# brain.py
import json
import os
import sys
import threading
import subprocess
import time
from datetime import datetime
from queue import Queue
from collections import deque

import requests
import sounddevice as sd
import webrtcvad
import numpy as np

from memory import MemoryDB

class KratixBrain:
    def __init__(self,
                 stt_url="http://localhost:8001/transcribe",
                 trans_url="http://localhost:8002/translate",
                 tts_url="http://localhost:8003/synthesize",
                 chat_url="http://localhost:8004/chat"):
        # service endpoints
        self.urls = {
            "stt":    stt_url,
            "trans":  trans_url,
            "tts":    tts_url,
            "chat":   chat_url,
        }

        # queues for pipelining
        self.queues = {
            "stt":    Queue(),
            "trans":  Queue(),
            "tts":    Queue(),
        }
        self.user_id = ""
        # user‐scoped memory
        self.memory = MemoryDB()

        # which micro‐services to auto‐spawn
        self.services = {}

        # settings (current and dest languages, toggles, etc.)
        self.settings = {
            "curr_lang":  "de",
            "dest_lang":  "en",
            "use_translate": False,
        }

        # prepare worker threads
        self._threads = [
            threading.Thread(target=self._record_loop, daemon=True),
            threading.Thread(target=self._stt_loop,    daemon=True),
            threading.Thread(target=self._trans_loop,  daemon=True),
            threading.Thread(target=self._tts_loop,    daemon=True),
        ]
        self.history = {}  # chat_id → [ messages ]
        # load system prompt once
        with open("./prompts/en/system_prompt.txt", "r", encoding="utf-8") as f:
            self.SYSTEM_PROMPT = f.read().strip()

    def start(self):
        """Start all background threads."""
        for t in self._threads:
            t.start()

    def stop(self):
        """Terminate all spawned services."""
        for name in list(self.services):
            self._stop_service(name)

    def get_datetime_message(self):
        now = datetime.now()
        return {
            "role": "system",
            "content": f"Current date: {now:%Y-%m-%d}. Current time: {now:%H:%M}Z"
        }

    def call_chat(self, text: str, chat_id: str, user_id: str) -> str:
        # 1) fetch or init history
        if chat_id not in self.history:
            self.history[chat_id] = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                self.get_datetime_message()
            ]

        # 2) prepend any stored facts for this user
        facts = self.memory.retrieve(chat_id, user_id)
        if facts:
            ctx = "\n".join(f"{k}: {v}" for k, v in facts.items())
            self.history[chat_id].append({"role": "system", "content": ctx})

        # 3) add the user message
        self.history[chat_id].append({"role": "user", "content": text})

        # 4) call your LLM endpoint
        payload = {
            "model": "google/gemma-3-12b",
            "stream": False,
            "messages": self.history[chat_id]
        }
        r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        full = data["choices"][0]["message"]["content"].strip()

        # 5) extract answer + JSON after `<-->`
        if "<-->" in full:
            answer, raw = full.split("<-->", 1)
            # strip everything outside the first {...}
            start, end = raw.find("{"), raw.rfind("}")
            js = raw[start:end + 1]
            obj = json.loads(js)
            # store facts if any
            if obj.get("facts"):
                # your JSON format: obj["facts"] is a dict already
                self.memory.store_facts(chat_id, user_id, obj["facts"])
            # you can also handle settings/task/actions here…
        else:
            answer = full

        # 6) append assistant to history
        self.history[chat_id].append({"role": "assistant", "content": answer.strip()})
        # 7) prune old turns
        max_turns = 8
        turns = self.history[chat_id][1:]  # keep system prompt at 0
        pruned = turns[-max_turns * 2:]
        self.history[chat_id] = [self.history[chat_id][0]] + pruned

        return answer.strip()
    # ─── recording until silence ─────────────────────────────────────────────
    def _record_loop(self):
        vad = webrtcvad.Vad(2)
        frame_len = int(16000 * 30 / 1000)
        pre_buf = deque(maxlen=int(500/30))
        in_speech = False
        silence_ct = 0
        buffer = []

        with sd.InputStream(samplerate=16000, channels=1,
                            dtype='int16', blocksize=frame_len) as stream:
            while True:
                pcm_frame, _ = stream.read(frame_len)
                pcm_bytes = pcm_frame.tobytes()
                pre_buf.append(pcm_bytes)

                try:
                    speech = vad.is_speech(pcm_bytes, 16000)
                except webrtcvad.Error:
                    continue

                if speech:
                    if not in_speech:
                        in_speech = True
                        buffer = list(pre_buf)
                    buffer.append(pcm_bytes)
                    silence_ct = 0
                elif in_speech:
                    buffer.append(pcm_bytes)
                    silence_ct += 1
                    if silence_ct * 30 >= 1000:
                        # hand off to STT
                        self.queues["stt"].put(b"".join(buffer))
                        in_speech = False

    # ─── STT worker ──────────────────────────────────────────────────────────
    def _stt_loop(self):
        while True:
            wav = self.queues["stt"].get()
            lang, text = self._call_service("stt", data=wav)
            if lang != "nn" and len(text.strip()) >= 3:
                self.settings["curr_lang"] = lang
                if self.settings["use_translate"]:
                    self.queues["trans"].put((text, lang))
                else:
                    reply = self.call_chat(text, chat_id="audio", user_id=self.user_id)
                    self.queues["tts"].put((reply, lang))

    # ─── Translation worker ──────────────────────────────────────────────────
    def _trans_loop(self):
        while True:
            text, src = self.queues["trans"].get()
            tr = self._call_service("trans", json={"text": text, "src": src, "dest": self.settings["dest_lang"]})
            if tr:
                self.queues["tts"].put((tr, self.settings["dest_lang"]))

    # ─── TTS worker ──────────────────────────────────────────────────────────
    def _tts_loop(self):
        while True:
            text, lang = self.queues["tts"].get()
            audio = self._call_service("tts", json={"text": text, "lang": lang}, stream=True)
            sd.play(audio, samplerate=22050); sd.wait()


    # ─── Generic HTTP + auto‐spawn service ─────────────────────────────────
    def _call_service(self, name, data=None, json=None, stream=False):
        self._ensure_service(name)
        r = requests.post(self.urls[name],
                          data=data, json=json, stream=stream,
                          headers={'Content-Type': 'application/octet-stream'} if data else {})
        r.raise_for_status()
        if stream:
            raw = b"".join(r.iter_content(4096))
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return pcm / np.iinfo(np.int16).max
        else:
            return r.json().get("language", ""), r.json().get("text", "") if name=="stt" else r.json().get("translation", "") if name=="trans" else r.json().get("reply", "")

    def _ensure_service(self, name):
        if name in self.services and self.services[name].poll() is None:
            return

        py = sys.executable
        cwd = os.path.dirname(__file__)
        env = os.environ.copy()

        cmd_map = {
            "stt":   [py, "-m", "uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "8001"],
            "trans": [py, "-m", "uvicorn", "translation_service:app", "--host", "0.0.0.0", "--port", "8002"],
            "tts":   [py, "-m", "uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8003"],
        }

        cmd = cmd_map[name]
        print(f">>> Starting {name}_service:", " ".join(cmd))
        # capture stdout/stderr so you can inspect logs if something goes wrong:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.services[name] = proc

        # optionally read a line to verify startup:
        # line = proc.stdout.readline()
        # print(f"[{name}] {line.strip()}")

        time.sleep(5)  # give it a moment to spin up

    def _stop_service(self, name):
        proc = self.services.get(name)
        if proc and proc.poll() is None:
            proc.terminate(); proc.wait(timeout=5)
            del self.services[name]

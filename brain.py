from queue import Queue
import threading
import sounddevice as sd
import webrtcvad
import numpy as np
from memory_raw import MemoryDB
from fetcher import InternetFetcher
from service_manager import ServiceManager
import json
import requests
from tool_handler import ToolHandler


class KratixBrain:
    def __init__(self):
        self.queues = {
            "stt": Queue(),
            "trans": Queue(),
            "tts": Queue(),
        }
        self.user_id = ""
        self.memory = MemoryDB()
        self.fetcher = InternetFetcher(timeout=3.0, max_retries=2)
        self.service_mgr = ServiceManager()

        self.settings = {
            "curr_lang": "en",
            "dest_lang": "de",
            "use_telegram": True,
            "use_translate": False,
            "use_tts": False,
            "use_stt": False,
            "use_model": "google/gemma-3-12b", # google/gemma-3-27b
        }

        self._threads = [
            threading.Thread(target=self._record_loop, daemon=True),
            threading.Thread(target=self._stt_loop, daemon=True),
            threading.Thread(target=self._trans_loop, daemon=True),
            threading.Thread(target=self._tts_loop, daemon=True),
        ]

        self.memory.set_context(813664714, "Klamenzui")
        self.memory.get_system_message()
        self.tool_handler = ToolHandler(self.memory, self.fetcher)

    def start(self):
        for t in self._threads:
            t.start()

    def stop(self):
        for name in ["stt", "trans", "tts"]:
            self.service_mgr.stop_service(name)

    def call_chat(self, content: any, chat_id: str, user_id: str, searched: bool = False) -> str:
        self.memory.set_context(chat_id, user_id)
        settings = self.memory.get_settings()
        history = self.memory.get_history(content)

        payload = {
            "model": settings.get("use_model", self.settings.get("use_model")),
            "stream": False,
            "messages": history
        }

        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=240)
            r.raise_for_status()
            data = r.json()
            full = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Chat error: {e}")
            return "Entschuldigung, der Chat-Service ist nicht verfügbar."

        try:
            answer, next_prompt = self.tool_handler.process_llm_response(
                full, chat_id=chat_id, user_id=user_id, searched=searched
            )
            self.memory.append_history(answer)
            if next_prompt:
                return self.call_chat(next_prompt, chat_id, user_id, searched=True)
        except Exception as e:
            print(e)
            answer = "Ein Fehler ist aufgetreten."
        return answer.strip()

    def _stt_loop(self):
        while True:
            try:
                wav = self.queues["stt"].get()
                lang, text = self.service_mgr.call_service("stt", data=wav)
                if lang != "nn" and len(text.strip()) >= 3:
                    self.settings["curr_lang"] = lang
                    if self.settings["use_translate"]:
                        self.queues["trans"].put((text, lang))
                    else:
                        reply = self.call_chat(text, chat_id="audio", user_id=self.user_id)
                        self.queues["tts"].put((reply, lang))
            except Exception as e:
                print(f"STT loop error: {e}")

    def _trans_loop(self):
        while True:
            try:
                text, src = self.queues["trans"].get()
                tr = self.service_mgr.call_service("trans", json={"text": text, "src": src, "dest": self.settings["dest_lang"]})
                if tr:
                    self.queues["tts"].put((tr, self.settings["dest_lang"]))
            except Exception as e:
                print(f"Translation loop error: {e}")

    def _tts_loop(self):
        while True:
            try:
                text, lang = self.queues["tts"].get()
                audio = self.service_mgr.call_service("tts", json={"text": text, "lang": lang}, stream=True)
                sd.play(audio, samplerate=22050)
                sd.wait()
            except Exception as e:
                print(f"TTS error: {e}")

    def _record_loop(self):
        try:
            vad = webrtcvad.Vad(2)
            frame_len = int(16000 * 30 / 1000)
            buffer, in_speech, silence_ct = [], False, 0
            pre_buf = []

            with sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=frame_len) as stream:
                while True:
                    if not self.settings["use_stt"]:
                        continue
                    try:
                        pcm_frame, _ = stream.read(frame_len)
                        pcm_bytes = pcm_frame.tobytes()
                        pre_buf.append(pcm_bytes)
                        if len(pre_buf) > 16:
                            pre_buf.pop(0)
                        speech = vad.is_speech(pcm_bytes, 16000)
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
                                self.queues["stt"].put(b"".join(buffer))
                                in_speech = False
                                buffer = []
                                silence_ct = 0
                    except Exception as e:
                        print(f"Recording loop error: {e}")
        except Exception as e:
            print(f"Fatal recording error: {e}")
            raise

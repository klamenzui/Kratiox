# Fixes für brain.py - Bessere Fehlerbehandlung und Debugging

import json
import os
import sys
import threading
import subprocess
import time
from datetime import datetime, timezone
from queue import Queue
from collections import deque

import requests
import sounddevice as sd
import webrtcvad
import numpy as np
from Tools.scripts.objgraph import ignore
from numba.np.numpy_support import is_array

from memory import MemoryDB

from fetcher import InternetFetcher


class KratixBrain:
    def __init__(self,
                 stt_url="http://localhost:8001/transcribe",
                 trans_url="http://localhost:8002/translate",
                 tts_url="http://localhost:8003/synthesize",
                 chat_url="http://localhost:8004/chat"):
        # service endpoints
        self.urls = {
            "stt": stt_url,
            "trans": trans_url,
            "tts": tts_url,
            "chat": chat_url,
        }

        # queues for pipelining
        self.queues = {
            "stt": Queue(),
            "trans": Queue(),
            "tts": Queue(),
        }
        self.user_id = ""
        # user‐scoped memory
        self.memory = MemoryDB()

        # which micro‐services to auto‐spawn
        self.services = {}
        self.service_ready = {}  # Track readiness separately

        # settings (current and dest languages, toggles, etc.)
        self.settings = {
            "curr_lang": "en",
            "dest_lang": "de",
            "use_telegram": True,
            "use_translate": False,
            "use_tts": False,
            "use_stt": False,
        }

        # prepare worker threads
        self._threads = [
            threading.Thread(target=self._record_loop, daemon=True),
            threading.Thread(target=self._stt_loop, daemon=True),
            threading.Thread(target=self._trans_loop, daemon=True),
            threading.Thread(target=self._tts_loop, daemon=True),
        ]
        self.history = {}  # chat_id → [ messages ]

        self.fetcher = InternetFetcher(timeout=3.0, max_retries=2)
        self.get_system_message(813664714, "Klamenzui")

    def start(self):
        """Start all background threads."""
        for t in self._threads:
            t.start()

    def stop(self):
        """Terminate all spawned services."""
        for name in list(self.services):
            self._stop_service(name)

    def get_system_message(self, chat_id, user_id):
        # history = self.memory.get_fact_history(chat_id, user_id, "company_name")
        facts_dict = self.memory.get_latest_facts(chat_id, user_id)
        facts = "\n".join(json.dumps(v, indent=4) for k, v in facts_dict.items()) if facts_dict else ""

        settings_dict = self.memory.get_settings(user_id)
        settings = "\n".join(f"{k}: {v!r}" for k, v in settings_dict.items()) if settings_dict else ""
        now = datetime.now(timezone.utc)
        try:
            sys_prompt = self.get_tpl_message("system_prompt", {
                "date": f"{now:%Y-%m-%d}",
                "time": f"{now:%H:%M}Z",
                "facts": facts,
                "settings": settings
            })
        except FileNotFoundError:
            print("Warning: system_prompt.txt not found, using default")
            sys_prompt = "You are a helpful AI assistant."
        with open(f"./prompts/current_system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(sys_prompt)
        return {"role": "system", "content": sys_prompt}

    def get_tpl_message(self, name, prompt_data):
        message = ""
        with open(f"./prompts/{self.settings.get('curr_lang', 'en')}/{name}.txt", "r", encoding="utf-8") as f:
            message = f.read().strip()
        for k, v in prompt_data.items():
            message = message.replace(f'%{k}%', f'{v}')
        return message

    def call_chat(self, text: any, chat_id: str, user_id: str, searched: bool = False) -> str:
        sys_message = self.get_system_message(chat_id, user_id)
        if chat_id not in self.history:
            self.history[chat_id] = [sys_message]
        else:
            self.history[chat_id][0] = sys_message

        # 3) add the user message
        self.history[chat_id].append({"role": "user", "content": text})

        # 4) call your LLM endpoint
        payload = {
            "model": "google/gemma-3-12b",
            "stream": False,
            "messages": self.history[chat_id]
        }

        print(f"DEBUG: Calling LLM with payload: {json.dumps(payload, indent=2)}")

        try:
            # Check if LLM server is running
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=180)
            print(f"DEBUG: LLM response status: {r.status_code}")
            if r.status_code != 200:
                print(f"DEBUG: LLM response text: {r.text}")
            r.raise_for_status()
            data = r.json()
            full = data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            print("ERROR: LLM server nicht erreichbar auf localhost:1234")
            return "Entschuldigung, der Chat-Service ist nicht verfügbar."
        except requests.exceptions.HTTPError as e:
            print(f"ERROR: LLM server HTTP error: {e}")
            print(f"Response: {r.text}")
            return "Entschuldigung, es gab ein Problem mit dem Chat-Service."
        except Exception as e:
            print(f"ERROR: Unexpected error calling LLM: {e}")
            return "Entschuldigung, es gab einen unerwarteten Fehler."
        try:
            sep: str = '###' if '###' in full else ''
            print(sep)
            sep = '<-->' if not sep and '<-->' in full else ''
            print(sep)
            sep = '```' if not sep and '```' in full else sep
            print(sep, full)
            if sep and sep in full:
                answer, raw = full.split(sep, 1)
                # strip everything outside the first {...}
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end != -1:
                    js = raw[start:end + 1]
                    try:
                        obj = json.loads(js)
                        print(f"Infos: {json.dumps(obj, indent=4)}")
                        # store facts if any
                        if obj.get("facts"):
                            self.memory.store_facts(chat_id, user_id, obj["facts"])
                        if obj.get("settings"):
                            self.memory.store_settings(user_id, obj["settings"])
                        if obj.get("action"):
                            action = obj.get("action")
                            if action.get("name") == "search" and not searched:
                                if action.get("type") == "text":
                                    results = self.fetcher.web_search(action.get("args"))
                                    print("web search: ", action)
                                    #messages = [
                                    #    {"type": "text", "text": answer}] if answer else []
                                    # messages.append({"type": "text","text": })
                                    return self.call_chat(self.get_tpl_message("web_search", {
                                                "query": action.get("args", {}).get('query'),
                                                "results": results,
                                            }), chat_id, user_id, True)
                                if action.get("type") == "crypto_price":
                                    results = self.fetcher.get_crypto_price(action.get("args"))
                                    return self.call_chat(self.get_tpl_message("web_search", {
                                        "query": action.get("args", {}).get('ids'),
                                        "results": results,
                                    }), chat_id, user_id, True)

                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON: {e}")
            else:
                answer = full
            answer = answer.replace(sep, "")
            # 6) append assistant to history
            self.history[chat_id].append({"role": "assistant", "content": answer.strip()})
            # 7) prune old turns
            max_turns = 8
            turns = self.history[chat_id][1:]  # keep system prompt at 0
            pruned = turns[-max_turns * 2:]
            self.history[chat_id] = [self.history[chat_id][0]] + pruned
        except Exception as e:
            print(e)
            answer = "an exception caused"
        return answer.strip()

    # ─── STT worker ──────────────────────────────────────────────────────────
    def _stt_loop(self):
        while True:
            try:
                wav = self.queues["stt"].get()
                lang, text = self.call_service("stt", data=wav)
                print(f"DEBUG: STT result - lang: {lang}, text: {text}")
                if lang != "nn" and len(text.strip()) >= 3:
                    self.settings["curr_lang"] = lang
                    if self.settings["use_translate"]:
                        self.queues["trans"].put((text, lang))
                    else:
                        reply = self.call_chat(text, chat_id="audio", user_id=self.user_id)
                        self.queues["tts"].put((reply, lang))
            except Exception as e:
                print(f"ERROR in STT loop: {e}")

    # ─── Translation worker ──────────────────────────────────────────────────
    def _trans_loop(self):
        while True:
            try:
                text, src = self.queues["trans"].get()
                tr = self.call_service("trans", json={"text": text, "src": src, "dest": self.settings["dest_lang"]})
                if tr:
                    self.queues["tts"].put((tr, self.settings["dest_lang"]))
            except Exception as e:
                print(f"ERROR in translation loop: {e}")

    # ─── TTS worker ──────────────────────────────────────────────────────────
    def _tts_loop(self):
        while True:
            try:
                text, lang = self.queues["tts"].get()
                print(f"DEBUG: TTS request - text: {text}, lang: {lang}")
                audio = self.call_service("tts", json={"text": text, "lang": lang}, stream=True)
                sd.play(audio, samplerate=22050)
                sd.wait()
            except Exception as e:
                print(f"TTS Error: {e}")

    # ─── Generic HTTP + auto‐spawn service ─────────────────────────────────
    def call_service(self, name, data=None, json=None, stream=False):
        self._ensure_service(name)

        try:
            r = requests.post(self.urls[name],
                              data=data, json=json, stream=stream,
                              headers={'Content-Type': 'application/octet-stream'} if data else {},
                              timeout=30)

            if r.status_code != 200:
                print(f"ERROR: Service {name} returned {r.status_code}: {r.text}")

            r.raise_for_status()

            if stream:
                raw = b"".join(r.iter_content(4096))
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                return pcm / np.iinfo(np.int16).max
            else:
                response_data = r.json()
                if name == "stt":
                    return response_data.get("language", ""), response_data.get("text", "")
                elif name == "trans":
                    return response_data.get("translation", "")
                else:
                    return response_data.get("reply", "")
        except Exception as e:
            print(f"ERROR calling service {name}: {e}")
            raise

    def _ensure_service(self, name):
        # Check if service is already running and ready
        if (name in self.services and
                self.services[name].poll() is None and
                self.service_ready.get(name, False)):
            return

        # If service exists but not ready, don't start another one
        if name in self.services and self.services[name].poll() is None:
            print(f"Service {name} is running but not ready, waiting...")
            if self._wait_for_service_ready(name):
                return
            else:
                print(f"Service {name} failed to become ready, restarting...")
                self._stop_service(name)

        py = sys.executable
        cwd = os.path.dirname(__file__)
        env = os.environ.copy()

        cmd_map = {
            "stt": [py, "-m", "uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "8001"],
            "trans": [py, "-m", "uvicorn", "translation_service:app", "--host", "0.0.0.0", "--port", "8002"],
            "tts": [py, "-m", "uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8003"],
        }

        cmd = cmd_map[name]
        print(f">>> Starting {name}_service:", " ".join(cmd))

        # Start the service process
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.services[name] = proc
        self.service_ready[name] = False

        # Wait for service to be ready with longer timeout for STT
        max_wait = 180 if name == "stt" else 60  # 3 minutes for STT, 1 minute for others

        if self._wait_for_service_ready(name, max_wait):
            print(f">>> Service {name} is ready!")
        else:
            print(f">>> Service {name} failed to start within {max_wait} seconds")

            # Try to get some output from the failed process
            try:
                # Don't wait too long for output from a failed process
                stdout, stderr = proc.communicate(timeout=2)
                if stdout:
                    print(f"STDOUT: {stdout}")
                if stderr:
                    print(f"STDERR: {stderr}")
            except subprocess.TimeoutExpired:
                print(f">>> Service {name} is still running but not responding")
                # Kill the process if it's stuck
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                    if stdout:
                        print(f"STDOUT: {stdout}")
                    if stderr:
                        print(f"STDERR: {stderr}")
                except:
                    pass
            except Exception as e:
                print(f"Could not get output from failed service: {e}")

            # Clean up the failed service
            self._stop_service(name)
            raise Exception(f"Service {name} failed to start within {max_wait} seconds")

    def _wait_for_service_ready(self, name, max_wait=60):
        """Wait for service to be ready with timeout."""
        wait_interval = 2
        waited = 0

        print(f">>> Waiting for {name} service to be ready (max {max_wait}s)...")

        while waited < max_wait:
            # Check if process is still alive
            proc = self.services.get(name)
            if proc and proc.poll() is not None:
                print(f">>> Service {name} process died (exit code: {proc.poll()})")
                return False

            # Check if service is ready
            if self._is_service_ready(name):
                self.service_ready[name] = True
                print(f">>> Service {name} is ready after {waited}s!")
                return True

            # Only print waiting message every 10 seconds to reduce spam
            if waited % 10 == 0:
                print(f">>> Still waiting for {name} service... ({waited}s/{max_wait}s)")

            time.sleep(wait_interval)
            waited += wait_interval

        print(f">>> Timeout waiting for {name} service ({max_wait}s)")
        return False

    def _is_service_ready(self, name):
        """Check if a service is ready to accept requests."""
        try:
            if name == "stt":
                # Test with minimal dummy data for STT
                test_data = b'\x00' * 320  # 20ms of silence at 16kHz
                response = requests.post(self.urls[name],
                                         data=test_data,
                                         timeout=10,
                                         headers={'Content-Type': 'application/octet-stream'})
                return response.status_code == 200
            elif name == "tts":
                # Test with simple text
                test_json = {"text": "test", "lang": "en"}
                response = requests.post(self.urls[name], json=test_json, timeout=10)
                return response.status_code == 200
            elif name == "trans":
                # Test with simple translation
                test_json = {"text": "test", "src": "en", "dest": "de"}
                response = requests.post(self.urls[name], json=test_json, timeout=10)
                return response.status_code == 200
            else:
                return False
        except requests.exceptions.ConnectionError:
            # Service not yet available
            return False
        except requests.exceptions.Timeout:
            # Service is responding but too slow
            return False
        except Exception as e:
            # Other errors might indicate the service is not ready
            return False

    def _stop_service(self, name):
        """Stop a service process."""
        proc = self.services.get(name)
        if proc:
            if proc.poll() is None:  # Process is still running
                print(f">>> Stopping {name} service...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f">>> Force killing {name} service...")
                    proc.kill()
                    proc.wait()

            # Clean up
            if name in self.services:
                del self.services[name]
            if name in self.service_ready:
                del self.service_ready[name]

    def _record_loop(self):
        """Audio recording loop with VAD."""
        try:
            vad = webrtcvad.Vad(2)
            frame_len = int(16000 * 30 / 1000)  # 30ms frames
            pre_buf = deque(maxlen=int(500 / 30))  # 500ms buffer
            in_speech = False
            silence_ct = 0
            buffer = []

            print(">>> Starting audio recording...")

            with sd.InputStream(samplerate=16000, channels=1,
                                dtype='int16', blocksize=frame_len) as stream:
                while True:
                    if not self.settings["use_stt"]: continue
                    try:
                        pcm_frame, _ = stream.read(frame_len)
                        pcm_bytes = pcm_frame.tobytes()
                        pre_buf.append(pcm_bytes)

                        # Check for speech
                        try:
                            speech = vad.is_speech(pcm_bytes, 16000)
                        except webrtcvad.Error:
                            continue

                        if speech:
                            if not in_speech:
                                print(">>> Speech detected, starting recording...")
                                in_speech = True
                                buffer = list(pre_buf)
                            buffer.append(pcm_bytes)
                            silence_ct = 0
                        elif in_speech:
                            buffer.append(pcm_bytes)
                            silence_ct += 1
                            if silence_ct * 30 >= 1000:  # 1 second of silence
                                print(">>> Speech ended, processing audio...")
                                # hand off to STT
                                self.queues["stt"].put(b"".join(buffer))
                                in_speech = False
                                buffer = []
                                silence_ct = 0
                    except Exception as e:
                        print(f"Error in recording loop: {e}")
                        time.sleep(0.1)  # Brief pause before continuing

        except Exception as e:
            print(f"FATAL: Recording loop failed: {e}")
            raise

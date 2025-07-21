# service_manager.py
import os
import sys
import time
import subprocess
import requests
import numpy as np

class ServiceManager:
    def __init__(self):
        self.urls = {
            "stt": "http://localhost:8001/transcribe",
            "trans": "http://localhost:8002/translate",
            "tts": "http://localhost:8003/synthesize",
        }
        self.services = {}
        self.service_ready = {}

    def call_service(self, name, data=None, json=None, stream=False):
        self._ensure_service(name)
        try:
            r = requests.post(
                self.urls[name],
                data=data, json=json, stream=stream,
                headers={'Content-Type': 'application/octet-stream'} if data else {},
                timeout=30
            )
            r.raise_for_status()
            if stream:
                raw = b"".join(r.iter_content(4096))
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                return pcm / np.iinfo(np.int16).max
            else:
                resp = r.json()
                if name == "stt":
                    return resp.get("language", ""), resp.get("text", "")
                elif name == "trans":
                    return resp.get("translation", "")
                else:
                    return resp.get("reply", "")
        except Exception as e:
            print(f"ERROR calling {name}: {e}")
            raise

    def _ensure_service(self, name):
        if name in self.services and self.services[name].poll() is None and self.service_ready.get(name):
            return
        if name in self.services and self.services[name].poll() is None:
            if self._wait_for_service_ready(name):
                return
            self._stop_service(name)

        py = sys.executable
        cwd = os.path.dirname(__file__)
        cmd_map = {
            "stt": [py, "-m", "uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "8001"],
            "trans": [py, "-m", "uvicorn", "translation_service:app", "--host", "0.0.0.0", "--port", "8002"],
            "tts": [py, "-m", "uvicorn", "tts_service:app", "--host", "0.0.0.0", "--port", "8003"],
        }

        proc = subprocess.Popen(
            cmd_map[name],
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.services[name] = proc
        self.service_ready[name] = False
        if self._wait_for_service_ready(name, max_wait=180 if name == "stt" else 60):
            self.service_ready[name] = True
        else:
            self._stop_service(name)
            raise Exception(f"{name} service failed to start")

    def _wait_for_service_ready(self, name, max_wait=60):
        waited = 0
        while waited < max_wait:
            if self.services[name].poll() is not None:
                return False
            if self._is_service_ready(name):
                return True
            time.sleep(2)
            waited += 2
        return False

    def _is_service_ready(self, name):
        try:
            if name == "stt":
                dummy = b'\x00' * 320
                r = requests.post(self.urls[name], data=dummy, timeout=5,
                                  headers={'Content-Type': 'application/octet-stream'})
                return r.status_code == 200
            elif name == "tts":
                return requests.post(self.urls[name], json={"text": "test", "lang": "en"}, timeout=5).status_code == 200
            elif name == "trans":
                return requests.post(self.urls[name], json={"text": "test", "src": "en", "dest": "de"}, timeout=5).status_code == 200
        except:
            return False

    def _stop_service(self, name):
        proc = self.services.get(name)
        if proc:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()
        self.services.pop(name, None)
        self.service_ready.pop(name, None)

    def stop_service(self, name):
        self.stop_service(name)

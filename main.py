# main.py
import socket

import sounddevice as sd
import webrtcvad
import numpy as np
import datetime
import time
import requests
import base64
import subprocess

from queue import Queue
from collections import deque
import threading

# =============================
# Konfiguration
# =============================
SAMPLE_RATE    = 16000
FRAME_MS       = 30
AGGRESSIVENESS = 2
SILENCE_LIMIT  = 1.0   # Sekunden Stille bis Segmentende
PRE_MS         = 500   # ms Vorpuffer für Speech-Start

STT_URL   = "http://localhost:8001/transcribe"
TRANS_URL = "http://localhost:8002/translate"
TTS_URL   = "http://localhost:8003/synthesize"
services = {}
settings = {
    "curr_lang": "de",
    "dest_lang": "en",
    "active_service": {
        "record": False
    }
}
def set_curr_lang(lang:str):
    curr_lang = settings.get("curr_lang")
    if lang == settings["dest_lang"]:
        settings["dest_lang"] = curr_lang
    settings["curr_lang"] = lang
# =============================
# Aufnahme bis Stille
# =============================
def record_until_silence():
    vad        = webrtcvad.Vad(AGGRESSIVENESS)
    frame_len  = int(SAMPLE_RATE * FRAME_MS/1000)
    pre_buf    = deque(maxlen=int(PRE_MS/FRAME_MS))
    speech     = []
    in_speech  = False
    silence_ct = 0

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='int16', blocksize=frame_len)
    with stream:
        while True:
            chunk, _ = stream.read(frame_len)
            pcm = chunk.tobytes()
            pre_buf.append(pcm)

            try:
                is_s = vad.is_speech(pcm, SAMPLE_RATE)
            except Exception as e:
                print('record_until_silence', e)
                continue

            if is_s:
                if not in_speech:
                    in_speech = True
                    speech = list(pre_buf)  # alles von vor dem Speech-Start
                speech.append(pcm)
                silence_ct = 0
            elif in_speech:
                speech.append(pcm)
                silence_ct += 1
                if silence_ct * FRAME_MS >= SILENCE_LIMIT * 1000:
                    return b"".join(speech)

# =============================
# HTTP-Wrapper
# =============================
def call_stt(wav_bytes):
    resp = requests.post(STT_URL,
                         data=wav_bytes,
                         headers={'Content-Type':'application/octet-stream'})
    resp.raise_for_status()
    js = resp.json()
    return js['language'], js['text']

def call_translate(text, src, dest):
    resp = requests.post(TRANS_URL, json={'text':text,'src':src,'dest':dest})
    resp.raise_for_status()
    res = resp.json()
    try:
        return res.get('translation','')
    except Exception as e:
        #print(e)
        #print(res)
        return ''

def call_tts(text, lang):
    resp = requests.post(TTS_URL, json={'text':text,'lang':lang}, stream=True)
    resp.raise_for_status()
    wav = b"".join(resp.iter_content(4096))
    arr = np.frombuffer(wav, dtype=np.int16).astype(np.float32)
    return arr / np.iinfo(np.int16).max

def play(arr):
    sd.play(arr, samplerate=22050)
    sd.wait()

def is_too_quiet(wav_bytes, threshold=100):
    samples = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    print(rms,  threshold)
    return rms < threshold

def is_valid_text(text, min_chars=5):
    txt = text.strip()
    # Mindestlänge, keine reinen Sonderzeichen
    return len(txt) >= min_chars and any(c.isalnum() for c in txt)

# =============================
# Queues für die Pipeline
# =============================
stt_queue = Queue()
trans_queue = Queue()
tts_queue = Queue()

# =============================
# Worker-Loops
# =============================
def record_loop():
    print("🎤 Recorder-Thread läuft...")
    while True:
        if not settings.get("active_service").get("record"):
            continue
        wav = record_until_silence()
        if is_too_quiet(wav):
            print('too quite')
            continue  # verwerfe Stille
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stt_queue.put((wav, ts))

def stt_loop():
    # ensure_ollama()
    print("📝 STT-Thread läuft...")
    while True:
        if not ensure_service("stt_service"):
            continue
        wav, ts = stt_queue.get()
        # in stt_loop:
        lang, text = call_stt(wav)
        if not is_valid_text(text) or lang == "nn":
            print('is not valid text')
            stt_queue.task_done()
            continue
        set_curr_lang(lang)
        print(f"📝 ({ts}) erkannt [{lang}]: {text}")
        # Ollama-Check
        #checked = call_ollama_http(text)

        #print(f"test Ollama-Korrektur → {checked}")
        stt_queue.task_done()
        if not settings.get("active_service").get("translation_service"):
            answer = call_chat(text)
            print(f"🔍 ({ts}) Ollama-answer → {answer}")
            tts_queue.put((answer, ts))
        else:
            trans_queue.put((text, lang, ts))

def translate_loop():
    print("🔄 Translate-Thread läuft...")
    while True:
        if not ensure_service("translation_service"):
            continue
        text, lang, ts = trans_queue.get()
        tr = call_translate(text, src=lang, dest=settings.get("dest_lang"))
        if tr:
            print(f"🔄 ({ts}) übersetzt → {tr}")
            tts_queue.put((tr, ts))
            trans_queue.task_done()

def tts_loop():
    print("🔊 TTS-Thread läuft...")
    while True:
        if not ensure_service("tts_service"):
            continue
        tr, ts = tts_queue.get()
        audio = call_tts(tr, lang=settings.get("dest_lang"))
        print(f"🔊 ({ts}) Abspielen …")
        play(audio)
        tts_queue.task_done()


def call_chat(user_text, chat_id="audio"):
    r = requests.post(
        "http://localhost:8004/chat",
        json={"chat_id": chat_id, "message": user_text},
        timeout=5
    )
    r.raise_for_status()
    return r.json()["reply"]

def start_service(script_path: str, args: list = None):
    """
    Startet das Script `script_path` (z.B. translation_service.py) als
    Hintergrundprozess, wenn es noch nicht läuft.
    """
    if args is None:
        args = []
    proc = services.get(script_path)
    # Wenn noch kein Prozess oder der alte schon beendet ist:
    if proc is None or proc.poll() is not None:
        cmd = ["python", f"{script_path}.py"] + args
        # Popen startet non-blocking
        services[script_path] = subprocess.Popen(cmd)
        print(f">>> Service '{script_path}' gestartet mit PID {services[script_path].pid}")
    else:
        print(f">>> Service '{script_path}' läuft bereits (PID {proc.pid})")

def stop_service(name: str):
    """
    Beendet den Prozess sauber, wenn er läuft.
    """
    proc = services.get(name)
    if proc is None:
        print(f"--- Service '{name}' war gar nicht aktiv.")
        return
    if proc.poll() is None:
        print(f"--- Beende Service '{name}' (PID {proc.pid}) …")
        proc.terminate()   # SIGTERM unter Unix, CTRL-BREAK u.U. unter Windows
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"*** Service '{name}' reagiert nicht, kill …")
            proc.kill()
    else:
        print(f"--- Service '{name}' lief schon beendet.")
    services.pop(name, None)

def ensure_service(service_name: str):
    is_running = services.get(service_name)
    is_active = settings.get("active_service").get(service_name)
    if is_active and not is_running:
        start_service("translation_service")
        # evtl. kurz warten, bis HTTP-Server hoch ist
        time.sleep(1)
    if not is_active and is_running:
        stop_service(service_name)
    return is_active
# =============================
# Main: Threads starten
# =============================
if __name__ == "__main__":
    # Starte alle Worker als Daemon-Threads
    threading.Thread(target=record_loop, daemon=True).start()
    threading.Thread(target=stt_loop, daemon=True).start()
    threading.Thread(target=translate_loop, daemon=True).start()
    threading.Thread(target=tts_loop, daemon=True).start()

    print("Pipeline ist aktiv. STRG-C zum Beenden.")
    try:
        # Halte das Haupt-Thread am Leben
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Beende…")
        # Am Programmende: alle noch laufenden Services beenden
        for name in list(services.keys()):
            stop_service(name)

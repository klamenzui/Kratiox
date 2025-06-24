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
DEST_LANG = "en"
is_stt_active = True
is_translation_active = False
is_tts_active = False
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
        wav = record_until_silence()
        if is_too_quiet(wav):
            print('too quite')
            continue  # verwerfe Stille
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stt_queue.put((wav, ts))

def stt_loop():
    ensure_ollama()
    print("📝 STT-Thread läuft...")
    while is_stt_active:
        wav, ts = stt_queue.get()
        # in stt_loop:
        lang, text = call_stt(wav)
        if not is_valid_text(text):
            print('is not valid text')
            stt_queue.task_done()
            continue
        print(f"📝 ({ts}) erkannt [{lang}]: {text}")
        # Ollama-Check
        checked = call_ollama_check(text)
        if checked != text:
            print(f"🔍 ({ts}) Ollama-Korrektur → {checked}")
        stt_queue.task_done()
        trans_queue.put((checked, lang, ts))

def translate_loop():
    print("🔄 Translate-Thread läuft...")
    while is_translation_active:
        text, lang, ts = trans_queue.get()
        tr = call_translate(text, src=lang, dest=DEST_LANG)
        if tr:
            print(f"🔄 ({ts}) übersetzt → {tr}")
            tts_queue.put((tr, ts))
            trans_queue.task_done()

def tts_loop():
    print("🔊 TTS-Thread läuft...")
    while is_tts_active:
        tr, ts = tts_queue.get()
        audio = call_tts(tr, lang=DEST_LANG)
        print(f"🔊 ({ts}) Abspielen …")
        play(audio)
        tts_queue.task_done()

# ollama serve
# ollama run llama3.2:latest
# deepseek-r1 | llama3.2 | devstral | llama4 |codellama
# =============================
# Ollama-Integration
# =============================
OLLAMA_PORT = 11434
OLLAMA_MODEL = "llama3.2:latest"

def ensure_ollama():
    """Stellt sicher, dass 'ollama serve' läuft."""
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", OLLAMA_PORT))
        s.close()
    except ConnectionRefusedError:
        print("▶ Ollama läuft noch nicht – starte 'ollama serve' …")
        subprocess.Popen(["ollama", "serve"])
        # kurz warten, bis der Server hoch ist
        time.sleep(2)

def call_ollama_check(text: str) -> str:
    """
    Fragt Ollama, ob der erkannte Text Fehler enthält, und gibt eine korrigierte Fassung zurück.
    """
    prompt = (
        "Du bist ein Korrektur-Tool. "
        "Überprüfe den folgenden erkannten Text auf Erkennungsfehler und "
        "gib nur den korrigierten Text aus:\n\n"
        f"{text}"
    )
    try:
        proc = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        else:
            return text
    except Exception as e:
        print("⚠️ Ollama-Check fehlgeschlagen:", e)
        return text

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

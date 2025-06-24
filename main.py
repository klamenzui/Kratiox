# main.py
import sounddevice as sd
import webrtcvad
import numpy as np
import datetime
import time
import requests
import base64

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

def is_too_quiet(wav_bytes, threshold=200):
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
    print("📝 STT-Thread läuft...")
    while True:
        wav, ts = stt_queue.get()
        # in stt_loop:
        lang, text = call_stt(wav)
        #if not is_valid_text(text):
        #    print('is not valid text')
        #    stt_queue.task_done()
        #    continue
        print(f"📝 ({ts}) erkannt [{lang}]: {text}")
        trans_queue.put((text, lang, ts))
        stt_queue.task_done()

def translate_loop():
    print("🔄 Translate-Thread läuft...")
    while True:
        text, lang, ts = trans_queue.get()
        tr = call_translate(text, src=lang, dest=DEST_LANG)
        if tr:
            print(f"🔄 ({ts}) übersetzt → {tr}")
            tts_queue.put((tr, ts))
            trans_queue.task_done()

def tts_loop():
    print("🔊 TTS-Thread läuft...")
    while True:
        tr, ts = tts_queue.get()
        audio = call_tts(tr, lang=DEST_LANG)
        print(f"🔊 ({ts}) Abspielen …")
        play(audio)
        tts_queue.task_done()

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

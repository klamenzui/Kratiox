# main.py
import sounddevice as sd
import webrtcvad
import numpy as np
from collections import deque
import datetime, requests, base64

SAMPLE_RATE    = 16000
FRAME_MS       = 30
AGGRESSIVENESS = 2
SILENCE_LIMIT  = 1.0
PRE_MS         = 500

STT_URL   = "http://localhost:8001/transcribe"
TRANS_URL = "http://localhost:8002/translate"
TTS_URL   = "http://localhost:8003/synthesize"
DEST_LANG = "en"

def record_until_silence():
    vad = webrtcvad.Vad(AGGRESSIVENESS)
    frame_len = int(SAMPLE_RATE * FRAME_MS/1000)
    pre_buf   = deque(maxlen=int(PRE_MS/FRAME_MS))
    speech, in_speech, cnt = [], False, 0

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='int16', blocksize=frame_len)
    with stream:
        print("🎤 Aufnahme läuft... (STRG-C zum Abbrechen)")
        while True:
            chunk, _ = stream.read(frame_len)
            pcm = chunk.tobytes()
            pre_buf.append(pcm)
            try:
                is_s = vad.is_speech(pcm, SAMPLE_RATE)
            except webrtcvad.Error:
                continue

            if is_s:
                if not in_speech:
                    in_speech = True
                    speech = list(pre_buf)
                speech.append(pcm)
                cnt = 0
            elif in_speech:
                speech.append(pcm)
                cnt += 1
                if cnt * FRAME_MS >= SILENCE_LIMIT*1000:
                    break

    return b"".join(speech)

def call_stt(wav_bytes):
    r = requests.post(STT_URL, data=wav_bytes,
                      headers={'Content-Type':'application/octet-stream'})
    r.raise_for_status()
    d = r.json()
    return d['language'], d['text']

def call_translate(text, src, dest):
    r = requests.post(TRANS_URL, json={'text':text,'src':src,'dest':dest})
    r.raise_for_status()
    res = r.json()
    try:
        return res.get('translation','')
    except Exception as e:
        print(e)
        print(res)

def call_tts(text, lang):
    # 1) POST mit JSON, 2) stream=True um nicht alles in den Speicher zu laden
    r = requests.post(
        TTS_URL,
        json={"text": text, "lang": lang},
        stream=True
    )
    r.raise_for_status()
    # 3) WAV-Bytes einlesen
    wav_bytes = b"".join(r.iter_content(chunk_size=4096))
    # 4) Zu NumPy-Array konvertieren (int16 → float32 in [-1,1])
    pcm = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32)
    pcm /= np.iinfo(np.int16).max
    return pcm

def play(arr):
    sd.play(arr, samplerate=22050)
    sd.wait()

def main():
    print("Recorder → STT → Translate → TTS Pipeline")
    while True:
        # Aufnahme bis Stille
        wav = record_until_silence()
        lang, text = call_stt(wav)
        print(f"📝 Erkannt ({lang}): {text}")

        tr = call_translate(text, src=lang, dest=DEST_LANG)
        print(f"🔄 Übersetzt → {tr}")

        # Hier wird n i c h t mehr r.json() aufgerufen
        audio = call_tts(tr, lang=DEST_LANG)
        play(audio)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Programm beendet.")
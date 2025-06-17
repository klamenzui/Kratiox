# pip install git+https://github.com/openai/whisper.git
# pip install TTS sentencepiece sounddevice numpy scipy noisereduce pydub webrtcvad

import os
import wave
import threading
import datetime
import whisper
import sounddevice as sd
import webrtcvad
import numpy as np
import torch
from queue import Queue
from collections import deque
from TTS.api import TTS
from transformers import MarianMTModel, MarianTokenizer  # Offline-Übersetzung via HuggingFace
from scipy.io.wavfile import write as wav_write

# =============================
# Konfiguration
# =============================
SAMPLE_RATE    = 16000
FRAME_DURATION = 30    # ms
AGGRESSIVENESS = 2
SILENCE_LIMIT  = 1.0   # Sekunden Stille bis Segmentende
PRE_SPEECH_MS  = 500   # ms Vorpuffer

# Speichert Dateien nur, wenn True
save_to_file = False

# =============================
# Pfad zu ffmpeg (falls erforderlich)
# =============================
project_root = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path   = os.path.join(project_root, "ffmpeg", "bin")
os.environ["PATH"] += os.pathsep + ffmpeg_path
result_path = 'result'
if not os.path.isdir(result_path):
    os.makedirs (result_path)

# =============================
# Zusätzliche Requirements
# =============================
# pip install sentencepiece transformers

# =============================
# Modelle global laden
# =============================
stt_model = whisper.load_model("medium")
device = "cuda" if torch.cuda.is_available() else "cpu"

TTS_MODEL_MAP = {
    "de": "tts_models/de/thorsten/vits",
    "en": "tts_models/en/vctk/vits",
    "ru": "tts_models/ru/v3_1/vits",
    "ua": "tts_models/multilingual/multi-dataset/vits",
}
tts_instances = {}

TRANSLATION_MODELS = {
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "uk-en": "Helsinki-NLP/opus-mt-mul-en",
    "en-uk": "Helsinki-NLP/opus-mt-en-mul",
}
translation_models = {}

audio_queue = Queue()
translation_queue = Queue()
translation_tts_queue = Queue()
translation_dest = "en"

# =============================
# Hilfsfunktionen
# =============================
def get_tts_for_lang(lang: str) -> TTS:
    if lang not in TTS_MODEL_MAP:
        raise ValueError(f"Sprache nicht unterstützt: {lang}")
    if lang not in tts_instances:
        tts = TTS(model_name=TTS_MODEL_MAP[lang], progress_bar=False)
        tts.to(device)
        tts_instances[lang] = tts
    return tts_instances[lang]


def transcribe_audio_frames(frames: list) -> (str, str):
    # PCM-Bytes → numpy float32 in [-1,1]
    audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    audio_np /= np.iinfo(np.int16).max
    # Whisper akzeptiert numpy arrays direkt
    result = stt_model.transcribe(audio_np)
    return result["language"], result["text"].strip()


def get_translation_pipeline(src: str, dest: str):
    key = f"{src}-{dest}"
    if key not in TRANSLATION_MODELS:
        raise ValueError(f"Keine Übersetzungs-Pipeline für {key}")
    if key not in translation_models:
        model_name = TRANSLATION_MODELS[key]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        translation_models[key] = (tokenizer, model)
    return translation_models[key]

# =============================
# Recorder-Thread: kontinuierliche Aufnahme + Segmentierung
# =============================
def record_loop():
    vad = webrtcvad.Vad(AGGRESSIVENESS)
    frame_size = int(SAMPLE_RATE * FRAME_DURATION / 1000)
    pre_buffer = deque(maxlen=int(PRE_SPEECH_MS / FRAME_DURATION))
    speech_frames = []
    in_speech = False
    silence_counter = 0

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             dtype='int16', blocksize=frame_size)
    with stream:
        print("🎤 [Recorder] Starte kontinuierliche Aufnahme...")
        while True:
            chunk, _ = stream.read(frame_size)
            pcm = chunk.tobytes()
            pre_buffer.append(pcm)
            try:
                is_speech = vad.is_speech(pcm, SAMPLE_RATE)
            except webrtcvad.Error:
                continue

            if is_speech:
                if not in_speech:
                    in_speech = True
                    speech_frames = list(pre_buffer)
                speech_frames.append(pcm)
                silence_counter = 0
            elif in_speech:
                speech_frames.append(pcm)
                silence_counter += 1
                if silence_counter * FRAME_DURATION >= SILENCE_LIMIT * 1000:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # optional speichern
                    if save_to_file:
                        wav_filename = f"speech_{timestamp}.wav"
                        wav_write(wav_filename, SAMPLE_RATE, np.frombuffer(b"".join(speech_frames), dtype=np.int16))
                    audio_queue.put((speech_frames, timestamp))
                    in_speech = False
                    silence_counter = 0
                    speech_frames = []
                    pre_buffer.clear()

# =============================
# Transkriptions-Thread
# =============================
def transcribe_loop():
    while True:
        item = audio_queue.get()
        if item is None:
            break
        frames, timestamp = item
        lang, text = transcribe_audio_frames(frames)
        print(f"📝 [Transcriber] Erkannt ({lang}): {text}")
        # optional Text speichern
        #if save_to_file:
        txt = f"{result_path}/text_{timestamp}.txt"
        with open(txt, 'w', encoding='utf-8') as f:
            f.write(f"[{lang}] {text}\n")
        translation_queue.put((text, timestamp, lang))
        audio_queue.task_done()

# =============================
# Übersetzungs-Thread (offline via MarianMT)
# =============================
def translation_loop():
    while True:
        item = translation_queue.get()
        if item is None:
            break
        text, timestamp, src_lang = item
        try:
            tokenizer, model = get_translation_pipeline(src_lang, translation_dest)
            inputs = tokenizer(text, return_tensors='pt', padding=True).to(device)
            tokens = model.generate(**inputs)
            translated = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"⚠️ Übersetzung fehlgeschlagen: {e}")
            translation_queue.task_done()
            continue
        print(f"🔄 [Translator] {translated}")
        #if save_to_file:
        fn = f"{result_path}/translation_{timestamp}.txt"
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(f"[{translation_dest}] {translated}\n")
        ####
        translation_tts_queue.put((translated, timestamp))
        translation_queue.task_done()

# =============================
# Übersetzungs-TTS-Thread
# =============================
def translation_tts_loop():
    while True:
        item = translation_tts_queue.get()
        if item is None:
            break
        translated, timestamp = item
        try:
            tts = get_tts_for_lang(translation_dest)
            # Parameter für Multi-Speaker/Multilingual-Modelle
            tts_kwargs = {"text": translated}
            if hasattr(tts, 'speakers') and tts.speakers:
                tts_kwargs['speaker'] = tts.speakers[0]
            langs = getattr(tts, 'languages', []) or []
            if translation_dest in langs:
                tts_kwargs['language'] = translation_dest
            # Direktes Audio erzeugen
            audio = tts.tts(**tts_kwargs)
            # Abspielen
            sd.play(audio, samplerate=22050)
            sd.wait()
            # Optional speichern
            if save_to_file:
                fname = f"translation_audio_{timestamp}.wav"
                wav_write(fname, SAMPLE_RATE, (audio * np.iinfo(np.int16).max).astype(np.int16))
            print(f"🔊 [Translator-TTS] Audio wiedergegeben für {timestamp}")
        except Exception as e:
            print(f"⚠️ TTS der Übersetzung fehlgeschlagen: {e}")
        translation_tts_queue.task_done()

# =============================
# Hauptprogramm
# =============================
if __name__ == "__main__":
    threading.Thread(target=record_loop, daemon=True).start()
    threading.Thread(target=transcribe_loop, daemon=True).start()
    threading.Thread(target=translation_loop, daemon=True).start()
    threading.Thread(target=translation_tts_loop, daemon=True).start()

    while True:
        print(f"\nSpeichern nach Datei: {'Ja' if save_to_file else 'Nein'}")
        print("1: Toggle save_to_file")
        print("2: Schreibe Text und höre es (Text-to-Speech)")
        print(f"3: Zielsprache für Übersetzung setzen (aktuell: {translation_dest})")
        print("0: Beenden")
        choice = input("Auswahl: ").strip()

        if choice == "1":
            save_to_file = not save_to_file
            print(f"✅ save_to_file = {save_to_file}")
        elif choice == "2":
            text = input("Gib den Text ein: ").strip()
            lang = input("Sprache wählen (de/en/ru/ua): ").strip().lower()
            if lang not in TTS_MODEL_MAP:
                print("⚠️ Sprache nicht unterstützt, nutze 'de'.")
                lang = "de"
            tts = get_tts_for_lang(lang)
            # direkt audio array erzeugen
            audio = tts.tts(text)
            sd.play(audio, samplerate=SAMPLE_RATE)
            sd.wait()
            if save_to_file:
                wav_write("output.wav", SAMPLE_RATE, (audio * np.iinfo(np.int16).max).astype(np.int16))
        elif choice == "3":
            translation_dest = input("Neue Übersetzungssprache (en/de/ru/ua): ").strip().lower()
            print(f"✅ Zielsprache gesetzt auf: {translation_dest}")
        elif choice == "0":
            audio_queue.put(None)
            translation_queue.put(None)
            translation_tts_queue.put(None)
            break
        else:
            print("Ungültige Eingabe.")

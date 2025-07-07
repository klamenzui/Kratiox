# telegram_bot.py
import os, requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from io import BytesIO
from pathlib import Path

# 1) Bestimme den Pfad zum Projekt-Root (wo dein main.py liegt)
BASE_DIR = Path(__file__).parent.resolve()
# 2) Baue den Pfad zur lokalen ffmpeg/bin
FFMPEG_BIN = BASE_DIR / "ffmpeg" / "bin"

# 3) Hänge ihn ans PATH-Environment an
os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

BOT_TOKEN = "8179518415:AAFmSjJA_7vii6AYjwVtvHmPCV67eLt3zDA"
CHATAPI_URL = "http://localhost:8004/chat"
STT_URL       = "http://localhost:8001/transcribe"
TTS_URL       = "http://localhost:8003/synthesize"


# Aufruf deines Chat-Service
def ask_kratix(chat_id, user, text):
    resp = requests.post(CHATAPI_URL, json={"chat_id": str(chat_id), "user": user, "message": text})
    resp.raise_for_status()
    print(resp.json())
    return resp.json()["reply"]


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hallo! Ich bin Kratix. Frag mich was!")


# alle Text-Nachrichten
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user  # Shortcut für update.message.from_user
    # Telegram-Nickname (Usernamen), falls gesetzt:
    username = user.username  # str oder None
    # Fallback: Vor- und Nachname zusammen:
    full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()

    # Jetzt kannst du beides verwenden:
    print(f"Eingehende Nachricht von @{username} ({full_name})")

    text = update.message.text
    chat_id = update.effective_chat.id
    # Rufe deinen Chat-Service auf
    reply = ask_kratix(chat_id, full_name, text)  # oder nur text
    await update.message.reply_text(reply)

# Utility: Whisper‐STT auf deinem Service aufrufen
def call_stt(pcm_bytes: bytes) -> dict:
    resp = requests.post(
        STT_URL,
        data=pcm_bytes,
        headers={"Content-Type": "application/octet-stream"}
    )
    resp.raise_for_status()
    res = resp.json()
    return res
# Handler für Sprachnachrichten
async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    name    = user.username or f"{user.first_name or ''} {user.last_name or ''}".strip()
    chat_id = update.effective_chat.id
    voice   = update.message.voice

    # 1) Lade OGG‐File in BytesIO
    file_obj = await context.bot.get_file(voice.file_id)
    ogg_bytes = await file_obj.download_as_bytearray()
    ogg_buf = BytesIO(ogg_bytes)

    # 2) Konvertiere OGG→WAV@16kHz mono
    from pydub import AudioSegment
    audio = AudioSegment.from_file(ogg_buf, format="ogg")
    audio = (
        audio
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )
    wav_buf = BytesIO()
    audio.export(wav_buf, format="wav")
    pcm_bytes = wav_buf.getvalue()


    # 3) STT: Whisper‐Service ansprechen
    try:
        res = call_stt(pcm_bytes)
        text = res.get("text")
        language = res.get("language")
    except Exception as e:
        print(e)
        return #await update.message.reply_text(f"STT fehlgeschlagen: {e}")

    # 4) Rückmeldung der Transkription, dann ins Chat‐Service
    #await update.message.reply_text(f"📝 Du hast gesagt:\n“{text}”")
    print(f"📝 Du hast auf {language} gesagt:\n“{text}”")
    # 5) Frage weiter an deinen Chat‐Service
    reply = ask_kratix(chat_id, name, text)
    #await update.message.reply_text(reply)
    await send_voice(update, context, language, reply)

async def send_voice(update, context, language, answer):
    try:
        # a) WAV-Bytes vom TTS-Service
        r = requests.post(TTS_URL, json={"text": answer, "lang": language}, stream=True)
        r.raise_for_status()
        wav_bytes = b"".join(r.iter_content(4096))
        from pydub import AudioSegment
        # b) In AudioSegment laden
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

        # c) Für Telegram Voice muss es Opus in OGG sein @48 kHz, Mono
        opus_buf = BytesIO()
        audio = audio.set_frame_rate(48000).set_channels(1)
        audio.export(
            opus_buf,
            format="ogg",
            codec="libopus"  # switch to libopus, non-experimental
        )

        opus_buf.seek(0)
        chat_id = update.effective_chat.id
        # 4) Als Sprachnachricht schicken
        await context.bot.send_voice(
            chat_id=chat_id,
            voice=opus_buf,
            caption=None  # oder z.B. answer[:100]
        )
    except Exception as e:
        # falls etwas schiefgeht, loggen und weitermachen
        print("⚠️ Sprachantwort fehlgeschlagen:", e)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    # Voice‐Messages
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    print("⭐ Telegram-Bot läuft…")
    app.run_polling()

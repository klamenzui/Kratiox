# telegram_bot.py
import numpy as np
from pathlib import Path
import os
from io import BytesIO

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from pydub import AudioSegment

from brain import KratixBrain


class TelegramBot:
    def __init__(self, token: str):
        # ensure ffmpeg from project/ffmpeg/bin is on PATH
        base = Path(__file__).parent
        os.environ["PATH"] = str(base / "ffmpeg" / "bin") + os.pathsep + os.environ["PATH"]

        self.brain = KratixBrain()
        self.brain.start()

        self.app = ApplicationBuilder().token(token).build()
        self._register_handlers()

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self.on_voice))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hallo! Ich bin Kratix, dein KI‐Assistent.")

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        name = user.username or f"{user.first_name} {user.last_name}".strip()
        chat_id = update.effective_chat.id
        text = update.message.text
        try:
            reply = self.brain.call_chat(text, chat_id=str(chat_id), user_id=name)
            await update.message.reply_text(reply)
        except Exception as e:
            print(e)

    async def on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        name = user.username or f"{user.first_name} {user.last_name}".strip()
        chat_id = update.effective_chat.id

        file_obj = await context.bot.get_file(update.message.voice.file_id)
        ogg = await file_obj.download_as_bytearray()
        audio = AudioSegment.from_file(BytesIO(ogg), format="ogg")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav = BytesIO()
        audio.export(wav, format="wav")
        wav_bytes = wav.getvalue()

        # STT
        lang, text = self.brain.call_service("stt", data=wav_bytes)
        # Chat
        reply = self.brain.call_chat(text, chat_id=str(chat_id), user_id=name)

        # TTS back to user
        pcm = self.brain.call_service("tts",
                                       json={"text": reply, "lang": lang},
                                       stream=True)
        # convert float32 back to int16 WAV bytes & then to OGG/Opus @48kHz
        raw_wav = (pcm * np.iinfo(np.int16).max).astype(np.int16).tobytes()
        reply_audio = AudioSegment.from_raw(BytesIO(raw_wav),
                                            sample_width=2, frame_rate=22050, channels=1)
        opus_buf = BytesIO()
        reply_audio.set_frame_rate(48000).export(
            opus_buf, format="ogg", codec="libopus")
        opus_buf.seek(0)

        await context.bot.send_voice(chat_id, voice=opus_buf)

    def run(self):
        print("⭐ Telegram‐Bot läuft…")
        self.app.run_polling()


if __name__ == "__main__":
    token = os.getenv("BOT_TOKEN") or "8179518415:AAFmSjJA_7vii6AYjwVtvHmPCV67eLt3zDA"
    TelegramBot(token=token).run()

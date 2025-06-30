# telegram_bot.py
import os, requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN   = "8179518415:AAFmSjJA_7vii6AYjwVtvHmPCV67eLt3zDA"
CHATAPI_URL = "http://localhost:8004/chat"


# Aufruf deines Chat-Service
def ask_kratix(chat_id, user, text):
    resp = requests.post(CHATAPI_URL, json={"chat_id":str(chat_id), "user": user, "message":text})
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

    text    = update.message.text
    chat_id = update.effective_chat.id
    # Rufe deinen Chat-Service auf
    reply   = ask_kratix(chat_id, full_name, text)  # oder nur text
    await update.message.reply_text(reply)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print("⭐ Telegram-Bot läuft…")
    app.run_polling()

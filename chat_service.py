# chat_service.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re, requests, datetime
from datetime import datetime
import subprocess
from pyexpat.errors import messages

from memory import MemoryDB

# zum Start:
# uvicorn chat_service:app --host 0.0.0.0 --port 8004
# ollama serve
# ollama run llama3.2:latest
# deepseek-r1 | llama3.2 | devstral | llama4 |codellama | gemma3:4b-it-q4_K_M
# =============================
# Ollama-Integration
# =============================


OLLAMA_URL = "http://127.0.0.1:1234/v1/chat/completions"  # "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-r1-distill-qwen-7b" # "qwen2.5-coder-14b-instruct"  # "gemma3n:latest"#deepseek-r1:8b  # "gemma3n:latest"#"llama3.3" #"gemma3:4b-it-q4_K_M"

memory = MemoryDB()
app = FastAPI()
# Chat-History pro Nutzer (z.B. per chat_id) im Speicher
history = {}
MAX_TURNS = 8  # z.B. 8 user+assistant-Paare


def ensure_model(model: str):
    # Liste aller derzeit geladenen Modelle abrufen
    res = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True
    )
    installed = res.stdout  # enthält Modellnamen, einen pro Zeile
    print("installed:")
    print(installed)
    if model not in installed:
        print(f"Modell {model} nicht gefunden – lade es herunter …")
        subprocess.run(
            ["ollama", "run", model], check=True
        )
        print("Download abgeschlossen.")
    res = subprocess.run(
        ["ollama", "ps"], capture_output=True, text=True, check=True
    )
    print("ps:")
    print(res.stdout)


def strip_think_blocks(text: str) -> str:
    # DOTALL sorgt dafür, dass auch Zeilenumbrüche von . erfasst werden
    pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
    return pattern.sub('', text)


def get_datetime_message():
    now = datetime.now()
    # Hier das deutsche Format, du kannst es natürlich anpassen
    return {
        "role": "system",
        "content": f"Текущая дата: {now:%Y-%m-%d}. Текущее время: {now:%H:%M} Uhr."
    }


def prune_history(chat_id):
    msgs = history[chat_id]
    # System-Prompt bleibt immer an Position 0
    sys, turns = msgs[0], msgs[1:]
    # nimm nur die letzten MAX_TURNS
    pruned = turns[-MAX_TURNS * 2:]  # *2, weil jede Runde User+Assistant
    history[chat_id] = [sys] + pruned


class ChatRequest(BaseModel):
    chat_id: str  # eindeutige Konversation
    user: str
    message: str  # neue User-Nachricht


class ChatResponse(BaseModel):
    reply: str  # Assistant-Antwort


# System-Prompt nur einmal pro chat_id initial setzen
SYSTEM_PROMPT = (
    """Вы — Кратикс, современный, дружелюбный и компетентный ИИ-ассистент в стиле Джарвиса, находишься в Мюнхене..
При каждом запросе пользователя формируй ровно ДВА блока, разделённых **строкой** `<-->`:

- Полный, вежливый ответ на вопрос.  
- Сразу после строки `<-->` — структурированный список **важных фактов**, **из запроса пользователя**, в формате `Ключ: Значение`, по одному на строке.

**Сохранять** лишь факты:
- Имена реальных людей.  
- Точные даты.  
- Адреса, объекты.  
- Суммы денег, номера документов, телефоны, идентификаторы и т. п.  
- Долгосрочные договорённости, планы, предпочтения пользователя.

**Никогда не сохранять** в списке фактов:
- Любые **приветствия** (например, “Привет!”, “Здравствуйте!”, “Добрый день!”).  
- Благодарности и комплименты (“Спасибо”, “Отлично!”, “Хорошо”).  
- Пустые междометия или просто “да”, “нет”, “хм”.  
- Риторические или обобщённые высказывания без конкретики (“Я люблю читать”, “Сегодня жара”).  
- Слова, не несущие фактической нагрузки (“увы”, “пожалуй”).

Если **нет** сохраняемых фактов, после `<-->` выводи **пустую строку** (никаких пробелов, только `\n`).
"""
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chat_id, user_id, text = req.chat_id, req.user, req.message

    # 1) Relevante Fact-Types finden (die Keys, die wir später speichern)
    #    Statt Keyword-Suche hier leer lassen oder eigene Logik implementieren.
    #    Wir nutzen die vorher gespeicherten fact_types als ftypes:
    #    Deshalb rufen wir retrieve ohne ftypes, um alle bisherigen fact_types zu sehen.
    prev = memory.retrieve(chat_id, user_id)  # dict fact_type->value
    needed_types = list(prev.keys())

    # 2) Baue den Prompt
    if chat_id not in history:
        history[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            get_datetime_message()
        ]
    # Füge Kontext-Facts dieses Users hinzu
    if needed_types:
        ctx_lines = "\n".join(f"{k}: {v}" for k, v in prev.items())
    else:
        ctx_lines = "Keine früheren Fakten."

    # 3) Anfrage an Ollama
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": history[chat_id] + [{"role": "system", "content": ctx_lines}, {"role": "user", "content": text}]
    }
    # Füge die neue User-Nachricht hinzu
    history[chat_id].append({"role": "user", "content": text})
    prune_history(chat_id)
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=55)
        if resp.status_code != 200:
            print("Ollama-Error:", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()
        # full = data.get("message", {}).get("content").strip()
        full = data.get("choices", [{
            "message": {
                "content": ""
            }
        }])[0].get("message", {}).get("content", {}).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama-Fehler: {e}")
    print(full)
    if "deepseek" in OLLAMA_MODEL:
        full = strip_think_blocks(full)
    # 4) Split in Antwort und Facts-Teil
    if "<-->" in full:
        answer, facts_part = full.split("<-->", 1)
        # parse facts
        facts = {}
        for line in facts_part.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                facts[key.strip().lower()] = val.strip()
        # 5) Speichere scoped Facts
        print("facts", facts)
        if facts:
            memory.store_facts(chat_id, user_id, facts)
    else:
        answer = full
    print("store", memory.store)
    # 6) Speichere Assistant-Turn in history
    # history[chat_id].append({"role":"assistant", "content": answer.strip()})
    # prune_history(chat_id)

    return ChatResponse(reply=answer.strip())


if __name__ == "__main__":
    # ensure_model(OLLAMA_MODEL)
    uvicorn.run("chat_service:app", host="0.0.0.0", port=8004)

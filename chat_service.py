# chat_service.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re, requests, datetime
from datetime import datetime

from pyexpat.errors import messages

from memory import MemoryDB

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "deepseek-r1:8b"#"gemma3n:latest"#"llama3.3" #"gemma3:4b-it-q4_K_M"

memory = MemoryDB()
app = FastAPI()
# Chat-History pro Nutzer (z.B. per chat_id) im Speicher
history = {}
MAX_TURNS = 8  # z.B. 8 user+assistant-Paare

def strip_think_blocks(text: str) -> str:
    # DOTALL sorgt dafür, dass auch Zeilenumbrüche von . erfasst werden
    pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
    return pattern.sub('', text)

def get_datetime_message():
    now = datetime.now()
    # Hier das deutsche Format, du kannst es natürlich anpassen
    return {
        "role": "system",
        "content": f"Ты находишься в Мюнхене. Текущая дата: {now:%Y-%m-%d}. Текущее время: {now:%H:%M} Uhr."
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
    # "Du bist Kratix, ein hochmoderner, freundlicher und äußerst kompetenter KI-Assistent, "
    # "ganz ähnlich wie Jarvis. Antworte stets klar, übersichtlich und höflich."
    # "ты - Кратикс, ультрасовременный, дружелюбный и чрезвычайно компетентный помощник ИИ, "
    # "очень похожий на Джарвиса. Всегда отвечайте четко, лаконично и дружелюбно. "

    """Вы — Кратикс, современный, дружелюбный и компетентный ИИ-ассистент в стиле Джарвиса.
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
"""Du bist Kratix, ein moderner, freundlicher und kompetenter KI-Assistent im Jarvis-Stil.

Bei jeder Nutzeranfrage erzeugst du ZWEI Teile, getrennt durch eine eigene Zeile mit exakt:
<-->

1) Eine ausführliche, höfliche Antwort auf die Frage.
2) Direkt nach der Trennzeile eine strukturierte Auflistung aller wichtigen Fakten im Key-Value-Format. 
   Jeder Eintrag auf einer neuen Zeile, z. B.:
   Name: Username
   Objekt: Mussterstr. 3 Nummer 28
   Datum: 2025-06-29
   Kosten: 6000

Wenn keine speicherwürdigen Infos im Text von Benutzer sind, gibst du nach `<-->` eine **leere Zeile** aus (keine weiteren Zeichen)."""


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
            {"role":"system", "content": SYSTEM_PROMPT},
            get_datetime_message()
        ]
    # Füge Kontext-Facts dieses Users hinzu
    if needed_types:
        ctx_lines = "\n".join(f"{k}: {v}" for k, v in prev.items())
    else:
        ctx_lines = "Keine früheren Fakten."


    # 3) Anfrage an Ollama
    payload = {
        "model":    OLLAMA_MODEL,
        "stream":   False,
        "messages": history[chat_id] + [{"role": "system", "content": ctx_lines}, {"role":"user", "content": text}]
    }
    # Füge die neue User-Nachricht hinzu
    history[chat_id].append({"role":"user", "content": text})
    prune_history(chat_id)
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        full = data.get("message", {}).get("content").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama-Fehler: {e}")
    print(full)
    full = strip_think_blocks(full)
    # 4) Split in Antwort und Facts-Teil
    if "<-->" in full:
        answer, facts_part = full.split("<-->", 1)
        # parse facts
        facts = {}
        for line in facts_part.strip().splitlines():
            if ":" in line:
                key, val = line.split(":",1)
                facts[key.strip().lower()] = val.strip()
        # 5) Speichere scoped Facts
        print("facts",facts)
        if facts:
            memory.store_facts(chat_id, user_id, facts)
    else:
        answer = full
    print("store",memory.store)
    # 6) Speichere Assistant-Turn in history
    #history[chat_id].append({"role":"assistant", "content": answer.strip()})
    #prune_history(chat_id)

    return ChatResponse(reply=answer.strip())


if __name__ == "__main__":
    uvicorn.run("chat_service:app", host="0.0.0.0", port=8004)
# zum Start:
# uvicorn chat_service:app --host 0.0.0.0 --port 8004
# ollama serve
# ollama run llama3.2:latest
# deepseek-r1 | llama3.2 | devstral | llama4 |codellama | gemma3:4b-it-q4_K_M
# =============================
# Ollama-Integration
# =============================
"""OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3:4b-it-q4_K_M"
# OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_URL = "http://localhost:11434/api/chat"
history = [
    {"role":"system", "content":
     "Du bist Kratix, ein hochmoderner, freundlicher und äußerst kompetenter KI-Assistent, "
     "ganz ähnlich wie Jarvis. Antworte stets klar, übersichtlich und höflich."}
]

def ensure_ollama():
    #Stellt sicher, dass 'ollama serve' läuft.
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", OLLAMA_PORT))
        s.close()
    except ConnectionRefusedError:
        print("▶ Ollama läuft noch nicht – starte 'ollama serve' …")
        subprocess.Popen(["ollama", "serve"])
        # kurz warten, bis der Server hoch ist
        time.sleep(2)

def ask_kratix(user_input: str):
    # 2) häng die neue User-Nachricht an
    history.append({"role":"user", "content": user_input})
    # 3) schick die gesamte History

    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": history
            },
            timeout=10
        )
        r.raise_for_status()
        resp = r.json()
        assistant_msg = resp.get("message", {}).get("content")  # oder resp["choices"][0]["message"]["content"]
        # 4) speicher die Assistant-Antwort in der History
        history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg.strip()
    except Exception as e:
        print("⚠️ Ollama-Check fehlgeschlagen:", e)
        return user_input

def call_ollama_http(text: str, timeout: float = 5.0) -> str:
    ""prompt = (
        "Du bist ein Korrektur-Tool. "
        "Überprüfe den folgenden erkannten Text auf Erkennungsfehler und "
        "gib nur den korrigierten Text aus:\n\n"
        f"{text}"
    )""
    prompt = ""
    Du bist Kratix, ein hochmoderner, freundlicher und äußerst kompetenter KI-Assistent, ganz ähnlich wie Jarvis. 
    Deine Aufgabe ist es, deinem Nutzer bei allen Fragen und Aufgaben zu helfen: technische Unterstützung, Recherchen, Code-Beispiele, Terminplanung und mehr. 
    Antworte stets klar, übersichtlich und höflich. 
    Verwende einen professionellen, dennoch persönlichen Tonfall, und biete bei Bedarf Nachfragen an, um das Problem besser zu verstehen. 
    Nutze dein breites Wissen und verliere nie die ruhige, hilfreiche „Jarvis“-Manier.
    ""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt + "\n\n### Nutzer:\n" + text,
        "stream": False,
        # optional: temperature bzw. andere Optionen
        "options": {"temperature": 0.0}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("response", "").strip()
        return data
    except Exception as e:
        print("⚠️ Ollama-Check fehlgeschlagen:", e)
        return text
"""

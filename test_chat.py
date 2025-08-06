import json
import ollama
from collections import deque
import sqlite3
import spacy

# Konfiguration
MODEL_NAME = "llama3.1"
MAX_HISTORY = 10  # maximale Anzahl Nachrichten im Kontext

# Message-Queue: speichert die letzten MAX_HISTORY Nachrichten
message_history = deque(maxlen=MAX_HISTORY)

# Externer Speicher für wichtige Informationen (hier In-Memory-SQLite)
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.execute(
    'CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)'
)
conn.commit()

# NLP-Pipeline laden (deutsches Modell für NER)
nlp = spacy.load("de_core_news_sm")

# Funktion: Automatische Extraktion wichtiger Infos via NER
def extract_and_store_important_info(user_message: str):
    doc = nlp(user_message)
    print([(w.text, w.pos_) for w in doc])
    for ent in doc.ents:
        print("--", ent.label_, "=", ent.text)
        # Speichere nur relevante Entitätstypen
        if ent.label_ in {"PER", "ORG", "LOC", "GPE", "DATE", "TIME"}:
            key = ent.label_  # z.B. "DATE"
            value = ent.text
            print(key, "=", value)


# Beispielaufrufe
if __name__ == '__main__':
    extract_and_store_important_info("es geht mir gut")
    extract_and_store_important_info("wie geht es dir?")
    extract_and_store_important_info("ich heisse jak und ich wohne in münchen")
    extract_and_store_important_info("Unser Projekt Apollo startet am 12. August 2025.")
    extract_and_store_important_info("Ich treffe mich mit Dr. Müller in Berlin um 14:00.")
    extract_and_store_important_info("Was ist der aktuelle Stand?")

# Hinweis:
# - Das Skript nutzt spaCy für die automatische Erkennung von Personen (PER), Organisationen (ORG),
#   Orten (LOC/GPE), Datum (DATE) und Zeit (TIME).
# - Erfasste Entitäten werden in der SQLite-DB abgelegt und bei jedem Aufruf in den System-Prompt eingefügt.

import requests

OLLAMA_PORT = 11434
OLLAMA_MODEL = "gemma3:4b-it-q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama_http(text: str, timeout: float = 5.0) -> str:
    prompt = (
        "Du bist ein Korrektur-Tool. "
        "Überprüfe den folgenden erkannten Text auf Erkennungsfehler und "
        "gib nur den korrigierten Text aus:\n\n"
        f"{text}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
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

raw_text = "Das haus ist schön und gemütlich. ich würde hier wohn"
corrected = call_ollama_http(raw_text)
print("Ollama →", corrected)
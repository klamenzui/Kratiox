import requests


"""


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

"""
from fetcher import InternetFetcher


class Test:
    def __init__(self):
        # …
        self.fetcher = InternetFetcher(timeout=3.0, max_retries=2)

    def some_api_call(self):
        # Beispiel: hole Coin-Preise von CoinGecko
        data = self.fetcher.get_json(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "solana", "vs_currencies": "usd"}
        )
        print(data)
        price = data.get("solana", {}).get("usd")
        return price

    def some_web_scrape(self):
        # Beispiel: scrappe eine Wetterseite
        soup = self.fetcher.get_html("https://example.com/weather/today")
        temp = soup.select_one(".temp").get_text()
        return temp

    def google_search(self, q):
        # Beispiel: scrappe eine Wetterseite
        result = self.fetcher.google_search(q)
        i = 0
        for k, v in result.items():
            with open(f"{i}.html", "w", encoding="utf-8") as f:
                f.write(v)
            i += 1
        return result


#print(Test().some_api_call())
#print(Test().google_search("weather in Hemau Germany today"))


#web_search("weather in Hemau Germany", "de-DE")
#web_search("sol price")
#web_search("what is python")
fetcher = InternetFetcher()

#fetcher.web_search({'query': 'current president of USA', 'region': 'us-en', 'timelimit': 'd', 'max_results': 3})
import json
import requests
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:1.5b"
def download_model(model_name: str):
    print(f"🔄 Lade Modell '{model_name}' herunter...")
    resp = requests.post(
        f"{OLLAMA_URL}/api/pull",
        json={"name": model_name},
        stream=True,
    )

    if resp.status_code != 200:
        print(f"❌ Fehler beim Herunterladen: {resp.status_code} - {resp.text}")
        return

    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "status" in data:
                    print(f"[{data['status']}]")
                elif "completed" in data and "total" in data:
                    done = data["completed"]
                    total = data["total"]
                    percent = int(done / total * 100)
                    print(f"📥 {done}/{total} Bytes ({percent}%)")
            except Exception as e:
                print(f"⚠️ Fehler beim Parsen: {e}")

    print("✅ Modell wurde vollständig heruntergeladen.")

def generate():
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": MODEL_NAME,  # oder dein Modellname
        "prompt": "Was ist der Sinn des Lebens?"
    })
    print(resp.text)
    return resp.json()
def models_list():
    resp = requests.get(f"{OLLAMA_URL}/api/tags")
    return resp.json()
#export OLLAMA_MODELS=/media/klamenzui/OS/dev/projects/python/Kratiox/ollama
#download_model(MODEL_NAME)
#models_list()
print(generate())

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
            i+=1
        return result


#print(Test().some_api_call())
#print(Test().google_search("weather in Hemau Germany today"))

from ddgs import DDGS

def web_search(query: str, region=None, max_results: int = 3) -> str:
    """
    Führt eine DuckDuckGo-Suche durch und gibt Title+Snippet + URL der Top-Ergebnisse zurück.
    """
    results = DDGS().text(query, region=region, max_results=max_results)
    if not results:
        return f"Keine Ergebnisse gefunden für «{query}»."
    lines = []
    i=0
    for r in results:
        title = r.get("title", "").strip()
        body  = r.get("body", "").strip()    # kurzer Auszug
        url   = r.get("href", "").strip()
        lines.append(f"{title}\n{body}\n{url}\n")
        with open(f"{i}.dd.html", "w", encoding="utf-8") as f:
            f.write(f"{title}\n{body}\n{url}\n")
        i+=1
    return "\n".join(lines)

web_search("weather in Hemau Germany", "de-DE")
#web_search("sol price")
#web_search("what is python")

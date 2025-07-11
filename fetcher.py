# fetcher.py
import time
import requests
from functools import lru_cache
from urllib.parse import urljoin
from requests.exceptions import RequestException
#from googlesearch import search
from ddgs import DDGS
try:
    from bs4 import BeautifulSoup

    HAS_BS = True
except ImportError:
    HAS_BS = False


class InternetFetcher:
    def __init__(self, base_url: str = "", timeout: float = 5.0, max_retries: int = 3, backoff: float = 0.5):
        """
        :param base_url: ein optionaler Basis‐URL, auf den relative Pfade gemountet werden können.
        :param timeout:  Timeout in Sekunden für jeden Request.
        :param max_retries: Bei Fehlern bis zu dieser Anzahl automatisch wiederholen.
        :param backoff:    exponentieller Backoff in Sekunden (wird multipliziert mit Retry‐Nr).
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.session = requests.Session()
        # z. B. hier Header zentral setzen:
        self.session.headers.update({
            "User-Agent": "KratixFetcher/1.0 (+https://example.com)"
        })

    def _full_url(self, path: str) -> str:
        return urljoin(self.base_url, path)

    def get_json(self, path: str, params: dict = None, headers: dict = None) -> dict:
        """
        Holt JSON von einer API.
        Wirft ValueError, wenn kein gültiges JSON zurückkommt.
        """
        url = self._full_url(path)
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (RequestException, ValueError) as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)

    def post_json(self, path: str, data=None, json=None, headers: dict = None) -> dict:
        """
        POST mit JSON-Antwort.
        """
        url = self._full_url(path)
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(url, data=data, json=json, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except (RequestException, ValueError):
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)

    @lru_cache(maxsize=128)
    def get_text(self, path: str, params: tuple = None) -> str:
        """
        Einfacher GET, gibt reinen Text zurück.
        Wird hier gecached, damit wiederholte Aufrufe schnellen Zugriff haben.
        """
        url = self._full_url(path)
        r = self.session.get(url, params=dict(params or []), timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def get_html(self, path: str, params: tuple = None) -> "BeautifulSoup":
        """
        Parsen einer HTML‐Seite via BeautifulSoup (falls installiert).
        """
        text = self.get_text(path, params)
        if not HAS_BS:
            raise RuntimeError("BeautifulSoup (bs4) nicht installiert")
        return BeautifulSoup(text, "html.parser")

    def web_search(self, params) -> str:
        """
        Gibt die URLs der ersten 'num' Google-Treffer zurück.
        """
        """
            Führt eine DuckDuckGo-Suche durch und gibt Title+Snippet + URL der Top-Ergebnisse zurück.
            """
        results = DDGS().text(**params)
        if not results:
            return f"Keine Ergebnisse gefunden für «{params}»."
        lines = []
        i = 0
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()  # kurzer Auszug
            url = r.get("href", "").strip()
            lines.append(f"{title}\n{body}\n{url}\n")
            with open(f"{i}.dd.html", "w", encoding="utf-8") as f:
                f.write(f"{title}\n{body}\n{url}\n")
            i += 1
        return "\n - ".join(lines)

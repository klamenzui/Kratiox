#import sqlite3
from datetime import datetime

class MemoryDB:
    def __init__(self):
        # Struktur: { chat_id: { user_id: { fact_type: [(value, timestamp), …] } } }
        self.store = {}

    def store_facts(self, chat_id: str, user_id: str, facts: dict):
        """Facts ist dict fact_type->value."""
        self.store.setdefault(chat_id, {}).setdefault(user_id, {})
        user_facts = self.store[chat_id][user_id]
        ts = datetime.utcnow().isoformat()
        for ftype, value in facts.items():
            user_facts.setdefault(ftype, []).append((value, ts))

    def retrieve(self, chat_id: str, user_id: str, ftypes: list[str]=None):
        """Gib nur die angefragten fact_types für genau diesen user zurück."""
        chat = self.store.get(chat_id, {})
        user = chat.get(user_id, {})
        if not ftypes:
            return user  # alle Facts dieses users
        return {k: user.get(k) for k in ftypes if k in user}

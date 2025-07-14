#import sqlite3
import json
import os
from datetime import datetime
class MemoryDB:
    def __init__(self):
        self.memory_path = 'memory.json'
        self.settings_path = 'settings.json'
        # Struktur: { chat_id: { user_id: { fact_type: [(value, timestamp), …] } } }
        self.store = {}
        self.settings = {}
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as file:
                content = file.read()
                self.store = json.loads(content)

        if os.path.exists(self.settings_path):
            with open(self.settings_path, "r") as file:
                content = file.read()
                self.settings = json.loads(content)

    def store_facts(self, chat_id: str, user_id: str, facts: dict):
        """Fügt nur hinzu wenn Wert sich geändert hat."""
        self.store.setdefault(chat_id, {}).setdefault(user_id, {})
        user_facts = self.store[chat_id][user_id]
        ts = datetime.now().isoformat()

        for ftype, value in facts.items():
            history = user_facts.setdefault(ftype, [])
            # Nur hinzufügen wenn sich der Wert geändert hat
            if not history or history[-1][0] != value:
                history.append((value, ts))
        with open(self.memory_path, 'w') as json_file:
            json_file.write(json.dumps(self.store, indent=4))

    def store_settings(self, settings: dict):


        for k, value in settings.items():
            self.settings[k] = value

        with open(self.settings_path, 'w') as json_file:
            json_file.write(json.dumps(self.settings, indent=4))

    def retrieve(self, chat_id: str, user_id: str, ftypes: list[str]=None):
        """Gib nur die angefragten fact_types für genau diesen user zurück."""
        chat = self.store.get(chat_id, {})
        user = chat.get(user_id, {})
        if not ftypes:
            return user  # alle Facts dieses users
        return {k: user.get(k) for k in ftypes if k in user}

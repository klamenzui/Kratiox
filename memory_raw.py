# memory_db.py
import sqlite3
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def resolve_group_id(self, chat_id, user_id, raw_gid: str) -> str:
        g = self.get_group(chat_id, user_id, raw_gid.replace("?", ""))
        # wenn neu, Raw beginnt mit '?'
        if not g:
            return uuid4().hex
        return g["group_id"]

    def _init_schema(self):
        c = self.conn.cursor()
        # -- verhindert doppelte Zeilen für identische (chat_id, user_id, fact_type, value)
        c.execute("""CREATE TABLE IF NOT EXISTS summaries (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id    TEXT    NOT NULL,
  user_id    TEXT    NOT NULL,
  text      TEXT    NOT NULL,       -- JSON-serialized
  timestamp  TEXT    NOT NULL,       -- ISO-Timestamp
);
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL          -- JSON-serialized dict
        )""")
        self.conn.commit()

    def store_settings(self, user_id: str, settings: Dict[str, Any]):
        data = json.dumps(settings, ensure_ascii=False)
        self.conn.execute("""
        INSERT INTO settings (user_id,data) VALUES (?,?)
        ON CONFLICT(user_id) DO UPDATE SET data=excluded.data
        """, (user_id, data))
        self.conn.commit()

    def get_settings(self, user_id: str) -> Dict[str, Any]:
        c = self.conn.cursor()
        c.execute("SELECT data FROM settings WHERE user_id=?", (user_id,))
        row = c.fetchone()
        return json.loads(row[0]) if row else {}

    def store_facts(self, chat_id, user_id, text):
        ts = datetime.now(timezone.utc).isoformat() + "Z"
        self.conn.execute("""
                    INSERT INTO facts (chat_id, user_id, text, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (chat_id, user_id, text, ts))
        self.conn.commit()

    def get_latest_facts(self, chat_id: str, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Liefert pro Gruppe (group_id) ein Dict aller aktuellen Facts:
          {
            "<group_id1>": {
                "type": "...",
                "parent_id": "...",         # falls gesetzt
                "<key1>": <value1>,
                "<key2>": <value2>,
                …
            },
            "<group_id2>": { … },
            …
          }
        """
        c = self.conn.cursor()
        # 1) Alle Gruppen des Users holen
        groups = self.get_groups(chat_id, user_id)
        if not groups:
            return {}

        # 2) Zu jeder Gruppe die jeweils neuesten Values je Key holen
        placeholders = ",".join("?" for _ in groups)
        params = list(groups.keys())
        c.execute(f"""
            SELECT f.group_id, f.key, f.value
            FROM facts AS f
            JOIN (
                SELECT group_id, key, MAX(timestamp) AS maxts
                FROM facts
                WHERE group_id IN ({placeholders})
                GROUP BY group_id, key
            ) AS sub
              ON f.group_id=sub.group_id
             AND f.key=sub.key
             AND f.timestamp=sub.maxts
        """, params)

        for row in c.fetchall():
            gid = row["group_id"]
            key = row["key"]
            val = json.loads(row["value"])
            groups[gid][key] = val

        return groups

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

    def _init_schema(self):
        c = self.conn.cursor()
        # -- verhindert doppelte Zeilen für identische (chat_id, user_id, fact_type, value)
        c.execute("""CREATE TABLE IF NOT EXISTS summaries (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id    TEXT    NOT NULL,
  user_id    TEXT    NOT NULL,
  category      TEXT    NOT NULL,       
  text      TEXT    NOT NULL,       
  timestamp  TEXT    NOT NULL       -- ISO-Timestamp
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

    def store_facts(self, chat_id, user_id, obj):
        type_ = obj.get("type")
        params = obj.get("data", {})
        params["chat_id"] = chat_id
        params["user_id"] = user_id
        params["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"

        if type_ == "INSERT":
            keys = list(params.keys())
            values = list(params.values())
            placeholders = ", ".join("?" for _ in keys)
            self.conn.execute(
                f"INSERT INTO summaries ({', '.join(keys)}) VALUES ({placeholders})",
                values
            )

        elif type_ == "UPDATE":
            self.conn.execute("""
                UPDATE summaries
                SET category = ?, text = ?, timestamp = ?
                WHERE id = ?
            """, (params["category"], params["text"], params["timestamp"], obj.get("id")))

        elif type_ == "DELETE":
            ids = obj.get("ids", [])
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                self.conn.execute(
                    f"DELETE FROM summaries WHERE id IN ({placeholders})", ids
                )

        self.conn.commit()


    def get_latest_facts(self, chat_id: str, user_id: str) -> Dict[str, Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(f"""
            SELECT id, chat_id, user_id, category, text, timestamp FROM summaries
            WHERE chat_id=? and user_id=?
            ORDER BY timestamp
        """, (chat_id, user_id))
        groups = {user_id: {}}
        for row in c.fetchall():
            groups[user_id][row["id"]] = {
                "id": row["id"],
                "category": row["category"],
                "text": row["text"],
                "timestamp": row["timestamp"]
            }
        return groups

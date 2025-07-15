# memory_db.py
import sqlite3
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id   TEXT NOT NULL,
            user_id   TEXT NOT NULL,
            fact_type TEXT NOT NULL,
            value     TEXT NOT NULL,       -- JSON-serialized
            timestamp TEXT NOT NULL        -- ISO 8601
        );""")
        #-- verhindert doppelte Zeilen für identische (chat_id, user_id, fact_type, value)
        c.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_facts_unique
          ON facts(chat_id, user_id, fact_type, value);
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL          -- JSON-serialized dict
        )""")
        self.conn.commit()

    def store_fact(self, chat_id: str, user_id: str, fact_type: str, value: Any):
        """
        Fügt einen neuen Fact ein oder aktualisiert bei gleicher (chat_id,user_id,fact_type,value)
        nur den timestamp.
        """
        ts = datetime.now(timezone.utc).isoformat() + "Z"
        val_json = json.dumps(value, ensure_ascii=False)
        c = self.conn.cursor()
        c.execute("""
        INSERT INTO facts(chat_id, user_id, fact_type, value, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(chat_id, user_id, fact_type, value) DO UPDATE
          SET timestamp=excluded.timestamp
        """, (chat_id, user_id, fact_type, val_json, ts))
        self.conn.commit()

    def get_latest_facts(self,
                         chat_id: str,
                         user_id: str
                        ) -> Dict[str, Any]:
        """
        Liefert für jeden fact_type den neuesten Wert.
        Rückgabe: { fact_type: value, … }
        """
        c = self.conn.cursor()
        c.execute("""
        SELECT f.fact_type, f.value
        FROM facts AS f
        WHERE f.chat_id = ?
          AND f.user_id = ?
          AND f.timestamp = (
              SELECT MAX(timestamp)
              FROM facts
              WHERE chat_id = f.chat_id
                AND user_id  = f.user_id
                AND fact_type= f.fact_type
          );
        """, (chat_id, user_id))
        rows = c.fetchall()
        return {ft: json.loads(v) for ft, v in rows}

    def get_fact_history(self,
                         chat_id: str,
                         user_id: str,
                         fact_type: str
                        ) -> List[Dict[str, Any]]:
        """
        Historie für einen fact_type:
        [ {"value":…, "timestamp":…}, … ]
        """
        c = self.conn.cursor()
        c.execute("""
        SELECT value, timestamp
        FROM facts
        WHERE chat_id=? AND user_id=? AND fact_type=?
        ORDER BY timestamp
        """, (chat_id, user_id, fact_type))
        return [
            {"value": json.loads(val), "timestamp": ts}
            for val, ts in c.fetchall()
        ]

    def store_settings(self, user_id: str, settings: Dict[str,Any]):
        data = json.dumps(settings, ensure_ascii=False)
        self.conn.execute("""
        INSERT INTO settings (user_id,data) VALUES (?,?)
        ON CONFLICT(user_id) DO UPDATE SET data=excluded.data
        """, (user_id, data))
        self.conn.commit()

    def get_settings(self, user_id: str) -> Dict[str,Any]:
        c = self.conn.cursor()
        c.execute("SELECT data FROM settings WHERE user_id=?", (user_id,))
        row = c.fetchone()
        return json.loads(row[0]) if row else {}

    def store_facts(self, chat_id, user_id, obj: dict):
        for k, v in obj.items():
            self.store_fact(chat_id, user_id, k, v)

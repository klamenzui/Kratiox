# memory_db.py mit Logging + Thread-Schutz + SQLite-Cleanup
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4
import threading
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("memory.log"),
        logging.StreamHandler()
    ]
)

class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.chat_id = ""
        self.user_id = ""
        self.lang = "en"
        self.history = {}  # chat_id → [ messages ]
        self._lock = threading.Lock()
        self._init_schema()
        self._cleanup_old_data(days=30)

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS summaries (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id    TEXT    NOT NULL,
            user_id    TEXT    NOT NULL,
            category   TEXT    NOT NULL,
            text       TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON summaries(chat_id, user_id)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                user_id TEXT PRIMARY KEY,
                data    TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def _cleanup_old_data(self, days=30):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() + "Z"
        cur = self.conn.cursor()
        cur.execute("DELETE FROM summaries WHERE timestamp < ?", (cutoff,))
        deleted = cur.rowcount
        self.conn.commit()
        logging.info("[Cleanup] Removed %d old entries older than %d days", deleted, days)

    def set_context(self, chat_id, user_id, lang="en"):
        self.chat_id = chat_id
        self.user_id = user_id
        self.lang = lang

    def get_tpl_message(self, name, prompt_data):
        try:
            with open(f"./prompts/{self.lang}/{name}.txt", "r", encoding="utf-8") as f:
                message = f.read().strip()
        except FileNotFoundError:
            logging.warning("Prompt template not found: %s", name)
            return ""
        for k, v in prompt_data.items():
            message = message.replace(f'%{k}%', f'{v}')
        return message

    def get_history(self, content):
        with self._lock:
            sys_message = self.get_system_message()
            if self.chat_id not in self.history:
                self.history[self.chat_id] = [sys_message]
            else:
                self.history[self.chat_id][0] = sys_message
            self.history[self.chat_id].append({"role": "user", "content": content})
            return self.history[self.chat_id]

    def append_history(self, answer):
        with self._lock:
            self.history[self.chat_id].append({"role": "assistant", "content": answer.strip()})
            max_turns = 8
            turns = self.history[self.chat_id][1:]
            pruned = turns[-max_turns * 2:]
            self.history[self.chat_id] = [self.history[self.chat_id][0]] + pruned
            try:
                with open(f"./prompts/current_system_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(self.history[self.chat_id][0]["content"])
                with open(f"./prompts/current_history.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(self.history, indent=4))
            except Exception as e:
                logging.warning("Unable to write current_system_prompt.txt: %s", e)

    def get_system_message(self):
        facts_dict = self.get_latest_facts()
        facts = "\n".join(json.dumps(v, indent=4) for v in facts_dict.values()) if facts_dict else ""
        settings_dict = self.get_settings()
        settings = "\n".join(f"{k}: {v!r}" for k, v in settings_dict.items()) if settings_dict else ""
        now = datetime.now(timezone.utc)
        try:
            sys_prompt = self.get_tpl_message("system_prompt_raw", {
                "date": f"{now:%Y-%m-%d}",
                "time": f"{now:%H:%M}Z",
                "memory": facts,
                "settings": settings
            })
        except Exception as e:
            logging.error("System prompt error: %s", e)
            sys_prompt = "You are a helpful AI assistant."
        return {"role": "system", "content": sys_prompt}

    def store_settings(self, settings: Dict[str, Any]):
        data = json.dumps(settings, ensure_ascii=False)
        self.conn.execute("""
            INSERT INTO settings (user_id,data) VALUES (?,?)
            ON CONFLICT(user_id) DO UPDATE SET data=excluded.data
        """, (self.user_id, data))
        self.conn.commit()

    def get_settings(self) -> Dict[str, Any]:
        c = self.conn.cursor()
        c.execute("SELECT data FROM settings WHERE user_id=?", (self.user_id,))
        row = c.fetchone()
        return json.loads(row[0]) if row else {}

    def store_facts(self, cmd_list: list):
        for obj in cmd_list:
            type_ = obj.get("type")
            params = obj.get("data", {})
            params["chat_id"] = self.chat_id
            params["user_id"] = self.user_id
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

    def get_latest_facts(self) -> Dict[str, Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute("""
            SELECT id, chat_id, user_id, category, text, timestamp FROM summaries
            WHERE chat_id=? and user_id=?
            ORDER BY timestamp
        """, (self.chat_id, self.user_id))
        facts = {}
        for row in c.fetchall():
            facts[row["id"]] = {
                "id": row["id"],
                "category": row["category"],
                "text": row["text"],
                "timestamp": row["timestamp"]
            }
        return facts

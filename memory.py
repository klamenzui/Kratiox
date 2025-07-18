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
        c.execute("""
        CREATE TABLE IF NOT EXISTS groups (
  group_id   TEXT    PRIMARY KEY,
  chat_id    TEXT    NOT NULL,
  user_id    TEXT    NOT NULL,
  type       TEXT    NOT NULL,       -- z.B. "person", "event", …
  parent_id  TEXT    NULL,           -- verweist auf groups.group_id
  created_at TEXT    NOT NULL,       -- ISO-Timestamp
  FOREIGN KEY(parent_id) REFERENCES groups(group_id)
  );
""")
        # -- verhindert doppelte Zeilen für identische (chat_id, user_id, fact_type, value)
        c.execute("""CREATE TABLE IF NOT EXISTS facts (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  group_id   TEXT    NOT NULL,       -- verweist auf groups.group_id
  key        TEXT    NOT NULL,
  value      TEXT    NOT NULL,       -- JSON-serialized
  timestamp  TEXT    NOT NULL,       -- ISO-Timestamp
  FOREIGN KEY(group_id) REFERENCES groups(group_id)
);
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id TEXT PRIMARY KEY,
            data    TEXT NOT NULL          -- JSON-serialized dict
        )""")
        self.conn.commit()

    def store_fact(self,
                   group_id: str,
                   key: str,
                   value: Any):
        """
        Legt einen neuen Fact in der Gruppe an.
        """
        ts = datetime.now(timezone.utc).isoformat() + "Z"
        val_j = json.dumps(value, ensure_ascii=False)
        self.conn.execute("""
            INSERT INTO facts (group_id, key, value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (group_id, key, val_j, ts))
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

    def store_facts(self, chat_id, user_id, array):
        new_ids = {}
        for grp in array:
            raw_gid = grp["group_id"]
            grp_id = self.resolve_group_id(chat_id, user_id, raw_gid)  # löst "?abcd" → echte ID
            new_ids[grp_id] = raw_gid
            parent_id = grp.get("parent_id")
            if parent_id:
                parent_id = new_ids.get(parent_id, parent_id)
            sem_type = grp.get("type")
            if sem_type is None:
                # vorhandenen Typ aus DB holen
                cur = self.conn.cursor()
                cur.execute("SELECT type FROM groups WHERE group_id=?", (grp_id,))
                row = cur.fetchone()
                sem_type = row[0] if row else "unknown"

            # 1) Gruppe anlegen oder aktualisieren
            self.ensure_group(
                group_id=grp_id,
                chat_id=chat_id,
                user_id=user_id,
                sem_type=sem_type,
                parent_id=parent_id
            )

            # 2) Alle übrigen Keys als Fact speichern
            for key, val in grp.items():
                if key in ("group_id", "type", "parent_id"):
                    continue
                self.store_fact(
                    group_id=grp_id,
                    key=key,
                    value=val
                )

    def get_group(self, chat_id: str,
                  user_id: str,
                  group_id: str):
        c = self.conn.cursor()
        c.execute("""
                        SELECT group_id, type, parent_id
                        FROM groups
                        WHERE chat_id=? AND user_id=? AND group_id=?
                        ORDER BY created_at ASC
                    """, (chat_id, user_id, group_id))
        return c.fetchone()

    def get_groups(self, chat_id: str,
                   user_id: str):
        c = self.conn.cursor()
        c.execute("""
                        SELECT group_id, type, parent_id
                        FROM groups
                        WHERE chat_id=? AND user_id=?
                        ORDER BY created_at ASC
                    """, (chat_id, user_id))
        groups = {
            row["group_id"]: {
                "group_id": row["group_id"],
                "type": row["type"],
                "parent_id": row["parent_id"]
            }
            for row in c.fetchall()
        }
        return groups

    def ensure_group(self,
                     group_id: str,
                     chat_id: str,
                     user_id: str,
                     sem_type: str,
                     parent_id: str | None = None):
        """
        Legt eine Gruppe an, wenn sie nicht existiert.
        Falls parent_id übergeben wird, wird sie gespeichert.
        """
        ts = datetime.now(timezone.utc).isoformat() + "Z"
        self.conn.execute("""
            INSERT INTO groups (group_id, chat_id, user_id, type, parent_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
              type       = excluded.type,
              parent_id  = COALESCE(excluded.parent_id, groups.parent_id)
        """, (group_id, chat_id, user_id, sem_type, parent_id, ts))
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

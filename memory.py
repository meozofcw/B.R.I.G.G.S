import sqlite3
from datetime import datetime

DB = "data/memory.db"

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,
            content TEXT,
            created_at TEXT
        )""")

def add_memory(kind: str, content: str):
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO memories(kind, content, created_at) VALUES(?,?,?)",
            (kind, content, datetime.utcnow().isoformat())
        )

def get_memories(limit=20):
    with sqlite3.connect(DB) as con:
        rows = con.execute(
            "SELECT kind, content, created_at FROM memories ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows[::-1]

import sqlite3
from datetime import datetime

DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            category TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(filename, category, confidence):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (timestamp, filename, category, confidence) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), filename, category, confidence)
    )
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
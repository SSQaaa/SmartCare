import sqlite3
from pathlib import Path
from typing import Optional

SMART_CARE_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = SMART_CARE_ROOT / "data" / "object_memory.db"


def get_connection(db_path: Path = DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT NOT NULL UNIQUE,
                display_name TEXT,
                video_path TEXT,
                dataset_dir TEXT,
                data_yaml TEXT,
                weights_path TEXT,
                status TEXT DEFAULT 'created',
                is_active INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_object_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT NOT NULL,
                alias TEXT NOT NULL UNIQUE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_object(
    object_name: str,
    display_name: str,
    video_path: str = "",
    dataset_dir: str = "",
    data_yaml: str = "",
    weights_path: str = "",
    status: str = "created",
    db_path: Path = DB_PATH,
):
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO personal_objects (
                object_name, display_name, video_path, dataset_dir, data_yaml, weights_path, status, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(object_name) DO UPDATE SET
                display_name=excluded.display_name,
                video_path=excluded.video_path,
                dataset_dir=excluded.dataset_dir,
                data_yaml=excluded.data_yaml,
                weights_path=excluded.weights_path,
                status=excluded.status,
                updated_at=CURRENT_TIMESTAMP
            """,
            (object_name, display_name, video_path, dataset_dir, data_yaml, weights_path, status),
        )
        if display_name:
            conn.execute(
                """
                INSERT INTO personal_object_aliases (object_name, alias, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(alias) DO UPDATE SET
                    object_name=excluded.object_name,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (object_name, display_name),
            )
        conn.execute(
            """
            INSERT INTO personal_object_aliases (object_name, alias, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(alias) DO UPDATE SET
                object_name=excluded.object_name,
                updated_at=CURRENT_TIMESTAMP
            """,
            (object_name, object_name),
        )
        conn.commit()
    finally:
        conn.close()


def activate_object(object_name: str, db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE personal_objects SET is_active = 0")
        conn.execute(
            """
            UPDATE personal_objects
            SET is_active = 1, updated_at = CURRENT_TIMESTAMP
            WHERE object_name = ?
            """,
            (object_name,),
        )
        conn.commit()
    finally:
        conn.close()


def update_status(object_name: str, status: str, weights_path: str = "", db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        if weights_path:
            conn.execute(
                """
                UPDATE personal_objects
                SET status = ?, weights_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE object_name = ?
                """,
                (status, weights_path, object_name),
            )
        else:
            conn.execute(
                """
                UPDATE personal_objects
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE object_name = ?
                """,
                (status, object_name),
            )
        conn.commit()
    finally:
        conn.close()


def get_active_object(db_path: Path = DB_PATH) -> Optional[sqlite3.Row]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT *
            FROM personal_objects
            WHERE is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
        return row
    finally:
        conn.close()


def get_object(object_name: str, db_path: Path = DB_PATH) -> Optional[sqlite3.Row]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM personal_objects WHERE object_name = ?",
            (object_name,),
        ).fetchone()
        return row
    finally:
        conn.close()


def get_object_aliases(db_path: Path = DB_PATH) -> dict[str, str]:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT alias, object_name
            FROM personal_object_aliases
            ORDER BY updated_at DESC
            """
        ).fetchall()
        return {row["alias"]: row["object_name"] for row in rows}
    finally:
        conn.close()

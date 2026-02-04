import os
import sqlite3

# 스크립트와 같은 폴더에 DB 생성 (실행 경로와 무관하게 항상 동일한 파일 사용)
_this_dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(_this_dir, "key_expressions.db")


def init_db():
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS key_expressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grade TEXT NOT NULL,
                publisher TEXT NOT NULL,
                unit TEXT NOT NULL,
                unit_theme TEXT,
                key_expression TEXT NOT NULL,
                example_sentence TEXT
            )
        """
        )
        conn.commit()
        try:
            cursor.execute("ALTER TABLE key_expressions ADD COLUMN unit_theme TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass


def clear_key_expressions():
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.execute("DELETE FROM key_expressions")
        conn.commit()


def insert_key_expression(
    grade, publisher, unit, key_expression, example_sentence=None, unit_theme=None
):
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO key_expressions (grade, publisher, unit, unit_theme, key_expression, example_sentence)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (grade, publisher, unit, unit_theme, key_expression, example_sentence),
        )
        conn.commit()


def get_key_expressions(grade, publisher, unit):
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT key_expression, example_sentence FROM key_expressions
            WHERE grade = ? AND publisher = ? AND unit = ?
        """,
            (grade, publisher, unit),
        )
        rows = cursor.fetchall()
        return [{"key_expression": row[0], "example_sentence": row[1]} for row in rows]


def get_unit_theme(grade, publisher, unit):
    """해당 단원의 소재/기능(예: 교통 수단 말하기). 없으면 빈 문자열."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT unit_theme FROM key_expressions
            WHERE grade = ? AND publisher = ? AND unit = ?
            LIMIT 1
        """,
            (grade, publisher, unit),
        )
        row = cursor.fetchone()
        return (row[0] or "").strip() if row and row[0] else ""


if __name__ == "__main__":
    init_db()
    print("Database initialized and table created.")
    print("DB path:", DATABASE_FILE)

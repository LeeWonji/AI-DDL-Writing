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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS expected_error_scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grade TEXT NOT NULL,
                publisher TEXT NOT NULL,
                unit TEXT NOT NULL,
                error_pattern TEXT NOT NULL,
                question TEXT NOT NULL,
                example_sentences TEXT,
                priority INTEGER DEFAULT 100
            )
        """
        )
        conn.commit()


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


def get_grade_options():
    """학년 목록 반환."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT grade FROM key_expressions
            WHERE grade IS NOT NULL AND TRIM(grade) != ''
            ORDER BY grade
        """
        )
        return [row[0] for row in cursor.fetchall() if row and row[0]]


def get_publisher_options(grade):
    """선택한 학년의 출판사 목록 반환."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT publisher FROM key_expressions
            WHERE grade = ? AND publisher IS NOT NULL AND TRIM(publisher) != ''
            ORDER BY publisher
        """,
            (grade,),
        )
        return [row[0] for row in cursor.fetchall() if row and row[0]]


def get_unit_options(grade, publisher):
    """선택한 학년/출판사의 단원 목록 반환."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT unit FROM key_expressions
            WHERE grade = ? AND publisher = ? AND unit IS NOT NULL AND TRIM(unit) != ''
            ORDER BY unit
        """,
            (grade, publisher),
        )
        return [row[0] for row in cursor.fetchall() if row and row[0]]


def get_expected_error_scripts(grade, publisher, unit):
    """선택한 학년/출판사/단원의 예상 오류 스크립트 목록 반환."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, error_pattern, question, example_sentences, priority
            FROM expected_error_scripts
            WHERE grade = ? AND publisher = ? AND unit = ?
            ORDER BY priority ASC, id ASC
        """,
            (grade, publisher, unit),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "error_pattern": row[1],
                "question": row[2],
                "example_sentences": row[3],
                "priority": row[4],
            }
            for row in rows
        ]


if __name__ == "__main__":
    init_db()
    print("Database initialized and table created.")
    print("DB path:", DATABASE_FILE)

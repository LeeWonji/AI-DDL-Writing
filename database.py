import csv
import os
import sqlite3

# 스크립트와 같은 폴더에 DB 생성 (실행 경로와 무관하게 항상 동일한 파일 사용)
_this_dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(_this_dir, "key_expressions.db")

# 예상 오류 스크립트 CSV 캐시: 경로 -> (mtime, 파싱된 행 dict 목록)
_expected_script_csv_cache = {}


def _expected_error_scripts_csv_path(grade):
    """5학년/6학년 각각 전용 CSV 경로. 그 외 학년은 None."""
    g = (grade or "").strip()
    if g not in ("5학년", "6학년"):
        return None
    return os.path.join(_this_dir, "data", f"expected_error_scripts_{g}.csv")


def _load_all_rows_from_expected_script_csv(csv_path):
    """CSV 전체 행을 읽어 캐시한다. 파일 없음/오류 시 빈 리스트."""
    try:
        file_mtime = os.path.getmtime(csv_path)
    except OSError:
        return []
    cached = _expected_script_csv_cache.get(csv_path)
    if cached and cached[0] == file_mtime:
        return cached[1]
    required = ("grade", "publisher", "unit", "error_pattern", "question")
    rows_out = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or any(
                c not in reader.fieldnames for c in required
            ):
                _expected_script_csv_cache[csv_path] = (file_mtime, [])
                return []
            for row in reader:
                rows_out.append(dict(row))
    except OSError:
        _expected_script_csv_cache[csv_path] = (file_mtime, [])
        return []
    _expected_script_csv_cache[csv_path] = (file_mtime, rows_out)
    return rows_out


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
        # 예상 오류 스크립트는 CSV만 사용 — 예전 DB 테이블이 있으면 제거
        try:
            cursor.execute("DROP TABLE IF EXISTS expected_error_scripts")
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
    """선택 학년에 해당하는 CSV만 읽어, 출판사·단원에 맞는 예상 오류 스크립트 목록 반환.

    5학년 → data/expected_error_scripts_5학년.csv
    6학년 → data/expected_error_scripts_6학년.csv
    """
    g_sel = (grade or "").strip()
    p_sel = (publisher or "").strip()
    u_sel = (unit or "").strip()
    csv_path = _expected_error_scripts_csv_path(g_sel)
    if not csv_path or not os.path.isfile(csv_path):
        return []

    raw_rows = _load_all_rows_from_expected_script_csv(csv_path)
    matched = []
    for i, row in enumerate(raw_rows):
        row_grade = (row.get("grade") or "").strip()
        row_pub = (row.get("publisher") or "").strip()
        row_unit = (row.get("unit") or "").strip()
        if row_pub != p_sel or row_unit != u_sel:
            continue
        # 행에 grade가 적혀 있으면 선택 학년과 반드시 일치 (파일 오염 방지)
        if row_grade and row_grade != g_sel:
            continue
        error_pattern = (row.get("error_pattern") or "").strip()
        question = (row.get("question") or "").strip()
        if not error_pattern or not question:
            continue
        id_raw = (row.get("id") or "").strip()
        try:
            row_id = int(id_raw) if id_raw else i + 1
        except ValueError:
            row_id = i + 1
        matched.append(
            {
                "id": row_id,
                "error_pattern": error_pattern,
                # CSV에 컬럼이 없으면 row.get이 빈 값 → 기존과 동일하게 패턴만으로 매칭
                "error_description": (row.get("error_description") or "").strip(),
                "question": question,
                "example_sentences": (row.get("example_sentences") or "").strip(),
            }
        )
    matched.sort(key=lambda r: (r["id"], r["error_pattern"]))
    return matched


if __name__ == "__main__":
    init_db()
    print("Database initialized and table created.")
    print("DB path:", DATABASE_FILE)

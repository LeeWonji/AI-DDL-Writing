import csv
import os
import sqlite3

from database import DATABASE_FILE, init_db


def import_expected_scripts(csv_path):
    # 테이블이 없는 경우를 대비해 먼저 초기화
    init_db()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = [
                "grade",
                "publisher",
                "unit",
                "error_pattern",
                "question",
                "example_sentences",
                "priority",
            ]
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"필수 컬럼 누락: {col}")

            inserted_count = 0
            skipped_count = 0

            for row in reader:
                # 입력값 정리
                grade = (row.get("grade") or "").strip()
                publisher = (row.get("publisher") or "").strip()
                unit = (row.get("unit") or "").strip()
                error_pattern = (row.get("error_pattern") or "").strip()
                question = (row.get("question") or "").strip()
                example_sentences = (row.get("example_sentences") or "").strip()
                priority_raw = (row.get("priority") or "").strip()

                # 필수값이 비어 있으면 건너뜀
                if not (grade and publisher and unit and error_pattern and question):
                    skipped_count += 1
                    continue

                # 우선순위 파싱 실패 시 기본값 100 사용
                try:
                    priority = int(priority_raw) if priority_raw else 100
                except ValueError:
                    priority = 100

                # 같은 단원/패턴/질문이 이미 있으면 중복 삽입 방지
                cursor.execute(
                    """
                    SELECT id FROM expected_error_scripts
                    WHERE grade = ? AND publisher = ? AND unit = ?
                      AND error_pattern = ? AND question = ?
                    LIMIT 1
                """,
                    (grade, publisher, unit, error_pattern, question),
                )
                if cursor.fetchone():
                    skipped_count += 1
                    continue

                cursor.execute(
                    """
                    INSERT INTO expected_error_scripts
                    (grade, publisher, unit, error_pattern, question, example_sentences, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        grade,
                        publisher,
                        unit,
                        error_pattern,
                        question,
                        example_sentences,
                        priority,
                    ),
                )
                inserted_count += 1

        conn.commit()

    print(f"완료: inserted={inserted_count}, skipped={skipped_count}")
    print(f"DB 파일: {DATABASE_FILE}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(
        base_dir, "data", "expected_error_scripts_ybm_choi_6_1.csv"
    )
    import_expected_scripts(default_csv)

import json
import os
import re
import unicodedata

from flask import Flask, request, jsonify, render_template
import openai
import pandas as pd

from database import (
    init_db,
    clear_key_expressions,
    insert_key_expression,
    get_key_expressions,
    get_unit_theme,
)

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

CORPUS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def normalize_unit(unit):
    """단원을 'Unit N' 형식으로 통일. 숫자만('1') 또는 'Unit 1' 모두 허용."""
    if unit is None:
        return ""
    u = str(unit).strip()
    if not u:
        return ""
    if u.isdigit():
        return f"Unit {u}"
    m = re.match(r"^unit\s*(\d+)$", u, re.IGNORECASE)
    if m:
        return f"Unit {m.group(1)}"
    return u


def _infer_grade_from_filename(filename: str) -> str:
    """
    학년별 CSV 파일명에서 grade를 추론.
    기대 파일명: key_expressions_grade3.csv ... key_expressions_grade6.csv
    """
    m = re.search(r"key_expressions[_-]?grade([3-6])\.csv$", filename, re.IGNORECASE)
    return m.group(1) if m else ""


def load_corpus_from_data():
    """data 폴더 CSV를 DB에 로드. unit_theme(단원 소재/기능) 포함."""
    if not os.path.isdir(CORPUS_DATA_DIR):
        print(f"[코퍼스] data 폴더 없음: {CORPUS_DATA_DIR}")
        return
    clear_key_expressions()
    loaded = 0
    for filename in os.listdir(CORPUS_DATA_DIR):
        if not filename.lower().endswith(".csv"):
            continue
        grade_from_file = _infer_grade_from_filename(filename)
        if not grade_from_file:
            # 학년별 CSV만 로드 (기존 통합 CSV 등은 무시)
            continue
        path = os.path.join(CORPUS_DATA_DIR, filename)
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="cp949")
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
            for _, row in df.iterrows():
                grade = grade_from_file
                publisher = row.get("publisher")
                unit = row.get("unit")
                key_expression = row.get("key_expression")
                if pd.isna(publisher) or pd.isna(unit) or pd.isna(key_expression):
                    continue
                grade = str(grade).strip()
                publisher = str(publisher).strip()
                unit = normalize_unit(unit)
                key_expression = str(key_expression).strip()
                if not grade or not publisher or not unit or not key_expression:
                    continue
                ex_val = row.get("example_sentence")
                example_sentence = (
                    str(ex_val).strip()
                    if pd.notna(ex_val) and str(ex_val).strip()
                    else None
                )
                unit_theme = (
                    row.get("unit_theme")
                    if "unit_theme" in row and pd.notna(row.get("unit_theme"))
                    else None
                )
                insert_key_expression(
                    grade,
                    publisher,
                    unit,
                    key_expression,
                    example_sentence,
                    unit_theme=unit_theme,
                )
                loaded += 1
        except Exception as e:
            print(f"[코퍼스 로드] {filename} 처리 중 오류: {e}")
    if loaded > 0:
        print(f"[코퍼스] 핵심 표현 {loaded}건 로드됨 (예문 포함)")
    else:
        print(f"[코퍼스] 로드된 행 없음. CSV 경로: {CORPUS_DATA_DIR}")
        print(
            "[코퍼스] key_expressions.db 삭제 후 서버 재시작하면 CSV가 다시 로드됩니다."
        )


init_db()
load_corpus_from_data()


def get_key_expressions_from_csv(grade, publisher, unit):
    """DB에 없을 때 CSV에서 해당 단원 핵심 표현만 읽어서 반환."""
    grade = str(grade).strip() if grade else ""
    publisher = str(publisher).strip() if publisher else ""
    unit = normalize_unit(unit)
    if not grade or not publisher or not unit:
        return []
    out = []
    filename = f"key_expressions_grade{grade}.csv"
    path = os.path.join(CORPUS_DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        for _, row in df.iterrows():
            p = row.get("publisher")
            u = row.get("unit")
            if pd.isna(p) or pd.isna(u):
                continue
            if str(p).strip() != publisher or normalize_unit(u) != unit:
                continue
            ke = row.get("key_expression")
            if pd.isna(ke) or not str(ke).strip():
                continue
            ex = row.get("example_sentence")
            out.append(
                {
                    "key_expression": str(ke).strip(),
                    "example_sentence": (
                        str(ex).strip() if pd.notna(ex) and str(ex).strip() else None
                    ),
                }
            )
    except Exception as e:
        print(f"[CSV fallback] {filename}: {e}")
    return out


def get_unit_theme_from_csv(grade, publisher, unit):
    """DB에 없을 때 CSV에서 해당 단원 unit_theme 반환."""
    grade = str(grade).strip() if grade else ""
    publisher = str(publisher).strip() if publisher else ""
    unit = normalize_unit(unit)
    if not grade or not publisher or not unit:
        return ""
    filename = f"key_expressions_grade{grade}.csv"
    path = os.path.join(CORPUS_DATA_DIR, filename)
    if not os.path.exists(path):
        return ""
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        for _, row in df.iterrows():
            p, u = row.get("publisher"), row.get("unit")
            if pd.isna(p) or pd.isna(u):
                continue
            if str(p).strip() != publisher or normalize_unit(u) != unit:
                continue
            th = row.get("unit_theme") if "unit_theme" in row else None
            if pd.notna(th) and str(th).strip():
                return str(th).strip()
            return ""
    except Exception as e:
        print(f"[CSV fallback] {filename}: {e}")
    return ""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload_key_expressions", methods=["POST"])
def upload_key_expressions():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                grade = row["grade"]
                publisher = row["publisher"]
                unit = normalize_unit(row["unit"])
                key_expression = row["key_expression"]
                example_sentence = (
                    row["example_sentence"]
                    if pd.notna(row.get("example_sentence"))
                    else None
                )
                unit_theme = (
                    row["unit_theme"]
                    if "unit_theme" in row and pd.notna(row.get("unit_theme"))
                    else None
                )
                insert_key_expression(
                    grade,
                    publisher,
                    unit,
                    key_expression,
                    example_sentence,
                    unit_theme=unit_theme,
                )
            return jsonify({"message": "Key expressions uploaded successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type, please upload a CSV"}), 400


def _parse_json_from_response(content):
    content = content.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        content = m.group(1).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    bracket = re.search(r"\[[\s\S]*\]", content)
    if bracket:
        try:
            return json.loads(bracket.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _contains_hangul(s):
    """문자열에 한글이 포함되어 있는지 여부."""
    if not s:
        return False
    return bool(re.search(r"[\uAC00-\uD7A3]", s))


def api_call_translate_korean_to_english(items):
    """한국어 error_span 목록을 영어로만 번역. items = [(error_id, error_span, sentence 또는 ''), ...] -> {error_id: english}"""
    if not items:
        return {}
    lines = []
    for t in items:
        eid, span = t[0], t[1]
        sentence = t[2] if len(t) > 2 else ""
        if sentence:
            lines.append(f"- id {eid}: '{span}' (문장: {sentence})")
        else:
            lines.append(f"- id {eid}: '{span}'")
    lines_block = "\n".join(lines)
    prompt = f"""아래 각 항목의 한국어를 **해당 문장 맥락에 맞는 영어 단어 또는 짧은 구 하나**로만 번역하세요. 예: "가"가 문장 끝에 있으면 "go", "안녕"이면 "Hi"/"Hello", "전주북초등학교에"면 "at Jeonju Buk Elementary School" 등. 응답에는 한글이 포함되면 안 됩니다.

{lines_block}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요.
[{{"error_id": 0, "correction": "Hi"}}, {{"error_id": 1, "correction": "go"}}, ...]"""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You translate Korean words/phrases to equivalent English only. Output must be a JSON array with error_id and correction. Every correction must be in English only, no Korean characters.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if not parsed or not isinstance(parsed, list):
            return {}
        return {
            item["error_id"]: (item.get("correction") or "").strip()
            for item in parsed
            if isinstance(item, dict) and "error_id" in item
        }
    except Exception as e:
        print(f"[API translate Korean] 실패: {e}")
        return {}


def api_call_1_extract_errors(writing):
    """ESL 전문가 역할 한 줄 지시로 오류 추출. 문법 규칙을 프롬프트에 나열하지 않음. 한국어도 번역 대상으로 포함."""
    prompt = f"""학생 작문:
\"\"\"
{writing}
\"\"\"

위 글에서 발견한 모든 오류를 아래 JSON 배열 형식으로만 출력하세요. 오류가 없으면 [] 출력.
- 문법·철자·굴절·전치사 등 모든 영어 오류를 포함하세요.
- **한국어가 한 글자라도 섞여 있으면 반드시** 그 부분을 빠짐없이 오류로 넣으세요. 인사말(안녕, 안녕하세요 등), 이름·단어(이원지야, 전주북초등학교에 등), 조사·어미(가, 에, 를 등) 모두 각각 별도 오류. description은 "한국어를 영어로 번역". 예: "안녕, I'm 이원지야. ... everyday 가" → error_span "안녕", "이원지야", "전주북초등학교에", "가" 각각 별도 항목.
- **error_span은 위 학생 작문 원문에 나온 문자열을 그대로 복사**하세요. 공백·문자 하나라도 다르면 안 됩니다. 실제로 잘못된 부분만 최소 단위로 나누되, 반드시 원문과 동일한 문자열을 사용하세요. (예: "too meet" → error_span "too"만. "이원지야"가 있으면 error_span은 "이원지야" 그대로.)
[{{"id": 0, "sentence": "오류가 포함된 문장", "error_span": "원문과 동일한 문자열", "description": "오류 유형"}}, ...]"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ESL/EFL error detector. List every grammatical, spelling, and usage error. If the writing contains ANY Korean (including greetings like 안녕, particles/endings like 가/에/를, names, place names), list EVERY such Korean piece as a separate error with description '한국어를 영어로 번역'. Do not skip any. Use for error_span the EXACT substring from the student writing. Output only a valid JSON array.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content
        parsed = _parse_json_from_response(raw)
        if parsed is None or not isinstance(parsed, list):
            return []
        return [
            e for e in parsed if isinstance(e, dict) and "id" in e and "sentence" in e
        ]
    except Exception as e:
        print(f"[API1] 오류 추출 실패: {e}")
        return []


def api_call_2_classify_errors_by_theme(errors, unit_theme, key_expressions_list):
    """단원 소재/기능(unit_theme)에 관련된 오류인지 판단. 오류 유형 나열 없이 소재·기능만으로 분류."""
    if not unit_theme and not key_expressions_list:
        return [
            {
                "error_id": e.get("id", i),
                "is_key_expression_error": False,
                "matched_key_expression": None,
            }
            for i, e in enumerate(errors)
        ]
    key_exprs_str = (
        ", ".join(repr(ke["key_expression"]) for ke in key_expressions_list)
        if key_expressions_list
        else "(없음)"
    )
    errors_json = json.dumps(
        [
            {
                "id": e.get("id", i),
                "sentence": e.get("sentence", ""),
                "error_span": e.get("error_span", ""),
            }
            for i, e in enumerate(errors)
        ],
        ensure_ascii=False,
    )
    prompt = f"""이 단원의 소재/기능: "{unit_theme or '해당 없음'}"
이 단원의 핵심 표현 목록: {key_exprs_str}

아래 오류 목록의 각 항목을 판단할 때, **오직** 다음 조건을 만족할 때만 is_key_expression_error: true 로 답하세요.
- 조건: error_span이 **핵심 표현 목록에 있는 표현을 잘못 쓴 것**일 때만 true
  예: 핵심 표현이 "by bus"인데 error_span이 "on bus", "in bus", "by a bus" → true
  예: 핵심 표현이 "want to"인데 error_span이 "want be", "want" (to 누락) → true
- 그 외는 **모두** false:
  - 시제 오류(과거/현재), 주어-동사 일치(has/have, like/likes), 철자(ofen→often), 전치사 오류 중 핵심 표현이 아닌 것 → false
  - 문장 안에 핵심 표현이 나와도, **해당 오류가 그 핵심 표현을 잘못 쓴 게 아니면** false
  - 애매하면 반드시 false

오류 목록 (JSON):
{errors_json}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요.
[{{"error_id": 0, "is_key_expression_error": true, "matched_key_expression": "by bus"}}, {{"error_id": 1, "is_key_expression_error": false, "matched_key_expression": null}}, ...]
- 오직 핵심 표현을 잘못 쓴 경우만 true, 나머지는 전부 false."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Classify errors strictly: set is_key_expression_error true ONLY when the error_span is a misuse of one of the listed key expressions (e.g. wrong preposition or missing part of that expression). All other errors (tense, subject-verb agreement, spelling, etc.) must be false. When in doubt, use false. Output only valid JSON array.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content
        parsed = _parse_json_from_response(raw)
        if parsed is None or not isinstance(parsed, list):
            return [
                {
                    "error_id": e.get("id", i),
                    "is_key_expression_error": False,
                    "matched_key_expression": None,
                }
                for i, e in enumerate(errors)
            ]
        return parsed
    except Exception as e:
        print(f"[API2] 분류 실패: {e}")
        return [
            {
                "error_id": e.get("id", i),
                "is_key_expression_error": False,
                "matched_key_expression": None,
            }
            for i, e in enumerate(errors)
        ]


def build_kwic_from_db(classifications, key_expressions_list):
    if not key_expressions_list:
        return "단원의 핵심 표현이 없습니다."
    # 예문: 행마다 1개 표시. 한 셀에 여러 예문은 | 또는 줄바꿈으로 구분 가능
    lines = []
    for ke in key_expressions_list:
        expr_raw = (ke.get("key_expression") or "").strip()
        ex = ke.get("example_sentence")
        if not ex or not str(ex).strip():
            continue
        # 한 셀에 여러 예문: "문장1|문장2" 또는 "문장1\n문장2" 형태 지원
        raw = str(ex).strip().replace("\r\n", "\n")
        sentences = []
        for part in raw.split("|"):
            for line in part.split("\n"):
                s = line.strip()
                if s:
                    sentences.append(s)
        if not sentences:
            sentences = [str(ex).strip()]
        parts = [p.strip() for p in expr_raw.replace("，", ",").split(",") if p.strip()]
        for sent in sentences:
            marked = sent
            for part in parts:
                if part in sent:
                    marked = sent.replace(part, f"**{part}**")
                    break
            if marked == sent and parts:
                marked = sent.replace(parts[0], f"**{parts[0]}**")
            lines.append(f"- {marked}")
    if not lines:
        return "이 단원에 등록된 핵심 표현 예문이 없습니다."
    return "\n".join(lines)


# 핵심 표현의 흔한 잘못된 형태 → 정답 표현 (후처리용)
KEY_EXPR_WRONG_FORMS = {
    "by bus": ["by a bus", "on bus", "in bus", "by the bus"],
    "by subway": ["by a subway", "on subway", "in subway"],
    "by taxi": ["by a taxi", "on taxi", "in taxi"],
    "on foot": ["by foot", "on the foot"],
    "want to": ["want be"],  # want to be에서 to 누락
}


def _strip_prompt_echo_from_corrected(text):
    """수정문에 섞여 나온 프롬프트/지시문 잔여를 제거한다."""
    if not text or not text.strip():
        return text
    # "The student wrote:" 같은 문구와 그 뒤 따옴표 블록 제거
    text = re.sub(
        r"\s*The student wrote\s*:?\s*\"\"\"?\s*.*$",
        "",
        text,
        flags=re.I | re.DOTALL,
    )
    text = re.sub(
        r"\s*Student (?:composition|writing)\s*:?\s*\"\"\"?\s*.*$",
        "",
        text,
        flags=re.I | re.DOTALL,
    )
    # 한 줄씩 보며 지시문 같은 줄 제거 (영문 메타 문구)
    skip_patterns = [
        r"^\s*Please output\s+.*$",
        r"^\s*Do not simply repeat\s+.*$",
        r"^\s*Output only\s+.*$",
        r"^\s*Student composition\s*:?\s*$",
        r"^\s*\(?Student composition\s*:?\s*\)?\s*$",
        r"^\s*수정된 글만\s+.*$",
        r"^\s*다음에\s+본문만\s+.*$",
    ]
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue
        if any(re.search(p, stripped, re.I) for p in skip_patterns):
            continue
        out.append(line)
    text = "\n".join(out).strip()
    return text


def _ensure_key_expressions_marked(text, mark_parts):
    """수정문에서 단원 핵심 표현이 **만 되어 있거나 잘못된 형태면 <mark>로 감싼다."""
    if not mark_parts:
        return text
    # 1) 잘못된 형태(want be, by a bus 등) → <mark>정답</mark>으로 먼저 치환
    for phrase in sorted(mark_parts, key=len, reverse=True):
        for wrong in KEY_EXPR_WRONG_FORMS.get(phrase, []):
            text = text.replace(f"**{wrong}**", f"<mark>{phrase}</mark>")
            text = text.replace(wrong, f"<mark>{phrase}</mark>")
    # 2) 공백 있는 구절이 "want **to**"처럼 끊겨 나온 경우 → <mark>want to</mark>
    for phrase in sorted(mark_parts, key=len, reverse=True):
        if " " not in phrase:
            continue
        words = phrase.split()
        for i in range(len(words)):
            variant = " ".join(words[:i] + [f"**{words[i]}**"] + words[i + 1 :])
            text = text.replace(variant, f"<mark>{phrase}</mark>")
    # 3) **정답** → <mark>정답</mark>, 그냥 정답 → <mark>정답</mark>
    parts = sorted(mark_parts, key=len, reverse=True)
    placeholders = {}
    for phrase in parts:
        text = text.replace(f"**{phrase}**", f"<mark>{phrase}</mark>")
    for phrase in parts:
        ph = f"\u200b__M{len(placeholders)}__\u200b"
        placeholders[ph] = f"<mark>{phrase}</mark>"
        text = text.replace(placeholders[ph], ph)
        text = text.replace(phrase, f"<mark>{phrase}</mark>")
        text = text.replace(ph, placeholders[ph])
    return text


def api_call_3_correct_non_key_expression_errors(
    writing, errors, classifications, key_expressions_list
):
    key_expr_error_ids = {
        c["error_id"] for c in classifications if c.get("is_key_expression_error")
    }
    # 핵심 표현 오류: (잘못된 부분 → 반드시 쓸 정답 표현)
    key_expr_corrections = []
    for c in classifications:
        if not c.get("is_key_expression_error"):
            continue
        eid = c.get("error_id")
        right = (c.get("matched_key_expression") or "").strip()
        if not right:
            continue
        err = next((e for i, e in enumerate(errors) if e.get("id", i) == eid), None)
        if not err:
            continue
        wrong = (err.get("error_span") or "").strip()
        if wrong and (wrong, right) not in key_expr_corrections:
            key_expr_corrections.append((wrong, right))
    key_expr_instruction = ""
    if key_expr_corrections:
        key_expr_instruction = (
            "다음은 단원 핵심 표현 오류입니다. 반드시 아래처럼 고치고, 고친 표현(정답)을 <mark>...</mark>로 감싸세요.\n"
            + "\n".join(
                f"  '{w}' → '{r}' (고친 뒤 <mark>{r}</mark>)"
                for w, r in key_expr_corrections
            )
            + "\n\n"
        )
    phrases_to_mark = [
        e.get("error_span", "").strip()
        for i, e in enumerate(errors)
        if e.get("id", i) in key_expr_error_ids and e.get("error_span")
    ]
    phrases_to_mark = [p for p in phrases_to_mark if p]
    # 단원 핵심 표현(정답형)도 수정문에 나오면 <mark>로 표시
    unit_key_phrases = []
    for ke in key_expressions_list or []:
        expr = (ke.get("key_expression") or "").strip()
        if not expr:
            continue
        for part in expr.replace("，", ",").split(","):
            part = part.strip()
            if part and part not in unit_key_phrases:
                unit_key_phrases.append(part)
    mark_parts = list(
        unit_key_phrases
    )  # 정답형만 마킹 목록에 (잘못된 건 위에서 고치라고 했음)
    for p in phrases_to_mark:
        if p not in mark_parts:
            mark_parts.append(p)
    mark_instruction = (
        f'수정된 글에서 다음 구절이 나오면 <mark>...</mark>로 감싸세요: {", ".join(repr(p) for p in mark_parts)}\n\n'
        if mark_parts
        else ""
    )

    non_key_errors = [
        e for i, e in enumerate(errors) if e.get("id", i) not in key_expr_error_ids
    ]
    if not non_key_errors:
        prompt = f"""다음 학생 작문에서 {key_expr_instruction}{mark_instruction or '잘못된 부분만'} <mark>...</mark>로 감싸 주세요. 한국어가 섞여 있으면 해당 부분을 영어로 바꾸어 수정문은 전부 영어로만 출력하세요.

학생 작문:
\"\"\"
{writing}
\"\"\"

수정된 글만 출력하세요. [수정된 글] 다음에 본문만 넣어 주세요."""
    else:
        non_key_json = json.dumps(
            [
                {
                    "sentence": e.get("sentence", ""),
                    "error_span": e.get("error_span", ""),
                    "description": e.get("description", ""),
                }
                for e in non_key_errors
            ],
            ensure_ascii=False,
        )
        rule0 = (
            f"0) 단원 핵심 표현 오류: 아래와 같이 반드시 정답 표현으로 고치고, 고친 표현을 <mark>...</mark>로 감싸세요.\n{key_expr_instruction}"
            if key_expr_instruction
            else ""
        )
        prompt = f"""다음 규칙을 지키세요.
{rule0}1) 아래 "마킹할 구절"에 있는 표현이 수정된 글에 나오면 반드시 <mark>...</mark>로 감싸세요.
2) "수정할 오류 목록"에 있는 오류를 고친 전체 문단을 출력하세요. **볼드는 반드시 단어(또는 구) 단위로만** 적용하세요. 문장 전체가 아니라 실제로 바꾼 단어/구만 **이렇게** 감싸세요. 예: She **likes** to **play** soccer.
3) 작문에 한국어가 섞여 있으면, 그 부분을 자연스러운 영어로 바꾸어 수정문은 **전부 영어**로만 출력하세요. 한국어를 영어로 바꾼 부분은 **볼드**로 표시하세요.
{mark_instruction}
[수정된 글] 다음에 수정된 전체 글만 출력하세요. 원문을 그대로 출력하지 마세요.

학생 작문:
\"\"\"
{writing}
\"\"\"

수정할 오류 목록 (아래 항목들을 반드시 반영한 수정문을 출력):
{non_key_json}"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Output ONLY [수정된 글] followed by the corrected paragraph—nothing else. No instructions, no 'Student composition', no 'Please output'. Just the header and the paragraph. If the writing contains Korean, translate to English and bold the translated part. Bold only the specific words you changed. Example: [수정된 글]\\nShe **likes** to **play** baseball.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        idx = raw.find("[수정된 글]")
        if idx != -1:
            raw = raw[idx + len("[수정된 글]") :].strip()
        # 프롬프트·지시문이 붙어 나온 경우 제거
        raw = _strip_prompt_echo_from_corrected(raw)
        # "학생 작문:" 이전까지만 사용 (수정문 손실 방지)
        if "\n학생 작문:" in raw or "\n\n학생 작문:" in raw:
            raw = re.split(r"\n\n?학생 작문\s*:", raw, maxsplit=1)[0].strip()
        if "Student composition" in raw:
            raw = re.split(
                r"\n?\s*Student composition\s*:?\s*", raw, maxsplit=1, flags=re.I
            )[-1].strip()
        raw = re.sub(r'^\s*"""\s*', "", raw, count=1)
        raw = re.sub(r'\s*"""\s*$', "", raw, count=1)
        raw = raw.strip()
        if not raw or len(raw) < 2:
            return writing
        # 핵심 표현이 노란색(<mark>)으로 확실히 표시되도록 후처리 (모델이 **만 쓴 경우 대비)
        raw = _ensure_key_expressions_marked(raw, mark_parts)
        return raw
    except Exception as e:
        print(f"[API3] 수정 실패: {e}")
        return writing


def api_call_get_corrections(non_key_errors):
    """일반 오류(비핵심표현)에 대해 오류별 정답 단어/구만 JSON으로 반환."""
    if not non_key_errors:
        return {}
    errors_json = json.dumps(
        [
            {
                "error_id": e.get("id", i),
                "sentence": e.get("sentence", ""),
                "error_span": e.get("error_span", ""),
            }
            for i, e in enumerate(non_key_errors)
        ],
        ensure_ascii=False,
    )
    prompt = f"""아래 오류 목록에서 각 error_span을 문맥에 맞게 고친 **한 단어 또는 짧은 구**만 제시하세요.
- 영어 오류: 예) "ofen"→"often", "wakes"→"wake", "in bus"→"by bus".
- **한국어(error_span에 한글이 있는 경우)**: 반드시 해당 한국어를 **영어 단어/구 하나**로만 번역하여 correction에 넣으세요. correction에는 한글이 있으면 안 됩니다. 예) "내"→"my", "최애"→"favorite", "영화"→"movie".
오류 목록 (JSON):
{errors_json}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요. 각 error_id에 대해 correction 하나씩. 한국어 오류의 correction은 반드시 영어만.
[{{"error_id": 0, "correction": "often"}}, {{"error_id": 1, "correction": "my"}}, ...]"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Output only a valid JSON array of {error_id, correction}. One correction per error. If error_span contains Korean characters, correction MUST be the English equivalent only—no Korean allowed in correction.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if not parsed or not isinstance(parsed, list):
            return {}
        return {
            item["error_id"]: (item.get("correction") or "").strip()
            for item in parsed
            if isinstance(item, dict) and "error_id" in item
        }
    except Exception as e:
        print(f"[API corrections] 실패: {e}")
        return {}


@app.route("/feedback", methods=["POST"])
def get_feedback():
    data = request.get_json()
    student_writing = (data.get("writing") or "").strip()
    if isinstance(student_writing, str) and student_writing:
        student_writing = unicodedata.normalize("NFC", student_writing)
    grade = data.get("grade")
    publisher = data.get("publisher")
    unit = data.get("unit")

    if not student_writing or not grade or not publisher or not unit:
        return jsonify({"error": "Missing required fields"}), 400

    unit = normalize_unit(unit)
    key_expressions_list = get_key_expressions(grade, publisher, unit)
    if not key_expressions_list:
        key_expressions_list = get_key_expressions_from_csv(grade, publisher, unit)
    unit_theme = get_unit_theme(grade, publisher, unit)
    if not unit_theme and key_expressions_list:
        unit_theme = get_unit_theme_from_csv(grade, publisher, unit)

    errors = api_call_1_extract_errors(student_writing)
    if not errors:
        kwic_examples = build_kwic_from_db([], key_expressions_list)
        return (
            jsonify(
                {
                    "feedback": {
                        "original_text": student_writing,
                        "error_list": [],
                        "kwic_examples": kwic_examples,
                    }
                }
            ),
            200,
        )

    classifications = api_call_2_classify_errors_by_theme(
        errors, unit_theme, key_expressions_list
    )
    kwic_examples = build_kwic_from_db(classifications, key_expressions_list)

    key_expr_error_ids = {
        c["error_id"] for c in classifications if c.get("is_key_expression_error")
    }
    classification_by_id = {c["error_id"]: c for c in classifications}
    non_key_errors = [
        e for i, e in enumerate(errors) if e.get("id", i) not in key_expr_error_ids
    ]
    corrections_map = api_call_get_corrections(non_key_errors)

    # 한글 오류는 전용 번역 API로만 보정 사용(일반 보정 API가 잘못 매핑되는 경우 방지)
    korean_retry = []
    for e in non_key_errors:
        eid = e.get("id")
        span = (e.get("error_span") or "").strip()
        desc = (e.get("description") or "").strip()
        is_korean_error = _contains_hangul(span) or "한국어" in desc or "번역" in desc
        if not is_korean_error:
            continue
        sentence = (e.get("sentence") or "").strip()
        korean_retry.append((eid, span, sentence))
    if korean_retry:
        translated = api_call_translate_korean_to_english(korean_retry)
        for eid, eng in translated.items():
            if eng and not _contains_hangul(eng):
                corrections_map[eid] = eng

    # 원문 + 오류별 correction, is_key_expression 포함
    error_list = []
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        c = classification_by_id.get(eid, {})
        is_key = bool(c.get("is_key_expression_error"))
        if is_key:
            correction = (c.get("matched_key_expression") or "").strip()
        else:
            try:
                correction = (
                    corrections_map.get(eid)
                    or (corrections_map.get(int(eid)) if isinstance(eid, str) else None)
                    or ""
                )
            except (ValueError, TypeError):
                correction = corrections_map.get(eid) or ""
        error_list.append(
            {
                "id": eid,
                "sentence": e.get("sentence", ""),
                "error_span": (e.get("error_span") or "").strip(),
                "description": e.get("description", ""),
                "correction": correction,
                "is_key_expression": is_key,
            }
        )

    return (
        jsonify(
            {
                "feedback": {
                    "original_text": student_writing,
                    "error_list": error_list,
                    "kwic_examples": kwic_examples,
                }
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)

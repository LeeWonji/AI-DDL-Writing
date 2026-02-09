import json
import os
import re
import time
import unicodedata

from flask import Flask, request, jsonify, render_template
import openai
import pandas as pd

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


# API 재시도 (일시적 오류 완화)
def _openai_chat_with_retry(
    model, messages, max_tokens=1500, temperature=0.0, max_retries=3
):
    last_err = None
    for attempt in range(max_retries):
        try:
            return openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            last_err = e
            print(f"[OpenAI] 시도 {attempt + 1}/{max_retries} 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
    raise last_err


CORPUS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOUVAIN_TAG_EXAMPLES_PATH = os.path.join(CORPUS_DATA_DIR, "louvain_tag_examples.csv")
_louvain_cache = None


def _load_louvain_error_tags():
    """louvain_tag_examples.csv 단일 파일 로드 (code, label_en, desc_en, question, example_sentence)."""
    global _louvain_cache
    if _louvain_cache is not None:
        return _louvain_cache
    if not os.path.exists(LOUVAIN_TAG_EXAMPLES_PATH):
        _louvain_cache = []
        return []
    try:
        try:
            df = pd.read_csv(LOUVAIN_TAG_EXAMPLES_PATH, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(LOUVAIN_TAG_EXAMPLES_PATH, encoding="cp949")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        rows = []
        for _, r in df.iterrows():
            code = r.get("code")
            if pd.isna(code) or not str(code).strip():
                continue
            code = str(code).strip()
            label_en = (
                (r.get("label_en") or "").strip() if pd.notna(r.get("label_en")) else ""
            )
            desc_en = (
                (r.get("desc_en") or "").strip() if pd.notna(r.get("desc_en")) else ""
            )
            question = (
                (r.get("question") or "").strip() if pd.notna(r.get("question")) else ""
            )
            ex = (r.get("Example sentences") or r.get("example_sentence") or "").strip()
            if pd.notna(ex) and ex:
                ex = ex.replace("|", "\n")
            rows.append(
                {
                    "code": code,
                    "label_en": label_en,
                    "desc_en": desc_en,
                    "question": question,
                    "example_sentence": ex,
                }
            )
        _louvain_cache = rows
        return rows
    except Exception as e:
        print(f"[louvain_tag_examples] 로드 실패: {e}")
        _louvain_cache = []
        return []


def _get_louvain_row_by_code(code):
    """louvain_tag_examples에서 code로 행 하나 반환."""
    for row in _load_louvain_error_tags():
        if row.get("code") == code:
            return row
    return None


def _match_error_to_csv(description, error_span=None, correction=None):
    """description 키워드로 매칭. AI 분류 실패 시 폴백용."""
    if not description or not isinstance(description, str):
        return None
    desc_lower = description.strip().lower()
    best_row = None
    best_count = 0
    for row in _load_louvain_error_tags():
        kw = row.get("label_en", "") + " " + row.get("desc_en", "")
        words = re.findall(r"[a-z0-9]{2,}", kw.lower())
        count = sum(
            1 for w in words if w in desc_lower and w not in ("error", "use", "other")
        )
        if count > best_count:
            best_count = count
            best_row = row
    return best_row


def _clean_example_sentence(text):
    """예문 텍스트 정리."""
    if not text or not isinstance(text, str):
        return text
    return text.strip()


@app.route("/")
def home():
    return render_template("index(2).html")


def _parse_json_from_response(content):
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    # 코드 블록 제거
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        content = m.group(1).strip()
    # 1) 전체 파싱 시도
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # 2) 첫 번째 '[' 와 마지막 ']' 사이 배열만 추출 (앞뒤 설명문 제거)
    start = content.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "[":
                depth += 1
            elif content[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break
    # 3) 정규식으로 배열 부분만
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
        response = _openai_chat_with_retry(
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
        response = _openai_chat_with_retry(
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
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if parsed is None or not isinstance(parsed, list):
            print(f"[API1] 파싱 실패. raw 앞 300자: {raw[:300] if raw else 'None'}")
            return []
        out = []
        for e in parsed:
            if not isinstance(e, dict) or "sentence" not in e:
                continue
            # id: 숫자 또는 문자열 허용
            eid = e.get("id", len(out))
            span = (e.get("error_span") or e.get("errorSpan") or "").strip()
            if not span:
                continue
            out.append(
                {
                    "id": eid,
                    "sentence": e.get("sentence", ""),
                    "error_span": span,
                    "description": (e.get("description") or "").strip(),
                }
            )
        return out
    except Exception as e:
        print(f"[API1] 오류 추출 실패: {e}")
        raise


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
        response = _openai_chat_with_retry(
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
        raise


def _build_louvain_code_list():
    """AI 오류 유형 분류용: code와 짧은 설명 문자열 (한 줄에 하나씩)."""
    lines = []
    for row in _load_louvain_error_tags():
        code = row.get("code")
        if not code:
            continue
        label = (row.get("label_en") or "").strip()
        desc = (row.get("desc_en") or "").strip()
        lines.append(f"{code}: {label}. {desc}" if desc else f"{code}: {label}")
    return "\n".join(lines)


def api_call_classify_errors_to_louvain_batch(
    errors, corrections_map, classification_by_id
):
    """오류별 (sentence, error_span, correction, description)을 주고, 각각에 맞는 louvain 코드 하나씩 반환. {error_id: code}."""
    if not errors:
        return {}
    code_list = _build_louvain_code_list()
    payload = []
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        c = classification_by_id.get(eid, {})
        if c.get("is_key_expression_error"):
            correction = (c.get("matched_key_expression") or "").strip()
        else:
            correction = (
                corrections_map.get(eid)
                or corrections_map.get(int(eid))
                or corrections_map.get(str(eid))
                or ""
            )
        payload.append(
            {
                "error_id": eid,
                "sentence": (e.get("sentence") or "").strip(),
                "error_span": (e.get("error_span") or "").strip(),
                "correction": (correction or "").strip(),
                "description": (e.get("description") or "").strip(),
            }
        )
    errors_json = json.dumps(payload, ensure_ascii=False)
    prompt = f"""Below are errors from an ESL student's writing. Each has sentence, error_span (the wrong part), correction (the fixed form), and description.
For each error, choose the ONE most appropriate error type code from the list below. Consider what kind of mistake it is (e.g. tense, subject-verb agreement, spelling, word order, verb complementation).
Return a JSON array with one object per error: {{"error_id": <id>, "code": "<CODE>"}}. Use only codes from the list.

Error type list (code + short description):
{code_list}

Errors (JSON):
{errors_json}

Response (JSON array only):
[{{"error_id": 0, "code": "GVT"}}, {{"error_id": 1, "code": "FS"}}, ...]"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ESL error classifier. Output only a valid JSON array of {error_id, code}. Each code must be exactly one of the codes from the list (e.g. GVT, FS, GVN, WO, XVCO).",
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
        out = {}
        valid_codes = {row.get("code") for row in _load_louvain_error_tags()}
        for item in parsed:
            if (
                not isinstance(item, dict)
                or "error_id" not in item
                or "code" not in item
            ):
                continue
            eid = item["error_id"]
            code = (item.get("code") or "").strip().upper()
            if code in valid_codes:
                out[eid] = code
                out[int(eid)] = code
                out[str(eid)] = code
        return out
    except Exception as e:
        print(f"[API classify to louvain] 실패: {e}")
        raise


@app.route("/feedback", methods=["POST"])
def get_feedback():
    data = request.get_json()
    student_writing = (data.get("writing") or "").strip()
    if isinstance(student_writing, str) and student_writing:
        student_writing = student_writing.replace("\r\n", "\n").replace("\r", "\n")
        student_writing = unicodedata.normalize("NFC", student_writing)

    if not student_writing:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        errors = api_call_1_extract_errors(student_writing)
    except Exception as e:
        print(f"[get_feedback] 오류 추출 API 실패: {e}")
        return (
            jsonify({"error": "API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요."}),
            503,
        )

    if not errors:
        return (
            jsonify(
                {
                    "feedback": {
                        "original_text": student_writing,
                        "error_list": [],
                        "kwic_examples": "",
                        "sentences": [],
                    }
                }
            ),
            200,
        )

    try:
        # (2) 버전: 단원/핵심 표현 없음 → 모든 오류를 일반 오류로 처리
        classification_by_id = {
            e.get("id", i): {
                "error_id": e.get("id", i),
                "is_key_expression_error": False,
                "matched_key_expression": None,
            }
            for i, e in enumerate(errors)
        }
        corrections_map = api_call_get_corrections(errors)
    except Exception as e:
        print(f"[get_feedback] 보정/분류 API 실패: {e}")
        return (
            jsonify({"error": "API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요."}),
            503,
        )

    # 한글 오류는 전용 번역 API로만 보정 사용(일반 보정 API가 잘못 매핑되는 경우 방지)
    korean_retry = []
    for e in errors:
        eid = e.get("id")
        span = (e.get("error_span") or "").strip()
        desc = (e.get("description") or "").strip()
        is_korean_error = _contains_hangul(span) or "한국어" in desc or "번역" in desc
        if not is_korean_error:
            continue
        sentence = (e.get("sentence") or "").strip()
        korean_retry.append((eid, span, sentence))
    if korean_retry:
        try:
            translated = api_call_translate_korean_to_english(korean_retry)
            for eid, eng in translated.items():
                if eng and not _contains_hangul(eng):
                    corrections_map[eid] = eng
        except Exception as e:
            print(f"[get_feedback] 한글 번역 API 실패(무시): {e}")

    try:
        ai_codes = api_call_classify_errors_to_louvain_batch(
            errors, corrections_map, classification_by_id
        )
    except Exception as e:
        print(f"[get_feedback] Louvain 분류 API 실패: {e}")
        return (
            jsonify({"error": "API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요."}),
            503,
        )

    # 원문 + 오류별 correction, 매칭된 질문·예문
    error_list = []
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        c = classification_by_id.get(eid, {})
        is_key = False  # (2) 버전: 단원 핵심 표현 미사용
        try:
            correction = (
                corrections_map.get(eid)
                or (corrections_map.get(int(eid)) if isinstance(eid, str) else None)
                or ""
            )
        except (ValueError, TypeError):
            correction = corrections_map.get(eid) or ""
        desc = (e.get("description") or "").strip()
        span = (e.get("error_span") or "").strip()
        sent = (e.get("sentence") or "").strip()
        # AI 분류 코드로 먼저 매칭, 없으면 기존 키워드 매칭
        code = ai_codes.get(eid) or ai_codes.get(int(eid)) or ai_codes.get(str(eid))
        matched = _get_louvain_row_by_code(code) if code else None
        if not matched:
            matched = _match_error_to_csv(desc, error_span=span, correction=correction)
        question = (matched.get("question") or "").strip() if matched else ""
        question_source = "csv" if (matched and question) else "ai"
        csv_ex = (matched.get("example_sentence") or "").strip() if matched else ""
        example_sentences = []
        if csv_ex:
            parts = [
                p.strip() for p in csv_ex.replace("|", "\n").split("\n") if p.strip()
            ][:5]
            example_sentences = [
                {"text": _clean_example_sentence(p), "source": "csv"} for p in parts
            ]
        error_list.append(
            {
                "id": eid,
                "sentence": e.get("sentence", ""),
                "error_span": span,
                "description": desc,
                "correction": correction,
                "is_key_expression": is_key,
                "question": question,
                "question_source": question_source,
                "example_sentences": example_sentences,
            }
        )

    return (
        jsonify(
            {
                "feedback": {
                    "original_text": student_writing,
                    "error_list": error_list,
                    "kwic_examples": "",
                    "sentences": [],
                }
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)

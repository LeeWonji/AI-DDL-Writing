import json
import os
import re
import time
import unicodedata

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import pandas as pd

app = Flask(__name__)
_openai_client = None


def _get_openai_client():
    """OPENAI_API_KEY를 사용하는 클라이언트. 한 번만 생성."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# API 재시도 (일시적 오류 완화)
def _openai_chat_with_retry(
    model, messages, max_tokens=1500, temperature=0.0, max_retries=3
):
    last_err = None
    for attempt in range(max_retries):
        try:
            client = _get_openai_client()
            return client.chat.completions.create(
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


def _cell_str(val):
    """CSV 셀 값을 안전하게 문자열로. float/NaN은 빈 문자열."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in ("nan", "none") else s


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
            code = _cell_str(r.get("code"))
            if not code:
                continue
            label_en = _cell_str(r.get("label_en"))
            desc_en = _cell_str(r.get("desc_en"))
            question = _cell_str(r.get("question"))
            ex = _cell_str(r.get("Example sentences") or r.get("example_sentence"))
            if ex:
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


@app.route("/api/random-topic", methods=["GET"])
def api_random_topic():
    """랜덤 글쓰기 주제 한 개를 OpenAI로 생성해 반환. 응답: { \"topic\": \"...\" }"""
    prompt = """Give one short English writing topic for elementary or middle school ESL students.
Reply with only the topic in one sentence or phrase, in Korean. No number, no explanation.
Example: 나의 주말 일과, 내가 좋아하는 음식 소개하기"""
    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You suggest one simple writing topic in Korean only. Output nothing but the topic text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        topic = raw.split("\n")[0].strip() if raw else ""
        if not topic:
            return jsonify({"error": "주제를 생성하지 못했습니다."}), 503
        return jsonify({"topic": topic}), 200
    except Exception as e:
        print(f"[api/random-topic] 실패: {e}")
        return (
            jsonify({"error": "주제를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."}),
            503,
        )


def _parse_json_from_response(content):
    if not content or not isinstance(content, str):
        return None
    content = content.strip()
    # 코드 블록 제거
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if m:
        content = m.group(1).strip()

    def _try_load(s):
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # trailing comma 제거 시도 (LLM이 자주 출력)
        t = re.sub(r",\s*([}\]])", r"\1", s)
        if t != s:
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                pass
        return None

    # 1) 전체 파싱 시도
    out = _try_load(content)
    if out is not None:
        return out
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
                    out = _try_load(content[start : i + 1])
                    if out is not None:
                        return out
                    break
    # 3) 정규식으로 배열 부분만
    bracket = re.search(r"\[[\s\S]*\]", content)
    if bracket:
        out = _try_load(bracket.group(0))
        if out is not None:
            return out
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

위 글에서 발견한 모든 오류를 나열하세요. 오류가 없으면 errors에 빈 배열을 넣으세요.
- 문법·철자·굴절·전치사 등 모든 영어 오류를 포함하세요.
- **error_span은 반드시 위 학생 작문 원문에 나온 문자열을 그대로 복사**하세요. (예: "thinked"가 있으면 error_span은 "thinked")
응답은 반드시 아래 형태의 JSON 하나만 출력하세요. 다른 설명 없이 JSON만.
{{"errors": [{{"id": 0, "sentence": "오류가 포함된 문장", "error_span": "원문과 동일한 문자열", "description": "오류 유형"}}, ...]}}"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ESL/EFL error detector. List every real error including: spelling (rabit→rabbit), tense (e.g. 'when I am a child' in a past context → should be 'when I was a child'), and grammar. Do NOT flag correct pronoun use: object pronouns (him, her, them, me) after a verb or preposition are correct. Only flag pronouns when case is wrong. Use for error_span the EXACT substring from the student writing. Reply with a single JSON object: {\"errors\": [...]}. No other text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content or ""
        # {"errors": [...]} 형태 또는 [...] 배열 형태 모두 허용
        parsed = _parse_json_from_response(raw)
        if isinstance(parsed, dict) and "errors" in parsed:
            parsed = parsed["errors"]
        if parsed is None or not isinstance(parsed, list):
            print(f"[API1] 파싱 실패. raw 앞 500자:\n{raw[:500] if raw else 'None'}")
            return []
        writing_nfc = unicodedata.normalize("NFC", writing)
        out = []
        span_keys = ("error_span", "errorSpan", "error", "wrong_word", "incorrect", "word", "phrase", "token", "text")

        def _span_in_writing(candidate):
            """NFC 정규화 후 원문 포함 여부 확인. 포함되면 원문에서의 실제 문자열 반환."""
            if not candidate or not isinstance(candidate, str):
                return None
            s = candidate.strip()
            if not s:
                return None
            s_nfc = unicodedata.normalize("NFC", s)
            idx = writing_nfc.find(s_nfc)
            if idx != -1:
                return writing_nfc[idx : idx + len(s_nfc)]
            if s in writing_nfc:
                return s
            return None

        for e in parsed:
            if not isinstance(e, dict):
                continue
            span = None
            for key in span_keys:
                val = e.get(key)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                span = _span_in_writing(str(val))
                if span:
                    break
            if not span:
                for _, val in e.items():
                    if isinstance(val, str) and len(val) >= 2:
                        span = _span_in_writing(val)
                        if span:
                            break
            if not span:
                continue
            sentence = (
                e.get("sentence")
                or e.get("sentence_text")
                or e.get("context")
                or ""
            )
            if isinstance(sentence, str):
                sentence = sentence.strip()
            else:
                sentence = ""
            eid = e.get("id", len(out))
            out.append(
                {
                    "id": eid,
                    "sentence": sentence,
                    "error_span": span,
                    "description": _cell_str(e.get("description")),
                }
            )
        if not out and writing.strip():
            print(f"[API1] 파싱된 배열은 있으나 유효한 오류 항목 없음. raw 앞 600자:\n{raw[:600]}")
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
    prompt = f"""아래 오류 목록에서 각 error_span에 대해, **문장에서 해당 부분을 그대로 대체했을 때 자연스러운 전체 대체문(구)**을 제시하세요.
- correction은 error_span과 **같은 자리**에 넣었을 때 문장이 자연스러워지도록, 필요한 만큼의 단어/구를 모두 포함하세요. (예: error_span이 "I am a child"이면 correction은 "I was a child"처럼 전체 구를 고친 형태로.)
- **대명사**: 주어 자리에는 I/he/she, **목적어 자리(동사·전치사 뒤)**에는 me/him/her/them을 씁니다. 목적어 자리에 이미 him/her 등이 맞게 쓰였으면 he/she로 '고치지' 마세요. (예: "I miss him"에서 him은 맞음 → 잘못 오류로 들어온 경우 correction은 그대로 "him" 또는 해당 문맥에 맞는 목적어 형태.)
- 영어 오류: 예) "ofen"→"often", "wakes"→"wake", "in bus"→"by bus", "I am a child"→"I was a child".
- **한국어(error_span에 한글이 있는 경우)**: 해당 한국어를 **영어로 번역한 단어/구**를 넣으세요. correction에는 한글이 있으면 안 됩니다. 예) "내"→"my", "최애"→"favorite".
오류 목록 (JSON):
{errors_json}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요. 각 error_id에 대해 correction 하나씩.
[{{"error_id": 0, "correction": "often"}}, {{"error_id": 1, "correction": "I was a child"}}, ...]"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "For each error, output the full replacement phrase that can replace error_span in the sentence and make it correct. Subject position: I/he/she; object position (after verb/preposition): me/him/her/them. Do NOT 'correct' correctly used object pronouns to subject form (e.g. 'him' in 'I miss him' is correct—do not suggest 'he'). Do not abbreviate: e.g. for 'I am a child' give 'I was a child'. Output only a valid JSON array of {error_id, correction}. If error_span contains Korean, correction MUST be the English equivalent only.",
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
    """AI 오류 유형 분류용: code, 설명, 질문 요약. 분류 시 correction과 질문이 맞는 코드를 고르기 위함."""
    lines = []
    for row in _load_louvain_error_tags():
        code = row.get("code")
        if not code:
            continue
        label = (row.get("label_en") or "").strip()
        desc = (row.get("desc_en") or "").strip()
        question = (row.get("question") or "").strip()
        part = f"{code}: {label}. {desc}" if desc else f"{code}: {label}"
        if question:
            part += f" [질문: {question[:80]}{'...' if len(question) > 80 else ''}]"
        lines.append(part)
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
    prompt = f"""Below are errors from an ESL student's writing. Each has sentence, error_span, **correction** (the actual fixed form), and description.
For each error, choose the ONE code from the list below whose **[질문]** (question) fits the correction. The code you choose determines which question and examples the learner will see—so pick the code whose question is about the same grammar point as the correction.
Examples: When the correction is the **same word with only spelling fixed** (e.g. rabit→rabbit, ofen→often), choose the **spelling code (FS)** so the learner sees the spelling question (철자), NOT noun number (복수 -s) or other codes. Correction "he"/"him" → code whose question is about I/me/he/him (GPP). Correction "myself" → code about myself/each other (GPF). Correction "was" → code whose question is about past/present/future tense (GVT).
Return a JSON array: {{"error_id": <id>, "code": "<CODE>"}}. Use only codes from the list.

Error type list (each line has code, description, and [질문: ...] that the learner will see):
{code_list}

Errors (JSON):
{errors_json}

Response (JSON array only):
[{{"error_id": 0, "code": "GVT"}}, {{"error_id": 1, "code": "GPP"}}, ...]"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an ESL error classifier. Pick the code whose [질문] matches the correction. Spelling-only fix (e.g. rabit→rabbit) → use FS (spelling), so the learner sees the spelling question. Tense fix (e.g. am→was) → use GVT. Pronoun he/him → GPP; myself → GPF. Output only a valid JSON array of {error_id, code}. Each code must be from the list.",
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
        if student_writing.strip():
            print(f"[get_feedback] 오류 0건 반환 (원문 길이: {len(student_writing)})")
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

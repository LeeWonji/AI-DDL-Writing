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
    """louvain_tag_examples.csv 단일 파일 로드 (code, label_en, desc_ko, question, example_sentence)."""
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
            desc_ko = _cell_str(r.get("desc_ko"))
            question = _cell_str(r.get("question"))
            ex = _cell_str(r.get("Example sentences") or r.get("example_sentence"))
            if ex:
                ex = ex.replace("|", "\n")
            rows.append(
                {
                    "code": code,
                    "label_en": label_en,
                    "desc_ko": desc_ko,
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
        kw = row.get("label_en", "") + " " + row.get("desc_ko", "")
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


def _ensure_correction_is_replacement_only(correction, error_span):
    """교정이 문장 전체가 아닌 error_span 대체어만 되도록 보정.
    예: correction='My school is good.', error_span='goods' -> 'good'"""
    if not correction or not error_span:
        return correction
    corr = correction.strip()
    span = (error_span or "").strip()
    if not corr or not span:
        return corr
    span_words = span.split()
    corr_words = corr.split()
    if len(span_words) == 1 and len(corr_words) >= 3 and corr.rstrip().endswith("."):
        last = corr_words[-1].rstrip(".,!?;:")
        if last and last.lower() != span.lower():
            return last
    return corr


@app.route("/")
def home():
    return render_template("index.html")


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
    # 4) 단일 객체 {...} 추출 (LLM이 배열 대신 객체만 반환할 때)
    brace = re.search(r"\{[\s\S]*\}", content)
    if brace:
        out = _try_load(brace.group(0))
        if out is not None:
            return out
    return None


def _contains_hangul(s):
    """문자열에 한글이 포함되어 있는지 여부."""
    if not s:
        return False
    return bool(re.search(r"[\uAC00-\uD7A3]", s))


def api_call_generate_example_sentences(items):
    """주어진 단어/구가 문장에 자연스럽게 쓰인 예문을 AI로 생성.
    items = [(error_id, word_or_phrase), ...] -> {error_id: [sentence1, sentence2, ...]}"""
    if not items:
        return {}
    lines = [
        f"- id {eid}: '{correction}'"
        for eid, correction in items
    ]
    lines_block = "\n".join(lines)
    prompt = f"""아래 각 단어/구가 문장에서 자연스럽게 쓰인 **초등·중등 ESL 수준의 짧은 예문**을 2~4개 만들어 주세요.
각 단어/구는 해당 문장에 반드시 포함되어야 합니다.
**중요**: 예문에서 그 단어/구가 쓰인 부분은 **이중별표**로 감싸세요. 예: 단어가 "good"이면 "School is **good**."처럼.

{lines_block}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요. example_sentences는 문자열 배열입니다.
[{{"error_id": 0, "example_sentences": ["Hello! **How** are you?", "Hi, my name is Tom."]}}, ...]"""
    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You write simple ESL-level example sentences in English. Each sentence must naturally contain the given word/phrase. Wrap the target word/phrase in **double asterisks** in each sentence (e.g. for 'good' write 'School is **good**.'). Output only a valid JSON array with error_id and example_sentences (array of strings).",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if not parsed:
            return {}
        if isinstance(parsed, dict) and "error_id" in parsed:
            parsed = [parsed]
        if not isinstance(parsed, list):
            return {}
        out = {}
        for item in parsed:
            if not isinstance(item, dict) or "error_id" not in item:
                continue
            eid = item["error_id"]
            sents = (
                item.get("example_sentences")
                or item.get("examples")
                or item.get("sentences")
                or []
            )
            if isinstance(sents, list):
                lst = [_clean_example_sentence(str(s)) for s in sents if s][:5]
            else:
                lst = []
            # int/str 키 모두 저장하여 lookup 호환
            out[eid] = lst
            out[int(eid)] = lst
            out[str(eid)] = lst
        return out
    except Exception as e:
        print(f"[API generate example sentences] 실패: {e}")
        return {}


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
        if not parsed:
            return {}
        if isinstance(parsed, dict) and "error_id" in parsed:
            parsed = [parsed]
        if not isinstance(parsed, list):
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
- **한글로 쓰인 부분**은 영어로 바꿔 써야 하므로 오류로 포함하세요. (예: "집" → description에 "한국어" 포함)
- **error_span은 반드시 위 학생 작문 원문에 나온 문자열을 그대로 복사**하세요. (예: "thinked"가 있으면 error_span은 "thinked")
응답은 반드시 아래 형태의 JSON 하나만 출력하세요. 다른 설명 없이 JSON만.
{{"errors": [{{"id": 0, "sentence": "오류가 포함된 문장", "error_span": "원문과 동일한 문자열", "description": "오류 유형"}}, ...]}}"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ESL/EFL error detector. List every real error including: spelling (rabit→rabbit), tense (e.g. 'when I am a child' in a past context → should be 'when I was a child'), grammar, and Korean words/phrases that should be written in English (use description '한국어' or 'Korean' for those). Do NOT flag correct pronoun use: object pronouns (him, her, them, me) after a verb or preposition are correct. Only flag pronouns when case is wrong. Use for error_span the EXACT substring from the student writing. Reply with a single JSON object: {\"errors\": [...]}. No other text.",
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
    prompt = f"""아래 오류 목록에서 각 error_span에 대해, **그 자리에 넣을 대체어(단어/구)만** 제시하세요.

**핵심**: correction을 원문의 error_span 자리에 넣었을 때, **전체 문장이 문법적으로 맞아야** 합니다.
- 예: "I have apples because it is cute too." 에서 error_span이 "it is cute too"이면, "because cute"가 되면 문법 오류. → correction은 "it is too cute" (순서 수정)
- 예: "goods"→"good", "I am a child"→"I was a child", "in bus"→"by bus"
- correction은 error_span과 **같은 자리**에 들어갈 구. 넣었을 때 문장이 자연스러워야 함.
- **절대** 문장 전체를 넣지 마세요. "goods"의 correction은 "good" 하나만.

- **대명사**: 주어→I/he/she, 목적어→me/him/her/them.
- **한국어**: 영어 번역만. "내"→"my", "최애"→"favorite".

오류 목록 (JSON):
{errors_json}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요.
[{{"error_id": 0, "correction": "often"}}, {{"error_id": 1, "correction": "good"}}, ...]"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Output the replacement for error_span. When inserted in the sentence, the result MUST be grammatically correct. E.g. 'it is cute too' in '...because it is cute too' → correction 'it is too cute' (not 'cute', which would give ungrammatical 'because cute'). Output only a valid JSON array of {error_id, correction}. If error_span contains Korean, correction MUST be the English equivalent only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if not parsed:
            return {}
        if isinstance(parsed, dict) and "error_id" in parsed:
            parsed = [parsed]
        if not isinstance(parsed, list):
            return {}
        span_by_id = {e.get("id", i): (e.get("error_span") or "").strip() for i, e in enumerate(non_key_errors)}
        out = {}
        for item in parsed:
            if not isinstance(item, dict) or "error_id" not in item:
                continue
            eid = item["error_id"]
            corr = (item.get("correction") or "").strip()
            span = span_by_id.get(eid) or span_by_id.get(int(eid)) or span_by_id.get(str(eid)) or ""
            corr = _ensure_correction_is_replacement_only(corr, span)
            if corr:
                out[eid] = corr
        return out
    except Exception as e:
        print(f"[API corrections] 실패: {e}")
        raise


def _validate_and_fix_cause_code(cause: str, code: str, valid_codes: set) -> str:
    """cause와 code 불일치 시 키워드 기반으로 code 보정."""
    c = (cause or "").strip().lower()
    if not c or code not in valid_codes:
        return code

    # WO(어순): cause가 어순/도치를 다루는데 LCS, GVT 등이면 WO로
    wo_keywords = ["어순", "도치", "주어", "동사", "위치", "순서", "배치", "바뀌어야", "앞에 와야"]
    has_wo = any(kw in c for kw in wo_keywords)
    has_question_inversion = "의문" in c and any(kw in c for kw in ["위치", "도치", "바뀌"])
    if (has_wo or has_question_inversion) and code in ("LCS", "GVT", "GA", "GPR"):
        return "WO" if "WO" in valid_codes else code

    # LCS(종속 접속사): cause가 접속사 등을 다루는데 WO, GVT면 LCS로
    lcs_keywords = ["접속사", "종속절", "이유", "조건"]
    if any(kw in c for kw in lcs_keywords) and code in ("WO", "GVT"):
        return "LCS" if "LCS" in valid_codes else code

    # GVT(시제): cause가 시제를 다루는데 WO, LCS면 GVT로
    gvt_keywords = ["시제", "과거", "현재", "미래"]
    if any(kw in c for kw in gvt_keywords) and code in ("WO", "LCS"):
        return "GVT" if "GVT" in valid_codes else code

    # XVPR(동사 전치사): cause가 동사+전치사인데 GA, LCS면 XVPR로
    has_xvpr = "전치사" in c and any(kw in c for kw in ["동사", "필요"])
    if has_xvpr and code in ("GA", "LCS"):
        return "XVPR" if "XVPR" in valid_codes else code

    # GADVO(부사 위치): cause가 부사 위치인데 LCS, GVT, WO면 GADVO로 (형용사 순서 제외)
    gadvo_keywords = ["부사", "위치", "자리", "형용사 앞", "동사 뒤"]
    has_gadvo = any(kw in c for kw in gadvo_keywords) and "부사" in c
    has_adj_order = "형용사" in c
    if has_gadvo and not has_adj_order and code in ("LCS", "GVT", "WO"):
        return "GADVO" if "GADVO" in valid_codes else code

    return code


def _build_louvain_code_list():
    """AI 오류 유형 분류용: code와 desc_ko만 사용. 분류 시 원인(cause)과 설명이 같은 오류 유형인 코드를 고르기 위함."""
    lines = []
    for row in _load_louvain_error_tags():
        code = row.get("code")
        if not code:
            continue
        desc = (row.get("desc_ko") or "").strip()
        part = f"{code}: {desc}" if desc else code
        lines.append(part)
    return "\n".join(lines)


def api_call_classify_errors_to_louvain_batch(
    errors, corrections_map, classification_by_id
):
    """오류별 (sentence, error_span, correction)을 주고, 먼저 correction 원인을 기술한 뒤 그에 맞는 louvain 코드를 반환. {error_id: code}."""
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
            }
        )
    errors_json = json.dumps(payload, ensure_ascii=False)
    prompt = f"""아래는 ESL 학생 작문의 오류 목록입니다. 각 항목에는 sentence, error_span, correction이 있습니다.

**1단계**: 각 오류에 대해, correction을 왜 그렇게 했는지 **원인(문법 규칙·이유)**을 한 문장으로 기술하세요.
예: "the date"→"for the date" → 원인: "동사 wait는 목적어 앞에 전치사 for가 필요하다"
예: "it is cute too"→"it is too cute" → 원인: "부사 too는 형용사 앞에 와야 한다"
예: "goods"→"good" → 원인: "철자 오류"
예: "a apple"→"an apple" → 원인: "모음 앞에서는 an을 쓴다"

**2단계**: 그 원인이 설명하는 오류 유형과 **같은 내용**을 다루는 코드를 아래 목록에서 하나 고르세요.
- 원인과 설명(desc_ko)이 같은 오류 유형을 다루는 코드를 선택해야 합니다.
- 단순히 correction에 포함된 단어(a, the 등)가 있어서가 아니라, **원인**에 맞는 코드를 고르세요.

오류 유형 목록 (각 줄에 code: 설명):
{code_list}

오류 목록 (JSON):
{errors_json}

응답은 반드시 아래 형식의 JSON 배열만 출력하세요. cause는 한글이나 영어 모두 가능합니다.
[{{"error_id": 0, "cause": "동사 wait는 전치사 for가 필요하다", "code": "XVPR"}}, {{"error_id": 1, "cause": "철자 오류", "code": "FS"}}, ...]"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "For each error: 1) State the CAUSE (why the correction was made—the grammar rule violated). 2) Pick the Louvain code whose description (설명) covers that SAME error type. Match by CAUSE, not by words in the correction. E.g. 'the date'→'for the date' cause: verb needs preposition 'for' → XVPR, not GA. Output only a valid JSON array of {error_id, cause, code}. Each code must be from the list.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_from_response(raw)
        if not parsed:
            return {}
        if isinstance(parsed, dict) and "error_id" in parsed:
            parsed = [parsed]
        if not isinstance(parsed, list):
            return {}
        out = {}
        causes_out = {}
        valid_codes = {row.get("code") for row in _load_louvain_error_tags()}
        for item in parsed:
            if not isinstance(item, dict) or "error_id" not in item or "code" not in item:
                continue
            eid = item["error_id"]
            code = (item.get("code") or "").strip().upper()
            cause = (item.get("cause") or "").strip()
            code = _validate_and_fix_cause_code(cause, code, valid_codes)
            if code in valid_codes:
                out[eid] = code
                out[int(eid)] = code
                out[str(eid)] = code
            if cause:
                causes_out[eid] = cause
                causes_out[int(eid)] = cause
                causes_out[str(eid)] = cause
        return out, causes_out
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
        # 단원/핵심 표현 없음 → 모든 오류를 일반 오류로 처리
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
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        span = (e.get("error_span") or "").strip()
        desc = (e.get("description") or "").strip()
        is_korean_error = _contains_hangul(span) or "한국어" in desc or "번역" in desc
        if not is_korean_error:
            continue
        sentence = (e.get("sentence") or "").strip()
        korean_retry.append((eid, span, sentence))
    korean_error_ids = []
    if korean_retry:
        korean_error_ids = [t[0] for t in korean_retry]
        try:
            translated = api_call_translate_korean_to_english(korean_retry)
            for eid, eng in translated.items():
                if eng and not _contains_hangul(eng):
                    corrections_map[eid] = eng
                    corrections_map[int(eid)] = eng
                    corrections_map[str(eid)] = eng
        except Exception as e:
            print(f"[get_feedback] 한글 번역 API 실패(무시): {e}")

    # 3. 한글→영어 교정된 뜻으로 예문 AI 생성 (KO 전용)
    ko_examples_map = {}
    if korean_error_ids:
        ko_items = [
            (eid, corrections_map.get(eid) or corrections_map.get(int(eid)) or corrections_map.get(str(eid)) or "")
            for eid in korean_error_ids
        ]
        ko_items = [(eid, c) for eid, c in ko_items if c]
        if ko_items:
            try:
                ko_examples_map = api_call_generate_example_sentences(ko_items)
            except Exception as e:
                print(f"[get_feedback] KO 예문 생성 실패(무시): {e}")

    try:
        ai_codes, ai_causes = api_call_classify_errors_to_louvain_batch(
            errors, corrections_map, classification_by_id
        )
    except Exception as e:
        print(f"[get_feedback] Louvain 분류 API 실패: {e}")
        return (
            jsonify({"error": "API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요."}),
            503,
        )

    # 한글 오류는 KO 유형으로 강제 매칭
    for eid in korean_error_ids:
        ai_codes[eid] = "KO"
        ai_codes[int(eid)] = "KO"
        ai_codes[str(eid)] = "KO"

    # 철자 오류는 FS로 강제 매칭 (description 또는 correction이 같은 단어 철자 수정인 경우)
    def _is_spelling_error(desc, error_span=None, correction=None):
        if desc:
            d = desc.lower()
            if "spelling" in d or "철자" in desc or "spell" in d or "typo" in d or "misspell" in d:
                return True
        # correction이 error_span과 같은 단어의 철자 수정인 경우 (편집 거리 1~2)
        if error_span and correction and isinstance(error_span, str) and isinstance(correction, str):
            sp, co = error_span.strip().lower(), correction.strip().lower()
            if sp and co and len(sp) >= 2 and len(co) >= 2:
                if sp == co:
                    return True
                diff = sum(1 for a, b in zip(sp, co) if a != b) + abs(len(sp) - len(co))
                if diff <= 2 and len(sp) <= 20:
                    return True
        return False

    spelling_error_ids = []
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        desc = (e.get("description") or "").strip()
        span = (e.get("error_span") or "").strip()
        corr = corrections_map.get(eid) or corrections_map.get(int(eid)) or corrections_map.get(str(eid)) or ""
        if _is_spelling_error(desc, error_span=span, correction=corr):
            spelling_error_ids.append(eid)
    for eid in spelling_error_ids:
        ai_codes[eid] = "FS"
        ai_codes[int(eid)] = "FS"
        ai_codes[str(eid)] = "FS"

    # FS(철자) 오류: AI 예문 생성
    fs_examples_map = {}
    fs_error_ids = [
        e.get("id", i)
        for i, e in enumerate(errors)
        if (ai_codes.get(e.get("id", i)) or ai_codes.get(int(e.get("id", i))) or ai_codes.get(str(e.get("id", i)))) == "FS"
    ]
    if fs_error_ids:
        fs_items = [
            (eid, corrections_map.get(eid) or corrections_map.get(int(eid)) or corrections_map.get(str(eid)) or "")
            for eid in fs_error_ids
        ]
        fs_items = [(eid, c) for eid, c in fs_items if c]
        if fs_items:
            try:
                fs_examples_map = api_call_generate_example_sentences(fs_items)
            except Exception as e:
                print(f"[get_feedback] FS 예문 생성 실패(무시): {e}")

    # 원문 + 오류별 correction, 매칭된 질문·예문
    error_list = []
    for i, e in enumerate(errors):
        eid = e.get("id", i)
        c = classification_by_id.get(eid, {})
        is_key = False  # 단원 핵심 표현 미사용
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
        correction_cause = (
            ai_causes.get(eid) or ai_causes.get(int(eid)) or ai_causes.get(str(eid)) or ""
        )
        matched = _get_louvain_row_by_code(code) if code else None
        if not matched:
            matched = _match_error_to_csv(desc, error_span=span, correction=correction)
        question = (matched.get("question") or "").strip() if matched else ""
        question_source = "csv" if (matched and question) else "ai"
        example_sentences = []
        if code == "KO":
            # KO: AI 생성 예문 사용 (없으면 개별 재시도)
            ko_ex = (
                ko_examples_map.get(eid)
                or ko_examples_map.get(int(eid))
                or ko_examples_map.get(str(eid))
                or []
            )
            if not ko_ex and correction:
                try:
                    fallback = api_call_generate_example_sentences([(eid, correction)])
                    ko_ex = fallback.get(eid) or fallback.get(int(eid)) or fallback.get(str(eid)) or []
                except Exception:
                    pass
            example_sentences = [{"text": t, "source": "ai"} for t in ko_ex]
        elif code == "FS":
            # FS(철자): AI 생성 예문 사용 (없으면 개별 재시도)
            fs_ex = (
                fs_examples_map.get(eid)
                or fs_examples_map.get(int(eid))
                or fs_examples_map.get(str(eid))
                or []
            )
            if not fs_ex and correction:
                try:
                    fallback = api_call_generate_example_sentences([(eid, correction)])
                    fs_ex = fallback.get(eid) or fallback.get(int(eid)) or fallback.get(str(eid)) or []
                except Exception:
                    pass
            example_sentences = [{"text": t, "source": "ai"} for t in fs_ex]
        else:
            csv_ex = (matched.get("example_sentence") or "").strip() if matched else ""
            if csv_ex:
                parts = [
                    p.strip() for p in csv_ex.replace("|", "\n").split("\n") if p.strip()
                ][:5]
                example_sentences = [
                    {"text": _clean_example_sentence(p), "source": "csv"} for p in parts
                ]
            else:
                # CSV에 예문 없음 → AI로 생성
                if correction:
                    try:
                        fallback = api_call_generate_example_sentences([(eid, correction)])
                        ai_ex = (
                            fallback.get(eid)
                            or fallback.get(int(eid))
                            or fallback.get(str(eid))
                            or []
                        )
                        example_sentences = [{"text": t, "source": "ai"} for t in ai_ex]
                    except Exception:
                        pass
        # 5개 미만일 경우 AI로 추가 생성하여 5개 채우기
        if len(example_sentences) < 5 and correction:
            try:
                fallback = api_call_generate_example_sentences([(eid, correction)])
                ai_ex = (
                    fallback.get(eid)
                    or fallback.get(int(eid))
                    or fallback.get(str(eid))
                    or []
                )
                existing_texts = {ex["text"] for ex in example_sentences}
                for t in ai_ex:
                    if len(example_sentences) >= 5:
                        break
                    if t not in existing_texts:
                        example_sentences.append({"text": t, "source": "ai"})
                        existing_texts.add(t)
            except Exception:
                pass
        error_list.append(
            {
                "id": eid,
                "sentence": e.get("sentence", ""),
                "error_span": span,
                "description": desc,
                "correction": correction,
                "correction_cause": correction_cause,
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

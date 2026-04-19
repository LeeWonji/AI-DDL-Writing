import json
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import pandas as pd
from database import (
    init_db,
    get_expected_error_scripts,
)

app = Flask(__name__)
_openai_client = None

# 서버 시작 시 필요한 테이블을 미리 생성
try:
    init_db()
except Exception as e:
    print(f"[init_db] 초기화 실패: {e}")


# 학년/출판사/단원 고정 선택지
GRADE_OPTIONS = ["5학년", "6학년"]
PUBLISHER_OPTIONS = ["YBM(최희경)", "YBM(김혜리)"]
UNIT_OPTIONS = [f"{i}단원" for i in range(1, 13)]

# API에 오류 설명이 있을 때 단원 스크립트 순위: 패턴 + CSV error_description (설명 비중 높음)
_SCRIPT_RANK_WEIGHT_PATTERN_WITH_DESC = 0.35
_SCRIPT_RANK_WEIGHT_DESCRIPTION_WITH_DESC = 0.65


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
    """louvain_tag_examples.csv 단일 파일 로드 (code, label_en, description, question, example_sentence)."""
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
            description = _cell_str(
                r.get("description") or r.get("desc_ko")
            )  # desc_ko: 예전 CSV 호환
            question = _cell_str(r.get("question"))
            ex = _cell_str(r.get("Example sentences") or r.get("example_sentence"))
            if ex:
                ex = ex.replace("|", "\n")
            rows.append(
                {
                    "code": code,
                    "label_en": label_en,
                    "description": description,
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
    """API description과 Louvain CSV의 description·label 정규화 유사도를 우선. 폴백으로 영어 토큰 겹침."""
    if not description or not isinstance(description, str):
        return None
    api_norm = _normalize_for_script_description_match(description)
    if not api_norm:
        return None
    desc_lower = description.strip().lower()
    best_row = None
    best_key = (-1.0, -1, -1)  # (전체유사도, 토큰히트, 라벨유사도) 내림차순 정렬용

    for row in _load_louvain_error_tags():
        label_en = (row.get("label_en") or "").strip()
        row_desc = (row.get("description") or "").strip()
        row_blob = _normalize_for_script_description_match(f"{label_en} {row_desc}")
        full_sim = (
            SequenceMatcher(None, api_norm, row_blob).ratio() if row_blob else 0.0
        )
        label_sim = (
            SequenceMatcher(
                None, api_norm, _normalize_for_script_description_match(label_en)
            ).ratio()
            if label_en
            else 0.0
        )
        kw = f"{label_en} {row_desc}"
        words = re.findall(r"[a-z0-9]{2,}", kw.lower())
        token_hits = sum(
            1 for w in words if w in desc_lower and w not in ("error", "use", "other")
        )
        # 설명 전체 매칭을 최우선, 동점 시 토큰·라벨 보조
        key = (full_sim, token_hits, label_sim)
        if key > best_key:
            best_key = key
            best_row = row
    return best_row


def _clean_example_sentence(text):
    """예문 텍스트 정리."""
    if not text or not isinstance(text, str):
        return text
    return text.strip()


def _strip_bold_markers_for_prompt(text):
    """LLM 프롬프트에 넣을 때 예문의 ** 강조 표시 제거."""
    if not text or not isinstance(text, str):
        return ""
    return text.replace("**", "").strip()


def _postprocess_replacement_with_step4_examples(
    replacement, error_span, example_texts, sentence
):
    """4단계 예문과 어긋나는 흔한 모델 오답을 규칙으로 보정 (예: feeling ↔ feel)."""
    r = (replacement or "").strip()
    span = (error_span or "").strip()
    if not r or not example_texts:
        return r
    ex_join = " ".join(example_texts).lower()
    rl = r.lower()
    span_l = span.lower()
    sent_l = (sentence or "").strip().lower()

    # 예문에 'feel the same' 패턴이 있는데 치환어가 feeling 등으로만 나온 경우
    if "feel the same" in ex_join:
        if rl in ("feeling", "feels", "felt"):
            if "same" in sent_l:
                return "feel the same"
            return "feel"
        if rl == "feel" and "same" in sent_l and "feel the same" in ex_join:
            return "feel the same"

    # 예문에 동사 feel(단어)이 있는데 모델이 명사/형용사용 feeling만 단독 출력
    if rl == "feeling":
        if re.search(r"\bfeel\b", ex_join) and not re.search(
            r"\bfeeling\b", ex_join
        ):
            return "feel"

    # be + 동사 이중 표현(span) + 예문이 feel 패턴일 때, 잘못된 -ing/활용만 고침
    if re.search(r"\b(am|is|are)\b", span_l) and re.search(r"\bfeel\b", span_l):
        if re.search(r"\bfeel\b", ex_join) and not re.search(r"\bfeeling\b", ex_join):
            if "feel the same" in ex_join and "same" in sent_l and rl in (
                "feeling",
                "feels",
                "felt",
                "feel",
            ):
                return "feel the same"
            if rl == "feeling":
                return "feel"

    return r


def _parse_script_examples(raw_examples):
    """예상 오류 스크립트의 예문 문자열을 배열로 변환."""
    if not raw_examples:
        return []
    text = str(raw_examples).strip()
    if not text:
        return []
    # JSON 배열 저장을 우선 지원
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_clean_example_sentence(str(x)) for x in parsed if str(x).strip()][:5]
    except Exception:
        pass
    # fallback: "|" 또는 줄바꿈 구분
    return [
        _clean_example_sentence(s)
        for s in text.replace("|", "\n").split("\n")
        if s and str(s).strip()
    ][:5]


def _expected_script_pattern_score(span, sentence, pattern_lower):
    """error_pattern과의 유사도. span만 보면 짧은 구에서 엉뚱한 패턴이 이기는 경우가 있어 문장도 함께 본다."""
    span_l = (span or "").strip().lower()
    sent_l = (sentence or "").strip().lower()
    r_span = SequenceMatcher(None, span_l, pattern_lower).ratio() if span_l else 0.0
    r_sent = SequenceMatcher(None, sent_l, pattern_lower).ratio() if sent_l else 0.0
    return max(r_span, r_sent)


def _normalize_for_script_description_match(text):
    """API 설명 vs CSV error_description 비교용(공백·대소문자·유니코드 정규화)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", str(text)).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _expected_script_row_key(row):
    """스크립트 행 구분 키(한 피드백에서 같은 행을 두 오류에 중복 배정하지 않기 위함)."""
    try:
        return ("id", int(row.get("id", 0)))
    except (TypeError, ValueError):
        return (
            "pq",
            (row.get("error_pattern") or "").strip(),
            (row.get("question") or "").strip(),
        )


def _rank_expected_script_matches(
    error_span, script_rows, sentence=None, api_error_description=None
):
    """유사도 내림차순, 동점 시 CSV id 오름차순인 (score, row) 목록.

    API에 오류 설명(description)이 있으면 반드시 그것과 CSV error_description을 맞춰 순위를 매긴다.
    이때 error_description이 비어 있는 스크립트 행은 후보에서 제외한다(문장 유사도만으로 고르지 않음).
    API 설명이 없을 때만 패턴·문장 유사도만 사용한다.
    """
    span_l = (error_span or "").strip().lower()
    sent_l = (sentence or "").strip()
    if (not span_l and not sent_l) or not script_rows:
        return []
    api_desc_raw = (api_error_description or "").strip()
    api_norm = _normalize_for_script_description_match(api_desc_raw)
    require_script_description = bool(api_norm)

    items = []
    for row in script_rows:
        pattern = (row.get("error_pattern") or "").strip()
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        pattern_score = _expected_script_pattern_score(
            error_span, sentence, pattern_lower
        )
        script_ed = (row.get("error_description") or "").strip()
        if require_script_description:
            # API 설명이 있는데 스크립트 설명이 없으면 후보 제외
            if not script_ed:
                continue
            desc_ratio = SequenceMatcher(
                None,
                api_norm,
                _normalize_for_script_description_match(script_ed),
            ).ratio()
            score = (
                _SCRIPT_RANK_WEIGHT_PATTERN_WITH_DESC * pattern_score
                + _SCRIPT_RANK_WEIGHT_DESCRIPTION_WITH_DESC * desc_ratio
            )
        else:
            score = pattern_score
        try:
            rid = int(row.get("id", 0))
        except (TypeError, ValueError):
            rid = 0
        items.append((score, rid, row))
    items.sort(key=lambda x: (-x[0], x[1]))
    return [(s, r) for s, rid, r in items]


def _match_expected_script(
    error_span,
    script_rows,
    sentence=None,
    used_row_keys=None,
    api_error_description=None,
):
    """패턴·설명으로 스크립트 행 선택. API에 description이 있으면 CSV error_description 없는 행은 후보에서 제외된다. used_row_keys가 있으면 이미 쓴 행은 건너뛴다."""
    ranked = _rank_expected_script_matches(
        error_span,
        script_rows,
        sentence=sentence,
        api_error_description=api_error_description,
    )
    if not ranked:
        return None

    if used_row_keys is None:
        best_score = ranked[0][0]
        tie = [r for s, r in ranked if abs(s - best_score) < 1e-9]

        def _row_id_obj(r):
            try:
                return int(r.get("id", 0))
            except (TypeError, ValueError):
                return 0

        return min(tie, key=_row_id_obj)

    for _score, row in ranked:
        key = _expected_script_row_key(row)
        if key not in used_row_keys:
            used_row_keys.add(key)
            return row
    # 단원 스크립트가 한 종류뿐이면 중복 허용(최상위 매칭)
    return ranked[0][1]


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


@app.route("/api/options/grades", methods=["GET"])
def api_option_grades():
    return jsonify({"grades": GRADE_OPTIONS}), 200


@app.route("/api/options/publishers", methods=["GET"])
def api_option_publishers():
    grade = (request.args.get("grade") or "").strip()
    if not grade:
        return jsonify({"error": "grade is required"}), 400
    if grade not in GRADE_OPTIONS:
        return jsonify({"publishers": []}), 200
    return jsonify({"publishers": PUBLISHER_OPTIONS}), 200


@app.route("/api/options/units", methods=["GET"])
def api_option_units():
    grade = (request.args.get("grade") or "").strip()
    publisher = (request.args.get("publisher") or "").strip()
    if not grade or not publisher:
        return jsonify({"error": "grade and publisher are required"}), 400
    if grade not in GRADE_OPTIONS or publisher not in PUBLISHER_OPTIONS:
        return jsonify({"units": []}), 200
    return jsonify({"units": UNIT_OPTIONS}), 200


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
- **오류가 난 최소 범위**를 고르세요. `I like to + 동사원형`이 **하고 싶다/정중한 요청** 뜻인데 `I would like to`가 빠진 경우, 잘못된 부분은 보통 **"like" 또는 "I like"** 입니다. **"to have a cake" 같은 to부정식 구만** 잡지 마세요.
응답은 반드시 아래 형태의 JSON 하나만 출력하세요. 다른 설명 없이 JSON만.
{{"errors": [{{"id": 0, "sentence": "오류가 포함된 문장", "error_span": "원문과 동일한 문자열", "description": "오류 유형"}}, ...]}}"""

    try:
        response = _openai_chat_with_retry(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ESL/EFL error detector. List every real error including: spelling, tense, grammar, and Korean that should be English. "
                        "Do NOT flag correct object pronouns after verbs/prepositions. "
                        "error_span must be the EXACT substring from the student text — choose the **minimal** wrong part. "
                        "For 'I like to + verb' meaning a wish/polite desire (same as 'I would like to'), the mistake is usually missing 'would': mark 'like' or 'I like', NOT the whole infinitive phrase 'to have ...'. "
                        "Reply with a single JSON object: {\"errors\": [...]}. No other text."
                    ),
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


def api_call_refine_correction_with_clues(
    sentence,
    error_span,
    draft_correction,
    step3_question,
    step4_example_texts,
):
    """5단계(정확한 표현): 3단계 질문·4단계 예문을 최우선으로 치환어를 정함."""
    sent = (sentence or "").strip()
    span = (error_span or "").strip()
    draft = (draft_correction or "").strip()
    q = (step3_question or "").strip()
    examples = [
        _strip_bold_markers_for_prompt(t)
        for t in (step4_example_texts or [])
        if _strip_bold_markers_for_prompt(t)
    ]
    if not span or not draft:
        return draft
    if not q and not examples:
        return draft

    examples_block = "\n".join(f"- {t}" for t in examples[:10]) if examples else "(없음)"
    q_block = q if q else "(없음)"

    # 3.5는 지시 이탈이 잦아, 5단계 보정만 더 안정적인 모델 사용(환경변수로 변경 가능)
    refine_model = os.getenv("OPENAI_STEP5_REFINE_MODEL", "gpt-4o-mini")

    user_prompt = f"""역할: 학생 화면의 **5단계(정확한 표현)** 에 넣을 영어 치환어만 정한다.

반드시 지킬 것:
- **3단계 질문**과 **4단계 예문**이 가리키는 문법·표현을 그대로 따른다. 초안 교정어는 힌트일 뿐이며, 질문/예문과 충돌하면 **초안은 버린다**.
- 4단계 예문에 나온 **동사 형태·구(phrase)** 를 복사해 쓴다. 예문이 "I feel the same." 형태면 error_span 자리에는 "feel the same" 또는 문맥상 맞는 같은 패턴를 넣는다. 예문이 동사원형 feel을 쓰는데 단독으로 "feeling"만 내지 않는다.
- 출력은 error_span과 **같은 위치**에 끼워 넣을 **영어 문자열 하나**만. JSON의 값으로만 제시한다.

【3단계 — 질문 단서】
{q_block}

【4단계 — 교과서 예문】
{examples_block}

【문맥】
- 전체 문장: {sent}
- error_span (이 부분을 바꿈): {span}
- (참고용, 틀릴 수 있음) 초안 교정어: {draft}

소규모 예시:
- error_span이 "am feel"이고 예문에 "I feel the same." 가 있으면 replacement는 "feel the same" 또는 "feel" 등 예문 패턴에 맞춘다. "feeling"만 단독으로 내지 않는다.

응답 형식(이것만, 설명 없음):
{{"replacement":"여기에 치환어"}}"""

    try:
        response = _openai_chat_with_retry(
            model=refine_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You choose the English text that replaces error_span in the sentence. "
                        "Step 3 (question) and step 4 (examples) are the source of truth; ignore the draft "
                        "if it conflicts. Match verb forms and phrases used in the examples (e.g. use 'feel' / "
                        "'feel the same' when examples do—not a lone 'feeling' if examples use base 'feel'). "
                        "Reply with ONLY valid JSON: {\"replacement\":\"...\"} — no markdown, no extra text."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        refined = ""
        parsed = _parse_json_from_response(raw)
        if isinstance(parsed, dict):
            refined = (parsed.get("replacement") or parsed.get("correction") or "").strip()
        if not refined:
            line = raw.split("\n")[0].strip()
            if (line.startswith('"') and line.endswith('"')) or (
                line.startswith("'") and line.endswith("'")
            ):
                line = line[1:-1].strip()
            refined = line
        if not refined:
            return draft
        refined = _ensure_correction_is_replacement_only(refined, span)
        refined = _postprocess_replacement_with_step4_examples(
            refined, span, examples, sent
        )
        refined = _ensure_correction_is_replacement_only(refined, span)
        return refined if refined else draft
    except Exception as e:
        print(f"[API refine_correction_with_clues] 실패: {e}")
        return draft


def _text_same_ignore_case(a, b):
    """두 문자열이 공백·대소문자 무시하고 동일한지."""
    return (a or "").strip().lower() == (b or "").strip().lower()


def api_call_salvage_correction_when_unchanged(
    sentence,
    error_span,
    step3_question,
    step4_example_texts,
):
    """교정어가 error_span과 같을 때(실질적 교정 실패): 3·4단계 단서만으로 정답 표현 재생성."""
    sent = (sentence or "").strip()
    span = (error_span or "").strip()
    q = (step3_question or "").strip()
    examples = [
        _strip_bold_markers_for_prompt(t)
        for t in (step4_example_texts or [])
        if _strip_bold_markers_for_prompt(t)
    ]
    if not sent or not span or (not q and not examples):
        return ""
    examples_block = "\n".join(f"- {t}" for t in examples[:10])
    q_block = q if q else "(없음)"
    model = os.getenv("OPENAI_STEP5_REFINE_MODEL", "gpt-4o-mini")

    user_prompt = f"""상황: 오류로 표시한 문자열과 모델이 낸 교정어가 **완전히 같아서** 학생에게는 아무것도 고쳐지지 않은 상태입니다. 표시된 구간(error_span)이 잘못 잡혔을 수 있습니다.

【학생 문장】{sent}
【표시된 error_span】{span}

【3단계 질문】
{q_block}

【4단계 예문】
{examples_block}

과제: 3·4단계와 **모순 없이** 이 오류를 반영한 **올바른 영어 표현**을 하나만 정하세요. 예문의 would like / I'd like 패턴을 따르세요.
- 5단계에 보여 줄 한 덩어리면 되며, 원문의 잘못된 span보다 **긴 구**여도 됩니다(예: I'd like to have a cake).

응답 형식만: {{"replacement":"..."}}"""

    try:
        response = _openai_chat_with_retry(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with ONLY valid JSON: {\"replacement\":\"...\"}. "
                        "The replacement is the correct English for the student, aligned with the question and examples. "
                        "It may be longer than the marked span if the span was misidentified."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_json_from_response(raw)
        if not isinstance(parsed, dict):
            return ""
        rep = (parsed.get("replacement") or "").strip()
        if not rep or _text_same_ignore_case(rep, span):
            return ""
        return rep
    except Exception as e:
        print(f"[API salvage_correction_when_unchanged] 실패: {e}")
        return ""


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
    """AI 오류 유형 분류용: code와 description만 사용. 분류 시 원인(cause)과 설명이 같은 오류 유형인 코드를 고르기 위함."""
    lines = []
    for row in _load_louvain_error_tags():
        code = row.get("code")
        if not code:
            continue
        desc = (row.get("description") or "").strip()
        part = f"{code}: {desc}" if desc else code
        lines.append(part)
    return "\n".join(lines)


def api_call_classify_errors_to_louvain_batch(
    errors, corrections_map, classification_by_id
):
    """오류별 sentence, error_span, correction, description을 주고 Louvain 코드를 반환. {error_id: code}."""
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
    prompt = f"""아래는 ESL 학생 작문의 오류 목록입니다. 각 항목에는 sentence, error_span, correction, **description**(오류 탐지 시스템이 붙인 설명)이 있습니다.

**0단계(필수)**: 각 오류의 **description**을 반드시 읽고, 아래 목록의 각 코드 **description**과 어떤 유형이 같은지에 초점을 맞추세요. 문장 표면이 비슷하다는 이유만으로 코드를 고르지 마세요.

**1단계**: correction을 왜 그렇게 했는지 **원인(문법 규칙·이유)**을 한 문장으로 기술하세요. 이때 **description**과 모순되지 않게 하세요.
예: "the date"→"for the date" → 원인: "동사 wait는 목적어 앞에 전치사 for가 필요하다"
예: "it is cute too"→"it is too cute" → 원인: "부사 too는 형용사 앞에 와야 한다"
예: "goods"→"good" → 원인: "철자 오류"
예: "a apple"→"an apple" → 원인: "모음 앞에서는 an을 쓴다"

**2단계**: **오류 탐지 description**, **원인(cause)**, 아래 목록의 **코드별 description**이 서로 같은 오류 유형을 가리키는 코드를 하나 고르세요.
- 세 가지(탐지 설명·원인·목록 설명)가 어긋나면 **목록의 코드 description**과 **탐지 description**을 우선 맞추세요.
- 단순히 correction에 들어 있는 단어(a, the 등)만 보고 고르지 마세요.

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
                    "content": "For each error you MUST use the detector's `description` field together with sentence/error_span/correction. Pick the Louvain code whose list description matches the same error TYPE as that detector description (and your stated cause). Do not pick a code based only on surface similarity of words in the sentence. Output only a valid JSON array of {error_id, cause, code}. Each code must be from the list.",
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
    data = request.get_json() or {}
    student_writing = (data.get("writing") or "").strip()
    writing_mode = (data.get("writing_mode") or "unit").strip().lower()
    selected_grade = (data.get("grade") or "").strip()
    selected_publisher = (data.get("publisher") or "").strip()
    selected_unit = (data.get("unit") or "").strip()
    if isinstance(student_writing, str) and student_writing:
        student_writing = student_writing.replace("\r\n", "\n").replace("\r", "\n")
        student_writing = unicodedata.normalize("NFC", student_writing)

    if not student_writing:
        return jsonify({"error": "Missing required fields"}), 400
    is_unit_mode = writing_mode == "unit"
    if is_unit_mode and (
        not selected_grade or not selected_publisher or not selected_unit
    ):
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

    expected_script_rows = []
    if is_unit_mode:
        try:
            expected_script_rows = get_expected_error_scripts(
                selected_grade, selected_publisher, selected_unit
            )
        except Exception as e:
            print(f"[get_feedback] 예상 오류 스크립트 로드 실패: {e}")
            expected_script_rows = []

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
    # 같은 요청에서 서로 다른 오류가 동일 expected_script 행(id)에 붙는 것 방지
    used_expected_script_row_keys = set()
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
        # 질문 폴백용: Louvain 코드 → 키워드 CSV (단원 스크립트에 질문 없을 때만 사용)
        code = ai_codes.get(eid) or ai_codes.get(int(eid)) or ai_codes.get(str(eid))
        correction_cause = (
            ai_causes.get(eid) or ai_causes.get(int(eid)) or ai_causes.get(str(eid)) or ""
        )
        # 단원 expected_script — KO·FS는 내용을 미리 적어둘 수 없어 스크립트 매칭 생략(Louvain·AI만).
        matched_script = None
        if code not in ("KO", "FS"):
            matched_script = _match_expected_script(
                span,
                expected_script_rows,
                sentence=sent,
                used_row_keys=used_expected_script_row_keys,
                api_error_description=desc,
            )
        question = ""
        question_source = "ai"
        source_track = "unexpected_mapping"
        example_sentences = []
        if matched_script:
            script_question = (matched_script.get("question") or "").strip()
            script_examples = _parse_script_examples(
                matched_script.get("example_sentences")
            )
            if script_question:
                question = script_question
                question_source = "expected_script"
                source_track = "expected_script"
            if script_examples:
                example_sentences = [
                    {"text": t, "source": "expected_script"} for t in script_examples
                ]
        # 스크립트에 질문이 없을 때만 Louvain(또는 설명 CSV)에서 질문
        matched = _get_louvain_row_by_code(code) if code else None
        if not matched:
            matched = _match_error_to_csv(desc, error_span=span, correction=correction)
        if not (question and question.strip()):
            louvain_q = (matched.get("question") or "").strip() if matched else ""
            if louvain_q:
                question = louvain_q
                question_source = "csv"
                source_track = "unexpected_mapping"
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
            if not example_sentences:
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
            if not example_sentences:
                example_sentences = [{"text": t, "source": "ai"} for t in fs_ex]
        else:
            if not example_sentences:
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
        # 3·4단계 단서용 예문 텍스트 (5단계 보정·salvage에서 공통 사용)
        ex_for_clues = [
            _strip_bold_markers_for_prompt((ex.get("text") or "").strip())
            for ex in example_sentences
            if (ex.get("text") or "").strip()
        ]
        # 5단계 표시용 correction: 3·4단서(질문·예문)와 맞게 초안을 재정렬
        if correction and span:
            if (question and question.strip()) or ex_for_clues:
                try:
                    correction = api_call_refine_correction_with_clues(
                        sentence=sent,
                        error_span=span,
                        draft_correction=correction,
                        step3_question=question or "",
                        step4_example_texts=ex_for_clues,
                    )
                except Exception as e:
                    print(f"[get_feedback] 5단계 교정어 보정 실패(초안 유지): {e}")
                # API 실패·모델 오류 시에도 예문 패턴과 맞추는 후처리 적용
                correction = _postprocess_replacement_with_step4_examples(
                    correction, span, ex_for_clues, sent
                )
                correction = _ensure_correction_is_replacement_only(correction, span)
        # 교정어가 여전히 error_span과 동일 → span 오지정으로 치환이 불가했을 가능성 → 단서만으로 재생성
        if (
            span
            and correction
            and _text_same_ignore_case(correction, span)
            and ((question and question.strip()) or ex_for_clues)
        ):
            try:
                salvaged = api_call_salvage_correction_when_unchanged(
                    sentence=sent,
                    error_span=span,
                    step3_question=question or "",
                    step4_example_texts=ex_for_clues,
                )
                if salvaged:
                    correction = salvaged
            except Exception as e:
                print(f"[get_feedback] 교정 salvage 실패(유지): {e}")
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
                "source_track": source_track,
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
    init_db()
    app.run(debug=True, port=5002)

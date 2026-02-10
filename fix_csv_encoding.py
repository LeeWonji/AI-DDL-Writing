#!/usr/bin/env python3
"""
CSV 한글이 ?로 깨지는 문제 수정 스크립트.

원인: Excel에서 'CSV(쉼표 구분)'로 저장하면 시스템 인코딩(CP949 등)으로 저장되거나,
     다른 프로그램이 UTF-8이 아닌 인코딩으로 저장하면 한글이 ?로 바뀝니다.

해결: 이 스크립트는 key_expressions_grade3.csv를 UTF-8(BOM)으로 다시 저장하고,
     첫 번째 데이터 행의 한글(script_kr, key_expression_kr)과 출판사명을 올바르게 넣습니다.
     다른 단원/출판사 데이터는 나중에 CSV를 UTF-8로 유지한 채로 편집하면 됩니다.

사용: 프로젝트 루트에서  python fix_csv_encoding.py
"""
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas 필요: pip install pandas")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "key_expressions_grade3.csv")

# 3·4학년 출판사 순서 (CSV 행 순서와 맞춤)
PUBLISHERS_34 = [
    "YBM(김)",
    "YBM(최)",
    "동아(윤)",
    "미래엔(강)",
    "아이스크림(박)",
    "천재(김)",
    "천재(이)",
    "천재(함)",
]


def main():
    if not os.path.exists(CSV_PATH):
        print(f"파일 없음: {CSV_PATH}")
        sys.exit(1)

    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            break
        except Exception:
            continue
    else:
        print("CSV를 읽을 수 없습니다. 인코딩을 확인하세요.")
        sys.exit(1)

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # 출판사가 깨진 행(YBM(?), ??(?) 등)을 실제 출판사명으로 교체 (행 순서로 매핑)
    pub_index = -1
    for i, row in df.iterrows():
        p = row.get("publisher")
        if pd.isna(p):
            continue
        p = str(p).strip()
        if not p:
            continue
        if "YBM" in p and "?" in p:
            pub_index += 1
            if pub_index < len(PUBLISHERS_34):
                df.at[i, "publisher"] = PUBLISHERS_34[pub_index]
        elif "?" in p or (len(p) <= 6 and "(" in p):
            pub_index += 1
            if pub_index < len(PUBLISHERS_34):
                df.at[i, "publisher"] = PUBLISHERS_34[pub_index]

    # 첫 번째로 script_en이 채워진 행 = YBM(김) 1단원 → 한글 컬럼 보정
    for i, row in df.iterrows():
        script_en = row.get("script_en")
        if pd.notna(script_en) and str(script_en).strip():
            df.at[i, "publisher"] = "YBM(김)"
            df.at[i, "unit_theme"] = "인사"
            df.at[i, "script_kr"] = "안녕, 나는 데이비드야. 만나서 반가워."
            df.at[i, "key_expression_kr"] = "나는 ~야, 만나서 반가워"
            break

    # 반드시 UTF-8 BOM으로 저장 (Excel에서도 한글이 깨지지 않도록)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {CSV_PATH} (인코딩: UTF-8 BOM)")
    print(
        "앞으로 이 CSV는 'UTF-8' 또는 'UTF-8 BOM'으로만 저장하세요. Excel에서는 'CSV UTF-8(쉼표 구분)'으로 저장하세요."
    )


if __name__ == "__main__":
    main()

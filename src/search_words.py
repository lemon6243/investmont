# search_words.py
# 목적: 병원/은행 데모용 실용단어가 AI Hub morpheme(전체 단어사전)에 있는지 검색
# 핵심: 무거운 keypoint 압축 불필요. morpheme JSON만 스캔 -> 지금 당장 실행 가능.
# 결과: 어떤 단어가 있고(몇 개), 어떤 단어는 없어서 웹캠으로 직접 찍어야 하는지 판정

import os
import json
from collections import Counter
from load_dataset import find_morpheme_files

# ------------------------------------------------------------
# 설정 (collect_all.py와 동일 경로)
# ------------------------------------------------------------
MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"

# 병원/은행 데모용 실용단어 (설계도 모듈6 기준)
TARGET_WORDS = [
    # 공통
    "신분증", "여기", "있다", "원하다", "얼마",
    # 병원
    "머리", "배", "아프다", "진료", "예약", "약", "처방", "병원",
    # 은행
    "계좌", "만들다", "돈", "찾다", "이체", "은행", "도와주다",
    # 자주 쓰는 기본
    "감사하다", "안녕하다", "네", "아니다",
]

# 유의어까지 같이 찾기 (한 개념을 여러 단어로 표현할 수 있으므로)
# key = 대표단어, value = 같이 검색할 후보들
SYNONYMS = {
    "아프다": ["아프다", "아픔", "통증", "아파"],
    "원하다": ["원하다", "싶다", "바라다"],
    "도와주다": ["도와주다", "돕다", "도움", "도와"],
    "만들다": ["만들다", "개설", "생성"],
    "찾다": ["찾다", "출금", "인출"],
    "이체": ["이체", "송금", "보내다"],
    "진료": ["진료", "진찰", "검진"],
    "예약": ["예약", "접수"],
}


def scan_all_labels(morpheme_files):
    """morpheme 전체를 읽어 등장한 모든 단어 라벨의 개수를 센다."""
    label_count = Counter()
    total = len(morpheme_files)
    for i, mf in enumerate(morpheme_files):
        if i % 40000 == 0:
            print(f"   ...스캔 {i}/{total}")
        try:
            with open(mf, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        if not m.get("data"):
            continue
        try:
            label = m["data"][0]["attributes"][0]["name"]
        except (KeyError, IndexError):
            continue
        label_count[label] += 1
    return label_count


def main():
    print("=" * 60)
    print("[단어 검색] 병원/은행 실용단어가 AI Hub에 있는지 확인")
    print("  (morpheme만 스캔 - keypoint 압축 불필요)")
    print("=" * 60)

    morpheme_files = find_morpheme_files(MORPHEME)
    print(f">> morpheme JSON 개수: {len(morpheme_files)}")
    print(">> 전체 단어 라벨 스캔 중... (시간이 조금 걸립니다)")

    label_count = scan_all_labels(morpheme_files)
    all_labels = set(label_count.keys())
    print(f">> AI Hub에 존재하는 서로 다른 단어 수: {len(all_labels)}\n")

    # ----------------------------------------------------
    # 목표 단어 판정
    # ----------------------------------------------------
    found = []       # (대표단어, 실제매칭단어, 개수)
    not_found = []   # 대표단어

    print("=" * 60)
    print(" 목표단어 검색 결과")
    print("=" * 60)
    print(f"{'목표단어':10s} {'상태':6s} {'매칭단어(개수)'}")
    print("-" * 60)

    for word in TARGET_WORDS:
        # 유의어 후보 목록 (없으면 자기 자신만)
        candidates = SYNONYMS.get(word, [word])
        # AI Hub 라벨 중 후보와 정확히 일치하거나 후보를 포함하는 것 찾기
        hits = []
        for cand in candidates:
            for label in all_labels:
                if cand == label or cand in label:
                    hits.append((label, label_count[label]))
        # 중복 제거
        hits = sorted(set(hits), key=lambda x: -x[1])

        if hits:
            matched_str = ", ".join(f"{l}({c})" for l, c in hits[:4])
            print(f"{word:10s} {'있음':6s} {matched_str}")
            found.append((word, hits))
        else:
            print(f"{word:10s} {'없음':6s} -")
            not_found.append(word)

    # ----------------------------------------------------
    # 요약
    # ----------------------------------------------------
    print("\n" + "=" * 60)
    print(" 요약")
    print("=" * 60)
    print(f">> AI Hub에 있는 단어: {len(found)}개")
    print(f"   {[w for w, _ in found]}")
    print(f">> 웹캠으로 직접 찍어야 할 단어: {len(not_found)}개")
    print(f"   {not_found}")

    # ----------------------------------------------------
    # 결과를 파일로 저장 (다음 단계에서 활용)
    # ----------------------------------------------------
    result = {
        "found": {w: [{"label": l, "count": c} for l, c in hits]
                  for w, hits in found},
        "not_found": not_found,
        "total_labels_in_aihub": len(all_labels),
    }
    with open("word_search_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n>> 상세 결과 저장: word_search_result.json")


if __name__ == "__main__":
    main()

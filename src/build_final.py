# build_final.py
# 목적: collect_all로 모은 전체 데이터에서, 샘플이 충분한 단어만 골라
#       최종 학습셋(X.npy, y.npy, groups.npy, label2idx.json)을 생성.
# 사용법: 모든 압축 수집이 끝난 뒤 딱 한 번 실행.

import os
import json
import numpy as np
from collections import Counter

ALL_DIR = "dataset_all"      # collect_all이 쌓은 폴더
OUT_DIR = "dataset"          # train.py가 읽는 최종 폴더

MIN_SAMPLES = 20    # 이 개수 이상 모인 단어만 학습에 사용 (원하는 대로 조정)
MAX_WORDS = None    # 상위 몇 개 단어만 쓸지 (None=조건 만족하는 전부)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(os.path.join(ALL_DIR, "X_all.npy"))
    y = np.load(os.path.join(ALL_DIR, "y_all.npy"), allow_pickle=True)
    g = np.load(os.path.join(ALL_DIR, "groups_all.npy"), allow_pickle=True)
    print(">> 전체 누적:", X.shape, " 서로 다른 단어:", len(set(y.tolist())))

    # 단어별 개수 세기
    counts = Counter(y.tolist())

    # 조건 만족하는 단어 선정 (개수 많은 순)
    chosen = [w for w, c in counts.most_common() if c >= MIN_SAMPLES]
    if MAX_WORDS is not None:
        chosen = chosen[:MAX_WORDS]
    chosen_set = set(chosen)
    print(f">> 선택된 단어 수: {len(chosen)} (MIN_SAMPLES={MIN_SAMPLES})")
    for w in chosen:
        print(f"     {w:10s} {counts[w]}")

    if not chosen:
        print(">> [경고] 조건을 만족하는 단어가 없습니다. MIN_SAMPLES를 낮추세요.")
        return

    # 선택된 단어에 해당하는 샘플만 추출
    mask = np.array([lbl in chosen_set for lbl in y.tolist()])
    Xf = X[mask]
    yf_words = y[mask]
    gf = g[mask]

    # 라벨 문자열 -> 숫자 인덱스
    label2idx = {w: i for i, w in enumerate(sorted(chosen))}
    yf = np.array([label2idx[w] for w in yf_words.tolist()], dtype=np.int64)

    # 저장
    np.save(os.path.join(OUT_DIR, "X.npy"), Xf.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "y.npy"), yf)
    np.save(os.path.join(OUT_DIR, "groups.npy"), gf)
    with open(os.path.join(OUT_DIR, "label2idx.json"), "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    print(">> 최종 학습셋 저장:", Xf.shape, " 클래스 수:", len(label2idx))
    print(">> 저장 위치 ->", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()

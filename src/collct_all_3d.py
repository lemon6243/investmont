# collect_all.py
# 목적: 압축을 하나씩 풀 때마다, 그 안의 "모든 단어" 클립을 누적 저장.
#       (특정 단어만 고르지 않음 -> 버리는 것 없이 전부 자산화)
# 사용법: 압축 하나 풀 때마다 실행 -> "이번에 추가된 샘플" 확인 -> 원본 폴더 삭제 -> 반복
# 주의: dataset_all/ 폴더는 절대 지우지 말 것 (여기에 데이터가 쌓임)

import os
import json
import numpy as np
from collections import Counter
from load_dataset_3d import (
    scan_video_dirs,
    find_morpheme_files,
    load_sequence,
    SEQ_LEN,
    FEATURE_DIM
)

# ------------------------------------------------------------
# 설정 (경로는 본인 환경에 맞게)
# ------------------------------------------------------------
MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"
KEYPOINT = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_keypoint"

ANGLES = None       # None=전 각도, ['F']=정면만
OUT_DIR = "dataset_all"   # 기존 dataset과 분리 (모든 단어 누적용)

PROCESSED_PATH = os.path.join(OUT_DIR, "processed_clips.json")
X_PATH = os.path.join(OUT_DIR, "X_all_3d.npy")
Y_PATH = os.path.join(OUT_DIR, "y_all.npy")      # 라벨을 "단어 문자열"로 저장
G_PATH = os.path.join(OUT_DIR, "groups_all.npy")


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def scan_morpheme_meta(morpheme_files, video_map):
    """morpheme을 읽어 (name, label, start, end) 목록 반환 (단어 필터 없음)"""
    meta = []
    for i, mf in enumerate(morpheme_files):
        if i % 40000 == 0:
            print(f"   ...morpheme {i}/{len(morpheme_files)}")
        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)
        if not m["data"]:
            continue
        name = m["metaData"]["name"].replace(".mp4", "")
        angle = name.split("_")[-1]
        if ANGLES and angle not in ANGLES:
            continue
        if name not in video_map:      # 지금 풀린 keypoint에 없으면 스킵
            continue
        label = m["data"][0]["attributes"][0]["name"]
        start = m["data"][0]["start"]
        end = m["data"][0]["end"]
        meta.append((name, label, start, end))
    return meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(">> keypoint 클립 폴더 스캔 중...")
    video_map = scan_video_dirs(KEYPOINT)
    print(">> 현재 풀린 keypoint 클립 폴더 개수:", len(video_map))

    morpheme_files = find_morpheme_files(MORPHEME)
    print(">> morpheme JSON 개수:", len(morpheme_files))

    print(">> morpheme 메타 스캔 중...")
    meta = scan_morpheme_meta(morpheme_files, video_map)
    print(">> 현재 매칭되는 클립 수:", len(meta))

    # ----------------------------------------------------
    # 기존 누적 데이터 & 처리이력 불러오기
    # ----------------------------------------------------
    if os.path.exists(X_PATH):
        X_old = np.load(X_PATH)
        y_old = np.load(Y_PATH, allow_pickle=True)   # 문자열 배열이라 pickle 허용
        g_old = np.load(G_PATH, allow_pickle=True)
        print(">> 기존 누적 데이터:", X_old.shape)
    else:
        X_old = np.zeros((0, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        y_old = np.array([], dtype=object)
        g_old = np.array([], dtype=object)

    processed = set(load_json(PROCESSED_PATH, []))
    print(">> 이미 처리한 클립 수:", len(processed))

    # ----------------------------------------------------
    # 새 클립 로딩 (단어 필터 없음, 아직 처리 안 한 것 전부)
    # ----------------------------------------------------
    print(">> 새 클립 로딩 중...")
    Xn, yn, gn = [], [], []
    added = 0
    for name, label, start, end in meta:
        if name in processed:          # 이미 처리한 클립은 건너뜀
            continue
        video_dir = video_map[name]
        seq = load_sequence(video_dir, start, end)
        if seq is None:
            continue
        Xn.append(seq)
        yn.append(label)               # 단어 문자열 그대로 저장
        gn.append(name.rsplit("_", 1)[0])
        processed.add(name)
        added += 1
        if added % 200 == 0:
            print(f"   ...새로 로딩 {added}개")

    print(">> 이번에 추가된 샘플:", added)

    # ----------------------------------------------------
    # 합치고 저장
    # ----------------------------------------------------
    if added > 0:
        Xn = np.array(Xn, dtype=np.float32)
        yn = np.array(yn, dtype=object)
        gn = np.array(gn, dtype=object)
        X = np.concatenate([X_old, Xn], axis=0)
        y = np.concatenate([y_old, yn], axis=0)
        g = np.concatenate([g_old, gn], axis=0)
    else:
        X, y, g = X_old, y_old, g_old

    np.save(X_PATH, X)
    np.save(Y_PATH, y)     # 문자열 배열 저장
    np.save(G_PATH, g)
    save_json(PROCESSED_PATH, sorted(processed))

    print(">> 누적 데이터 총량:", X.shape, " 서로 다른 단어 수:", len(set(y.tolist())))
    print(">> 저장 완료 ->", os.path.abspath(OUT_DIR))

    # 단어별 샘플 수 요약 (상위 20개만 표시)
    per_word = Counter(y.tolist())
    top20 = per_word.most_common(20)
    print(">> 단어별 샘플 수 상위 20:")
    for w, c in top20:
        print(f"     {w:10s} {c}")


if __name__ == "__main__":
    main()

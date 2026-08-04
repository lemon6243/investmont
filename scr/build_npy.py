# build_npy.py  (누적/이어붙이기 버전)
# 사용법:
#   - 처음 실행: 상위 TOP_N 단어를 자동 선정해 target_words.json에 고정 + 데이터 누적
#   - 이후 실행(새 keypoint zip 풀고): 같은 단어 목록으로 새 클립만 추가 누적
# 특징: 이미 처리한 클립은 processed_clips.json에 기록 -> 중복 방지

import os
import json
import numpy as np
from collections import Counter
from load_dataset import (
    scan_video_dirs, find_morpheme_files, load_sequence,
    SEQ_LEN, FEATURE_DIM
)

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"
KEYPOINT = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_keypoint"

TOP_N = 10          # 첫 실행 때만 사용 (단어 목록 고정 후엔 무시됨)
ANGLES = None       # None=전 각도, ['F']=정면만
OUT_DIR = "dataset"

TARGET_PATH = os.path.join(OUT_DIR, "target_words.json")
PROCESSED_PATH = os.path.join(OUT_DIR, "processed_clips.json")
X_PATH = os.path.join(OUT_DIR, "X.npy")
Y_PATH = os.path.join(OUT_DIR, "y.npy")
G_PATH = os.path.join(OUT_DIR, "groups.npy")
L_PATH = os.path.join(OUT_DIR, "label2idx.json")


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def scan_morpheme_meta(morpheme_files, video_map):
    """morpheme을 읽어 (name, label, start, end) 목록과 단어별 개수 반환"""
    word_count = Counter()
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
        word_count[label] += 1
        meta.append((name, label, start, end))
    return meta, word_count


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(">> keypoint 클립 폴더 스캔 중...")
    video_map = scan_video_dirs(KEYPOINT)
    print(">> 현재 풀린 keypoint 클립 폴더 개수:", len(video_map))

    morpheme_files = find_morpheme_files(MORPHEME)
    print(">> morpheme JSON 개수:", len(morpheme_files))

    print(">> morpheme 메타 스캔 중...")
    meta, word_count = scan_morpheme_meta(morpheme_files, video_map)
    print(">> 현재 매칭되는 클립 수:", len(meta))

    # ----------------------------------------------------
    # 목표 단어 목록: 처음이면 선정+저장, 있으면 불러오기(고정)
    # ----------------------------------------------------
    target = load_json(TARGET_PATH, None)
    if target is None:
        target_words = [w for w, c in word_count.most_common(TOP_N)]
        label2idx = {w: i for i, w in enumerate(sorted(target_words))}
        save_json(TARGET_PATH, {"words": target_words, "label2idx": label2idx})
        save_json(L_PATH, label2idx)
        print(f">> [최초] 상위 {TOP_N}단어 고정:", target_words)
    else:
        target_words = target["words"]
        label2idx = target["label2idx"]
        print(f">> [기존] 고정된 {len(target_words)}단어 사용")

    top_set = set(target_words)

    # ----------------------------------------------------
    # 기존 누적 데이터 & 처리이력 불러오기
    # ----------------------------------------------------
    if os.path.exists(X_PATH):
        X_old = np.load(X_PATH)
        y_old = np.load(Y_PATH)
        g_old = np.load(G_PATH)
        print(">> 기존 누적 데이터:", X_old.shape)
    else:
        X_old = np.zeros((0, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        y_old = np.zeros((0,), dtype=np.int64)
        g_old = np.array([], dtype=object)

    processed = set(load_json(PROCESSED_PATH, []))
    print(">> 이미 처리한 클립 수:", len(processed))

    # ----------------------------------------------------
    # 새 클립만 로딩 (목표 단어 & 아직 처리 안 한 것)
    # ----------------------------------------------------
    print(">> 새 클립 로딩 중...")
    Xn, yn, gn = [], [], []
    added = 0
    for name, label, start, end in meta:
        if label not in top_set:
            continue
        if name in processed:          # 이미 처리한 클립은 건너뜀
            continue
        video_dir = video_map[name]
        seq = load_sequence(video_dir, start, end)
        if seq is None:
            continue
        Xn.append(seq)
        yn.append(label2idx[label])
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
        yn = np.array(yn, dtype=np.int64)
        gn = np.array(gn)
        X = np.concatenate([X_old, Xn], axis=0)
        y = np.concatenate([y_old, yn], axis=0)
        g = np.concatenate([g_old, gn], axis=0)
    else:
        X, y, g = X_old, y_old, g_old

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    np.save(G_PATH, g)
    save_json(PROCESSED_PATH, sorted(processed))

    print(">> 누적 데이터 총량:", X.shape, " 라벨수:", len(set(y.tolist())))
    print(">> 저장 완료 ->", os.path.abspath(OUT_DIR))

    # 단어별 샘플 수 요약
    idx2label = {v: k for k, v in label2idx.items()}
    per_word = Counter(idx2label[int(v)] for v in y)
    print(">> 단어별 샘플 수:", dict(per_word))


if __name__ == "__main__":
    main()

# collect_all_3d.py
# 목적: 압축을 하나씩 풀 때마다 그 안의 "모든 단어" 클립을 누적 저장.
#       인식용(X_all_3d)과 아바타용 원본3D(X_raw_3d)를 함께 쌓는다.
# 사용법: 압축 하나 풀 때마다 실행 -> "이번에 추가된 샘플" 확인 -> 원본 삭제 -> 반복
# 주의: dataset_all_3d/ 폴더는 절대 지우지 말 것 (여기에 데이터가 쌓임)

import os
import json
import shutil
import numpy as np
from collections import Counter
from load_dataset_3d import (
    scan_video_dirs,
    find_morpheme_files,
    load_sequence,
    SEQ_LEN,
    FEATURE_DIM,
    N_KEYPOINTS,
)

# ------------------------------------------------------------
# 설정 (경로는 본인 환경에 맞게)
# ------------------------------------------------------------
MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"
KEYPOINT = r"D:\수어 영상\1.Training\[라벨]13_real_word_keypoint"

ANGLES = None                 # None=전 각도, ['F']=정면만
OUT_DIR = "dataset_all_3d"    # 2D와 완전히 분리된 폴더

X_PATH = os.path.join(OUT_DIR, "X_all_3d.npy")     # 인식용 (N,60,150)
XR_PATH = os.path.join(OUT_DIR, "X_raw_3d.npy")    # 아바타용 (N,60,50,3)
Y_PATH = os.path.join(OUT_DIR, "y_all_3d.npy")
G_PATH = os.path.join(OUT_DIR, "groups_all_3d.npy")
PROCESSED_PATH = os.path.join(OUT_DIR, "processed_clips_3d.json")


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json_atomic(path, obj):
    """임시 파일에 먼저 쓰고 성공하면 이름 교체 (저장 중 깨짐 방지)"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_npy_atomic(path, arr):
    """임시 .npy에 먼저 저장하고 성공하면 이름 교체.
    저장 도중 디스크가 꽉 차도 기존 path 파일은 그대로 보존됨."""
    tmp = path + ".tmp.npy"
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def check_free_space(target_dir, need_bytes, margin=1.5):
    """저장에 필요한 공간이 있는지 미리 확인. margin=여유 배수."""
    free = shutil.disk_usage(target_dir).free
    required = int(need_bytes * margin)
    print(f">> 디스크 여유: {free/1e9:.2f} GB, 필요(여유 포함): {required/1e9:.2f} GB")
    if free < required:
        raise OSError(
            f"디스크 공간 부족: {free/1e9:.2f} GB 남음, "
            f"약 {required/1e9:.2f} GB 필요. 공간을 확보한 뒤 다시 실행하세요."
        )


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
        if name not in video_map:
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
        Xr_old = np.load(XR_PATH)
        y_old = np.load(Y_PATH, allow_pickle=True)
        g_old = np.load(G_PATH, allow_pickle=True)
        print(">> 기존 누적 인식용:", X_old.shape, " 아바타용:", Xr_old.shape)

        # --- 무결성 체크: 네 배열 길이가 어긋나면 즉시 중단 ---
        if not (len(X_old) == len(Xr_old) == len(y_old) == len(g_old)):
            raise ValueError(
                f"기존 데이터 길이 불일치! "
                f"X={len(X_old)}, X_raw={len(Xr_old)}, y={len(y_old)}, g={len(g_old)}\n"
                f"-> 저장이 중간에 끊겼거나 파일이 손상된 상태입니다. "
                f"dataset_all_3d 의 파일들을 정리한 뒤 처음부터 다시 수집하세요."
            )
    else:
        X_old = np.zeros((0, SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        Xr_old = np.zeros((0, SEQ_LEN, N_KEYPOINTS, 3), dtype=np.float32)
        y_old = np.array([], dtype=object)
        g_old = np.array([], dtype=object)

    processed = set(load_json(PROCESSED_PATH, []))
    print(">> 이미 처리한 클립 수:", len(processed))

    # X가 없는데 processed 기록만 남아 있는 위험 상태 감지
    if len(X_old) == 0 and len(processed) > 0:
        raise ValueError(
            f"X_all_3d.npy 는 비었는데 processed 기록은 {len(processed)}개입니다.\n"
            f"-> 이전 저장이 깨진 상태입니다. dataset_all_3d 안의 "
            f"X_all_3d / X_raw_3d / y_all_3d / groups_all_3d / processed_clips_3d "
            f"파일을 모두 지우고 처음부터 다시 수집하세요."
        )

    # ----------------------------------------------------
    # 새 클립 로딩 (단어 필터 없음, 아직 처리 안 한 것 전부)
    # ----------------------------------------------------
    print(">> 새 클립 로딩 중...")
    Xn, Xr, yn, gn = [], [], [], []
    added = 0
    for name, label, start, end in meta:
        if name in processed:
            continue
        video_dir = video_map[name]
        x_norm, x_raw = load_sequence(video_dir, start, end)
        if x_norm is None:
            continue
        Xn.append(x_norm)
        Xr.append(x_raw)
        yn.append(label)
        gn.append(name.rsplit("_", 1)[0])
        processed.add(name)
        added += 1
        if added % 200 == 0:
            print(f"   ...새로 로딩 {added}개")

    print(">> 이번에 추가된 샘플:", added)

    # ----------------------------------------------------
    # 합치기
    # ----------------------------------------------------
    if added > 0:
        Xn = np.array(Xn, dtype=np.float32)
        Xr = np.array(Xr, dtype=np.float32)
        yn = np.array(yn, dtype=object)
        gn = np.array(gn, dtype=object)
        X = np.concatenate([X_old, Xn], axis=0)
        Xr_all = np.concatenate([Xr_old, Xr], axis=0)
        y = np.concatenate([y_old, yn], axis=0)
        g = np.concatenate([g_old, gn], axis=0)
    else:
        X, Xr_all, y, g = X_old, Xr_old, y_old, g_old

    # ----------------------------------------------------
    # 저장 (디스크 공간 확인 -> atomic 저장)
    # ----------------------------------------------------
    # 3D는 X_raw가 가장 크므로 두 배열 합산 크기로 여유 공간 확인
    check_free_space(OUT_DIR, X.nbytes + Xr_all.nbytes)

    save_npy_atomic(X_PATH, X)         # 인식용
    save_npy_atomic(XR_PATH, Xr_all)   # 아바타용 (가장 큼)
    save_npy_atomic(Y_PATH, y)
    save_npy_atomic(G_PATH, g)
    save_json_atomic(PROCESSED_PATH, sorted(processed))

    print(">> 누적 인식용:", X.shape, " 아바타용:", Xr_all.shape,
          " 서로 다른 단어 수:", len(set(y.tolist())))
    print(">> 저장 완료 ->", os.path.abspath(OUT_DIR))

    per_word = Counter(y.tolist())
    print(">> 단어별 샘플 수 상위 20:")
    for w, c in per_word.most_common(20):
        print(f"     {w:10s} {c}")


if __name__ == "__main__":
    main()

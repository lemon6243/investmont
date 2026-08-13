# load_dataset_3d.py
# 목적: AI Hub WORD 3D 데이터를 (인식용 시퀀스, 아바타용 원본 시퀀스, 라벨)로 로딩
# 흐름: morpheme JSON에서 라벨/구간 -> 영상 폴더 프레임 JSON 로딩
#       -> 공통포맷(원본 3D + conf) -> 신뢰도 보정
#       -> 인식용은 정규화, 아바타용은 원본 유지 -> 고정 길이로 맞춤
# 주의: 경로에 대괄호[]가 있어 glob 대신 os.walk 사용

import os
import json
import numpy as np
from keypoint_schema_3d import (
    openpose_to_common_3d,
    fill_low_conf_sequence,
    normalize_3d,
    flatten_3d,
    N_KEYPOINTS,
    FEATURE_DIM,
)

FPS = 30          # AI Hub 수어영상 기준 30fps
SEQ_LEN = 60      # 시퀀스 고정 길이(프레임)


def _read_frame_raw(json_path):
    """
    keypoint JSON 한 개(=한 프레임) -> (kp (50,3), conf (50,))  원본 3D
    """
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return openpose_to_common_3d(d["people"])


def _fix_length(seq, seq_len):
    """(T, ...) -> (seq_len, ...) 균등 샘플링 또는 뒤쪽 0패딩"""
    T = seq.shape[0]
    if T >= seq_len:
        idx = np.linspace(0, T - 1, seq_len).astype(int)
        return seq[idx]
    pad_shape = (seq_len - T,) + seq.shape[1:]
    pad = np.zeros(pad_shape, dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def load_sequence(video_dir, start_sec, end_sec):
    """
    한 영상 폴더의 프레임을 시간순으로 읽어 start~end 구간을 시퀀스로.
    반환:
      x_norm : (SEQ_LEN, FEATURE_DIM)  인식용 (정규화+평탄화)
      x_raw  : (SEQ_LEN, N_KEYPOINTS, 3)  아바타용 (원본 3D, 신뢰도 보정됨)
    실패 시 (None, None)
    """
    files = []
    for dp, dn, fns in os.walk(video_dir):
        for fn in fns:
            if fn.endswith("_keypoints.json"):
                files.append(os.path.join(dp, fn))
    files = sorted(files)
    if not files:
        return None, None

    s = int(start_sec * FPS)
    e = int(end_sec * FPS)
    s = max(0, s)
    e = min(len(files), e)
    if e <= s:
        s, e = 0, len(files)

    kp_list, conf_list = [], []
    for fp in files[s:e]:
        kp, conf = _read_frame_raw(fp)
        kp_list.append(kp)
        conf_list.append(conf)
    if not kp_list:
        return None, None

    seq_kp = np.stack(kp_list)      # (T, 50, 3)  원본
    seq_conf = np.stack(conf_list)  # (T, 50)

    # 1) 신뢰도 보정 (원본 3D 기준으로 먼저 수행)
    seq_kp = fill_low_conf_sequence(seq_kp, seq_conf)

    # 2) 아바타용: 원본 3D 그대로 (정규화 X) — 길이만 맞춤
    x_raw = _fix_length(seq_kp, SEQ_LEN)              # (SEQ_LEN, 50, 3)

    # 3) 인식용: 프레임별 정규화 후 평탄화 — 길이 맞춤
    norm_frames = np.stack([flatten_3d(normalize_3d(f)) for f in seq_kp])
    x_norm = _fix_length(norm_frames, SEQ_LEN)        # (SEQ_LEN, 150)

    return x_norm, x_raw


def scan_video_dirs(keypoint_root):
    """keypoint_root 아래 모든 영상 폴더를 한 번만 스캔 -> {폴더명: 경로}"""
    mapping = {}
    for dp, dn, fns in os.walk(keypoint_root):
        base = os.path.basename(dp)
        if base.startswith("NIA_SL_"):
            mapping[base] = dp
    return mapping


def find_morpheme_files(morpheme_root):
    """morpheme JSON 전체 경로 목록 (대괄호 경로 안전)"""
    files = []
    for dp, dn, fns in os.walk(morpheme_root):
        for fn in fns:
            if fn.endswith(".json"):
                files.append(os.path.join(dp, fn))
    return sorted(files)


def build_dataset(morpheme_root, keypoint_root, angles=None, limit=None,
                  verbose=True):
    """
    반환:
      X_norm  (N, SEQ_LEN, FEATURE_DIM)      인식용
      X_raw   (N, SEQ_LEN, N_KEYPOINTS, 3)   아바타용 원본 3D
      y       (N,)  라벨
      groups  (N,)  세션 구분(각도 뗀 이름)
    """
    if verbose:
        print(">> MORPHEME 경로 존재?:", os.path.exists(morpheme_root))
        print(">> KEYPOINT 경로 존재?:", os.path.exists(keypoint_root))
        print(">> keypoint 영상 폴더 스캔 중...")

    video_map = scan_video_dirs(keypoint_root)
    if verbose:
        print(">> keypoint 영상 폴더 개수:", len(video_map))

    morpheme_files = find_morpheme_files(morpheme_root)
    if verbose:
        print(">> morpheme JSON 개수:", len(morpheme_files))

    Xn, Xr, y, groups = [], [], [], []
    used, not_found = 0, 0

    for mf in morpheme_files:
        if limit and used >= limit:
            break
        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)

        name = m["metaData"]["name"].replace(".mp4", "")
        angle = name.split("_")[-1]
        if angles and angle not in angles:
            continue
        if not m["data"]:
            continue

        label = m["data"][0]["attributes"][0]["name"]
        start = m["data"][0]["start"]
        end = m["data"][0]["end"]

        video_dir = video_map.get(name)
        if video_dir is None:
            not_found += 1
            if verbose and not_found <= 5:
                print(">> keypoint 폴더 못찾음:", name)
            continue

        x_norm, x_raw = load_sequence(video_dir, start, end)
        if x_norm is None:
            continue

        Xn.append(x_norm)
        Xr.append(x_raw)
        y.append(label)
        groups.append(name.rsplit("_", 1)[0])
        used += 1

    if verbose:
        print(">> 최종 사용 샘플:", used, " / keypoint 못찾음:", not_found)

    return (np.array(Xn, dtype=np.float32),
            np.array(Xr, dtype=np.float32),
            np.array(y),
            np.array(groups))


if __name__ == "__main__":
    MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"
    KEYPOINT = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_keypoint"

    Xn, Xr, y, groups = build_dataset(MORPHEME, KEYPOINT, angles=None, limit=5)
    print("X_norm shape:", Xn.shape)   # 기대: (5, 60, 150)
    print("X_raw  shape:", Xr.shape)   # 기대: (5, 60, 50, 3)
    print("labels:", set(y))
    print("groups:", len(set(groups)))

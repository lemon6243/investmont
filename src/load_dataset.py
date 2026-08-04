# load_dataset.py
# 목적: AI Hub WORD 데이터를 (시퀀스, 라벨) 쌍으로 로딩
# 흐름: morpheme JSON에서 라벨/구간 읽기 -> 해당 영상 폴더의 프레임 JSON 로딩
#       -> start~end 구간만 잘라 공통포맷 변환+정규화 -> 고정길이로 맞춤
# 주의: 경로에 대괄호[]가 있어 glob 대신 os.walk 사용

import os
import json
import numpy as np
from keypoint_schema import openpose_to_common, normalize, flatten, FEATURE_DIM

FPS = 30            # AI Hub 수어영상 기준 30fps
SEQ_LEN = 60        # 시퀀스 고정 길이(프레임). 짧으면 패딩, 길면 균등 샘플링


def load_frame(json_path):
    """keypoint JSON 한 개(=한 프레임) -> 정규화된 (FEATURE_DIM,) 벡터"""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    kp = openpose_to_common(d["people"])
    kp = normalize(kp)
    return flatten(kp)


def load_sequence(video_dir, start_sec, end_sec):
    """
    한 영상 폴더 안의 프레임들을 시간순으로 읽어 start~end 구간만 시퀀스로.
    반환: (SEQ_LEN, FEATURE_DIM)
    """
    files = []
    for dp, dn, fns in os.walk(video_dir):
        for fn in fns:
            if fn.endswith("_keypoints.json"):
                files.append(os.path.join(dp, fn))
    files = sorted(files)
    if not files:
        return None

    s = int(start_sec * FPS)
    e = int(end_sec * FPS)
    s = max(0, s)
    e = min(len(files), e)
    if e <= s:
        s, e = 0, len(files)   # 구간이 이상하면 전체 사용

    seq = [load_frame(fp) for fp in files[s:e]]
    seq = np.stack(seq)        # (T, FEATURE_DIM)

    # 고정 길이로 맞추기
    T = seq.shape[0]
    if T >= SEQ_LEN:
        idx = np.linspace(0, T - 1, SEQ_LEN).astype(int)  # 균등 샘플링
        seq = seq[idx]
    else:
        pad = np.zeros((SEQ_LEN - T, FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)          # 뒤를 0으로 패딩
    return seq


def scan_video_dirs(keypoint_root):
    """
    keypoint_root 아래의 모든 '영상 폴더'를 한 번만 스캔해서
    {폴더이름: 전체경로} 사전으로 반환. (매번 재탐색 방지 -> 속도 대폭 향상)
    영상 폴더 = NIA_SL_WORD...._D 처럼 keypoint json을 직접 담은 폴더
    """
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
    morpheme_root: morpheme JSON들이 있는 최상위 폴더
    keypoint_root: 영상별 하위폴더(NIA_SL_WORD..._D 등)가 있는 최상위 폴더
    angles: 사용할 각도 리스트, 예 ['F'] 또는 None(전체)
    limit: 사용할 morpheme 최대 개수(빠른 검증용)
    반환: X (N, SEQ_LEN, FEATURE_DIM), y (N,) 라벨, groups (N,) 세션구분
    """
    if verbose:
        print(">> MORPHEME 경로 존재?:", os.path.exists(morpheme_root))
        print(">> KEYPOINT 경로 존재?:", os.path.exists(keypoint_root))
        print(">> keypoint 영상 폴더 스캔 중... (한 번만)")

    video_map = scan_video_dirs(keypoint_root)
    if verbose:
        print(">> keypoint 영상 폴더 개수:", len(video_map))

    morpheme_files = find_morpheme_files(morpheme_root)
    if verbose:
        print(">> morpheme JSON 개수:", len(morpheme_files))

    X, y, groups = [], [], []
    used = 0
    not_found = 0

    for mf in morpheme_files:
        if limit and used >= limit:
            break

        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)

        name = m["metaData"]["name"].replace(".mp4", "")  # 폴더명과 동일
        angle = name.split("_")[-1]                        # D/F/L/R/U
        if angles and angle not in angles:
            continue
        if not m["data"]:
            continue

        label = m["data"][0]["attributes"][0]["name"]      # 예: "고민"
        start = m["data"][0]["start"]
        end = m["data"][0]["end"]

        video_dir = video_map.get(name)
        if video_dir is None:
            not_found += 1
            if verbose and not_found <= 5:
                print(">> keypoint 폴더 못찾음:", name)
            continue

        seq = load_sequence(video_dir, start, end)
        if seq is None:
            continue

        X.append(seq)
        y.append(label)
        groups.append(name.rsplit("_", 1)[0])  # 각도 뗀 이름 = 세션 그룹키
        used += 1

    if verbose:
        print(">> 최종 사용 샘플:", used, " / keypoint 못찾음:", not_found)

    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    # 경로는 본인 환경에 맞게 수정 (대괄호 있어도 OK)
    MORPHEME = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_morpheme"
    KEYPOINT = r"C:\Users\LG\Downloads\수어 영상\1.Training\[라벨]01_real_word_keypoint"

    # 우선 5개만, 전체 각도로 빠르게 동작 확인
    X, y, groups = build_dataset(MORPHEME, KEYPOINT, angles=None, limit=500)
    print("X shape:", X.shape)      # 기대: (5, 60, 100)
    print("labels:", set(y))
    print("groups:", len(set(groups)))

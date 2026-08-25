# send_mh_live.py
# 지금 있는 단어 3D를 UE로 30fps 송출
#   1) dataset_all_3d/X_raw_3d.npy  (있으면 이걸 우선)
#   2) 없으면 data/*.json 한 클립
#
#   python send_mh_live.py
#   python send_mh_live.py --word 머리
#   python send_mh_live.py --idx 3
#   python send_mh_live.py --file          # UDP 대신 파일 (플러그인 없을 때)
#   python send_mh_live.py --arms-only     # 팔 6본만 (첫 테스트 추천)

import os
import json
import time
import socket
import argparse
import numpy as np

from keypoint_schema_3d import to_unreal, POSE_ORDER, N_HAND
from pos_to_rot import (
    BONES,
    positions_to_local_quats,
    trim_seq,
    pick_best_clip,
    frame_is_empty,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA_DIR = os.path.join(HERE, "dataset_all_3d")
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(ROOT, "dataset_all_3d")

XR_PATH = os.path.join(DATA_DIR, "X_raw_3d.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all_3d.npy")
JSON_DIR = os.path.join(ROOT, "data")
if not os.path.isdir(JSON_DIR):
    JSON_DIR = os.path.join(HERE, "data")

UE_IP, UE_PORT = "127.0.0.1", 7755
FPS = 30
LIVE_FILE = os.path.join(ROOT, "Saved", "sign_live.json")

ARM_BONES = {
    "upperarm_l", "lowerarm_l", "hand_l",
    "upperarm_r", "lowerarm_r", "hand_r",
}


def find_npy():
    for p in (XR_PATH, os.path.join(HERE, "dataset_all_3d", "X_raw_3d.npy")):
        if os.path.exists(p):
            return p
    return None


def load_from_npy(word=None, idx=None):
    xr_path = find_npy()
    if xr_path is None:
        return None, None
    y_path = os.path.join(os.path.dirname(xr_path), "y_all_3d.npy")
    Xr = np.load(xr_path)
    y = np.load(y_path, allow_pickle=True)
    print(f">> npy: {xr_path}  {Xr.shape}  단어 {len(set(map(str, y)))}개")

    if word:
        hits = [i for i, lab in enumerate(y) if str(lab) == word]
        if not hits:
            print(">> 그 단어 없음. 있는 것 일부:", sorted(set(map(str, y)))[:30])
            raise SystemExit(1)
        clips = [Xr[i] for i in hits]
        seq = trim_seq(pick_best_clip(clips))
        return seq, word

    i = 0 if idx is None else int(idx)
    i = max(0, min(i, len(Xr) - 1))
    return trim_seq(Xr[i]), str(y[i])


def load_from_json_folder():
    from keypoint_schema_3d import openpose_to_common_3d
    files = sorted(
        os.path.join(JSON_DIR, f)
        for f in os.listdir(JSON_DIR)
        if f.endswith("_keypoints.json")
    )
    if not files:
        return None, None
    frames = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        kp, _ = openpose_to_common_3d(d["people"])
        frames.append(kp)
    seq = np.stack(frames)
    print(f">> json 클립: {len(files)}프레임  ({files[0]})")
    return trim_seq(seq), "json_clip"


def seq_to_ue(seq):
    """OpenPose 3D (T,50,3) -> UE cm (T,50,3)"""
    return np.stack([to_unreal(f, scale=100.0, recenter=True) for f in seq])


def pack_frame(ue_frame, bones, t, word, arms_only):
    pts = ue_frame.reshape(-1).astype(float).tolist()
    q = {}
    for name, arr in bones.items():
        if arms_only and name not in ARM_BONES:
            continue
        q[name] = [float(x) for x in arr[t]]
    return {
        "type": "frame",
        "word": word,
        "i": int(t),
        "n": int(ue_frame.shape[0]) if False else None,  # 아래에서 덮음
        "space": "bone",      # ← 추가
        "mode": "replace",    # ← 추가
        "pts": pts,
        "q": q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="", help="재생할 단어. 없으면 첫 클립")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--file", action="store_true", help="UDP 대신 Saved/sign_live.json")
    ap.add_argument("--arms-only", action="store_true", default=True)
    ap.add_argument("--all-bones", action="store_true")
    ap.add_argument("--fps", type=float, default=FPS)
    args = ap.parse_args()
    arms_only = not args.all_bones

    seq, word = load_from_npy(args.word or None, args.idx)
    if seq is None:
        seq, word = load_from_json_folder()
    if seq is None:
        raise SystemExit(">> X_raw_3d.npy 도 data/*.json 도 없음")

    # 빈 패딩 프레임 제거 후 UE 좌표로 변환한 뒤 회전 계산
    # (축이 UE와 같아야 팔이 등 뒤로 안 접힘)
    ue = seq_to_ue(seq)
    bones, root = positions_to_local_quats(ue)
    T = ue.shape[0]
    print(f">> 재생: '{word}'  T={T}  bones={len(bones)}  arms_only={arms_only}")
    print(f">> 손목 UE 좌표 예시 f0: L={ue[0, POSE_ORDER.index('l_wrist')]}  "
          f"R={ue[0, POSE_ORDER.index('r_wrist')]}")

    sock = None
    if args.file:
        os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)
        print(">> 파일 모드:", LIVE_FILE)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f">> UDP {UE_IP}:{UE_PORT}")

    print(">> UE Play 후 이 창을 켜 두세요. 종료: Ctrl+C")
    dt = 1.0 / max(args.fps, 1.0)

    try:
        while True:
            for t in range(T):
                msg = pack_frame(ue[t], bones, t, word, arms_only)
                msg["n"] = T   # 총 프레임 수 덮어쓰기

                raw = json.dumps(msg, ensure_ascii=False).encode("utf-8")
                if sock is not None:
                    sock.sendto(raw, (UE_IP, UE_PORT))
                else:
                    tmp = LIVE_FILE + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(raw.decode("utf-8"))
                    os.replace(tmp, LIVE_FILE)
                time.sleep(dt)
            time.sleep(0.35)
    except KeyboardInterrupt:
        print("\n>> 중지")



if __name__ == "__main__":
    main()

"inbox/        ← 여기다 mp4 넣으면 됨
processing/
done/
failed/
out/pos/      (T,50,3) npy
out/rots/     회전 npz"


# watch_extract.py
# inbox 에 영상 넣으면 자동 추출. 이미 처리한 파일은 건너뜀.
# 중단했다가 다시 켜도 이어서 함.

import os
import time
import json
import shutil
import hashlib
import traceback
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from keypoint_schema import mediapipe_to_common, N_KEYPOINTS
from pos_to_rot import positions_to_local_quats, trim_seq

ROOT = Path(__file__).resolve().parent / "mocap_inbox"
INBOX = ROOT / "inbox"
PROC = ROOT / "processing"
DONE = ROOT / "done"
FAIL = ROOT / "failed"
OUT_POS = ROOT / "out" / "pos"
OUT_ROT = ROOT / "out" / "rots"
STATE = ROOT / "state.json"

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
POLL_SEC = 3


def setup():
    for p in (INBOX, PROC, DONE, FAIL, OUT_POS, OUT_ROT):
        p.mkdir(parents=True, exist_ok=True)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": {}, "failed": {}}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def file_id(path: Path):
    h = hashlib.sha1()
    h.update(str(path.stat().st_size).encode())
    h.update(path.name.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def extract_positions(video_path: Path):
    """MediaPipe Holistic -> (T,50,3). z는 상대값(2.5D)."""
    mp_holistic = mp.solutions.holistic
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없음")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    i = 0
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            kp2 = mediapipe_to_common(res, w, h, swap_lr=False)  # (50,2)
            z = np.zeros((N_KEYPOINTS, 1), np.float32)

            # pose world z 가 있으면 상체 8점에만 사용
            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                # nose, (neck=어깨중점), r_sh, r_el, r_wr, l_sh, l_el, l_wr
                mp_idx = [0, None, 12, 14, 16, 11, 13, 15]
                for ji, mi in enumerate(mp_idx):
                    if mi is None:
                        z[ji, 0] = (lm[11].z + lm[12].z) * 0.5
                    else:
                        z[ji, 0] = lm[mi].z

            kp = np.concatenate([kp2, z], axis=1).astype(np.float32)
            frames.append(kp)
            i += 1
            if i % 60 == 0:
                print(f"      frame {i}/{total or '?'}")

    cap.release()
    if not frames:
        raise RuntimeError("프레임 0개")
    return np.stack(frames)


def process_one(src: Path, st: dict):
    fid = file_id(src)
    stem = src.stem
    print(f"\n▶ {src.name}")

    work = PROC / src.name
    if src.parent != PROC:
        shutil.move(str(src), work)
    else:
        work = src

    try:
        pos = extract_positions(work)
        pos = trim_seq(pos)
        pos_path = OUT_POS / f"{stem}.npy"
        np.save(pos_path, pos)

        bones, root = positions_to_local_quats(pos)
        rot_path = OUT_ROT / f"{stem}.npz"
        np.savez_compressed(rot_path, root=root, **bones)

        shutil.move(str(work), DONE / work.name)
        st["done"][fid] = {
            "name": work.name,
            "frames": int(pos.shape[0]),
            "pos": str(pos_path),
            "rot": str(rot_path),
        }
        save_state(st)
        print(f"   OK  T={pos.shape[0]}  -> {rot_path.name}")
    except Exception:
        err = traceback.format_exc()
        print(err)
        (FAIL / f"{work.name}.error.txt").write_text(err, encoding="utf-8")
        try:
            shutil.move(str(work), FAIL / work.name)
        except OSError:
            pass
        st["failed"][fid] = work.name
        save_state(st)


def pending():
    files = [p for p in INBOX.iterdir() if p.suffix.lower() in VIDEO_EXT]
    leftover = [p for p in PROC.iterdir() if p.suffix.lower() in VIDEO_EXT]
    return leftover + sorted(files, key=lambda p: p.stat().st_mtime)


def main():
    setup()
    st = load_state()
    print("=" * 50)
    print("  inbox 감시 시작")
    print(f"  넣는 곳: {INBOX}")
    print("  종료: Ctrl+C  (다시 켜면 이어서 함)")
    print("=" * 50)

    try:
        while True:
            jobs = pending()
            if not jobs:
                time.sleep(POLL_SEC)
                continue
            for src in jobs:
                fid = file_id(src)
                if fid in st["done"]:
                    print(f"skip (이미 완료): {src.name}")
                    shutil.move(str(src), DONE / src.name)
                    continue
                process_one(src, st)
    except KeyboardInterrupt:
        print("\n>> 중단. state 저장됨. 다시 실행하면 이어서 함.")


if __name__ == "__main__":
    main()

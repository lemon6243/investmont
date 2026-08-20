# preview3d.py - 원본 3D 위치(X_raw_3d)를 직접 그리는 상반신 스켈레톤 뷰어
#   FK 근사 없이 실제 데이터 위치를 그대로 렌더링한다. 손가락까지 전부 표시.
#
# 사용법:
#   python preview3d.py 가다
#   python preview3d.py            # 첫 단어
#
# 조작: [스페이스] 재생/정지  [q/ESC] 종료
import os
import sys
import numpy as np
import cv2
from collections import defaultdict

from keypoint_schema_3d import POSE_ORDER, N_POSE, N_HAND
from pos_to_rot import (
    JOINTS, J, HAND_CHAINS, pick_best_clip, trim_seq, frame_is_empty
)

DATA_DIR = "dataset_all_3d"
XR_PATH = os.path.join(DATA_DIR, "X_raw_3d.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all_3d.npy")

# ---- 연결 구조 ----
# 상반신 몸통/팔 (pose 조인트 이름끼리)
BODY_LINKS = [
    ("neck", "nose"),
    ("neck", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("neck", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
]
# 손목 -> 손 루트 연결 (r_wrist -> rhand_0)
WRIST_TO_HAND = [
    ("r_wrist", "rhand_0"),
    ("l_wrist", "lhand_0"),
]
# 손가락 링크: hand_start(0) 기준 (a,b) 쌍. 5손가락 x 각 마디
FINGER_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
]

# ---- 화면 ----
W, H = 720, 720
BG_TOP = (40, 20, 10)
BG_BOT = (15, 8, 4)
NEON_ARM = (255, 220, 60)    # 청록 (BGR)
NEON_HAND = (255, 150, 220)  # 보라
NEON_BODY = (200, 180, 80)
HEAD_COL = (255, 200, 120)
JOINT_COL = (255, 255, 255)


def load_clip(word):
    Xr = np.load(XR_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    by_word = defaultdict(list)
    for seq, label in zip(Xr, y):
        by_word[str(label)].append(seq)
    if word not in by_word:
        return None
    return trim_seq(pick_best_clip(by_word[word]))


def make_bg():
    img = np.zeros((H, W, 3), np.uint8)
    for yy in range(H):
        a = yy / H
        img[yy, :] = [int(BG_TOP[c]*(1-a) + BG_BOT[c]*a) for c in range(3)]
    return img


def draw_glow_line(img, p1, p2, color, thick=3):
    cv2.line(img, p1, p2, tuple(int(c*0.35) for c in color), thick+6, cv2.LINE_AA)
    cv2.line(img, p1, p2, tuple(int(c*0.7) for c in color), thick+2, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)


def compute_view(seq):
    """전체 프레임의 x,y 범위로 화면 스케일/중심 자동 계산."""
    pts = seq.reshape(-1, 3)
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts = pts[valid]
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    span = max(xmax - xmin, ymax - ymin) + 1e-6
    scale = (H * 0.7) / span
    cx = W / 2 + (xmin + xmax) / 2 * scale     # x 반전에 맞춰 부호 +
    cy = H / 2 - (ymin + ymax) / 2 * scale     # y 그대로 더하기에 맞춤
    return scale, cx, cy



# 뷰 회전 각도 (라디안). 0.6 = 약 34도 비스듬히
VIEW_ANGLE = 0.6
_COS = np.cos(VIEW_ANGLE)
_SIN = np.sin(VIEW_ANGLE)


def proj(p, scale, cx, cy):
    # x축 기준으로 z를 섞어 3/4 뷰 (깊이감)
    #   화면가로 = -x, 깊이(z)를 가로에 일부 투영
    z_rel = p[2] - 2.3   # z 중심 대략 보정 (어깨 z≈2.3)
    x = int(cx - (p[0] * _COS + z_rel * _SIN) * scale)
    y = int(cy + p[1] * scale)
    return x, y





def valid(p):
    return np.linalg.norm(p) > 1e-6


def draw_frame(kp, scale, cx, cy):
    img = make_bg()

    def P(name):
        return kp[J[name]]

    # 몸통/팔
    for a, b in BODY_LINKS:
        pa, pb = P(a), P(b)
        if valid(pa) and valid(pb):
            draw_glow_line(img, proj(pa, scale, cx, cy),
                           proj(pb, scale, cx, cy), NEON_ARM, 4)

    # 손목 -> 손루트
    for a, b in WRIST_TO_HAND:
        pa, pb = P(a), P(b)
        if valid(pa) and valid(pb):
            draw_glow_line(img, proj(pa, scale, cx, cy),
                           proj(pb, scale, cx, cy), NEON_HAND, 2)

    # 손가락 (양손)
    for hand_prefix in ("rhand", "lhand"):
        base = J[f"{hand_prefix}_0"]
        for a, b in FINGER_PAIRS:
            pa, pb = kp[base + a], kp[base + b]
            if valid(pa) and valid(pb):
                draw_glow_line(img, proj(pa, scale, cx, cy),
                               proj(pb, scale, cx, cy), NEON_HAND, 2)

    # 머리 (nose 위치에 큰 원 + 글로우)
    nose = P("nose")
    if valid(nose):
        hx, hy = proj(nose, scale, cx, cy)
        cv2.circle(img, (hx, hy), 26, tuple(int(c*0.3) for c in HEAD_COL), -1, cv2.LINE_AA)
        cv2.circle(img, (hx, hy), 20, HEAD_COL, 2, cv2.LINE_AA)

    # 조인트 점 (몸통/팔만 크게, 손가락은 작게)
    for name in ["neck", "r_shoulder", "l_shoulder",
                 "r_elbow", "l_elbow", "r_wrist", "l_wrist"]:
        p = P(name)
        if valid(p):
            x, y = proj(p, scale, cx, cy)
            cv2.circle(img, (x, y), 5, JOINT_COL, -1, cv2.LINE_AA)
    for hand_prefix in ("rhand", "lhand"):
        base = J[f"{hand_prefix}_0"]
        for i in range(N_HAND):
            p = kp[base + i]
            if valid(p):
                x, y = proj(p, scale, cx, cy)
                cv2.circle(img, (x, y), 2, JOINT_COL, -1, cv2.LINE_AA)

    return img


def main():
    args = sys.argv[1:]
    word = args[0] if args else None

    if word is None:
        # 첫 단어
        y = np.load(Y_PATH, allow_pickle=True)
        word = sorted(set(str(w) for w in y))[0]

    seq = load_clip(word)
    if seq is None:
        print(f"단어 없음: {word}")
        return

    T = len(seq)
    scale, cx, cy = compute_view(seq)
    print(f"[{word}] {T}프레임  |  스페이스=재생/정지  q=종료")

    t = 0
    playing = True
    while True:
        img = draw_frame(seq[t], scale, cx, cy)
        cv2.putText(img, f"{word}  f{t}/{T-1}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("preview3d", img)

        key = cv2.waitKey(40) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            playing = not playing
        if playing:
            t = (t + 1) % T

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

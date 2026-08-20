# -*- coding: utf-8 -*-
"""
avatar_preview.py
저장된 키포인트 시퀀스를 "홀로그램 네온 뼈대"로 재생.
데이터/좌표는 그대로, 그리는 방식만 홀로그램형으로 교체.

조작:
  space : 일시정지/재생
  n     : 다음 단어(샘플)
  q     : 종료
"""

import os
import numpy as np
import cv2

from keypoint_schema import POSE_ORDER, N_POSE, N_HAND, FEATURE_DIM

# ------------------------------------------------------------
# 0) 설정
# ------------------------------------------------------------
DATA_DIR = "dataset_all"
X_PATH = os.path.join(DATA_DIR, "X_all.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all.npy")

CANVAS = 600
FPS = 20

# ===== 홀로그램/네온 스타일 색상 (BGR) =====
BG_TOP = (30, 18, 12)       # 배경 그라데이션 위(짙은 남색톤)
BG_BOT = (12, 8, 6)         # 배경 아래(거의 검정)
NEON_ARM = (255, 180, 60)   # 팔: 파란 네온
NEON_HAND = (180, 220, 40)  # 손: 청록 네온
NEON_BODY = (200, 120, 40)  # 몸통: 진한 파랑
JOINT = (255, 230, 180)     # 관절 노드: 밝은 하늘빛

ARM_TH = 5
FINGER_TH = 2
JOINT_R = 6

# ------------------------------------------------------------
# 1) 관절 인덱스 / 뼈대
# ------------------------------------------------------------
def _p(name):
    return POSE_ORDER.index(name)

I_NOSE, I_NECK = _p("nose"), _p("neck")
I_RSH, I_REL, I_RWR = _p("r_shoulder"), _p("r_elbow"), _p("r_wrist")
I_LSH, I_LEL, I_LWR = _p("l_shoulder"), _p("l_elbow"), _p("l_wrist")

ARM_BONES = [
    (I_RSH, I_REL), (I_REL, I_RWR),   # 오른팔
    (I_LSH, I_LEL), (I_LEL, I_LWR),   # 왼팔
]

# 손 21점 표준 연결
HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

LHAND_START = N_POSE
RHAND_START = N_POSE + N_HAND

# ------------------------------------------------------------
# 2) 좌표 변환
# ------------------------------------------------------------
def to_pixel(kp_xy):
    scale = CANVAS * 0.16
    cx, cy = CANVAS // 2, int(CANVAS * 0.52)
    return int(cx + kp_xy[0] * scale), int(cy + kp_xy[1] * scale)


def is_valid(kp_xy):
    return not (abs(kp_xy[0]) < 1e-6 and abs(kp_xy[1]) < 1e-6)


# ------------------------------------------------------------
# 3) 홀로그램 파츠 그리기 헬퍼
# ------------------------------------------------------------
def _bg():
    """세로 그라데이션 배경 (어두운 홀로그램 무대)"""
    img = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    for y in range(CANVAS):
        t = y / CANVAS
        img[y, :] = [int(BG_BOT[i] * t + BG_TOP[i] * (1 - t)) for i in range(3)]
    return img


def draw_glow_line(img, p1, p2, color, thickness):
    """네온 선: 굵은 반투명 glow + 가는 밝은 코어"""
    glow = img.copy()
    cv2.line(glow, p1, p2, color, thickness + 10, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, img, 0.65, 0, img)
    cv2.line(img, p1, p2, color, thickness + 3, cv2.LINE_AA)
    cv2.line(img, p1, p2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)


def draw_joint(img, p, color=JOINT, r=JOINT_R):
    """빛나는 관절 노드"""
    glow = img.copy()
    cv2.circle(glow, p, r + 6, color, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.4, img, 0.6, 0, img)
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p, max(1, r - 3), (255, 255, 255), -1, cv2.LINE_AA)


def draw_hand(img, kp, start):
    """손: 네온 선 + 관절 노드"""
    pts = [kp[start + i] for i in range(N_HAND)]
    if not all(is_valid(pt) for pt in [pts[0], pts[5], pts[17]]):
        return  # 손 감지 안 됨
    for a, b in HAND_BONES:
        if is_valid(pts[a]) and is_valid(pts[b]):
            draw_glow_line(img, to_pixel(pts[a]), to_pixel(pts[b]),
                           NEON_HAND, FINGER_TH)
    for i in range(N_HAND):
        if is_valid(pts[i]):
            draw_joint(img, to_pixel(pts[i]), NEON_HAND, r=3)


def draw_face(img, nose_px, neck_px):
    """목 위에 반투명 홀로그램 헤드 + 스캔라인"""
    dx = nose_px[0] - neck_px[0]
    dy = nose_px[1] - neck_px[1]
    r = max(24, int((dx**2 + dy**2) ** 0.5 * 0.9))
    fx, fy = nose_px[0], nose_px[1] - r // 3

    glow = img.copy()
    cv2.circle(glow, (fx, fy), r, NEON_ARM, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.25, img, 0.75, 0, img)
    cv2.circle(img, (fx, fy), r, NEON_ARM, 2, cv2.LINE_AA)
    # 스캔라인 느낌 (가로줄 두 개)
    cv2.line(img, (fx - r, fy - r // 3), (fx + r, fy - r // 3),
             NEON_ARM, 1, cv2.LINE_AA)
    cv2.line(img, (fx - r, fy + r // 4), (fx + r, fy + r // 4),
             NEON_ARM, 1, cv2.LINE_AA)


# ------------------------------------------------------------
# 4) 한 프레임 -> 홀로그램 이미지
# ------------------------------------------------------------
def draw_frame(frame_vec):
    kp = frame_vec.reshape(-1, 2)   # (50,2)
    img = _bg()

    r_sh, l_sh = kp[I_RSH], kp[I_LSH]
    neck, nose = kp[I_NECK], kp[I_NOSE]

    # --- 몸통: 어깨선 + 목 (네온 선) ---
    if is_valid(r_sh) and is_valid(l_sh) and is_valid(neck):
        draw_glow_line(img, to_pixel(r_sh), to_pixel(l_sh), NEON_BODY, ARM_TH)
        draw_glow_line(img, to_pixel(neck), to_pixel(nose), NEON_BODY, ARM_TH - 1)

    # --- 팔 (네온 선) ---
    for a, b in ARM_BONES:
        if is_valid(kp[a]) and is_valid(kp[b]):
            draw_glow_line(img, to_pixel(kp[a]), to_pixel(kp[b]), NEON_ARM, ARM_TH)

    # --- 얼굴 ---
    if is_valid(nose) and is_valid(neck):
        draw_face(img, to_pixel(nose), to_pixel(neck))

    # --- 팔 관절 노드 ---
    for idx in [I_RSH, I_REL, I_RWR, I_LSH, I_LEL, I_LWR, I_NECK]:
        if is_valid(kp[idx]):
            draw_joint(img, to_pixel(kp[idx]))

    # --- 손 (팔보다 나중에 그려서 위에 오도록) ---
    draw_hand(img, kp, LHAND_START)
    draw_hand(img, kp, RHAND_START)

    return img


# ------------------------------------------------------------
# 5) 메인 (단독 실행 테스트용)
# ------------------------------------------------------------
def main():
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"{X_PATH} 가 없습니다. 먼저 collect_all.py로 데이터를 만드세요.")

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    print(f">> 로드: {X.shape}, 단어 수 {len(set(y.tolist()))}")
    print("   space=일시정지  n=다음단어  q=종료")

    idx, f, paused = 0, 0, False
    delay = int(1000 / FPS)

    while True:
        seq = X[idx]
        word = str(y[idx])
        img = draw_frame(seq[f])

        cv2.putText(img, f"WORD: {word}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {f+1}/{len(seq)}  sample {idx+1}/{len(X)}",
                    (15, CANVAS - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 160), 1, cv2.LINE_AA)

        cv2.imshow("SignBridge - Hologram Avatar Preview", img)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            idx = (idx + 1) % len(X); f = 0; continue

        if not paused:
            f += 1
            if f >= len(seq):
                f = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

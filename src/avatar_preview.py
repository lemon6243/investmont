# -*- coding: utf-8 -*-
"""
avatar_preview.py
모듈 9(2단계): 저장된 키포인트 시퀀스를 "살 붙인 2D 캐릭터"로 재생.
데이터/좌표는 그대로, 그리는 방식만 캐릭터형으로 교체.

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

# 캐릭터 색상 (BGR)
SKIN   = (180, 210, 245)   # 살구색 피부
BODY   = (200, 150, 90)    # 몸통/옷 (파란톤)
OUTLINE= (60, 60, 60)      # 외곽선
FACE_LN= (70, 70, 70)      # 얼굴 이목구비

# 두께(정규화 스케일 기준 - to_pixel의 scale과 함께 조정)
ARM_TH   = 16   # 팔 두께(px)
FINGER_TH= 7    # 손가락 두께(px)

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
# 손바닥 외곽(살 채우기용): 손목-엄지뿌리-검지뿌리-...-소지뿌리
PALM_POLY = [0, 1, 5, 9, 13, 17]

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
# 3) 캐릭터 파츠 그리기 헬퍼
# ------------------------------------------------------------
def draw_capsule(img, p1, p2, thickness, fill, outline):
    """두 점을 잇는 '통통한 캡슐'(팔다리) : 외곽선 있는 두꺼운 선 + 끝 원"""
    # 외곽선(약간 더 두껍게)
    cv2.line(img, p1, p2, outline, thickness + 4, cv2.LINE_AA)
    cv2.circle(img, p1, (thickness + 4)//2, outline, -1, cv2.LINE_AA)
    cv2.circle(img, p2, (thickness + 4)//2, outline, -1, cv2.LINE_AA)
    # 내부 채움
    cv2.line(img, p1, p2, fill, thickness, cv2.LINE_AA)
    cv2.circle(img, p1, thickness//2, fill, -1, cv2.LINE_AA)
    cv2.circle(img, p2, thickness//2, fill, -1, cv2.LINE_AA)


def draw_hand(img, kp, start):
    """손: 손바닥을 살로 채우고 손가락을 캡슐로"""
    pts = [kp[start + i] for i in range(N_HAND)]
    if not all(is_valid(pt) for pt in [pts[0], pts[5], pts[17]]):
        return  # 손 감지 안 됨

    # 손바닥 채우기
    poly = []
    for i in PALM_POLY:
        if is_valid(pts[i]):
            poly.append(to_pixel(pts[i]))
    if len(poly) >= 3:
        poly_np = np.array(poly, dtype=np.int32)
        cv2.fillPoly(img, [poly_np], SKIN, cv2.LINE_AA)
        cv2.polylines(img, [poly_np], True, OUTLINE, 2, cv2.LINE_AA)

    # 손가락(캡슐)
    for a, b in HAND_BONES:
        if is_valid(pts[a]) and is_valid(pts[b]):
            draw_capsule(img, to_pixel(pts[a]), to_pixel(pts[b]),
                         FINGER_TH, SKIN, OUTLINE)


def draw_face(img, nose_px, neck_px):
    """목 위에 얼굴 원 + 눈/입"""
    # 얼굴 크기: 목-코 거리 기반
    dx = nose_px[0] - neck_px[0]
    dy = nose_px[1] - neck_px[1]
    r = max(24, int((dx**2 + dy**2) ** 0.5 * 0.9))
    # 얼굴 중심: 코 위치에서 위로 살짝
    fx, fy = nose_px[0], nose_px[1] - r // 3

    cv2.circle(img, (fx, fy), r, SKIN, -1, cv2.LINE_AA)
    cv2.circle(img, (fx, fy), r, OUTLINE, 2, cv2.LINE_AA)
    # 눈
    eye_dx, eye_dy = r // 3, r // 6
    cv2.circle(img, (fx - eye_dx, fy - eye_dy), max(2, r//10), FACE_LN, -1, cv2.LINE_AA)
    cv2.circle(img, (fx + eye_dx, fy - eye_dy), max(2, r//10), FACE_LN, -1, cv2.LINE_AA)
    # 입(살짝 웃는 곡선)
    cv2.ellipse(img, (fx, fy + r//4), (r//3, r//4), 0, 20, 160, FACE_LN, 2, cv2.LINE_AA)


# ------------------------------------------------------------
# 4) 한 프레임 -> 캐릭터 이미지
# ------------------------------------------------------------
def draw_frame(frame_vec):
    kp = frame_vec.reshape(-1, 2)   # (50,2)
    img = np.full((CANVAS, CANVAS, 3), 245, dtype=np.uint8)  # 연한 배경

    r_sh, l_sh = kp[I_RSH], kp[I_LSH]
    neck, nose = kp[I_NECK], kp[I_NOSE]

    # --- 몸통(어깨~허리 사다리꼴) ---
    if is_valid(r_sh) and is_valid(l_sh) and is_valid(neck):
        r_sh_px, l_sh_px = to_pixel(r_sh), to_pixel(l_sh)
        # 허리는 어깨 아래로 살짝 (정규화상 아래 방향으로)
        hip_off = int(CANVAS * 0.22 * 0.9)   # scale * 0.9 만큼 아래
        r_hip = (r_sh_px[0] + int((l_sh_px[0]-r_sh_px[0])*0.15), r_sh_px[1] + hip_off)
        l_hip = (l_sh_px[0] - int((l_sh_px[0]-r_sh_px[0])*0.15), l_sh_px[1] + hip_off)
        body_poly = np.array([r_sh_px, l_sh_px, l_hip, r_hip], dtype=np.int32)
        cv2.fillPoly(img, [body_poly], BODY, cv2.LINE_AA)
        cv2.polylines(img, [body_poly], True, OUTLINE, 2, cv2.LINE_AA)
        # 목
        draw_capsule(img, to_pixel(neck), to_pixel(nose), ARM_TH, SKIN, OUTLINE)

    # --- 팔(캡슐) ---
    for a, b in ARM_BONES:
        if is_valid(kp[a]) and is_valid(kp[b]):
            draw_capsule(img, to_pixel(kp[a]), to_pixel(kp[b]), ARM_TH, SKIN, OUTLINE)

    # --- 얼굴 ---
    if is_valid(nose) and is_valid(neck):
        draw_face(img, to_pixel(nose), to_pixel(neck))

    # --- 손 (팔보다 나중에 그려서 위에 오도록) ---
    draw_hand(img, kp, LHAND_START)
    draw_hand(img, kp, RHAND_START)

    return img


# ------------------------------------------------------------
# 5) 메인
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"frame {f+1}/{len(seq)}  sample {idx+1}/{len(X)}",
                    (15, CANVAS - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)

        cv2.imshow("SignBridge - Avatar Preview (character)", img)

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

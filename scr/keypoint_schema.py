# keypoint_schema.py
# 목적: AI Hub(OpenPose) 데이터와 웹캠(MediaPipe) 입력을 "같은 형식"으로 통일
# 핵심: 두 소스가 공통으로 안정적으로 잡는 관절만 골라 하나의 스펙으로 정의

import numpy as np

# ------------------------------------------------------------
# 1) 공통 스펙 정의
#    - 손: 양손 각 21점 (OpenPose hand == MediaPipe hand, 완전 동일)
#    - 상체: 어깨/팔꿈치/손목/코 등 수어에 꼭 필요한 점만 선별
# ------------------------------------------------------------

# OpenPose BODY_25에서 우리가 쓸 관절의 인덱스
# (0코 1목 2오른어깨 3오른팔꿈치 4오른손목 5왼어깨 6왼팔꿈치 7왼손목 ...)
OPENPOSE_POSE_USE = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
    "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
}
# 우리가 최종적으로 쓸 상체 관절 순서 (이 순서를 MediaPipe에서도 똑같이 맞춘다)
POSE_ORDER = ["nose", "neck",
              "r_shoulder", "r_elbow", "r_wrist",
              "l_shoulder", "l_elbow", "l_wrist"]

N_POSE = len(POSE_ORDER)   # 8
N_HAND = 21                # 손 한쪽
# 최종 관절 수 = 상체8 + 왼손21 + 오른손21 = 50점, 각 (x,y) → 100차원
N_KEYPOINTS = N_POSE + N_HAND * 2   # 50
FEATURE_DIM = N_KEYPOINTS * 2       # 100


def _to_xy(flat, n):
    """OpenPose flat 배열 [x,y,c, x,y,c ...] -> (n,2) 배열 (x,y만)"""
    arr = np.array(flat, dtype=np.float32).reshape(-1, 3)
    return arr[:n, :2]


def openpose_to_common(people: dict) -> np.ndarray:
    """
    AI Hub keypoint JSON의 'people' 딕셔너리 하나(=한 프레임)를
    공통 포맷 (N_KEYPOINTS, 2) 배열로 변환.
    """
    pose_all = _to_xy(people.get("pose_keypoints_2d", []), 25)   # (25,2)
    lhand = _to_xy(people.get("hand_left_keypoints_2d", []), 21) # (21,2)
    rhand = _to_xy(people.get("hand_right_keypoints_2d", []), 21)# (21,2)

    # 상체는 선별된 관절만 순서대로
    pose_sel = np.stack([pose_all[OPENPOSE_POSE_USE[j]] for j in POSE_ORDER])  # (8,2)

    # 손이 비어있으면(감지 실패) 0으로 채움
    if lhand.shape[0] < 21:
        lhand = np.zeros((21, 2), dtype=np.float32)
    if rhand.shape[0] < 21:
        rhand = np.zeros((21, 2), dtype=np.float32)

    return np.concatenate([pose_sel, lhand, rhand], axis=0)  # (50,2)


def normalize(kp: np.ndarray) -> np.ndarray:
    """
    위치·크기 정규화: 사람이 화면 어디에 있든 같은 동작이 같은 값이 되도록.
    - 원점: 두 어깨의 중점 (몸 중심)
    - 스케일: 두 어깨 사이 거리
    입력/출력: (N_KEYPOINTS, 2)
    """
    kp = kp.copy()
    r_sh = kp[POSE_ORDER.index("r_shoulder")]
    l_sh = kp[POSE_ORDER.index("l_shoulder")]
    center = (r_sh + l_sh) / 2.0
    scale = np.linalg.norm(r_sh - l_sh)
    if scale < 1e-6:
        scale = 1.0
    kp = (kp - center) / scale
    return kp


def flatten(kp: np.ndarray) -> np.ndarray:
    """(N_KEYPOINTS, 2) -> (FEATURE_DIM,) 1차원, 모델 입력용"""
    return kp.reshape(-1).astype(np.float32)

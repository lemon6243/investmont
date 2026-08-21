# keypoint_schema_3d.py
# 목적: AI Hub(OpenPose 3D) 데이터를 두 용도로 변환
#   (1) 인식용: 어깨 중앙 원점 + 어깨너비 스케일 정규화 (카메라 거리 무관)
#   (2) 아바타용: 정규화하지 않은 원본 3D + 언리얼 좌표계 변환
# 핵심: 원본 3D 좌표는 절대 버리지 않는다. 정규화는 인식 입력에만 적용.

import numpy as np

# ------------------------------------------------------------
# OpenPose BODY25 중 사용할 관절 (상체)
# ------------------------------------------------------------
OPENPOSE_POSE_USE = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
    "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
}

POSE_ORDER = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
]

N_POSE = len(POSE_ORDER)          # 8
N_HAND = 21                       # 손 한쪽
N_KEYPOINTS = N_POSE + N_HAND * 2 # 50
FEATURE_DIM = N_KEYPOINTS * 3     # 150  (x,y,z)

# 신뢰도 임계값: 이 값보다 낮으면 "감지 실패"로 간주
CONF_MIN = 0.05

# 언리얼 좌표계 변환 스케일 (m -> cm). 필요시 조정.
UNREAL_SCALE = 100.0


# ------------------------------------------------------------
# 내부: OpenPose 3D flat 배열 -> (N,3) xyz + (N,) conf
# OpenPose 3D 포맷: [x, y, z, c, x, y, z, c, ...]  (관절당 4개)
# ------------------------------------------------------------
def _to_xyz_conf(flat, expected_count, key_name=""):
    """
    반환: xyz (expected_count, 3), conf (expected_count,)
    - 키가 비었으면 0으로 채우고 conf=0 (감지 실패로 표시)
    - 길이가 모자라면 뒤를 0 패딩
    """
    arr = np.array(flat, dtype=np.float32)

    if arr.size == 0:
        if key_name:
            # 진짜로 3D 키가 없는 프레임을 조용히 넘기지 않도록 표시
            # (대량 처리 중엔 로그가 많아지므로 필요할 때만 주석 해제)
            # print(f"[경고] '{key_name}' 비어있음 (감지 실패 프레임)")
            pass
        return (np.zeros((expected_count, 3), np.float32),
                np.zeros((expected_count,), np.float32))

    arr = arr.reshape(-1, 4)          # [x,y,z,c]
    xyz = arr[:, :3]
    conf = arr[:, 3]

    n = xyz.shape[0]
    if n < expected_count:
        xyz = np.concatenate(
            [xyz, np.zeros((expected_count - n, 3), np.float32)], axis=0)
        conf = np.concatenate(
            [conf, np.zeros((expected_count - n,), np.float32)], axis=0)

    return xyz[:expected_count], conf[:expected_count]


# ------------------------------------------------------------
# AIHub OpenPose 3D JSON(people 딕셔너리) -> 공통포맷
# 반환: kp (50,3), conf (50,)  ← 정규화 전 원본 3D
# ------------------------------------------------------------
def openpose_to_common_3d(people):
    """
    people: keypoint JSON의 'people' 딕셔너리 하나(=한 프레임).
            (이 데이터셋은 people이 리스트가 아니라 딕셔너리 1개)
    """
    pose_all, pose_c = _to_xyz_conf(
        people.get("pose_keypoints_3d", []), 25, "pose_keypoints_3d")
    lhand, lhand_c = _to_xyz_conf(
        people.get("hand_left_keypoints_3d", []), 21, "hand_left_keypoints_3d")
    rhand, rhand_c = _to_xyz_conf(
        people.get("hand_right_keypoints_3d", []), 21, "hand_right_keypoints_3d")

    # 상체는 선별 관절만 순서대로
    pose_sel = np.stack([pose_all[OPENPOSE_POSE_USE[n]] for n in POSE_ORDER])
    pose_sel_c = np.stack([pose_c[OPENPOSE_POSE_USE[n]] for n in POSE_ORDER])

    kp = np.concatenate([pose_sel, lhand, rhand], axis=0)      # (50,3)
    conf = np.concatenate([pose_sel_c, lhand_c, rhand_c], axis=0)  # (50,)
    return kp, conf


# ------------------------------------------------------------
# 신뢰도 마스킹: c가 낮은 관절을 직전 유효 프레임 값으로 채움
# 시퀀스 단위로 적용 (프레임 간 연속성 사용)
# ------------------------------------------------------------
def fill_low_conf_sequence(seq_kp, seq_conf):
    """
    seq_kp:   (T, 50, 3)  원본 3D 시퀀스
    seq_conf: (T, 50)     프레임별 관절 신뢰도
    반환: 보정된 seq_kp (T, 50, 3)
    - c < CONF_MIN 인 관절은 직전 유효 프레임 값으로 대체
    - 첫 프레임부터 실패면 0 유지
    """
    seq = seq_kp.copy()
    T, J, _ = seq.shape
    last_valid = np.zeros((J, 3), np.float32)
    has_valid = np.zeros((J,), dtype=bool)

    for t in range(T):
        bad = seq_conf[t] < CONF_MIN
        # 유효한 관절은 last_valid 갱신
        good = ~bad
        last_valid[good] = seq[t][good]
        has_valid[good] = True
        # 실패한 관절 중 과거 유효값이 있으면 그 값으로 대체
        fill = bad & has_valid
        seq[t][fill] = last_valid[fill]
    return seq


# ------------------------------------------------------------
# (1) 인식용 정규화: 어깨 중앙 원점 + 어깨너비 스케일
#     카메라 거리(z 절대값)에 무관하게 만든다.
# ------------------------------------------------------------
def normalize_3d(kp):
    kp = kp.copy()
    r_sh = kp[POSE_ORDER.index("r_shoulder")]
    l_sh = kp[POSE_ORDER.index("l_shoulder")]
    center = (r_sh + l_sh) / 2.0
    scale = np.linalg.norm(r_sh - l_sh)
    if scale < 1e-6:
        scale = 1.0
    return (kp - center) / scale


# ------------------------------------------------------------
# (2) 아바타용: 언리얼 좌표계로 변환
#     OpenPose(카메라) 좌표계: x 오른쪽, y 아래, z 카메라에서 멀어짐
#     언리얼 좌표계: x 앞, y 오른쪽, z 위 (왼손, cm)
#     매핑: UEx = z, UEy = x, UEz = -y  (필요시 데모 보며 조정)
# ------------------------------------------------------------
def to_unreal(kp, scale=UNREAL_SCALE, recenter=True):
    """
    kp: (50,3) 원본 3D (정규화 전)
    반환: (50,3) 언리얼 좌표계(cm)
    recenter=True 면 어깨 중앙을 원점으로 이동(아바타 배치 편의).
    """
    kp = kp.copy()
    if recenter:
        r_sh = kp[POSE_ORDER.index("r_shoulder")]
        l_sh = kp[POSE_ORDER.index("l_shoulder")]
        center = (r_sh + l_sh) / 2.0
        kp = kp - center

    x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]
    ue = np.stack([z, x, -y], axis=1) * scale   # y → -y 로 수정 (주석과 일치)

    return ue.astype(np.float32)



# ------------------------------------------------------------
# 모델 입력용 평탄화: (50,3) -> (150,)
# ------------------------------------------------------------
def flatten_3d(kp):
    return kp.reshape(-1).astype(np.float32)

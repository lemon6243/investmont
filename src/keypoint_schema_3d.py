import numpy as np

# ------------------------------------------------------------
# OpenPose BODY25 중 사용할 관절
# ------------------------------------------------------------

OPENPOSE_POSE_USE = {
    "nose": 0,
    "neck": 1,

    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,

    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
}

POSE_ORDER = [
    "nose",
    "neck",

    "r_shoulder",
    "r_elbow",
    "r_wrist",

    "l_shoulder",
    "l_elbow",
    "l_wrist",
]

N_POSE = len(POSE_ORDER)

N_HAND = 21

N_KEYPOINTS = N_POSE + (N_HAND * 2)

FEATURE_DIM = N_KEYPOINTS * 3

# ------------------------------------------------------------
# OpenPose 3D
# [x,y,z,c] -> (N,3)
# ------------------------------------------------------------

def _to_xyz(flat, expected_count):

    arr = np.array(flat, dtype=np.float32)

    if len(arr) == 0:
        return np.zeros((expected_count, 3), dtype=np.float32)

    arr = arr.reshape(-1, 4)

    xyz = arr[:, :3]

    if xyz.shape[0] < expected_count:

        pad = np.zeros(
            (expected_count - xyz.shape[0], 3),
            dtype=np.float32
        )

        xyz = np.concatenate(
            [xyz, pad],
            axis=0
        )

    return xyz[:expected_count]


# ------------------------------------------------------------
# AIHub OpenPose JSON -> 공통포맷
# 결과:
# (50,3)
# ------------------------------------------------------------

def openpose_to_common_3d(people):

    pose_all = _to_xyz(
        people.get("pose_keypoints_3d", []),
        25
    )

    lhand = _to_xyz(
        people.get("hand_left_keypoints_3d", []),
        21
    )

    rhand = _to_xyz(
        people.get("hand_right_keypoints_3d", []),
        21
    )

    pose_sel = np.stack(
        [
            pose_all[OPENPOSE_POSE_USE[name]]
            for name in POSE_ORDER
        ]
    )

    return np.concatenate(
        [
            pose_sel,
            lhand,
            rhand
        ],
        axis=0
    )


# ------------------------------------------------------------
# 정규화
#
# 어깨 중앙 기준
# ------------------------------------------------------------

def normalize_3d(kp):

    kp = kp.copy()

    r_sh = kp[
        POSE_ORDER.index("r_shoulder")
    ]

    l_sh = kp[
        POSE_ORDER.index("l_shoulder")
    ]

    center = (r_sh + l_sh) / 2.0

    scale = np.linalg.norm(
        r_sh - l_sh
    )

    if scale < 1e-6:
        scale = 1.0

    kp = (kp - center) / scale

    return kp


# ------------------------------------------------------------
# 모델 입력용
# (50,3) -> (150,)
# ------------------------------------------------------------

def flatten_3d(kp):

    return kp.reshape(-1).astype(np.float32)

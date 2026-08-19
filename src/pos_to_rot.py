# pos_to_rot.py
# X_raw_3d.npy (N,T,50,3) 위치 -> 로컬 쿼터니언
# 출력: dataset_all_3d/rots/{단어}.npz

import os
import json
import numpy as np
from collections import defaultdict

from keypoint_schema_3d import POSE_ORDER, N_POSE, N_HAND

DATA_DIR = "dataset_all_3d"
XR_PATH = os.path.join(DATA_DIR, "X_raw_3d.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all_3d.npy")
OUT_DIR = os.path.join(DATA_DIR, "rots")

# 50점 이름
JOINTS = (
    list(POSE_ORDER)
    + [f"lhand_{i}" for i in range(N_HAND)]
    + [f"rhand_{i}" for i in range(N_HAND)]
)
J = {n: i for i, n in enumerate(JOINTS)}

# (부모, 자식, 본이름)  — 자식 방향으로 본이 향함
HAND_CHAINS = [
    (0, 1, "thumb_01"), (1, 2, "thumb_02"), (2, 3, "thumb_03"),
    (0, 5, "index_01"), (5, 6, "index_02"), (6, 7, "index_03"),
    (0, 9, "middle_01"), (9, 10, "middle_02"), (10, 11, "middle_03"),
    (0, 13, "ring_01"), (13, 14, "ring_02"), (14, 15, "ring_03"),
    (0, 17, "pinky_01"), (17, 18, "pinky_02"), (18, 19, "pinky_03"),
]

BONES = [
    ("neck", "nose", "head"),
    ("neck", "r_shoulder", "clavicle_r"),
    ("r_shoulder", "r_elbow", "upperarm_r"),
    ("r_elbow", "r_wrist", "lowerarm_r"),
    ("r_wrist", "rhand_0", "hand_r"),
    ("neck", "l_shoulder", "clavicle_l"),
    ("l_shoulder", "l_elbow", "upperarm_l"),
    ("l_elbow", "l_wrist", "lowerarm_l"),
    ("l_wrist", "lhand_0", "hand_l"),
]
for a, b, name in HAND_CHAINS:
    BONES.append((f"lhand_{a}", f"lhand_{b}", f"{name}_l"))
    BONES.append((f"rhand_{a}", f"rhand_{b}", f"{name}_r"))

# 자식 -> 부모 (로컬 회전용)
PARENT = {child: parent for parent, child, _ in BONES}


def _valid(p):
    return np.linalg.norm(p) > 1e-8


def quat_from_to(a, b):
    """벡터 a를 b로 돌리는 쿼터니언 (x,y,z,w). Unreal 호환."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    c = np.dot(a, b)
    if c > 0.9999:
        return np.array([0, 0, 0, 1], np.float32)
    if c < -0.9999:
        axis = np.cross(a, np.array([1, 0, 0], np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0, 1, 0], np.float32))
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        return np.array([axis[0], axis[1], axis[2], 0], np.float32)
    v = np.cross(a, b)
    q = np.array([v[0], v[1], v[2], 1.0 + c], np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], np.float32)


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float32)


def quat_rotate(q, v):
    qv = np.array([v[0], v[1], v[2], 0], np.float32)
    return quat_mul(quat_mul(q, qv), quat_inv(q))[:3]


def frame_is_empty(kp, min_ratio=0.3):
    valid = np.linalg.norm(kp, axis=1) > 1e-6
    return valid.mean() < min_ratio


def pick_best_clip(clips):
    """유효 관절 비율이 가장 높은 클립 1개."""
    scores = []
    for seq in clips:
        good = 0
        total = 0
        for f in seq:
            v = np.linalg.norm(f, axis=1) > 1e-6
            good += v.sum()
            total += len(v)
        scores.append(good / max(total, 1))
    return clips[int(np.argmax(scores))]


def build_rest_pose(seq):
    """
    첫 유효 프레임을 rest로 씀.
    (T포즈가 없어도 '이 클립의 시작 자세'가 rest가 되면
     상대 회전이 안정적이다.)
    """
    for f in seq:
        if not frame_is_empty(f):
            return f.copy()
    return seq[0].copy()

# 언리얼 좌표계(x 앞, y 오른쪽, z 위) 기준 T포즈 본 방향.
# send_mh_live.py가 seq_to_ue로 이미 언리얼 좌표로 바꿔서 넘기므로
# rest를 언리얼 T포즈로 정의하면 계산된 회전이 뼈 0도(T포즈)에 맞는다.
TPOSE_DIR = {
    "head":       np.array([0, 0, 1], np.float32),    # 목->코: 위
    "clavicle_r": np.array([0, 1, 0], np.float32),    # 목->오른어깨: 오른쪽(+Y)
    "upperarm_r": np.array([0, 1, 0], np.float32),    # 오른팔: 오른쪽
    "lowerarm_r": np.array([0, 1, 0], np.float32),
    "hand_r":     np.array([0, 1, 0], np.float32),
    "clavicle_l": np.array([0, -1, 0], np.float32),   # 목->왼어깨: 왼쪽(-Y)
    "upperarm_l": np.array([0, -1, 0], np.float32),   # 왼팔: 왼쪽
    "lowerarm_l": np.array([0, -1, 0], np.float32),
    "hand_l":     np.array([0, -1, 0], np.float32),
}

# 손가락: T포즈에서 손가락은 팔과 같은 방향(바깥쪽)으로 뻗어 있다고 가정.
# 오른손 손가락은 +Y, 왼손 손가락은 -Y.
for _a, _b, _name in HAND_CHAINS:
    TPOSE_DIR[f"{_name}_r"] = np.array([0, 1, 0], np.float32)
    TPOSE_DIR[f"{_name}_l"] = np.array([0, -1, 0], np.float32)


def positions_to_local_quats(seq):
    """
    seq: (T,50,3) 월드 위치
    return: bones dict -> (T,4) 로컬 xyzw, root (T,3)
    """
    rest = build_rest_pose(seq)
    T = seq.shape[0]
    out = {name: np.zeros((T, 4), np.float32) for _, _, name in BONES}
    for _, _, name in BONES:
        out[name][:, 3] = 1.0  # identity

    root = np.zeros((T, 3), np.float32)
    rsh, lsh = J["r_shoulder"], J["l_shoulder"]

    # rest 본 방향 (월드)
        # rest 본 방향: 클립 첫 프레임이 아니라 고정 T포즈를 기준으로 사용.
    # (이래야 모든 클립이 언리얼 T포즈와 같은 기준을 갖고,
    #  팔이 엉뚱하게 돌지 않는다.)
    rest_dir = {}
    for parent, child, name in BONES:
        if name in TPOSE_DIR:
            rest_dir[name] = TPOSE_DIR[name].copy()
        else:
            # 정의 안 된 본은 안전하게 첫 프레임 방향 사용
            rest_dir[name] = rest[J[child]] - rest[J[parent]]


    for t in range(T):
        kp = seq[t]
        if _valid(kp[rsh]) and _valid(kp[lsh]):
            root[t] = (kp[rsh] + kp[lsh]) * 0.5

        world_q = {}  # 조인트 월드 회전
        # 목은 단위
        world_q["neck"] = np.array([0, 0, 0, 1], np.float32)
        world_q["r_wrist"] = np.array([0, 0, 0, 1], np.float32)
        world_q["l_wrist"] = np.array([0, 0, 0, 1], np.float32)

        for parent, child, name in BONES:
            a = kp[J[parent]]
            b = kp[J[child]]
            if not (_valid(a) and _valid(b) and _valid(rest_dir[name])):
                continue
            cur = b - a
            if np.linalg.norm(cur) < 1e-6:
                continue

            # 월드 스윙: rest_dir -> 현재 방향
            q_world = quat_from_to(rest_dir[name], cur)

            q_parent = world_q.get(parent, np.array([0, 0, 0, 1], np.float32))
            q_local = quat_mul(quat_inv(q_parent), q_world)
            out[name][t] = q_local
            world_q[child] = q_world

    return out, root


def trim_seq(seq):
    n = len(seq)
    s = 0
    while s < n and frame_is_empty(seq[s]):
        s += 1
    e = n - 1
    while e > s and frame_is_empty(seq[e]):
        e -= 1
    return seq[s:e + 1] if s < e else seq


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Xr = np.load(XR_PATH)                          # (N,T,50,3)
    y = np.load(Y_PATH, allow_pickle=True)

    by_word = defaultdict(list)
    for seq, label in zip(Xr, y):
        by_word[str(label)].append(seq)

    catalog = {}
    print(f">> 단어 {len(by_word)}개, 클립 {len(y)}개")

    for i, (word, clips) in enumerate(sorted(by_word.items()), 1):
        seq = trim_seq(pick_best_clip(clips))
        bones, root = positions_to_local_quats(seq)
        path = os.path.join(OUT_DIR, f"{word}.npz")
        np.savez_compressed(
            path,
            root=root,
            joint_names=np.array(JOINTS),
            **bones,
        )
        catalog[word] = {
            "file": f"{word}.npz",
            "frames": int(seq.shape[0]),
            "clips_available": len(clips),
        }
        if i % 20 == 0 or i == len(by_word):
            print(f"   {i}/{len(by_word)}  {word}  T={seq.shape[0]}")

    with open(os.path.join(OUT_DIR, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(">> 저장:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()

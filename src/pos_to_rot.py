# pos_to_rot.py
# X_raw_3d.npy (N,T,50,3) 위치 -> 로컬 쿼터니언
# 출력: dataset_all_3d/rots/{단어}.npz
#
# [rest 기준: 데이터 첫 유효 프레임(팔 내린 자세)로 통일]
#   - 모든 본이 "데이터 첫 프레임 = 0도" 기준.
#   - 언리얼에서는 팔 6개 본을 Add to Existing 으로 설정.

import os
import re
import json
import numpy as np
from collections import defaultdict

from keypoint_schema_3d import POSE_ORDER, N_POSE, N_HAND

# pos_to_rot.py 상단에 추가
# ============================================================
# MetaHuman Reference Pose Offset
# Unreal 에디터 → Skeletal Mesh → Bone Transform → Local Rotation
# 집에서 확인 후 값을 채워넣으세요.
# ============================================================
MH_REF_POSE_LOCAL = {
    # TODO: 집에서 UE 에디터에서 본별 Bone Space Local Rotation 확인 후 채우기
    "upperarm_l": np.array([0, 0, 0, 1], np.float32),  # placeholder
    "lowerarm_l": np.array([0, 0, 0, 1], np.float32),
    "hand_l":      np.array([0, 0, 0, 1], np.float32),
    "upperarm_r": np.array([0, 0, 0, 1], np.float32),
    "lowerarm_r": np.array([0, 0, 0, 1], np.float32),
    "hand_r":      np.array([0, 0, 0, 1], np.float32),
}


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
    ("r_wrist", "rhand_9", "hand_r"),
    ("neck", "l_shoulder", "clavicle_l"),
    ("l_shoulder", "l_elbow", "upperarm_l"),
    ("l_elbow", "l_wrist", "lowerarm_l"),
    ("l_wrist", "lhand_9", "hand_l"),
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


def make_basis(primary, secondary):
    """
    primary(주축=본 방향)와 secondary(보조축=손바닥 법선)로
    정규직교 3x3 회전행렬을 만든다. 열: [x축, y축, z축].
    x축 = primary 방향, z축 = primary와 secondary에 수직, y축 = z×x.
    실패(평행/영벡터) 시 None.
    """
    x = primary / (np.linalg.norm(primary) + 1e-8)
    s = secondary / (np.linalg.norm(secondary) + 1e-8)
    z = np.cross(x, s)
    zn = np.linalg.norm(z)
    if zn < 1e-4:            # primary와 secondary가 거의 평행
        return None
    z = z / zn
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).astype(np.float32)  # (3,3)


def rotmat_to_quat(R):
    """3x3 회전행렬 -> 쿼터니언 (x,y,z,w)."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def palm_normal(kp, hand_start):
    """손목(0), 검지뿌리(5), 새끼뿌리(17)로 손바닥 법선 계산. 실패 시 None."""
    p0 = kp[hand_start + 0]
    p5 = kp[hand_start + 5]
    p17 = kp[hand_start + 17]
    if min(np.linalg.norm(p0), np.linalg.norm(p5), np.linalg.norm(p17)) < 1e-6:
        return None
    n = np.cross(p5 - p0, p17 - p0)
    ln = np.linalg.norm(n)
    if ln < 1e-6:
        return None
    return (n / ln).astype(np.float32)


# basis 방식을 쓸 본과, 그 본이 참조할 손의 시작 인덱스
#   lowerarm/hand 는 손바닥 법선으로 twist를 정의한다.
#   왼손 시작 = 8, 오른손 시작 = 29 (JOINTS 순서: pose8 + lhand21 + rhand21)
BASIS_BONES = {
    "lowerarm_r": 29, "hand_r": 29,
    "lowerarm_l": 8,  "hand_l": 8,
}


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


# [참고용으로만 남김 — 현재 파이프라인에서는 사용하지 않음]
# 언리얼 T포즈 기준으로 되돌리고 싶을 때(A안) 참조하는 값.
# 지금은 rest_dir 를 데이터 첫 프레임 방향으로 통일하므로 쓰이지 않는다.
TPOSE_DIR = {
    "head":       np.array([0, 0, 1], np.float32),
    "clavicle_r": np.array([0, 1, 0], np.float32),
    "upperarm_r": np.array([0, 1, 0], np.float32),
    "lowerarm_r": np.array([0, 1, 0], np.float32),
    "hand_r":     np.array([0, 1, 0], np.float32),
    "clavicle_l": np.array([0, -1, 0], np.float32),
    "upperarm_l": np.array([0, -1, 0], np.float32),
    "lowerarm_l": np.array([0, -1, 0], np.float32),
    "hand_l":     np.array([0, -1, 0], np.float32),
}
for _a, _b, _name in HAND_CHAINS:
    TPOSE_DIR[f"{_name}_r"] = np.array([0, 1, 0], np.float32)
    TPOSE_DIR[f"{_name}_l"] = np.array([0, -1, 0], np.float32)


def positions_to_local_quats(seq):
    """
    seq: (T,50,3) 월드 위치 (언리얼 좌표계)
    return: (out 로컬쿼터니언 dict, root (T,3), bone_len dict)

    [rest 기준 통일] 모든 본이 '데이터 첫 프레임 = 0도'. 언리얼 Add to Existing.

    [손바닥 법선 안정화 - 강화판]
      1) 부호 정렬: 직전 법선과 내적 음수면 뒤집기.
      2) 각도 게이트: 부호 정렬 후에도 직전과 내적이 THRESH 미만이면
         (= 급격히 흔들린 노이즈 프레임) 이번 법선을 버리고 직전 값 유지.

    [뼈 길이] rest 프레임에서 부모->자식 거리를 재서 bone_len 으로 반환.
             (FK 미리보기에서 관절 위치 복원에 사용)
    """
    NORMAL_SIM_THRESH = 0.6   # 수어의 빠른 손목 회전 대응 (cos⁻¹(0.6) ≈ 53°)

    rest = build_rest_pose(seq)
    T = seq.shape[0]
    out = {name: np.zeros((T, 4), np.float32) for _, _, name in BONES}
    for _, _, name in BONES:
        out[name][:, 3] = 1.0  # identity

    root = np.zeros((T, 3), np.float32)
    rsh, lsh = J["r_shoulder"], J["l_shoulder"]

    # rest 본 방향 + 뼈 길이 (데이터 첫 유효 프레임 기준)
    rest_dir = {}
    bone_len = {}
    for parent, child, name in BONES:
        d = rest[J[child]] - rest[J[parent]]
        rest_dir[name] = d
        bone_len[name] = float(np.linalg.norm(d))   # rest에서의 뼈 길이

    rest_basis = {}
    rest_normal = {}
    for name, hstart in BASIS_BONES.items():
        parent = child = None
        for p, c, nm in BONES:
            if nm == name:
                parent, child = p, c
                break
        prim = rest[J[child]] - rest[J[parent]]
        sec = palm_normal(rest, hstart)
        rest_basis[name] = make_basis(prim, sec) if sec is not None else None
        rest_normal[name] = sec

    prev_normal = {name: rest_normal[name] for name in BASIS_BONES}

    for t in range(T):
        kp = seq[t]
        if _valid(kp[rsh]) and _valid(kp[lsh]):
            root[t] = (kp[rsh] + kp[lsh]) * 0.5

        world_q = {}
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

            if name in BASIS_BONES and rest_basis.get(name) is not None:
                sec = palm_normal(kp, BASIS_BONES[name])
                if sec is not None:
                    ref = prev_normal.get(name)
                    if ref is not None:
                        # 1) 부호 정렬
                        if np.dot(sec, ref) < 0:
                            sec = -sec
                        # 2) 각도 게이트: 부호 맞춰도 급격히 어긋나면 직전 유지
                        if np.dot(sec, ref) < NORMAL_SIM_THRESH:
                            sec = ref
                    prev_normal[name] = sec
                else:
                    # 법선 계산 실패 시에도 직전 값으로 대체
                    sec = prev_normal.get(name)
                Rcur = make_basis(cur, sec) if sec is not None else None
                if Rcur is not None:
                    Rrel = Rcur @ rest_basis[name].T
                    q_world = rotmat_to_quat(Rrel)
                else:
                    q_world = quat_from_to(rest_dir[name], cur)
            else:
                q_world = quat_from_to(rest_dir[name], cur)

            q_parent = world_q.get(parent, np.array([0, 0, 0, 1], np.float32))
            q_local = quat_mul(quat_inv(q_parent), q_world)   # ← 이 줄 복구!
            
            # === MetaHuman Reference Pose 보정 ===
            if name in MH_REF_POSE_LOCAL:
                ref = MH_REF_POSE_LOCAL[name]
                q_local = quat_mul(q_local, ref)
            
            out[name][t] = q_local
            world_q[child] = q_world

    return out, root, bone_len


def trim_seq(seq):
    n = len(seq)
    s = 0
    while s < n and frame_is_empty(seq[s]):
        s += 1
    e = n - 1
    while e > s and frame_is_empty(seq[e]):
        e -= 1
    return seq[s:e + 1] if s < e else seq


def safe_name(word):
    """파일 이름에 못 쓰는 문자 제거 (줄바꿈/탭/윈도우 금지문자)."""
    s = word.strip()                          # 앞뒤 공백/줄바꿈 제거
    s = re.sub(r'[\r\n\t]', '', s)            # 내부 줄바꿈/탭 제거
    s = re.sub(r'[<>:"/\\|?*]', '_', s)       # 윈도우 파일명 금지문자 -> _
    s = s.strip()
    return s if s else "unknown"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(">>> [뼈길이 저장 버전] 실행 중 <<<")

    Xr = np.load(XR_PATH)                          # (N,T,50,3)
    y = np.load(Y_PATH, allow_pickle=True)

    by_word = defaultdict(list)
    for seq, label in zip(Xr, y):
        by_word[str(label)].append(seq)

    catalog = {}
    print(f">> 단어 {len(by_word)}개, 클립 {len(y)}개")

    bone_names = [nm for _, _, nm in BONES]   # 저장 순서 고정

    used = {}  # 정리 후 이름 충돌 방지
    for i, (word, clips) in enumerate(sorted(by_word.items()), 1):
        seq = trim_seq(pick_best_clip(clips))
        bones, root, bone_len = positions_to_local_quats(seq)

        name = safe_name(word)
        # 정리 후 이름이 겹치면 뒤에 번호 붙이기
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0

        bone_lens = np.array([bone_len[nm] for nm in bone_names], np.float32)

        path = os.path.join(OUT_DIR, f"{name}.npz")
        np.savez_compressed(
            path,
            root=root,
            joint_names=np.array(JOINTS),
            bone_names=np.array(bone_names),
            bone_lens=bone_lens,
            **bones,
        )
        catalog[word] = {                 # catalog 키는 원본 라벨 유지
            "file": f"{name}.npz",        # 실제 저장 파일명
            "frames": int(seq.shape[0]),
            "clips_available": len(clips),
        }
        if i % 20 == 0 or i == len(by_word):
            print(f"   {i}/{len(by_word)}  {name}  T={seq.shape[0]}")

    with open(os.path.join(OUT_DIR, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(">> 저장:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()

# preview3d.py
# 원본 위치(네온, 손가락 포함) + FK 복원 스켈레톤(대조용) 오버레이
# 사용법:
#   python preview3d.py            # catalog 첫 단어
#   python preview3d.py 가다        # 특정 단어
# 키: F = FK 뼈대 on/off,  Space = 일시정지,  마우스드래그 = 회전, 휠 = 확대

import os
import sys
import json
import numpy as np
import open3d as o3d
from collections import defaultdict

from pos_to_rot import (
    J, BONES, HAND_CHAINS,
    pick_best_clip, trim_seq, _valid,
    build_rest_pose,
    quat_mul, quat_rotate,
)

DATA_DIR = "dataset_all_3d"
ROTS_DIR = os.path.join(DATA_DIR, "rots")
XR_PATH = os.path.join(DATA_DIR, "X_raw_3d.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all_3d.npy")

# FK 루트(부모가 BONES에 없는 관절) — 이 위치는 원본에서 앵커로 가져옴
ROOT_JOINTS = ["neck", "r_wrist", "l_wrist"]

# 손가락 본 판별 (FK 대조는 팔까지만; 손가락 FK는 별도 앵커 필요해서 제외)
_FINGER = {
    "thumb_01", "thumb_02", "thumb_03",
    "index_01", "index_02", "index_03",
    "middle_01", "middle_02", "middle_03",
    "ring_01", "ring_02", "ring_03",
    "pinky_01", "pinky_02", "pinky_03",
}


def _is_finger(name):
    return name.endswith(("_r", "_l")) and name[:-2] in _FINGER


FK_BONES = [b for b in BONES if not _is_finger(b[2])]


# ---------- rest 방향: pos_to_rot 와 동일하게 ----------
def build_rest_dirs(seq):
    """pos_to_rot 의 build_rest_pose 로 rest 프레임을 잡고, 본별 rest 방향 계산."""
    rest = build_rest_pose(seq)          # pos_to_rot 와 동일 로직
    rest_dir = {}
    for parent, child, name in BONES:
        d = rest[J[child]] - rest[J[parent]]
        n = np.linalg.norm(d)
        rest_dir[name] = (d / n) if n > 1e-8 else np.array([1, 0, 0], np.float32)
    return rest_dir


# ---------- FK: 월드 회전 누적 (pos_to_rot 저장 방식의 정확한 역) ----------
def fk_frame(data, rest_dir, bone_len, root_pos, t):
    """
    각 본:
      world_q(child) = world_q(parent) * local_q(자식)   ← 저장 방식의 역
      cur_dir        = rotate(world_q(child), rest_dir)
      pos(child)     = pos(parent) + cur_dir * bone_len
    root 관절의 world_q 는 identity 로 시작(원본에서도 그렇게 잡았음).
    """
    pos = {rj: root_pos.get(rj, np.zeros(3, np.float32)).copy()
           for rj in ROOT_JOINTS}
    world_q = {rj: np.array([0, 0, 0, 1], np.float32) for rj in ROOT_JOINTS}

    remaining = list(FK_BONES)
    for _ in range(6):  # 여러 패스로 부모부터 채움
        nxt = []
        for parent, child, name in remaining:
            if parent not in pos:
                nxt.append((parent, child, name))
                continue
            local_q = data[name][t] if name in data.files \
                else np.array([0, 0, 0, 1], np.float32)
            q_parent = world_q.get(parent, np.array([0, 0, 0, 1], np.float32))
            q_world = quat_mul(q_parent, local_q)          # 누적
            cur_dir = quat_rotate(q_world, rest_dir[name])
            n = np.linalg.norm(cur_dir)
            if n > 1e-8:
                cur_dir = cur_dir / n
            pos[child] = pos[parent] + cur_dir * bone_len[name]
            world_q[child] = q_world
        remaining = nxt
        if not remaining:
            break
    return pos


# ---------- 좌표 변환: 화면용 y 뒤집기 ----------
def flip_y(p):
    q = np.array(p, np.float64).copy()
    q[..., 1] *= -1.0
    return q


# ---------- 지오메트리 ----------
def make_lineset(points, line_pairs, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(line_pairs, np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(line_pairs), 1)).astype(np.float64)
    )
    return ls


def make_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(points), 1)).astype(np.float64)
    )
    return pcd


# 원본(네온) — 손가락 포함 전체
def orig_geoms(kp):
    pts = flip_y(kp)
    lines = []
    for parent, child, _ in BONES:
        a, b = J[parent], J[child]
        if _valid(kp[a]) and _valid(kp[b]):
            lines.append([a, b])
    ls = make_lineset(pts, lines, [0.0, 0.9, 1.0])   # 시안 네온 선
    # 관절점: 유효한 것만
    valid_idx = [i for i in range(len(kp)) if _valid(kp[i])]
    jpts = pts[valid_idx]
    pcd = make_pcd(jpts, [1.0, 0.2, 0.8])            # 마젠타 관절
    return ls, pcd


# FK(대조) — 팔까지, 얇은 노란선
def fk_geoms(pos_dict):
    names = list(pos_dict.keys())
    idx = {n: i for i, n in enumerate(names)}
    pts = flip_y(np.array([pos_dict[n] for n in names], np.float32))
    lines = []
    for parent, child, _ in FK_BONES:
        if parent in idx and child in idx:
            lines.append([idx[parent], idx[child]])
    ls = make_lineset(pts, lines, [1.0, 0.85, 0.0])  # 노란선
    return ls


def load_word(word):
    Xr = np.load(XR_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    by_word = defaultdict(list)
    for seq, label in zip(Xr, y):
        by_word[str(label)].append(seq)
    if word not in by_word:
        print(f"단어 없음: {word}")
        print("예시:", list(by_word.keys())[:10])
        sys.exit(1)
    # 원본 좌표 그대로 (y 안 뒤집음 — 회전 계산과 좌표 일치용)
    return trim_seq(pick_best_clip(by_word[word]))


def main():
    args = sys.argv[1:]
    with open(os.path.join(ROTS_DIR, "catalog.json"), encoding="utf-8") as f:
        cat = json.load(f)
    word = args[0] if args else next(iter(cat.keys()))
    print(f">> 단어: {word}")

    fname = cat[word]["file"] if word in cat else f"{word}.npz"
    data = np.load(os.path.join(ROTS_DIR, fname), allow_pickle=True)
    bone_names = list(data["bone_names"])
    bone_len = {n: float(l) for n, l in zip(bone_names, data["bone_lens"])}

    seq = load_word(word)                 # 원본 좌표 (y 안 뒤집음)
    T = min(len(seq), len(data["head"]))
    rest_dir = build_rest_dirs(seq)

    print(f">> 프레임 {T}개.  F=FK대조 on/off,  Space=정지,  드래그=회전, 휠=확대")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"preview3d - {word}", width=1000, height=800)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.04, 0.04, 0.07])
    opt.point_size = 9.0
    opt.line_width = 3.0

    def build(t):
        kp = seq[t]
        o_ls, o_pcd = orig_geoms(kp)
        root_pos = {rj: kp[J[rj]].astype(np.float32) for rj in ROOT_JOINTS}
        pos = fk_frame(data, rest_dir, bone_len, root_pos, t)
        f_ls = fk_geoms(pos)
        return o_ls, o_pcd, f_ls

    o_ls, o_pcd, f_ls = build(0)
    vis.add_geometry(o_ls)
    vis.add_geometry(o_pcd)
    vis.add_geometry(f_ls)

    state = {"t": 0, "cnt": 0, "paused": False, "show_fk": True}

    def toggle_fk(v):
        state["show_fk"] = not state["show_fk"]
        # 끄면 선을 비움, 켜면 다시 채움 (다음 update 에서)
        if not state["show_fk"]:
            f_ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), np.int32))
            v.update_geometry(f_ls)
        return False

    def toggle_pause(v):
        state["paused"] = not state["paused"]
        return False

    vis.register_key_callback(ord("F"), toggle_fk)
    vis.register_key_callback(ord(" "), toggle_pause)

    def update(vis):
        if state["paused"]:
            return False
        state["cnt"] += 1
        if state["cnt"] % 3 != 0:   # 속도 조절
            return False

        t = state["t"]
        kp = seq[t]

        # 원본
        new_o_ls, new_o_pcd = orig_geoms(kp)
        o_ls.points = new_o_ls.points
        o_ls.lines = new_o_ls.lines
        o_ls.colors = new_o_ls.colors
        o_pcd.points = new_o_pcd.points
        o_pcd.colors = new_o_pcd.colors

        # FK
        if state["show_fk"]:
            root_pos = {rj: kp[J[rj]].astype(np.float32) for rj in ROOT_JOINTS}
            pos = fk_frame(data, rest_dir, bone_len, root_pos, t)
            new_f = fk_geoms(pos)
            f_ls.points = new_f.points
            f_ls.lines = new_f.lines
            f_ls.colors = new_f.colors

        vis.update_geometry(o_ls)
        vis.update_geometry(o_pcd)
        vis.update_geometry(f_ls)

        state["t"] = (t + 1) % T
        return False

    vis.register_animation_callback(update)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

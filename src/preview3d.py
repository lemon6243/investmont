# preview3d.py
# 네온/홀로그램 스타일: 원본(굵은 시안 실린더 + 마젠타 구체) + FK 대조(노란 얇은 선)
# 사용법:
#   python preview3d.py            # catalog 첫 단어
#   python preview3d.py 가다        # 특정 단어
# 키: F = FK대조 on/off,  Space = 정지,  드래그 = 회전, 휠 = 확대

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

ROOT_JOINTS = ["neck", "r_wrist", "l_wrist"]

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

# 색상
COL_BONE = [0.0, 0.85, 1.0]      # 시안 뼈
COL_JOINT = [1.0, 0.15, 0.75]    # 마젠타 관절
COL_FK = [1.0, 0.85, 0.0]        # 노란 FK 대조선

# 두께 (데이터 스케일이 약 미터 단위라 작게)
R_BONE = 0.012
R_JOINT = 0.020
R_FINGER_BONE = 0.006
R_FINGER_JOINT = 0.010


# ---------- rest 방향 (pos_to_rot 와 동일) ----------
def build_rest_dirs(seq):
    rest = build_rest_pose(seq)
    rest_dir = {}
    for parent, child, name in BONES:
        d = rest[J[child]] - rest[J[parent]]
        n = np.linalg.norm(d)
        rest_dir[name] = (d / n) if n > 1e-8 else np.array([1, 0, 0], np.float32)
    return rest_dir


# ---------- FK: 월드 회전 누적 ----------
def fk_frame(data, rest_dir, bone_len, root_pos, t):
    pos = {rj: root_pos.get(rj, np.zeros(3, np.float32)).copy()
           for rj in ROOT_JOINTS}
    world_q = {rj: np.array([0, 0, 0, 1], np.float32) for rj in ROOT_JOINTS}
    remaining = list(FK_BONES)
    for _ in range(6):
        nxt = []
        for parent, child, name in remaining:
            if parent not in pos:
                nxt.append((parent, child, name))
                continue
            local_q = data[name][t] if name in data.files \
                else np.array([0, 0, 0, 1], np.float32)
            q_parent = world_q.get(parent, np.array([0, 0, 0, 1], np.float32))
            q_world = quat_mul(q_parent, local_q)
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


# ---------- 좌표: 화면용 y 뒤집기 ----------
def flip_y(p):
    q = np.array(p, np.float64)
    q = q.copy()
    q[..., 1] *= -1.0
    return q


# ---------- 실린더를 두 점 사이에 놓는 변환 ----------
def cylinder_transform(p0, p1):
    """+z 축 기준 단위 실린더를 p0->p1 로 놓는 4x4 변환과 길이."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-8:
        return None, 0.0
    z = d / L
    # z축(0,0,1)을 z 로 돌리는 회전
    zaxis = np.array([0, 0, 1], np.float64)
    v = np.cross(zaxis, z)
    c = np.dot(zaxis, z)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1]).astype(np.float64)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], np.float64)
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (p0 + p1) / 2.0   # 중점
    return T, L


# ---------- 재사용 메시 풀 ----------
class BonePool:
    """본마다 실린더 1개, 관절마다 구체 1개를 미리 만들고 변환만 갱신."""

    def __init__(self, bones, joint_names, color_bone, color_joint,
                 r_bone, r_joint, r_bone_f, r_joint_f):
        self.bones = bones
        self.joint_names = joint_names
        self.cyls = []       # (mesh, name)
        self.base_cyl = []   # 원본 정점 (변환 전) 캐시
        self.spheres = {}    # joint -> mesh
        self.base_sph = {}   # joint -> 원본 정점

        for parent, child, name in bones:
            r = r_bone_f if _is_finger(name) else r_bone
            m = o3d.geometry.TriangleMesh.create_cylinder(
                radius=r, height=1.0, resolution=8, split=1)
            m.paint_uniform_color(color_bone)
            m.compute_vertex_normals()
            self.cyls.append([m, name])
            self.base_cyl.append(np.asarray(m.vertices).copy())

        for jn in joint_names:
            r = r_joint_f if (jn.startswith(("lhand_", "rhand_"))) else r_joint
            s = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=8)
            s.paint_uniform_color(color_joint)
            s.compute_vertex_normals()
            self.spheres[jn] = s
            self.base_sph[jn] = np.asarray(s.vertices).copy()

    def add_to(self, vis):
        for m, _ in self.cyls:
            vis.add_geometry(m)
        for s in self.spheres.values():
            vis.add_geometry(s)

    def update(self, vis, pos_dict):
        # 실린더: 부모-자식 위치로 변환
        for (m, name), base in zip(self.cyls, self.base_cyl):
            parent = child = None
            for p, c, nm in self.bones:
                if nm == name:
                    parent, child = p, c
                    break
            if parent in pos_dict and child in pos_dict:
                T, L = cylinder_transform(pos_dict[parent], pos_dict[child])
                if T is None:
                    verts = base.copy()
                    verts[:] = pos_dict.get(parent, [0, 0, 0])
                else:
                    v = base.copy()
                    v[:, 2] *= L                     # 길이 스케일 (z 높이)
                    vh = np.c_[v, np.ones(len(v))]
                    v = (T @ vh.T).T[:, :3]
                    verts = v
            else:
                verts = base.copy() * 0.0            # 숨김(원점에 뭉침)
            m.vertices = o3d.utility.Vector3dVector(verts)
            m.compute_vertex_normals()
            vis.update_geometry(m)

        # 구체: 관절 위치로 평행이동
        for jn, s in self.spheres.items():
            base = self.base_sph[jn]
            if jn in pos_dict:
                verts = base + np.asarray(pos_dict[jn], np.float64)
            else:
                verts = base * 0.0
            s.vertices = o3d.utility.Vector3dVector(verts)
            s.compute_vertex_normals()
            vis.update_geometry(s)


# ---------- FK 대조선 ----------
def fk_lineset(pos_dict):
    names = list(pos_dict.keys())
    idx = {n: i for i, n in enumerate(names)}
    pts = flip_y(np.array([pos_dict[n] for n in names], np.float32))
    lines = [[idx[p], idx[c]] for p, c, _ in FK_BONES
             if p in idx and c in idx]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(COL_FK, (len(lines), 1)).astype(np.float64))
    return ls


# ---------- 원본 관절/본 위치 dict (화면좌표, y뒤집힘) ----------
def orig_positions(kp):
    """유효한 관절만 이름->화면좌표(y뒤집음). 본은 양끝 유효할 때만 그려짐."""
    pos = {}
    for name, i in J.items():
        if _valid(kp[i]):
            p = kp[i].astype(np.float64).copy()
            p[1] *= -1.0
            pos[name] = p
    return pos


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
    return trim_seq(pick_best_clip(by_word[word]))   # 원본 좌표(y 안뒤집음)


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

    seq = load_word(word)
    T = min(len(seq), len(data["head"]))
    rest_dir = build_rest_dirs(seq)

    print(f">> 프레임 {T}개.  F=FK대조,  Space=정지,  드래그=회전, 휠=확대")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"preview3d - {word}", width=1000, height=820)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.03, 0.03, 0.06])
    opt.light_on = True
    opt.line_width = 2.0
    opt.mesh_show_back_face = True

    # 화면좌표 기준 본 리스트(원본 본: 손가락 포함 전체)
    all_joint_names = list(J.keys())
    pool = BonePool(BONES, all_joint_names,
                    COL_BONE, COL_JOINT,
                    R_BONE, R_JOINT, R_FINGER_BONE, R_FINGER_JOINT)
    pool.add_to(vis)

    # FK 대조선
    kp0 = seq[0]
    root_pos0 = {rj: kp0[J[rj]].astype(np.float32) for rj in ROOT_JOINTS}
    f_ls = fk_lineset(fk_frame(data, rest_dir, bone_len, root_pos0, 0))
    vis.add_geometry(f_ls)

    # 초기 원본
    pool.update(vis, orig_positions(kp0))

    state = {"t": 0, "cnt": 0, "paused": False, "show_fk": True}

    def toggle_fk(v):
        state["show_fk"] = not state["show_fk"]
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
        if state["cnt"] % 3 != 0:
            return False

        t = state["t"]
        kp = seq[t]

        # 원본 실린더/구체 갱신
        pool.update(vis, orig_positions(kp))

        # FK 대조선
        if state["show_fk"]:
            root_pos = {rj: kp[J[rj]].astype(np.float32) for rj in ROOT_JOINTS}
            new_f = fk_lineset(fk_frame(data, rest_dir, bone_len, root_pos, t))
            f_ls.points = new_f.points
            f_ls.lines = new_f.lines
            f_ls.colors = new_f.colors
            vis.update_geometry(f_ls)

        state["t"] = (t + 1) % T
        return False

    vis.register_animation_callback(update)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

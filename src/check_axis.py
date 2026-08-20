# check_axis.py - json 클립의 좌표가 어느 방향으로 움직이는지 확인
import os
import json
import numpy as np
from keypoint_schema_3d import openpose_to_common_3d, POSE_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON_DIR = os.path.join(ROOT, "data")
if not os.path.isdir(JSON_DIR):
    JSON_DIR = os.path.join(HERE, "data")

files = sorted(
    os.path.join(JSON_DIR, f)
    for f in os.listdir(JSON_DIR)
    if f.endswith("_keypoints.json")
)
print(f">> json {len(files)}프레임  폴더: {JSON_DIR}")

frames = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    kp, _ = openpose_to_common_3d(d["people"])
    frames.append(kp)
seq = np.stack(frames)  # (T,50,3) 원본 좌표
print(f">> {seq.shape[0]}프레임 로드됨")

rw = POSE_ORDER.index("r_wrist")
lw = POSE_ORDER.index("l_wrist")

def valid_frames(seq, j):
    return [t for t in range(len(seq)) if np.linalg.norm(seq[t, j]) > 1e-6]

for name, j in [("오른손목", rw), ("왼손목", lw)]:
    vf = valid_frames(seq, j)
    if not vf:
        print(f"{name}: 유효 프레임 없음")
        continue
    first = seq[vf[0], j]
    last = seq[vf[-1], j]
    xs = np.array([seq[t, j] for t in vf])
    print(f"\n{name}:")
    print(f"  시작 xyz: {first.round(3)}")
    print(f"  끝   xyz: {last.round(3)}")
    print(f"  변화량 : {(last - first).round(3)}  (x, y, z 각각)")
    print(f"  x 범위: [{xs[:,0].min():.3f}, {xs[:,0].max():.3f}]")
    print(f"  y 범위: [{xs[:,1].min():.3f}, {xs[:,1].max():.3f}]")
    print(f"  z 범위: [{xs[:,2].min():.3f}, {xs[:,2].max():.3f}]")


    # 한 프레임 전체 관절의 축별 범위 (깊이가 평면인지 확인)
mid = seq[len(seq)//2]  # 중간 프레임
valid = mid[np.linalg.norm(mid, axis=1) > 1e-6]
print("\n[중간 프레임 전체 관절]")
print(f"  x 범위: [{valid[:,0].min():.3f}, {valid[:,0].max():.3f}]  폭 {np.ptp(valid[:,0]):.3f}")
print(f"  y 범위: [{valid[:,1].min():.3f}, {valid[:,1].max():.3f}]  폭 {np.ptp(valid[:,1]):.3f}")
print(f"  z 범위: [{valid[:,2].min():.3f}, {valid[:,2].max():.3f}]  폭 {np.ptp(valid[:,2]):.3f}")

# 어깨-팔꿈치-손목 z값 비교 (손과 몸의 좌표계가 맞는지)
rs = POSE_ORDER.index("r_shoulder")
re = POSE_ORDER.index("r_elbow")
rw2 = POSE_ORDER.index("r_wrist")
m = seq[len(seq)//2]
print("\n[오른팔 z값 - 중간 프레임]")
print(f"  어깨 z: {m[rs,2]:.3f}")
print(f"  팔꿈치 z: {m[re,2]:.3f}")
print(f"  손목 z: {m[rw2,2]:.3f}")
print(f"  어깨 전체: {m[rs].round(3)}")
print(f"  손목 전체: {m[rw2].round(3)}")

# 여러 프레임에서 오른팔 방향(손목-어깨) 벡터 확인
print("\n[오른팔 방향벡터 (손목-어깨), 프레임별]")
for t in [0, len(seq)//4, len(seq)//2, 3*len(seq)//4, len(seq)-1]:
    sh = seq[t, rs]
    wr = seq[t, rw2]
    if np.linalg.norm(sh) < 1e-6 or np.linalg.norm(wr) < 1e-6:
        print(f"  f{t}: (유효하지 않음)")
        continue
    d = wr - sh
    print(f"  f{t}: dx={d[0]:+.3f} dy={d[1]:+.3f} dz={d[2]:+.3f}")

# ── 오른손 검지 회전 검증 (손가락 방향-only 방식 확인) ──
print("\n[오른손 검지 마디별 방향 - 프레임별]")
# rhand 인덱스: 손목29, 검지뿌리34, 검지2관절35, 검지3관절36, 검지끝37
rh = rhand_start  # 29
tip_chain = [
    ("index_01(뿌리->2관절)", 5, 6),
    ("index_02(2관절->3관절)", 6, 7),
    ("index_03(3관절->끝)", 7, 8),
]
for label, a, b in tip_chain:
    print(f"  [{label}]")
    for t in [0, len(seq)//4, len(seq)//2, 3*len(seq)//4, len(seq)-1]:
        pa, pb = seq[t, rh+a], seq[t, rh+b]
        if min(np.linalg.norm(pa), np.linalg.norm(pb)) < 1e-6:
            print(f"    f{t}: (데이터 없음)"); continue
        d = pb - pa
        d = d / (np.linalg.norm(d) + 1e-8)
        print(f"    f{t}: 방향={d.round(3)}")

# ── rest 프레임에서 손가락이 향하는 방향 ──
print("\n[rest 프레임 손가락 방향 (TPOSE_DIR 가정 검증)]")
ri2, rf2 = first_valid_frame(seq)
for hand_name, hstart in [("오른손", rhand_start), ("왼손", lhand_start)]:
    p_wrist = rf2[hstart + 0]
    p_idx_base = rf2[hstart + 5]   # 검지뿌리
    p_mid_tip = rf2[hstart + 12]   # 중지끝
    if min(np.linalg.norm(p_wrist), np.linalg.norm(p_mid_tip)) < 1e-6:
        print(f"  {hand_name}: 데이터 없음"); continue
    finger_dir = p_mid_tip - p_wrist
    finger_dir = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    print(f"  {hand_name} 손목->중지끝 방향: {finger_dir.round(3)}")

    # ============================================================
# 손가락 검증 (독립 실행 - 위쪽 변수에 의존 안 함)
# ============================================================
import numpy as np

# 손 시작 인덱스 (JOINTS 순서: pose8 + lhand21 + rhand21)
_LHAND = 8
_RHAND = 29

def _first_valid(s):
    for i, f in enumerate(s):
        if (np.linalg.norm(f, axis=1) > 1e-6).mean() > 0.3:
            return i, f
    return 0, s[0]

print("\n[오른손 검지 마디별 방향 - 프레임별]")
tip_chain = [
    ("index_01(뿌리->2관절)", 5, 6),
    ("index_02(2관절->3관절)", 6, 7),
    ("index_03(3관절->끝)", 7, 8),
]
for label, a, b in tip_chain:
    print(f"  [{label}]")
    for t in [0, len(seq)//4, len(seq)//2, 3*len(seq)//4, len(seq)-1]:
        pa, pb = seq[t, _RHAND+a], seq[t, _RHAND+b]
        if min(np.linalg.norm(pa), np.linalg.norm(pb)) < 1e-6:
            print(f"    f{t}: (데이터 없음)"); continue
        d = pb - pa
        d = d / (np.linalg.norm(d) + 1e-8)
        print(f"    f{t}: 방향={d.round(3)}")

print("\n[rest 프레임 손가락 방향 (TPOSE_DIR 가정 검증)]")
ri2, rf2 = _first_valid(seq)
print(f"  rest 프레임: f{ri2}")
for hand_name, hstart in [("오른손", _RHAND), ("왼손", _LHAND)]:
    p_wrist = rf2[hstart + 0]
    p_mid_tip = rf2[hstart + 12]   # 중지끝
    if min(np.linalg.norm(p_wrist), np.linalg.norm(p_mid_tip)) < 1e-6:
        print(f"  {hand_name}: 데이터 없음"); continue
    finger_dir = p_mid_tip - p_wrist
    finger_dir = finger_dir / (np.linalg.norm(finger_dir) + 1e-8)
    print(f"  {hand_name} 손목->중지끝 방향: {finger_dir.round(3)}")


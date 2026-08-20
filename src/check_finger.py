# check_finger.py - 손가락 방향/rest 검증 (독립 실행)
import os, json
import numpy as np
from keypoint_schema_3d import openpose_to_common_3d

JSON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
if not os.path.isdir(JSON_DIR):
    JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

files = sorted(os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR)
               if f.endswith("_keypoints.json"))
frames = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    kp, _ = openpose_to_common_3d(d["people"])
    frames.append(kp)
seq = np.stack(frames)
print(f">> {len(files)}프레임 로드  ({JSON_DIR})")

_LHAND, _RHAND = 8, 29

def first_valid(s):
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

print("\n[rest 프레임 손가락 방향]")
ri, rf = first_valid(seq)
print(f"  rest 프레임: f{ri}")
for name, hstart in [("오른손", _RHAND), ("왼손", _LHAND)]:
    p_wrist = rf[hstart + 0]
    p_mid_tip = rf[hstart + 12]
    if min(np.linalg.norm(p_wrist), np.linalg.norm(p_mid_tip)) < 1e-6:
        print(f"  {name}: 데이터 없음"); continue
    fd = p_mid_tip - p_wrist
    fd = fd / (np.linalg.norm(fd) + 1e-8)
    print(f"  {name} 손목->중지끝 방향: {fd.round(3)}")

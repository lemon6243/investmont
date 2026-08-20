# check_hand_debug.py - 가다 단어에서 hand_r 튐의 진짜 원인 추적
import numpy as np
from pos_to_rot import (
    J, BONES, BASIS_BONES, palm_normal, make_basis,
    build_rest_pose, pick_best_clip, trim_seq, _valid
)
from collections import defaultdict

Xr = np.load("dataset_all_3d/X_raw_3d.npy")
y = np.load("dataset_all_3d/y_all_3d.npy", allow_pickle=True)

by_word = defaultdict(list)
for seq, label in zip(Xr, y):
    by_word[str(label)].append(seq)

seq = trim_seq(pick_best_clip(by_word["가다"]))
rest = build_rest_pose(seq)

hstart = BASIS_BONES["hand_r"]   # 29
rw = J["r_wrist"]
rh0 = J["rhand_0"]

print("frame | 손목유효 rhand0유효 검5유효 검17유효 | 법선 | cur(hand방향)")
for t in range(0, 12):
    kp = seq[t]
    p0 = kp[hstart + 0]
    p5 = kp[hstart + 5]
    p17 = kp[hstart + 17]
    n = palm_normal(kp, hstart)
    a = kp[rw]
    b = kp[rh0]
    cur = b - a
    curn = cur / (np.linalg.norm(cur) + 1e-8)
    print(f"f{t:2d} | "
          f"{_valid(a)!s:5} {_valid(b)!s:5} {_valid(p5)!s:5} {_valid(p17)!s:5} | "
          f"n={None if n is None else n.round(3)} | cur={curn.round(3)}")


print("\n--- 대안: 손목 -> 중지뿌리(rhand_9) 방향으로 cur 잡으면? ---")
rh9 = J["rhand_9"]
for t in range(0, 12):
    kp = seq[t]
    a = kp[rw]
    b = kp[rh9]
    cur = b - a
    curn = cur / (np.linalg.norm(cur) + 1e-8)
    print(f"f{t:2d} | cur(손목->중지뿌리)={curn.round(3)}")

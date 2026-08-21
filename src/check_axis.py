import numpy as np
from keypoint_schema_3d import to_unreal, POSE_ORDER

X = np.load("dataset_all_3d/X_raw_3d.npy")
y = np.load("dataset_all_3d/y_all_3d.npy", allow_pickle=True)

R_WRIST = POSE_ORDER.index("r_wrist")   # 4
L_WRIST = POSE_ORDER.index("l_wrist")   # 7

def first_clip(word):
    idx = np.where(y == word)[0][0]
    seq = X[idx]
    valid = np.any(seq.reshape(len(seq), -1) != 0, axis=1)
    return seq[valid]

for word in ["위", "아래", "왼쪽", "오른쪽"]:
    seq = first_clip(word)
    F = len(seq)
    t0, t1 = 0, F // 2
    print("=" * 64)
    print(f"[{word}]  유효프레임 {F}")
    for label, j in [("r_wrist", R_WRIST), ("l_wrist", L_WRIST)]:
        # 원본
        rx0, ry0, rz0 = seq[t0, j]
        rx1, ry1, rz1 = seq[t1, j]
        # 언리얼 변환 후 (현재 코드)
        ue0 = to_unreal(seq[t0])[j]
        ue1 = to_unreal(seq[t1])[j]
        print(f"  {label:8s} 원본 y: {ry0:6.2f}→{ry1:6.2f}   "
              f"UE z: {ue0[2]:8.1f}→{ue1[2]:8.1f}")

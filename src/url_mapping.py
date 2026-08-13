import numpy as np, os
from keypoint_schema_3d import POSE_ORDER

XR = np.load(os.path.join("dataset_all_3d", "X_raw_3d.npy"))
frame = XR[0][0]   # 첫 클립 첫 프레임 (50,3) — 변환 전 원본

print("변환 전 원본 상체 8개 관절 (x, y, z):")
for i, name in enumerate(POSE_ORDER):
    print(f"  {name:12s} x={frame[i,0]:8.2f}  y={frame[i,1]:8.2f}  z={frame[i,2]:8.2f}")

print("\n원본 전체 50관절 범위:")
print("  x:", round(float(XR[0][0][:,0].min()),1), "~", round(float(XR[0][0][:,0].max()),1))
print("  y:", round(float(XR[0][0][:,1].min()),1), "~", round(float(XR[0][0][:,1].max()),1))
print("  z:", round(float(XR[0][0][:,2].min()),1), "~", round(float(XR[0][0][:,2].max()),1))

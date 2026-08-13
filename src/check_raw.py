import numpy as np, os
from keypoint_schema_3d import to_unreal, POSE_ORDER

XR = np.load(os.path.join("dataset_all_3d", "X_raw_3d.npy"))
clip = XR[0]          # (60,50,3)
frame = clip[0]       # 첫 프레임 (50,3)

ue = to_unreal(frame)   # (50,3) 언리얼 좌표
print("상체 8개 관절 좌표(언리얼 변환 후):")
for i, name in enumerate(POSE_ORDER):
    print(f"  {name:12s} x={ue[i,0]:8.2f}  y={ue[i,1]:8.2f}  z={ue[i,2]:8.2f}")

print("\n전체 50관절 범위:")
print("  x:", round(float(ue[:,0].min()),1), "~", round(float(ue[:,0].max()),1))
print("  y:", round(float(ue[:,1].min()),1), "~", round(float(ue[:,1].max()),1))
print("  z:", round(float(ue[:,2].min()),1), "~", round(float(ue[:,2].max()),1))

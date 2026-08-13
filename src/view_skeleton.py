import numpy as np, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keypoint_schema_3d import POSE_ORDER

XR = np.load(os.path.join("dataset_all_3d", "X_raw_3d.npy"))
Y  = np.load(os.path.join("dataset_all_3d", "y_all_3d.npy"), allow_pickle=True)

CLIP_IDX = 0
clip = XR[CLIP_IDX]   # (60,50,3)
print("재생 단어:", Y[CLIP_IDX])

R_SH, L_SH = POSE_ORDER.index("r_shoulder"), POSE_ORDER.index("l_shoulder")

def to_view(kp):
    kp = kp.copy()
    center = (kp[R_SH] + kp[L_SH]) / 2.0
    kp = kp - center
    x, y, z = kp[:,0], kp[:,1], kp[:,2]
    return np.stack([z, x, -y], axis=1)   # 앞뒤, 좌우, 위

POSE_BONES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7)]
HAND = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
LB = [(a+8,b+8) for a,b in HAND]
RB = [(a+29,b+29) for a,b in HAND]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

def draw(f):
    ax.clear()
    p = to_view(clip[f])
    ax.scatter(p[:,0], p[:,1], p[:,2], s=15, c='k')
    for a,b in POSE_BONES:
        ax.plot([p[a,0],p[b,0]],[p[a,1],p[b,1]],[p[a,2],p[b,2]], 'g-', lw=2)
    for a,b in LB+RB:
        ax.plot([p[a,0],p[b,0]],[p[a,1],p[b,1]],[p[a,2],p[b,2]], color='orange', lw=1)
    ax.set_xlim(-0.5,0.5); ax.set_ylim(-0.5,0.5); ax.set_zlim(-0.5,0.5)
    ax.set_title(f"{Y[CLIP_IDX]}  frame {f}")

ani = FuncAnimation(fig, draw, frames=clip.shape[0], interval=50, repeat=True)
plt.show()

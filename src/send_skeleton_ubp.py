# send_skeleton_udp.py
# X_raw_3d.npy에서 클립 하나를 골라 언리얼로 프레임별 관절 좌표를 UDP 송출
import numpy as np, socket, json, time, os
from keypoint_schema_3d import POSE_ORDER

DATA_DIR = "dataset_all_3d"
XR = np.load(os.path.join(DATA_DIR, "X_raw_3d.npy"))   # (N,60,50,3)
Y  = np.load(os.path.join(DATA_DIR, "y_all_3d.npy"), allow_pickle=True)

CLIP_IDX = 0                    # 재생할 클립 번호
FPS = 30
UE_IP, UE_PORT = "127.0.0.1", 7755

VIEW_SCALE = 300.0    # 화면 표시용 배율 (작으면 키우고 크면 줄이세요)

R_SH = POSE_ORDER.index("r_shoulder")
L_SH = POSE_ORDER.index("l_shoulder")


def to_unreal_v2(kp):
    """
    이 데이터셋 원본축:
      x=좌우, y=위아래(아래로 +), z=깊이(카메라 거리)
    언리얼:
      x=앞뒤, y=좌우, z=위
    매핑: UEx = z(깊이), UEy = x(좌우), UEz = -y(위아래 뒤집기)
    어깨 중앙을 원점으로 이동 후 배율 적용.
    """
    kp = kp.copy()
    center = (kp[R_SH] + kp[L_SH]) / 2.0
    kp = kp - center
    x, y, z = kp[:, 0], kp[:, 1], kp[:, 2]
    ue = np.stack([z, x, -y], axis=1) * VIEW_SCALE
    return ue.astype(np.float32)

def clamp_outliers(ue, limit=150.0):
    """몸 중심에서 limit(cm) 이상 떨어진 관절은 튄 값으로 보고 잘라냄."""
    ue = ue.copy()
    center = ue.mean(axis=0)
    d = np.linalg.norm(ue - center, axis=1)
    bad = d > limit
    # 튄 점은 중심 방향으로 limit 거리까지 끌어당김
    if bad.any():
        dir_unit = (ue[bad] - center) / (d[bad][:, None] + 1e-6)
        ue[bad] = center + dir_unit * limit
    return ue


clip = XR[CLIP_IDX]            # (60,50,3)
print("재생 단어:", Y[CLIP_IDX], " 프레임:", clip.shape[0])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:                     # 반복 재생
    for f in range(clip.shape[0]):
        ue = to_unreal_v2(clip[f])
        
        msg = json.dumps({"pts": ue.reshape(-1).tolist()})
        sock.sendto(msg.encode("utf-8"), (UE_IP, UE_PORT))
        time.sleep(1.0 / FPS)
    time.sleep(0.5)

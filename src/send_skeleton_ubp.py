# send_skeleton_udp.py
# X_raw_3d.npy에서 클립 하나를 골라 언리얼로 프레임별 관절 좌표를 UDP 송출
import numpy as np, socket, json, time, os
from keypoint_schema_3d import to_unreal

DATA_DIR = "dataset_all_3d"
XR = np.load(os.path.join(DATA_DIR, "X_raw_3d.npy"))   # (N,60,50,3)
Y  = np.load(os.path.join(DATA_DIR, "y_all_3d.npy"), allow_pickle=True)

CLIP_IDX = 0                    # 재생할 클립 번호
FPS = 30
UE_IP, UE_PORT = "127.0.0.1", 7755

clip = XR[CLIP_IDX]            # (60,50,3)
print("재생 단어:", Y[CLIP_IDX], " 프레임:", clip.shape[0])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:                     # 반복 재생
    for f in range(clip.shape[0]):
        ue = to_unreal(clip[f])            # (50,3) 언리얼 좌표(cm)
        msg = json.dumps({"pts": ue.reshape(-1).tolist()})
        sock.sendto(msg.encode("utf-8"), (UE_IP, UE_PORT))
        time.sleep(1.0 / FPS)
    time.sleep(0.5)

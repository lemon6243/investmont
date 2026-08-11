# webcam_infer.py
# 모듈 4: 웹캠 실시간 수어 인식 (연속 모드 + 트리거 모드 전환 가능)
# 흐름: 웹캠 -> MediaPipe Holistic -> 공통포맷 변환 -> 정규화 -> 60프레임 시퀀스 -> LSTM 예측 -> 화면 출력
#
# 조작키
#   m     : 모드 전환 (연속 <-> 트리거)
#   space : (트리거 모드에서) 60프레임 녹화 시작
#   c     : 현재 시퀀스 버퍼 비우기
#   q     : 종료

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.nn.utils.rnn import pack_padded_sequence

import mediapipe as mp
from keypoint_schema import mediapipe_to_common, normalize, flatten, FEATURE_DIM
from tts import make_tts
tts = make_tts("offline")   # 또는 "online"
last_spoken = None          # 마지막으로 읽은 단어 (중복 방지)


# ------------------------------------------------------------
# 0) 설정
# ------------------------------------------------------------
CKPT_PATH = os.path.join("dataset", "sign_lstm.pt")
SWAP_LR = False   # check_lr.py로 검증된 값
USE_FLIP = False  # check_lr.py로 검증된 값
CONF_THRESHOLD = 0.30  # 이 확률 미만이면 "인식 안됨"으로 표시

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# 1) 모델 정의 (train.py의 SignLSTM과 동일해야 함)
# ------------------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_cls),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]
        return self.fc(last)

# ------------------------------------------------------------
# 2) 체크포인트 로드 (학습 때 저장한 메타데이터 그대로 사용)
# ------------------------------------------------------------
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"모델을 찾을 수 없습니다: {CKPT_PATH}  (먼저 train.py를 실행하세요)")

ckpt = torch.load(CKPT_PATH, map_location=device)
SEQ_LEN = ckpt["seq_len"]
idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}

model = SignLSTM(
    ckpt["in_dim"], ckpt["hidden"], ckpt["n_classes"],
    num_layers=ckpt["num_layers"], dropout=ckpt["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("=" * 55)
print("[모듈 4] 웹캠 실시간 수어 인식 시작")
print(f"  클래스 수: {ckpt['n_classes']}  SEQ_LEN: {SEQ_LEN}")
print(f"  학습 테스트 정확도: {ckpt.get('test_acc', '?')}")
print("  m=모드전환  space=녹화(트리거)  c=버퍼비우기  q=종료")
print("=" * 55)

# ------------------------------------------------------------
# 3) 예측 함수: (T, FEATURE_DIM) 시퀀스 -> (단어, 확률)
# ------------------------------------------------------------
def predict(seq_frames):
    """seq_frames: list of (FEATURE_DIM,) numpy 배열"""
    if len(seq_frames) == 0:
        return None, 0.0
    arr = np.stack(seq_frames)  # (T, FEATURE_DIM)
    T = arr.shape[0]
    # SEQ_LEN에 맞춰 패딩(뒤쪽 0) 또는 뒤에서 SEQ_LEN개만 사용
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, FEATURE_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
        length = T
    else:
        arr = arr[-SEQ_LEN:]
        length = SEQ_LEN

    x = torch.tensor(arr[None, ...], dtype=torch.float32, device=device)  # (1, SEQ_LEN, F)
    lengths = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        logits = model(x, lengths)
        prob = torch.softmax(logits, dim=1)[0]
        conf, idx = prob.max(0)
    return idx2label[int(idx)], float(conf)

# ------------------------------------------------------------
# 4) 한 프레임 -> 공통포맷 정규화 벡터
# ------------------------------------------------------------
def frame_to_vector(results, w, h):
    common = mediapipe_to_common(results, w, h, swap_lr=SWAP_LR)  # (50,2)
    norm = normalize(common)
    return flatten(norm)  # (FEATURE_DIM,)

# ------------------------------------------------------------
# 5) 메인 루프
# ------------------------------------------------------------
mp_holistic = mp.solutions.holistic

mode = "continuous"          # "continuous" 또는 "trigger"
buffer = deque(maxlen=SEQ_LEN)   # 연속 모드용 최근 프레임 버퍼
recording = False            # 트리거 모드 녹화 중 여부
record_frames = []           # 트리거 모드 녹화 버퍼
last_label, last_conf = "-", 0.0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 다른 앱이 웹캠을 쓰고 있는지 확인하세요.")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if USE_FLIP:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        vec = frame_to_vector(results, w, h)

        # ----- 모드별 동작 -----
        if mode == "continuous":
            buffer.append(vec)
            # 버퍼가 어느 정도 차면 매 프레임 예측
            if len(buffer) >= max(10, SEQ_LEN // 2):
                label, conf = predict(list(buffer))
                if conf >= CONF_THRESHOLD:
                    last_label, last_conf = label, conf

        elif mode == "trigger":
            if recording:
                record_frames.append(vec)
                if len(record_frames) >= SEQ_LEN:
                    label, conf = predict(record_frames)
                    last_label, last_conf = label, conf
                    recording = False
                    record_frames = []

                    # ← 여기부터 추가: 예측이 확정되면 음성으로 읽기
                    if last_label != "-" and last_label != last_spoken:
                        tts.speak(last_label)
                        last_spoken = last_label

        # ----- 화면 표시 -----
        mode_txt = "CONTINUOUS" if mode == "continuous" else "TRIGGER"
        cv2.putText(image, f"MODE: {mode_txt}  (m to switch)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if mode == "trigger":
            if recording:
                status = f"REC {len(record_frames)}/{SEQ_LEN}"
                cv2.putText(image, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(image, "press SPACE to record", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # 예측 결과 (큰 글씨)
        result_txt = f"{last_label}  ({last_conf*100:.0f}%)"
        cv2.putText(image, result_txt, (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        cv2.imshow("SignBridge - Realtime (Module 4)", image)

        # ----- 키 입력 -----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "trigger" if mode == "continuous" else "continuous"
            buffer.clear()
            record_frames = []
            recording = False
            last_label, last_conf = "-", 0.0
        elif key == ord('c'):
            buffer.clear()
            record_frames = []
            recording = False
            last_label, last_conf = "-", 0.0
        elif key == ord(' ') and mode == "trigger":
            recording = True
            record_frames = []

cap.release()
cv2.destroyAllWindows()
print("[완료] 모듈 4 종료.")

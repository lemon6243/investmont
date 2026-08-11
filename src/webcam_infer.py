# webcam_infer.py
# 모듈 4: 웹캠 실시간 수어 인식 (연속 모드 + 트리거 모드 전환 가능)
# 트리거 모드에서는 단어를 누적해 문장으로 조립 -> LLM 문장화 -> TTS 음성 출력
#
# 조작키
#   m         : 모드 전환 (연속 <-> 트리거)
#   space     : (트리거 모드) 단어 1개 녹화(60프레임) -> 문장 버퍼에 추가
#   enter     : (트리거 모드) 지금까지 쌓인 단어들로 문장 완성 + 음성 출력
#   backspace : (트리거 모드) 마지막에 추가한 단어 1개 취소
#   c         : 버퍼/문장 모두 비우기
#   q         : 종료

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
from sentence_builder import build_sentence, APILLM   # ← 추가: 문장 조립 모듈

tts = make_tts("offline")   # 또는 "online"

# ← 추가: LLM 백엔드 준비 (환경변수 CLOVA_API_KEY 없으면 규칙+폴백만 작동)
sentence_llm = APILLM(model="HCX-DASH-002")


# ------------------------------------------------------------
# 0) 설정
# ------------------------------------------------------------
CKPT_PATH = os.path.join("dataset", "sign_lstm.pt")
SWAP_LR = False   # check_lr.py로 검증된 값
USE_FLIP = False  # check_lr.py로 검증된 값
CONF_THRESHOLD = 0.30  # 이 확률 미만이면 단어를 문장에 추가하지 않음

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
print("  m=모드전환  space=단어녹화  enter=문장완성  backspace=취소  c=비우기  q=종료")
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
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, FEATURE_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
        length = T
    else:
        arr = arr[-SEQ_LEN:]
        length = SEQ_LEN

    x = torch.tensor(arr[None, ...], dtype=torch.float32, device=device)
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

mode = "continuous"              # "continuous" 또는 "trigger"
buffer = deque(maxlen=SEQ_LEN)   # 연속 모드용 최근 프레임 버퍼
recording = False                # 트리거 모드 녹화 중 여부
record_frames = []               # 트리거 모드 녹화 버퍼
last_label, last_conf = "-", 0.0

sentence_words = []              # ← 추가: 트리거로 쌓은 단어들
current_sentence = ""            # ← 추가: 완성된 문장

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

                    # ← 변경: 여기서 단어를 읽지 않고, 문장 버퍼에 "추가"만 한다
                    if last_label != "-" and last_conf >= CONF_THRESHOLD:
                        sentence_words.append(last_label)
                        current_sentence = ""   # 새 단어 추가되면 이전 완성문장 초기화

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
                cv2.putText(image, "SPACE=word  ENTER=sentence  BKSP=undo", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # ← 추가: 지금까지 쌓인 단어들 표시
            words_txt = " + ".join(sentence_words) if sentence_words else "(none)"
            cv2.putText(image, f"WORDS: {words_txt}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 방금 인식한 단어 (작게)
        result_txt = f"last: {last_label} ({last_conf*100:.0f}%)"
        cv2.putText(image, result_txt, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ← 추가: 완성된 문장 (크게)
        if current_sentence:
            cv2.putText(image, current_sentence, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

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
            sentence_words = []          # ← 추가
            current_sentence = ""        # ← 추가
        elif key == ord('c'):
            buffer.clear()
            record_frames = []
            recording = False
            last_label, last_conf = "-", 0.0
            sentence_words = []          # ← 추가
            current_sentence = ""        # ← 추가
        elif key == ord(' ') and mode == "trigger":
            recording = True
            record_frames = []
        elif key in (13, 10) and mode == "trigger":   # ← 추가: Enter = 문장 완성
            if sentence_words:
                current_sentence = build_sentence(sentence_words, llm=sentence_llm)
                print(f"[문장] {sentence_words}  ->  {current_sentence}")
                tts.speak(current_sentence)      # 문장 전체를 음성으로
                sentence_words = []              # 다음 문장을 위해 비움
        elif key == 8 and mode == "trigger":          # ← 추가: Backspace = 마지막 단어 취소
            if sentence_words:
                removed = sentence_words.pop()
                print(f"[취소] {removed}")

cap.release()
cv2.destroyAllWindows()
print("[완료] 모듈 4 종료.")

# collect_webcam.py
# 목적: 웹캠으로 수어 단어를 직접 녹화·저장 (도메인 격차 해소용 데이터 수집)
# 핵심: 학습/추론과 "완전히 같은" 파이프라인으로 저장
#        (mediapipe_to_common -> normalize -> flatten, swap_lr=False, flip 없음)
#
# 저장 구조: webcam_data/<단어>/<번호>.npy   (각 파일: (SEQ_LEN, FEATURE_DIM))
#            webcam_data/<단어>/meta.json     (녹화 정보)
#
# 조작키
#   space : SEQ_LEN 프레임 녹화 시작
#   n     : 새 단어 입력 (터미널에 입력)
#   c     : 직전 녹화 취소(마지막 파일 삭제)
#   q     : 종료

import os
import cv2
import json
import numpy as np
from datetime import datetime

import mediapipe as mp
from keypoint_schema import mediapipe_to_common, normalize, flatten, FEATURE_DIM

# ------------------------------------------------------------
# 0) 설정 (webcam_infer.py와 반드시 동일해야 함)
# ------------------------------------------------------------
SEQ_LEN = 60          # 한 클립당 프레임 수 (train.py의 SEQ_LEN과 동일)
SWAP_LR = False       # check_lr.py로 검증된 값
USE_FLIP = False      # check_lr.py로 검증된 값
SAVE_ROOT = "webcam_data"

os.makedirs(SAVE_ROOT, exist_ok=True)

# ------------------------------------------------------------
# 1) 단어 폴더에 저장하는 함수
# ------------------------------------------------------------
def next_index(word_dir):
    """해당 단어 폴더에서 다음 저장 번호를 구함 (기존 파일 이어서)"""
    if not os.path.isdir(word_dir):
        return 1
    nums = []
    for f in os.listdir(word_dir):
        if f.endswith(".npy"):
            try:
                nums.append(int(os.path.splitext(f)[0]))
            except ValueError:
                pass
    return (max(nums) + 1) if nums else 1


def save_clip(word, frames):
    """frames: list of (FEATURE_DIM,) -> (SEQ_LEN, FEATURE_DIM)로 맞춰 저장.
       반환: 저장한 파일 경로"""
    word_dir = os.path.join(SAVE_ROOT, word)
    os.makedirs(word_dir, exist_ok=True)

    arr = np.stack(frames).astype(np.float32)  # (T, FEATURE_DIM)
    T = arr.shape[0]
    if T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, FEATURE_DIM), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:SEQ_LEN]

    idx = next_index(word_dir)
    path = os.path.join(word_dir, f"{idx:03d}.npy")
    np.save(path, arr)

    # 메타 기록 (누적)
    meta_path = os.path.join(word_dir, "meta.json")
    meta = {"word": word, "clips": []}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["clips"].append({
        "file": f"{idx:03d}.npy",
        "frames_recorded": T,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return path


def count_clips(word):
    """해당 단어의 현재 저장된 클립 수"""
    word_dir = os.path.join(SAVE_ROOT, word)
    if not os.path.isdir(word_dir):
        return 0
    return len([f for f in os.listdir(word_dir) if f.endswith(".npy")])


# ------------------------------------------------------------
# 2) 프레임 -> 공통포맷 정규화 벡터 (학습/추론과 동일)
# ------------------------------------------------------------
def frame_to_vector(results, w, h):
    common = mediapipe_to_common(results, w, h, swap_lr=SWAP_LR)  # (50,2)
    norm = normalize(common)
    return flatten(norm)  # (FEATURE_DIM,)


# ------------------------------------------------------------
# 3) 시작할 단어 입력
# ------------------------------------------------------------
print("=" * 55)
print("[웹캠 데이터 수집] 도메인 격차 해소용 직접 녹화 도구")
print("=" * 55)
current_word = input("녹화할 단어를 입력하세요 (예: 걷다): ").strip()
if not current_word:
    current_word = "테스트"
print(f">> 현재 단어: '{current_word}'  (기존 {count_clips(current_word)}개)")
print("  space=녹화시작  n=단어변경  c=직전취소  q=종료")

# ------------------------------------------------------------
# 4) 메인 루프
# ------------------------------------------------------------
mp_holistic = mp.solutions.holistic

recording = False
rec_frames = []
last_saved_path = None   # 직전 저장 파일 (취소용)

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

        # 녹화 중이면 프레임 벡터 누적
        if recording:
            vec = frame_to_vector(results, w, h)
            rec_frames.append(vec)
            if len(rec_frames) >= SEQ_LEN:
                last_saved_path = save_clip(current_word, rec_frames)
                print(f"  >> 저장: {last_saved_path}  "
                      f"(누적 {count_clips(current_word)}개)")
                recording = False
                rec_frames = []

        # ----- 화면 표시 -----
        cv2.putText(image, f"WORD: {current_word}  (saved: {count_clips(current_word)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if recording:
            cv2.putText(image, f"REC {len(rec_frames)}/{SEQ_LEN}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(image, "SPACE=record  n=new word  c=cancel  q=quit",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("SignBridge - Webcam Data Collect", image)

        # ----- 키 입력 -----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            rec_frames = []
        elif key == ord('n'):
            # 새 단어 입력 (터미널에서)
            cv2.destroyAllWindows()
            new_word = input("새 단어를 입력하세요: ").strip()
            if new_word:
                current_word = new_word
                print(f">> 현재 단어: '{current_word}'  (기존 {count_clips(current_word)}개)")
            recording = False
            rec_frames = []
        elif key == ord('c'):
            # 직전 저장 취소
            if last_saved_path and os.path.exists(last_saved_path):
                os.remove(last_saved_path)
                print(f"  >> 취소: {last_saved_path} 삭제")
                last_saved_path = None
            else:
                print("  >> 취소할 직전 파일이 없습니다.")

cap.release()
cv2.destroyAllWindows()
print(f"[완료] 웹캠 데이터 수집 종료. 저장 위치: {os.path.abspath(SAVE_ROOT)}")

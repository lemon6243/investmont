# -*- coding: utf-8 -*-
"""
text_to_sign.py
"타이핑 -> 수어 캐릭터 재생" 뼈대.
흐름: 한국어 입력 -> text_to_gloss -> 각 글로스의 키포인트 시퀀스 -> 이어붙여 캐릭터 재생

조작: 터미널에 문장 입력 -> 캐릭터 창에서 재생 -> 창 닫히면 다음 입력.
      (재생 중 q = 현재 재생 중단)
"""

import os
import numpy as np
import cv2

from avatar_preview import draw_frame, CANVAS, FPS   # 캐릭터 렌더링 재사용
from text_to_gloss import text_to_gloss, GlossLLM

DATA_DIR = "dataset_all"
X_PATH = os.path.join(DATA_DIR, "X_all.npy")
Y_PATH = os.path.join(DATA_DIR, "y_all.npy")

# ------------------------------------------------------------
# 보간(모션 블렌딩): 두 자세 사이를 매끄럽게 잇는 중간 프레임 생성
# ------------------------------------------------------------
TRANSITION_FRAMES = 12   # 단어 사이에 끼워 넣을 전환 프레임 수 (많을수록 느리고 부드러움)


def _ease(t):
    """ease-in-out 곡선: 0~1 입력을 부드러운 0~1로. (시작/끝 느리고 중간 빠름)"""
    return t * t * (3 - 2 * t)   # smoothstep


def interpolate(pose_a, pose_b, n_frames):
    """
    pose_a, pose_b: (100,) 두 프레임 벡터
    두 자세 사이를 n_frames개의 중간 프레임으로 채워 반환. (list of (100,))
    감지 실패(0)인 관절은 보간하지 않고 목표값(pose_b) 사용.
    """
    a = pose_a.reshape(-1, 2)   # (50,2)
    b = pose_b.reshape(-1, 2)

    # 각 관절이 양쪽 다 유효한지(0이 아닌지) 마스크
    valid_a = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    valid_b = ~((np.abs(b[:, 0]) < 1e-6) & (np.abs(b[:, 1]) < 1e-6))
    both = valid_a & valid_b     # 양쪽 다 유효할 때만 보간

    frames = []
    for i in range(1, n_frames + 1):
        t = _ease(i / (n_frames + 1))
        mid = b.copy()                       # 기본은 목표 자세
        mid[both] = a[both] * (1 - t) + b[both] * t   # 유효한 관절만 선형보간
        frames.append(mid.reshape(-1).astype(np.float32))
    return frames



def build_word_bank(X, y):
    """단어 -> 대표 시퀀스 1개 매핑. (지금은 각 단어의 첫 샘플 사용)"""
    bank = {}
    for seq, label in zip(X, y):
        label = str(label)
        if label not in bank:
            bank[label] = seq   # (SEQ_LEN, 100)
    return bank


# ------------------------------------------------------------
# 보간 + 중립자세 + 스무딩
# ------------------------------------------------------------
TRANSITION_FRAMES = 8    # 단어<->중립 전환 프레임 수
NEUTRAL_HOLD = 0         # 중립 자세를 잠깐 유지하는 프레임 수
SMOOTH_WINDOW = 6        # 스무딩 창 크기(홀수 권장: 3,5,7). 클수록 부드럽고 뭉개짐


def _ease(t):
    """ease-in-out (smoothstep)"""
    return t * t * (3 - 2 * t)


def make_neutral_pose(bank):
    """
    중립(rest) 자세 생성: 여러 단어의 '첫 프레임'을 평균.
    대개 단어 시작은 손을 아래/앞에 둔 준비자세라 평균이 자연스러운 중립이 됨.
    감지실패(0)는 평균에서 제외.
    """
    firsts = [seq[0].reshape(-1, 2) for seq in bank.values()]
    stack = np.stack(firsts)            # (W, 50, 2)
    # 0인 점은 마스킹해서 평균
    mask = ~((np.abs(stack[..., 0]) < 1e-6) & (np.abs(stack[..., 1]) < 1e-6))
    mask = mask[..., None]              # (W,50,1)
    summed = (stack * mask).sum(axis=0)
    cnt = mask.sum(axis=0)
    cnt[cnt == 0] = 1
    neutral = (summed / cnt).reshape(-1).astype(np.float32)  # (100,)
    return neutral


def interpolate(pose_a, pose_b, n_frames):
    """두 자세 사이 n_frames개 중간 프레임 (감지실패 관절은 목표값 사용)"""
    a = pose_a.reshape(-1, 2)
    b = pose_b.reshape(-1, 2)
    va = ~((np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) < 1e-6))
    vb = ~((np.abs(b[:, 0]) < 1e-6) & (np.abs(b[:, 1]) < 1e-6))
    both = va & vb
    out = []
    for i in range(1, n_frames + 1):
        t = _ease(i / (n_frames + 1))
        mid = b.copy()
        mid[both] = a[both] * (1 - t) + b[both] * t
        out.append(mid.reshape(-1).astype(np.float32))
    return out

def _frame_is_empty(frame_vec, min_valid_ratio=0.3):
    """
    한 프레임이 '비었는지' 판단.
    유효(0이 아닌) 관절 비율이 min_valid_ratio 미만이면 빈 프레임으로 봄.
    (손·팔이 거의 안 잡힌 프레임 = 캐릭터가 사라지는 프레임)
    """
    kp = frame_vec.reshape(-1, 2)
    valid = ~((np.abs(kp[:, 0]) < 1e-6) & (np.abs(kp[:, 1]) < 1e-6))
    return valid.mean() < min_valid_ratio


def trim_empty_frames(seq):
    """
    클립 앞뒤의 빈 프레임을 잘라낸 (start:end+1) 구간 반환.
    전부 비어있으면 원본 그대로 반환(안전장치).
    """
    n = len(seq)
    start = 0
    while start < n and _frame_is_empty(seq[start]):
        start += 1
    end = n - 1
    while end > start and _frame_is_empty(seq[end]):
        end -= 1
    if start >= end:          # 유효 구간이 없으면 원본 유지
        return seq
    return seq[start:end + 1]



def smooth_sequence(frames, window=SMOOTH_WINDOW):
    """
    전체 프레임 시퀀스에 시간축 이동평균 스무딩.
    frames: list of (100,) -> 같은 길이 list.
    감지실패(0)가 섞인 관절은 그 프레임에서 스무딩 제외(0 오염 방지).
    """
    if len(frames) < 3:
        return frames
    arr = np.stack(frames)              # (T, 100)
    T = arr.shape[0]
    half = window // 2
    kp = arr.reshape(T, -1, 2)          # (T, 50, 2)
    valid = ~((np.abs(kp[..., 0]) < 1e-6) & (np.abs(kp[..., 1]) < 1e-6))  # (T,50)

    out = kp.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        win = kp[lo:hi]                 # (w,50,2)
        wv = valid[lo:hi][..., None]    # (w,50,1)
        s = (win * wv).sum(axis=0)
        c = wv.sum(axis=0)
        c[c == 0] = 1
        avg = s / c                     # (50,2)
        # 현재 프레임에서 유효한 관절만 스무딩 값으로 교체
        m = valid[t]
        out[t][m] = avg[m]
    return [out[t].reshape(-1).astype(np.float32) for t in range(T)]


def build_full_sequence(gloss, bank, neutral):
    """글로스 -> [중립] 단어1 [중립] 단어2 ... (빈 프레임 제거 + 이음매 보간 + 전체 스무딩)"""
    frames = []
    prev = neutral.copy()
    for word in gloss:
        seq = bank.get(word)
        if seq is None:
            continue
        seq = trim_empty_frames(seq)          # ← 추가: 앞뒤 빈 프레임 잘라내기

        frames += interpolate(prev, neutral, TRANSITION_FRAMES)
        frames += [neutral] * NEUTRAL_HOLD
        frames += interpolate(neutral, seq[0], TRANSITION_FRAMES)
        frames += [seq[f] for f in range(len(seq))]
        prev = seq[-1]
    frames += interpolate(prev, neutral, TRANSITION_FRAMES)
    return smooth_sequence(frames)



def play_gloss(gloss, bank, neutral):
    """조립된 전체 시퀀스를 한 번에 재생."""
    if not gloss:
        print("   (재생할 단어 없음)")
        return
    full = build_full_sequence(gloss, bank, neutral)
    delay = int(1000 / FPS)
    label = "  ".join(gloss)
    for img_vec in full:
        img = draw_frame(img_vec)
        cv2.putText(img, label, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("SignBridge - Text to Sign (character)", img)
        if (cv2.waitKey(delay) & 0xFF) == ord('q'):
            return




def main():
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"{X_PATH} 없음. collect_all.py 먼저 실행.")

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True)
    bank = build_word_bank(X, y)
    neutral = make_neutral_pose(bank)   # ← 추가: 중립 자세 생성
    available_words = set(bank.keys())
    print(f">> 보유 단어 {len(available_words)}개 로드 완료")

    llm = GlossLLM()   # CLOVA_API_KEY 있으면 사용, 없으면 규칙/폴백

    print("=" * 50)
    print(" 한국어 문장을 입력하면 캐릭터가 수어로 재생합니다.")
    print(" (그냥 엔터 또는 'exit' 입력 시 종료)")
    print("=" * 50)

    while True:
        text = input("\n문장 입력> ").strip()
        if text == "" or text.lower() == "exit":
            break
        gloss = text_to_gloss(text, available_words, llm=llm)
        print(f"   글로스: {gloss}")
        if not gloss:
            print("   보유 단어로 표현 가능한 게 없습니다.")
            continue
        play_gloss(gloss, bank, neutral)   # ← neutral 추가
        cv2.destroyAllWindows()

    print(">> 종료")


if __name__ == "__main__":
    main()

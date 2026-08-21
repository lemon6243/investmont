import numpy as np

X = np.load("dataset_all_3d/X_raw_3d.npy")   # (59960, 60, 50, 3)
y = np.load("dataset_all_3d/y_all_3d.npy", allow_pickle=True)

def clip_for(word):
    idx = np.where(y == word)[0][0]
    return X[idx], idx

# 방향 단어 4개의 첫 클립을 로드
for word in ["왼쪽", "오른쪽", "위", "아래"]:
    seq, idx = clip_for(word)          # (60, 50, 3)
    # 유효 프레임만 (전부 0인 패딩 프레임 제외)
    valid = np.any(seq.reshape(60, -1) != 0, axis=1)
    seq = seq[valid]
    F = len(seq)
    print("=" * 60)
    print(f"[{word}] idx={idx}, 유효프레임={F}")

    # 각 조인트가 프레임 동안 얼마나 움직였는지 (이동량 큰 = 손)
    motion = np.linalg.norm(seq.max(0) - seq.min(0), axis=1)  # (50,)
    top = np.argsort(motion)[::-1][:6]
    print("  가장 많이 움직인 조인트 인덱스:", top.tolist())
    for j in top:
        x0, y0, z0 = seq[0, j]
        x1, y1, z1 = seq[F//2, j]     # 중간 프레임
        print(f"    joint[{j}] x: {x0:8.1f} → {x1:8.1f}   "
              f"y: {y0:8.1f} → {y1:8.1f}   z: {z0:8.1f} → {z1:8.1f}")

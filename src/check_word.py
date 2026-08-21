import numpy as np

X = np.load("dataset_all_3d/X_raw_3d.npy")          # (59960, 60, 50, 3)
y = np.load("dataset_all_3d/y_all_3d.npy", allow_pickle=True)

print("X shape:", X.shape)
print("전체 클립 수:", len(y))

# 고유 단어 목록과 개수
uniq, counts = np.unique(y, return_counts=True)
print("고유 단어 수:", len(uniq))
print("샘플 단어 30개:", list(uniq[:30]))

# 방향/움직임이 뚜렷한 단어가 있는지 검색
for kw in ["왼쪽", "오른쪽", "위", "아래", "앞", "뒤", "가다", "오다"]:
    hits = np.where(y == kw)[0]
    if len(hits):
        print(f"'{kw}' → 인덱스 {hits[:6].tolist()} (총 {len(hits)}개)")
    else:
        print(f"'{kw}' → 없음")

# 같은 단어 5각도의 3D가 실제로 동일한지 확인 (첫 단어 기준)
first_word = y[0]
idxs = np.where(y == first_word)[0][:5]
print(f"\n[{first_word}] 5각도 클립 인덱스: {idxs.tolist()}")
if len(idxs) >= 2:
    diff = np.abs(X[idxs[0]] - X[idxs[1]]).max()
    print(f"  각도0 vs 각도1 최대 좌표 차이: {diff:.4f}")
    print("  → 0에 가까우면 5각도 3D가 동일(정상), 크면 각도별로 다른 좌표계")

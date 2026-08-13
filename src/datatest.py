import os
D = "dataset_all"
for f in ["X_all.npy", "y_all.npy", "groups_all.npy", "processed_clips.json"]:
    p = os.path.join(D, f)
    if os.path.exists(p):
        os.remove(p)
        print("삭제:", f)
print("초기화 완료 - 이제 0부터 다시 쌓입니다")

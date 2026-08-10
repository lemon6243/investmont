# check_data.py
# 목적: 지금까지 X.npy에 실제로 몇 개가 쌓였고, 단어 분포가 어떤지 확인
import os, json
import numpy as np
from collections import Counter

OUT_DIR = "dataset"

X = np.load(os.path.join(OUT_DIR, "X.npy"))
y = np.load(os.path.join(OUT_DIR, "y.npy"))
print(">> 현재 저장된 총 샘플 수:", X.shape)

with open(os.path.join(OUT_DIR, "processed_clips.json"), encoding="utf-8") as f:
    processed = json.load(f)
print(">> 처리 완료로 기록된 클립 수:", len(processed))

with open(os.path.join(OUT_DIR, "label2idx.json"), encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
per_word = Counter(idx2label[int(v)] for v in y)
print(">> 단어별 샘플 수:", dict(per_word))

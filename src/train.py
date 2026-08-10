# train.py
# 목적: 저장된 .npy 데이터로 수어 단어 분류 모델(LSTM) 학습
# 핵심1: 세션(사람) 단위로 train/test 분할 -> 데이터 누수 방지
# 핵심2: 시퀀스 실제 길이를 반영(pack_padded_sequence)해 패딩 프레임 영향 제거
# 핵심3: 추론에 필요한 모든 정보를 하나의 체크포인트로 저장

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------
# 0) 설정 & 재현성
# ------------------------------------------------------------
DATA_DIR = "dataset"
EPOCHS = 60
BATCH = 32
LR = 1e-3
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> device:", device)

# ------------------------------------------------------------
# 1) 데이터 로드
# ------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X.npy"))           # (N, SEQ_LEN, FEATURE_DIM)
y = np.load(os.path.join(DATA_DIR, "y.npy"))           # (N,)
groups = np.load(os.path.join(DATA_DIR, "groups.npy"), allow_pickle=True) # (N,)
with open(os.path.join(DATA_DIR, "label2idx.json"), encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
n_classes = len(label2idx)

SEQ_LEN = X.shape[1]
FEATURE_DIM = X.shape[2]
print(">> X:", X.shape, " classes:", n_classes,
      " SEQ_LEN:", SEQ_LEN, " FEATURE_DIM:", FEATURE_DIM)

# ------------------------------------------------------------
# 1-1) 각 시퀀스의 실제 길이 계산 (넷째 개선)
#   - load_dataset에서 짧은 시퀀스는 "뒤쪽"을 전부 0으로 패딩했음.
#   - 따라서 "전부 0인 프레임"을 뒤에서부터 세어 실제 길이를 역산한다.
#   - 주의: 정규화 후에도 유효 프레임이 전부 정확히 0이 될 확률은 매우 낮으므로
#           이 역산은 실용적으로 안전하다. (완전 정밀한 길이는 build 단계에서
#           별도 저장하는 것이 이상적 — 아래 '한계' 설명 참고)
# ------------------------------------------------------------
def compute_lengths(X_arr):
    # 프레임별로 "모든 값이 0인가?"를 판단 -> 패딩 프레임 마스크
    nonzero_frame = np.any(X_arr != 0.0, axis=2)   # (N, SEQ_LEN) True=유효 프레임
    lengths = nonzero_frame.sum(axis=1).astype(np.int64)  # (N,)
    # 전부 0인 비정상 샘플이 있으면 최소 길이 1로 보정 (pack 요구사항)
    lengths = np.clip(lengths, 1, SEQ_LEN)
    return lengths

lengths = compute_lengths(X)
print(">> 실제 길이 통계: min", int(lengths.min()),
      " max", int(lengths.max()), " mean", round(float(lengths.mean()), 1))

# ------------------------------------------------------------
# 2) 세션(그룹) 단위 train/test 분할 (누수 방지)
# ------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups))
Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y[train_idx], y[test_idx]
ltr, lte = lengths[train_idx], lengths[test_idx]
print(">> train:", Xtr.shape[0], " test:", Xte.shape[0])

# ------------------------------------------------------------
# 2-1) 클래스 가중치 (불균형 대응)
# ------------------------------------------------------------
class_count = np.bincount(ytr, minlength=n_classes).astype(np.float32)
class_count[class_count == 0] = 1.0  # 0 나눗셈 방지
class_weight = class_count.sum() / (n_classes * class_count)
class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
print(">> class weight:", np.round(class_weight.cpu().numpy(), 2))

# ------------------------------------------------------------
# 3) 텐서/로더  (X, y, length 세 가지를 함께 묶는다)
# ------------------------------------------------------------
tr = TensorDataset(
    torch.tensor(Xtr, dtype=torch.float32),
    torch.tensor(ytr, dtype=torch.long),
    torch.tensor(ltr, dtype=torch.long),
)
te = TensorDataset(
    torch.tensor(Xte, dtype=torch.float32),
    torch.tensor(yte, dtype=torch.long),
    torch.tensor(lte, dtype=torch.long),
)
tr_loader = DataLoader(tr, batch_size=BATCH, shuffle=True)
te_loader = DataLoader(te, batch_size=BATCH)

# ------------------------------------------------------------
# 4) 모델: 2층 LSTM(패킹 지원) -> 실제 마지막 은닉상태 -> 분류
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
        # lengths: (B,) 각 시퀀스의 실제 길이 (CPU 텐서여야 함)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # LSTM에 패킹된 시퀀스를 넣으면 패딩 프레임은 계산에서 제외됨
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, B, hidden) -> 마지막 층의 은닉상태 사용
        last = h_n[-1]              # (B, hidden) = 실제 마지막 프레임의 은닉상태
        return self.fc(last)

model = SignLSTM(FEATURE_DIM, HIDDEN, n_classes,
                 num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss(weight=class_weight)

# ------------------------------------------------------------
# 5) 평가 함수
# ------------------------------------------------------------
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, lb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, lb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total else 0

# ------------------------------------------------------------
# 6) 학습 루프 (베스트 체크포인트 저장)
# ------------------------------------------------------------
CKPT_PATH = os.path.join(DATA_DIR, "sign_lstm.pt")
best_te_acc = -1.0

def save_checkpoint(path, te_acc):
    """추론에 필요한 모든 정보를 하나로 저장 (셋째 개선)"""
    ckpt = {
        "model_state": model.state_dict(),
        "in_dim": FEATURE_DIM,
        "hidden": HIDDEN,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "n_classes": n_classes,
        "seq_len": SEQ_LEN,
        "label2idx": label2idx,
        "idx2label": {int(k): v for k, v in idx2label.items()},
        "test_acc": te_acc,
    }
    torch.save(ckpt, path)

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0
    for xb, yb, lb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb, lb), yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        tr_acc = evaluate(tr_loader)
        te_acc = evaluate(te_loader)
        print(f"[{epoch:3d}] loss={loss_sum/len(tr_loader):.3f}  "
              f"train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

        # 테스트 정확도가 개선되면 저장
        if te_acc > best_te_acc:
            best_te_acc = te_acc
            save_checkpoint(CKPT_PATH, te_acc)
            print(f"      >> best 갱신 (test_acc={te_acc:.3f}) -> 저장")

# ------------------------------------------------------------
# 7) 마지막 상태도 별도 저장 (베스트와 비교용)
# ------------------------------------------------------------
save_checkpoint(os.path.join(DATA_DIR, "sign_lstm_last.pt"), evaluate(te_loader))
print(">> 학습 종료. best test_acc:", round(best_te_acc, 3))
print(">> 베스트 모델:", os.path.abspath(CKPT_PATH))
print(">> 마지막 모델:", os.path.abspath(os.path.join(DATA_DIR, "sign_lstm_last.pt")))

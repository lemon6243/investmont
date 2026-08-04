# train.py
# 목적: 저장된 .npy 데이터로 수어 단어 분류 모델(LSTM) 학습
# 핵심: 세션(사람) 단위로 train/test 분할 -> 데이터 누수 방지

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = "dataset"
EPOCHS = 60
BATCH = 32
LR = 1e-3
HIDDEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> device:", device)

# ------------------------------------------------------------
# 1) 데이터 로드
# ------------------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X.npy"))          # (N, 60, 100)
y = np.load(os.path.join(DATA_DIR, "y.npy"))          # (N,)
groups = np.load(os.path.join(DATA_DIR, "groups.npy"))# (N,)
with open(os.path.join(DATA_DIR, "label2idx.json"), encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}
n_classes = len(label2idx)
print(">> X:", X.shape, " classes:", n_classes)

# ------------------------------------------------------------
# 2) 세션(그룹) 단위 train/test 분할 (누수 방지)
# ------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
Xtr, Xte = X[train_idx], X[test_idx]
ytr, yte = y[train_idx], y[test_idx]
print(">> train:", Xtr.shape[0], " test:", Xte.shape[0])

# ------------------------------------------------------------
# 3) 텐서/로더
# ------------------------------------------------------------
tr = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
te = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
tr_loader = DataLoader(tr, batch_size=BATCH, shuffle=True)
te_loader = DataLoader(te, batch_size=BATCH)

# ------------------------------------------------------------
# 4) 모델: 2층 LSTM -> 마지막 은닉상태 -> 분류
# ------------------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, in_dim, hidden, n_cls):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_cls)
        )

    def forward(self, x):
        out, _ = self.lstm(x)      # (B, T, H)
        last = out[:, -1, :]       # 마지막 프레임의 은닉상태
        return self.fc(last)

model = SignLSTM(X.shape[2], HIDDEN, n_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.CrossEntropyLoss()

# ------------------------------------------------------------
# 5) 학습 루프
# ------------------------------------------------------------
def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total else 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        tr_acc = evaluate(tr_loader)
        te_acc = evaluate(te_loader)
        print(f"[{epoch:3d}] loss={loss_sum/len(tr_loader):.3f}  "
              f"train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")

# ------------------------------------------------------------
# 6) 모델 저장
# ------------------------------------------------------------
torch.save(model.state_dict(), os.path.join(DATA_DIR, "sign_lstm.pt"))
print(">> 모델 저장 완료:", os.path.join(DATA_DIR, "sign_lstm.pt"))

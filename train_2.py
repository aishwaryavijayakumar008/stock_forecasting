import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.informer_utils import (
    preprocess,
    build_sequences,
    feature_cols,
    time_split_df,
    InformerTrendReturn
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./price/"
CSV_FILES = glob.glob(os.path.join(DATA_DIR, "*_with_sentiment.csv"))

SEQ_LEN = 90
HORIZON = 30
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-4
TRAIN_SPLIT = 0.7

SAVE_DIR = "informer_regression_ckpts"
EVAL_SEQ_DIR = "eval_sequences_reg"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EVAL_SEQ_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class PriceDataset(Dataset):
    def __init__(self, X, ytrend, yreg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ytrend = torch.tensor(ytrend, dtype=torch.float32)
        self.yreg = torch.tensor(yreg, dtype=torch.float32)  # SHAPE FIX: (N,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ytrend[idx], self.yreg[idx]


# ============================================================
# INFORMER MODEL (TREND + REGRESSION)
# ============================================================

class InformerTrendReturnRegression(nn.Module):
    def __init__(self, input_dim=len(feature_cols), d_model=128):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 5000, d_model))
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        # Load your original Informer encoder
        base = InformerTrendReturn(input_dim)
        self.encoder = base.encoder

        # Add LSTM refinement
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

        # Output heads (same structure, new regression)
        self.trend_head = nn.Linear(d_model, 1)
        self.reg_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.norm_in(x)
        x = self.dropout(x)

        x = self.encoder(x)
        x, _ = self.lstm(x)

        last = x[:, -1, :]

        pred_t = torch.sigmoid(self.trend_head(last))
        pred_r = self.reg_head(last)

        return pred_t, pred_r


# ============================================================
# LOAD + BUILD DATA
# ============================================================

X_train_all, ytrend_all, yreg_all = [], [], []

print("\n=== Loading tickers and creating train/test sequences ===\n")

for file in CSV_FILES:
    ticker = os.path.basename(file).split("_")[0]
    print(f"Processing {ticker}...")

    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = preprocess(df)

    df_train, df_test = time_split_df(df, split_ratio=TRAIN_SPLIT)

    # TRAIN sequences
    X_tr, ytrend_tr, ybins_unused, pct_tr, dates_tr, bins = build_sequences(
        df_train, SEQ_LEN, HORIZON
    )
    yreg_tr = pct_tr.reshape(-1, 1)

    X_train_all.append(X_tr)
    ytrend_all.append(ytrend_tr)
    yreg_all.append(yreg_tr)

    # TEST sequences saved for evaluation
    if len(df_test) > SEQ_LEN + HORIZON:
        X_te, ytrend_te, ybins_unused2, pct_te, dates_te, _ = build_sequences(
            df_test, SEQ_LEN, HORIZON, bins=bins
        )

        out = os.path.join(EVAL_SEQ_DIR, f"{ticker}_eval.npz")
        np.savez(
            out,
            X=X_te,
            ytrend=ytrend_te,
            pct=pct_te,
            dates=dates_te
        )
        print(f"  Saved eval sequences → {out}")
    else:
        print("  Not enough test rows. Skipped.")


# MERGE TRAIN
X_train = np.concatenate(X_train_all, axis=0)
ytrend_train = np.concatenate(ytrend_all, axis=0)
yreg_train = np.concatenate(yreg_all, axis=0)

train_loader = DataLoader(
    PriceDataset(X_train, ytrend_train, yreg_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# TRAINING SETUP
# ============================================================

model = InformerTrendReturnRegression().to(DEVICE)

loss_trend_fn = nn.BCELoss()
loss_reg_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


# ============================================================
# TRAIN LOOP
# ============================================================

print("\n🚀 Training Started...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()

    losses = []
    accs = []
    maes = []

    for xb, ytb, yrb in train_loader:
        xb, ytb, yrb = xb.to(DEVICE), ytb.to(DEVICE), yrb.to(DEVICE)

        optimizer.zero_grad()
        pred_t, pred_r = model(xb)

        # losses
        lt = loss_trend_fn(pred_t, ytb)
        lr = loss_reg_fn(pred_r, yrb)

        loss = lt + lr * 0.5
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # trend accuracy
        pred_label = (pred_t > 0.5).float()
        accs.append((pred_label == ytb).float().mean().item())

        # regression MAE
        maes.append(torch.abs(pred_r - yrb).mean().item())

    scheduler.step()

    print(
        f"Epoch {epoch} "
        f"| Loss={np.mean(losses):.5f} "
        f"| TrendAcc={np.mean(accs):.4f} "
        f"| MAE={np.mean(maes):.5f}"
    )

    if epoch % 25 == 0:
        ckpt_path = f"{SAVE_DIR}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

torch.save(model.state_dict(), f"{SAVE_DIR}/final.pt")
print("\n🎉 Training Complete!\n")

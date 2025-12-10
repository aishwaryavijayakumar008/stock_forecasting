import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils.informer_utils import (
    preprocess,
    build_sequences,
    feature_cols,
    InformerTrendReturn,
    time_split_df,
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./price/"
CSV_FILES = glob.glob(os.path.join(DATA_DIR, "*_with_sentiment.csv"))

SEQ_LEN = 90
HORIZON = 30
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
NUM_BINS = 5

TRAIN_SPLIT = 0.7        # CONFIGURABLE %
RESUME_FROM_LAST = True

SAVE_DIR = "multi_ticker_informer_checkpoints"
EVAL_SEQ_DIR = "eval_sequences"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EVAL_SEQ_DIR, exist_ok=True)

# ============================================================
# DATASET CLASS
# ============================================================

class PriceDataset(Dataset):
    def __init__(self, X, ytrend, ybins):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ytrend = torch.tensor(ytrend, dtype=torch.float32)
        self.ybins = torch.tensor(ybins, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ytrend[idx], self.ybins[idx]


# ============================================================
# LOAD ALL TICKERS + SPLIT + BUILD SEQUENCES
# ============================================================

X_train_all, ytrend_train_all, ybins_train_all = [], [], []

print("\n=== Loading tickers and generating train/test sequences ===\n")

for file in CSV_FILES:
    ticker = os.path.basename(file).split("_")[0]
    print(f"Processing {ticker}...")

    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = preprocess(df)

    # -----------------------------
    # TIME SPLIT (CONFIGURABLE)
    # -----------------------------
    df_train, df_test = time_split_df(df, split_ratio=TRAIN_SPLIT)
    print(f"  Train rows = {len(df_train)}, Test rows = {len(df_test)}")

    # -----------------------------
    # Build training sequences
    # -----------------------------
    X_tr, ytrend_tr, ybins_tr, pct_tr, dates_tr, bins = build_sequences(
        df_train, SEQ_LEN, HORIZON
    )

    print(f"  Train sequences = {len(X_tr)}")

    X_train_all.append(X_tr)
    ytrend_train_all.append(ytrend_tr)
    ybins_train_all.append(ybins_tr)

    # -----------------------------
    # Build test sequences using SAME BINS
    # -----------------------------
    if len(df_test) > SEQ_LEN + HORIZON:
        X_te, ytrend_te, ybins_te, pct_te, dates_te, _ = build_sequences(
            df_test, SEQ_LEN, HORIZON, bins=bins
        )

        # Save evaluation file per ticker
        outpath = os.path.join(EVAL_SEQ_DIR, f"{ticker}_eval.npz")
        np.savez(
            outpath,
            X=X_te, ytrend=ytrend_te, ybins=ybins_te, pct=pct_te, dates=dates_te, bins=bins
        )
        print(f"  Saved eval sequences to {outpath}")

    else:
        print("  Not enough rows for test sequences. Skipping eval save.")

# Merge all tickers for training
X_train = np.concatenate(X_train_all, axis=0)
ytrend_train = np.concatenate(ytrend_train_all, axis=0)
ybins_train = np.concatenate(ybins_train_all, axis=0)

print("\nTotal training sequences:", len(X_train))

train_loader = DataLoader(
    PriceDataset(X_train, ytrend_train, ybins_train),
    batch_size=BATCH_SIZE, shuffle=True
)

# ============================================================
# MODEL SETUP
# ====================================================AMZN========

model = InformerTrendReturn(input_dim=len(feature_cols))
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_trend_fn = nn.BCELoss()
loss_bins_fn = nn.CrossEntropyLoss()

# Resume checkpoint
if RESUME_FROM_LAST:
    ckpts = sorted(glob.glob(f"{SAVE_DIR}/*.pt"))
    if ckpts:
        last = ckpts[-1]
        print("Resuming from:", last)
        model.load_state_dict(torch.load(last))

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n🚀 Training Started\n")

for epoch in range(1, EPOCHS + 1):
    losses, trend_accs, bin_accs = [], [], []

    for xb, ytb, ybb in train_loader:

        optimizer.zero_grad()
        pred_t, pred_b = model(xb)

        lt = loss_trend_fn(pred_t, ytb)
        lb = loss_bins_fn(pred_b, ybb)

        loss = lt + lb
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pred_t_label = (pred_t.detach() > 0.5).float()
        trend_accs.append((pred_t_label == ytb).float().mean().item())

        pred_b_label = torch.argmax(pred_b.detach(), dim=1)
        bin_accs.append((pred_b_label == ybb).float().mean().item())

    print(f"Epoch {epoch} | Loss={np.mean(losses):.5f} "
          f"| TrendAcc={np.mean(trend_accs):.4f} "
          f"| BinAcc={np.mean(bin_accs):.4f}")

    if epoch % 10 == 0:
        ckpt_path = f"{SAVE_DIR}/epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)

torch.save(model.state_dict(), f"{SAVE_DIR}/final.pt")
print("\n🎉 Training Complete!")

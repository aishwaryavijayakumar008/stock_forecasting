import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.informer_utils import InformerTrendReturn, feature_cols

EVAL_DIR = "eval_sequences"
MODEL_PATH = "multi_ticker_informer_checkpoints/final.pt"
OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 90

# Load model
model = InformerTrendReturn(input_dim=len(feature_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

eval_files = [f for f in os.listdir(EVAL_DIR) if f.endswith(".npz")]

for file in eval_files:
    ticker = file.split("_")[0]
    print(f"\nEvaluating {ticker}")

    data = np.load(os.path.join(EVAL_DIR, file), allow_pickle=True)

    X = data["X"]
    ytrend = data["ytrend"]
    ybins = data["ybins"]
    pct = data["pct"]
    dates = data["dates"]

    preds_trend, preds_bin = [], []

    for i in range(len(X)):
        xt = torch.tensor(X[i:i+1], dtype=torch.float32)
        pt, pb = model(xt)

        preds_trend.append(pt.item())
        preds_bin.append(torch.argmax(pb).item())

    preds_trend = np.array(preds_trend)
    preds_bin = np.array(preds_bin)

    # TREND METRICS
    y_true_bin = ytrend.reshape(-1)
    y_pred_bin = (preds_trend > 0.5).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    print(f"  Trend Accuracy = {acc:.4f}")
    print(f"  Trend F1       = {f1:.4f}")

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{ticker}_eval_output.csv")
    pd.DataFrame({
        "date": dates,
        "pct_return": pct,
        "true_trend": y_true_bin,
        "pred_trend": preds_trend,
        "pred_trend_label": y_pred_bin,
        "true_bin": ybins,
        "pred_bin": preds_bin
    }).to_csv(out_csv, index=False)

    print(f"  Saved CSV: {out_csv}")

    # Plot trend predictions
    plt.figure(figsize=(12,4))
    plt.plot(preds_trend, label="Predicted Trend (Prob)")
    plt.plot(y_true_bin, label="True Trend")
    plt.legend()
    plt.title(f"{ticker} Trend Prediction Timeline")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_trend_plot.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{ticker} Trend Confusion Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_trend_cm.png"))
    plt.close()

print("\n🎉 Evaluation Complete!")

import os
import glob
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

from utils.informer_utils import feature_cols, InformerTrendReturn


EVAL_DIR = "eval_sequences"
SAVE_MODEL_PATH = "/home/gl-961/ASH/python_examples/stock_forecasting/checkpoints_temp/epoch_160.pt"

OUT_DIR = "evaluation_results"
TREND_DIR = f"{OUT_DIR}/TREND"
RETURN_DIR = f"{OUT_DIR}/RETURN"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)

os.makedirs(f"{TREND_DIR}/plots", exist_ok=True)
os.makedirs(f"{RETURN_DIR}/plots", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = InformerTrendReturn(input_dim=len(feature_cols), num_bins=5).to(DEVICE)
model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
model.eval()



def expected_return_from_bins(probs, bins):
    centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    centers = torch.tensor(centers, dtype=torch.float32).to(DEVICE)
    return (probs * centers).sum(dim=1)


def plot_confusion(cm, path, title):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down","Up"],
                yticklabels=["Down","Up"])
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_roc(y_true, y_scores, path, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    rauc = auc(fpr, tpr)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"AUC={rauc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_pr(y_true, y_scores, path, title):
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_line(true, pred, path, title):
    plt.figure(figsize=(10,5))
    plt.plot(true, label="True", linewidth=1)
    plt.plot(pred, label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_scatter(true, pred, path, title):
    plt.figure(figsize=(7,6))
    plt.scatter(true, pred, alpha=0.4)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_residuals(true, pred, path, title):
    plt.figure(figsize=(7,6))
    sns.histplot(true - pred, bins=50, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_rolling_error(true, pred, path, title, window=50):
    errors = np.abs(true - pred)
    rolling = pd.Series(errors).rolling(window).mean()
    plt.figure(figsize=(10,5))
    plt.plot(rolling)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"Rolling MAE ({window})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_distribution(true, pred, path, title):
    plt.figure(figsize=(7,6))
    sns.kdeplot(true, fill=True, label="True")
    sns.kdeplot(pred, fill=True, label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_heatmap(true, pred, path, title, bins=5):
    df = pd.DataFrame({"true": true, "pred": pred})
    df["true_bin"] = pd.qcut(df["true"], bins, duplicates="drop")
    df["pred_bin"] = pd.qcut(df["pred"], bins, duplicates="drop")
    heat = pd.crosstab(df["true_bin"], df["pred_bin"])

    plt.figure(figsize=(10,8))
    sns.heatmap(heat, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def evaluate(folder, prefix):

    files = glob.glob(os.path.join(folder, "*.npz"))
    if not files:
        return

    true_trend, pred_trend, prob_trend = [], [], []
    true_ret, pred_ret = [], []

    for f in files:
        d = np.load(f, allow_pickle=True)

        X = torch.tensor(d["X"], dtype=torch.float32).to(DEVICE)
        ytrend = d["ytrend"].reshape(-1)
        pct = d["pct"]
        bins = d["bins"]

        with torch.no_grad():
            pred_t, pred_b = model(X)
            prob = pred_t.cpu().numpy().flatten()
            pred_label = (prob > 0.5).astype(int)
            probs_bins = torch.softmax(pred_b, dim=1)
            pred_returns = expected_return_from_bins(probs_bins, bins).cpu().numpy()

        true_trend.extend(ytrend)
        pred_trend.extend(pred_label)
        prob_trend.extend(prob)

        true_ret.extend(pct)
        pred_ret.extend(pred_returns)

    true_trend = np.array(true_trend)
    pred_trend = np.array(pred_trend)
    prob_trend = np.array(prob_trend)
    true_ret = np.array(true_ret)
    pred_ret = np.array(pred_ret)


    cm = confusion_matrix(true_trend, pred_trend, labels=[0,1])
    plot_confusion(cm, f"{TREND_DIR}/plots/{prefix}_trend_confusion.png",
                   f"Trend Confusion Matrix ({prefix})")

    plot_roc(true_trend, prob_trend,
             f"{TREND_DIR}/plots/{prefix}_trend_roc.png",
             f"ROC Curve ({prefix})")

    plot_pr(true_trend, prob_trend,
            f"{TREND_DIR}/plots/{prefix}_trend_pr.png",
            f"Precision-Recall Curve ({prefix})")

    trend_metrics = {
        "accuracy": accuracy_score(true_trend, pred_trend),
        "precision": precision_score(true_trend, pred_trend, zero_division=0),
        "recall": recall_score(true_trend, pred_trend, zero_division=0),
        "f1": f1_score(true_trend, pred_trend, zero_division=0)
    }

    pd.DataFrame([trend_metrics]).to_csv(
        f"{TREND_DIR}/{prefix}_trend_metrics.csv", index=False
    )


    plot_line(true_ret, pred_ret,
              f"{RETURN_DIR}/plots/{prefix}_return_line.png",
              f"True vs Predicted Returns ({prefix})")

    plot_scatter(true_ret, pred_ret,
                 f"{RETURN_DIR}/plots/{prefix}_return_scatter.png",
                 f"Scatter: True vs Predicted ({prefix})")

    plot_residuals(true_ret, pred_ret,
                   f"{RETURN_DIR}/plots/{prefix}_return_residuals.png",
                   f"Residual Distribution ({prefix})")

    plot_rolling_error(true_ret, pred_ret,
                       f"{RETURN_DIR}/plots/{prefix}_return_rolling_mae.png",
                       f"Rolling MAE ({prefix})")

    plot_distribution(true_ret, pred_ret,
                      f"{RETURN_DIR}/plots/{prefix}_return_distribution.png",
                      f"Return Distribution Comparison ({prefix})")

    plot_heatmap(true_ret, pred_ret,
                 f"{RETURN_DIR}/plots/{prefix}_return_heatmap.png",
                 f"Regression Heatmap ({prefix})")

    directional_accuracy = np.mean(
        np.sign(true_ret) == np.sign(pred_ret)
    )

    ret_metrics = {
        "mae": mean_absolute_error(true_ret, pred_ret),
        "rmse": np.sqrt(mean_squared_error(true_ret, pred_ret)),
        "r2": r2_score(true_ret, pred_ret),
        "directional_accuracy": directional_accuracy
    }

    pd.DataFrame([ret_metrics]).to_csv(
        f"{RETURN_DIR}/{prefix}_return_metrics.csv", index=False
    )

    mag_bins = pd.qcut(np.abs(true_ret), q=3, labels=["Small", "Medium", "Large"])
    mag_df = pd.DataFrame({
        "magnitude": mag_bins,
        "abs_error": np.abs(true_ret - pred_ret)
    })

    mag_summary = mag_df.groupby("magnitude").mean()
    mag_summary.to_csv(
        f"{RETURN_DIR}/{prefix}_return_error_by_magnitude.csv"
    )


evaluate(EVAL_DIR, "EVAL")
print("\n All evaluation complete. Check evaluation_results/\n")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# PREPROCESSING HELPERS (Shared)
# ============================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period).mean()
    avg_loss = loss.ewm(alpha=1/period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal, macd - signal

def add_time_features(df):
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    return df

def smooth_close(series):
    return series.ewm(alpha=0.15).mean()

def compute_log_returns(series):
    return np.log(series / series.shift(1))

def compute_atr(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def scale_columns(df, cols):
    for col in cols:
        m = df[col].mean()
        s = df[col].std() + 1e-9
        df[col] = (df[col] - m) / s
    return df


# ============================================================
# MASTER PREPROCESS FUNCTION (TRAIN + EVAL)
# ============================================================

def preprocess(df):
    df = df.copy()

    # Fix numeric columns
    price_cols = ["open","high","low","close","volume"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=price_cols)

    df = add_time_features(df)
    df["close_smooth"] = smooth_close(df["close"])
    df["log_return"] = compute_log_returns(df["close"])
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["sma30"] = df["close"].rolling(30).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["atr"] = compute_atr(df)
    df["vol20"] = df["log_return"].rolling(20).std()
    df["vol50"] = df["log_return"].rolling(50).mean()
    df["volume_norm"] = np.log(df["volume"] + 1)

    df = df.dropna().reset_index(drop=True)

    scale_cols = [
        "rsi","macd","macd_signal","macd_hist",
        "log_return","sma30","sma50","sma200",
        "atr","vol20","vol50"
    ]
    df = scale_columns(df, scale_cols)

    df["avg_sentiment"] = (df["avg_sentiment"] - df["avg_sentiment"].mean()) / \
                          (df["avg_sentiment"].std() + 1e-9)

    return df


# ============================================================
# FEATURE LIST
# ============================================================

feature_cols = [
    "open","high","low","close_smooth",
    "volume_norm","log_return",
    "rsi","macd","macd_signal","macd_hist",
    "sma30","sma50","sma200",
    "atr","vol20","vol50",
    "dow_sin","dow_cos","month_sin","month_cos",
    "avg_sentiment","news_flag"
]


# ============================================================
# BUILD TRAIN OR TEST SEQUENCES
# ============================================================

def build_sequences(df, seq_len, horizon, bins=None):
    """
    If bins is None → training mode → compute bins from pct returns
    If bins is provided → evaluation mode → reuse same bins
    """
    X_list, ytrend_list, ybins_list, pct_list, dates_list = [], [], [], [], []
    close_arr = df["close"].values

    pct_all = []

    # First pass to compute all pct returns
    for i in range(len(df) - seq_len - horizon):
        now = close_arr[i + seq_len]
        future = close_arr[i + seq_len + horizon]
        pct = (future - now) / now
        pct_all.append(pct)

    pct_all = np.array(pct_all, dtype=np.float32)

    # Training mode: compute bins
    if bins is None:
        bins = np.quantile(pct_all, [0,0.2,0.4,0.6,0.8,1.0])

    # Build sequences
    idx = 0
    for i in range(len(df) - seq_len - horizon):

        X_list.append(df[feature_cols].iloc[i:i+seq_len].values)
        pct = pct_all[idx]
        idx += 1

        ytrend_list.append(1 if pct > 0 else 0)

        bin_id = np.digitize(pct, bins) - 1
        if bin_id == len(bins) - 1:
            bin_id -= 1

        ybins_list.append(bin_id)
        pct_list.append(pct)
        dates_list.append(df["date"].iloc[i + seq_len + horizon])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(ytrend_list, dtype=np.float32).reshape(-1,1),
        np.array(ybins_list, dtype=np.int64),
        np.array(pct_list, dtype=np.float32),
        np.array(dates_list),
        bins
    )


# ============================================================
# INFORMER MODEL
# ============================================================

class ProbSparseAttention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(Q.size(-1))
        top_k = max(1, int(scores.size(-1)*0.02))
        vals, idx = torch.topk(scores, top_k, dim=-1)
        sparse = torch.full_like(scores, -1e9)
        sparse.scatter_(-1, idx, vals)
        attn = torch.softmax(sparse, dim=-1)
        return torch.matmul(attn, V)

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.attn = ProbSparseAttention()
        self.ff = nn.Sequential(nn.Linear(d_model,256), nn.ReLU(), nn.Linear(256,d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x,x,x))
        x = self.norm2(x + self.ff(x))
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model=128, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model) for _ in range(layers)])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class InformerTrendReturn(nn.Module):
    def __init__(self, input_dim=len(feature_cols), d_model=128, num_bins=5):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1,5000,d_model))
        self.encoder = InformerEncoder(d_model, layers=3)
        self.trend_head = nn.Linear(d_model, 1)
        self.return_bins = nn.Linear(d_model, num_bins)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        last = x[:, -1, :]
        return torch.sigmoid(self.trend_head(last)), self.return_bins(last)


def time_split_df(df, split_ratio=0.8):
    """
    Splits a DF into train and test by time order.

    split_ratio = fraction of rows used for training.
    Example:
        0.8 = 80% train, 20% test
    """
    n = len(df)
    split = int(n * split_ratio)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_test = df.iloc[split:].reset_index(drop=True)
    return df_train, df_test

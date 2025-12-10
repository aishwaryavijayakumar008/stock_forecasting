import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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




feature_cols = [
    "open","high","low","close_smooth",
    "volume_norm","log_return",
    "rsi","macd","macd_signal","macd_hist",
    "sma30","sma50","sma200",
    "atr","vol20","vol50",
    "dow_sin","dow_cos","month_sin","month_cos",
    "avg_sentiment","news_flag"
]


def build_sequences(df, seq_len, horizon, bins=None):
    X_list, ytrend_list, ybins_list, pct_list, dates_list = [], [], [], [], []
    close_arr = df["close"].values

    pct_all = []


    for i in range(len(df) - seq_len - horizon):
        now = close_arr[i + seq_len]
        future = close_arr[i + seq_len + horizon]
        pct = (future - now) / now
        pct_all.append(pct)

    pct_all = np.array(pct_all, dtype=np.float32)


    if bins is None:
        bins = np.quantile(pct_all, [0,0.2,0.4,0.6,0.8,1.0])

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

class ProbSparseAttention(nn.Module):
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1))

        # Keep only top 2% scores
        top_k = max(1, int(scores.size(-1) * 0.02))
        vals, idx = torch.topk(scores, top_k, dim=-1)

        sparse_scores = torch.full_like(scores, -1e9)
        sparse_scores.scatter_(-1, idx, vals)

        attn = torch.softmax(sparse_scores, dim=-1)
        return torch.matmul(attn, V)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.attn = ProbSparseAttention()
        self.dropout = nn.Dropout(0.15)                
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        att = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(att))

        # Feed-forward block with dropout
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))

        return x


class InformerEncoder(nn.Module):
    def __init__(self, d_model=128, layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [InformerEncoderLayer(d_model) for _ in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InformerTrendReturn(nn.Module):
    def __init__(self, input_dim=len(feature_cols), d_model=128, num_bins=5):
        super().__init__()

        self.embed = nn.Linear(input_dim, d_model)

        # NEW: Add LayerNorm on embeddings
        self.norm_in = nn.LayerNorm(d_model)

        # Positional encoding
        self.pos = nn.Parameter(torch.randn(1, 5000, d_model))

        # Encoder with dropout
        self.encoder = InformerEncoder(d_model=d_model, layers=3)

        # NEW: Dropout before heads
        self.dropout = nn.Dropout(0.15)

        # Output heads (unchanged)
        self.trend_head = nn.Linear(d_model, 1)
        self.return_bins = nn.Linear(d_model, num_bins)

    def forward(self, x):

        # NEW: Noise injection (small, improves generalization)
        x = x + torch.randn_like(x) * 0.01

        # Embed + positional encoding + normalization
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.norm_in(x)

        # Transformer encoder
        x = self.encoder(x)

        # Extract last token
        last = x[:, -1, :]

        # NEW: Dropout before classification heads
        last = self.dropout(last)

        # Outputs
        trend = torch.sigmoid(self.trend_head(last))
        bins = self.return_bins(last)

        return trend, bins

def time_split_df(df, split_ratio=0.8):
    n = len(df)
    split = int(n * split_ratio)
    df_train = df.iloc[:split].reset_index(drop=True)
    df_test = df.iloc[split:].reset_index(drop=True)
    return df_train, df_test
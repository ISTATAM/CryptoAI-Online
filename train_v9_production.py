import json
import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# CryptoAI V9
# Production Model Builder
#
# 只建立目前要進行 Paper Trading 的：
# 1D / AI >= 70%
#
# 使用 V7 固定資料集
# Target 與 V8 完全相同：
# signal t close
# entry  = t+1 close
# exit   = t+2 close
# ============================================================

TRAINING_FILE = Path("training/training_1d.csv")
OUTPUT_DIR = Path("v9_model")

FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "rsi_14",
    "close_ema10_ratio",
    "close_ema20_ratio",
    "close_ema50_ratio",
    "ema10_ema20_ratio",
    "ema20_ema50_ratio",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "atr_pct",
    "volatility_6",
    "volatility_24",
    "volume_ratio_20",
    "volume_change",
    "candle_return",
    "high_low_range",
    "taker_buy_ratio",
    "hour",
    "day_of_week",
]

INTERVAL = "1d"
SIGNAL_THRESHOLD = 0.70

MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 42,
}


def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            sha.update(block)

    return sha.hexdigest()


def load_data():

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"找不到固定資料集：{TRAINING_FILE}"
        )

    df = pd.read_csv(TRAINING_FILE)

    required = FEATURES + [
        "symbol",
        "open_time",
        "close",
        "quote_volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"固定資料缺少欄位：{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    numeric_columns = FEATURES + [
        "close",
        "quote_volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
        )
        .sort_values(
            ["symbol", "open_time"]
        )
        .reset_index(drop=True)
    )

    return df


def rebuild_v8_target(df):

    group = df.groupby("symbol")

    df["entry_time"] = (
        group["open_time"].shift(-1)
    )

    df["entry_close"] = (
        group["close"].shift(-1)
    )

    df["exit_time"] = (
        group["open_time"].shift(-2)
    )

    df["exit_close"] = (
        group["close"].shift(-2)
    )

    expected_entry = (
        df["open_time"]
        + pd.Timedelta(days=1)
    )

    expected_exit = (
        df["open_time"]
        + pd.Timedelta(days=2)
    )

    valid = (
        (df["entry_time"] == expected_entry)
        & (df["exit_time"] == expected_exit)
        & df["entry_close"].notna()
        & df["exit_close"].notna()
    )

    df = df[valid].copy()

    df["tradable_return"] = (
        df["exit_close"]
        / df["entry_close"]
        - 1
    )

    df["tradable_target"] = (
        df["tradable_return"] > 0
    ).astype(int)

    return df


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CryptoAI V9 Production Model")
    print("=" * 65)

    dataset_hash = sha256_file(
        TRAINING_FILE
    )

    print()
    print("固定資料 SHA256：")
    print(dataset_hash)

    df = load_data()

    df = rebuild_v8_target(df)

    print()
    print(f"訓練資料：{len(df):,}")
    print(
        f"幣種："
        f"{df['symbol'].nunique()}"
    )
    print(
        "日期："
        f"{df['open_time'].min()} "
        "→ "
        f"{df['open_time'].max()}"
    )
    print(
        "Tradable 上漲比例："
        f"{df['tradable_target'].mean():.2%}"
    )

    # ========================================================
    # Production model
    #
    # Paper Trading 已經是未來資料，
    # 因此這裡可以使用固定資料集內所有可用歷史資料訓練。
    # ========================================================

    model = HistGradientBoostingClassifier(
        **MODEL_PARAMETERS
    )

    model.fit(
        df[FEATURES],
        df["tradable_target"],
    )

    model_path = (
        OUTPUT_DIR
        / "cryptoai_v9_1d.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "model": model,
                "features": FEATURES,
                "interval": INTERVAL,
                "signal_threshold":
                    SIGNAL_THRESHOLD,
                "dataset_sha256":
                    dataset_hash,
                "model_parameters":
                    MODEL_PARAMETERS,
            },
            f,
        )

    metadata = {
        "version": "V9",
        "purpose": "forward paper trading",
        "interval": INTERVAL,
        "signal_threshold":
            SIGNAL_THRESHOLD,
        "training_rows":
            int(len(df)),
        "symbols":
            int(df["symbol"].nunique()),
        "training_start":
            str(df["open_time"].min()),
        "training_end":
            str(df["open_time"].max()),
        "positive_ratio":
            float(
                df["tradable_target"].mean()
            ),
        "dataset_sha256":
            dataset_hash,
        "features":
            FEATURES,
        "model_parameters":
            MODEL_PARAMETERS,
        "execution_rule": {
            "signal": "completed t candle",
            "entry": "t+1 close",
            "exit": "t+2 close",
        },
    }

    with (
        OUTPUT_DIR
        / "model_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    model_hash = sha256_file(
        model_path
    )

    with (
        OUTPUT_DIR
        / "model_sha256.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{model_hash}  "
            "cryptoai_v9_1d.pkl\n"
        )

    print()
    print("正式模型建立完成")
    print(
        f"模型：{model_path}"
    )
    print(
        f"模型 SHA256："
        f"{model_hash}"
    )
    print()
    print(
        "Paper Trading 門檻："
        "AI >= 70%"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()

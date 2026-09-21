
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CryptoAI Online V3
# AI 特徵工程
# ============================================================

DATA_FOLDER = Path("history_data")

OUTPUT_FOLDER = Path("training")

INTERVALS = ["15m", "4h", "1d"]

EXCLUDED_BASE_ASSETS = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP",
    "DAI", "USD1", "PYUSD", "RLUSD", "USDD",
    "EUR", "TRY", "BRL", "GBP", "AUD", "JPY"
}

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


def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = average_gain / average_loss

    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50
    )

    return rsi


def build_symbol_features(filepath, interval):

    symbol = filepath.name.replace(
        f"_{interval}.csv",
        ""
    )

    base_asset = symbol.removesuffix("USDT")

    if base_asset in EXCLUDED_BASE_ASSETS:

        print(
            f"  SKIP {symbol}: 穩定幣或法幣"
        )

        return None

    df = pd.read_csv(filepath)

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True
    )

    df = df.sort_values(
        "open_time"
    ).drop_duplicates(
        "open_time"
    ).reset_index(drop=True)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    returns = close.pct_change(
        fill_method=None
    )

    # --------------------------------------------------------
    # 歷史報酬
    # --------------------------------------------------------

    for period in [1, 3, 6, 12, 24]:

        df[f"return_{period}"] = (
            close.pct_change(
                periods=period,
                fill_method=None
            )
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = calculate_rsi(
        close
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema10 = close.ewm(
        span=10,
        adjust=False
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["close_ema10_ratio"] = (
        close / ema10 - 1
    )

    df["close_ema20_ratio"] = (
        close / ema20 - 1
    )

    df["close_ema50_ratio"] = (
        close / ema50 - 1
    )

    df["ema10_ema20_ratio"] = (
        ema10 / ema20 - 1
    )

    df["ema20_ema50_ratio"] = (
        ema20 / ema50 - 1
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_pct"] = (
        macd / close
    )

    df["macd_signal_pct"] = (
        macd_signal / close
    )

    df["macd_hist_pct"] = (
        (macd - macd_signal) / close
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(
        14
    ).mean()

    df["atr_pct"] = (
        atr / close
    )

    # --------------------------------------------------------
    # 波動率
    # --------------------------------------------------------

    df["volatility_6"] = (
        returns.rolling(6).std()
    )

    df["volatility_24"] = (
        returns.rolling(24).std()
    )

    # --------------------------------------------------------
    # 成交量
    # --------------------------------------------------------

    average_volume = volume.rolling(
        20
    ).mean()

    df["volume_ratio_20"] = (
        volume / average_volume
    )

    df["volume_change"] = (
        volume.pct_change(
            fill_method=None
        )
    )

    # --------------------------------------------------------
    # K棒
    # --------------------------------------------------------

    df["candle_return"] = (
        close / open_price - 1
    )

    df["high_low_range"] = (
        (high - low) / open_price
    )

    # --------------------------------------------------------
    # 主動買方成交比例
    # --------------------------------------------------------

    df["taker_buy_ratio"] = (
        df["taker_buy_volume"]
        / volume.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 時間
    # --------------------------------------------------------

    df["hour"] = (
        df["open_time"].dt.hour
    )

    df["day_of_week"] = (
        df["open_time"].dt.dayofweek
    )

    # --------------------------------------------------------
    # 預測目標：下一根 K 線
    # --------------------------------------------------------

    df["future_close"] = (
        close.shift(-1)
    )

    df["future_return"] = (
        df["future_close"] / close - 1
    )

    df["target"] = (
        df["future_return"] > 0
    ).astype(int)

    # 避免把最後一根尚無答案的 K 線拿來訓練
    df = df[
        df["future_close"].notna()
    ].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + [
            "future_return",
            "target",
        ]
    )

    if df.empty:
        return None

    df["symbol"] = symbol

    df["interval"] = interval

    columns = [
        "symbol",
        "interval",
        "open_time",
        "close",
        "quote_volume",
    ] + FEATURES + [
        "future_close",
        "future_return",
        "target",
    ]

    return df[columns]


def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("CryptoAI Online V3")
    print("AI 特徵工程")
    print("=" * 65)

    summary = []

    for interval in INTERVALS:

        print()
        print(
            f"開始處理週期：{interval}"
        )

        folder = (
            DATA_FOLDER / interval
        )

        files = sorted(
            folder.glob(
                f"*_{interval}.csv"
            )
        )

        datasets = []

        for index, filepath in enumerate(
            files,
            start=1
        ):

            print(
                f"[{index}/{len(files)}] "
                f"{filepath.name}",
                flush=True
            )

            result = build_symbol_features(
                filepath,
                interval
            )

            if result is None:
                continue

            datasets.append(result)

        if not datasets:

            print(
                f"沒有可用的 {interval} 訓練資料"
            )

            continue

        combined = pd.concat(
            datasets,
            ignore_index=True
        )

        combined = combined.sort_values(
            ["open_time", "symbol"]
        ).reset_index(drop=True)

        output_file = (
            OUTPUT_FOLDER
            / f"training_{interval}.csv"
        )

        combined.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        up_count = int(
            combined["target"].sum()
        )

        down_count = (
            len(combined) - up_count
        )

        print()
        print(
            f"完成：{interval}"
        )

        print(
            f"幣種："
            f"{combined['symbol'].nunique()}"
        )

        print(
            f"訓練筆數：{len(combined):,}"
        )

        print(
            f"上漲：{up_count:,}"
        )

        print(
            f"下跌或持平：{down_count:,}"
        )

        print(
            f"儲存：{output_file}"
        )

        summary.append({
            "interval": interval,
            "symbols": int(
                combined["symbol"].nunique()
            ),
            "rows": len(combined),
            "up_count": up_count,
            "down_count": down_count,
            "first_time": str(
                combined["open_time"].min()
            ),
            "last_time": str(
                combined["open_time"].max()
            ),
        })

    summary_file = (
        OUTPUT_FOLDER
        / "training_summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 65)
    print("全部特徵工程完成")
    print("=" * 65)

    for item in summary:

        print(
            f"{item['interval']}: "
            f"{item['symbols']} 幣，"
            f"{item['rows']:,} 筆"
        )

    if not summary:

        raise RuntimeError(
            "所有週期都沒有產生訓練資料"
        )


if __name__ == "__main__":
    main()

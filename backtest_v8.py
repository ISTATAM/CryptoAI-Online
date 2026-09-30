import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# CryptoAI Online V8
# Tradable Target / 可交易目標嚴格驗證版
# ============================================================

TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("v8_reports")

INTERVALS = ["15m", "4h", "1d"]

FEATURES = [
    "return_1", "return_3", "return_6", "return_12", "return_24",
    "rsi_14",
    "close_ema10_ratio", "close_ema20_ratio", "close_ema50_ratio",
    "ema10_ema20_ratio", "ema20_ema50_ratio",
    "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "atr_pct",
    "volatility_6", "volatility_24",
    "volume_ratio_20", "volume_change",
    "candle_return", "high_low_range",
    "taker_buy_ratio",
    "hour", "day_of_week",
]

INITIAL_CAPITAL = 100.0
MAX_POSITIONS = 5
POSITION_FRACTION = 0.20
MIN_ORDER_USDT = 5.0

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
COSTS = [0.002, 0.003, 0.005]

MIN_VOLUME_24H = 1_000_000

INITIAL_TRAIN_RATIO = 0.50
TEST_FOLDS = 3

CONFIG = {
    "15m": {
        "duration": pd.Timedelta(minutes=15),
        "volume_bars": 96,
    },
    "4h": {
        "duration": pd.Timedelta(hours=4),
        "volume_bars": 6,
    },
    "1d": {
        "duration": pd.Timedelta(days=1),
        "volume_bars": 1,
    },
}


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def load_and_rebuild_target(interval):
    path = TRAINING_DIR / f"training_{interval}.csv"

    if not path.exists():
        raise FileNotFoundError(f"找不到固定資料：{path}")

    df = pd.read_csv(path)

    required = FEATURES + [
        "symbol",
        "open_time",
        "close",
        "quote_volume",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{interval} 缺少欄位：{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True
    )

    numeric_columns = FEATURES + [
        "close",
        "quote_volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values(["symbol", "open_time"])
        .reset_index(drop=True)
    )

    duration = CONFIG[interval]["duration"]
    volume_bars = CONFIG[interval]["volume_bars"]

    # --------------------------------------------------------
    # 歷史當下可以知道的 24h 成交額
    # --------------------------------------------------------

    df["volume_24h_usdt"] = (
        df.groupby("symbol")["quote_volume"]
        .transform(
            lambda s: s.rolling(
                volume_bars,
                min_periods=volume_bars
            ).sum()
        )
    )

    group = df.groupby("symbol")

    # ========================================================
    # V8 最重要的改變
    #
    # t K線收盤
    # ↓
    # AI 才取得完整 t 資料
    # ↓
    # 下一根 K 線視為等待/執行階段
    # ↓
    # entry = t+1 close
    # exit  = t+2 close
    #
    # AI 的 target 直接改成：
    # exit_close > entry_close
    #
    # 模型學習目標與回測交易目標完全一致
    # ========================================================

    df["entry_time"] = group["open_time"].shift(-1)
    df["entry_close"] = group["close"].shift(-1)

    df["exit_time"] = group["open_time"].shift(-2)
    df["exit_close"] = group["close"].shift(-2)

    expected_entry = df["open_time"] + duration
    expected_exit = df["open_time"] + duration * 2

    valid = (
        (df["entry_time"] == expected_entry)
        & (df["exit_time"] == expected_exit)
        & df["entry_close"].notna()
        & df["exit_close"].notna()
        & df["volume_24h_usdt"].notna()
    )

    df = df[valid].copy()

    df["tradable_return"] = (
        df["exit_close"] / df["entry_close"] - 1
    )

    df["tradable_target"] = (
        df["tradable_return"] > 0
    ).astype(int)

    df = df.sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    print()
    print("=" * 60)
    print(interval)
    print(f"資料：{len(df):,}")
    print(f"幣種：{df['symbol'].nunique()}")
    print(
        "上漲比例："
        f"{df['tradable_target'].mean():.2%}"
    )

    return df


def create_walk_forward_predictions(df, interval):
    unique_times = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    first_test_index = int(
        len(unique_times) * INITIAL_TRAIN_RATIO
    )

    boundaries = np.linspace(
        first_test_index,
        len(unique_times),
        TEST_FOLDS + 1,
        dtype=int
    )

    duration = CONFIG[interval]["duration"]

    predictions = []

    for fold in range(TEST_FOLDS):
        start_index = boundaries[fold]
        end_index = boundaries[fold + 1]

        if end_index <= start_index:
            continue

        test_start = unique_times.iloc[start_index]
        test_end = unique_times.iloc[end_index - 1]

        # ----------------------------------------------------
        # V8 防止標籤穿越測試邊界
        #
        # target 使用到 t+2 close
        # 所以 train 的 exit_time 必須早於 test_start
        # ----------------------------------------------------

        train = df[
            df["exit_time"] < test_start
        ].copy()

        test = df[
            (df["open_time"] >= test_start)
            & (df["open_time"] <= test_end)
        ].copy()

        if len(train) < 500:
            raise ValueError(
                f"{interval} Fold {fold + 1} "
                f"訓練資料不足：{len(train)}"
            )

        if len(test) < 100:
            raise ValueError(
                f"{interval} Fold {fold + 1} "
                f"測試資料不足：{len(test)}"
            )

        model = make_model()

        model.fit(
            train[FEATURES],
            train["tradable_target"]
        )

        test["ai_score"] = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test["fold"] = fold + 1

        predictions.append(test)

        print(
            f"Fold {fold + 1}"
            f"｜Train {len(train):,}"
            f"｜Test {len(test):,}"
            f"｜{test_start}"
            f" → {test_end}"
        )

    return pd.concat(
        predictions,
        ignore_index=True
    )


def simulate(predictions, threshold, cost):
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_drawdown = 0.0

    trades = []
    equity_rows = []

    for timestamp, group in predictions.groupby(
        "open_time",
        sort=True
    ):

        candidates = group[
            (group["ai_score"] >= threshold)
            & (
                group["volume_24h_usdt"]
                >= MIN_VOLUME_24H
            )
        ].copy()

        if candidates.empty:
            continue

        candidates = (
            candidates
            .sort_values(
                ["ai_score", "symbol"],
                ascending=[False, True]
            )
            .head(MAX_POSITIONS)
        )

        position_size = (
            equity * POSITION_FRACTION
        )

        if position_size < MIN_ORDER_USDT:
            continue

        timestamp_pnl = 0.0

        for _, row in candidates.iterrows():
            gross_return = float(
                row["tradable_return"]
            )

            net_return = gross_return - cost

            pnl = position_size * net_return

            timestamp_pnl += pnl

            trades.append({
                "signal_time": str(row["open_time"]),
                "entry_time": str(row["entry_time"]),
                "exit_time": str(row["exit_time"]),
                "symbol": row["symbol"],
                "fold": int(row["fold"]),
                "ai_score": float(row["ai_score"]),
                "volume_24h_usdt": float(
                    row["volume_24h_usdt"]
                ),
                "position_usdt": position_size,
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl_usdt": pnl,
            })

        equity = max(
            0.0,
            equity + timestamp_pnl
        )

        peak = max(peak, equity)

        drawdown = equity / peak - 1

        max_drawdown = min(
            max_drawdown,
            drawdown
        )

        equity_rows.append({
            "time": str(timestamp),
            "equity": equity,
            "drawdown": drawdown,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    if trades_df.empty:
        summary = {
            "final_capital": INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": None,
            "average_net_return": None,
        }

        return summary, trades_df, equity_df

    summary = {
        "final_capital": float(equity),

        "total_return": float(
            equity / INITIAL_CAPITAL - 1
        ),

        "max_drawdown": float(
            max_drawdown
        ),

        "trades": int(
            len(trades_df)
        ),

        "win_rate": float(
            (trades_df["net_return"] > 0).mean()
        ),

        "average_net_return": float(
            trades_df["net_return"].mean()
        ),
    }

    return summary, trades_df, equity_df


def concentration_analysis(trades):
    if trades.empty:
        return {}

    ordered = trades.sort_values(
        "pnl_usdt",
        ascending=False
    )

    total = float(
        trades["pnl_usdt"].sum()
    )

    result = {
        "total_trade_pnl": total
    }

    for count in [1, 5, 10]:
        removed = float(
            ordered.head(count)["pnl_usdt"].sum()
        )

        result[f"best_{count}_pnl"] = removed

        result[
            f"remaining_after_best_{count}"
        ] = total - removed

        if total != 0:
            result[f"best_{count}_share"] = (
                removed / total
            )
        else:
            result[f"best_{count}_share"] = None

    return result


def fold_analysis(trades):
    if trades.empty:
        return []

    output = []

    for fold, group in trades.groupby("fold"):
        output.append({
            "fold": int(fold),
            "trades": int(len(group)),
            "pnl": float(
                group["pnl_usdt"].sum()
            ),
            "win_rate": float(
                (group["net_return"] > 0).mean()
            ),
            "average_net_return": float(
                group["net_return"].mean()
            ),
        })

    return output


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CryptoAI Online V8")
    print("可交易目標嚴格驗證")
    print("=" * 70)

    all_results = []

    for interval in INTERVALS:
        data = load_and_rebuild_target(
            interval
        )

        predictions = (
            create_walk_forward_predictions(
                data,
                interval
            )
        )

        predictions.to_csv(
            OUTPUT_DIR
            / f"predictions_{interval}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        for threshold in THRESHOLDS:
            for cost in COSTS:

                summary, trades, equity = simulate(
                    predictions,
                    threshold,
                    cost
                )

                concentration = (
                    concentration_analysis(trades)
                )

                folds = fold_analysis(trades)

                result = {
                    "interval": interval,
                    "threshold": threshold,
                    "cost": cost,
                    **summary,
                    "concentration": concentration,
                    "folds": folds,
                }

                all_results.append(result)

                label = (
                    f"{interval}_"
                    f"{int(threshold * 100)}_"
                    f"{int(cost * 10000)}"
                )

                trades.to_csv(
                    OUTPUT_DIR
                    / f"trades_{label}.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

                equity.to_csv(
                    OUTPUT_DIR
                    / f"equity_{label}.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"{interval} "
                    f">={threshold:.0%} "
                    f"成本 {cost:.2%}"
                    f"｜資產 "
                    f"{summary['final_capital']:.2f}"
                    f"｜報酬 "
                    f"{summary['total_return']:.2%}"
                    f"｜回撤 "
                    f"{summary['max_drawdown']:.2%}"
                    f"｜交易 "
                    f"{summary['trades']}"
                )

    with (
        OUTPUT_DIR / "v8_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2
        )

    pd.json_normalize(
        all_results
    ).to_csv(
        OUTPUT_DIR / "v8_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("V8 完成")
    print(
        f"完成測試："
        f"{len(all_results)} 組"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

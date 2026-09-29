import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# ============================================================
# CryptoAI Online V7
# 嚴格 Walk-Forward 驗證
# ============================================================

TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("v7_reports")

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
MIN_ORDER = 5.0

THRESHOLDS = [0.65, 0.70, 0.75]
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


def model_factory():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def load_data(interval):
    path = TRAINING_DIR / f"training_{interval}.csv"

    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")

    df = pd.read_csv(path)

    required = FEATURES + [
        "symbol", "open_time",
        "close", "quote_volume",
        "future_return", "target"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{interval} 缺少欄位：{missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    numeric = FEATURES + [
        "close", "quote_volume",
        "future_return", "target"
    ]

    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values(["symbol", "open_time"])
        .reset_index(drop=True)
    )

    duration = CONFIG[interval]["duration"]
    volume_bars = CONFIG[interval]["volume_bars"]

    # --------------------------------------------------------
    # 僅使用當時已知成交額
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

    # --------------------------------------------------------
    # V7 成交延遲
    #
    # 訊號使用時間 t 已完成 K 線。
    # 真實情況不能假設在 t 的 close 瞬間已經成交。
    #
    # 保守處理：
    # 訊號完成後，跳過原本 target 的下一根，
    # 使用再下一根的 close 作為退出價格。
    #
    # entry = 下一根 K 線 close
    # exit  = 再下一根 K 線 close
    #
    # 這不是逐 tick 撮合，但比 V5/V6 更保守。
    # --------------------------------------------------------

    group = df.groupby("symbol")

    df["entry_time"] = group["open_time"].shift(-1)
    df["entry_close"] = group["close"].shift(-1)

    df["exit_time"] = group["open_time"].shift(-2)
    df["exit_close"] = group["close"].shift(-2)

    expected_entry = df["open_time"] + duration
    expected_exit = df["open_time"] + duration * 2

    valid = (
        (df["entry_time"] == expected_entry)
        & (df["exit_time"] == expected_exit)
        & df["volume_24h_usdt"].notna()
        & df["entry_close"].notna()
        & df["exit_close"].notna()
    )

    df = df[valid].copy()

    df["execution_return"] = (
        df["exit_close"] / df["entry_close"] - 1
    )

    df = df.sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    print(
        f"{interval}: {len(df):,} 筆，"
        f"{df['symbol'].nunique()} 幣"
    )

    return df


def walk_forward(df, interval):
    times = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    first_test = int(len(times) * INITIAL_TRAIN_RATIO)

    boundaries = np.linspace(
        first_test,
        len(times),
        TEST_FOLDS + 1,
        dtype=int
    )

    duration = CONFIG[interval]["duration"]

    outputs = []

    for fold in range(TEST_FOLDS):
        start = boundaries[fold]
        end = boundaries[fold + 1]

        if end <= start:
            continue

        test_start = times.iloc[start]
        test_end = times.iloc[end - 1]

        # 訓練標籤不能跨越測試期。
        train = df[
            df["open_time"] + duration < test_start
        ].copy()

        test = df[
            (df["open_time"] >= test_start)
            & (df["open_time"] <= test_end)
        ].copy()

        if len(train) < 500 or len(test) < 100:
            raise ValueError(
                f"{interval} Fold {fold + 1} 資料不足"
            )

        print(
            f"{interval} Fold {fold + 1}: "
            f"train={len(train):,}, test={len(test):,}"
        )

        model = model_factory()

        model.fit(
            train[FEATURES],
            train["target"].astype(int)
        )

        test["ai_score"] = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        test["fold"] = fold + 1

        outputs.append(test)

    return pd.concat(
        outputs,
        ignore_index=True
    )


def simulate(predictions, threshold, cost):
    equity = INITIAL_CAPITAL
    peak = equity
    max_drawdown = 0.0

    trades = []
    curve = []

    for timestamp, group in predictions.groupby(
        "open_time",
        sort=True
    ):
        candidates = group[
            (group["ai_score"] >= threshold)
            & (group["volume_24h_usdt"] >= MIN_VOLUME_24H)
        ].copy()

        if candidates.empty:
            continue

        candidates = (
            candidates.sort_values(
                ["ai_score", "symbol"],
                ascending=[False, True]
            )
            .head(MAX_POSITIONS)
        )

        amount = equity * POSITION_FRACTION

        if amount < MIN_ORDER:
            continue

        total_pnl = 0.0

        for _, row in candidates.iterrows():
            gross = float(row["execution_return"])
            net = gross - cost
            pnl = amount * net

            total_pnl += pnl

            trades.append({
                "signal_time": str(row["open_time"]),
                "entry_time": str(row["entry_time"]),
                "exit_time": str(row["exit_time"]),
                "symbol": row["symbol"],
                "fold": int(row["fold"]),
                "ai_score": float(row["ai_score"]),
                "volume_24h_usdt": float(row["volume_24h_usdt"]),
                "position_usdt": amount,
                "gross_return": gross,
                "net_return": net,
                "pnl_usdt": pnl,
            })

        equity = max(0.0, equity + total_pnl)
        peak = max(peak, equity)

        drawdown = equity / peak - 1
        max_drawdown = min(max_drawdown, drawdown)

        curve.append({
            "time": str(timestamp),
            "equity": equity,
            "drawdown": drawdown,
        })

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    if trades_df.empty:
        return {
            "final_capital": INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": None,
        }, trades_df, curve_df

    summary = {
        "final_capital": float(equity),
        "total_return": float(equity / INITIAL_CAPITAL - 1),
        "max_drawdown": float(max_drawdown),
        "trades": int(len(trades_df)),
        "win_rate": float(
            (trades_df["net_return"] > 0).mean()
        ),
        "average_net_return": float(
            trades_df["net_return"].mean()
        ),
    }

    return summary, trades_df, curve_df


def concentration_test(trades):
    if trades.empty:
        return {}

    ordered = trades.sort_values(
        "pnl_usdt",
        ascending=False
    )

    total = float(trades["pnl_usdt"].sum())

    result = {
        "total_trade_pnl": total
    }

    for n in [1, 5, 10]:
        removed = float(
            ordered.head(n)["pnl_usdt"].sum()
        )

        result[f"best_{n}_pnl"] = removed
        result[f"remaining_after_best_{n}"] = (
            total - removed
        )

        if total != 0:
            result[f"best_{n}_share"] = (
                removed / total
            )
        else:
            result[f"best_{n}_share"] = None

    return result


def fold_analysis(trades):
    if trades.empty:
        return []

    rows = []

    for fold, group in trades.groupby("fold"):
        rows.append({
            "fold": int(fold),
            "trades": int(len(group)),
            "pnl": float(group["pnl_usdt"].sum()),
            "win_rate": float(
                (group["net_return"] > 0).mean()
            ),
            "average_net_return": float(
                group["net_return"].mean()
            ),
        })

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CryptoAI Online V7")
    print("嚴格驗證版")
    print("=" * 70)

    results = []

    for interval in INTERVALS:
        df = load_data(interval)

        predictions = walk_forward(
            df,
            interval
        )

        predictions.to_csv(
            OUTPUT_DIR / f"predictions_{interval}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        for threshold in THRESHOLDS:
            for cost in COSTS:
                summary, trades, curve = simulate(
                    predictions,
                    threshold,
                    cost
                )

                concentration = concentration_test(
                    trades
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

                results.append(result)

                label = (
                    f"{interval}_"
                    f"{int(threshold * 100)}_"
                    f"{int(cost * 10000)}"
                )

                trades.to_csv(
                    OUTPUT_DIR / f"trades_{label}.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

                curve.to_csv(
                    OUTPUT_DIR / f"equity_{label}.csv",
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"{interval} "
                    f">={threshold:.0%} "
                    f"成本 {cost:.2%}"
                    f"｜資產 {summary['final_capital']:.2f}"
                    f"｜報酬 {summary['total_return']:.2%}"
                    f"｜回撤 {summary['max_drawdown']:.2%}"
                    f"｜交易 {summary['trades']}"
                )

                if concentration:
                    print(
                        "   最大10筆獲利占比：",
                        f"{concentration['best_10_share']:.2%}"
                        if concentration["best_10_share"] is not None
                        else "N/A"
                    )

    with (
        OUTPUT_DIR / "v7_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    pd.json_normalize(results).to_csv(
        OUTPUT_DIR / "v7_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("V7 完成")
    print(f"完成測試：{len(results)} 組")
    print("=" * 70)


if __name__ == "__main__":
    main()

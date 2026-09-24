
import json
from pathlib import Path
 
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# CryptoAI Online V5
# Walk-Forward + 100 USDT 現貨投資組合回測
# ============================================================

TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("backtest_reports")

INTERVALS = ["15m", "4h", "1d"]

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

INITIAL_CAPITAL = 100.0

MAX_POSITIONS = 5
POSITION_FRACTION = 0.20
MIN_ORDER_USDT = 5.0

FEE_COST = 0.002
SLIPPAGE_COST = 0.001

TOTAL_COST = FEE_COST + SLIPPAGE_COST

THRESHOLDS = [0.65, 0.70, 0.75]

MIN_24H_QUOTE_VOLUME = 1_000_000

# 最初 50% 時間作為初始訓練區間
INITIAL_TRAIN_RATIO = 0.50

# 後面分成三個 Walk-Forward 測試區間
TEST_FOLDS = 3

INTERVAL_CONFIG = {
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


def create_model():

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def load_data(interval):

    path = (
        TRAINING_DIR
        / f"training_{interval}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"找不到訓練資料：{path}"
        )

    df = pd.read_csv(path)

    required = FEATURES + [
        "symbol",
        "open_time",
        "quote_volume",
        "future_return",
        "target",
    ]

    missing = [
        name
        for name in required
        if name not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{interval} 缺少欄位：{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    numeric_columns = FEATURES + [
        "quote_volume",
        "future_return",
        "target",
    ]

    for name in numeric_columns:
        df[name] = pd.to_numeric(
            df[name],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required,
    )

    df = df.sort_values(
        ["symbol", "open_time"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 歷史成交額
    # 使用訊號當時及之前已完成的 K 線
    # 不使用未來成交額
    # --------------------------------------------------------

    bars = INTERVAL_CONFIG[
        interval
    ]["volume_bars"]

    df["volume_24h_usdt"] = (
        df.groupby("symbol")["quote_volume"]
        .transform(
            lambda series: series.rolling(
                window=bars,
                min_periods=bars,
            ).sum()
        )
    )

    # --------------------------------------------------------
    # 排除缺 K 線的標籤
    # 如果下一筆不是緊接的一根 K 線，
    # 就不能把它當作下一根的預測答案。
    # --------------------------------------------------------

    duration = INTERVAL_CONFIG[
        interval
    ]["duration"]

    next_time = (
        df.groupby("symbol")["open_time"]
        .shift(-1)
    )

    valid_next_bar = (
        next_time - df["open_time"]
        == duration
    )

    df = df[
        valid_next_bar
        & df["volume_24h_usdt"].notna()
    ].copy()

    df = df.sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"{interval} 沒有可用資料"
        )

    print(
        f"載入 {interval}："
        f"{len(df):,} 筆，"
        f"{df['symbol'].nunique()} 幣"
    )

    return df


def create_folds(df, interval):

    unique_times = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    first_test_index = int(
        len(unique_times)
        * INITIAL_TRAIN_RATIO
    )

    remaining = (
        len(unique_times)
        - first_test_index
    )

    boundaries = np.linspace(
        first_test_index,
        len(unique_times),
        TEST_FOLDS + 1,
        dtype=int,
    )

    duration = INTERVAL_CONFIG[
        interval
    ]["duration"]

    for fold_number in range(
        TEST_FOLDS
    ):

        start_index = boundaries[
            fold_number
        ]

        end_index = boundaries[
            fold_number + 1
        ]

        if end_index <= start_index:
            continue

        test_start = unique_times.iloc[
            start_index
        ]

        test_end = unique_times.iloc[
            end_index - 1
        ]

        # 訓練標籤不能跨越測試起點
        train = df[
            df["open_time"] + duration
            < test_start
        ].copy()

        test = df[
            (df["open_time"] >= test_start)
            & (df["open_time"] <= test_end)
        ].copy()

        yield (
            fold_number + 1,
            train,
            test,
            test_start,
            test_end,
        )


def generate_walk_forward_predictions(
    df,
    interval,
):

    all_predictions = []

    for (
        fold_number,
        train,
        test,
        test_start,
        test_end,
    ) in create_folds(
        df,
        interval,
    ):

        print()
        print(
            f"{interval} Fold {fold_number}"
        )

        print(
            f"訓練：{len(train):,} 筆"
        )

        print(
            f"測試：{len(test):,} 筆"
        )

        print(
            f"測試期間："
            f"{test_start} → {test_end}"
        )

        if (
            len(train) < 500
            or len(test) < 100
        ):

            raise ValueError(
                f"{interval} Fold "
                f"{fold_number} 資料不足"
            )

        model = create_model()

        model.fit(
            train[FEATURES],
            train["target"].astype(int),
        )

        probabilities = (
            model.predict_proba(
                test[FEATURES]
            )[:, 1]
        )

        result = test[
            [
                "symbol",
                "open_time",
                "future_return",
                "target",
                "volume_24h_usdt",
            ]
        ].copy()

        result["ai_score"] = (
            probabilities
        )

        result["fold"] = (
            fold_number
        )

        all_predictions.append(
            result
        )

        print(
            "AI 預測完成"
        )

    if not all_predictions:
        raise RuntimeError(
            f"{interval} 沒有產生預測"
        )

    return pd.concat(
        all_predictions,
        ignore_index=True,
    ).sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)


def simulate_portfolio(
    predictions,
    threshold,
):

    equity = INITIAL_CAPITAL

    peak_equity = equity

    max_drawdown = 0.0

    trades = []
    equity_history = []

    skipped_liquidity = 0
    skipped_min_order = 0

    # 每個時間點是一個投資決策
    # 所有同時訊號共用同一筆資金

    for timestamp, group in (
        predictions.groupby(
            "open_time",
            sort=True,
        )
    ):

        candidates = group[
            group["ai_score"]
            >= threshold
        ].copy()

        if candidates.empty:
            continue

        before_liquidity = len(
            candidates
        )

        candidates = candidates[
            candidates["volume_24h_usdt"]
            >= MIN_24H_QUOTE_VOLUME
        ].copy()

        skipped_liquidity += (
            before_liquidity
            - len(candidates)
        )

        if candidates.empty:
            continue

        # 同一時間最多選五個幣
        # 優先使用較高 AI 分數
        candidates = (
            candidates.sort_values(
                "ai_score",
                ascending=False,
            ).head(
                MAX_POSITIONS
            )
        )

        equity_before = equity

        position_amount = (
            equity_before
            * POSITION_FRACTION
        )

        if (
            position_amount
            < MIN_ORDER_USDT
        ):

            skipped_min_order += (
                len(candidates)
            )

            continue

        total_pnl = 0.0

        for _, row in (
            candidates.iterrows()
        ):

            gross_return = float(
                row["future_return"]
            )

            net_return = (
                gross_return
                - TOTAL_COST
            )

            pnl = (
                position_amount
                * net_return
            )

            total_pnl += pnl

            trades.append({
                "time": str(timestamp),
                "symbol": row["symbol"],
                "fold": int(
                    row["fold"]
                ),
                "threshold": threshold,
                "ai_score": float(
                    row["ai_score"]
                ),
                "volume_24h_usdt": float(
                    row["volume_24h_usdt"]
                ),
                "position_usdt":
                    position_amount,
                "gross_return":
                    gross_return,
                "net_return":
                    net_return,
                "pnl_usdt": pnl,
            })

        # 同一根 K 線的持倉一起結算
        equity += total_pnl

        equity = max(
            equity,
            0.0,
        )

        peak_equity = max(
            peak_equity,
            equity,
        )

        drawdown = (
            equity / peak_equity - 1
        )

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

        equity_history.append({
            "time": str(timestamp),
            "equity": equity,
            "drawdown": drawdown,
        })

    trades_df = pd.DataFrame(
        trades
    )

    equity_df = pd.DataFrame(
        equity_history
    )

    if trades_df.empty:

        summary = {
            "threshold": threshold,
            "initial_capital":
                INITIAL_CAPITAL,
            "final_capital":
                INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": None,
            "average_net_return": None,
            "skipped_liquidity":
                skipped_liquidity,
            "skipped_min_order":
                skipped_min_order,
        }

        return (
            summary,
            trades_df,
            equity_df,
        )

    summary = {
        "threshold": threshold,

        "initial_capital":
            INITIAL_CAPITAL,

        "final_capital":
            float(equity),

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
            (
                trades_df["net_return"]
                > 0
            ).mean()
        ),

        "average_net_return": float(
            trades_df["net_return"]
            .mean()
        ),

        "skipped_liquidity": int(
            skipped_liquidity
        ),

        "skipped_min_order": int(
            skipped_min_order
        ),
    }

    return (
        summary,
        trades_df,
        equity_df,
    )


def run_interval(interval):

    print()
    print("=" * 70)
    print(
        f"CryptoAI V5：{interval}"
    )
    print("=" * 70)

    df = load_data(
        interval
    )

    predictions = (
        generate_walk_forward_predictions(
            df,
            interval,
        )
    )

    prediction_path = (
        OUTPUT_DIR
        / f"walkforward_{interval}.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8",
    )

    reports = []

    for threshold in THRESHOLDS:

        (
            summary,
            trades,
            equity_history,
        ) = simulate_portfolio(
            predictions,
            threshold,
        )

        reports.append(
            summary
        )

        label = int(
            threshold * 100
        )

        trades.to_csv(
            OUTPUT_DIR
            / f"trades_{interval}_{label}.csv",
            index=False,
            encoding="utf-8",
        )

        equity_history.to_csv(
            OUTPUT_DIR
            / f"equity_{interval}_{label}.csv",
            index=False,
            encoding="utf-8",
        )

        print()
        print(
            f"{interval}｜"
            f"AI >= {label}%"
        )

        print(
            "最終資產："
            f"{summary['final_capital']:.2f}"
            " USDT"
        )

        print(
            "總報酬："
            f"{summary['total_return']:.2%}"
        )

        print(
            "最大回撤："
            f"{summary['max_drawdown']:.2%}"
        )

        print(
            "交易次數："
            f"{summary['trades']}"
        )

        if (
            summary["win_rate"]
            is not None
        ):

            print(
                "扣成本勝率："
                f"{summary['win_rate']:.2%}"
            )

        print(
            "流動性不足排除："
            f"{summary['skipped_liquidity']}"
        )

    report_path = (
        OUTPUT_DIR
        / f"portfolio_{interval}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            reports,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return reports


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CryptoAI Online V5")
    print("Walk-Forward + 100 USDT")
    print("=" * 70)

    print(
        f"本金：{INITIAL_CAPITAL} USDT"
    )

    print(
        f"最多持倉：{MAX_POSITIONS}"
    )

    print(
        f"單幣比例：{POSITION_FRACTION:.0%}"
    )

    print(
        f"總假設成本：{TOTAL_COST:.2%}"
    )

    print(
        "最低 24H 成交額："
        f"{MIN_24H_QUOTE_VOLUME:,} USDT"
    )

    all_reports = {}

    failures = {}

    for interval in INTERVALS:

        try:

            all_reports[interval] = (
                run_interval(
                    interval
                )
            )

        except Exception as error:

            failures[interval] = (
                str(error)
            )

            print(
                f"{interval} 失敗："
                f"{type(error).__name__}: "
                f"{error}"
            )

    summary = {
        "settings": {
            "initial_capital":
                INITIAL_CAPITAL,
            "max_positions":
                MAX_POSITIONS,
            "position_fraction":
                POSITION_FRACTION,
            "total_cost":
                TOTAL_COST,
            "minimum_24h_volume":
                MIN_24H_QUOTE_VOLUME,
        },
        "results": all_reports,
        "failures": failures,
    }

    with (
        OUTPUT_DIR
        / "v5_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("V5 回測完成")
    print("=" * 70)

    for interval, reports in (
        all_reports.items()
    ):

        for report in reports:

            print(
                f"{interval} "
                f">={report['threshold']:.0%}"
                "｜資產 "
                f"{report['final_capital']:.2f}"
                " USDT"
                "｜回撤 "
                f"{report['max_drawdown']:.2%}"
                "｜交易 "
                f"{report['trades']}"
            )

    print(
        f"成功週期："
        f"{len(all_reports)}"
    )

    print(
        f"失敗週期："
        f"{len(failures)}"
    )

    if failures:

        raise RuntimeError(
            "部分週期回測失敗，"
            "請檢查上方錯誤"
        )


if __name__ == "__main__":
    main()

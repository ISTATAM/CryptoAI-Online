
import json
from pathlib import Path

import pandas as pd


# ==================================================
# CryptoAI Online V6
# 策略成本壓力測試與風險分析
# ==================================================

INPUT_DIR = Path("backtest_reports")
OUTPUT_DIR = Path("v6_reports")

INTERVALS = ["15m", "4h", "1d"]

THRESHOLDS = [0.65, 0.70, 0.75]

COSTS = [0.002, 0.003, 0.005]

INITIAL_CAPITAL = 100.0

MAX_POSITIONS = 5

POSITION_FRACTION = 0.20

MIN_ORDER_USDT = 5.0

MIN_VOLUME_24H = 1_000_000


def load_predictions(interval):

    path = (
        INPUT_DIR
        / f"walkforward_{interval}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"找不到：{path}"
        )

    df = pd.read_csv(path)

    required = [
        "symbol",
        "open_time",
        "future_return",
        "volume_24h_usdt",
        "ai_score",
        "fold",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{interval} 缺少欄位：{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    numeric_columns = [
        "future_return",
        "volume_24h_usdt",
        "ai_score",
        "fold",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    ).copy()

    df = df.sort_values(
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    return df


def simulate(
    predictions,
    threshold,
    cost,
):

    equity = INITIAL_CAPITAL

    peak = INITIAL_CAPITAL

    max_drawdown = 0.0

    trades = []

    equity_rows = []

    for timestamp, group in (
        predictions.groupby(
            "open_time",
            sort=True,
        )
    ):

        candidates = group[
            (group["ai_score"] >= threshold)
            &
            (
                group["volume_24h_usdt"]
                >= MIN_VOLUME_24H
            )
        ].copy()

        if candidates.empty:
            continue

        candidates = (
            candidates.sort_values(
                ["ai_score", "symbol"],
                ascending=[False, True],
            )
            .head(MAX_POSITIONS)
        )

        amount = (
            equity
            * POSITION_FRACTION
        )

        if amount < MIN_ORDER_USDT:
            continue

        total_pnl = 0.0

        for _, row in candidates.iterrows():

            gross_return = float(
                row["future_return"]
            )

            net_return = (
                gross_return - cost
            )

            pnl = (
                amount * net_return
            )

            total_pnl += pnl

            trades.append({
                "time": str(timestamp),
                "symbol": row["symbol"],
                "fold": int(row["fold"]),
                "ai_score": float(
                    row["ai_score"]
                ),
                "volume_24h_usdt": float(
                    row["volume_24h_usdt"]
                ),
                "position_usdt": amount,
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl_usdt": pnl,
            })

        equity = max(
            0.0,
            equity + total_pnl,
        )

        peak = max(
            peak,
            equity,
        )

        drawdown = (
            equity / peak - 1
        )

        max_drawdown = min(
            max_drawdown,
            drawdown,
        )

        equity_rows.append({
            "time": str(timestamp),
            "equity": equity,
            "drawdown": drawdown,
        })

    trades_df = pd.DataFrame(trades)

    equity_df = pd.DataFrame(
        equity_rows
    )

    if trades_df.empty:

        summary = {
            "threshold": threshold,
            "cost": cost,
            "initial_capital":
                INITIAL_CAPITAL,
            "final_capital":
                INITIAL_CAPITAL,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "win_rate": None,
            "average_net_return": None,
        }

        return (
            summary,
            trades_df,
            equity_df,
        )

    summary = {
        "threshold": threshold,
        "cost": cost,

        "initial_capital":
            INITIAL_CAPITAL,

        "final_capital":
            float(equity),

        "total_return":
            float(
                equity / INITIAL_CAPITAL - 1
            ),

        "max_drawdown":
            float(max_drawdown),

        "trades":
            int(len(trades_df)),

        "win_rate":
            float(
                (
                    trades_df["net_return"]
                    > 0
                ).mean()
            ),

        "average_net_return":
            float(
                trades_df["net_return"]
                .mean()
            ),
    }

    return (
        summary,
        trades_df,
        equity_df,
    )


def analyze_trades(
    trades,
    interval,
    threshold,
    cost,
):

    if trades.empty:
        return

    label = (
        f"{interval}_"
        f"{int(threshold * 100)}_"
        f"{int(cost * 10000)}"
    )

    # ----------------------------------------------
    # 每個幣種的表現
    # ----------------------------------------------

    coins = (
        trades.groupby("symbol")
        .agg(
            trades=(
                "pnl_usdt",
                "size",
            ),
            total_pnl=(
                "pnl_usdt",
                "sum",
            ),
            average_net_return=(
                "net_return",
                "mean",
            ),
        )
        .reset_index()
    )

    coins = coins.sort_values(
        "total_pnl",
        ascending=False,
    )

    coins.to_csv(
        OUTPUT_DIR
        / f"coins_{label}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------
    # 每個 Walk-Forward 測試區間
    # ----------------------------------------------

    folds = (
        trades.groupby("fold")
        .agg(
            trades=(
                "pnl_usdt",
                "size",
            ),
            total_pnl=(
                "pnl_usdt",
                "sum",
            ),
            average_net_return=(
                "net_return",
                "mean",
            ),
        )
        .reset_index()
    )

    folds.to_csv(
        OUTPUT_DIR
        / f"folds_{label}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------
    # 每月獲利與虧損
    # ----------------------------------------------

    monthly = trades.copy()

    monthly["month"] = (
        pd.to_datetime(
            monthly["time"],
            utc=True,
        )
        .dt.strftime("%Y-%m")
    )

    monthly = (
        monthly.groupby("month")
        .agg(
            trades=(
                "pnl_usdt",
                "size",
            ),
            total_pnl=(
                "pnl_usdt",
                "sum",
            ),
        )
        .reset_index()
    )

    monthly.to_csv(
        OUTPUT_DIR
        / f"monthly_{label}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CryptoAI Online V6")
    print("策略壓力測試")
    print("=" * 65)

    results = []

    for interval in INTERVALS:

        predictions = load_predictions(
            interval
        )

        print()
        print(
            f"{interval}："
            f"{len(predictions):,} 筆預測"
        )

        for threshold in THRESHOLDS:

            for cost in COSTS:

                (
                    summary,
                    trades,
                    equity,
                ) = simulate(
                    predictions,
                    threshold,
                    cost,
                )

                summary["interval"] = (
                    interval
                )

                results.append(
                    summary
                )

                label = (
                    f"{interval}_"
                    f"{int(threshold * 100)}_"
                    f"{int(cost * 10000)}"
                )

                trades.to_csv(
                    OUTPUT_DIR
                    / f"trades_{label}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                equity.to_csv(
                    OUTPUT_DIR
                    / f"equity_{label}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                analyze_trades(
                    trades,
                    interval,
                    threshold,
                    cost,
                )

                print(
                    f"{interval} "
                    f">={threshold:.0%} "
                    f"成本 {cost:.2%}"
                    "｜資產 "
                    f"{summary['final_capital']:.2f}"
                    " USDT"
                    "｜回撤 "
                    f"{summary['max_drawdown']:.2%}"
                    "｜交易 "
                    f"{summary['trades']}"
                )

    summary_df = pd.DataFrame(
        results
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "v6_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_DIR
        / "v6_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 65)
    print("V6 全部完成")
    print(
        f"完成測試："
        f"{len(results)} 組"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()

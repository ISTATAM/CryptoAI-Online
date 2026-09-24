
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# CryptoAI Online V4
# 三週期 AI 訓練與時間序列驗證
# ============================================================

TRAINING_FOLDER = Path("training")
MODEL_FOLDER = Path("models")
REPORT_FOLDER = Path("reports")

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

# 保留最後 20% 時間作測試
TEST_RATIO = 0.20

# 現貨單次買入與賣出的合計假設成本
# 0.002 = 0.20%
ROUND_TRIP_COST = 0.002

# 只研究做多，不建立放空交易
LONG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]

MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100


def create_model():

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )


def load_dataset(interval):

    filepath = (
        TRAINING_FOLDER
        / f"training_{interval}.csv"
    )

    if not filepath.exists():

        raise FileNotFoundError(
            f"找不到 {filepath}"
        )

    df = pd.read_csv(filepath)

    required = (
        FEATURES
        + [
            "symbol",
            "open_time",
            "future_return",
            "target",
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{interval} 缺少欄位：{missing}"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        utc=True,
    )

    for column in (
        FEATURES + ["future_return", "target"]
    ):

        df[column] = pd.to_numeric(
            df[column],
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
        ["open_time", "symbol"]
    ).reset_index(drop=True)

    return df


def split_by_time(df, interval):

    # 同一個時間點的所有幣種必須在同一側
    unique_times = (
        df["open_time"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    split_index = int(
        len(unique_times)
        * (1 - TEST_RATIO)
    )

    if (
        split_index < 2
        or split_index >= len(unique_times)
    ):

        raise ValueError(
            f"{interval} 可用時間點不足"
        )

    split_time = unique_times.iloc[
        split_index
    ]

    # V3 的答案是「下一根 K 線」
    # 訓練集最後一個時間點的答案
    # 可能來自測試集第一個時間點
    # 因此在邊界留出一根 K 線的間隔

    interval_duration = {
        "15m": pd.Timedelta(minutes=15),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }[interval]

    train = df[
        df["open_time"]
        + interval_duration
        < split_time
    ].copy()

    test = df[
        df["open_time"] >= split_time
    ].copy()

    return train, test, split_time


def calculate_threshold_results(
    test,
    probabilities,
):

    results = []

    for threshold in LONG_THRESHOLDS:

        mask = (
            probabilities >= threshold
        )

        selected = test.loc[
            mask
        ].copy()

        count = len(selected)

        if count == 0:

            results.append({
                "threshold": threshold,
                "signals": 0,
                "direction_accuracy": None,
                "after_cost_win_rate": None,
                "average_gross_return": None,
                "average_net_return": None,
                "sum_net_return": None,
            })

            continue

        gross_returns = (
            selected["future_return"]
            .to_numpy(dtype=float)
        )

        net_returns = (
            gross_returns
            - ROUND_TRIP_COST
        )

        results.append({
            "threshold": threshold,
            "signals": int(count),

            "direction_accuracy": float(
                np.mean(
                    gross_returns > 0
                )
            ),

            "after_cost_win_rate": float(
                np.mean(
                    net_returns > 0
                )
            ),

            "average_gross_return": float(
                np.mean(gross_returns)
            ),

            "average_net_return": float(
                np.mean(net_returns)
            ),

            "sum_net_return": float(
                np.sum(net_returns)
            ),
        })

    return results


def train_interval(interval):

    print()
    print("=" * 70)
    print(f"開始訓練：{interval}")
    print("=" * 70)

    df = load_dataset(interval)

    print(
        f"總資料：{len(df):,} 筆"
    )

    print(
        f"幣種：{df['symbol'].nunique()}"
    )

    train, test, split_time = (
        split_by_time(
            df,
            interval,
        )
    )

    if len(train) < MIN_TRAIN_ROWS:

        raise ValueError(
            f"{interval} 訓練資料不足"
        )

    if len(test) < MIN_TEST_ROWS:

        raise ValueError(
            f"{interval} 測試資料不足"
        )

    print(
        f"訓練筆數：{len(train):,}"
    )

    print(
        f"測試筆數：{len(test):,}"
    )

    print(
        "訓練期間：",
        train["open_time"].min(),
        "→",
        train["open_time"].max(),
    )

    print(
        "測試期間：",
        test["open_time"].min(),
        "→",
        test["open_time"].max(),
    )

    x_train = train[FEATURES]
    y_train = train["target"].astype(int)

    x_test = test[FEATURES]
    y_test = test["target"].astype(int)

    if y_train.nunique() < 2:

        raise ValueError(
            f"{interval} 訓練集只有一種漲跌答案"
        )

    model = create_model()

    print()
    print("正在訓練 AI...")

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_test
    )

    probabilities = (
        model.predict_proba(
            x_test
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    up_ratio = float(
        y_test.mean()
    )

    baseline_accuracy = max(
        up_ratio,
        1 - up_ratio,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).tolist()

    print()
    print(
        f"AI 準確率：{accuracy:.2%}"
    )

    print(
        "基準準確率："
        f"{baseline_accuracy:.2%}"
    )

    print(
        "測試集上漲比例："
        f"{up_ratio:.2%}"
    )

    print(
        "混淆矩陣：",
        matrix,
    )

    threshold_results = (
        calculate_threshold_results(
            test,
            probabilities,
        )
    )

    print()
    print("現貨做多訊號驗證")
    print(
        "假設單次完整交易成本："
        f"{ROUND_TRIP_COST:.2%}"
    )

    for result in threshold_results:

        threshold = (
            result["threshold"]
        )

        count = result["signals"]

        if count == 0:

            print(
                f">={threshold:.0%}："
                "沒有訊號"
            )

            continue

        print(
            f">={threshold:.0%}："
            f"{count:,} 筆"
            "｜方向正確率 "
            f"{result['direction_accuracy']:.2%}"
            "｜扣成本勝率 "
            f"{result['after_cost_win_rate']:.2%}"
            "｜平均淨報酬 "
            f"{result['average_net_return']:.4%}"
        )

    # --------------------------------------------------------
    # 儲存測試期間的逐筆預測
    # --------------------------------------------------------

    predictions_df = test[
        [
            "symbol",
            "open_time",
            "future_return",
            "target",
        ]
    ].copy()

    predictions_df["ai_up_probability"] = (
        probabilities
    )

    predictions_df["predicted_up"] = (
        predictions
    )

    predictions_df["net_long_return"] = (
        predictions_df["future_return"]
        - ROUND_TRIP_COST
    )

    prediction_file = (
        REPORT_FOLDER
        / f"predictions_{interval}.csv"
    )

    predictions_df.to_csv(
        prediction_file,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 測試完成後，用全部已知歷史資料
    # 重新訓練供下一階段即時預測的模型
    # 注意：這個正式模型沒有獨立測試成績
    # 上面的測試成績屬於先前的時間切分模型
    # --------------------------------------------------------

    print()
    print("正在建立正式預測模型...")

    production_model = (
        create_model()
    )

    production_model.fit(
        df[FEATURES],
        df["target"].astype(int),
    )

    model_package = {
        "model": production_model,
        "features": FEATURES,
        "interval": interval,
        "training_rows": int(
            len(df)
        ),
        "last_training_time": str(
            df["open_time"].max()
        ),
        "round_trip_cost_assumption":
            ROUND_TRIP_COST,
    }

    model_file = (
        MODEL_FOLDER
        / f"crypto_model_{interval}.pkl"
    )

    with model_file.open(
        "wb"
    ) as file:

        pickle.dump(
            model_package,
            file,
        )

    print(
        f"模型已儲存：{model_file}"
    )

    report = {
        "interval": interval,

        "symbols": int(
            df["symbol"].nunique()
        ),

        "total_rows": int(
            len(df)
        ),

        "train_rows": int(
            len(train)
        ),

        "test_rows": int(
            len(test)
        ),

        "split_time": str(
            split_time
        ),

        "train_start": str(
            train["open_time"].min()
        ),

        "train_end": str(
            train["open_time"].max()
        ),

        "test_start": str(
            test["open_time"].min()
        ),

        "test_end": str(
            test["open_time"].max()
        ),

        "accuracy": float(
            accuracy
        ),

        "baseline_accuracy":
            baseline_accuracy,

        "test_up_ratio":
            up_ratio,

        "confusion_matrix":
            matrix,

        "round_trip_cost":
            ROUND_TRIP_COST,

        "threshold_results":
            threshold_results,
    }

    report_file = (
        REPORT_FOLDER
        / f"report_{interval}.json"
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"報告已儲存：{report_file}"
    )

    return report


def main():

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("CryptoAI Online V4")
    print("三週期 AI 模型訓練")
    print("=" * 70)

    reports = []
    failures = []

    for interval in INTERVALS:

        try:

            report = train_interval(
                interval
            )

            reports.append(
                report
            )

        except Exception as error:

            print()
            print(
                f"{interval} 訓練失敗："
                f"{type(error).__name__}: {error}"
            )

            failures.append({
                "interval": interval,
                "error": str(error),
            })

    summary = {
        "successful_models": len(
            reports
        ),

        "failed_models": len(
            failures
        ),

        "reports": reports,
        "failures": failures,
    }

    summary_file = (
        REPORT_FOLDER
        / "model_summary.json"
    )

    with summary_file.open(
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
    print("V4 AI 訓練總結")
    print("=" * 70)

    for report in reports:

        print(
            f"{report['interval']}："
            f"AI {report['accuracy']:.2%}"
            "｜基準 "
            f"{report['baseline_accuracy']:.2%}"
        )

    print(
        f"成功模型：{len(reports)}"
    )

    print(
        f"失敗模型：{len(failures)}"
    )

    if failures:

        raise RuntimeError(
            "部分模型訓練失敗，"
            "請查看上方錯誤"
        )


if __name__ == "__main__":
    main()

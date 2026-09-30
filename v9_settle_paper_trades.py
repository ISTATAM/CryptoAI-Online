import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# ============================================================
# CryptoAI V9-3
# Paper Trading Settlement Engine
#
# V8 / V9 固定交易規則：
#
# signal = t 已完成的 1D K線
# entry  = t+1 的 close
# exit   = t+2 的 close
#
# WAIT_ENTRY -> WAIT_EXIT -> CLOSED
# ============================================================

BASE_URL = "https://data-api.binance.vision"

SIGNAL_FILE = Path(
    "paper_trading/signals.csv"
)

SUMMARY_FILE = Path(
    "paper_trading/settlement_summary.json"
)

COSTS = {
    "0_2": 0.002,
    "0_3": 0.003,
    "0_5": 0.005,
}


def api_get(path, params=None):

    url = BASE_URL + path

    if params:
        url += "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent":
                "CryptoAI-V9-Settlement/1.0"
        },
    )

    with urlopen(
        request,
        timeout=30
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


def get_daily_kline(symbol, open_time):

    target = pd.to_datetime(
        open_time,
        utc=True
    )

    start_ms = int(
        target.timestamp() * 1000
    )

    raw = api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "limit": 1,
        },
    )

    if not raw:
        return None

    kline = raw[0]

    actual_open_time = pd.to_datetime(
        int(kline[0]),
        unit="ms",
        utc=True
    )

    # 必須精確是我們要的那根 K 線。
    if actual_open_time != target:
        return None

    close_time_ms = int(kline[6])

    now_ms = int(
        datetime.now(timezone.utc)
        .timestamp() * 1000
    )

    # 尚未完成的 K 線不能拿來結算。
    if close_time_ms >= now_ms:
        return None

    return {
        "open_time": actual_open_time,
        "close_time": pd.to_datetime(
            close_time_ms,
            unit="ms",
            utc=True
        ),
        "close": float(kline[4]),
    }


def ensure_columns(df):

    defaults = {
        "entry_time": "",
        "entry_price": "",
        "exit_time": "",
        "exit_price": "",
        "gross_return": "",
        "net_return_cost_0_2": "",
        "net_return_cost_0_3": "",
        "net_return_cost_0_5": "",
        "direction_correct": "",
        "closed_at_utc": "",
    }

    for column, default in defaults.items():

        if column not in df.columns:
            df[column] = default

    return df


def main():

    print("=" * 70)
    print("CryptoAI V9-3 Settlement Engine")
    print("=" * 70)

    if not SIGNAL_FILE.exists():

        print()
        print(
            "目前還沒有 signals.csv"
        )

        print(
            "代表目前沒有 AI >= 70% "
            "的 Paper Trading 訊號。"
        )

        print(
            "本次不需要結算。"
        )

        print("=" * 70)

        return

    df = pd.read_csv(
        SIGNAL_FILE,
        dtype=str,
        keep_default_na=False
    )

    if df.empty:

        print(
            "signals.csv 為空，"
            "本次沒有交易需要處理。"
        )

        return

    df = ensure_columns(df)

    entry_updates = 0
    closed_updates = 0
    waiting_entry = 0
    waiting_exit = 0
    already_closed = 0
    errors = []

    for index, row in df.iterrows():

        status = row["status"].strip()

        symbol = row["symbol"].strip()

        try:

            signal_time = pd.to_datetime(
                row["signal_open_time"],
                utc=True
            )

            expected_entry_time = (
                signal_time
                + pd.Timedelta(days=1)
            )

            expected_exit_time = (
                signal_time
                + pd.Timedelta(days=2)
            )

            # ================================================
            # STEP 1
            # WAIT_ENTRY
            # ================================================

            if status == "WAIT_ENTRY":

                entry = get_daily_kline(
                    symbol,
                    expected_entry_time
                )

                if entry is None:

                    waiting_entry += 1

                    print(
                        f"{symbol:12s} "
                        "WAIT_ENTRY "
                        "（進場 K 線尚未完成）"
                    )

                    continue

                df.at[
                    index,
                    "entry_time"
                ] = str(
                    entry["open_time"]
                )

                df.at[
                    index,
                    "entry_price"
                ] = str(
                    entry["close"]
                )

                df.at[
                    index,
                    "status"
                ] = "WAIT_EXIT"

                status = "WAIT_EXIT"

                entry_updates += 1

                print(
                    f"{symbol:12s} "
                    f"ENTRY = "
                    f"{entry['close']}"
                )

            # ================================================
            # STEP 2
            # WAIT_EXIT
            #
            # 同一次執行允許直接繼續檢查 EXIT。
            # 如果 Action 漏跑幾天，可以一次補齊。
            # ================================================

            if status == "WAIT_EXIT":

                entry_price_text = str(
                    df.at[
                        index,
                        "entry_price"
                    ]
                ).strip()

                if not entry_price_text:

                    # 舊紀錄若只有 WAIT_EXIT，
                    # 但 entry_price 不存在，
                    # 自動補回正確 entry K 線。
                    entry = get_daily_kline(
                        symbol,
                        expected_entry_time
                    )

                    if entry is None:

                        waiting_entry += 1

                        print(
                            f"{symbol:12s} "
                            "缺少 ENTRY，等待資料"
                        )

                        continue

                    entry_price = float(
                        entry["close"]
                    )

                    df.at[
                        index,
                        "entry_time"
                    ] = str(
                        entry["open_time"]
                    )

                    df.at[
                        index,
                        "entry_price"
                    ] = str(
                        entry_price
                    )

                else:

                    entry_price = float(
                        entry_price_text
                    )

                exit_kline = get_daily_kline(
                    symbol,
                    expected_exit_time
                )

                if exit_kline is None:

                    waiting_exit += 1

                    print(
                        f"{symbol:12s} "
                        "WAIT_EXIT "
                        "（出場 K 線尚未完成）"
                    )

                    continue

                exit_price = float(
                    exit_kline["close"]
                )

                gross_return = (
                    exit_price
                    / entry_price
                    - 1
                )

                net_02 = (
                    gross_return
                    - COSTS["0_2"]
                )

                net_03 = (
                    gross_return
                    - COSTS["0_3"]
                )

                net_05 = (
                    gross_return
                    - COSTS["0_5"]
                )

                df.at[
                    index,
                    "exit_time"
                ] = str(
                    exit_kline["open_time"]
                )

                df.at[
                    index,
                    "exit_price"
                ] = str(
                    exit_price
                )

                df.at[
                    index,
                    "gross_return"
                ] = str(
                    gross_return
                )

                df.at[
                    index,
                    "net_return_cost_0_2"
                ] = str(
                    net_02
                )

                df.at[
                    index,
                    "net_return_cost_0_3"
                ] = str(
                    net_03
                )

                df.at[
                    index,
                    "net_return_cost_0_5"
                ] = str(
                    net_05
                )

                df.at[
                    index,
                    "direction_correct"
                ] = (
                    "TRUE"
                    if gross_return > 0
                    else "FALSE"
                )

                df.at[
                    index,
                    "closed_at_utc"
                ] = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                df.at[
                    index,
                    "status"
                ] = "CLOSED"

                closed_updates += 1

                print(
                    f"{symbol:12s} "
                    f"CLOSED "
                    f"Entry {entry_price} "
                    f"Exit {exit_price} "
                    f"Gross "
                    f"{gross_return:+.2%} "
                    f"Net(0.3%) "
                    f"{net_03:+.2%}"
                )

            elif status == "CLOSED":

                already_closed += 1

        except Exception as exc:

            errors.append({
                "symbol": symbol,
                "error": str(exc),
            })

            print(
                f"{symbol:12s} "
                f"ERROR: {exc}"
            )

    # ========================================================
    # CLOSED 紀錄之後不重新計算
    # 只保存這次必要的狀態更新。
    # ========================================================

    df.to_csv(
        SIGNAL_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    status_counts = (
        df["status"]
        .value_counts()
        .to_dict()
    )

    closed_df = df[
        df["status"] == "CLOSED"
    ].copy()

    statistics = {
        "closed_trades":
            int(len(closed_df)),
        "win_rate_gross":
            None,
        "average_net_0_2":
            None,
        "average_net_0_3":
            None,
        "average_net_0_5":
            None,
    }

    if not closed_df.empty:

        gross = pd.to_numeric(
            closed_df["gross_return"],
            errors="coerce"
        )

        net02 = pd.to_numeric(
            closed_df[
                "net_return_cost_0_2"
            ],
            errors="coerce"
        )

        net03 = pd.to_numeric(
            closed_df[
                "net_return_cost_0_3"
            ],
            errors="coerce"
        )

        net05 = pd.to_numeric(
            closed_df[
                "net_return_cost_0_5"
            ],
            errors="coerce"
        )

        statistics = {
            "closed_trades":
                int(len(closed_df)),

            "win_rate_gross":
                float(
                    (gross > 0).mean()
                ),

            "average_net_0_2":
                float(net02.mean()),

            "average_net_0_3":
                float(net03.mean()),

            "average_net_0_5":
                float(net05.mean()),
        }

    summary = {
        "version": "V9-3",
        "settled_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "entry_updates":
            entry_updates,
        "closed_updates":
            closed_updates,
        "waiting_entry":
            waiting_entry,
        "waiting_exit":
            waiting_exit,
        "already_closed":
            already_closed,
        "status_counts":
            status_counts,
        "statistics":
            statistics,
        "errors":
            errors,
    }

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with SUMMARY_FILE.open(
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
    print("=" * 70)
    print("V9-3 結算完成")
    print(
        f"新增 ENTRY："
        f"{entry_updates}"
    )
    print(
        f"新增 CLOSED："
        f"{closed_updates}"
    )
    print(
        f"等待 ENTRY："
        f"{waiting_entry}"
    )
    print(
        f"等待 EXIT："
        f"{waiting_exit}"
    )
    print(
        f"目前 CLOSED："
        f"{len(closed_df)}"
    )
    print(
        f"錯誤："
        f"{len(errors)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

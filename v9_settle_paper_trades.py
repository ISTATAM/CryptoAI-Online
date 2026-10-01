import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://data-api.binance.vision"
SIGNAL_FILE = Path("paper_trading/signals.csv")
SUMMARY_FILE = Path("paper_trading/settlement_summary.json")
COSTS = {"0_2": 0.002, "0_3": 0.003, "0_5": 0.005}


def api_get(path, params=None):
    url = BASE_URL + path
    if params:
        url += "?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "CryptoAI-V9-Settlement/1.0"})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_daily_kline(symbol, open_time):
    target = pd.to_datetime(open_time, utc=True)
    start_ms = int(target.timestamp() * 1000)
    raw = api_get("/api/v3/klines", {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ms,
        "limit": 1,
    })
    if not raw:
        return None
    k = raw[0]
    actual_open = pd.to_datetime(int(k[0]), unit="ms", utc=True)
    if actual_open != target:
        return None
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if int(k[6]) >= now_ms:
        return None
    return {
        "open_time": actual_open,
        "close_time": pd.to_datetime(int(k[6]), unit="ms", utc=True),
        "close": float(k[4]),
    }


def ensure_columns(df):
    defaults = {
        "entry_time": "", "entry_price": "", "exit_time": "", "exit_price": "",
        "gross_return": "", "net_return_cost_0_2": "", "net_return_cost_0_3": "",
        "net_return_cost_0_5": "", "direction_correct": "", "closed_at_utc": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def write_summary(payload):
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 70)
    print("CryptoAI V9-3 Settlement Engine")
    print("=" * 70)

    if not SIGNAL_FILE.exists():
        print("目前還沒有 signals.csv，本次不需要結算。")
        write_summary({
            "version": "V9-3",
            "settled_at_utc": datetime.now(timezone.utc).isoformat(),
            "message": "No signals.csv yet",
            "status_counts": {},
            "statistics": {"closed_trades": 0},
            "errors": [],
        })
        return

    df = pd.read_csv(SIGNAL_FILE, dtype=str, keep_default_na=False)
    if df.empty:
        print("signals.csv 為空，本次沒有交易需要處理。")
        write_summary({
            "version": "V9-3",
            "settled_at_utc": datetime.now(timezone.utc).isoformat(),
            "message": "signals.csv is empty",
            "status_counts": {},
            "statistics": {"closed_trades": 0},
            "errors": [],
        })
        return

    df = ensure_columns(df)
    entry_updates = closed_updates = waiting_entry = waiting_exit = already_closed = 0
    errors = []

    for index, row in df.iterrows():
        status = row["status"].strip()
        symbol = row["symbol"].strip()
        try:
            signal_time = pd.to_datetime(row["signal_open_time"], utc=True)
            entry_time = signal_time + pd.Timedelta(days=1)
            exit_time = signal_time + pd.Timedelta(days=2)

            if status == "WAIT_ENTRY":
                entry = get_daily_kline(symbol, entry_time)
                if entry is None:
                    waiting_entry += 1
                    print(f"{symbol:12s} WAIT_ENTRY")
                    continue
                df.at[index, "entry_time"] = str(entry["open_time"])
                df.at[index, "entry_price"] = str(entry["close"])
                df.at[index, "status"] = "WAIT_EXIT"
                status = "WAIT_EXIT"
                entry_updates += 1
                print(f"{symbol:12s} ENTRY = {entry['close']}")

            if status == "WAIT_EXIT":
                entry_text = str(df.at[index, "entry_price"]).strip()
                if not entry_text:
                    entry = get_daily_kline(symbol, entry_time)
                    if entry is None:
                        waiting_entry += 1
                        continue
                    entry_price = float(entry["close"])
                    df.at[index, "entry_time"] = str(entry["open_time"])
                    df.at[index, "entry_price"] = str(entry_price)
                else:
                    entry_price = float(entry_text)

                exit_k = get_daily_kline(symbol, exit_time)
                if exit_k is None:
                    waiting_exit += 1
                    print(f"{symbol:12s} WAIT_EXIT")
                    continue

                exit_price = float(exit_k["close"])
                gross = exit_price / entry_price - 1
                df.at[index, "exit_time"] = str(exit_k["open_time"])
                df.at[index, "exit_price"] = str(exit_price)
                df.at[index, "gross_return"] = str(gross)
                df.at[index, "net_return_cost_0_2"] = str(gross - COSTS["0_2"])
                df.at[index, "net_return_cost_0_3"] = str(gross - COSTS["0_3"])
                df.at[index, "net_return_cost_0_5"] = str(gross - COSTS["0_5"])
                df.at[index, "direction_correct"] = "TRUE" if gross > 0 else "FALSE"
                df.at[index, "closed_at_utc"] = datetime.now(timezone.utc).isoformat()
                df.at[index, "status"] = "CLOSED"
                closed_updates += 1
                print(f"{symbol:12s} CLOSED Gross {gross:+.2%}")
            elif status == "CLOSED":
                already_closed += 1
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
            print(f"{symbol:12s} ERROR: {exc}")

    df.to_csv(SIGNAL_FILE, index=False, encoding="utf-8-sig")
    status_counts = df["status"].value_counts().to_dict()
    closed = df[df["status"] == "CLOSED"].copy()
    stats = {"closed_trades": int(len(closed)), "win_rate_gross": None,
             "average_net_0_2": None, "average_net_0_3": None, "average_net_0_5": None}
    if not closed.empty:
        gross = pd.to_numeric(closed["gross_return"], errors="coerce")
        n02 = pd.to_numeric(closed["net_return_cost_0_2"], errors="coerce")
        n03 = pd.to_numeric(closed["net_return_cost_0_3"], errors="coerce")
        n05 = pd.to_numeric(closed["net_return_cost_0_5"], errors="coerce")
        stats.update({
            "win_rate_gross": float((gross > 0).mean()),
            "average_net_0_2": float(n02.mean()),
            "average_net_0_3": float(n03.mean()),
            "average_net_0_5": float(n05.mean()),
        })

    write_summary({
        "version": "V9-3",
        "settled_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_updates": entry_updates,
        "closed_updates": closed_updates,
        "waiting_entry": waiting_entry,
        "waiting_exit": waiting_exit,
        "already_closed": already_closed,
        "status_counts": status_counts,
        "statistics": stats,
        "errors": errors,
    })

    print("=" * 70)
    print("V9-3 結算完成")
    print(f"新增 ENTRY：{entry_updates}")
    print(f"新增 CLOSED：{closed_updates}")
    print(f"等待 ENTRY：{waiting_entry}")
    print(f"等待 EXIT：{waiting_exit}")
    print(f"目前 CLOSED：{len(closed)}")
    print(f"錯誤：{len(errors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

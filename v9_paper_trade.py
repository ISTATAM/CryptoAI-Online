import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# ============================================================
# CryptoAI V9-2
# Forward Paper Trading Signal Generator
# ============================================================

MODEL_FILE = Path("v9_model/cryptoai_v9_1d.pkl")

DATA_DIR = Path("paper_trading")
SIGNAL_FILE = DATA_DIR / "signals.csv"
LATEST_FILE = DATA_DIR / "latest_scan.csv"
SUMMARY_FILE = DATA_DIR / "latest_summary.json"

BASE_URL = "https://data-api.binance.vision"

INTERVAL = "1d"
KLINE_LIMIT = 200

SIGNAL_THRESHOLD = 0.70
MIN_24H_QUOTE_VOLUME = 1_000_000

EXCLUDED_BASE_ASSETS = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP",
    "DAI", "USD1", "PYUSD", "RLUSD", "USDD",
    "EUR", "TRY", "BRL", "GBP", "AUD", "JPY",
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


def api_get(path, params=None):
    url = BASE_URL + path

    if params:
        url += "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "CryptoAI-V9-PaperTrading/1.0"
        },
    )

    with urlopen(request, timeout=30) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"找不到 V9 正式模型：{MODEL_FILE}"
        )

    with open(MODEL_FILE, "rb") as f:
        package = pickle.load(f)

    model = package["model"]

    model_features = package["features"]

    if model_features != FEATURES:
        raise ValueError(
            "模型特徵與 V9 程式不一致，停止預測。"
        )

    threshold = float(
        package.get(
            "signal_threshold",
            SIGNAL_THRESHOLD
        )
    )

    print("正式模型載入成功")
    print(
        "模型資料 SHA256：",
        package.get("dataset_sha256")
    )
    print(
        f"正式門檻：{threshold:.0%}"
    )

    return model, threshold


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
        (average_loss == 0)
        & (average_gain > 0),
        100
    )

    rsi = rsi.mask(
        (average_gain == 0)
        & (average_loss > 0),
        0
    )

    rsi = rsi.mask(
        (average_gain == 0)
        & (average_loss == 0),
        50
    )

    return rsi


def get_spot_symbols():
    info = api_get("/api/v3/exchangeInfo")

    symbols = []

    for item in info["symbols"]:
        if item["status"] != "TRADING":
            continue

        if item["quoteAsset"] != "USDT":
            continue

        if item["baseAsset"] in EXCLUDED_BASE_ASSETS:
            continue

        permissions = item.get("permissions", [])

        if permissions and "SPOT" not in permissions:
            continue

        symbols.append(item["symbol"])

    return set(symbols)


def get_liquid_symbols():
    valid_symbols = get_spot_symbols()

    tickers = api_get("/api/v3/ticker/24hr")

    rows = []

    for ticker in tickers:
        symbol = ticker.get("symbol")

        if symbol not in valid_symbols:
            continue

        try:
            quote_volume = float(
                ticker["quoteVolume"]
            )
        except Exception:
            continue

        if quote_volume < MIN_24H_QUOTE_VOLUME:
            continue

        rows.append({
            "symbol": symbol,
            "quote_volume_24h":
                quote_volume,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df.sort_values(
            "quote_volume_24h",
            ascending=False
        )
        .reset_index(drop=True)
    )


def download_klines(symbol):
    raw = api_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": KLINE_LIMIT,
        },
    )

    if not raw:
        raise ValueError("沒有 K 線")

    rows = []

    now_ms = int(
        datetime.now(timezone.utc).timestamp()
        * 1000
    )

    for k in raw:
        # Binance close time 位於 index 6。
        # 尚未真正收盤的當日 K 線不能交給模型。
        close_time_ms = int(k[6])

        if close_time_ms >= now_ms:
            continue

        rows.append({
            "open_time":
                pd.to_datetime(
                    int(k[0]),
                    unit="ms",
                    utc=True
                ),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "taker_buy_volume":
                float(k[9]),
        })

    df = pd.DataFrame(rows)

    if len(df) < 60:
        raise ValueError(
            f"完整歷史 K 線不足：{len(df)}"
        )

    return (
        df.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )


def build_features(df):
    df = df.copy()

    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    returns = close.pct_change(
        fill_method=None
    )

    for period in [1, 3, 6, 12, 24]:
        df[f"return_{period}"] = (
            close.pct_change(
                periods=period,
                fill_method=None
            )
        )

    df["rsi_14"] = calculate_rsi(close)

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

    df["macd_pct"] = macd / close

    df["macd_signal_pct"] = (
        macd_signal / close
    )

    df["macd_hist_pct"] = (
        (macd - macd_signal) / close
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(14).mean()

    df["atr_pct"] = atr / close

    df["volatility_6"] = (
        returns.rolling(6).std()
    )

    df["volatility_24"] = (
        returns.rolling(24).std()
    )

    average_volume = (
        volume.rolling(20).mean()
    )

    df["volume_ratio_20"] = (
        volume / average_volume
    )

    df["volume_change"] = (
        volume.pct_change(
            fill_method=None
        )
    )

    df["candle_return"] = (
        close / open_price - 1
    )

    df["high_low_range"] = (
        (high - low) / open_price
    )

    df["taker_buy_ratio"] = (
        df["taker_buy_volume"]
        / volume.replace(0, np.nan)
    )

    df["hour"] = (
        df["open_time"].dt.hour
    )

    df["day_of_week"] = (
        df["open_time"].dt.dayofweek
    )

    return df


def read_existing_signals():
    if not SIGNAL_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(SIGNAL_FILE)
    except Exception:
        return pd.DataFrame()


def save_signals(new_signals):
    existing = read_existing_signals()

    new_df = pd.DataFrame(new_signals)

    if existing.empty:
        combined = new_df
    elif new_df.empty:
        combined = existing
    else:
        combined = pd.concat(
            [existing, new_df],
            ignore_index=True
        )

    if combined.empty:
        return

    # 同一根訊號 K 線、同一幣只能存在一次。
    combined = combined.drop_duplicates(
        subset=[
            "symbol",
            "signal_open_time"
        ],
        keep="first"
    )

    combined = combined.sort_values(
        [
            "signal_open_time",
            "ai_score"
        ],
        ascending=[True, False]
    )

    combined.to_csv(
        SIGNAL_FILE,
        index=False,
        encoding="utf-8-sig"
    )


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("CryptoAI V9-2 Forward Paper Trading")
    print("=" * 70)

    model, threshold = load_model()

    liquid = get_liquid_symbols()

    if liquid.empty:
        raise RuntimeError(
            "沒有取得符合條件的 USDT 現貨"
        )

    print(
        f"目前流動性合格："
        f"{len(liquid)} 幣"
    )

    scan_rows = []
    signal_rows = []

    failures = []

    recorded_at = (
        datetime.now(timezone.utc)
        .isoformat()
    )

    for index, item in liquid.iterrows():
        symbol = item["symbol"]

        try:
            candles = download_klines(
                symbol
            )

            features = build_features(
                candles
            )

            latest = features.iloc[-1]

            values = latest[FEATURES]

            if values.isna().any():
                raise ValueError(
                    "最新 K 線特徵含 NaN"
                )

            if not np.isfinite(
                values.astype(float)
            ).all():
                raise ValueError(
                    "最新 K 線特徵非有限數值"
                )

            x = pd.DataFrame(
                [
                    values.astype(float)
                    .to_dict()
                ],
                columns=FEATURES
            )

            ai_score = float(
                model.predict_proba(x)[0, 1]
            )

            signal_open_time = str(
                latest["open_time"]
            )

            signal_close = float(
                latest["close"]
            )

            live_quote_volume = float(
                item["quote_volume_24h"]
            )

            triggered = (
                ai_score >= threshold
            )

            scan_rows.append({
                "recorded_at_utc":
                    recorded_at,
                "symbol": symbol,
                "signal_open_time":
                    signal_open_time,
                "signal_close":
                    signal_close,
                "ai_score":
                    ai_score,
                "ai_percent":
                    ai_score * 100,
                "quote_volume_24h":
                    live_quote_volume,
                "triggered":
                    triggered,
            })

            if triggered:
                # V8/V9 規則：
                #
                # signal = t completed close
                # entry  = t+1 close
                # exit   = t+2 close
                #
                # 現在只建立 pending 紀錄。
                signal_rows.append({
                    "recorded_at_utc":
                        recorded_at,
                    "interval": "1d",
                    "symbol": symbol,
                    "signal_open_time":
                        signal_open_time,
                    "signal_close":
                        signal_close,
                    "ai_score":
                        ai_score,
                    "quote_volume_24h":
                        live_quote_volume,
                    "threshold":
                        threshold,
                    "status":
                        "WAIT_ENTRY",
                    "entry_time": "",
                    "entry_price": "",
                    "exit_time": "",
                    "exit_price": "",
                    "gross_return": "",
                    "net_return_cost_0_2":
                        "",
                    "net_return_cost_0_3":
                        "",
                    "net_return_cost_0_5":
                        "",
                    "direction_correct":
                        "",
                })

            print(
                f"{symbol:12s} "
                f"AI {ai_score:7.2%}"
                + (
                    "  << SIGNAL"
                    if triggered
                    else ""
                )
            )

            time.sleep(0.05)

        except Exception as exc:
            failures.append({
                "symbol": symbol,
                "error": str(exc)
            })

            print(
                f"{symbol:12s} "
                f"ERROR: {exc}"
            )

    scan_df = pd.DataFrame(scan_rows)

    if not scan_df.empty:
        scan_df = scan_df.sort_values(
            "ai_score",
            ascending=False
        )

        scan_df.to_csv(
            LATEST_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    save_signals(signal_rows)

    summary = {
        "version": "V9-2",
        "recorded_at_utc":
            recorded_at,
        "interval": "1d",
        "threshold":
            threshold,
        "liquid_symbols":
            int(len(liquid)),
        "successful_scans":
            int(len(scan_rows)),
        "failures":
            int(len(failures)),
        "new_signal_candidates":
            int(len(signal_rows)),
        "top_predictions": (
            scan_df.head(10)[
                [
                    "symbol",
                    "ai_score",
                    "signal_open_time"
                ]
            ].to_dict("records")
            if not scan_df.empty
            else []
        ),
        "failed_symbols":
            failures,
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 70)
    print("V9-2 掃描完成")
    print(
        f"成功掃描："
        f"{len(scan_rows)}"
    )
    print(
        f"失敗："
        f"{len(failures)}"
    )
    print(
        f"AI >= {threshold:.0%}："
        f"{len(signal_rows)}"
    )

    if not scan_df.empty:
        print()
        print("目前 AI 前 10 名：")

        for _, row in (
            scan_df.head(10).iterrows()
        ):
            print(
                f"{row['symbol']:12s} "
                f"{row['ai_score']:.2%}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()

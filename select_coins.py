import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================
# CryptoAI Online - 高流動性幣種篩選器
# ============================================================

BASE_URL = "https://data-api.binance.vision"

TARGET_COINS = 100

# 24H 成交額至少 1,000 萬 USDT
MIN_24H_QUOTE_VOLUME = 10_000_000

# 先檢查成交額前 150 名
CANDIDATE_POOL = 150

# 至少需要 90 天歷史資料
MIN_HISTORY_DAYS = 90

# 最近 30 天，至少 24 天成交額達標
MIN_ACTIVE_DAYS = 24

# 每日成交額最低門檻
MIN_DAILY_QUOTE_VOLUME = 3_000_000

OUTPUT_FOLDER = Path("output")

EXCLUDED_ASSETS = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP",
    "DAI", "USD1", "PYUSD", "RLUSD", "USDD",
    "EUR", "TRY", "BRL", "GBP", "AUD", "JPY"
}

def get_json(endpoint, params=""):

    url = BASE_URL + endpoint + params

    request = Request(
        url,
        headers={
            "User-Agent": "CryptoAI-Online/1.0",
            "Accept": "application/json"
        }
    )

    for attempt in range(3):

        try:

            with urlopen(
                request,
                timeout=30
            ) as response:

                return json.load(response)

        except (HTTPError, URLError, TimeoutError):

            if attempt == 2:
                raise

            time.sleep(2 + attempt * 2)


def get_candidates():

    print("取得 Binance 現貨交易對...")

    exchange_info = get_json(
        "/api/v3/exchangeInfo"
    )

    print("取得 24H 成交額...")

    tickers = get_json(
        "/api/v3/ticker/24hr"
    )

    valid_symbols = set()

    for item in exchange_info["symbols"]:

        if (
            item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
            and item.get("isSpotTradingAllowed", False)
            and item["baseAsset"] not in EXCLUDED_ASSETS
        ):

            valid_symbols.add(
                item["symbol"]
            )

    candidates = []

    for ticker in tickers:

        symbol = ticker["symbol"]

        if symbol not in valid_symbols:
            continue

        try:

            quote_volume = float(
                ticker["quoteVolume"]
            )

            price = float(
                ticker["lastPrice"]
            )

            change_24h = float(
                ticker["priceChangePercent"]
            )

        except (ValueError, KeyError, TypeError):
            continue

        if quote_volume < MIN_24H_QUOTE_VOLUME:
            continue

        if price <= 0:
            continue

        candidates.append({
            "symbol": symbol,
            "price": price,
            "quote_volume_24h": quote_volume,
            "change_24h": change_24h
        })

    candidates.sort(
        key=lambda x: x["quote_volume_24h"],
        reverse=True
    )

    print(
        f"符合24H成交額門檻：{len(candidates)} 個"
    )

    return candidates[:CANDIDATE_POOL]


def check_history(symbol):

    # 取得最近約 120 根日 K
    candles = get_json(
        "/api/v3/klines",
        (
            f"?symbol={symbol}"
            "&interval=1d"
            "&limit=120"
        )
    )

    if not candles:

        return False, "沒有歷史K線", {}

    # Binance K線時間使用毫秒
    server_time = get_json(
        "/api/v3/time"
    )["serverTime"]

    # 排除尚未收盤的日K
    completed = [
        candle
        for candle in candles
        if candle[6] < server_time
    ]

    if len(completed) < MIN_HISTORY_DAYS:

        return (
            False,
            f"歷史不足：{len(completed)} 天",
            {}
        )

    recent_30 = completed[-30:]

    daily_volumes = [
        float(candle[7])
        for candle in recent_30
    ]

    active_days = sum(
        volume >= MIN_DAILY_QUOTE_VOLUME
        for volume in daily_volumes
    )

    if active_days < MIN_ACTIVE_DAYS:

        return (
            False,
            f"成交額不穩定：{active_days}/30 天達標",
            {}
        )

    average_volume = (
        sum(daily_volumes)
        / len(daily_volumes)
    )

    return True, "PASS", {
        "history_days_checked": len(completed),
        "active_days_30d": active_days,
        "average_daily_quote_volume_30d": average_volume
    }


def save_csv(filename, rows, fieldnames):

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    filepath = OUTPUT_FOLDER / filename

    with filepath.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"已儲存：{filepath}")


def main():

    print("=" * 75)

    print(
        "CryptoAI Online V1 - 高流動性幣種篩選"
    )

    print("=" * 75)

    print(
        f"目標幣數：{TARGET_COINS}"
    )

    print(
        f"最低24H成交額："
        f"{MIN_24H_QUOTE_VOLUME:,.0f} USDT"
    )

    print(
        f"最低歷史天數：{MIN_HISTORY_DAYS}"
    )

    print()

    candidates = get_candidates()

    selected = []

    rejected = []

    for index, coin in enumerate(
        candidates,
        start=1
    ):

        if len(selected) >= TARGET_COINS:
            break

        symbol = coin["symbol"]

        print(
            f"[{index}/{len(candidates)}] "
            f"檢查 {symbol}..."
        )

        try:

            passed, reason, stats = check_history(
                symbol
            )

            if not passed:

                print(
                    f"  跳過：{reason}"
                )

                rejected.append({
                    "symbol": symbol,
                    "reason": reason
                })

                continue

            result = {
                "rank": len(selected) + 1,
                **coin,
                **stats
            }

            selected.append(result)

            print(
                f"  PASS！"
                f"成功 {len(selected)}/{TARGET_COINS}"
            )

        except Exception as error:

            reason = (
                f"{type(error).__name__}: {error}"
            )

            print(
                f"  ERROR：{reason}"
            )

            rejected.append({
                "symbol": symbol,
                "reason": reason
            })

        time.sleep(0.1)

    print()

    print("=" * 75)

    print("篩選完成")

    print("=" * 75)

    print(
        f"成功：{len(selected)}"
    )

    print(
        f"跳過：{len(rejected)}"
    )

    print()

    for coin in selected:

        print(
            f"{coin['rank']:3d}. "
            f"{coin['symbol']:15s} "
            f"24H成交額："
            f"{coin['quote_volume_24h']:,.0f} USDT "
            f"｜30日平均："
            f"{coin['average_daily_quote_volume_30d']:,.0f} USDT"
        )

    selected_fields = [
        "rank",
        "symbol",
        "price",
        "quote_volume_24h",
        "change_24h",
        "history_days_checked",
        "active_days_30d",
        "average_daily_quote_volume_30d"
    ]

    save_csv(
        "selected_coins.csv",
        selected,
        selected_fields
    )

    save_csv(
        "rejected_coins.csv",
        rejected,
        ["symbol", "reason"]
    )

    summary = {
        "scan_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "target_coins": TARGET_COINS,

        "selected_count": len(selected),

        "rejected_count": len(rejected),

        "minimum_24h_quote_volume":
            MIN_24H_QUOTE_VOLUME,

        "minimum_history_days":
            MIN_HISTORY_DAYS
    }

    with (
        OUTPUT_FOLDER / "summary.json"
    ).open(
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

    print("所有結果已儲存到 output 資料夾")

    print("=" * 75)


if __name__ == "__main__":
    main()

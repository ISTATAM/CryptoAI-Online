
import csv
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# ============================================================
# CryptoAI Online V2
# Binance 雲端歷史資料下載
# ============================================================

BASE_URL = "https://data-api.binance.vision"

SELECTED_FILE = Path("output/selected_coins.csv")

DATA_FOLDER = Path("history_data")

ZIP_FILE = Path("output/cryptoai_history.zip")

# 第一階段：先驗證雲端下載流程
INTERVALS = {
    "15m": 30,
    "4h": 365,
    "1d": 730,
}

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

KLINE_LIMIT = 1000

MAX_RETRIES = 5

REQUEST_DELAY = 0.12

FIELDS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]


def get_json(endpoint, params=None):

    url = BASE_URL + endpoint

    if params:
        url += "?" + urlencode(params)

    last_error = None

    for attempt in range(MAX_RETRIES):

        request = Request(
            url,
            headers={
                "User-Agent": "CryptoAI-Online/2.0",
                "Accept": "application/json",
            },
        )

        try:

            with urlopen(
                request,
                timeout=40,
            ) as response:

                return json.load(response)

        except HTTPError as error:

            last_error = error

            # 地區限制或無效請求，不盲目重試
            if error.code in (400, 403, 451):
                raise

            if error.code == 429:

                wait = 10 * (attempt + 1)

                print(
                    f"API 限流，等待 {wait} 秒",
                    flush=True,
                )

                time.sleep(wait)

            else:

                time.sleep(
                    2 * (attempt + 1)
                )

        except (
            URLError,
            TimeoutError,
        ) as error:

            last_error = error

            time.sleep(
                2 * (attempt + 1)
            )

    raise RuntimeError(
        f"API 多次請求失敗：{last_error}"
    )


def load_symbols():

    if not SELECTED_FILE.exists():

        raise FileNotFoundError(
            "找不到 output/selected_coins.csv，"
            "請先執行 select_coins.py"
        )

    symbols = []

    with SELECTED_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            symbol = row["symbol"].strip()

            if symbol:
                symbols.append(symbol)

    # 去除重複，保留原本排名
    symbols = list(
        dict.fromkeys(symbols)
    )

    return symbols


def get_server_time():

    result = get_json(
        "/api/v3/time"
    )

    return int(
        result["serverTime"]
    )


def download_klines(
    symbol,
    interval,
    days,
    server_time,
):

    interval_ms = INTERVAL_MS[interval]

    start_time = (
        server_time
        - days * 24 * 60 * 60 * 1000
    )

    all_rows = []

    while start_time < server_time:

        candles = get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": server_time,
                "limit": KLINE_LIMIT,
            },
        )

        if not candles:
            break

        for candle in candles:

            # 排除尚未收盤的 K 線
            if int(candle[6]) >= server_time:
                continue

            all_rows.append(candle)

        last_open_time = int(
            candles[-1][0]
        )

        next_start = (
            last_open_time
            + interval_ms
        )

        if next_start <= start_time:
            break

        start_time = next_start

        if len(candles) < KLINE_LIMIT:
            break

        time.sleep(REQUEST_DELAY)

    # 避免 API 分頁產生重複 K 線
    unique = {
        int(row[0]): row
        for row in all_rows
    }

    return [
        unique[key]
        for key in sorted(unique)
    ]


def save_klines(
    symbol,
    interval,
    candles,
):

    folder = (
        DATA_FOLDER / interval
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    filepath = (
        folder
        / f"{symbol}_{interval}.csv"
    )

    with filepath.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(FIELDS)

        for candle in candles:

            writer.writerow([
                datetime.fromtimestamp(
                    int(candle[0]) / 1000,
                    tz=timezone.utc,
                ).isoformat(),

                candle[1],
                candle[2],
                candle[3],
                candle[4],
                candle[5],

                datetime.fromtimestamp(
                    int(candle[6]) / 1000,
                    tz=timezone.utc,
                ).isoformat(),

                candle[7],
                candle[8],
                candle[9],
                candle[10],
            ])

    return filepath


def create_zip():

    ZIP_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:

        for filepath in (
            DATA_FOLDER.rglob("*.csv")
        ):

            archive.write(
                filepath,
                arcname=str(filepath),
            )

    return (
        ZIP_FILE.stat().st_size
        / 1024
        / 1024
    )


def main():

    print("=" * 65)
    print("CryptoAI Online V2")
    print("雲端歷史 K 線下載")
    print("=" * 65)

    symbols = load_symbols()

    print(
        f"本次候選幣：{len(symbols)}"
    )

    if not symbols:

        raise RuntimeError(
            "沒有可下載的幣種"
        )

    server_time = get_server_time()

    print(
        "Binance UTC 時間：",
        datetime.fromtimestamp(
            server_time / 1000,
            tz=timezone.utc,
        ).isoformat(),
    )

    DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for interval, days in (
        INTERVALS.items()
    ):

        print()
        print("=" * 65)

        print(
            f"週期：{interval}"
            f"｜歷史：{days} 天"
        )

        print("=" * 65)

        for index, symbol in enumerate(
            symbols,
            start=1,
        ):

            print(
                f"[{index}/{len(symbols)}] "
                f"{symbol} {interval}",
                flush=True,
            )

            try:

                candles = download_klines(
                    symbol,
                    interval,
                    days,
                    server_time,
                )

                if not candles:

                    print(
                        "  跳過：沒有完整 K 線"
                    )

                    results.append({
                        "symbol": symbol,
                        "interval": interval,
                        "rows": 0,
                        "status": "NO_DATA",
                        "error": "",
                    })

                    continue

                filepath = save_klines(
                    symbol,
                    interval,
                    candles,
                )

                print(
                    f"  OK：{len(candles):,} 根"
                )

                results.append({
                    "symbol": symbol,
                    "interval": interval,
                    "rows": len(candles),
                    "status": "SUCCESS",
                    "error": "",
                })

            except Exception as error:

                print(
                    f"  ERROR：{error}"
                )

                results.append({
                    "symbol": symbol,
                    "interval": interval,
                    "rows": 0,
                    "status": "FAILED",
                    "error": str(error),
                })

    OUTPUT_FOLDER = Path("output")

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        OUTPUT_FOLDER
        / "history_download_report.csv"
    )

    with report_file.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "symbol",
                "interval",
                "rows",
                "status",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    success = sum(
        row["status"] == "SUCCESS"
        for row in results
    )

    failed = sum(
        row["status"] == "FAILED"
        for row in results
    )

    no_data = sum(
        row["status"] == "NO_DATA"
        for row in results
    )

    total_rows = sum(
        row["rows"]
        for row in results
    )

    print()
    print("=" * 65)
    print("下載結果")
    print("=" * 65)

    print(
        f"成功檔案：{success}"
    )

    print(
        f"失敗檔案：{failed}"
    )

    print(
        f"無資料檔案：{no_data}"
    )

    print(
        f"總 K 線：{total_rows:,}"
    )

    print()
    print("開始壓縮歷史資料...")

    zip_size = create_zip()

    print(
        f"ZIP 大小：{zip_size:.2f} MB"
    )

    print(
        f"檔案位置：{ZIP_FILE}"
    )

    print("=" * 65)

    if success == 0:

        raise RuntimeError(
            "所有歷史資料下載皆失敗"
        )


if __name__ == "__main__":
    main()

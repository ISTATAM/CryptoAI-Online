import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen

BASE_URL = "https://api.binance.com"

MIN_24H_VOLUME = 10_000_000
TOP_N = 100

EXCLUDED_ASSETS = {
    "USDT", "USDC", "FDUSD", "TUSD", "USDP",
    "DAI", "EUR", "TRY", "BRL", "GBP",
    "AUD", "JPY"
}


def get_json(endpoint):
    request = Request(
        BASE_URL + endpoint,
        headers={"User-Agent": "CryptoAI-Online/1.0"}
    )

    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    print("=" * 60)
    print("CryptoAI Online - Binance API 測試")
    print("=" * 60)

    exchange_info = get_json("/api/v3/exchangeInfo")
    tickers = get_json("/api/v3/ticker/24hr")

    valid_symbols = set()

    for item in exchange_info["symbols"]:
        if (
            item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
            and item.get("isSpotTradingAllowed", False)
            and item["baseAsset"] not in EXCLUDED_ASSETS
        ):
            valid_symbols.add(item["symbol"])

    coins = []

    for ticker in tickers:
        symbol = ticker["symbol"]

        if symbol not in valid_symbols:
            continue

        volume = float(ticker["quoteVolume"])

        if volume < MIN_24H_VOLUME:
            continue

        coins.append({
            "symbol": symbol,
            "price": float(ticker["lastPrice"]),
            "volume": volume
        })

    coins.sort(
        key=lambda coin: coin["volume"],
        reverse=True
    )

    selected = coins[:TOP_N]

    print(f"符合最低成交額條件：{len(coins)} 個")
    print(f"最終選取：{len(selected)} 個")
    print()

    for index, coin in enumerate(selected, start=1):
        print(
            f"{index:3d}. "
            f"{coin['symbol']:15s} "
            f"價格：{coin['price']:.8f} "
            f"24H成交額：{coin['volume']:,.0f} USDT"
        )

    print()
    print("測試完成！")
    print(
        "UTC時間：",
        datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    main()

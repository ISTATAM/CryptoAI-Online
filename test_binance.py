import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_URL = "https://data-api.binance.vision"


def get_json(endpoint):

    url = BASE_URL + endpoint

    request = Request(
        url,
        headers={
            "User-Agent": "CryptoAI-Online/1.0",
            "Accept": "application/json"
        }
    )

    with urlopen(request, timeout=30) as response:
        return json.load(response)


def test_api(name, endpoint):

    print()
    print("=" * 60)
    print("測試：", name)

    try:

        data = get_json(endpoint)

        print("結果：SUCCESS")

        if isinstance(data, list):

            print("取得資料數量：", len(data))

            if len(data) > 0:
                print("第一筆資料：", data[0])

        elif isinstance(data, dict):

            if "symbols" in data:

                print(
                    "交易對數量：",
                    len(data["symbols"])
                )

            else:

                print("資料：", data)

    except HTTPError as error:

        print("結果：FAILED")
        print("HTTP：", error.code)

        print(
            error.read().decode(
                "utf-8",
                errors="replace"
            )[:500]
        )

    except Exception as error:

        print("結果：FAILED")
        print("原因：", str(error))


def main():

    print("CryptoAI Online - Binance 市場資料測試")

    test_api(
        "BTC 15分鐘 K線",
        "/api/v3/klines"
        "?symbol=BTCUSDT"
        "&interval=15m"
        "&limit=5"
    )

    test_api(
        "BTC 4小時 K線",
        "/api/v3/klines"
        "?symbol=BTCUSDT"
        "&interval=4h"
        "&limit=5"
    )

    test_api(
        "全部交易對",
        "/api/v3/exchangeInfo"
    )

    test_api(
        "全部幣種24H成交額",
        "/api/v3/ticker/24hr"
    )

    print()
    print("=" * 60)
    print("市場資料測試完成")


if __name__ == "__main__":
    main()

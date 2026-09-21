import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API_TESTS = {
    "Binance Global": (
        "https://api.binance.com/api/v3/time"
    ),
    "Binance Data": (
        "https://data-api.binance.vision/api/v3/time"
    ),
    "Binance US": (
        "https://api.binance.us/api/v3/time"
    ),
    "CoinGecko": (
        "https://api.coingecko.com/api/v3/ping"
    ),
}


def test_api(name, url):

    print()
    print("=" * 50)
    print(f"測試：{name}")
    print(f"網址：{url}")

    request = Request(
        url,
        headers={
            "User-Agent": "CryptoAI-Online/1.0",
            "Accept": "application/json",
        },
    )

    try:

        with urlopen(request, timeout=20) as response:

            data = json.load(response)

            print("結果：SUCCESS")
            print("HTTP：", response.status)
            print("資料：", data)

    except HTTPError as error:

        print("結果：FAILED")
        print("HTTP：", error.code)

        try:
            print(
                "原因：",
                error.read().decode(
                    "utf-8",
                    errors="replace"
                )[:500]
            )
        except Exception:
            pass

    except URLError as error:

        print("結果：FAILED")
        print("原因：", error.reason)

    except Exception as error:

        print("結果：FAILED")
        print("原因：", str(error))


def main():

    print("CryptoAI Online - 雲端 API 連線診斷")

    for name, url in API_TESTS.items():

        test_api(name, url)

    print()
    print("=" * 50)
    print("全部測試完成")


if __name__ == "__main__":
    main()

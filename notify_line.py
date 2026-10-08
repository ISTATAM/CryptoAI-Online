"""掃描完成後，由 GitHub 雲端執行 LINE 通知。"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    url = os.environ.get("V10_NOTIFY_URL", "").strip()
    secret = os.environ.get("V10_NOTIFY_SECRET", "").strip()

    if (
        not url.startswith("https://")
        or not url.endswith("/notify")
        or not secret
    ):
        raise ValueError(
            "請設定 V10_NOTIFY_URL 與 V10_NOTIFY_SECRET；"
            "網址必須使用 HTTPS 並以 /notify 結尾"
        )

    path = Path(
        os.environ.get(
            "V10_SUMMARY_PATH",
            "paper_trading/latest_summary.json",
        )
    )

    summary = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(summary.get("top_predictions"), list):
        raise ValueError("top_predictions 必須為陣列")

    stamp = datetime.fromisoformat(
        summary["recorded_at_utc"].replace("Z", "+00:00")
    )

    if stamp.tzinfo is None:
        raise ValueError("recorded_at_utc 必須包含時區")

    age = (datetime.now(timezone.utc) - stamp).total_seconds()

    if not -600 <= age <= 36 * 3600:
        raise ValueError("掃描資料已過期，或時間在未來")

    payload = json.dumps(
        {"summary": summary},
        ensure_ascii=False,
    ).encode("utf-8")

    for attempt in range(5):
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {secret}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                result = json.load(response)

            if result.get("ok") is not True:
                raise ValueError("Worker 未回傳 ok=true")

            print(
                "V10 通知完成：",
                json.dumps(result, ensure_ascii=False),
            )
            return

        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode(
                "utf-8",
                errors="replace",
            )
            print(
                f"通知 HTTP {error.code}：{detail}",
                file=sys.stderr,
            )

            if error.code not in (
                408, 429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"通知設定或資料錯誤 HTTP {error.code}"
                ) from None

        except (urllib.error.URLError, TimeoutError):
            print(
                "通知連線暫時失敗，稍後重試。",
                file=sys.stderr,
            )

        if attempt < 4:
            time.sleep(min(2 ** (attempt + 1), 16))

    raise RuntimeError(
        "通知重試後仍失敗；請查看 Worker Logs"
    )


if __name__ == "__main__":
    main()

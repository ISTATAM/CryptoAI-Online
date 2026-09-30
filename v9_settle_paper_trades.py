name: CryptoAI V9 Paper Trading

on:
  workflow_dispatch:

  schedule:
    # Binance 1D K線 UTC 00:00 收盤後，
    # 預留 15 分鐘再執行。
    - cron: '15 0 * * *'

permissions:
  actions: read
  contents: write

concurrency:
  group: cryptoai-v9-paper-trading
  cancel-in-progress: false

jobs:
  paper-trade:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Download project
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install packages
        run: |
          python -m pip install --upgrade pip
          python -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "scikit-learn>=1.5,<2"

      - name: Find V9 production model
        id: find-model
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          RUN_ID=$(gh run list \
            --workflow "build_v9_model.yml" \
            --status success \
            --limit 1 \
            --json databaseId \
            --jq '.[0].databaseId')

          if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
            echo "找不到成功的 V9 Production Model"
            exit 1
          fi

          echo "V9 Model Run ID: $RUN_ID"
          echo "run_id=$RUN_ID" >> "$GITHUB_OUTPUT"

      - name: Download locked V9 model
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p v9_model

          gh run download \
            "${{ steps.find-model.outputs.run_id }}" \
            --name cryptoai-v9-production-model \
            --dir downloaded_model

          cp -r downloaded_model/. v9_model/

          echo "正式模型："
          ls -lh v9_model/

          echo ""
          echo "模型 SHA256："
          cat v9_model/model_sha256.txt

      # ======================================================
      # 先處理昨天以前留下來的 Paper Trade
      # ======================================================

      - name: Settle existing paper trades
        run: python v9_settle_paper_trades.py

      # ======================================================
      # 再掃描今天最新完成的 1D K線
      # ======================================================

      - name: Run V9 paper trading scanner
        run: python v9_paper_trade.py

      # ======================================================
      # 再結算一次
      #
      # 正常新訊號不會立刻成交。
      # 這一步主要用來增加中斷恢復能力。
      # ======================================================

      - name: Recheck paper trade settlement
        run: python v9_settle_paper_trades.py

      # ======================================================
      # 永久寫回 GitHub
      # ======================================================

      - name: Save paper trading history
        run: |
          git config user.name "CryptoAI V9 Bot"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          git add paper_trading/

          if git diff --cached --quiet; then
            echo "沒有新的 Paper Trading 資料"
            exit 0
          fi

          git commit -m "Update V9 paper trading data"

          git pull --rebase origin main

          git push origin main

      - name: Upload latest V9 report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: cryptoai-v9-paper-trading-latest
          path: |
            paper_trading/
          if-no-files-found: warn
          retention-days: 30

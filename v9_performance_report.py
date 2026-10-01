import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path("paper_trading")
SIGNAL_FILE = DATA_DIR / "signals.csv"
PERFORMANCE_CSV = DATA_DIR / "performance_trades.csv"
SUMMARY_FILE = DATA_DIR / "performance_summary.json"
INITIAL_CAPITAL = 100.0
MAX_POSITIONS = 5
POSITION_FRACTION = 0.20
COST_COLUMNS = {
    "0.2%": "net_return_cost_0_2",
    "0.3%": "net_return_cost_0_3",
    "0.5%": "net_return_cost_0_5",
}


def safe_float_series(series):
    return pd.to_numeric(series, errors="coerce")


def max_drawdown(equity):
    if not equity:
        return 0.0
    s = pd.Series(equity, dtype=float)
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())


def build_curve(closed, return_col):
    capital = INITIAL_CAPITAL
    curve = [capital]
    rows = []
    work = closed.copy()
    work["_ret"] = safe_float_series(work[return_col])
    work["_exit"] = pd.to_datetime(work["exit_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_ret", "_exit"]).sort_values(["_exit", "ai_score"], ascending=[True, False])

    for exit_time, group in work.groupby("_exit", sort=True):
        selected = group.sort_values("ai_score", ascending=False).head(MAX_POSITIONS)
        start_capital = capital
        pnl = 0.0
        for _, row in selected.iterrows():
            allocation = start_capital * POSITION_FRACTION
            trade_pnl = allocation * float(row["_ret"])
            pnl += trade_pnl
            rows.append({
                "symbol": row["symbol"],
                "signal_open_time": row["signal_open_time"],
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "ai_score": float(row["ai_score"]),
                "return_column": return_col,
                "net_return": float(row["_ret"]),
                "allocated_usdt": allocation,
                "pnl_usdt": trade_pnl,
            })
        capital += pnl
        curve.append(capital)
    return capital, curve, rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    if not SIGNAL_FILE.exists():
        payload = {
            "version": "V9-4",
            "updated_at_utc": now,
            "message": "No signals.csv yet",
            "initial_capital_usdt": INITIAL_CAPITAL,
            "total_signals": 0,
            "closed_trades": 0,
        }
        SUMMARY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("V9-4：目前沒有 signals.csv。")
        return

    df = pd.read_csv(SIGNAL_FILE, dtype=str, keep_default_na=False)
    total = len(df)
    status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
    closed = df[df["status"] == "CLOSED"].copy() if "status" in df.columns else pd.DataFrame()
    results = {}
    export_rows = []

    for label, col in COST_COLUMNS.items():
        if col not in closed.columns or closed.empty:
            results[label] = {
                "final_capital_usdt": INITIAL_CAPITAL,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "trades_used": 0,
                "win_rate_pct": None,
                "average_trade_return_pct": None,
            }
            continue

        r = safe_float_series(closed[col]).dropna()
        final_capital, curve, rows = build_curve(closed, col)
        export_rows.extend(rows)
        results[label] = {
            "final_capital_usdt": round(final_capital, 4),
            "total_return_pct": round((final_capital / INITIAL_CAPITAL - 1) * 100, 4),
            "max_drawdown_pct": round(max_drawdown(curve) * 100, 4),
            "trades_used": int(len(rows)),
            "win_rate_pct": round(float((r > 0).mean()) * 100, 4) if len(r) else None,
            "average_trade_return_pct": round(float(r.mean()) * 100, 4) if len(r) else None,
        }

    if export_rows:
        pd.DataFrame(export_rows).to_csv(PERFORMANCE_CSV, index=False, encoding="utf-8-sig")

    payload = {
        "version": "V9-4",
        "updated_at_utc": now,
        "rules": {
            "initial_capital_usdt": INITIAL_CAPITAL,
            "max_positions_per_exit_time": MAX_POSITIONS,
            "position_fraction_of_starting_equity": POSITION_FRACTION,
            "note": "Forward paper-trading performance; not live executable fills.",
        },
        "total_signals": int(total),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "closed_trades": int(len(closed)),
        "performance_by_cost": results,
    }
    SUMMARY_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("CryptoAI V9-4 Forward Performance")
    print(f"總訊號：{total} | CLOSED：{len(closed)}")
    for label, item in results.items():
        print(f"成本 {label}: {item['final_capital_usdt']:.2f} USDT | 報酬 {item['total_return_pct']:+.2f}% | DD {item['max_drawdown_pct']:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcp_trader.data.binance import get_futures_klines, get_funding_rates


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main():
    p = argparse.ArgumentParser(description="Fetch Binance futures data and cache to Parquet")
    p.add_argument("symbol", help="e.g., BTCUSDT")
    p.add_argument("start", help="ISO datetime UTC, e.g., 2024-01-01T00:00:00")
    p.add_argument("end", help="ISO datetime UTC, e.g., 2025-01-01T00:00:00")
    p.add_argument("--interval", default="1h", help="Kline interval, e.g., 1m,5m,1h,4h,1d")
    p.add_argument("--outdir", default="data", help="Output directory for Parquet files")
    args = p.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Fetching klines...")
    kl = get_futures_klines(args.symbol, args.interval, start, end)
    kpath = outdir / f"{args.symbol}_{args.interval}_klines.parquet"
    if not kl.empty:
        kl.to_parquet(kpath, index=False)
        print(f"Saved {len(kl)} klines -> {kpath}")
    else:
        print("No kline data returned.")

    print("Fetching funding rates...")
    fr = get_funding_rates(args.symbol, start, end)
    fpath = outdir / f"{args.symbol}_funding.parquet"
    if not fr.empty:
        fr.to_parquet(fpath, index=False)
        print(f"Saved {len(fr)} funding rows -> {fpath}")
    else:
        print("No funding data returned.")


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcp_trader.data.okx import get_candles, get_funding_rate_history


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main():
    p = argparse.ArgumentParser(description="Fetch OKX public data and cache to Parquet")
    p.add_argument("inst_id", help="e.g., BTC-USDT-SWAP")
    p.add_argument("start", help="ISO datetime UTC, e.g., 2024-01-01T00:00:00")
    p.add_argument("end", help="ISO datetime UTC, e.g., 2025-01-01T00:00:00")
    p.add_argument("--bar", default="1H", help="Candle bar size, e.g., 1m,5m,1H,4H,1D")
    p.add_argument("--outdir", default="data", help="Output directory")
    args = p.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Fetching candles...")
    candles = get_candles(args.inst_id, args.bar, start, end)
    cpath = outdir / f"{args.inst_id}_{args.bar}_okx_candles.parquet"
    if not candles.empty:
        candles.to_parquet(cpath, index=False)
        print(f"Saved {len(candles)} candles -> {cpath}")
    else:
        print("No candle data returned.")

    print("Fetching funding rate history...")
    fr = get_funding_rate_history(args.inst_id, start, end)
    fpath = outdir / f"{args.inst_id}_okx_funding.parquet"
    if not fr.empty:
        fr.to_parquet(fpath, index=False)
        print(f"Saved {len(fr)} funding rows -> {fpath}")
    else:
        print("No funding data returned.")


if __name__ == "__main__":
    main()

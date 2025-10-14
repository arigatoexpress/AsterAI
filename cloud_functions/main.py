"""
Cloud Function for scheduled market data ingestion.
Deploys to GCP Cloud Functions with Cloud Scheduler trigger.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from mcp_trader.data.bigquery_client import BigQueryClient
from mcp_trader.data.binance import get_futures_klines, get_funding_rates
from mcp_trader.data.okx import get_candles, get_funding_rate_history
from mcp_trader.config import get_pairs_with_external_data, PRIORITY_SYMBOLS


def ingest_market_data(event: Dict[str, Any], context) -> str:
    """
    Cloud Function entry point for market data ingestion.
    Triggered by Cloud Scheduler every hour.
    """
    project_id = os.environ.get("GCP_PROJECT")
    dataset_id = os.environ.get("BIGQUERY_DATASET", "market_data")
    
    if not project_id:
        raise ValueError("GCP_PROJECT environment variable not set")
    
    print(f"Starting market data ingestion for project {project_id}")
    
    client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)
    
    # Get current time and last hour window
    now = datetime.now(timezone.utc)
    end_time = now.replace(minute=0, second=0, microsecond=0)  # Top of hour
    start_time = end_time - timedelta(hours=1)
    
    print(f"Fetching data from {start_time} to {end_time}")
    
    # Get Aster/Jewster trading pairs with external data sources
    trading_pairs = get_pairs_with_external_data()
    
    # Filter to priority symbols for cost efficiency
    priority_pairs = [pair for pair in trading_pairs if pair.symbol in PRIORITY_SYMBOLS]
    
    # Create symbol/venue mapping for ingestion
    symbols = []
    for pair in priority_pairs:
        if pair.binance_symbol:
            symbols.append((pair.binance_symbol, "binance", pair.symbol))
        if pair.okx_symbol:
            symbols.append((pair.okx_symbol, "okx", pair.symbol))
    
    results = []
    
    for external_symbol, venue, aster_symbol in symbols:
        try:
            print(f"Processing {aster_symbol} ({external_symbol} from {venue})...")
            
            # Check if we already have data for this period
            latest_ts = client.get_latest_timestamp(aster_symbol, venue, "1h" if venue == "binance" else "1H")
            if latest_ts and latest_ts >= start_time:
                print(f"Data already exists for {aster_symbol} at {latest_ts}, skipping...")
                continue
            
            if venue == "binance":
                # Fetch Binance data
                klines = get_futures_klines(external_symbol, "1h", start_time, end_time, limit=100)
                if not klines.empty:
                    client.insert_ohlcv_data(klines, aster_symbol, venue, "1h")
                    results.append(f"✅ {aster_symbol} (binance): {len(klines)} OHLCV rows")
                
                funding = get_funding_rates(external_symbol, start_time, end_time, limit=100)
                if not funding.empty:
                    client.insert_funding_data(funding, aster_symbol, venue)
                    results.append(f"✅ {aster_symbol} (binance): {len(funding)} funding rows")
                    
            elif venue == "okx":
                # Fetch OKX data
                candles = get_candles(external_symbol, "1H", start_time, end_time, limit=100)
                if not candles.empty:
                    client.insert_ohlcv_data(candles, aster_symbol, venue, "1H")
                    results.append(f"✅ {aster_symbol} (okx): {len(candles)} OHLCV rows")
                
                funding = get_funding_rate_history(external_symbol, start_time, end_time, limit=100)
                if not funding.empty:
                    client.insert_funding_data(funding, aster_symbol, venue)
                    results.append(f"✅ {aster_symbol} (okx): {len(funding)} funding rows")
                    
        except Exception as e:
            error_msg = f"❌ Error processing {aster_symbol} ({external_symbol} from {venue}): {str(e)}"
            print(error_msg)
            results.append(error_msg)
    
    result_summary = "\n".join(results)
    print(f"Ingestion complete:\n{result_summary}")
    
    return result_summary


# For local testing
if __name__ == "__main__":
    # Test the function locally
    test_event = {}
    test_context = type('Context', (), {})()
    result = ingest_market_data(test_event, test_context)
    print(result)

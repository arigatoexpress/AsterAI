#!/usr/bin/env python3
"""
Setup GCP infrastructure for market data pipeline.
Run this once to create BigQuery tables and Cloud Functions.
"""

import argparse
from datetime import datetime, timezone, timedelta

from mcp_trader.data.bigquery_client import BigQueryClient


def setup_bigquery_tables(project_id: str, dataset_id: str = "market_data"):
    """Create BigQuery dataset and tables."""
    print(f"Setting up BigQuery in project {project_id}...")
    
    client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)
    client.ensure_dataset_exists()
    client.ensure_ohlcv_table_exists()
    client.ensure_funding_table_exists()
    
    print("✅ BigQuery setup complete!")


def initial_data_load(project_id: str, dataset_id: str = "market_data"):
    """Placeholder for initial data load.

    Aster DEX historical data ingestion will be handled by the live data
    pipeline. Skipping external exchange bootstrap to keep the system
    Aster-only per current scope.
    """
    print("⏭️  Skipping initial historical data load (Aster-only scope)")


def main():
    parser = argparse.ArgumentParser(description="Setup GCP market data pipeline")
    parser.add_argument("project_id", help="GCP Project ID")
    parser.add_argument("--dataset", default="market_data", help="BigQuery dataset name")
    parser.add_argument("--skip-data", action="store_true", help="Skip initial data load")
    args = parser.parse_args()
    
    # Setup BigQuery
    setup_bigquery_tables(args.project_id, args.dataset)
    
    # Load initial data
    if not args.skip_data:
        initial_data_load(args.project_id, args.dataset)
    
    print("\n🎉 GCP setup complete!")
    print(f"Next steps:")
    print(f"1. Deploy Cloud Functions: gcloud functions deploy ingest-market-data")
    print(f"2. Setup Cloud Scheduler: python scripts/setup_scheduler.py {args.project_id}")
    print(f"3. View data in BigQuery console")


if __name__ == "__main__":
    main()

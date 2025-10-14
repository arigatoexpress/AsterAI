#!/usr/bin/env python3
"""
Data quality monitoring dashboard.
Checks freshness, completeness, and alerts on issues.
"""

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any
from google.cloud import bigquery

from mcp_trader.data.bigquery_client import BigQueryClient
from mcp_trader.config import PRIORITY_SYMBOLS


class DataQualityMonitor:
    """Monitor data quality metrics."""
    
    def __init__(self, project_id: str, dataset_id: str = "market_data"):
        self.client = BigQueryClient(project_id=project_id, dataset_id=dataset_id)
        self.bq = self.client.client
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.alerts: List[str] = []
    
    def check_data_freshness(self, max_age_hours: int = 2) -> Dict[str, Any]:
        """Check how recent the data is for each symbol."""
        query = f"""
        SELECT 
          symbol,
          venue,
          MAX(timestamp) as latest_timestamp,
          TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(timestamp), HOUR) as hours_old
        FROM `{self.project_id}.{self.dataset_id}.ohlcv`
        WHERE symbol IN UNNEST(@priority_symbols)
        GROUP BY symbol, venue
        ORDER BY hours_old DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("priority_symbols", "STRING", PRIORITY_SYMBOLS)
            ]
        )
        
        results = self.bq.query(query, job_config=job_config).to_dataframe()
        
        stale_data = results[results['hours_old'] > max_age_hours]
        if len(stale_data) > 0:
            for _, row in stale_data.iterrows():
                self.alerts.append(
                    f"🚨 Stale data: {row['symbol']} on {row['venue']} "
                    f"is {row['hours_old']:.1f} hours old"
                )
        
        return {
            'status': 'OK' if len(stale_data) == 0 else 'WARNING',
            'stale_count': len(stale_data),
            'results': results.to_dict('records')
        }
    
    def check_data_completeness(self, days_back: int = 7) -> Dict[str, Any]:
        """Check for missing hourly candles."""
        query = f"""
        WITH date_range AS (
          SELECT 
            symbol,
            venue,
            GENERATE_TIMESTAMP_ARRAY(
              TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY),
              CURRENT_TIMESTAMP(),
              INTERVAL 1 HOUR
            ) as expected_timestamps
          FROM (
            SELECT DISTINCT symbol, venue 
            FROM `{self.project_id}.{self.dataset_id}.ohlcv`
            WHERE symbol IN UNNEST(@priority_symbols)
          )
        ),
        actual_data AS (
          SELECT DISTINCT
            symbol,
            venue,
            TIMESTAMP_TRUNC(timestamp, HOUR) as hour
          FROM `{self.project_id}.{self.dataset_id}.ohlcv`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY)
            AND interval = '1h'
            AND symbol IN UNNEST(@priority_symbols)
        )
        SELECT 
          dr.symbol,
          dr.venue,
          ARRAY_LENGTH(dr.expected_timestamps) as expected_count,
          COUNT(DISTINCT ad.hour) as actual_count,
          ROUND(COUNT(DISTINCT ad.hour) / ARRAY_LENGTH(dr.expected_timestamps) * 100, 2) as completeness_pct
        FROM date_range dr
        LEFT JOIN UNNEST(dr.expected_timestamps) as expected_hour
        LEFT JOIN actual_data ad 
          ON dr.symbol = ad.symbol 
          AND dr.venue = ad.venue 
          AND expected_hour = ad.hour
        GROUP BY dr.symbol, dr.venue, dr.expected_timestamps
        ORDER BY completeness_pct ASC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
                bigquery.ArrayQueryParameter("priority_symbols", "STRING", PRIORITY_SYMBOLS)
            ]
        )
        
        results = self.bq.query(query, job_config=job_config).to_dataframe()
        
        incomplete_data = results[results['completeness_pct'] < 95.0]
        if len(incomplete_data) > 0:
            for _, row in incomplete_data.iterrows():
                self.alerts.append(
                    f"⚠️  Incomplete data: {row['symbol']} on {row['venue']} "
                    f"has {row['completeness_pct']:.1f}% completeness "
                    f"({row['actual_count']}/{row['expected_count']} candles)"
                )
        
        return {
            'status': 'OK' if len(incomplete_data) == 0 else 'WARNING',
            'incomplete_count': len(incomplete_data),
            'results': results.to_dict('records')
        }
    
    def check_for_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate data points."""
        query = f"""
        SELECT 
          symbol,
          venue,
          timestamp,
          COUNT(*) as duplicate_count
        FROM `{self.project_id}.{self.dataset_id}.ohlcv`
        WHERE symbol IN UNNEST(@priority_symbols)
        GROUP BY symbol, venue, timestamp
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        LIMIT 100
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("priority_symbols", "STRING", PRIORITY_SYMBOLS)
            ]
        )
        
        results = self.bq.query(query, job_config=job_config).to_dataframe()
        
        if len(results) > 0:
            total_dupes = results['duplicate_count'].sum()
            self.alerts.append(
                f"❌ Found {len(results)} duplicate timestamp groups "
                f"({total_dupes} total duplicate rows)"
            )
        
        return {
            'status': 'OK' if len(results) == 0 else 'ERROR',
            'duplicate_groups': len(results),
            'results': results.to_dict('records')
        }
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check for invalid OHLCV relationships."""
        query = f"""
        SELECT 
          symbol,
          venue,
          timestamp,
          open,
          high,
          low,
          close,
          CASE
            WHEN high < low THEN 'high < low'
            WHEN high < open THEN 'high < open'
            WHEN high < close THEN 'high < close'
            WHEN low > open THEN 'low > open'
            WHEN low > close THEN 'low > close'
            WHEN open <= 0 THEN 'open <= 0'
            WHEN high <= 0 THEN 'high <= 0'
            WHEN low <= 0 THEN 'low <= 0'
            WHEN close <= 0 THEN 'close <= 0'
            ELSE 'unknown'
          END as quality_issue
        FROM `{self.project_id}.{self.dataset_id}.ohlcv`
        WHERE symbol IN UNNEST(@priority_symbols)
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          AND (
            high < low 
            OR high < open 
            OR high < close
            OR low > open
            OR low > close
            OR open <= 0
            OR high <= 0
            OR low <= 0
            OR close <= 0
          )
        LIMIT 100
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("priority_symbols", "STRING", PRIORITY_SYMBOLS)
            ]
        )
        
        results = self.bq.query(query, job_config=job_config).to_dataframe()
        
        if len(results) > 0:
            self.alerts.append(
                f"❌ Found {len(results)} rows with data quality issues"
            )
        
        return {
            'status': 'OK' if len(results) == 0 else 'ERROR',
            'bad_rows': len(results),
            'results': results.to_dict('records')
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all data quality checks."""
        print("🔍 Running data quality checks...\n")
        
        results = {}
        
        # Check freshness
        print("1. Checking data freshness...")
        results['freshness'] = self.check_data_freshness()
        print(f"   Status: {results['freshness']['status']}\n")
        
        # Check completeness
        print("2. Checking data completeness...")
        results['completeness'] = self.check_data_completeness()
        print(f"   Status: {results['completeness']['status']}\n")
        
        # Check for duplicates
        print("3. Checking for duplicates...")
        results['duplicates'] = self.check_for_duplicates()
        print(f"   Status: {results['duplicates']['status']}\n")
        
        # Check data quality
        print("4. Checking data quality...")
        results['quality'] = self.check_data_quality()
        print(f"   Status: {results['quality']['status']}\n")
        
        # Print alerts
        if self.alerts:
            print("⚠️  ALERTS:")
            for alert in self.alerts:
                print(f"   {alert}")
        else:
            print("✅ All checks passed!")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Monitor data quality")
    parser.add_argument("project_id", help="GCP Project ID")
    parser.add_argument("--dataset", default="market_data", help="BigQuery dataset")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()
    
    monitor = DataQualityMonitor(args.project_id, args.dataset)
    results = monitor.run_all_checks()
    
    if args.json:
        import json
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

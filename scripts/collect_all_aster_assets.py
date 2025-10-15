#!/usr/bin/env python3
"""
Collect Data for ALL 200+ Aster DEX Assets with Rate Limiting
Optimized batch processing with exponential backoff and API compliance.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitedAsterCollector:
    """Rate-limited collector for all Aster DEX assets."""

    def __init__(self):
        self.base_url = "https://fapi.asterdex.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AsterAI-Trading-Bot/1.0',
            'Accept': 'application/json'
        })

        # Data directories
        self.data_dir = Path("data/historical/real_aster_only")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting settings
        self.requests_per_minute = 30  # Conservative rate limiting
        self.request_interval = 60 / self.requests_per_minute
        self.last_request_time = 0
        self.rate_limit_hits = 0
        self.backoff_multiplier = 2
        self.max_backoff_time = 300  # 5 minutes max backoff

        # Collection settings
        self.batch_size = 10  # Process 10 assets at a time
        self.batch_pause = 60  # 1 minute pause between batches
        self.asset_timeout = 3600  # 1 hour timeout per asset

        # Progress tracking
        self.collection_stats = {
            'start_time': None,
            'end_time': None,
            'total_assets': 0,
            'processed_assets': 0,
            'successful_assets': 0,
            'failed_assets': 0,
            'total_requests': 0,
            'rate_limit_hits': 0,
            'total_data_points': 0,
            'average_quality': 0.0
        }

    def _rate_limit_wait(self, backoff_time: float = 0):
        """Enforce rate limiting with exponential backoff."""
        elapsed = time.time() - self.last_request_time
        wait_time = max(self.request_interval, backoff_time)

        if elapsed < wait_time:
            actual_wait = wait_time - elapsed
            logger.debug(f"Rate limiting: waiting {actual_wait:.1f} seconds")
            time.sleep(actual_wait)

        self.last_request_time = time.time()

    def _handle_rate_limit(self, response) -> float:
        """Handle rate limit responses with exponential backoff."""
        if response.status_code == 429:
            self.rate_limit_hits += 1
            self.collection_stats['rate_limit_hits'] += 1

            # Exponential backoff: 30s, 60s, 120s, 240s, 300s max
            backoff_time = min(30 * (self.backoff_multiplier ** self.rate_limit_hits), self.max_backoff_time)

            logger.warning(f"⚠️  Rate limit hit! Backing off for {backoff_time} seconds")
            return backoff_time

        # Reset backoff on successful request
        if response.status_code == 200:
            self.rate_limit_hits = 0

        return 0

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols from exchange info."""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        self._rate_limit_wait()

        try:
            response = self.session.get(url, timeout=10)
            backoff_time = self._handle_rate_limit(response)

            if backoff_time > 0:
                time.sleep(backoff_time)
                # Retry once after backoff
                response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                symbols = [s.get('symbol') for s in data.get('symbols', []) if isinstance(s, dict)]
                logger.info(f"✅ Retrieved {len(symbols)} symbols from Aster DEX")
                return sorted(symbols)
            else:
                logger.error(f"❌ Failed to get symbols: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"❌ Symbol retrieval error: {e}")
            return []

    def get_historical_klines(self, symbol: str, interval: str = "1h",
                            days: int = 30, limit: int = 500) -> pd.DataFrame:
        """Get historical klines with optimized request batching."""
        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_klines = []
        current_ts = start_ts
        asset_start_time = time.time()

        logger.info(f"📊 Collecting {symbol} {interval} data ({days} days)")

        while current_ts < end_ts and (time.time() - asset_start_time) < self.asset_timeout:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_ts,
                'limit': min(limit, 1000)
            }

            self._rate_limit_wait()
            self.collection_stats['total_requests'] += 1

            try:
                response = self.session.get(url, params=params, timeout=15)
                backoff_time = self._handle_rate_limit(response)

                if backoff_time > 0:
                    time.sleep(backoff_time)
                    continue

                if response.status_code == 200:
                    klines = response.json()
                    if not klines:
                        break

                    all_klines.extend(klines)
                    last_time = klines[-1][0]
                    current_ts = last_time + 1

                    # Break if we got less than requested (end of data)
                    if len(klines) < limit:
                        break

                else:
                    logger.warning(f"   ⚠️  API error {response.status_code} for {symbol}")
                    break

            except Exception as e:
                logger.error(f"   ❌ Error collecting {symbol}: {e}")
                break

        # Convert to DataFrame
        if all_klines:
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.drop(['close_time', 'trades', 'taker_buy_quote_volume', 'ignore'], axis=1)
            df = df.set_index('timestamp').sort_index()

            logger.info(f"✅ Collected {len(df)} rows for {symbol} {interval}")
            return df
        else:
            logger.warning(f"❌ No {interval} data collected for {symbol}")
            return pd.DataFrame()

    def get_asset_snapshot(self, symbol: str) -> Tuple[Optional[Dict], Optional[List]]:
        """Get orderbook and recent trades snapshot."""
        orderbook = None
        trades = None

        # Get orderbook
        try:
            url = f"{self.base_url}/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': 100}

            self._rate_limit_wait()
            self.collection_stats['total_requests'] += 1

            response = self.session.get(url, params=params, timeout=10)
            backoff_time = self._handle_rate_limit(response)

            if backoff_time > 0:
                time.sleep(backoff_time)
                response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                orderbook = response.json()

        except Exception as e:
            logger.warning(f"   Orderbook failed for {symbol}: {e}")

        # Get recent trades
        try:
            url = f"{self.base_url}/fapi/v1/trades"
            params = {'symbol': symbol, 'limit': 50}

            self._rate_limit_wait()
            self.collection_stats['total_requests'] += 1

            response = self.session.get(url, params=params, timeout=10)
            backoff_time = self._handle_rate_limit(response)

            if backoff_time > 0:
                time.sleep(backoff_time)
                response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                trades = response.json()

        except Exception as e:
            logger.warning(f"   Trades failed for {symbol}: {e}")

        return orderbook, trades

    def collect_single_asset(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """Collect all data for a single asset with rate limiting."""
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        logger.info(f"🔄 Processing {symbol}...")

        asset_data = {
            'symbol': symbol,
            'success': False,
            'timeframes': {},
            'orderbook': None,
            'trades': None,
            'quality_score': 0.0,
            'data_points': 0,
            'requests_used': 0,
            'processing_time': 0,
            'error': None
        }

        start_time = time.time()

        try:
            # Collect historical data
            total_points = 0
            successful_timeframes = 0
            requests_used = 0

            for tf in timeframes:
                df = self.get_historical_klines(symbol, tf, days=30, limit=500)
                requests_used += max(1, len(df) // 500 + 1) if not df.empty else 1

                if not df.empty:
                    asset_data['timeframes'][tf] = df
                    total_points += len(df)
                    successful_timeframes += 1
                else:
                    logger.warning(f"   ❌ {tf}: No data")

            # Collect snapshots (orderbook + trades = 2 requests)
            orderbook, trades = self.get_asset_snapshot(symbol)
            requests_used += 2

            if orderbook:
                asset_data['orderbook'] = orderbook
            if trades:
                asset_data['trades'] = trades

            # Calculate quality score
            if successful_timeframes > 0:
                completeness = successful_timeframes / len(timeframes)
                data_density = min(1.0, total_points / (successful_timeframes * 720))  # Expected points per timeframe
                has_orderbook = 1.0 if orderbook else 0.0
                has_trades = 1.0 if trades else 0.0

                asset_data['quality_score'] = (
                    completeness * 0.4 +
                    data_density * 0.3 +
                    has_orderbook * 0.2 +
                    has_trades * 0.1
                )
                asset_data['data_points'] = total_points
                asset_data['success'] = True

            asset_data['requests_used'] = requests_used
            asset_data['processing_time'] = time.time() - start_time

            return asset_data

        except Exception as e:
            logger.error(f"❌ Failed to collect {symbol}: {e}")
            asset_data['error'] = str(e)
            asset_data['processing_time'] = time.time() - start_time
            return asset_data

    def save_asset_data(self, asset_data: Dict[str, Any]):
        """Save collected asset data."""
        symbol = asset_data['symbol']

        # Save each timeframe
        for tf, df in asset_data.get('timeframes', {}).items():
            filename = f"{symbol}_{tf}.parquet"
            filepath = self.data_dir / filename
            df.to_parquet(filepath)

        # Save orderbook
        if asset_data.get('orderbook'):
            filename = f"{symbol}_orderbook.json"
            filepath = self.data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(asset_data['orderbook'], f, indent=2)

        # Save trades
        if asset_data.get('trades'):
            filename = f"{symbol}_trades.json"
            filepath = self.data_dir / filename
            with open(filepath, 'w') as f:
                json.dump(asset_data['trades'], f, indent=2)

    def process_batch(self, symbols: List[str], batch_num: int, total_batches: int) -> Dict[str, Any]:
        """Process a batch of symbols."""
        logger.info(f"\n📦 Batch {batch_num}/{total_batches}: Processing {len(symbols)} assets")

        batch_results = {}
        batch_successful = 0
        batch_requests = 0

        for symbol in symbols:
            asset_data = self.collect_single_asset(symbol)
            batch_results[symbol] = asset_data
            batch_requests += asset_data.get('requests_used', 1)

            if asset_data['success']:
                batch_successful += 1
                self.save_asset_data(asset_data)
                self.collection_stats['total_data_points'] += asset_data['data_points']
            else:
                self.collection_stats['failed_assets'] += 1

        batch_success_rate = batch_successful / len(symbols) if symbols else 0
        logger.info(f"✅ Batch {batch_num} complete: {batch_successful}/{len(symbols)} successful ({batch_success_rate*100:.1f}%)")

        # Pause between batches (unless it's the last batch)
        if batch_num < total_batches:
            logger.info(f"⏳ Pausing {self.batch_pause} seconds before next batch...")
            time.sleep(self.batch_pause)

        return batch_results

    def run_full_collection(self, max_assets: int = None, priority_symbols: List[str] = None):
        """Run full collection for all assets with rate limiting."""
        print("""
================================================================================
        Complete Aster DEX Asset Collection (200+ Assets)
      Rate-Limited Batch Processing with Exponential Backoff
================================================================================
        """)

        self.collection_stats['start_time'] = datetime.now()

        # Get all available symbols
        logger.info("🔍 Discovering all Aster DEX symbols...")
        all_symbols = self.get_all_symbols()

        if not all_symbols:
            logger.error("❌ Cannot retrieve symbol list from Aster DEX")
            return {}

        self.collection_stats['total_assets'] = len(all_symbols)
        logger.info(f"📊 Found {len(all_symbols)} total symbols on Aster DEX")

        # Prioritize symbols if specified
        if priority_symbols:
            priority_set = set(priority_symbols)
            priority_list = [s for s in all_symbols if s in priority_set]
            remaining = [s for s in all_symbols if s not in priority_set]
            selected_symbols = priority_list + remaining[:max_assets - len(priority_list)] if max_assets else priority_list + remaining
        else:
            selected_symbols = all_symbols[:max_assets] if max_assets else all_symbols

        logger.info(f"🎯 Selected {len(selected_symbols)} symbols for collection")

        # Create batches
        batches = []
        for i in range(0, len(selected_symbols), self.batch_size):
            batches.append(selected_symbols[i:i + self.batch_size])

        logger.info(f"📦 Created {len(batches)} batches of {self.batch_size} assets each")
        logger.info(f"⏱️  Estimated time: ~{len(batches) * (self.batch_pause/60):.1f} minutes (plus processing time)")

        # Process batches
        all_results = {}

        for batch_num, batch_symbols in enumerate(batches, 1):
            try:
                batch_results = self.process_batch(batch_symbols, batch_num, len(batches))
                all_results.update(batch_results)
                self.collection_stats['processed_assets'] += len(batch_symbols)
                self.collection_stats['successful_assets'] += sum(1 for r in batch_results.values() if r.get('success'))

            except KeyboardInterrupt:
                logger.warning(f"⚠️  Collection interrupted at batch {batch_num}")
                break
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                continue

        # Finalize collection
        self.collection_stats['end_time'] = datetime.now()

        if self.collection_stats['successful_assets'] > 0:
            self.collection_stats['average_quality'] = sum(
                r.get('quality_score', 0) for r in all_results.values() if r.get('success')
            ) / self.collection_stats['successful_assets']

        # Save comprehensive summary
        summary = {
            'collection_stats': self.collection_stats,
            'rate_limiting': {
                'requests_per_minute': self.requests_per_minute,
                'total_requests': self.collection_stats['total_requests'],
                'rate_limit_hits': self.collection_stats['rate_limit_hits'],
                'backoff_events': self.rate_limit_hits
            },
            'batch_processing': {
                'batch_size': self.batch_size,
                'batch_pause_seconds': self.batch_pause,
                'total_batches': len(batches),
                'completed_batches': len([b for b in range(1, len(batches)+1) if b <= len(all_results) // self.batch_size + 1])
            },
            'assets': all_results,
            'collection_timestamp': datetime.now().isoformat()
        }

        summary_file = self.data_dir / "full_collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print final results
        self.print_final_summary(summary)
        return summary

    def print_final_summary(self, summary: Dict):
        """Print comprehensive collection summary."""
        stats = summary['collection_stats']
        rate_stats = summary['rate_limiting']

        print(f"\n{'='*80}")
        print("COMPLETE ASTER DEX COLLECTION SUMMARY")
        print(f"{'='*80}\n")

        duration = None
        if stats.get('start_time') and stats.get('end_time'):
            duration = (stats['end_time'] - stats['start_time']).total_seconds()
            print(f"⏱️  Duration: {duration/60:.1f} minutes")
        print(f"📊 Total Assets Available: {stats['total_assets']}")
        print(f"🎯 Assets Processed: {stats['processed_assets']}")
        print(f"✅ Successful Collections: {stats['successful_assets']}")
        print(f"❌ Failed Collections: {stats['failed_assets']}")
        print(f"📈 Success Rate: {stats['successful_assets']/max(stats['processed_assets'],1)*100:.1f}%")
        print(f"⭐ Average Quality Score: {stats['average_quality']:.2f}")
        print(f"📈 Total Data Points: {stats['total_data_points']:,}")

        print(f"\n🔗 API Usage:")
        print(f"   Requests Made: {rate_stats['total_requests']}")
        print(f"   Rate Limits Hit: {rate_stats['rate_limit_hits']}")
        print(f"   Requests/Minute: {rate_stats['requests_per_minute']}")

        print(f"\n📦 Batch Processing:")
        batch_stats = summary['batch_processing']
        print(f"   Batch Size: {batch_stats['batch_size']}")
        print(f"   Total Batches: {batch_stats['total_batches']}")
        print(f"   Batch Pause: {batch_stats['batch_pause_seconds']}s")

        if stats['successful_assets'] > 0:
            print(f"\n🏆 Top 5 Assets by Quality:")
            successful_assets = [(k, v) for k, v in summary['assets'].items() if v.get('success')]
            top_assets = sorted(successful_assets, key=lambda x: x[1]['quality_score'], reverse=True)[:5]

            for symbol, data in top_assets:
                print(f"   🥇 {symbol}: {data['quality_score']:.2f} ({data['data_points']} points)")

        print(f"""
================================================================================
                🎉 Collection Complete!
      {stats['successful_assets']} assets collected with rate limiting compliance
================================================================================

Data saved to: {self.data_dir}
Summary saved to: {self.data_dir}/full_collection_summary.json

Next steps:
1. Review full_collection_summary.json for detailed results
2. Proceed to feature engineering: python scripts/validate_confluence_features.py
3. Train models: python local_training/train_confluence_model.py
        """)


def main():
    """Main execution."""
    import argparse
    parser = argparse.ArgumentParser(description="Collect data for all Aster DEX assets with rate limiting")
    parser.add_argument('--max-assets', type=int, default=50, help='Maximum assets to collect (default: 50)')
    parser.add_argument('--batch-size', type=int, default=5, help='Assets per batch (default: 5)')
    parser.add_argument('--requests-per-minute', type=int, default=20, help='API requests per minute (default: 20)')
    parser.add_argument('--priority-symbols', nargs='*', help='Priority symbols to collect first')

    args = parser.parse_args()

    collector = RateLimitedAsterCollector()
    collector.batch_size = args.batch_size
    collector.requests_per_minute = args.requests_per_minute
    collector.request_interval = 60 / args.requests_per_minute

    logger.info(f"🚀 Starting collection: max {args.max_assets} assets, {args.requests_per_minute} req/min, batch size {args.batch_size}")

    if args.priority_symbols:
        logger.info(f"🎯 Priority symbols: {', '.join(args.priority_symbols)}")

    summary = collector.run_full_collection(
        max_assets=args.max_assets,
        priority_symbols=args.priority_symbols
    )

    if summary and summary['collection_stats']['successful_assets'] > 0:
        logger.info("✅ Full collection completed successfully")
    else:
        logger.error("❌ Full collection failed or was interrupted")


if __name__ == "__main__":
    main()




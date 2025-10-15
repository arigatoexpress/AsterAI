#!/usr/bin/env python3
"""
Collect Real Historical Data from Aster DEX Only
No synthetic data, rate-limited, validated collection.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealAsterDataCollector:
    """
    Collect historical data from Aster DEX only.
    No synthetic data, rate-limited, comprehensive validation.
    """

    def __init__(self, output_dir: str = "data/historical/real_aster_only"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collector = None

        # Rate limiting (conservative)
        self.request_delay = 0.2  # 200ms between requests (5 requests/second)
        self.max_requests_per_minute = 200  # 200 requests/minute
        self.batch_size = 10  # Process 10 symbols at a time
        self.batch_delay = 60  # 1 minute between batches

        # Data requirements
        self.min_data_points = 1000  # Minimum 1000 data points (about 40+ days at 1h)
        self.max_missing_pct = 0.05  # Maximum 5% missing data
        self.required_timeframes = ["1h", "4h", "1d"]

        # Time range (6 months)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)

        # Track rate limiting
        self.requests_this_minute = 0
        self.minute_start_time = time.time()

    async def initialize(self):
        """Initialize the collector."""
        self.collector = AsterDEXDataCollector()
        await self.collector.initialize()
        logger.info("✅ Real Aster data collector initialized")

    async def load_discovered_assets(self, discovery_file: str = "data/aster_assets_discovery.json") -> Dict[str, Dict]:
        """Load assets from discovery phase."""
        discovery_path = Path(discovery_file)

        if not discovery_path.exists():
            logger.error(f"❌ Discovery file not found: {discovery_path}")
            logger.info("💡 Run scripts/discover_aster_assets.py first")
            return {}

        with open(discovery_path) as f:
            discovery_data = json.load(f)

        assets = discovery_data.get('assets', {})
        trainable_assets = {k: v for k, v in assets.items() if v.get('is_trainable', False)}

        logger.info(f"📋 Loaded {len(trainable_assets)} trainable assets from discovery")
        return trainable_assets

    async def collect_all_real_data(self) -> Dict[str, Dict]:
        """Collect data for all trainable assets from Aster DEX only."""
        logger.info(f"\n{'='*70}")
        logger.info("COLLECTING REAL ASTER DEX DATA ONLY (NO SYNTHETIC)")
        logger.info(f"{'='*70}\n")

        # Load discovered assets
        assets = await self.load_discovered_assets()
        if not assets:
            return {}

        # Process in batches to respect rate limits
        asset_symbols = list(assets.keys())
        results = {}

        for i in range(0, len(asset_symbols), self.batch_size):
            batch_symbols = asset_symbols[i:i+self.batch_size]

            logger.info(f"\n📦 Processing batch {i//self.batch_size + 1} of {(len(asset_symbols) + self.batch_size - 1)//self.batch_size}")
            logger.info(f"   Symbols: {batch_symbols}")

            # Process batch
            batch_results = await self.process_batch(batch_symbols)
            results.update(batch_results)

            # Rate limiting between batches
            if i + self.batch_size < len(asset_symbols):
                logger.info(f"⏳ Rate limiting: waiting {self.batch_delay} seconds...")
                await asyncio.sleep(self.batch_delay)

        # Generate final report
        await self.generate_collection_report(results)

        return results

    async def process_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Process a batch of symbols."""
        batch_results = {}

        for symbol in symbols:
            try:
                logger.info(f"📥 Collecting {symbol}...")

                # Collect data with rate limiting
                symbol_result = await self.collect_symbol_data(symbol)
                batch_results[symbol] = symbol_result

                # Rate limiting between symbols
                await self.enforce_rate_limit()

            except Exception as e:
                logger.error(f"❌ Failed to collect {symbol}: {e}")
                batch_results[symbol] = {
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                }

        return batch_results

    async def collect_symbol_data(self, symbol: str) -> Dict:
        """Collect all required data for a symbol."""
        result = {
            'symbol': symbol,
            'success': False,
            'timeframes_collected': [],
            'total_data_points': 0,
            'data_quality_score': 0,
            'collection_errors': []
        }

        try:
            # Test basic connectivity for this symbol
            logger.info(f"   🔗 Testing connectivity for {symbol}...")

            # Test orderbook
            orderbook = await self.collector.collect_orderbook_data(symbol)
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                result['collection_errors'].append("No orderbook data available")
                return result

            # Test recent trades
            trades = await self.collector.collect_recent_trades(symbol, limit=5)
            if not trades or len(trades) == 0:
                result['collection_errors'].append("No recent trades available")
                return result

            logger.info(f"   ✅ Connectivity confirmed (Orderbook: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks)")

            # Collect historical data for each timeframe
            all_dataframes = {}
            total_points = 0

            for timeframe in self.required_timeframes:
                try:
                    logger.info(f"   📊 Collecting {timeframe} data...")

                    df = await self.collector.collect_historical_data(
                        symbol=symbol,
                        start_date=self.start_date.strftime("%Y-%m-%d"),
                        end_date=self.end_date.strftime("%Y-%m-%d"),
                        interval=timeframe,
                        limit=1000
                    )

                    if df.empty:
                        result['collection_errors'].append(f"No {timeframe} data available")
                        continue

                    # Validate data quality
                    quality_score = self.validate_dataframe(df)
                    if quality_score < 0.7:
                        result['collection_errors'].append(f"Poor {timeframe} data quality: {quality_score:.2f}")
                        continue

                    # Save dataframe
                    all_dataframes[timeframe] = df
                    total_points += len(df)

                    logger.info(f"   ✅ {timeframe}: {len(df)} points, quality: {quality_score:.2f}")

                except Exception as e:
                    error_msg = f"{timeframe} collection failed: {str(e)}"
                    result['collection_errors'].append(error_msg)
                    logger.warning(f"   ⚠️  {error_msg}")

            # Check if we have sufficient data
            if len(all_dataframes) == 0:
                result['collection_errors'].append("No timeframes successfully collected")
                return result

            if total_points < self.min_data_points:
                result['collection_errors'].append(f"Insufficient data points: {total_points} < {self.min_data_points}")
                return result

            # Save all dataframes
            for timeframe, df in all_dataframes.items():
                output_file = self.output_dir / f"{symbol}_{timeframe}.parquet"
                df.to_parquet(output_file, compression='snappy')

            # Calculate overall quality
            avg_quality = sum(self.validate_dataframe(df) for df in all_dataframes.values()) / len(all_dataframes)

            result.update({
                'success': True,
                'timeframes_collected': list(all_dataframes.keys()),
                'total_data_points': total_points,
                'data_quality_score': avg_quality,
                'date_range': {
                    'start': self.start_date.strftime("%Y-%m-%d"),
                    'end': self.end_date.strftime("%Y-%m-%d")
                }
            })

            logger.info(f"   🎉 {symbol} collected successfully: {total_points} total points")

        except Exception as e:
            result['collection_errors'].append(f"Unexpected error: {str(e)}")
            logger.error(f"   ❌ Unexpected error for {symbol}: {e}")

        return result

    def validate_dataframe(self, df: pd.DataFrame) -> float:
        """Validate dataframe quality."""
        if df.empty:
            return 0.0

        quality_score = 1.0

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            quality_score *= 0.5
            logger.warning(f"Missing columns: {missing_cols}")

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > self.max_missing_pct:
            quality_score *= (1 - missing_pct * 2)

        # Check data reasonableness
        if (df['close'] <= 0).any():
            quality_score *= 0.8

        # Check for extreme outliers
        returns = df['close'].pct_change().abs()
        extreme_changes = (returns > 0.5).sum()
        if extreme_changes > len(df) * 0.05:  # More than 5% extreme changes
            quality_score *= 0.9

        # Check volume is reasonable
        if (df['volume'] <= 0).any():
            quality_score *= 0.9

        return max(0.0, quality_score)

    async def enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()

        # Reset counter if minute has passed
        if current_time - self.minute_start_time >= 60:
            self.requests_this_minute = 0
            self.minute_start_time = current_time

        # Check if we need to wait
        if self.requests_this_minute >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                self.requests_this_minute = 0
                self.minute_start_time = time.time()

        # Add delay between requests
        await asyncio.sleep(self.request_delay)
        self.requests_this_minute += 1

    async def generate_collection_report(self, results: Dict[str, Dict]):
        """Generate comprehensive collection report."""
        logger.info(f"\n{'='*70}")
        logger.info("COLLECTION REPORT")
        logger.info(f"{'='*70}\n")

        # Summary statistics
        successful = sum(1 for r in results.values() if r.get('success'))
        total_symbols = len(results)
        total_data_points = sum(r.get('total_data_points', 0) for r in results.values())

        print(f"📊 Collection Summary:")
        print(f"   Total symbols processed: {total_symbols}")
        print(f"   Successfully collected: {successful}")
        print(f"   Success rate: {successful/total_symbols*100:.1f}%")
        print(f"   Total data points: {total_data_points:,}")

        # Quality analysis
        successful_results = [r for r in results.values() if r.get('success')]
        if successful_results:
            avg_quality = sum(r['data_quality_score'] for r in successful_results) / len(successful_results)
            print(f"   Average data quality: {avg_quality:.2f}")

        # Error analysis
        all_errors = []
        for result in results.values():
            all_errors.extend(result.get('collection_errors', []))

        if all_errors:
            print(f"\n⚠️  Top errors:")
            error_counts = {}
            for error in all_errors:
                error_counts[error] = error_counts.get(error, 0) + 1

            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {error}: {count} occurrences")

        # Save detailed report
        report_file = self.output_dir / "collection_report.json"

        report = {
            'collection_timestamp': datetime.now().isoformat(),
            'parameters': {
                'start_date': self.start_date.strftime("%Y-%m-%d"),
                'end_date': self.end_date.strftime("%Y-%m-%d"),
                'required_timeframes': self.required_timeframes,
                'min_data_points': self.min_data_points,
                'rate_limit_requests_per_minute': self.max_requests_per_minute
            },
            'summary': {
                'total_symbols': total_symbols,
                'successful_collections': successful,
                'success_rate': successful/total_symbols if total_symbols > 0 else 0,
                'total_data_points': total_data_points,
                'average_quality_score': avg_quality if successful_results else 0
            },
            'results': results
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n💾 Detailed report saved to {report_file}")

        # Save summary CSV for easy viewing
        summary_data = []
        for symbol, result in results.items():
            summary_data.append({
                'symbol': symbol,
                'success': result.get('success', False),
                'data_points': result.get('total_data_points', 0),
                'quality_score': result.get('data_quality_score', 0),
                'timeframes': ','.join(result.get('timeframes_collected', [])),
                'errors': '; '.join(result.get('collection_errors', []))
            })

        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "collection_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"💾 Summary CSV saved to {summary_csv}")

    async def close(self):
        """Close connections."""
        if self.collector:
            await self.collector.close()


async def main():
    """Main execution."""
    print("""
================================================================================
      Real Aster DEX Data Collection (No Synthetic Data)
         Rate-Limited, Validated, Production-Ready
================================================================================
    """)

    collector = RealAsterDataCollector()

    try:
        await collector.initialize()

        # Check if discovery has been run
        discovery_file = Path("data/aster_assets_discovery.json")
        if not discovery_file.exists():
            logger.error("❌ Asset discovery not found!")
            logger.info("💡 Run scripts/discover_aster_assets.py first")
            return

        # Collect data
        results = await collector.collect_all_real_data()

        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)

        print(f"""
================================================================================
                   🎉 Collection Complete!
         {successful}/{total} assets collected successfully
================================================================================

Data saved to: data/historical/real_aster_only/

Key achievements:
✅ No synthetic data used - only real Aster DEX data
✅ Rate-limited requests ({collector.max_requests_per_minute}/min)
✅ Comprehensive data validation
✅ Multi-timeframe collection (1h, 4h, 1d)
✅ Quality scoring for each dataset

Next steps:
1. Review collection_report.json for detailed results
2. Check collection_summary.csv for quick overview
3. Proceed to feature engineering with validated data
4. Train models on high-quality real data only
        """)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Collection interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""
Real-Time Market Data Collection for Adaptive AI Trading System

This script collects recent market data from Aster DEX for adaptive learning.
Focuses on current market conditions rather than historical backtesting.

Goal: Enable continuous learning from real-time market data to optimize
trading strategies for current market conditions and maximize profits.
"""

import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_trader.backtesting.historical_data_collector import HistoricalDataCollector
from mcp_trader.config import get_settings
from mcp_trader.security.secrets import get_secret_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCollectionManager:
    """Manages the collection of historical trading data."""

    def __init__(self):
        self.settings = get_settings()
        self.collector = HistoricalDataCollector()
        self.stats = {
            'pairs_collected': 0,
            'total_klines': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

    async def initialize_secrets(self):
        """Load API credentials."""
        try:
            sm = get_secret_manager()
            sm.load_secrets_from_file()
            logger.info("API credentials loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load API credentials: {e}")
            raise

    async def collect_top_pairs_data(self, num_pairs: int = 20, intervals: list = None):
        """
        Collect historical data for the top traded pairs.

        Args:
            num_pairs: Number of top pairs to collect
            intervals: Time intervals to collect (1h, 4h, 1d)
        """
        if intervals is None:
            intervals = ['1h', '4h', '1d']

        logger.info(f"Starting collection of top {num_pairs} pairs with intervals: {intervals}")
        self.stats['start_time'] = datetime.now()

        try:
            # Get top traded pairs
            top_pairs = await self.collector.get_top_traded_pairs(num_pairs)
            if not top_pairs:
                logger.error("No trading pairs found")
                return

            logger.info(f"Top {len(top_pairs)} pairs to collect: {top_pairs}")

                # Collect recent data for adaptive learning (last 30 days)
            collected_data = await self.collector.collect_recent_market_data(
                symbols=top_pairs,
                intervals=intervals,
                days_back=30  # Focus on recent market conditions
            )

            # Update statistics
            self.stats['pairs_collected'] = len(collected_data)
            total_klines = 0

            for symbol, interval_data in collected_data.items():
                for interval, df in interval_data.items():
                    if df is not None and not df.empty:
                        total_klines += len(df)
                        logger.info(f"Collected {len(df)} {interval} klines for {symbol}")

            self.stats['total_klines'] = total_klines

            logger.info(f"Data collection completed successfully!")
            logger.info(f"Pairs collected: {self.stats['pairs_collected']}")
            logger.info(f"Total klines: {self.stats['total_klines']}")

        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            self.stats['errors'] += 1
            raise

        finally:
            self.stats['end_time'] = datetime.now()

    async def update_existing_data(self):
        """Update existing historical data with latest market data."""
        logger.info("Starting data update process...")

        try:
            updated_data = await self.collector.update_historical_data()

            update_count = 0
            for symbol, interval_data in updated_data.items():
                for interval, df in interval_data.items():
                    if df is not None:
                        update_count += 1

            logger.info(f"Data update completed. Updated {update_count} datasets.")

        except Exception as e:
            logger.error(f"Error updating data: {e}")
            raise

    def print_collection_report(self):
        """Print a comprehensive report of the data collection."""
        print("\n" + "="*80)
        print("🎯 ASTER DEX HISTORICAL DATA COLLECTION REPORT")
        print("="*80)

        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"⏱️  Duration: {duration}")

        print(f"📊 Pairs Collected: {self.stats['pairs_collected']}")
        print(f"📈 Total Klines: {self.stats['total_klines']:,}")
        print(f"❌ Errors: {self.stats['errors']}")

        if self.stats['pairs_collected'] > 0:
            avg_klines_per_pair = self.stats['total_klines'] / self.stats['pairs_collected']
            print(f"📊 Average Klines per Pair: {avg_klines_per_pair:,.0f}")

        # Check data directory
        data_dir = Path("data/historical")
        if data_dir.exists():
            total_files = len(list(data_dir.rglob("*.csv")))
            total_size = sum(f.stat().st_size for f in data_dir.rglob("*.csv"))
            print(f"💾 Files Created: {total_files}")
            print(f"💽 Total Size: {total_size / (1024*1024):.1f} MB")

        print("\n🎯 PROGRESS TOWARD $1M GOAL:")
        print("✅ Foundation: Historical data collection"        print("✅ Analysis: Multi-timeframe data available"        print("✅ AI Ready: Data prepared for machine learning"        print("🔄 Next: Backtesting framework implementation"        print("🔄 Next: AI strategy development"        print("🔄 Next: Autonomous trading deployment"        print("🎯 Target: $1M by end of 2026"        print("="*80)

    async def verify_data_quality(self):
        """Verify the quality of collected data."""
        logger.info("Verifying data quality...")

        data_dir = Path("data/historical")
        if not data_dir.exists():
            logger.warning("Data directory does not exist")
            return

        quality_report = {
            'total_files': 0,
            'empty_files': 0,
            'corrupted_files': 0,
            'oldest_data': None,
            'newest_data': None,
            'total_rows': 0
        }

        for csv_file in data_dir.rglob("*.csv"):
            quality_report['total_files'] += 1

            try:
                df = pd.read_csv(csv_file)
                quality_report['total_rows'] += len(df)

                if df.empty:
                    quality_report['empty_files'] += 1
                    continue

                # Check date range
                df.index = pd.to_datetime(df.iloc[:, 0])
                oldest = df.index.min()
                newest = df.index.max()

                if quality_report['oldest_data'] is None or oldest < quality_report['oldest_data']:
                    quality_report['oldest_data'] = oldest
                if quality_report['newest_data'] is None or newest > quality_report['newest_data']:
                    quality_report['newest_data'] = newest

            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
                quality_report['corrupted_files'] += 1

        print("\n📊 DATA QUALITY REPORT:")
        print(f"Total Files: {quality_report['total_files']}")
        print(f"Empty Files: {quality_report['empty_files']}")
        print(f"Corrupted Files: {quality_report['corrupted_files']}")
        print(f"Total Rows: {quality_report['total_rows']:,}")
        if quality_report['oldest_data']:
            print(f"Date Range: {quality_report['oldest_data']} to {quality_report['newest_data']}")

        return quality_report


async def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description="Collect historical data from Aster DEX")
    parser.add_argument('--pairs', type=int, default=20, help='Number of top pairs to collect')
    parser.add_argument('--intervals', nargs='+', default=['1h', '4h', '1d'],
                       help='Time intervals to collect')
    parser.add_argument('--update-only', action='store_true',
                       help='Only update existing data, do not collect fresh')
    parser.add_argument('--verify', action='store_true',
                       help='Verify data quality after collection')

    args = parser.parse_args()

    manager = DataCollectionManager()

    try:
        # Initialize
        await manager.initialize_secrets()

        if args.update_only:
            # Update existing data
            await manager.update_existing_data()
        else:
            # Collect fresh data
            await manager.collect_top_pairs_data(args.pairs, args.intervals)

        # Print report
        manager.print_collection_report()

        # Verify data quality if requested
        if args.verify:
            await manager.verify_data_quality()

        print("\n🎉 Data collection completed successfully!")
        print("💡 Next steps:")
        print("   1. Run backtesting: python3 run_backtesting.py")
        print("   2. Train AI models: python3 train_ai_models.py")
        print("   3. Deploy autonomous trading: python3 deploy_autonomous.py")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        print(f"\n❌ Data collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

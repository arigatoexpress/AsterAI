#!/usr/bin/env python3
"""
Collect 6 Months of Historical Data from Aster DEX
Multi-asset confluence trading system data collection.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiAssetDataCollector:
    """
    Collect historical data for multiple assets for confluence trading.
    """
    
    def __init__(self, output_dir: str = "data/historical/aster_dex"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = None
        
        # Target assets for multi-asset confluence
        self.target_symbols = [
            "BTCUSDT",   # Bitcoin - reference asset
            "ETHUSDT",   # Ethereum - reference asset
            "SOLUSDT",   # Solana
            "SUIUSDT",   # Sui
            "ASTERUSDT", # Aster
            "BNBUSDT",   # Binance Coin
            "ADAUSDT",   # Cardano
            "DOTUSDT",   # Polkadot
            "AVAXUSDT",  # Avalanche
            "MATICUSDT", # Polygon
        ]
        
        # Multiple timeframes for different analysis
        self.intervals = ["1h", "4h", "1d"]
        
        # 6 months of data
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)
        
        logger.info(f"📊 Multi-Asset Data Collector initialized")
        logger.info(f"   Assets: {len(self.target_symbols)}")
        logger.info(f"   Intervals: {self.intervals}")
        logger.info(f"   Date range: {self.start_date.date()} to {self.end_date.date()}")
    
    async def initialize(self):
        """Initialize the data collector."""
        self.collector = AsterDEXDataCollector()
        await self.collector.initialize()
        logger.info("✅ Collector initialized")
    
    async def collect_all_data(self):
        """Collect data for all assets and intervals."""
        total_tasks = len(self.target_symbols) * len(self.intervals)
        completed = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting data collection: {total_tasks} tasks")
        logger.info(f"{'='*60}\n")
        
        for symbol in self.target_symbols:
            for interval in self.intervals:
                try:
                    logger.info(f"📥 [{completed+1}/{total_tasks}] Collecting {symbol} {interval}...")
                    
                    # Collect historical data
                    df = await self.collector.collect_historical_data(
                        symbol=symbol,
                        start_date=self.start_date.strftime("%Y-%m-%d"),
                        end_date=self.end_date.strftime("%Y-%m-%d"),
                        interval=interval,
                        limit=1000
                    )
                    
                    if not df.empty:
                        # Save to parquet format (efficient for large datasets)
                        output_file = self.output_dir / f"{symbol}_{interval}.parquet"
                        df.to_parquet(output_file, compression='snappy')
                        
                        logger.info(f"   ✅ Saved {len(df)} records to {output_file.name}")
                        logger.info(f"   📅 Range: {df.index.min()} to {df.index.max()}")
                        logger.info(f"   💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                        
                        # Save summary statistics
                        self._save_data_summary(symbol, interval, df)
                    else:
                        logger.warning(f"   ⚠️  No data collected for {symbol} {interval}")
                    
                    completed += 1
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"   ❌ Error collecting {symbol} {interval}: {e}")
                    completed += 1
                    continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Data collection complete: {completed}/{total_tasks} tasks")
        logger.info(f"{'='*60}\n")
    
    def _save_data_summary(self, symbol: str, interval: str, df: pd.DataFrame):
        """Save data quality summary."""
        summary = {
            'symbol': symbol,
            'interval': interval,
            'records': len(df),
            'start_date': str(df.index.min()),
            'end_date': str(df.index.max()),
            'price_min': float(df['close'].min()),
            'price_max': float(df['close'].max()),
            'price_mean': float(df['close'].mean()),
            'volume_total': float(df['volume'].sum()),
            'missing_values': int(df.isnull().sum().sum()),
            'collection_time': datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "data_summary.csv"
        
        # Append or create summary file
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary])
        
        summary_df.to_csv(summary_file, index=False)
    
    async def validate_data_quality(self):
        """Validate collected data quality."""
        logger.info("\n🔍 Validating data quality...")
        
        summary_file = self.output_dir / "data_summary.csv"
        if not summary_file.exists():
            logger.warning("No summary file found")
            return
        
        summary_df = pd.read_csv(summary_file)
        
        logger.info(f"\n📊 Data Quality Report:")
        logger.info(f"   Total datasets: {len(summary_df)}")
        logger.info(f"   Symbols: {summary_df['symbol'].nunique()}")
        logger.info(f"   Intervals: {summary_df['interval'].nunique()}")
        logger.info(f"   Total records: {summary_df['records'].sum():,}")
        logger.info(f"   Date range: {summary_df['start_date'].min()} to {summary_df['end_date'].max()}")
        
        # Check for issues
        missing_data = summary_df[summary_df['records'] < 100]
        if not missing_data.empty:
            logger.warning(f"\n⚠️  Datasets with < 100 records:")
            for _, row in missing_data.iterrows():
                logger.warning(f"   - {row['symbol']} {row['interval']}: {row['records']} records")
        
        # Check for missing values
        data_with_nulls = summary_df[summary_df['missing_values'] > 0]
        if not data_with_nulls.empty:
            logger.warning(f"\n⚠️  Datasets with missing values:")
            for _, row in data_with_nulls.iterrows():
                logger.warning(f"   - {row['symbol']} {row['interval']}: {row['missing_values']} nulls")
        
        logger.info("\n✅ Data quality validation complete")
    
    async def close(self):
        """Close the collector."""
        if self.collector:
            await self.collector.close()
        logger.info("✅ Collector closed")


async def main():
    """Main execution function."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║       Aster DEX Historical Data Collection (6 Months)         ║
║          Multi-Asset Confluence Trading System                 ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    collector = MultiAssetDataCollector()
    
    try:
        # Initialize collector
        await collector.initialize()
        
        # Collect all data
        await collector.collect_all_data()
        
        # Validate data quality
        await collector.validate_data_quality()
        
        print("""
╔════════════════════════════════════════════════════════════════╗
║                  🎉 Data Collection Complete!                  ║
╚════════════════════════════════════════════════════════════════╝

Next steps:
1. Review data/historical/aster_dex/data_summary.csv
2. Proceed to feature engineering (Phase 3)
3. Train confluence models (Phase 4)

Data location: data/historical/aster_dex/
        """)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Data collection interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Data collection failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())





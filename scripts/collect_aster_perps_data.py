#!/usr/bin/env python3
"""
Aster DEX Perpetual Contracts Data Collection
Collects comprehensive data for all assets available on Aster DEX perps
Including both crypto and stock perpetual contracts
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import logging
from typing import Dict, List, Set
import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector
from scripts.collect_multi_source_crypto import MultiSourceCryptoCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsterPerpsDataCollector:
    """
    Collects data for all Aster DEX perpetual contracts
    - Crypto perpetuals (200+ assets)
    - Stock perpetuals (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA)
    """
    
    def __init__(self, output_dir: str = "data/historical/aster_perps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stock perpetuals on Aster DEX
        self.stock_perps = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
        
        self.collection_report = {
            'timestamp': datetime.now().isoformat(),
            'crypto_perps': {'total': 0, 'collected': 0, 'failed': []},
            'stock_perps': {'total': len(self.stock_perps), 'collected': 0, 'failed': []},
            'total_assets': 0,
            'total_data_points': 0
        }
    
    async def collect_all_perps(self):
        """Main method to collect all Aster DEX perpetual contracts data"""
        print("""
╔════════════════════════════════════════════════════════════════╗
║          Aster DEX Perpetual Contracts Data Collection         ║
║          Crypto Perps + Stock Perps (24/7 Trading)             ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        # Step 1: Get all available crypto perpetuals from Aster DEX
        logger.info("Step 1: Discovering available crypto perpetuals on Aster DEX...")
        crypto_symbols = await self._get_aster_crypto_perps()
        
        # Step 2: Collect data for crypto perpetuals
        logger.info(f"\nStep 2: Collecting data for {len(crypto_symbols)} crypto perpetuals...")
        await self._collect_crypto_perps(crypto_symbols)
        
        # Step 3: Collect data for stock perpetuals
        logger.info(f"\nStep 3: Collecting data for {len(self.stock_perps)} stock perpetuals...")
        await self._collect_stock_perps()
        
        # Step 4: Generate comprehensive report
        logger.info("\nStep 4: Generating collection report...")
        self._generate_report()
        
        return self.collection_report
    
    async def _get_aster_crypto_perps(self) -> List[str]:
        """Get all available crypto perpetual symbols from Aster DEX"""
        collector = AsterDEXDataCollector()
        
        try:
            await collector.initialize()
            
            # Get all available symbols
            symbols = collector.available_symbols
            logger.info(f"✅ Found {len(symbols)} crypto perpetual contracts on Aster DEX")
            
            # Filter for USDT pairs (perpetuals)
            perp_symbols = [s.replace('USDT', '') for s in symbols if 'USDT' in s]
            
            # Remove duplicates and sort
            unique_symbols = sorted(list(set(perp_symbols)))
            
            self.collection_report['crypto_perps']['total'] = len(unique_symbols)
            
            logger.info(f"📊 Filtered to {len(unique_symbols)} unique crypto assets")
            logger.info(f"   Top 20: {', '.join(unique_symbols[:20])}")
            
            return unique_symbols
            
        finally:
            await collector.cleanup()
    
    async def _collect_crypto_perps(self, symbols: List[str]):
        """Collect historical data for crypto perpetuals with multi-source backup"""
        collector = MultiSourceCryptoCollector(
            output_dir=str(self.output_dir / "crypto")
        )
        
        try:
            await collector.initialize()
            
            # Prioritize high-value assets
            priority_assets = [
                'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT',
                'MATIC', 'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM', 'ALGO',
                'VET', 'FIL', 'AAVE', 'SAND', 'MANA', 'AXS', 'THETA', 'EGLD'
            ]
            
            # Reorder: priority assets first, then others
            priority_first = [s for s in priority_assets if s in symbols]
            others = [s for s in symbols if s not in priority_assets]
            ordered_symbols = priority_first + others
            
            logger.info(f"📈 Collecting {len(ordered_symbols)} crypto perpetuals...")
            logger.info(f"   Priority assets (first): {len(priority_first)}")
            logger.info(f"   Other assets: {len(others)}")
            
            successful = 0
            for i, symbol in enumerate(ordered_symbols, 1):
                try:
                    logger.info(f"  [{i}/{len(ordered_symbols)}] Collecting {symbol}...")
                    result = await collector.collect_asset_data(symbol)
                    
                    if result['consolidated'] is not None and len(result['consolidated']) > 0:
                        successful += 1
                        points = len(result['consolidated'])
                        sources = result['metadata']['sources_used']
                        self.collection_report['total_data_points'] += points
                        logger.info(f"    ✅ {symbol}: {points} points from {sources}")
                    else:
                        self.collection_report['crypto_perps']['failed'].append(symbol)
                        logger.warning(f"    ⚠️  {symbol}: No data available")
                    
                    # Progress update every 10 assets
                    if i % 10 == 0:
                        logger.info(f"  📊 Progress: {successful}/{i} successful ({successful/i*100:.1f}%)")
                    
                except Exception as e:
                    self.collection_report['crypto_perps']['failed'].append(symbol)
                    logger.error(f"    ❌ {symbol}: {str(e)[:100]}")
            
            self.collection_report['crypto_perps']['collected'] = successful
            logger.info(f"\n✅ Crypto Perps: {successful}/{len(ordered_symbols)} collected")
            
        finally:
            await collector.cleanup()
    
    async def _collect_stock_perps(self):
        """Collect data for stock perpetuals (24/7 trading on Aster DEX)"""
        import yfinance as yf
        
        logger.info(f"📈 Collecting stock perpetual data for: {', '.join(self.stock_perps)}")
        
        stock_output = self.output_dir / "stocks"
        stock_output.mkdir(exist_ok=True)
        
        successful = 0
        for symbol in self.stock_perps:
            try:
                logger.info(f"  Collecting {symbol}...")
                
                # Download historical data (5 years)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="5y", interval="1d")
                
                if df is not None and len(df) > 0:
                    # Save to parquet
                    output_file = stock_output / f"{symbol}_stock_perp.parquet"
                    df.to_parquet(output_file)
                    
                    successful += 1
                    points = len(df)
                    self.collection_report['total_data_points'] += points
                    logger.info(f"    ✅ {symbol}: {points} daily candles")
                else:
                    self.collection_report['stock_perps']['failed'].append(symbol)
                    logger.warning(f"    ⚠️  {symbol}: No data")
                
            except Exception as e:
                self.collection_report['stock_perps']['failed'].append(symbol)
                logger.error(f"    ❌ {symbol}: {str(e)[:100]}")
        
        self.collection_report['stock_perps']['collected'] = successful
        logger.info(f"\n✅ Stock Perps: {successful}/{len(self.stock_perps)} collected")
    
    def _generate_report(self):
        """Generate comprehensive collection report"""
        self.collection_report['total_assets'] = (
            self.collection_report['crypto_perps']['collected'] +
            self.collection_report['stock_perps']['collected']
        )
        
        # Save report
        report_file = self.output_dir / "aster_perps_collection_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.collection_report, f, indent=2)
        
        # Create summary file with asset list
        summary = {
            'collection_date': self.collection_report['timestamp'],
            'total_assets': self.collection_report['total_assets'],
            'total_data_points': self.collection_report['total_data_points'],
            'crypto_perps': {
                'count': self.collection_report['crypto_perps']['collected'],
                'failed_count': len(self.collection_report['crypto_perps']['failed'])
            },
            'stock_perps': {
                'count': self.collection_report['stock_perps']['collected'],
                'assets': [s for s in self.stock_perps 
                          if s not in self.collection_report['stock_perps']['failed']]
            }
        }
        
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("ASTER DEX PERPETUALS COLLECTION REPORT")
        print("="*70)
        print(f"\n📊 Total Assets Collected: {self.collection_report['total_assets']}")
        print(f"📈 Total Data Points: {self.collection_report['total_data_points']:,}")
        
        print(f"\n🔸 Crypto Perpetuals:")
        print(f"   Collected: {self.collection_report['crypto_perps']['collected']}")
        print(f"   Failed: {len(self.collection_report['crypto_perps']['failed'])}")
        
        print(f"\n📊 Stock Perpetuals:")
        print(f"   Collected: {self.collection_report['stock_perps']['collected']}")
        print(f"   Available: {', '.join([s for s in self.stock_perps if s not in self.collection_report['stock_perps']['failed']])}")
        
        if self.collection_report['crypto_perps']['failed']:
            print(f"\n⚠️  Failed Crypto Assets ({len(self.collection_report['crypto_perps']['failed'])}):")
            for asset in self.collection_report['crypto_perps']['failed'][:20]:
                print(f"   • {asset}")
            if len(self.collection_report['crypto_perps']['failed']) > 20:
                print(f"   ... and {len(self.collection_report['crypto_perps']['failed']) - 20} more")
        
        print(f"\n✅ Report saved to: {report_file}")
        print(f"✅ Summary saved to: {summary_file}")
        print(f"✅ Data directory: {self.output_dir.absolute()}")
        print("\n" + "="*70)


async def main():
    collector = AsterPerpsDataCollector()
    try:
        report = await collector.collect_all_perps()
        return report
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Starting Aster DEX Perpetuals Data Collection...")
    try:
        asyncio.run(main())
        print("\n✅ Collection completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")


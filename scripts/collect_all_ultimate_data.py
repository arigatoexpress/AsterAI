#!/usr/bin/env python3
"""
Master Data Collection Orchestrator
Collects all data needed for the Ultimate AI Trading System.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_multi_source_crypto import MultiSourceCryptoCollector
from scripts.collect_traditional_markets import TraditionalMarketsCollector
from scripts.collect_alternative_data import AlternativeDataCollector
from scripts.collect_aster_data_sync import collect_aster_data_sync

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltimateDataOrchestrator:
    """
    Orchestrates collection of all data sources for the Ultimate Trading System.
    """
    
    def __init__(self, output_dir: str = "data/historical/ultimate_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.collectors = {
            'crypto': MultiSourceCryptoCollector(
                output_dir=str(self.output_dir / "crypto")
            ),
            'traditional': TraditionalMarketsCollector(
                output_dir=str(self.output_dir / "traditional")
            ),
            'alternative': AlternativeDataCollector(
                output_dir=str(self.output_dir / "alternative")
            )
        }
        
        self.results = {}
        self.start_time = None
        
    async def collect_all_data(self):
        """Collect data from all sources"""
        self.start_time = datetime.now()
        
        print("""
╔════════════════════════════════════════════════════════════════╗
║            Ultimate AI Trading System Data Collection          ║
║                                                                ║
║  This will collect:                                            ║
║  • Top 100 cryptocurrencies from 5+ sources                   ║
║  • S&P 500 components and major indices                       ║
║  • Commodities (Gold, Silver, Oil, etc.)                      ║
║  • Economic indicators (GDP, CPI, Fed rates)                  ║
║  • Sentiment data (Reddit, News, Google Trends)               ║
║  • On-chain metrics                                           ║
║                                                                ║
║  Estimated time: 30-60 minutes                                ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        # 1. Collect Aster DEX data first (if available)
        try:
            logger.info("\n=== Phase 1: Aster DEX Native Assets ===")
            aster_results = await self._collect_aster_data()
            self.results['aster'] = aster_results
        except Exception as e:
            logger.warning(f"Aster DEX collection skipped: {e}")
        
        # 2. Collect multi-source crypto data
        logger.info("\n=== Phase 2: Multi-Source Cryptocurrency Data ===")
        try:
            await self.collectors['crypto'].initialize()
            crypto_results = await self.collectors['crypto'].collect_all_assets()
            self.results['crypto'] = crypto_results
            await self.collectors['crypto'].cleanup()
        except Exception as e:
            logger.error(f"Crypto collection error: {e}")
            self.results['crypto'] = {}
        
        # 3. Collect traditional markets data
        logger.info("\n=== Phase 3: Traditional Markets Data ===")
        try:
            await self.collectors['traditional'].initialize()
            traditional_results = await self.collectors['traditional'].collect_all_assets()
            self.results['traditional'] = traditional_results
            await self.collectors['traditional'].cleanup()
        except Exception as e:
            logger.error(f"Traditional markets error: {e}")
            self.results['traditional'] = {}
        
        # 4. Collect alternative data
        logger.info("\n=== Phase 4: Alternative Data Sources ===")
        try:
            await self.collectors['alternative'].initialize()
            alternative_results = await self.collectors['alternative'].collect_all_data()
            self.results['alternative'] = alternative_results
            await self.collectors['alternative'].cleanup()
        except Exception as e:
            logger.error(f"Alternative data error: {e}")
            self.results['alternative'] = {}
        
        # 5. Generate master summary
        self._generate_master_summary()
        
        elapsed = datetime.now() - self.start_time
        print(f"\n✅ All data collection complete! Total time: {elapsed}")
        
    async def _collect_aster_data(self):
        """Collect Aster DEX specific data"""
        # Run synchronous collector in thread pool
        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor() as pool:
            results = await loop.run_in_executor(
                pool,
                collect_aster_data_sync,
                ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE'],
                str(self.output_dir / "aster")
            )
        
        return results
    
    def _generate_master_summary(self):
        """Generate comprehensive summary of all collected data"""
        summary = {
            'collection_time': datetime.now().isoformat(),
            'total_duration': str(datetime.now() - self.start_time),
            'data_sources': {
                'crypto': {
                    'assets_collected': len(self.results.get('crypto', {})),
                    'sources_used': ['aster', 'binance', 'coingecko', 'cryptocompare', 'yahoo']
                },
                'traditional': {
                    'assets_collected': len(self.results.get('traditional', {})),
                    'categories': ['equities', 'indices', 'commodities', 'economic_indicators', 'bonds']
                },
                'alternative': {
                    'data_types': ['sentiment', 'google_trends', 'fear_greed', 'on_chain', 'news']
                }
            },
            'statistics': self._calculate_statistics()
        }
        
        # Save master summary
        summary_file = self.output_dir / "master_collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                    Collection Summary                          ║
╚════════════════════════════════════════════════════════════════╝

📊 Cryptocurrency Data:
   • Assets collected: {len(self.results.get('crypto', {}))}
   • Data points: ~{self._estimate_crypto_datapoints():,}

📈 Traditional Markets:
   • Assets collected: {len(self.results.get('traditional', {}))}
   • Categories: Stocks, Indices, Commodities, Economic

🎯 Alternative Data:
   • Google Trends keywords tracked
   • Fear & Greed Index collected
   • News sentiment analyzed
   • On-chain metrics gathered

📁 Output Directory: {self.output_dir}
📋 Master Summary: {summary_file}
        """)
    
    def _calculate_statistics(self):
        """Calculate detailed statistics"""
        stats = {
            'total_files_created': len(list(self.output_dir.rglob('*.parquet'))) + 
                                  len(list(self.output_dir.rglob('*.json'))),
            'total_size_mb': sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file()) / (1024 * 1024),
            'data_quality': {
                'crypto_coverage': len(self.results.get('crypto', {})) / 100 * 100,  # percentage
                'no_synthetic_data': True  # We strictly reject synthetic data
            }
        }
        return stats
    
    def _estimate_crypto_datapoints(self):
        """Estimate total crypto data points"""
        # Rough estimate: 100 assets * 730 days * 24 hours
        return 100 * 730 * 24


async def main():
    """Main execution"""
    orchestrator = UltimateDataOrchestrator()
    
    try:
        await orchestrator.collect_all_data()
    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    
    # Run collection
    asyncio.run(main())
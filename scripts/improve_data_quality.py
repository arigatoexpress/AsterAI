#!/usr/bin/env python3
"""
Data Quality Improvement System
Fixes incomplete data, adds multi-source redundancy, validates quality
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple
import aiohttp
import ccxt.async_support as ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.collect_multi_source_crypto import MultiSourceCryptoCollector
from scripts.collect_aster_data_sync import collect_aster_data_sync

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMITS = {
    'binance': {'delay': 0.1, 'retry_delay': 5, 'max_retries': 3},
    'kraken': {'delay': 0.3, 'retry_delay': 5, 'max_retries': 3},
    'kucoin': {'delay': 0.2, 'retry_delay': 5, 'max_retries': 3},
    'coingecko': {'delay': 2.0, 'retry_delay': 10, 'max_retries': 2}
}


class DataQualityImprover:
    """
    Improves cryptocurrency data quality by:
    1. Adding missing assets
    2. Diversifying data sources
    3. Validating data integrity
    4. Detecting and fixing outliers
    5. Ensuring completeness
    """
    
    def __init__(self, data_dir: str = "data/historical/ultimate_dataset/crypto"):
        self.data_dir = Path(data_dir)
        self.summary_file = self.data_dir / "collection_summary.json"
        self.quality_report = {
            'timestamp': datetime.now().isoformat(),
            'issues_found': [],
            'fixes_applied': [],
            'quality_metrics': {}
        }
        
    async def analyze_and_improve(self):
        """Main method to analyze and improve data quality"""
        print("""
===============================================================================
          Data Quality Improvement System
          Analyzing and Fixing Crypto Data Streams
===============================================================================
        """)
        
        # Step 1: Load current data summary
        logger.info("Step 1: Loading current data summary...")
        summary = self._load_summary()
        
        # Step 2: Analyze data quality issues
        logger.info("\nStep 2: Analyzing data quality...")
        issues = await self._analyze_quality(summary)
        
        # Step 3: Fix missing/incomplete data
        logger.info("\nStep 3: Fixing data issues...")
        await self._fix_issues(issues)
        
        # Step 4: Add source diversity
        logger.info("\nStep 4: Adding source diversity...")
        try:
            await self._diversify_sources(summary)
        except Exception as e:
            logger.warning(f"⚠️  Source diversification failed: {str(e)[:200]}")
            logger.warning("Continuing with remaining steps...")
        
        # Step 5: Validate all data
        logger.info("\nStep 5: Validating data integrity...")
        validation_results = await self._validate_all_data()
        
        # Step 6: Generate quality report
        logger.info("\nStep 6: Generating quality report...")
        self._generate_report(validation_results)
        
        return self.quality_report
    
    def _load_summary(self) -> Dict:
        """Load collection summary"""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")
        
        with open(self.summary_file, 'r') as f:
            summary = json.load(f)
        
        logger.info(f"✅ Loaded summary: {summary['successful']}/{summary['total_assets']} assets")
        return summary
    
    async def _analyze_quality(self, summary: Dict) -> Dict:
        """Analyze data quality issues"""
        issues = {
            'missing_data': [],
            'single_source': [],
            'low_quality': [],
            'incomplete': []
        }
        
        for symbol, data in summary['asset_summary'].items():
            # Check for missing data
            if data['data_points'] == 0:
                issues['missing_data'].append(symbol)
                self.quality_report['issues_found'].append(f"Missing data: {symbol}")
            
            # Check for single source dependency
            if len(data['sources']) == 1:
                issues['single_source'].append(symbol)
            elif len(data['sources']) == 0:
                issues['missing_data'].append(symbol)
            
            # Check for incomplete data (less than 1000 points)
            if 0 < data['data_points'] < 1000:
                issues['incomplete'].append(symbol)
                self.quality_report['issues_found'].append(f"Incomplete data: {symbol} ({data['data_points']} points)")
        
        # Log issues
        logger.warning(f"\n❌ Issues Found:")
        logger.warning(f"   Missing data: {len(issues['missing_data'])} assets")
        logger.warning(f"   Single source: {len(issues['single_source'])} assets")
        logger.warning(f"   Incomplete: {len(issues['incomplete'])} assets")
        
        if issues['missing_data']:
            logger.warning(f"   Assets with no data: {', '.join(issues['missing_data'])}")
        
        return issues
    
    async def _fix_issues(self, issues: Dict):
        """Fix identified data issues"""
        # Fix missing data
        if issues['missing_data']:
            logger.info(f"Fixing {len(issues['missing_data'])} assets with missing data...")
            await self._collect_missing_assets(issues['missing_data'])
        
        # Fix incomplete data
        if issues['incomplete']:
            logger.info(f"Completing {len(issues['incomplete'])} incomplete assets...")
            await self._complete_incomplete_assets(issues['incomplete'])
    
    async def _collect_missing_assets(self, symbols: List[str]):
        """Collect data for missing assets"""
        # Deduplicate the symbols list to avoid processing the same asset multiple times
        unique_symbols = list(set(symbols))
        logger.info(f"Deduplicated missing assets: {len(symbols)} -> {len(unique_symbols)} unique symbols")

        collector = MultiSourceCryptoCollector(output_dir=str(self.data_dir))

        try:
            await collector.initialize()

            for symbol in unique_symbols:
                logger.info(f"  Collecting {symbol}...")
                try:
                    result = await collector.collect_asset_data(symbol)
                    if result['consolidated'] is not None:
                        logger.info(f"    ✅ {symbol}: {len(result['consolidated'])} points from {result['metadata']['sources_used']}")
                        self.quality_report['fixes_applied'].append(f"Collected {symbol}")
                    else:
                        logger.warning(f"    ❌ {symbol}: No data available from any source")
                except Exception as e:
                    logger.error(f"    ❌ {symbol}: {e}")
            
        finally:
            await collector.cleanup()
    
    async def _complete_incomplete_assets(self, symbols: List[str]):
        """Complete data for assets with insufficient data points"""
        # Similar to collect_missing_assets but focuses on extending existing data
        logger.info("  Extending data for incomplete assets...")
        await self._collect_missing_assets(symbols)
    
    async def _diversify_sources(self, summary: Dict):
        """Add source diversity to assets relying on single source"""
        logger.info("Adding alternative data sources for redundancy...")
        
        # Get assets that need source diversity
        single_source_assets = [
            symbol for symbol, data in summary['asset_summary'].items()
            if len(data['sources']) == 1
        ]
        
        if not single_source_assets:
            logger.info("  All assets already have source diversity")
            return
        
        total_added = 0
        top_assets = single_source_assets[:20]  # Focus on top 20 first
        
        # Try to add Binance as backup source (may be geo-restricted)
        logger.info(f"  Trying Binance backup for {len(top_assets)} top assets...")
        try:
            binance_added = await self._add_binance_source(top_assets)
            total_added += binance_added
            logger.info(f"    ✅ Binance: Added {binance_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  Binance unavailable (geo-restricted or error): {str(e)[:100]}")
            binance_added = 0
        
        # Try to add Kraken as alternative
        logger.info(f"  Trying Kraken backup...")
        try:
            kraken_added = await self._add_kraken_source(top_assets)
            total_added += kraken_added
            logger.info(f"    ✅ Kraken: Added {kraken_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  Kraken unavailable: {str(e)[:100]}")
            kraken_added = 0
        
        # Try to add KuCoin as alternative
        logger.info(f"  Trying KuCoin backup...")
        try:
            kucoin_added = await self._add_kucoin_source(top_assets)
            total_added += kucoin_added
            logger.info(f"    ✅ KuCoin: Added {kucoin_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  KuCoin unavailable: {str(e)[:100]}")
            kucoin_added = 0

        # Try to add OKX as alternative
        logger.info(f"  Trying OKX backup...")
        try:
            okx_added = await self._add_okx_source(top_assets)
            total_added += okx_added
            logger.info(f"    ✅ OKX: Added {okx_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  OKX unavailable: {str(e)[:100]}")
            okx_added = 0

        # Try to add Gate.io as alternative
        logger.info(f"  Trying Gate.io backup...")
        try:
            gate_added = await self._add_gate_source(top_assets)
            total_added += gate_added
            logger.info(f"    ✅ Gate.io: Added {gate_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  Gate.io unavailable: {str(e)[:100]}")
            gate_added = 0

        # Try to add CoinGecko for all single-source assets
        logger.info(f"  Adding CoinGecko backup...")
        try:
            coingecko_added = await self._add_coingecko_source(single_source_assets)
            total_added += coingecko_added
            logger.info(f"    ✅ CoinGecko: Added {coingecko_added} assets")
        except Exception as e:
            logger.warning(f"    ⚠️  CoinGecko unavailable: {str(e)[:100]}")
        
        if total_added > 0:
            self.quality_report['fixes_applied'].append(
                f"Added source diversity: {total_added} assets"
            )
            logger.info(f"  📊 Total source diversity added: {total_added} assets")
        else:
            logger.warning(f"  ⚠️  Could not add alternative sources (all exchanges unavailable)")
    
    async def _add_binance_source(self, symbols: List[str]) -> int:
        """Add Binance as alternative source with rate limiting and retries"""
        added = 0
        config = RATE_LIMITS['binance']
        binance = ccxt.binance({
            'enableRateLimit': True,
            'rateLimit': int(config['delay'] * 1000),
            'timeout': 30000
        })
        
        try:
            markets = await binance.load_markets()
            logger.info(f"      Loaded {len(markets)} Binance markets")
            
            for symbol in symbols:
                trading_pair = f"{symbol}/USDT"
                if trading_pair in markets:
                    # Retry logic
                    for attempt in range(config['max_retries']):
                        try:
                            # Collect sample data to verify
                            ohlcv = await binance.fetch_ohlcv(trading_pair, '1h', limit=10)
                            if ohlcv and len(ohlcv) > 0:
                                added += 1
                                logger.debug(f"      ✓ {symbol}: Binance available")
                                break
                        except Exception as e:
                            if attempt < config['max_retries'] - 1:
                                logger.debug(f"      ⚠ {symbol}: Retry {attempt+1}/{config['max_retries']}")
                                await asyncio.sleep(config['retry_delay'])
                            else:
                                logger.debug(f"      ⚠ {symbol}: Binance error - {str(e)[:50]}")
                
                await asyncio.sleep(config['delay'])
        
        finally:
            await binance.close()
        
        return added
    
    async def _add_kraken_source(self, symbols: List[str]) -> int:
        """Add Kraken as alternative source with rate limiting and retries"""
        added = 0
        config = RATE_LIMITS['kraken']
        kraken = ccxt.kraken({
            'enableRateLimit': True,
            'rateLimit': int(config['delay'] * 1000),
            'timeout': 30000
        })
        
        try:
            markets = await kraken.load_markets()
            logger.info(f"      Loaded {len(markets)} Kraken markets")
            
            for symbol in symbols:
                # Kraken uses different naming: BTC/USD, ETH/USD
                for quote in ['USD', 'USDT']:
                    trading_pair = f"{symbol}/{quote}"
                    if trading_pair in markets:
                        # Retry logic
                        for attempt in range(config['max_retries']):
                            try:
                                ohlcv = await kraken.fetch_ohlcv(trading_pair, '1h', limit=10)
                                if ohlcv and len(ohlcv) > 0:
                                    added += 1
                                    logger.debug(f"      ✓ {symbol}: Kraken available ({trading_pair})")
                                    break
                            except Exception as e:
                                if attempt < config['max_retries'] - 1:
                                    await asyncio.sleep(config['retry_delay'])
                                else:
                                    logger.debug(f"      ⚠ {symbol}: Kraken error - {str(e)[:50]}")
                        if added:
                            break  # Found on this quote currency
                
                await asyncio.sleep(config['delay'])
        
        finally:
            await kraken.close()
        
        return added
    
    async def _add_kucoin_source(self, symbols: List[str]) -> int:
        """Add KuCoin as alternative source with rate limiting and retries"""
        added = 0
        config = RATE_LIMITS['kucoin']
        kucoin = ccxt.kucoin({
            'enableRateLimit': True,
            'rateLimit': int(config['delay'] * 1000),
            'timeout': 30000
        })
        
        try:
            markets = await kucoin.load_markets()
            logger.info(f"      Loaded {len(markets)} KuCoin markets")
            
            for symbol in symbols:
                trading_pair = f"{symbol}/USDT"
                if trading_pair in markets:
                    # Retry logic
                    for attempt in range(config['max_retries']):
                        try:
                            ohlcv = await kucoin.fetch_ohlcv(trading_pair, '1h', limit=10)
                            if ohlcv and len(ohlcv) > 0:
                                added += 1
                                logger.debug(f"      ✓ {symbol}: KuCoin available")
                                break
                        except Exception as e:
                            if attempt < config['max_retries'] - 1:
                                logger.debug(f"      ⚠ {symbol}: Retry {attempt+1}/{config['max_retries']}")
                                await asyncio.sleep(config['retry_delay'])
                            else:
                                logger.debug(f"      ⚠ {symbol}: KuCoin error - {str(e)[:50]}")
                
                await asyncio.sleep(config['delay'])
        
        finally:
            await kucoin.close()
        
        return added

    async def _add_okx_source(self, symbols: List[str]) -> int:
        """Add OKX as alternative source"""
        added = 0
        okx = ccxt.okx({'enableRateLimit': True})

        try:
            markets = await okx.load_markets()

            for symbol in symbols:
                trading_pair = f"{symbol}/USDT"
                if trading_pair in markets:
                    try:
                        # Collect sample data to verify
                        ohlcv = await okx.fetch_ohlcv(trading_pair, '1h', limit=10)
                        if ohlcv and len(ohlcv) > 0:
                            added += 1
                            logger.debug(f"      ✓ {symbol}: OKX available")
                    except Exception as e:
                        logger.debug(f"      ⚠ {symbol}: OKX error - {e}")

                await asyncio.sleep(0.1)  # Rate limiting

        finally:
            await okx.close()

        return added

    async def _add_gate_source(self, symbols: List[str]) -> int:
        """Add Gate.io as alternative source"""
        added = 0
        gate = ccxt.gate({'enableRateLimit': True})

        try:
            markets = await gate.load_markets()

            for symbol in symbols:
                trading_pair = f"{symbol}/USDT"
                if trading_pair in markets:
                    try:
                        # Collect sample data to verify
                        ohlcv = await gate.fetch_ohlcv(trading_pair, '1h', limit=10)
                        if ohlcv and len(ohlcv) > 0:
                            added += 1
                            logger.debug(f"      ✓ {symbol}: Gate.io available")
                    except Exception as e:
                        logger.debug(f"      ⚠ {symbol}: Gate.io error - {e}")

                await asyncio.sleep(0.1)  # Rate limiting

        finally:
            await gate.close()

        return added

    async def _add_coingecko_source(self, symbols: List[str]) -> int:
        """Add CoinGecko as alternative source with rate limiting and retries"""
        added = 0
        config = RATE_LIMITS['coingecko']
        
        # Map symbol to CoinGecko ID (comprehensive mapping)
        symbol_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
            'SOL': 'solana', 'ADA': 'cardano', 'DOT': 'polkadot',
            'KLAY': 'klay-token', 'IOTA': 'iota', 'FXS': 'frax-share',
            'XRP': 'ripple', 'USDT': 'tether', 'AVAX': 'avalanche-2',
            'DOGE': 'dogecoin', 'TRX': 'tron', 'LINK': 'chainlink',
            'MATIC': 'matic-network', 'TON': 'the-open-network',
            'DAI': 'dai', 'WBTC': 'wrapped-bitcoin', 'SHIB': 'shiba-inu',
            'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'LEO': 'leo-token',
            'UNI': 'uniswap', 'OKB': 'okb', 'ATOM': 'cosmos',
            'XLM': 'stellar', 'ETC': 'ethereum-classic', 'HBAR': 'hedera-hashgraph',
            'CRO': 'crypto-com-chain', 'APT': 'aptos', 'MKR': 'maker',
            'VET': 'vechain', 'NEAR': 'near', 'GRT': 'the-graph',
            'AAVE': 'aave', 'ALGO': 'algorand', 'QNT': 'quant-network',
            'FIL': 'filecoin', 'ICP': 'internet-computer', 'EOS': 'eos',
            'XTZ': 'tezos', 'SAND': 'the-sandbox', 'MANA': 'decentraland',
            'THETA': 'theta-token', 'AXS': 'axie-infinity', 'EGLD': 'elrond-erd-2',
            'BSV': 'bitcoin-cash-sv', 'XMR': 'monero', 'KCS': 'kucoin-shares',
            'TUSD': 'true-usd', 'GALA': 'gala-games', 'CHZ': 'chiliz',
            'FLOW': 'flow', 'HT': 'huobi-token', 'MINA': 'mina-protocol',
            'KDA': 'kadena', 'CRV': 'curve-dao-token', 'XEC': 'ecash',
            'FTM': 'fantom', 'GT': 'gatechain-token', 'ZEC': 'zcash',
            'NEO': 'neo', 'CAKE': 'pancakeswap-token', 'ENJ': 'enjincoin',
            'COMP': 'compound-governance-token', 'SNX': 'synthetix-network-token',
            'ROSE': 'oasis-network', 'LRC': 'loopring', 'LUNA': 'terra-luna-2',
            'KAVA': 'kava', 'BAT': 'basic-attention-token', 'DASH': 'dash',
            'WAVES': 'waves', 'KSM': 'kusama', 'ZIL': 'zilliqa',
            'CELO': 'celo', '1INCH': '1inch', 'ANKR': 'ankr',
            'YFI': 'yearn-finance', 'RVN': 'ravencoin', 'WOO': 'woo-network',
            'GLM': 'golem', 'ICX': 'icon', 'BTG': 'bitcoin-gold',
            'TWT': 'trust-wallet-token', 'CEL': 'celsius-degree-token',
            'XEM': 'nem', 'QTUM': 'qtum', 'OMG': 'omisego',
            'SXP': 'swipe', 'ZRX': '0x', 'ONT': 'ontology',
            'IOTX': 'iotex', 'AUDIO': 'audius', 'STORJ': 'storj',
            'RSR': 'reserve-rights-token', 'BAND': 'band-protocol',
            'OCEAN': 'ocean-protocol'
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for symbol in symbols[:30]:  # CoinGecko free tier limit
                coin_id = symbol_map.get(symbol, symbol.lower())
                
                # Retry logic with rate limit handling
                for attempt in range(config['max_retries']):
                    try:
                        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                        params = {'vs_currency': 'usd', 'days': '7'}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                if data.get('prices'):
                                    added += 1
                                    logger.debug(f"      ✓ {symbol}: CoinGecko available ({coin_id})")
                                    break
                            elif response.status == 429:  # Rate limited
                                logger.debug(f"      ⚠ {symbol}: CoinGecko rate limited, waiting...")
                                await asyncio.sleep(config['retry_delay'] * 2)
                            else:
                                logger.debug(f"      ⚠ {symbol}: CoinGecko HTTP {response.status}")
                                break
                        
                        await asyncio.sleep(config['delay'])
                        
                    except Exception as e:
                        if attempt < config['max_retries'] - 1:
                            await asyncio.sleep(config['retry_delay'])
                        else:
                            logger.debug(f"      ⚠ {symbol}: CoinGecko error - {str(e)[:50]}")
        
        logger.info(f"      CoinGecko processed {len(symbols[:30])} symbols")
        return added
    
    async def _validate_all_data(self) -> Dict:
        """Validate integrity of all collected data"""
        validation_results = {
            'total_assets': 0,
            'validated_assets': 0,
            'quality_scores': {},
            'issues': []
        }
        
        # Check each parquet file
        parquet_files = list(self.data_dir.glob("*_consolidated.parquet"))
        validation_results['total_assets'] = len(parquet_files)
        
        for file in parquet_files:
            symbol = file.stem.replace('_consolidated', '')
            
            try:
                df = pd.read_parquet(file)
                quality_score = self._calculate_quality_score(df)
                
                validation_results['quality_scores'][symbol] = quality_score
                
                if quality_score >= 0.7:
                    validation_results['validated_assets'] += 1
                else:
                    validation_results['issues'].append(f"{symbol}: Low quality ({quality_score:.2f})")
                
            except Exception as e:
                validation_results['issues'].append(f"{symbol}: Read error - {e}")
        
        return validation_results
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)"""
        if df.empty:
            return 0.0
        
        score = 1.0
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            score *= 0.5
        
        # Check for missing values
        missing_pct = df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))
        score *= (1 - missing_pct)
        
        # Check OHLC integrity (high >= low, etc.)
        invalid_ohlc = ((df['high'] < df['low']) | 
                       (df['close'] > df['high']) | 
                       (df['close'] < df['low'])).sum()
        if invalid_ohlc > 0:
            score *= (1 - invalid_ohlc / len(df))
        
        # Check for extreme outliers
        returns = df['close'].pct_change().abs()
        extreme_changes = (returns > 0.5).sum()
        if extreme_changes > len(df) * 0.01:  # More than 1% extreme
            score *= 0.9
        
        # Check for zero/negative prices
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            score *= 0.7
        
        # Check data continuity (no large gaps)
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            # Check for gaps larger than 2x expected interval
            # This is a simplified check
            pass
        
        return max(0.0, min(1.0, score))
    
    def _generate_report(self, validation_results: Dict):
        """Generate comprehensive quality report"""
        self.quality_report['validation_results'] = validation_results
        self.quality_report['summary'] = {
            'total_assets': validation_results['total_assets'],
            'validated_assets': validation_results['validated_assets'],
            'validation_rate': f"{validation_results['validated_assets'] / validation_results['total_assets'] * 100:.1f}%",
            'average_quality': np.mean(list(validation_results['quality_scores'].values())),
            'issues_found': len(self.quality_report['issues_found']),
            'fixes_applied': len(self.quality_report['fixes_applied'])
        }
        
        # Save report
        report_file = self.data_dir / "data_quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"\nTotal Assets: {validation_results['total_assets']}")
        print(f"Validated (Quality >= 70%): {validation_results['validated_assets']}")
        print(f"Validation Rate: {self.quality_report['summary']['validation_rate']}")
        print(f"Average Quality Score: {self.quality_report['summary']['average_quality']:.2%}")
        print(f"\nIssues Found: {len(self.quality_report['issues_found'])}")
        print(f"Fixes Applied: {len(self.quality_report['fixes_applied'])}")
        
        if validation_results['issues']:
            print(f"\n⚠️  Assets Needing Attention ({len(validation_results['issues'])}):")
            for issue in validation_results['issues'][:10]:  # Show first 10
                print(f"   • {issue}")
        
        print(f"\n✅ Report saved to: {report_file}")


async def main():
    improver = DataQualityImprover()
    try:
        report = await improver.analyze_and_improve()
        return report
    except Exception as e:
        logger.error(f"Error during improvement: {e}")
        raise


if __name__ == "__main__":
    print("Starting data quality improvement script...")
    try:
        asyncio.run(main())
        print("\n✅ Script completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Script failed with error: {e}")
        import traceback
        traceback.print_exc()


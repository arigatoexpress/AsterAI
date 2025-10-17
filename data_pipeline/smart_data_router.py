"""
Smart Data Router with VPN Awareness
Intelligently routes data requests to optimal source based on:
1. VPN latency considerations
2. Source availability and health
3. Data freshness requirements
4. Cost optimization

Key Features:
- Automatic failover between data sources
- VPN-aware routing (prefer non-VPN when possible)
- Health monitoring and adaptive routing
- Connection pooling across all sources
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path

# Import collectors
from data_pipeline.binance_vpn_optimizer import VPNOptimizedBinanceCollector
from data_pipeline.aster_dex_data_collector import AsterDEXDataCollector

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    priority: int  # Lower = higher priority
    vpn_required: bool
    vpn_location: Optional[str]
    avg_latency_ms: float
    rate_limit_per_min: int
    cost_per_1k_requests: float
    has_perpetuals: bool = True
    has_spot: bool = True


class SmartDataRouter:
    """
    Intelligent routing for multi-source data collection
    
    Features:
    - VPN-aware routing (prefer non-VPN sources when possible)
    - Automatic failover on source failure
    - Health monitoring with exponential moving average
    - Connection pooling for all sources
    - Cost optimization
    """
    
    def __init__(self, vpn_location: str = "iceland", output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/historical/multi_source")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vpn_location = vpn_location

        # Define data sources with VPN considerations
        self.sources = {
            'aster': DataSourceConfig(
                name='aster',
                priority=1,  # Highest priority (native platform)
                vpn_required=False,  # Direct access
                vpn_location=None,
                avg_latency_ms=50,
                rate_limit_per_min=300,
                cost_per_1k_requests=0.0,
                has_perpetuals=True,
                has_spot=True
            ),
            'binance_vpn': DataSourceConfig(
                name='binance_vpn',
                priority=2,  # Second priority
                vpn_required=True,  # Requires Proton VPN Iceland
                vpn_location='iceland',
                avg_latency_ms=200,  # Through VPN
                rate_limit_per_min=1200,
                cost_per_1k_requests=0.0,
                has_perpetuals=True,
                has_spot=True
            ),
            'coingecko': DataSourceConfig(
                name='coingecko',
                priority=3,
                vpn_required=False,
                vpn_location=None,
                avg_latency_ms=300,
                rate_limit_per_min=30,  # Free tier limit
                cost_per_1k_requests=0.0,
                has_perpetuals=False,  # Spot only
                has_spot=True
            ),
        }
        
        # Health tracking (0.0 = dead, 1.0 = perfect)
        self.source_health = {name: 1.0 for name in self.sources}
        
        # Initialize collectors
        self.collectors = {}
        self._initialized = False
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'source_usage': {name: 0 for name in self.sources},
            'failures': {name: 0 for name in self.sources},
        }
        
        logger.info("Smart Data Router initialized with VPN-aware routing")
    
    async def initialize(self):
        """Initialize all data collectors"""
        if self._initialized:
            return
        
        try:
            # Initialize Aster collector (non-VPN)
            logger.info("Initializing Aster DEX collector...")
            self.collectors['aster'] = AsterDEXDataCollector()
            await self.collectors['aster'].initialize()
            
            # Initialize Binance VPN collector
            logger.info(f"Initializing Binance VPN-optimized collector for {self.vpn_location}...")
            self.collectors['binance_vpn'] = VPNOptimizedBinanceCollector(vpn_location=self.vpn_location)
            await self.collectors['binance_vpn'].initialize()
            
            # CoinGecko would go here (not implemented yet)
            # self.collectors['coingecko'] = CoinGeckoCollector()
            
            self._initialized = True
            logger.info(f"✅ Smart Router initialized with {len(self.collectors)} sources")
            
        except Exception as e:
            logger.error(f"Failed to initialize Smart Router: {e}")
            raise
    
    async def collect_symbol(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Collect data with intelligent source routing
        
        Logic:
        1. Try sources in optimal order (considering VPN, health, latency)
        2. Return first successful result
        3. Update health scores based on success/failure
        4. Track performance statistics
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            timeframe: Candle timeframe ('1h', '4h', '1d')
            limit: Number of candles to fetch
            use_cache: Whether to use caching
        
        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        
        if not self._initialized:
            await self.initialize()
        
        self.stats['total_requests'] += 1
        
        # Get optimal source order
        sorted_sources = self._get_optimal_source_order(symbol, timeframe)
        
        # Try each source until one succeeds
        for source_name in sorted_sources:
            if source_name not in self.collectors:
                continue
            
            try:
                logger.debug(f"Trying {source_name} for {symbol}")
                start_time = datetime.now()
                
                # Fetch from source
                data = await self._fetch_from_source(
                    source_name,
                    symbol,
                    timeframe,
                    limit,
                    use_cache
                )
                
                if data is not None and not data.empty:
                    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                    
                    self.stats['successful_requests'] += 1
                    self.stats['source_usage'][source_name] += 1
                    self._update_source_health(source_name, success=True, latency_ms=latency_ms)
                    
                    logger.info(f"✅ {symbol} from {source_name} ({latency_ms:.0f}ms, {len(data)} candles)")
                    return data
                else:
                    logger.debug(f"⚠️ {source_name} returned empty data for {symbol}")
                    
            except Exception as e:
                logger.warning(f"❌ {source_name} failed for {symbol}: {e}")
                self.stats['failures'][source_name] += 1
                self._update_source_health(source_name, success=False)
                continue
        
        logger.error(f"All sources failed for {symbol}")
        return None
    
    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols with intelligent routing
        
        Optimizations:
        - Batch requests to sources that support it
        - Parallel collection with asyncio.gather
        - Smart fallback on failures
        """
        
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"Collecting {len(symbols)} symbols via smart routing...")
        
        # Collect in parallel for speed
        tasks = [
            self.collect_symbol(symbol, timeframe, limit)
            for symbol in symbols
        ]
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Package results
        results = {}
        for symbol, result in zip(symbols, results_list):
            if isinstance(result, Exception):
                logger.warning(f"Failed {symbol}: {result}")
                results[symbol] = None
            else:
                results[symbol] = result
        
        # Log summary
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"✅ Collected {successful}/{len(symbols)} symbols")
        
        return results
    
    async def _fetch_from_source(
        self,
        source_name: str,
        symbol: str,
        timeframe: str,
        limit: int,
        use_cache: bool
    ) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        
        collector = self.collectors.get(source_name)
        if not collector:
            return None
        
        if source_name == 'aster':
            # Aster DEX collector
            return await collector.collect_historical_data(
                symbol=f"{symbol}USDT",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d'),
                interval=timeframe,
                limit=limit
            )
            
        elif source_name == 'binance_vpn':
            # Binance VPN-optimized collector
            results = await collector.collect_with_vpn_optimization(
                symbols=[symbol],
                timeframe=timeframe,
                limit=limit,
                use_cache=use_cache
            )
            return results.get(symbol)
        
        else:
            logger.warning(f"Unknown source: {source_name}")
            return None
    
    def _get_optimal_source_order(self, symbol: str, timeframe: str) -> List[str]:
        """
        Determine optimal source order based on:
        1. Source health (recent success rate)
        2. VPN overhead (prefer non-VPN when possible)
        3. Rate limit availability
        4. Historical latency
        5. Feature support (perpetuals vs spot)
        
        Returns:
            List of source names in optimal order
        """
        
        scores = {}
        for name, config in self.sources.items():
            if name not in self.collectors:
                continue
            
            score = 0
            
            # Health weight (most important - 100 points)
            score += self.source_health[name] * 100
            
            # Priority weight (50 points)
            score += (10 - config.priority) * 5
            
            # VPN penalty (prefer non-VPN sources - 30 points)
            if config.vpn_required:
                score -= 30
            else:
                score += 10  # Bonus for non-VPN
            
            # Latency penalty (lower is better - 20 points)
            score -= config.avg_latency_ms / 10
            
            # Rate limit consideration (10 points)
            if config.rate_limit_per_min > 100:
                score += 10
            
            scores[name] = score
        
        # Sort by score (highest first)
        sorted_sources = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        logger.debug(f"Optimal source order for {symbol}: {sorted_sources}")
        logger.debug(f"Scores: {scores}")
        
        return sorted_sources
    
    def _update_source_health(
        self,
        source: str,
        success: bool,
        latency_ms: Optional[float] = None
    ):
        """
        Update source health score using exponential moving average
        
        Health score represents recent reliability:
        - 1.0 = Perfect (100% success)
        - 0.5 = Moderate (50% success)
        - 0.0 = Dead (0% success)
        """
        
        alpha = 0.1  # Learning rate (adjust slowly)
        current_health = self.source_health[source]
        new_value = 1.0 if success else 0.0
        
        # Update health with EMA
        self.source_health[source] = current_health * (1 - alpha) + new_value * alpha
        
        # Update latency if provided
        if success and latency_ms is not None:
            config = self.sources[source]
            config.avg_latency_ms = config.avg_latency_ms * 0.9 + latency_ms * 0.1
        
        logger.debug(f"{source} health: {self.source_health[source]:.2f} "
                    f"(latency: {self.sources[source].avg_latency_ms:.0f}ms)")
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        
        total = self.stats['total_requests']
        successful = self.stats['successful_requests']
        success_rate = successful / max(1, total)
        
        report = {
            'total_requests': total,
            'successful_requests': successful,
            'success_rate': success_rate,
            'source_usage': self.stats['source_usage'],
            'source_health': self.source_health,
            'failures': self.stats['failures'],
            'sources': {
                name: {
                    'config': {
                        'priority': config.priority,
                        'vpn_required': config.vpn_required,
                        'latency_ms': config.avg_latency_ms,
                    },
                    'health': self.source_health[name],
                    'usage_count': self.stats['source_usage'][name],
                    'failure_count': self.stats['failures'][name],
                }
                for name, config in self.sources.items()
            }
        }
        
        return report
    
    async def close(self):
        """Close all collector connections"""
        for name, collector in self.collectors.items():
            try:
                if hasattr(collector, 'close'):
                    await collector.close()
                    logger.info(f"Closed {name} collector")
            except Exception as e:
                logger.warning(f"Error closing {name}: {e}")


async def test_smart_router():
    """Test smart router with multiple symbols"""
    router = SmartDataRouter()
    
    try:
        # Test symbols
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'AVAX']
        
        print("🔍 Testing Smart Data Router with VPN awareness...")
        print(f"Symbols: {symbols}\n")
        
        start = datetime.now()
        results = await router.collect_multiple_symbols(
            symbols=symbols,
            timeframe='1h',
            limit=100
        )
        duration = (datetime.now() - start).total_seconds()
        
        print(f"\n✅ Collection complete in {duration:.2f}s")
        print(f"Successfully collected: {sum(1 for v in results.values() if v is not None)}/{len(symbols)}")
        
        # Show performance report
        report = router.get_performance_report()
        print(f"\n📊 Performance Report:")
        print(f"  Total requests: {report['total_requests']}")
        print(f"  Success rate: {report['success_rate']:.1%}")
        print(f"\n  Source Usage:")
        for source, count in report['source_usage'].items():
            health = report['source_health'][source]
            print(f"    {source}: {count} requests (health: {health:.2f})")
        
    finally:
        await router.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_smart_router())


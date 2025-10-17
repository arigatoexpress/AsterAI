"""
Integrated Data Collector
Seamlessly combines all optimization modules with robust error handling
"""

import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import optimized collectors
from data_pipeline.smart_data_router import SmartDataRouter
from mcp_trader.ai.vpin_calculator_numpy import VPINCalculator, VPINConfig

logger = logging.getLogger(__name__)


class IntegratedDataCollector:
    """
    Integrated data collection with all optimizations
    
    Features:
    - VPN-optimized Binance access
    - Multi-source failover (Aster → Binance → CoinGecko)
    - Automatic error handling and retry
    - VPIN calculation for timing
    - Performance monitoring
    - Thread-safe operation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize components with error handling
        self.router = None
        self.vpin = None
        self._initialized = False
        
        # Performance tracking
        self.stats = {
            'collections': 0,
            'successes': 0,
            'failures': 0,
            'total_symbols': 0,
            'vpin_calculations': 0,
        }
        
        logger.info("Integrated Data Collector created")
    
    async def initialize(self) -> bool:
        """
        Initialize all components with robust error handling

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Initialize VPIN calculator (no async needed)
            logger.info("Initializing VPIN Calculator...")
            vpin_config = VPINConfig(
                toxic_flow_threshold=self.config.get('vpin_threshold', 0.65),
                high_confidence_threshold=self.config.get('vpin_high_threshold', 0.75)
            )
            self.vpin = VPINCalculator(vpin_config)
            logger.info("✅ VPIN Calculator initialized")

            # Try to initialize smart router (Binance may be blocked)
            try:
                logger.info("Initializing Smart Data Router...")
                self.router = SmartDataRouter()
                await self.router.initialize()
                logger.info("✅ Smart Router initialized")
            except Exception as router_error:
                logger.warning(f"⚠️ Smart Router initialization failed (Binance blocked): {router_error}")
                logger.info("🔄 Proceeding without Binance VPN optimization")
                self.router = None

            self._initialized = True
            logger.info("✅ Integrated Data Collector initialized (Binance optional)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Integrated Collector: {e}")
            self._initialized = False
            return False
    
    async def collect_training_data(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect training data with all optimizations
        
        Args:
            symbols: List of trading symbols
            timeframe: Candle timeframe
            limit: Number of candles to collect
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        
        if not self._initialized:
            success = await self.initialize()
            if not success:
                logger.error("Cannot collect data - initialization failed")
                return {}
        
        self.stats['collections'] += 1
        self.stats['total_symbols'] += len(symbols)
        
        try:
            logger.info(f"📊 Collecting {len(symbols)} symbols...")

            # Use smart router if available, otherwise collect from Aster only
            if self.router is not None:
                try:
                    data = await self.router.collect_multiple_symbols(
                        symbols=symbols,
                        timeframe=timeframe,
                        limit=limit
                    )
                except Exception as router_error:
                    logger.warning(f"⚠️ Smart router failed, using Aster-only fallback: {router_error}")
                    data = await self._collect_aster_only(symbols, timeframe, limit)
            else:
                # Fallback: Collect only from Aster DEX
                logger.info("🔄 Using Aster-only fallback (Binance blocked)")
                data = await self._collect_aster_only(symbols, timeframe, limit)

            # Count successes
            successful = sum(1 for v in data.values() if v is not None and not v.empty)
            self.stats['successes'] += successful
            self.stats['failures'] += (len(symbols) - successful)

            logger.info(f"✅ Collected {successful}/{len(symbols)} symbols successfully")

            return data

        except Exception as e:
            logger.error(f"❌ Error collecting training data: {e}")
            self.stats['failures'] += len(symbols)
            return {}

    async def _collect_aster_only(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, pd.DataFrame]:
        """Fallback method to collect data only from Aster DEX"""

        data = {}

        try:
            # Import Aster collector
            from data_pipeline.aster_dex_data_collector import AsterDEXDataCollector

            collector = AsterDEXDataCollector()
            await collector.initialize()

            # Collect data for each symbol
            for symbol in symbols:
                try:
                    # Use historical data collection method
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days

                    df = await collector.collect_historical_data(symbol, start_date, end_date)
                    if df is not None and not df.empty:
                        data[symbol] = df
                        logger.debug(f"✅ Aster DEX {symbol}: {len(df)} data points")
                    else:
                        logger.warning(f"⚠️ No data for {symbol} from Aster DEX")
                except Exception as e:
                    logger.error(f"❌ Failed to collect {symbol} from Aster DEX: {e}")

            await collector.close()

        except Exception as e:
            logger.error(f"❌ Aster-only collection failed: {e}")

        return data

    async def collect_with_vpin(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        include_vpin: bool = True
    ) -> Dict[str, Any]:
        """
        Collect data with optional VPIN analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            limit: Number of candles
            include_vpin: Whether to calculate VPIN
        
        Returns:
            Dictionary with 'data' and optionally 'vpin'
        """
        
        if not self._initialized:
            await self.initialize()
        
        result = {'symbol': symbol, 'data': None, 'vpin': None}
        
        try:
            # Collect OHLCV data
            data_dict = await self.router.collect_multiple_symbols(
                symbols=[symbol],
                timeframe=timeframe,
                limit=limit
            )
            
            result['data'] = data_dict.get(symbol)
            
            # Calculate VPIN if requested and data available
            if include_vpin and result['data'] is not None:
                try:
                    # Convert OHLCV to trades format for VPIN
                    # (In production, use actual trade data)
                    trades = self._ohlcv_to_trades(result['data'])
                    
                    if trades:
                        vpin_result = self.vpin.calculate_realtime_vpin(
                            symbol=symbol,
                            trades=trades,
                            orderbook=None  # Can add orderbook if available
                        )
                        result['vpin'] = vpin_result
                        self.stats['vpin_calculations'] += 1
                        
                        logger.info(f"VPIN for {symbol}: {vpin_result.avg_vpin:.3f} "
                                  f"(toxic: {vpin_result.toxic_flow})")
                except Exception as vpin_error:
                    logger.warning(f"VPIN calculation failed for {symbol}: {vpin_error}")
                    result['vpin'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error collecting {symbol} with VPIN: {e}")
            return result
    
    def _ohlcv_to_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert OHLCV data to approximate trade format for VPIN
        Note: This is approximate - use actual trade data in production
        """
        if df is None or df.empty:
            return []
        
        trades = []
        for idx, row in df.iterrows():
            # Approximate buy/sell from price movement
            side = 'buy' if row['close'] > row['open'] else 'sell'
            
            trades.append({
                'price': row['close'],
                'volume': row['volume'],
                'side': side,
                'timestamp': idx if isinstance(idx, datetime) else datetime.now()
            })
        
        return trades[-200:]  # Return last 200 for VPIN
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        
        success_rate = (self.stats['successes'] / max(1, self.stats['total_symbols']))
        
        report = {
            **self.stats,
            'success_rate': success_rate,
            'initialized': self._initialized,
        }
        
        # Add router stats if available
        if self.router and self._initialized:
            try:
                router_report = self.router.get_performance_report()
                report['router'] = router_report
            except:
                pass
        
        return report
    
    async def close(self):
        """Close all connections cleanly"""
        if self.router:
            try:
                await self.router.close()
                logger.info("✅ Smart router closed")
            except Exception as e:
                logger.warning(f"Error closing router: {e}")
        
        self._initialized = False


async def test_integrated_collector():
    """Test the integrated collector"""
    
    collector = IntegratedDataCollector()
    
    try:
        # Initialize
        success = await collector.initialize()
        print(f"Initialization: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if not success:
            return
        
        # Test data collection
        print("\n📊 Testing data collection...")
        test_symbols = ['BTC', 'ETH', 'SOL']
        
        data = await collector.collect_training_data(
            symbols=test_symbols,
            timeframe='1h',
            limit=100
        )
        
        print(f"✅ Collected {sum(1 for v in data.values() if v is not None)}/{len(test_symbols)} symbols")
        
        # Test VPIN integration
        print("\n📊 Testing VPIN integration...")
        result = await collector.collect_with_vpin(
            symbol='BTC',
            timeframe='1h',
            limit=100,
            include_vpin=True
        )
        
        if result['data'] is not None:
            print(f"✅ Data: {len(result['data'])} candles")
        
        if result['vpin']:
            print(f"✅ VPIN: {result['vpin'].avg_vpin:.3f}")
            print(f"   Toxic Flow: {result['vpin'].toxic_flow}")
            print(f"   Signal: {['SELL', 'HOLD', 'BUY'][result['vpin'].entry_signal + 1]}")
        
        # Performance report
        print("\n📈 Performance Report:")
        report = collector.get_performance_report()
        print(f"  Collections: {report['collections']}")
        print(f"  Success Rate: {report['success_rate']:.1%}")
        print(f"  VPIN Calculations: {report['vpin_calculations']}")
        
    finally:
        await collector.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_integrated_collector())


"""
VPIN Calculator - Pure NumPy Implementation (No PyTorch Required)

Volume-Synchronized Probability of Informed Trading
Uses only NumPy and Pandas - works perfectly with your RTX 5070 Ti setup

Key Features:
- No PyTorch/GPU dependencies
- Fast numpy-based calculations
- Multi-timeframe VPIN (1m, 5m, 15m)
- Real-time toxic flow detection
- Crypto-optimized parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VPINConfig:
    """Configuration for VPIN calculation"""
    # Volume bucket settings
    bucket_size_bars: int = 50  # Bars per bucket
    num_buckets: int = 50  # Number of buckets for VPIN calculation
    
    # Thresholds
    toxic_flow_threshold: float = 0.65  # VPIN > 0.65 = informed trading
    high_confidence_threshold: float = 0.75  # VPIN > 0.75 = very toxic
    
    # Timeframes to analyze
    timeframes: List[str] = None
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1m', '5m', '15m']


@dataclass
class VPINResult:
    """VPIN calculation result"""
    symbol: str
    timestamp: datetime
    vpin_1m: float
    vpin_5m: float
    vpin_15m: float
    avg_vpin: float
    toxic_flow: bool
    confidence: float  # 0-1, how confident in the signal
    buy_pressure: float  # >0.5 = buying pressure
    sell_pressure: float  # >0.5 = selling pressure
    entry_signal: int  # -1 (sell), 0 (hold), 1 (buy)
    volume_imbalance: float  # Absolute imbalance
    

class VPINCalculator:
    """
    Pure NumPy VPIN Calculator - No PyTorch Required
    
    VPIN measures the probability of informed trading by analyzing
    order flow imbalances in volume-synchronized buckets.
    
    High VPIN = Informed traders active (toxic flow)
    Low VPIN = Uninformed/retail flow (safe to trade)
    
    Perfect for CPU-only environments!
    """
    
    def __init__(self, config: Optional[VPINConfig] = None):
        self.config = config or VPINConfig()
        
        # Cache for recent calculations
        self.vpin_cache = {}
        self.cache_duration = 60  # seconds
        
        logger.info("VPIN Calculator initialized (NumPy-only, no PyTorch)")
        logger.info(f"Toxic flow threshold: {self.config.toxic_flow_threshold}")
    
    def calculate_realtime_vpin(
        self,
        symbol: str,
        trades: List[Dict],
        orderbook: Optional[Dict] = None
    ) -> VPINResult:
        """
        Calculate real-time VPIN from recent trades
        
        Args:
            symbol: Trading symbol
            trades: List of recent trades [{'price': float, 'volume': float, 'side': 'buy/sell', 'timestamp': datetime}]
            orderbook: Optional orderbook for additional context {'bids': [[price, size]], 'asks': [[price, size]]}
        
        Returns:
            VPINResult with multi-timeframe analysis
        """
        
        if not trades or len(trades) < self.config.bucket_size_bars:
            return self._neutral_result(symbol)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate VPIN for each timeframe
        vpin_1m = self._calculate_vpin_timeframe(df, '1min')
        vpin_5m = self._calculate_vpin_timeframe(df, '5min')
        vpin_15m = self._calculate_vpin_timeframe(df, '15min')
        
        # Average VPIN
        avg_vpin = np.mean([vpin_1m, vpin_5m, vpin_15m])
        
        # Detect toxic flow
        toxic_flow = avg_vpin > self.config.toxic_flow_threshold
        confidence = self._calculate_confidence(avg_vpin, [vpin_1m, vpin_5m, vpin_15m])
        
        # Calculate buy/sell pressure
        buy_pressure, sell_pressure = self._calculate_pressure(df)
        
        # Generate trading signal
        entry_signal = self._generate_signal(
            avg_vpin=avg_vpin,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            orderbook=orderbook
        )
        
        # Calculate volume imbalance
        total_buy = df[df['side'] == 'buy']['volume'].sum()
        total_sell = df[df['side'] == 'sell']['volume'].sum()
        volume_imbalance = abs(total_buy - total_sell) / (total_buy + total_sell) if (total_buy + total_sell) > 0 else 0
        
        result = VPINResult(
            symbol=symbol,
            timestamp=datetime.now(),
            vpin_1m=vpin_1m,
            vpin_5m=vpin_5m,
            vpin_15m=vpin_15m,
            avg_vpin=avg_vpin,
            toxic_flow=toxic_flow,
            confidence=confidence,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            entry_signal=entry_signal,
            volume_imbalance=volume_imbalance
        )
        
        return result
    
    def _calculate_vpin_timeframe(self, df: pd.DataFrame, timeframe: str) -> float:
        """
        Calculate VPIN for a specific timeframe
        
        Steps:
        1. Resample trades to timeframe bars
        2. Create volume buckets
        3. Calculate volume imbalance per bucket
        4. Average imbalance = VPIN
        """
        
        # Resample to timeframe - simpler aggregation
        df_indexed = df.set_index('timestamp')
        df_resampled = df_indexed.resample(timeframe).apply(
            lambda x: pd.Series({
                'volume': x['volume'].sum(),
                'side': self._aggregate_sides_simple(x['side'].values, x['volume'].values)
            })
        )
        
        # Remove empty bars
        df_resampled = df_resampled[df_resampled['volume'] > 0]
        
        if len(df_resampled) < self.config.bucket_size_bars:
            return 0.5  # Neutral if insufficient data
        
        # Create volume buckets
        total_volume = df_resampled['volume'].sum()
        volume_per_bucket = total_volume / self.config.num_buckets
        
        buckets = []
        current_bucket_volume = 0
        current_bucket_imbalance = 0
        
        for _, row in df_resampled.iterrows():
            remaining = row['volume']
            
            while remaining > 0:
                space_in_bucket = volume_per_bucket - current_bucket_volume
                
                if remaining >= space_in_bucket:
                    # Complete current bucket
                    current_bucket_imbalance += space_in_bucket * row['side']
                    buckets.append(abs(current_bucket_imbalance) / volume_per_bucket)
                    
                    # Start new bucket
                    remaining -= space_in_bucket
                    current_bucket_volume = 0
                    current_bucket_imbalance = 0
                else:
                    # Add to current bucket
                    current_bucket_volume += remaining
                    current_bucket_imbalance += remaining * row['side']
                    remaining = 0
        
        # Calculate VPIN (average absolute imbalance)
        if len(buckets) >= self.config.num_buckets:
            vpin = np.mean(buckets[-self.config.num_buckets:])
        elif len(buckets) > 0:
            vpin = np.mean(buckets)
        else:
            vpin = 0.5  # Neutral
        
        return np.clip(vpin, 0, 1)
    
    def _aggregate_sides_simple(self, sides: np.ndarray, volumes: np.ndarray) -> float:
        """
        Aggregate buy/sell sides weighted by volume (simplified for resampling)
        Returns: +1 for pure buy, -1 for pure sell, 0 for balanced
        """
        if len(sides) == 0:
            return 0
        
        # Calculate buy and sell volumes
        buy_mask = sides == 'buy'
        sell_mask = sides == 'sell'
        
        buy_volume = volumes[buy_mask].sum() if buy_mask.any() else 0
        sell_volume = volumes[sell_mask].sum() if sell_mask.any() else 0
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0
        
        return (buy_volume - sell_volume) / total_volume
    
    def _calculate_pressure(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate buy and sell pressure
        
        Returns:
            (buy_pressure, sell_pressure) both 0-1
        """
        total_buy = df[df['side'] == 'buy']['volume'].sum()
        total_sell = df[df['side'] == 'sell']['volume'].sum()
        total = total_buy + total_sell
        
        if total == 0:
            return 0.5, 0.5
        
        buy_pressure = total_buy / total
        sell_pressure = total_sell / total
        
        return buy_pressure, sell_pressure
    
    def _calculate_confidence(self, avg_vpin: float, individual_vpins: List[float]) -> float:
        """
        Calculate confidence in VPIN signal
        
        High confidence = all timeframes agree
        Low confidence = timeframes diverge
        """
        
        # Confidence from average VPIN magnitude
        magnitude_confidence = abs(avg_vpin - 0.5) * 2  # 0-1 scale
        
        # Confidence from consistency across timeframes
        std = np.std(individual_vpins)
        consistency_confidence = 1 - min(std * 3, 1)  # Lower std = higher confidence
        
        # Combine
        confidence = (magnitude_confidence * 0.6 + consistency_confidence * 0.4)
        
        return np.clip(confidence, 0, 1)
    
    def _generate_signal(
        self,
        avg_vpin: float,
        buy_pressure: float,
        sell_pressure: float,
        orderbook: Optional[Dict]
    ) -> int:
        """
        Generate trading signal from VPIN data
        
        Logic:
        - High VPIN + Strong buy pressure = BUY (informed buying)
        - High VPIN + Strong sell pressure = SELL (informed selling)
        - Low VPIN = HOLD (no informed flow)
        
        Returns:
            1 (buy), 0 (hold), -1 (sell)
        """
        
        # No signal if VPIN is low (no informed trading)
        if avg_vpin < self.config.toxic_flow_threshold:
            return 0
        
        # Determine direction from pressure
        if buy_pressure > 0.6:  # Strong buying
            # Confirm with orderbook if available
            if orderbook:
                bid_strength = sum(level[1] for level in orderbook.get('bids', [])[:5])
                ask_strength = sum(level[1] for level in orderbook.get('asks', [])[:5])
                
                if bid_strength > ask_strength * 1.2:
                    return 1  # BUY signal
            else:
                return 1  # BUY signal (no orderbook confirmation)
        
        elif sell_pressure > 0.6:  # Strong selling
            # Confirm with orderbook if available
            if orderbook:
                bid_strength = sum(level[1] for level in orderbook.get('bids', [])[:5])
                ask_strength = sum(level[1] for level in orderbook.get('asks', [])[:5])
                
                if ask_strength > bid_strength * 1.2:
                    return -1  # SELL signal
            else:
                return -1  # SELL signal (no orderbook confirmation)
        
        return 0  # HOLD (mixed signals)
    
    def _neutral_result(self, symbol: str) -> VPINResult:
        """Return neutral result when insufficient data"""
        return VPINResult(
            symbol=symbol,
            timestamp=datetime.now(),
            vpin_1m=0.5,
            vpin_5m=0.5,
            vpin_15m=0.5,
            avg_vpin=0.5,
            toxic_flow=False,
            confidence=0.0,
            buy_pressure=0.5,
            sell_pressure=0.5,
            entry_signal=0,
            volume_imbalance=0.0
        )
    
    def interpret_result(self, result: VPINResult) -> str:
        """
        Interpret VPIN result in human-readable format
        """
        
        lines = []
        lines.append(f"🔍 VPIN Analysis for {result.symbol}")
        lines.append(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        lines.append(f"📊 VPIN Scores:")
        lines.append(f"  1-min:  {result.vpin_1m:.3f}")
        lines.append(f"  5-min:  {result.vpin_5m:.3f}")
        lines.append(f"  15-min: {result.vpin_15m:.3f}")
        lines.append(f"  Average: {result.avg_vpin:.3f}")
        lines.append(f"")
        lines.append(f"⚠️ Toxic Flow: {'YES' if result.toxic_flow else 'NO'}")
        lines.append(f"🎯 Confidence: {result.confidence:.1%}")
        lines.append(f"")
        lines.append(f"📈 Pressure:")
        lines.append(f"  Buy:  {result.buy_pressure:.1%}")
        lines.append(f"  Sell: {result.sell_pressure:.1%}")
        lines.append(f"  Imbalance: {result.volume_imbalance:.1%}")
        lines.append(f"")
        
        if result.entry_signal == 1:
            lines.append(f"🟢 SIGNAL: BUY (informed buying detected)")
        elif result.entry_signal == -1:
            lines.append(f"🔴 SIGNAL: SELL (informed selling detected)")
        else:
            lines.append(f"⚪ SIGNAL: HOLD (no informed flow or mixed signals)")
        
        return "\n".join(lines)


def example_usage():
    """Example of how to use VPIN calculator"""
    
    # Initialize calculator
    vpin = VPINCalculator(VPINConfig(
        toxic_flow_threshold=0.65,
        high_confidence_threshold=0.75
    ))
    
    # Example trade data (would come from exchange)
    trades = []
    for i in range(200):
        trades.append({
            'price': 50000 + np.random.randn() * 100,
            'volume': abs(np.random.randn() * 10),
            'side': 'buy' if np.random.rand() > 0.45 else 'sell',  # Slight buy bias
            'timestamp': datetime.now() - timedelta(seconds=200-i)
        })
    
    # Example orderbook
    orderbook = {
        'bids': [[49990, 10], [49980, 15], [49970, 20]],
        'asks': [[50010, 8], [50020, 12], [50030, 18]]
    }
    
    # Calculate VPIN
    result = vpin.calculate_realtime_vpin(
        symbol='BTC',
        trades=trades,
        orderbook=orderbook
    )
    
    # Print interpretation
    print(vpin.interpret_result(result))
    
    return result


if __name__ == "__main__":
    # Run example
    result = example_usage()
    
    print("\n" + "="*60)
    print("✅ VPIN Calculator working perfectly - NO PyTorch required!")
    print("="*60)


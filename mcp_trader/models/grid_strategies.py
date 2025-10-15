"""
Grid trading strategies for perps.
Includes various grid spacing and sizing methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from ..models.base import BaseGridStrategy, TradingSignal


class LinearGridStrategy(BaseGridStrategy):
    """Linear grid with equal spacing."""
    
    def __init__(self, **kwargs):
        super().__init__("LinearGrid", **kwargs)
        self.grid_spacing = kwargs.get('grid_spacing', 0.01)  # 1%
        self.max_levels = kwargs.get('max_levels', 10)
    
    def calculate_grid_levels(self, current_price: float) -> List[float]:
        """Calculate linear grid levels."""
        levels = []
        
        # Buy levels (below current price)
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 - self.grid_spacing * i)
            levels.append(level)
        
        # Sell levels (above current price)
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 + self.grid_spacing * i)
            levels.append(level)
        
        return sorted(levels)


class FibonacciGridStrategy(BaseGridStrategy):
    """Fibonacci-based grid spacing."""
    
    def __init__(self, **kwargs):
        super().__init__("FibonacciGrid", **kwargs)
        self.fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0]
        self.max_levels = kwargs.get('max_levels', 9)
    
    def calculate_grid_levels(self, current_price: float) -> List[float]:
        """Calculate Fibonacci grid levels."""
        levels = []
        
        for ratio in self.fib_ratios[:self.max_levels]:
            # Buy levels
            buy_level = current_price * (1 - ratio * 0.01)  # Scale down
            levels.append(buy_level)
            
            # Sell levels
            sell_level = current_price * (1 + ratio * 0.01)
            levels.append(sell_level)
        
        return sorted(levels)


class VolatilityGridStrategy(BaseGridStrategy):
    """Volatility-adjusted grid spacing."""
    
    def __init__(self, **kwargs):
        super().__init__("VolatilityGrid", **kwargs)
        self.lookback_periods = kwargs.get('lookback_periods', 24)  # 24 hours
        self.vol_multiplier = kwargs.get('vol_multiplier', 1.0)
        self.max_levels = kwargs.get('max_levels', 10)
    
    def calculate_grid_levels(self, current_price: float, data: pd.DataFrame = None) -> List[float]:
        """Calculate volatility-adjusted grid levels."""
        if data is None or len(data) < self.lookback_periods:
            # Fallback to linear grid
            return LinearGridStrategy().calculate_grid_levels(current_price)
        
        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(self.lookback_periods)
        volatility = recent_returns.std() * np.sqrt(24)  # Daily volatility
        
        # Adjust grid spacing based on volatility
        base_spacing = 0.01  # 1%
        adjusted_spacing = base_spacing * (1 + volatility * self.vol_multiplier)
        
        levels = []
        
        # Buy levels
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 - adjusted_spacing * i)
            levels.append(level)
        
        # Sell levels
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 + adjusted_spacing * i)
            levels.append(level)
        
        return sorted(levels)


class AdaptiveGridStrategy(BaseGridStrategy):
    """Adaptive grid that adjusts based on market conditions."""
    
    def __init__(self, **kwargs):
        super().__init__("AdaptiveGrid", **kwargs)
        self.trend_threshold = kwargs.get('trend_threshold', 0.02)  # 2%
        self.vol_threshold = kwargs.get('vol_threshold', 0.05)  # 5%
        self.max_levels = kwargs.get('max_levels', 10)
    
    def calculate_grid_levels(self, current_price: float, data: pd.DataFrame = None) -> List[float]:
        """Calculate adaptive grid levels based on market conditions."""
        if data is None or len(data) < 24:
            return LinearGridStrategy().calculate_grid_levels(current_price)
        
        # Analyze market conditions
        recent_data = data.tail(24)
        returns = recent_data['close'].pct_change().dropna()
        
        # Calculate trend strength
        sma_short = recent_data['close'].rolling(8).mean().iloc[-1]
        sma_long = recent_data['close'].rolling(24).mean().iloc[-1]
        trend_strength = abs(sma_short - sma_long) / sma_long
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(24)
        
        # Adjust grid parameters based on conditions
        if trend_strength > self.trend_threshold:
            # Strong trend - wider spacing, fewer levels
            spacing = 0.02  # 2%
            levels_count = 5
        elif volatility > self.vol_threshold:
            # High volatility - wider spacing
            spacing = 0.015  # 1.5%
            levels_count = 8
        else:
            # Normal conditions - standard grid
            spacing = 0.01  # 1%
            levels_count = 10
        
        levels = []
        
        # Buy levels
        for i in range(1, levels_count + 1):
            level = current_price * (1 - spacing * i)
            levels.append(level)
        
        # Sell levels
        for i in range(1, levels_count + 1):
            level = current_price * (1 + spacing * i)
            levels.append(level)
        
        return sorted(levels)


class KellyGridStrategy(BaseGridStrategy):
    """Kelly Criterion-based position sizing for grid trading."""
    
    def __init__(self, **kwargs):
        super().__init__("KellyGrid", **kwargs)
        self.kelly_fraction = kwargs.get('kelly_fraction', 0.25)  # Conservative Kelly
        self.win_rate = kwargs.get('win_rate', 0.6)  # Expected win rate
        self.avg_win_loss_ratio = kwargs.get('avg_win_loss_ratio', 1.2)
        self.max_levels = kwargs.get('max_levels', 10)
        self.base_spacing = kwargs.get('base_spacing', 0.01)
    
    def calculate_position_size(self, level: float, current_price: float) -> float:
        """Calculate position size using Kelly Criterion."""
        # Calculate expected value for this grid level
        price_ratio = level / current_price
        if level < current_price:
            # Buy level - profit if price goes up
            expected_return = (current_price - level) / level
        else:
            # Sell level - profit if price goes down
            expected_return = (level - current_price) / current_price
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        b = expected_return
        p = self.win_rate
        q = 1 - p
        
        kelly_f = (b * p - q) / b if b > 0 else 0
        kelly_f = max(0, min(kelly_f, self.kelly_fraction))  # Cap at max Kelly
        
        return kelly_f
    
    def calculate_grid_levels(self, current_price: float) -> List[float]:
        """Calculate grid levels with Kelly-based sizing."""
        levels = []
        
        # Buy levels
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 - self.base_spacing * i)
            levels.append(level)
        
        # Sell levels
        for i in range(1, self.max_levels + 1):
            level = current_price * (1 + self.base_spacing * i)
            levels.append(level)
        
        return sorted(levels)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals with Kelly-based position sizing."""
        signals = []
        
        for idx, row in data.iterrows():
            current_price = row['close']
            
            if self.base_price is None:
                self.base_price = current_price
                self.grid_levels = self.calculate_grid_levels(current_price)
            
            # Check if price hit any grid level
            for level in self.grid_levels:
                if abs(current_price - level) / level < 0.001:  # Within 0.1%
                    signal = 1 if current_price < self.base_price else -1
                    position_size = self.calculate_position_size(level, current_price)
                    confidence = min(position_size * 2, 1.0)  # Scale confidence
                    
                    signals.append(TradingSignal(
                        timestamp=row['timestamp'],
                        symbol=row.get('symbol', 'UNKNOWN'),
                        signal=signal,
                        confidence=confidence,
                        price=current_price,
                        metadata={
                            'grid_level': level,
                            'base_price': self.base_price,
                            'position_size': position_size,
                            'kelly_fraction': self.kelly_fraction
                        }
                    ))
        
        return signals


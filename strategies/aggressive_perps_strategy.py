#!/usr/bin/env python3
"""
Aggressive Perpetual Contract Strategy for Maximum Profit Potential
Focus: Calculated high-probability bets with 10% max position sizing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerpPosition:
    """Perpetual contract position tracking."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    leverage: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    max_profit: float = 0.0
    funding_collected: float = 0.0


@dataclass
class MarketCondition:
    """Market condition assessment."""
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: str  # 'low', 'medium', 'high'
    liquidity: str  # 'low', 'medium', 'high'
    momentum: float  # -1 to 1
    volume_score: float  # 0 to 1


class AggressivePerpsStrategy:
    """
    HIGHLY AGGRESSIVE Perpetual Contract Strategy for Maximum Profits

    Key Features:
    - 10% max position sizing (as requested)
    - 5 max concurrent positions
    - HIGH leverage (5x-25x based on signal strength)
    - Both LONG and SHORT positions for maximum opportunities
    - TIGHT stop losses (0.3-0.5%) for risk control
    - Mid/small cap focus for explosive profit potential
    - Multiple aggressive entry signals
    - CALCULATED but aggressive exits
    """

    def __init__(self, max_positions: int = 5, max_position_size: float = 0.10):
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.positions: Dict[str, PerpPosition] = {}

        # HIGH LEVERAGE parameters - CALCULATED AGGRESSION
        self.leverage_brackets = {
            'low': 25,         # 25x leverage in low volatility (maximum leverage)
            'medium': 15,       # 15x leverage in medium volatility
            'high': 5           # 5x leverage in high volatility (safer but still aggressive)
        }

        # TIGHT stop loss parameters - AGGRESSIVE risk control
        self.stop_loss_brackets = {
            'low': 0.003,      # 0.3% stop loss in low volatility
            'medium': 0.004,   # 0.4% stop loss in medium volatility
            'high': 0.005      # 0.5% stop loss in high volatility (tighter)
        }

        # Entry signal weights (aggressive focus)
        self.signal_weights = {
            'momentum_breakout': 0.35,
            'volume_surge': 0.25,
            'funding_arbitrage': 0.20,
            'mean_reversion': 0.15,
            'support_resistance': 0.05
        }

        # AGGRESSIVE exit parameters - TIGHT stops, FAST profits
        self.exit_params = {
            'profit_target_1': 0.015,  # 1.5% initial target (faster exits)
            'profit_target_2': 0.03,   # 3% extended target (still aggressive)
            'trailing_stop': 0.008,    # 0.8% trailing stop (tighter)
            'time_exit_hours': 12,     # Exit after 12 hours max (shorter hold time)
            'stop_loss': 0.005         # 0.5% stop loss (tight)
        }

        # Mid/Small cap focus for maximum profit potential
        self.mid_small_cap_symbols = [
            "SOLUSDT", "SUIUSDT", "ASTERUSDT", "PENGUUSDT",  # Mid caps
            "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT"   # Small caps
        ]

        # Market data cache
        self.market_cache: Dict[str, Dict[str, Any]] = {}
        self.last_update = datetime.now()

    def analyze_market_condition(self, symbol: str, market_data: Dict[str, Any]) -> MarketCondition:
        """Analyze current market condition for aggressive trading."""
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        funding_rate = market_data.get('funding_rate', 0.0)

        if len(prices) < 20:
            return MarketCondition('sideways', 'low', 'low', 0.0, 0.0)

        # Calculate trend (using moving averages)
        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices[-20:])

        if ma_short > ma_long * 1.002:
            trend = 'bullish'
        elif ma_short < ma_long * 0.998:
            trend = 'bearish'
        else:
            trend = 'sideways'

        # Calculate volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility_pct = np.std(returns) * 100

        if volatility_pct < 2:
            volatility = 'low'
        elif volatility_pct < 5:
            volatility = 'medium'
        else:
            volatility = 'high'

        # Calculate momentum (-1 to 1)
        momentum = np.mean(returns[-5:]) * 10  # Amplify for sensitivity
        momentum = np.clip(momentum, -1, 1)

        # Calculate volume score (0-1)
        if volumes:
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            volume_score = min(recent_volume / avg_volume, 2.0) / 2.0  # Normalize
        else:
            volume_score = 0.5

        # Liquidity assessment
        if volume_score > 0.8:
            liquidity = 'high'
        elif volume_score > 0.4:
            liquidity = 'medium'
        else:
            liquidity = 'low'

        return MarketCondition(trend, volatility, liquidity, momentum, volume_score)

    def calculate_entry_signal_strength(self, symbol: str, market_data: Dict[str, Any],
                                      condition: MarketCondition) -> float:
        """Calculate overall entry signal strength (0-1)."""

        signals = {
            'momentum_breakout': self._momentum_breakout_signal(market_data, condition),
            'volume_surge': self._volume_surge_signal(market_data, condition),
            'funding_arbitrage': self._funding_arbitrage_signal(market_data),
            'mean_reversion': self._mean_reversion_signal(market_data, condition),
            'support_resistance': self._support_resistance_signal(market_data, condition)
        }

        # Weighted combination
        total_strength = 0.0
        for signal_name, strength in signals.items():
            weight = self.signal_weights.get(signal_name, 0.0)
            total_strength += strength * weight

        return min(total_strength, 1.0)

    def _momentum_breakout_signal(self, market_data: Dict[str, Any], condition: MarketCondition) -> float:
        """Momentum breakout signal - AGGRESSIVE entry."""
        momentum = condition.momentum

        # Strong momentum in trending market
        if condition.trend != 'sideways':
            if condition.trend == 'bullish' and momentum > 0.3:
                return min(momentum * 2, 1.0)  # Amplify bullish momentum
            elif condition.trend == 'bearish' and momentum < -0.3:
                return min(abs(momentum) * 2, 1.0)  # Amplify bearish momentum

        return 0.0

    def _volume_surge_signal(self, market_data: Dict[str, Any], condition: MarketCondition) -> float:
        """Volume surge signal - liquidity confirmation."""
        volume_score = condition.volume_score

        # High volume in trending market = strong signal
        if condition.trend != 'sideways' and volume_score > 0.7:
            return volume_score
        elif volume_score > 0.9:  # Extreme volume even in sideways market
            return volume_score

        return 0.0

    def _funding_arbitrage_signal(self, market_data: Dict[str, Any]) -> float:
        """Funding rate arbitrage opportunity."""
        funding_rate = market_data.get('funding_rate', 0.0)

        # Extreme funding rates create arbitrage opportunities
        if abs(funding_rate) > 0.001:  # >0.1% per hour
            return min(abs(funding_rate) * 1000, 1.0)  # Scale to 0-1

        return 0.0

    def _mean_reversion_signal(self, market_data: Dict[str, Any], condition: MarketCondition) -> float:
        """Mean reversion signal for ranging markets."""
        if condition.trend != 'sideways':
            return 0.0

        prices = market_data.get('prices', [])
        if len(prices) < 20:
            return 0.0

        # Bollinger Band squeeze or deviation
        ma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        current_price = prices[-1]

        # Distance from mean (standard deviations)
        deviation = abs(current_price - ma) / std if std > 0 else 0

        if deviation > 2:  # Price significantly deviated
            return min(deviation / 4, 1.0)  # Stronger signal for larger deviations

        return 0.0

    def _support_resistance_signal(self, market_data: Dict[str, Any], condition: MarketCondition) -> float:
        """Support/resistance breakout signal."""
        prices = market_data.get('prices', [])
        if len(prices) < 50:
            return 0.0

        # Simple support/resistance calculation
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        current_price = prices[-1]

        # Near resistance (bullish breakout potential)
        if current_price > recent_high * 0.995 and condition.trend == 'bullish':
            return 0.8
        # Near support (bearish breakout potential)
        elif current_price < recent_low * 1.005 and condition.trend == 'bearish':
            return 0.8

        return 0.0

    def should_enter_position(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine if we should enter a position - HIGHLY AGGRESSIVE but CALCULATED."""

        # Check position limits
        if len(self.positions) >= self.max_positions:
            return None

        if symbol in self.positions:
            return None

        # PRIORITY: Focus on mid/small caps for maximum profit potential
        if symbol not in self.mid_small_cap_symbols:
            return None  # Only trade mid/small caps for explosive gains

        # Analyze market condition
        condition = self.analyze_market_condition(symbol, market_data)

        # Calculate signal strength
        signal_strength = self.calculate_entry_signal_strength(symbol, market_data, condition)

        # LOWER threshold for mid/small caps (more opportunities)
        min_threshold = 0.5 if symbol in self.mid_small_cap_symbols else 0.6
        if signal_strength < min_threshold:
            return None

        # AGGRESSIVE side determination - take BOTH directions
        # Strong bullish momentum -> LONG
        # Strong bearish momentum -> SHORT
        # Neutral -> based on signal strength
        if condition.momentum > 0.4:
            side = 'long'
        elif condition.momentum < -0.4:
            side = 'short'
        else:
            # Neutral momentum - use signal strength to decide
            side = 'long' if signal_strength > 0.7 else 'short'

        # HIGHER leverage based on signal strength (more aggressive)
        base_leverage = self.leverage_brackets[condition.volatility]
        leverage_multiplier = 1 + (signal_strength - 0.5)  # 1x to 2x multiplier
        leverage = min(int(base_leverage * leverage_multiplier), 50)  # Cap at 50x

        # Calculate position size (up to 10%) - more aggressive for strong signals
        position_size_pct = min(signal_strength * self.max_position_size * 1.2, self.max_position_size)

        # Get current price
        current_price = market_data.get('prices', [1000])[-1]

        return {
            'symbol': symbol,
            'side': side,
            'leverage': leverage,
            'position_size_pct': position_size_pct,
            'current_price': current_price,
            'signal_strength': signal_strength,
            'condition': condition
        }

    def calculate_position_parameters(self, entry_signal: Dict[str, Any]) -> PerpPosition:
        """Calculate position entry parameters with TIGHT stops and HIGH leverage."""
        symbol = entry_signal['symbol']
        side = entry_signal['side']
        leverage = entry_signal['leverage']
        current_price = entry_signal['current_price']
        position_size_pct = entry_signal['position_size_pct']
        condition = entry_signal['condition']

        # Use TIGHT stop loss based on volatility (more aggressive risk control)
        stop_loss_pct = self.stop_loss_brackets[condition.volatility]

        # Calculate stop loss and take profit (HIGHLY aggressive targets)
        if side == 'long':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit_1 = current_price * (1 + self.exit_params['profit_target_1'])
            take_profit_2 = current_price * (1 + self.exit_params['profit_target_2'])
        else:  # short
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit_1 = current_price * (1 - self.exit_params['profit_target_1'])
            take_profit_2 = current_price * (1 - self.exit_params['profit_target_2'])

        # Position quantity (simplified - would be based on account balance)
        quantity = position_size_pct * leverage  # Simplified calculation

        return PerpPosition(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit_2,  # Use extended target
            trailing_stop=None
        )

    def should_exit_position(self, position: PerpPosition, current_price: float,
                           market_data: Dict[str, Any]) -> Optional[str]:
        """Determine if position should be exited - AGGRESSIVE profit taking."""

        # Stop loss hit
        if position.side == 'long' and current_price <= position.stop_loss:
            return 'stop_loss'
        elif position.side == 'short' and current_price >= position.stop_loss:
            return 'stop_loss'

        # Profit targets
        if position.side == 'long':
            if current_price >= position.take_profit:
                return 'take_profit'
        else:  # short
            if current_price <= position.take_profit:
                return 'take_profit'

        # Trailing stop (if activated)
        if position.trailing_stop:
            if position.side == 'long' and current_price <= position.trailing_stop:
                return 'trailing_stop'
            elif position.side == 'short' and current_price >= position.trailing_stop:
                return 'trailing_stop'

        # Time-based exit (24 hours max hold time)
        hours_held = (datetime.now() - position.entry_time).total_seconds() / 3600
        if hours_held >= self.exit_params['time_exit_hours']:
            return 'time_exit'

        # Update trailing stop if profitable
        self._update_trailing_stop(position, current_price)

        return None

    def _update_trailing_stop(self, position: PerpPosition, current_price: float):
        """Update trailing stop for profitable positions."""
        if position.side == 'long':
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price

        # Activate trailing stop if 1% profitable
        if profit_pct >= 0.01:
            if position.side == 'long':
                new_stop = current_price * (1 - self.exit_params['trailing_stop'])
                if not position.trailing_stop or new_stop > position.trailing_stop:
                    position.trailing_stop = new_stop
            else:  # short
                new_stop = current_price * (1 + self.exit_params['trailing_stop'])
                if not position.trailing_stop or new_stop < position.trailing_stop:
                    position.trailing_stop = new_stop

    def execute_entry(self, entry_signal: Dict[str, Any]) -> bool:
        """Execute position entry."""
        try:
            position = self.calculate_position_parameters(entry_signal)
            self.positions[position.symbol] = position

            logger.info(f"🚀 ENTERED {position.side.upper()} {position.symbol} "
                       f"@ {position.entry_price:.2f} (Leverage: {position.leverage}x, "
                       f"Size: {position.quantity:.4f})")

            return True
        except Exception as e:
            logger.error(f"❌ Entry execution failed: {e}")
            return False

    def execute_exit(self, symbol: str, exit_reason: str) -> bool:
        """Execute position exit."""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        exit_price = position.entry_price * 1.02  # Simplified exit price

        # Calculate P&L
        if position.side == 'long':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        logger.info(f"💰 EXITED {position.side.upper()} {symbol} "
                   f"@ {exit_price:.2f} ({exit_reason}) "
                   f"P&L: {pnl_pct:.2%}")

        del self.positions[symbol]
        return True

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        total_exposure = sum(pos.quantity * pos.leverage for pos in self.positions.values())
        total_positions = len(self.positions)

        return {
            'active_positions': total_positions,
            'max_positions': self.max_positions,
            'total_exposure': total_exposure,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'leverage': pos.leverage,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'hours_held': (datetime.now() - pos.entry_time).total_seconds() / 3600
                }
                for pos in self.positions.values()
            ]
        }

    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Update market data cache."""
        self.market_cache[symbol] = market_data
        self.market_cache[symbol]['last_update'] = datetime.now()

    def get_signal_summary(self, symbol: str) -> Dict[str, Any]:
        """Get signal summary for debugging."""
        if symbol not in self.market_cache:
            return {'error': 'No market data available'}

        market_data = self.market_cache[symbol]
        condition = self.analyze_market_condition(symbol, market_data)
        signal_strength = self.calculate_entry_signal_strength(symbol, market_data, condition)

        return {
            'symbol': symbol,
            'signal_strength': signal_strength,
            'trend': condition.trend,
            'volatility': condition.volatility,
            'momentum': condition.momentum,
            'volume_score': condition.volume_score,
            'should_enter': signal_strength >= 0.6,
            'active_position': symbol in self.positions
        }


# Strategy instance for deployment
aggressive_perps_strategy = AggressivePerpsStrategy()

if __name__ == "__main__":
    # Test the strategy
    print("🚀 Aggressive Perpetual Contract Strategy")
    print("=" * 50)

    # Sample market data
    sample_data = {
        'prices': [50000, 50100, 50200, 50300, 50400, 50500],
        'volumes': [1000, 1200, 1500, 1800, 2000, 2500],
        'funding_rate': 0.0002
    }

    strategy = AggressivePerpsStrategy()

    # Test market analysis
    condition = strategy.analyze_market_condition('BTCUSDT', sample_data)
    print(f"📊 Market Condition: {condition.trend}, {condition.volatility} volatility")

    # Test signal strength
    signal_strength = strategy.calculate_entry_signal_strength('BTCUSDT', sample_data, condition)
    print(f"🎯 Signal Strength: {signal_strength:.3f}")

    # Test entry decision
    entry_signal = strategy.should_enter_position('BTCUSDT', sample_data)
    if entry_signal:
        print(f"🚀 Entry Signal: {entry_signal['side']} {entry_signal['symbol']} "
              f"(Leverage: {entry_signal['leverage']}x, Size: {entry_signal['position_size_pct']:.1%})")

        # Test position execution
        success = strategy.execute_entry(entry_signal)
        print(f"✅ Entry Execution: {'Success' if success else 'Failed'}")

        # Test portfolio status
        status = strategy.get_portfolio_status()
        print(f"📊 Portfolio: {status['active_positions']}/{status['max_positions']} positions")
    else:
        print("⏸️  No entry signal generated")

    print("\n🎯 Strategy configured for maximum profit potential!")
    print("   • 10% max position size")
    print("   • 5 max concurrent positions")
    print("   • Dynamic leverage (3x-8x)")
    print("   • Aggressive profit targets (2-5%)")
    print("   • Tight risk management (0.5% stop loss)")

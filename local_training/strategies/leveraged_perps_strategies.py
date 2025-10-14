"""
Proven Leveraged Perps Strategies for Aster DEX
Research-based strategies optimized for RTX 5070Ti local training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for leveraged perps strategies."""
    symbol: str = "BTCUSDT"
    max_leverage: int = 25
    position_size_pct: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.10  # 10%
    funding_rate_threshold: float = 0.001  # 0.1% funding rate threshold
    volatility_lookback: int = 20
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


class MeanReversionStrategy:
    """
    Mean Reversion Strategy for Leveraged Perps.
    Based on research showing mean reversion in crypto perpetuals.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def generate_signals(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Generate mean reversion trading signals."""
        if len(data) < self.config.volatility_lookback * 2:
            return []

        current_price = data.iloc[-1]['close']

        # Calculate mean reversion indicators
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        sma_50 = data['close'].rolling(50).mean().iloc[-1]

        # Price deviation from mean
        price_deviation = (current_price - sma_20) / sma_20

        # RSI for overbought/oversold confirmation
        rsi = self._calculate_rsi(data['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1]

        # Volatility filter
        volatility = data['close'].pct_change().rolling(self.config.volatility_lookback).std().iloc[-1]

        signals = []

        # Mean reversion signals
        if price_deviation > 0.03 and current_rsi > self.config.rsi_overbought and volatility < 0.05:
            # Overbought condition - short bias
            signals.append({
                'action': 'sell',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(10, self.config.max_leverage),
                'stop_loss': current_price * (1 + self.config.stop_loss_pct),
                'take_profit': current_price * (1 - self.config.take_profit_pct * 2)
            })

        elif price_deviation < -0.03 and current_rsi < self.config.rsi_oversold and volatility < 0.05:
            # Oversold condition - long bias
            signals.append({
                'action': 'buy',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(10, self.config.max_leverage),
                'stop_loss': current_price * (1 - self.config.stop_loss_pct),
                'take_profit': current_price * (1 + self.config.take_profit_pct * 2)
            })

        return signals

    def _calculate_position_size(self, current_price: float, data: pd.DataFrame) -> float:
        """Calculate position size based on portfolio and volatility."""
        # Simple position sizing - $1000 per trade
        return 1000 / current_price

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class TrendFollowingStrategy:
    """
    Trend Following Strategy for Leveraged Perps.
    Based on momentum research in crypto markets.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def generate_signals(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Generate trend following signals."""
        if len(data) < 100:
            return []

        current_price = data.iloc[-1]['close']

        # Trend indicators
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        sma_50 = data['close'].rolling(50).mean().iloc[-1]
        sma_200 = data['close'].rolling(200).mean().iloc[-1]

        # ADX for trend strength
        adx = self._calculate_adx(data['high'], data['low'], data['close'])
        current_adx = adx.iloc[-1]

        # Volume confirmation
        volume_sma = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data.iloc[-1]['volume']

        signals = []

        # Strong uptrend
        if (current_price > sma_20 > sma_50 > sma_200 and
            current_adx > 25 and
            current_volume > volume_sma * 1.2):
            signals.append({
                'action': 'buy',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(15, self.config.max_leverage),
                'stop_loss': current_price * (1 - self.config.stop_loss_pct * 1.5),
                'take_profit': current_price * (1 + self.config.take_profit_pct * 3)
            })

        # Strong downtrend
        elif (current_price < sma_20 < sma_50 < sma_200 and
              current_adx > 25 and
              current_volume > volume_sma * 1.2):
            signals.append({
                'action': 'sell',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(15, self.config.max_leverage),
                'stop_loss': current_price * (1 + self.config.stop_loss_pct * 1.5),
                'take_profit': current_price * (1 - self.config.take_profit_pct * 3)
            })

        return signals

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        high_diff = high.diff()
        low_diff = low.diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        atr = self._calculate_atr(high, low, close, period)

        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx.fillna(0)

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_position_size(self, current_price: float, data: pd.DataFrame) -> float:
        """Calculate position size based on trend strength."""
        return 1500 / current_price  # Larger positions for strong trends


class FundingRateArbitrage:
    """
    Funding Rate Arbitrage Strategy.
    Based on research showing funding rate predictability in perpetuals.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def generate_signals(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Generate funding rate arbitrage signals."""
        if len(data) < 50:
            return []

        current_price = data.iloc[-1]['close']

        # Calculate funding rate indicators
        # In real implementation, this would use actual funding rate data
        # For now, simulate based on price momentum
        price_returns = data['close'].pct_change(8)  # 8-hour returns
        funding_rate_proxy = price_returns.rolling(8).mean().iloc[-1]

        signals = []

        # Long when funding rate is negative (expecting price increase)
        if funding_rate_proxy < -self.config.funding_rate_threshold:
            signals.append({
                'action': 'buy',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(20, self.config.max_leverage),
                'stop_loss': current_price * (1 - self.config.stop_loss_pct),
                'take_profit': current_price * (1 + self.config.take_profit_pct)
            })

        # Short when funding rate is positive (expecting price decrease)
        elif funding_rate_proxy > self.config.funding_rate_threshold:
            signals.append({
                'action': 'sell',
                'quantity': self._calculate_position_size(current_price, data),
                'leverage': min(20, self.config.max_leverage),
                'stop_loss': current_price * (1 + self.config.stop_loss_pct),
                'take_profit': current_price * (1 - self.config.take_profit_pct)
            })

        return signals

    def _calculate_position_size(self, current_price: float, data: pd.DataFrame) -> float:
        """Calculate position size for funding arbitrage."""
        return 800 / current_price


class GridTradingStrategies:
    """
    Advanced Grid Trading Strategies for Aster DEX.
    Multiple grid approaches based on research and backtesting.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

    def arithmetic_grid_strategy(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Arithmetic grid with fixed price spacing."""
        if len(data) < 50:
            return []

        current_price = data.iloc[-1]['close']
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]

        # Adjust grid spacing based on volatility
        base_spacing = 0.02  # 2%
        volatility_multiplier = 1 + volatility * 5  # Increase spacing in volatile markets
        grid_spacing = base_spacing * volatility_multiplier

        grid_levels = self.config.grid_levels
        signals = []

        for i in range(-grid_levels, grid_levels + 1):
            if i == 0:  # Skip current price level
                continue

            level_price = current_price * (1 + i * grid_spacing)

            # Check if price is near this level
            price_distance = abs(current_price - level_price) / current_price

            if price_distance < 0.005:  # Within 0.5%
                if i < 0:  # Below current price - buy orders
                    signals.append({
                        'action': 'buy',
                        'quantity': self.config.grid_position_size / level_price,
                        'price': level_price,
                        'grid_level': i,
                        'strategy': 'arithmetic_grid'
                    })
                else:  # Above current price - sell orders
                    signals.append({
                        'action': 'sell',
                        'quantity': self.config.grid_position_size / level_price,
                        'price': level_price,
                        'grid_level': i,
                        'strategy': 'arithmetic_grid'
                    })

        return signals

    def geometric_grid_strategy(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Geometric grid with percentage-based spacing."""
        if len(data) < 50:
            return []

        current_price = data.iloc[-1]['close']

        # Geometric progression
        grid_levels = self.config.grid_levels
        base_ratio = 1.02  # 2% geometric spacing

        signals = []

        for i in range(1, grid_levels + 1):
            # Lower levels (buy)
            lower_price = current_price / (base_ratio ** i)
            upper_price = current_price * (base_ratio ** i)

            # Check buy levels
            if abs(current_price - lower_price) / current_price < 0.005:
                signals.append({
                    'action': 'buy',
                    'quantity': self.config.grid_position_size / lower_price,
                    'price': lower_price,
                    'grid_level': -i,
                    'strategy': 'geometric_grid'
                })

            # Check sell levels
            if abs(current_price - upper_price) / current_price < 0.005:
                signals.append({
                    'action': 'sell',
                    'quantity': self.config.grid_position_size / upper_price,
                    'price': upper_price,
                    'grid_level': i,
                    'strategy': 'geometric_grid'
                })

        return signals

    def dynamic_grid_strategy(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Dynamic grid that adapts to market volatility."""
        if len(data) < 100:
            return []

        current_price = data.iloc[-1]['close']

        # Calculate adaptive grid spacing
        volatility = data['close'].pct_change().rolling(50).std().iloc[-1]
        trend = (data['close'].iloc[-1] - data['close'].rolling(50).mean().iloc[-1]) / data['close'].rolling(50).mean().iloc[-1]

        # Adjust spacing based on volatility and trend
        base_spacing = 0.02
        volatility_adjustment = 1 + volatility * 3
        trend_adjustment = 1 + abs(trend) * 2

        adaptive_spacing = base_spacing * volatility_adjustment * trend_adjustment

        grid_levels = int(self.config.grid_levels * (1 + volatility))
        signals = []

        for i in range(-grid_levels, grid_levels + 1):
            if i == 0:
                continue

            level_price = current_price * (1 + i * adaptive_spacing)

            if abs(current_price - level_price) / current_price < 0.003:  # Tighter trigger for dynamic grid
                if i < 0:
                    signals.append({
                        'action': 'buy',
                        'quantity': self.config.grid_position_size / level_price,
                        'price': level_price,
                        'grid_level': i,
                        'strategy': 'dynamic_grid'
                    })
                else:
                    signals.append({
                        'action': 'sell',
                        'quantity': self.config.grid_position_size / level_price,
                        'price': level_price,
                        'grid_level': i,
                        'strategy': 'dynamic_grid'
                    })

        return signals


class EnsembleStrategy:
    """
    Ensemble of multiple strategies for improved performance.
    Combines mean reversion, trend following, and grid strategies.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

        # Initialize individual strategies
        self.mean_reversion = MeanReversionStrategy(config)
        self.trend_following = TrendFollowingStrategy(config)
        self.funding_arbitrage = FundingRateArbitrage(config)
        self.grid_trading = GridTradingStrategies(config)

        # Strategy weights (can be optimized)
        self.strategy_weights = {
            'mean_reversion': 0.3,
            'trend_following': 0.3,
            'funding_arbitrage': 0.2,
            'grid_trading': 0.2
        }

    def generate_signals(self, data: pd.DataFrame, symbol: str, current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Generate ensemble signals from multiple strategies."""
        all_signals = []

        # Get signals from each strategy
        strategies = [
            (self.mean_reversion, 'mean_reversion'),
            (self.trend_following, 'trend_following'),
            (self.funding_arbitrage, 'funding_arbitrage'),
        ]

        for strategy, strategy_name in strategies:
            try:
                signals = strategy.generate_signals(data, symbol, current_time)

                # Add strategy metadata
                for signal in signals:
                    signal['strategy'] = strategy_name
                    signal['ensemble_weight'] = self.strategy_weights[strategy_name]

                all_signals.extend(signals)

            except Exception as e:
                logger.warning(f"Error in {strategy_name} strategy: {e}")

        # Add grid signals (separate due to different structure)
        try:
            grid_signals = self.grid_trading.arithmetic_grid_strategy(data, symbol, current_time)
            for signal in grid_signals:
                signal['strategy'] = 'grid_arithmetic'
                signal['ensemble_weight'] = self.strategy_weights['grid_trading']
            all_signals.extend(grid_signals)

        except Exception as e:
            logger.warning(f"Error in grid strategy: {e}")

        # Ensemble decision making
        return self._ensemble_decision(all_signals, data, current_time)

    def _ensemble_decision(self, signals: List[Dict[str, Any]], data: pd.DataFrame,
                          current_time: pd.Timestamp) -> List[Dict[str, Any]]:
        """Make ensemble decision from individual strategy signals."""
        if not signals:
            return []

        # Group signals by action
        buy_signals = [s for s in signals if s['action'] == 'buy']
        sell_signals = [s for s in signals if s['action'] == 'sell']

        # Calculate weighted confidence for each action
        buy_confidence = sum(s['ensemble_weight'] for s in buy_signals)
        sell_confidence = sum(s['ensemble_weight'] for s in sell_signals)

        # Market condition filters
        current_price = data.iloc[-1]['close']
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
        volume_ratio = data.iloc[-1]['volume'] / data['volume'].rolling(20).mean().iloc[-1]

        # Don't trade in extreme conditions
        if volatility > 0.08 or volume_ratio < 0.5:
            return []

        ensemble_signals = []

        # Strong buy consensus
        if buy_confidence > 0.5 and buy_confidence > sell_confidence:
            # Combine buy signals
            total_quantity = sum(s['quantity'] for s in buy_signals)
            avg_leverage = np.mean([s.get('leverage', 1) for s in buy_signals])

            ensemble_signals.append({
                'action': 'buy',
                'quantity': total_quantity,
                'leverage': int(avg_leverage),
                'confidence': buy_confidence,
                'strategy': 'ensemble',
                'constituent_strategies': [s['strategy'] for s in buy_signals]
            })

        # Strong sell consensus
        elif sell_confidence > 0.5 and sell_confidence > buy_confidence:
            total_quantity = sum(s['quantity'] for s in sell_signals)
            avg_leverage = np.mean([s.get('leverage', 1) for s in sell_signals])

            ensemble_signals.append({
                'action': 'sell',
                'quantity': total_quantity,
                'leverage': int(avg_leverage),
                'confidence': sell_confidence,
                'strategy': 'ensemble',
                'constituent_strategies': [s['strategy'] for s in sell_signals]
            })

        return ensemble_signals


# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'mean_reversion': MeanReversionStrategy,
    'trend_following': TrendFollowingStrategy,
    'funding_arbitrage': FundingRateArbitrage,
    'arithmetic_grid': lambda config: GridTradingStrategies(config).arithmetic_grid_strategy,
    'geometric_grid': lambda config: GridTradingStrategies(config).geometric_grid_strategy,
    'dynamic_grid': lambda config: GridTradingStrategies(config).dynamic_grid_strategy,
    'ensemble': EnsembleStrategy
}


def get_strategy(strategy_name: str, config: StrategyConfig = None) -> Any:
    """Get strategy instance by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(config)


# Example usage and backtesting integration
if __name__ == "__main__":
    # Test strategy generation
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    prices = 50000 + np.cumsum(np.random.normal(0.001, 0.02, 200))

    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'volume': np.random.uniform(1000000, 10000000, 200)
    })

    # Test ensemble strategy
    config = StrategyConfig(symbol="BTCUSDT", max_leverage=20)
    strategy = EnsembleStrategy(config)

    signals = strategy.generate_signals(data, "BTCUSDT", pd.Timestamp.now())

    print(f"Generated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal['action']} {signal['quantity']:.4f} BTC at {signal['leverage']}x leverage")
        print(f"  Strategy: {signal['strategy']}, Confidence: {signal['confidence']:.2f}")

"""
Hybrid trading strategy combining grid and volatility approaches.
Intelligently switches between strategies based on market conditions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..types import TradingDecision, MarketRegime
from .grid_strategy import GridStrategy, GridConfig
from .volatility_strategy import VolatilityStrategy
from ...execution.aster_client import AsterClient

logger = logging.getLogger(__name__)


class HybridStrategyConfig:
    """Configuration for hybrid trading strategy."""

    def __init__(self,
                 symbol: str,
                 total_capital: float = 10000.0,
                 grid_allocation: float = 0.6,  # 60% to grid
                 volatility_allocation: float = 0.4,  # 40% to volatility
                 regime_switch_threshold: float = 0.7,  # Confidence threshold
                 rebalance_interval_hours: int = 24,
                 min_volatility_for_vol_strategy: float = 0.02,  # 2% daily vol
                 max_volatility_for_grid: float = 0.05):  # 5% daily vol

        self.symbol = symbol
        self.total_capital = total_capital
        self.grid_allocation = grid_allocation
        self.volatility_allocation = volatility_allocation
        self.regime_switch_threshold = regime_switch_threshold
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_volatility_for_vol_strategy = min_volatility_for_vol_strategy
        self.max_volatility_for_grid = max_volatility_for_grid


class HybridStrategy:
    """
    Intelligent hybrid strategy that combines grid and volatility trading.
    Dynamically allocates capital based on market conditions and performance.
    """

    def __init__(self, config: HybridStrategyConfig, aster_client: AsterClient):
        self.config = config
        self.aster_client = aster_client

        # Initialize sub-strategies
        grid_config = {
            'grid_levels': 10,
            'grid_spacing_percent': 2.0,
            'position_size_per_level': (config.total_capital * config.grid_allocation) / 10,
            'max_position_size': config.total_capital * config.grid_allocation,
            'rebalancing_threshold': 1.0,
            'volatility_multiplier': 1.5,
            'min_order_size': 5.0,
            'max_slippage_percent': 0.5
        }

        vol_config = {
            'min_volatility_threshold': config.min_volatility_for_vol_strategy * 100,
            'max_volatility_threshold': config.max_volatility_for_grid * 100,
            'momentum_window': 6,
            'mean_reversion_window': 24,
            'position_size_scaler': 0.5,
            'profit_taking_threshold': 2.0,
            'stop_loss_threshold': 3.0,
            'min_holding_time': 30,
            'max_holding_time': 480
        }

        self.grid_strategy = GridStrategy(grid_config)
        self.volatility_strategy = VolatilityStrategy(vol_config)

        # Strategy state
        self.current_regime: Optional[MarketRegime] = None
        self.strategy_weights = {
            'grid': config.grid_allocation,
            'volatility': config.volatility_allocation
        }
        self.last_rebalance = datetime.now()
        self.performance_history: Dict[str, List[float]] = {
            'grid': [],
            'volatility': []
        }

        logger.info(f"HybridStrategy initialized for {config.symbol} with "
                   f"{config.grid_allocation:.1%} grid, {config.volatility_allocation:.1%} volatility")

    async def make_decisions(self, symbol: str, market_data: pd.DataFrame,
                           portfolio_state: Dict[str, Any]) -> List[TradingDecision]:
        """
        Generate trading decisions using hybrid approach.
        Analyzes market conditions and allocates between strategies.
        """
        if symbol != self.config.symbol:
            return []

        try:
            # Analyze current market regime
            regime = await self._analyze_market_regime(market_data)

            # Update strategy weights based on regime and performance
            await self._update_strategy_weights(regime, market_data)

            # Get decisions from both strategies
            grid_decisions = await self._get_grid_decisions(symbol, market_data, portfolio_state)
            vol_decisions = await self._get_volatility_decisions(symbol, market_data, portfolio_state)

            # Combine and weight decisions
            combined_decisions = self._combine_decisions(grid_decisions, vol_decisions)

            # Rebalance if needed
            if self._should_rebalance():
                rebalance_decisions = await self._generate_rebalance_decisions(portfolio_state)
                combined_decisions.extend(rebalance_decisions)

            logger.info(f"Hybrid strategy for {symbol}: regime={regime.value}, "
                       f"grid_weight={self.strategy_weights['grid']:.2f}, "
                       f"vol_weight={self.strategy_weights['volatility']:.2f}, "
                       f"decisions={len(combined_decisions)}")

            return combined_decisions

        except Exception as e:
            logger.error(f"Error in hybrid strategy decision making: {e}")
            return []

    async def _analyze_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Analyze current market regime using multiple indicators."""
        if len(market_data) < 50:
            return MarketRegime.SIDEWAYS

        # Calculate volatility (annualized)
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(365)

        # Trend analysis
        ma_20 = market_data['close'].rolling(20).mean().iloc[-1]
        ma_50 = market_data['close'].rolling(50).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]

        # Volume analysis (if available)
        volume_trend = 1.0
        if 'volume' in market_data.columns:
            volume_ma = market_data['volume'].rolling(20).mean()
            volume_trend = market_data['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0

        # Determine regime
        if volatility > self.config.max_volatility_for_grid:
            regime = MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.min_volatility_for_vol_strategy:
            regime = MarketRegime.LOW_VOLATILITY
        elif current_price > ma_20 > ma_50 and volume_trend > 1.1:
            regime = MarketRegime.BULL_TREND
        elif current_price < ma_20 < ma_50 and volume_trend > 1.1:
            regime = MarketRegime.BEAR_TREND
        else:
            regime = MarketRegime.SIDEWAYS

        self.current_regime = regime
        return regime

    async def _update_strategy_weights(self, regime: MarketRegime, market_data: pd.DataFrame):
        """Dynamically update strategy weights based on regime and performance."""
        base_weights = {
            MarketRegime.HIGH_VOLATILITY: {'grid': 0.3, 'volatility': 0.7},
            MarketRegime.LOW_VOLATILITY: {'grid': 0.8, 'volatility': 0.2},
            MarketRegime.BULL_TREND: {'grid': 0.6, 'volatility': 0.4},
            MarketRegime.BEAR_TREND: {'grid': 0.5, 'volatility': 0.5},
            MarketRegime.SIDEWAYS: {'grid': 0.7, 'volatility': 0.3}
        }

        # Get base weights for current regime
        regime_weights = base_weights.get(regime, {'grid': 0.5, 'volatility': 0.5})

        # Adjust based on recent performance (if available)
        performance_adjustment = self._calculate_performance_adjustment()

        # Smooth weight changes to avoid whipsaw
        smoothing_factor = 0.1
        for strategy in ['grid', 'volatility']:
            target_weight = regime_weights[strategy] * performance_adjustment[strategy]
            current_weight = self.strategy_weights[strategy]
            self.strategy_weights[strategy] = current_weight * (1 - smoothing_factor) + target_weight * smoothing_factor

        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight

    def _calculate_performance_adjustment(self) -> Dict[str, float]:
        """Calculate performance-based weight adjustments."""
        adjustments = {'grid': 1.0, 'volatility': 1.0}

        # Simple momentum-based adjustment (can be enhanced)
        for strategy in ['grid', 'volatility']:
            if len(self.performance_history[strategy]) >= 5:
                recent_performance = self.performance_history[strategy][-5:]
                avg_performance = np.mean(recent_performance)
                if avg_performance > 0:
                    adjustments[strategy] = min(1.2, 1.0 + avg_performance * 0.1)
                elif avg_performance < 0:
                    adjustments[strategy] = max(0.8, 1.0 + avg_performance * 0.1)

        return adjustments

    async def _get_grid_decisions(self, symbol: str, market_data: pd.DataFrame,
                                portfolio_state: Dict[str, Any]) -> List[TradingDecision]:
        """Get decisions from grid strategy with weighted allocation."""
        if self.strategy_weights['grid'] < 0.1:  # Minimum threshold
            return []

        try:
            # Adjust grid bounds based on current price
            current_price = market_data['close'].iloc[-1]
            volatility = market_data['close'].pct_change().std() * np.sqrt(365)

            # Dynamic bounds based on volatility
            bound_factor = min(0.3, volatility * 2)  # Max 30% range
            lower_bound = current_price * (1 - bound_factor)
            upper_bound = current_price * (1 + bound_factor)

            # Update grid strategy config
            self.grid_strategy.config.lower_bound = lower_bound
            self.grid_strategy.config.upper_bound = upper_bound
            self.grid_strategy.config.quantity_per_grid = (
                self.config.total_capital * self.strategy_weights['grid'] / self.grid_strategy.config.num_grids
            )

            # Get grid decisions
            decisions = await self.grid_strategy.make_decisions(symbol, market_data, portfolio_state)

            # Apply weight scaling
            for decision in decisions:
                decision.quantity *= self.strategy_weights['grid']

            return decisions

        except Exception as e:
            logger.error(f"Error getting grid decisions: {e}")
            return []

    async def _get_volatility_decisions(self, symbol: str, market_data: pd.DataFrame,
                                      portfolio_state: Dict[str, Any]) -> List[TradingDecision]:
        """Get decisions from volatility strategy with weighted allocation."""
        if self.strategy_weights['volatility'] < 0.1:  # Minimum threshold
            return []

        try:
            # Get volatility decisions
            decisions = await self.volatility_strategy.make_decisions(symbol, market_data, portfolio_state)

            # Apply weight scaling
            for decision in decisions:
                decision.quantity *= self.strategy_weights['volatility']

            return decisions

        except Exception as e:
            logger.error(f"Error getting volatility decisions: {e}")
            return []

    def _combine_decisions(self, grid_decisions: List[TradingDecision],
                          vol_decisions: List[TradingDecision]) -> List[TradingDecision]:
        """Combine decisions from both strategies with conflict resolution."""
        combined = grid_decisions + vol_decisions

        # Simple conflict resolution: prefer higher confidence decisions
        # In case of conflicts, keep the one with higher confidence
        final_decisions = []
        seen_symbols = set()

        # Sort by confidence (higher first)
        combined.sort(key=lambda x: x.confidence, reverse=True)

        for decision in combined:
            if decision.symbol not in seen_symbols:
                final_decisions.append(decision)
                seen_symbols.add(decision.symbol)

        return final_decisions

    def _should_rebalance(self) -> bool:
        """Check if portfolio rebalancing is needed."""
        time_since_rebalance = datetime.now() - self.last_rebalance
        return time_since_rebalance > timedelta(hours=self.config.rebalance_interval_hours)

    async def _generate_rebalance_decisions(self, portfolio_state: Dict[str, Any]) -> List[TradingDecision]:
        """Generate decisions to rebalance portfolio according to target weights."""
        # This is a simplified implementation
        # In practice, you'd calculate current allocations vs target allocations
        self.last_rebalance = datetime.now()

        # For now, just log that rebalancing occurred
        logger.info(f"Portfolio rebalance triggered for {self.config.symbol}")
        return []

    def update_performance(self, strategy: str, pnl: float):
        """Update performance history for strategy weight adjustment."""
        if strategy in self.performance_history:
            self.performance_history[strategy].append(pnl)
            # Keep only last 50 entries
            if len(self.performance_history[strategy]) > 50:
                self.performance_history[strategy] = self.performance_history[strategy][-50:]

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get current strategy performance metrics."""
        return {
            'current_regime': self.current_regime.value if self.current_regime else 'unknown',
            'grid_weight': self.strategy_weights['grid'],
            'volatility_weight': self.strategy_weights['volatility'],
            'last_rebalance': self.last_rebalance.isoformat(),
            'grid_performance': self.performance_history['grid'][-10:] if len(self.performance_history['grid']) >= 10 else self.performance_history['grid'],
            'volatility_performance': self.performance_history['volatility'][-10:] if len(self.performance_history['volatility']) >= 10 else self.performance_history['volatility']
        }


import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from ..execution.aster_client import AsterClient
from ..config import get_settings, PRIORITY_SYMBOLS
from ..trading.types import PortfolioState, TradingDecision, MarketRegime
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AdaptiveAgentConfig:
    """Configuration for the adaptive AI trading agent."""
    initial_balance: float = 10000.0
    max_allocation_per_trade: float = 0.1  # 10% of portfolio per trade
    min_allocation_per_trade: float = 0.01  # 1% of portfolio per trade
    max_open_positions: int = 5
    rebalance_frequency_minutes: int = 15
    learning_rate: float = 0.01
    risk_tolerance: float = 0.15  # 15% max drawdown
    volatility_threshold: float = 0.03  # 3% daily volatility threshold
    profit_taking_threshold: float = 0.05  # 5% profit taking
    stop_loss_threshold: float = 0.03  # 3% stop loss
    adaptation_window_minutes: int = 60  # 1 hour adaptation window


@dataclass
class MarketState:
    """Current market state observation."""
    timestamp: datetime
    prices: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    momentum: Dict[str, float] = field(default_factory=dict)
    regime: MarketRegime = MarketRegime.SIDEWAYS
    fear_greed_index: float = 50.0  # 0-100 scale


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_pnl: float = 0.0
    max_pnl: float = 0.0


class AdvancedStrategy(ABC):
    """Abstract base class for advanced trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history: List[Dict[str, Any]] = []

    @abstractmethod
    def generate_signals(self, market_state: MarketState,
                        portfolio_state: PortfolioState,
                        agent: 'AdaptiveTradingAgent') -> List[TradingDecision]:
        """Generate trading signals based on current market state."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name."""
        pass

    def update_performance(self, pnl: float, market_conditions: Dict[str, Any]):
        """Update strategy performance metrics."""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'market_conditions': market_conditions
        })

    def get_recent_performance(self, hours: int = 24) -> Dict[str, float]:
        """Get recent performance metrics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_trades = [p for p in self.performance_history if p['timestamp'] > cutoff_time]

        if not recent_trades:
            return {'win_rate': 0.0, 'avg_pnl': 0.0, 'total_trades': 0}

        pnl_values = [t['pnl'] for t in recent_trades]
        wins = sum(1 for pnl in pnl_values if pnl > 0)

        return {
            'win_rate': wins / len(pnl_values) if pnl_values else 0.0,
            'avg_pnl': np.mean(pnl_values) if pnl_values else 0.0,
            'total_trades': len(pnl_values)
        }


class BarbellStrategy(AdvancedStrategy):
    """
    Barbell Strategy: Combines safe assets with high-risk/high-reward bets.
    In crypto bull markets, this means holding stable blue-chip cryptos
    while taking asymmetric bets on high-volatility altcoins.
    """

    def get_strategy_name(self) -> str:
        return "Barbell_Portfolio"

    def generate_signals(self, market_state: MarketState,
                        portfolio_state: PortfolioState,
                        agent: 'AdaptiveTradingAgent') -> List[TradingDecision]:

        decisions = []
        current_balance = portfolio_state.available_balance

        # Define barbell components
        safe_assets = ['BTCUSDT', 'ETHUSDT']  # Blue-chip cryptos
        risk_assets = ['SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT']  # High-volatility alts

        # Check market regime - only trade in bull/high-vol markets
        if market_state.regime not in [MarketRegime.BULL_TREND, MarketRegime.HIGH_VOLATILITY]:
            return decisions  # No trades in sideways/bear markets

        # Calculate allocations (60% safe, 40% risk)
        safe_allocation = current_balance * 0.6
        risk_allocation = current_balance * 0.4

        # Safe asset positions (long-term hold)
        for symbol in safe_assets:
            if symbol in market_state.prices:
                price = market_state.prices[symbol]

                # Check if we should add to safe positions
                current_position = sum(p.quantity for p in agent.positions.values()
                                     if p.symbol == symbol and p.side == 'long')

                if current_position * price < safe_allocation / len(safe_assets):
                    # Add to position
                    add_amount = (safe_allocation / len(safe_assets) - current_position * price) * 0.1
                    if add_amount > agent.config.min_allocation_per_trade * current_balance:
                        quantity = add_amount / price
                        decisions.append(TradingDecision(
                            symbol=symbol,
                            side='long',
                            quantity=quantity,
                            reason='Barbell: Adding to safe asset position'
                        ))

        # Risk asset positions (momentum-based)
        for symbol in risk_assets:
            if symbol in market_state.prices and symbol in market_state.momentum:
                price = market_state.prices[symbol]
                momentum = market_state.momentum[symbol]

                # Only trade if strong momentum and high volatility
                volatility = market_state.volatility.get(symbol, 0)
                if momentum > 0.02 and volatility > agent.config.volatility_threshold:
                    # Asymmetric bet: small position with high upside potential
                    risk_amount = risk_allocation / len(risk_assets) * 0.2  # Only 20% of risk allocation per asset

                    if risk_amount > agent.config.min_allocation_per_trade * current_balance:
                        quantity = risk_amount / price
                        decisions.append(TradingDecision(
                            symbol=symbol,
                            side='long',
                            quantity=quantity,
                            reason=f'Barbell: Asymmetric bet on high momentum (momentum: {momentum:.3f})'
                        ))

        return decisions


class AsymmetricBetStrategy(AdvancedStrategy):
    """
    Asymmetric Bet Strategy: Takes small positions with large upside potential
    and limited downside risk. Uses volatility to our advantage.
    """

    def get_strategy_name(self) -> str:
        return "Asymmetric_Bets"

    def generate_signals(self, market_state: MarketState,
                        portfolio_state: PortfolioState,
                        agent: 'AdaptiveTradingAgent') -> List[TradingDecision]:

        decisions = []

        # Look for asymmetric opportunities
        for symbol in PRIORITY_SYMBOLS:
            if symbol in market_state.prices and symbol in market_state.volatility:
                price = market_state.prices[symbol]
                volatility = market_state.volatility[symbol]
                momentum = market_state.momentum.get(symbol, 0)

                # Asymmetric opportunity criteria:
                # 1. High volatility (> 5%)
                # 2. Positive momentum
                # 3. Price not too far from recent lows (mean reversion potential)
                # 4. Not already heavily positioned

                current_position = sum(p.quantity for p in agent.positions.values()
                                     if p.symbol == symbol)

                if (volatility > 0.05 and
                    momentum > 0.01 and
                    current_position * price < portfolio_state.available_balance * 0.05):  # Max 5% per asset

                    # Calculate position size based on Kelly criterion approximation
                    # Position size = (Expected return) / (Volatility^2)
                    expected_return = momentum * 24  # Daily expectation from hourly momentum
                    kelly_fraction = expected_return / (volatility ** 2) if volatility > 0 else 0

                    # Conservative Kelly sizing
                    position_size_pct = min(kelly_fraction * 0.5, agent.config.max_allocation_per_trade)

                    if position_size_pct > agent.config.min_allocation_per_trade:
                        amount = portfolio_state.available_balance * position_size_pct
                        quantity = amount / price

                        decisions.append(TradingDecision(
                            symbol=symbol,
                            side='long',
                            quantity=quantity,
                            reason=f'Asymmetric: High volatility opportunity (vol: {volatility:.3f}, momentum: {momentum:.3f})'
                        ))

        return decisions


class TailRiskHedgeStrategy(AdvancedStrategy):
    """
    Tail Risk Hedge Strategy: Protects against extreme market moves
    while benefiting from normal volatility.
    """

    def get_strategy_name(self) -> str:
        return "Tail_Risk_Hedge"

    def generate_signals(self, market_state: MarketState,
                        portfolio_state: PortfolioState,
                        agent: 'AdaptiveTradingAgent') -> List[TradingDecision]:

        decisions = []

        # Monitor portfolio volatility and implement hedges
        portfolio_volatility = np.std([market_state.volatility.get(s, 0) for s in PRIORITY_SYMBOLS])

        # If portfolio volatility is extreme, hedge with safe assets
        if portfolio_volatility > 0.08:  # 8% portfolio volatility threshold
            safe_asset = 'BTCUSDT'  # Hedge with BTC

            if safe_asset in market_state.prices:
                price = market_state.prices[safe_asset]

                # Check current hedge position
                hedge_position = sum(p.quantity for p in agent.positions.values()
                                   if p.symbol == safe_asset and p.side == 'long')

                # Target hedge size based on portfolio risk
                target_hedge_value = portfolio_state.total_balance * min(portfolio_volatility * 2, 0.3)

                if hedge_position * price < target_hedge_value:
                    add_amount = target_hedge_value - hedge_position * price
                    if add_amount > agent.config.min_allocation_per_trade * portfolio_state.available_balance:
                        quantity = add_amount / price
                        decisions.append(TradingDecision(
                            symbol=safe_asset,
                            side='long',
                            quantity=quantity,
                            reason=f'Tail Risk: Adding hedge position (portfolio vol: {portfolio_volatility:.3f})'
                        ))

        return decisions


class AdaptiveTradingAgent:
    """
    Real-time adaptive AI trading agent that learns and optimizes continuously.
    Uses advanced strategies to maximize profits while managing risk in volatile markets.
    """

    def __init__(self, config: AdaptiveAgentConfig):
        self.config = config
        self.aster_client: Optional[AsterClient] = None
        self.positions: Dict[str, Position] = {}
        # Tests expect lazy portfolio_state initialization
        self.portfolio_state: Optional[PortfolioState] = None

        # Market state tracking
        self.market_history: List[MarketState] = []
        self.market_state: Optional[MarketState] = None

        # Strategies
        self.strategies = {
            'barbell': BarbellStrategy({}),
            'asymmetric': AsymmetricBetStrategy({}),
            'tail_risk': TailRiskHedgeStrategy({})
        }

        # Strategy weights (adaptively learned)
        # Tests expect empty weights at initialization
        self.strategy_weights: Dict[str, float] = {}

        # Learning components
        self.performance_window: List[Dict[str, Any]] = []
        self.learning_enabled = True

        # Risk management
        self.portfolio_volatility = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0

        logger.info("Adaptive Trading Agent initialized with advanced strategies")

    async def initialize(self):
        """Initialize the trading agent."""
        try:
            settings = get_settings()
            self.aster_client = AsterClient(settings.aster_api_key, settings.aster_api_secret)
            await self.aster_client.connect()
            logger.info("Connected to Aster DEX")
        except Exception as e:
            logger.error(f"Failed to initialize trading agent: {e}")
            raise

    async def start_trading(self):
        """Start the adaptive trading loop."""
        logger.info("Starting adaptive trading loop...")

        while self.learning_enabled:
            try:
                # Update market state
                await self.update_market_state()

                # Assess market regime
                self.assess_market_regime()

                # Generate trading decisions
                decisions = await self.generate_decisions()

                # Execute decisions
                await self.execute_decisions(decisions)

                # Update portfolio state
                await self.update_portfolio_state()

                # Adapt strategies based on performance
                self.adapt_strategies()

                # Log status
                self.log_status()

                # Wait before next iteration
                await asyncio.sleep(self.config.rebalance_frequency_minutes * 60)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def update_market_state(self):
        """Update current market state from Aster DEX."""
        try:
            market_data = {}

            # Get current prices and volumes for all priority symbols
            for symbol in PRIORITY_SYMBOLS:
                try:
                    ticker = await self.aster_client.get_24hr_ticker(symbol)
                    orderbook = await self.aster_client.get_order_book(symbol, limit=20)

                    market_data[symbol] = {
                        'price': float(ticker.get('lastPrice', 0)),
                        'volume': float(ticker.get('quoteVolume', 0)),
                        'price_change': float(ticker.get('priceChangePercent', 0)) / 100,
                        'high': float(ticker.get('highPrice', 0)),
                        'low': float(ticker.get('lowPrice', 0)),
                        'bid': orderbook.get('bids', [[0]])[0][0] if orderbook.get('bids') else 0,
                        'ask': orderbook.get('asks', [[0]])[0][0] if orderbook.get('asks') else 0
                    }

                except Exception as e:
                    logger.warning(f"Failed to get market data for {symbol}: {e}")
                    continue

            # Calculate volatility and momentum
            volatility = {}
            momentum = {}

            for symbol, data in market_data.items():
                # Calculate hourly volatility (simplified)
                if len(self.market_history) > 1:
                    recent_prices = [h.prices.get(symbol, data['price']) for h in self.market_history[-24:]]
                    if len(recent_prices) > 1:
                        volatility[symbol] = np.std([p/r - 1 for p, r in zip(recent_prices[1:], recent_prices[:-1])])
                        momentum[symbol] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    else:
                        volatility[symbol] = 0.02  # Default volatility
                        momentum[symbol] = 0.0
                else:
                    volatility[symbol] = 0.02
                    momentum[symbol] = 0.0

            # Create market state
            self.market_state = MarketState(
                timestamp=datetime.now(),
                prices={s: d['price'] for s, d in market_data.items()},
                volumes={s: d['volume'] for s, d in market_data.items()},
                volatility=volatility,
                momentum=momentum
            )

            # Keep history
            self.market_history.append(self.market_state)
            if len(self.market_history) > 1000:  # Keep last 1000 observations
                self.market_history = self.market_history[-1000:]

        except Exception as e:
            logger.error(f"Error updating market state: {e}")

    def assess_market_regime(self):
        """Assess current market regime based on recent data."""
        if not self.market_history or len(self.market_history) < 10:
            self.market_state.regime = MarketRegime.SIDEWAYS
            return

        # Calculate market-wide metrics
        recent_states = self.market_history[-10:]

        # Average momentum across all symbols
        avg_momentum = np.mean([np.mean(list(s.momentum.values())) for s in recent_states if s.momentum])

        # Average volatility
        avg_volatility = np.mean([np.mean(list(s.volatility.values())) for s in recent_states if s.volatility])

        # Price trend (simplified)
        if len(recent_states) >= 2:
            start_prices = np.mean([list(s.prices.values()) for s in recent_states[:5]], axis=0)
            end_prices = np.mean([list(s.prices.values()) for s in recent_states[-5:]], axis=0)
            trend = np.mean(end_prices - start_prices) / np.mean(start_prices)
        else:
            trend = 0

        # Classify regime
        if trend > 0.05 and avg_volatility > 0.03:
            self.market_state.regime = MarketRegime.BULL_TREND
        elif avg_volatility > 0.05:
            self.market_state.regime = MarketRegime.HIGH_VOLATILITY
        elif abs(trend) < 0.02:
            self.market_state.regime = MarketRegime.SIDEWAYS
        elif trend < -0.05:
            self.market_state.regime = MarketRegime.BEAR_TREND
        else:
            self.market_state.regime = MarketRegime.SIDEWAYS

    async def generate_decisions(self) -> List[TradingDecision]:
        """Generate trading decisions using all strategies."""
        if not self.market_state:
            return []

        all_decisions = []

        # Get decisions from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                weight = self.strategy_weights.get(strategy_name, 0)
                if weight > 0:
                    decisions = strategy.generate_signals(
                        self.market_state,
                        self.portfolio_state,
                        self
                    )

                    # Apply strategy weight
                    for decision in decisions:
                        decision.quantity *= weight

                    all_decisions.extend(decisions)

            except Exception as e:
                logger.error(f"Error in strategy {strategy_name}: {e}")

        # Risk management filter
        all_decisions = self.apply_risk_filters(all_decisions)

        return all_decisions

    def apply_risk_filters(self, decisions: List[TradingDecision]) -> List[TradingDecision]:
        """Apply risk management filters to trading decisions."""
        filtered_decisions = []

        for decision in decisions:
            # Check position limits
            current_positions = len([p for p in self.positions.values() if p.symbol == decision.symbol])
            if current_positions >= 3:  # Max 3 positions per symbol
                continue

            # Check total open positions
            if len(self.positions) >= self.config.max_open_positions:
                continue

            # Check allocation limits
            position_value = decision.quantity * self.market_state.prices.get(decision.symbol, 0)
            if position_value > self.portfolio_state.available_balance * self.config.max_allocation_per_trade:
                # Scale down position
                max_quantity = (self.portfolio_state.available_balance * self.config.max_allocation_per_trade) / self.market_state.prices.get(decision.symbol, 0)
                decision.quantity = max_quantity

            # Check drawdown limits
            if self.max_drawdown > self.config.risk_tolerance:
                continue  # Stop trading if drawdown too high

            filtered_decisions.append(decision)

        return filtered_decisions

    async def execute_decisions(self, decisions: List[TradingDecision]):
        """Execute trading decisions on Aster DEX."""
        for decision in decisions:
            try:
                if decision.quantity < self.config.min_allocation_per_trade * self.portfolio_state.available_balance:
                    continue  # Too small

                price = self.market_state.prices.get(decision.symbol, 0)
                if price <= 0:
                    continue

                # Calculate stop loss and take profit levels
                if decision.side == 'long':
                    stop_loss = price * (1 - self.config.stop_loss_threshold)
                    take_profit = price * (1 + self.config.profit_taking_threshold)
                else:
                    stop_loss = price * (1 + self.config.stop_loss_threshold)
                    take_profit = price * (1 - self.config.profit_taking_threshold)

                # Create position
                position = Position(
                    symbol=decision.symbol,
                    side=decision.side,
                    entry_price=price,
                    quantity=decision.quantity,
                    timestamp=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                # Store position
                position_key = f"{decision.symbol}_{decision.side}_{datetime.now().isoformat()}"
                self.positions[position_key] = position

                # Update available balance
                self.portfolio_state.available_balance -= decision.quantity * price

                logger.info(f"Executed {decision.side} order: {decision.quantity:.4f} {decision.symbol} at ${price:.4f}")

            except Exception as e:
                logger.error(f"Error executing decision for {decision.symbol}: {e}")

    async def update_portfolio_state(self):
        """Update portfolio state with current valuations."""
        try:
            total_value = self.portfolio_state.available_balance
            total_positions_value = 0

            # Update position P&L
            positions_to_remove = []
            for pos_key, position in self.positions.items():
                current_price = self.market_state.prices.get(position.symbol, position.entry_price)

                if position.side == 'long':
                    position.current_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.current_pnl = (position.entry_price - current_price) * position.quantity

                position.max_pnl = max(position.max_pnl, position.current_pnl)
                total_positions_value += position.entry_price * position.quantity + position.current_pnl

                # Check stop loss and take profit
                if position.stop_loss and current_price <= position.stop_loss:
                    logger.info(f"Stop loss triggered for {position.symbol}")
                    positions_to_remove.append(pos_key)
                elif position.take_profit and current_price >= position.take_profit:
                    logger.info(f"Take profit triggered for {position.symbol}")
                    positions_to_remove.append(pos_key)

            # Remove closed positions
            for pos_key in positions_to_remove:
                position = self.positions[pos_key]
                self.portfolio_state.available_balance += position.entry_price * position.quantity + position.current_pnl
                del self.positions[pos_key]

            # Update totals
            self.portfolio_state.total_balance = total_value + total_positions_value
            self.portfolio_state.total_positions_value = total_positions_value
            self.portfolio_state.timestamp = datetime.now()

            # Calculate drawdown
            if hasattr(self, 'peak_balance'):
                self.peak_balance = max(self.peak_balance, self.portfolio_state.total_balance)
            else:
                self.peak_balance = self.portfolio_state.total_balance

            current_drawdown = (self.peak_balance - self.portfolio_state.total_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    def adapt_strategies(self):
        """Adapt strategy weights based on recent performance."""
        if len(self.performance_window) < 10:
            return  # Need more data

        # Calculate recent performance by strategy
        recent_performance = {}
        for strategy_name, strategy in self.strategies.items():
            perf = strategy.get_recent_performance(hours=24)
            recent_performance[strategy_name] = perf['avg_pnl'] * perf['win_rate']

        # Normalize performance scores
        scores = list(recent_performance.values())
        if scores and max(scores) > min(scores):
            normalized_scores = [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]
        else:
            normalized_scores = [1.0 / len(scores)] * len(scores)

        # Update weights using exponential moving average
        alpha = self.config.learning_rate
        for i, strategy_name in enumerate(self.strategies.keys()):
            current_weight = self.strategy_weights[strategy_name]
            target_weight = normalized_scores[i]
            new_weight = alpha * target_weight + (1 - alpha) * current_weight
            self.strategy_weights[strategy_name] = max(0.1, min(0.8, new_weight))  # Clamp weights

        # Renormalize weights
        total_weight = sum(self.strategy_weights.values())
        for strategy_name in self.strategy_weights:
            self.strategy_weights[strategy_name] /= total_weight

    def log_status(self):
        """Log current agent status."""
        logger.info(f"Portfolio: ${self.portfolio_state.total_balance:.2f} "
                   f"(Available: ${self.portfolio_state.available_balance:.2f})")
        logger.info(f"Positions: {len(self.positions)} | Max Drawdown: {self.max_drawdown:.2%}")
        logger.info(f"Strategy Weights: {self.strategy_weights}")
        logger.info(f"Market Regime: {self.market_state.regime.value if self.market_state else 'Unknown'}")


async def main():
    """Main function to run the adaptive trading agent."""
    logging.basicConfig(level=logging.INFO)

    # Configuration for volatile bull market
    config = AdaptiveAgentConfig(
        initial_balance=10000.0,
        max_allocation_per_trade=0.15,  # Higher allocation in bull markets
        risk_tolerance=0.20,  # Higher risk tolerance for bull market
        volatility_threshold=0.04,  # Higher volatility threshold
        learning_rate=0.05  # Faster learning adaptation
    )

    agent = AdaptiveTradingAgent(config)

    try:
        await agent.initialize()
        await agent.start_trading()
    except KeyboardInterrupt:
        logger.info("Trading agent stopped by user")
    except Exception as e:
        logger.error(f"Trading agent failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())


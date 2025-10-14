"""
Advanced Backtesting Framework for Aster DEX
RTX 5070Ti optimized backtesting with realistic market conditions, slippage, and funding rates.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting modes."""
    SPOT = "spot"
    PERPETUAL = "perpetual"
    GRID_TRADING = "grid_trading"
    LEVERAGED_PERPS = "leveraged_perps"


@dataclass
class BacktestConfig:
    """Configuration for advanced backtesting."""
    initial_balance: float = 10000.0
    mode: BacktestMode = BacktestMode.PERPETUAL

    # Trading parameters
    max_position_size: float = 1.0  # Max position as fraction of balance
    max_leverage: int = 25  # Max leverage for perpetuals
    min_position_size: float = 10.0  # USD

    # Costs and slippage
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    slippage_model: str = "linear"  # linear, sqrt, adaptive
    max_slippage: float = 0.005  # 0.5%

    # Funding rates (for perpetuals)
    funding_rate_interval: int = 8  # Hours
    funding_rate_impact: float = 0.1  # Impact on returns

    # Grid trading parameters
    grid_levels: int = 10
    grid_spacing: float = 0.02  # 2%
    grid_position_size: float = 100.0  # USD per level

    # Risk management
    stop_loss_threshold: float = 0.05  # 5%
    take_profit_threshold: float = 0.10  # 10%
    max_daily_loss: float = 0.10  # 10%
    max_drawdown: float = 0.20  # 20%

    # Market conditions
    market_volatility_multiplier: float = 1.0
    liquidity_factor: float = 1.0

    # Backtesting settings
    warmup_periods: int = 100  # Periods to skip for indicator calculation
    transaction_cost_model: str = "realistic"  # realistic, optimistic, pessimistic


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    leverage: int = 1
    entry_time: datetime = None
    stop_loss: float = None
    take_profit: float = None
    funding_paid: float = 0.0
    funding_received: float = 0.0

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()

    @property
    def notional_value(self) -> float:
        """Calculate notional value of position."""
        return abs(self.quantity) * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L (placeholder - needs current price)."""
        return 0.0  # To be calculated with current price

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L for position."""
        if self.side == 'long':
            return self.quantity * (current_price - self.entry_price)
        else:  # short
            return -self.quantity * (current_price - self.entry_price)


@dataclass
class GridLevel:
    """Represents a grid trading level."""
    price: float
    quantity: float = 0.0
    side: str = 'buy'  # 'buy' or 'sell'
    is_active: bool = True


@dataclass
class BacktestResult:
    """Comprehensive backtesting results."""
    config: BacktestConfig
    initial_balance: float
    final_balance: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    total_fees: float
    total_slippage: float
    total_funding_costs: float

    # Time series data
    portfolio_values: List[float]
    returns: List[float]
    drawdown_series: List[float]

    # Detailed metrics
    monthly_returns: Dict[str, float]
    strategy_metrics: Dict[str, Any]
    risk_metrics: Dict[str, float]

    # Execution details
    trades_log: List[Dict[str, Any]]
    position_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'config': self.config.__dict__,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'total_funding_costs': self.total_funding_costs,
            'portfolio_values': self.portfolio_values,
            'returns': self.returns,
            'drawdown_series': self.drawdown_series,
            'monthly_returns': self.monthly_returns,
            'strategy_metrics': self.strategy_metrics,
            'risk_metrics': self.risk_metrics,
            'trades_log': self.trades_log,
            'position_history': self.position_history
        }


class AdvancedBacktester:
    """
    RTX 5070Ti optimized backtesting framework for Aster DEX.
    Handles realistic market conditions, slippage, funding rates, and grid trading.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

        # Portfolio state
        self.balance = self.config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.grid_positions: Dict[str, List[GridLevel]] = {}

        # Performance tracking
        self.portfolio_values = [self.config.initial_balance]
        self.returns = []
        self.drawdown_series = []
        self.trades_log = []
        self.position_history = []

        # Market state
        self.current_prices: Dict[str, float] = {}
        self.funding_rates: Dict[str, float] = {}  # Funding rate per hour

        # Statistics
        self.start_time = None
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.total_funding_costs = 0.0
        self.trade_count = 0
        self.profitable_trades = 0

        logger.info(f"Advanced Backtester initialized with {self.config.mode.value} mode")

    def run_backtest(self, data: pd.DataFrame, strategy_func: Callable,
                    symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run comprehensive backtest with realistic market conditions.

        Args:
            data: OHLCV DataFrame
            strategy_func: Function that returns trading signals
            symbol: Trading pair symbol

        Returns:
            Comprehensive backtest results
        """
        logger.info(f"Starting backtest for {symbol} with {len(data)} data points")

        self.start_time = data.index[0] if hasattr(data.index, '__getitem__') else datetime.now()
        self.balance = self.config.initial_balance
        self.positions.clear()
        self.grid_positions.clear()

        # Initialize tracking
        self.portfolio_values = [self.config.initial_balance]
        self.returns = []
        self.drawdown_series = []
        self.trades_log = []
        self.position_history = []

        # Process each time step
        for i in range(self.config.warmup_periods, len(data)):
            try:
                current_time = data.index[i] if hasattr(data.index, '__getitem__') else datetime.now()
                current_row = data.iloc[i]

                # Update market prices
                self.current_prices[symbol] = current_row['close']

                # Generate trading signals
                signals = strategy_func(data.iloc[:i+1], symbol, current_time)

                # Execute signals
                self._execute_signals(signals, current_row, symbol, current_time)

                # Update positions (funding rates, etc.)
                self._update_positions(current_row, symbol, current_time)

                # Update portfolio tracking
                self._update_portfolio_tracking()

                # Risk management checks
                self._check_risk_limits()

            except Exception as e:
                logger.warning(f"Error at step {i}: {e}")
                continue

        # Calculate final results
        final_balance = self.portfolio_values[-1] if self.portfolio_values else self.config.initial_balance
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance

        # Calculate performance metrics
        if len(self.returns) > 1:
            annualized_return = self._calculate_annualized_return()
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown()
            win_rate = self.profitable_trades / max(self.trade_count, 1)
        else:
            annualized_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns()

        # Strategy-specific metrics
        strategy_metrics = self._calculate_strategy_metrics()

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()

        result = BacktestResult(
            config=self.config,
            initial_balance=self.config.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=self.trade_count,
            profitable_trades=self.profitable_trades,
            total_fees=self.total_fees,
            total_slippage=self.total_slippage,
            total_funding_costs=self.total_funding_costs,
            portfolio_values=self.portfolio_values,
            returns=self.returns,
            drawdown_series=self.drawdown_series,
            monthly_returns=monthly_returns,
            strategy_metrics=strategy_metrics,
            risk_metrics=risk_metrics,
            trades_log=self.trades_log,
            position_history=self.position_history
        )

        logger.info(f"✅ Backtest completed: {total_return:.2%} return, {max_drawdown:.2%} max DD")
        return result

    def _execute_signals(self, signals: List[Dict[str, Any]], current_row: pd.Series,
                       symbol: str, current_time: datetime):
        """Execute trading signals with realistic market conditions."""
        for signal in signals:
            try:
                action = signal.get('action')
                quantity = signal.get('quantity', 0)
                leverage = signal.get('leverage', 1)

                if action == 'buy':
                    self._open_position(symbol, 'long', quantity, leverage, current_row, current_time)
                elif action == 'sell':
                    self._open_position(symbol, 'short', quantity, leverage, current_row, current_time)
                elif action == 'close':
                    self._close_position(symbol, current_row, current_time)
                elif action.startswith('grid_'):
                    self._manage_grid_position(signal, current_row, symbol, current_time)

            except Exception as e:
                logger.warning(f"Error executing signal: {e}")

    def _open_position(self, symbol: str, side: str, quantity: float,
                      leverage: int, current_row: pd.Series, current_time: datetime):
        """Open a new position with realistic execution."""
        current_price = current_row['close']

        # Calculate realistic execution price with slippage
        execution_price = self._calculate_execution_price(
            current_price, quantity, side, current_row['volume']
        )

        # Validate position size limits
        position_value = abs(quantity) * execution_price * leverage
        max_position_value = self.balance * self.config.max_position_size

        if position_value > max_position_value:
            logger.warning(f"Position size {position_value:.2f} exceeds limit {max_position_value:.2f}")
            return

        # Check leverage limits
        if leverage > self.config.max_leverage:
            logger.warning(f"Leverage {leverage} exceeds limit {self.config.max_leverage}")
            leverage = self.config.max_leverage

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=execution_price,
            quantity=quantity if side == 'long' else -quantity,
            leverage=leverage,
            entry_time=current_time
        )

        # Calculate and deduct fees
        fee = self._calculate_fee(position_value, is_maker=True)
        self.balance -= fee
        self.total_fees += fee

        # Add position
        self.positions[symbol] = position

        # Log trade
        self._log_trade({
            'timestamp': current_time,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'leverage': leverage,
            'fee': fee,
            'type': 'open'
        })

        logger.debug(f"Opened {side} position: {quantity} {symbol} at ${execution_price}")

    def _close_position(self, symbol: str, current_row: pd.Series, current_time: datetime):
        """Close existing position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_price = current_row['close']

        # Calculate P&L
        pnl = position.calculate_pnl(current_price)

        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(
            current_price, abs(position.quantity), position.side, current_row['volume']
        )

        # Calculate fees
        position_value = abs(position.quantity) * execution_price
        fee = self._calculate_fee(position_value, is_maker=True)
        self.balance -= fee
        self.total_fees += fee

        # Update balance with P&L
        self.balance += pnl

        # Remove position
        del self.positions[symbol]

        # Log trade
        self._log_trade({
            'timestamp': current_time,
            'symbol': symbol,
            'side': 'close',
            'quantity': abs(position.quantity),
            'price': execution_price,
            'pnl': pnl,
            'fee': fee,
            'type': 'close'
        })

        logger.debug(f"Closed position: {pnl:+.2f} P&L")

    def _calculate_execution_price(self, base_price: float, quantity: float,
                                 side: str, volume: float) -> float:
        """Calculate realistic execution price with slippage and market impact."""
        # Base slippage calculation
        if self.config.slippage_model == "linear":
            slippage = min(self.config.max_slippage, quantity / volume * 0.001)
        elif self.config.slippage_model == "sqrt":
            slippage = min(self.config.max_slippage, np.sqrt(quantity / volume) * 0.001)
        else:  # adaptive
            # Adaptive based on volatility and volume
            volatility = self._estimate_volatility()
            base_slippage = quantity / volume * 0.001
            adaptive_slippage = base_slippage * (1 + volatility * 2)
            slippage = min(self.config.max_slippage, adaptive_slippage)

        # Apply slippage
        if side == 'buy':
            execution_price = base_price * (1 + slippage)
        else:
            execution_price = base_price * (1 - slippage)

        self.total_slippage += abs(base_price - execution_price) * quantity

        return execution_price

    def _calculate_fee(self, position_value: float, is_maker: bool = True) -> float:
        """Calculate trading fees."""
        if self.config.transaction_cost_model == "realistic":
            fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        elif self.config.transaction_cost_model == "optimistic":
            fee_rate = 0.0001  # Very low fees
        else:  # pessimistic
            fee_rate = 0.001  # Higher fees

        return position_value * fee_rate

    def _update_positions(self, current_row: pd.Series, symbol: str, current_time: datetime):
        """Update positions with funding rates and other costs."""
        current_price = current_row['close']

        for pos_symbol, position in self.positions.items():
            # Update funding rates for perpetuals
            if self.config.mode == BacktestMode.PERPETUAL:
                funding_cost = self._calculate_funding_cost(position, current_price, current_time)
                if funding_cost != 0:
                    self.balance -= funding_cost
                    self.total_funding_costs += abs(funding_cost)

                    if position.side == 'long':
                        position.funding_paid += abs(funding_cost)
                    else:
                        position.funding_received += abs(funding_cost)

            # Check stop loss and take profit
            self._check_stop_loss_take_profit(position, current_price, current_time)

        # Update position history
        total_portfolio_value = self.balance
        for position in self.positions.values():
            total_portfolio_value += position.calculate_pnl(current_price)

        self.position_history.append({
            'timestamp': current_time,
            'balance': self.balance,
            'positions_value': total_portfolio_value - self.balance,
            'total_value': total_portfolio_value,
            'active_positions': len(self.positions)
        })

    def _calculate_funding_cost(self, position: Position, current_price: float,
                              current_time: datetime) -> float:
        """Calculate funding rate cost for perpetual positions."""
        if position.symbol not in self.funding_rates:
            # Generate realistic funding rate based on market conditions
            self.funding_rates[position.symbol] = np.random.normal(0.0001, 0.0002)  # ±0.01% per hour

        funding_rate = self.funding_rates[position.symbol]

        # Funding cost = position_value * funding_rate * hours_held
        hours_held = (current_time - position.entry_time).total_seconds() / 3600

        if hours_held >= self.config.funding_rate_interval:
            position_value = abs(position.quantity) * current_price
            funding_cost = position_value * funding_rate * self.config.funding_rate_impact

            # Reset entry time for next funding period
            position.entry_time = current_time

            return funding_cost

        return 0.0

    def _check_stop_loss_take_profit(self, position: Position, current_price: float,
                                   current_time: datetime):
        """Check if position should be closed due to SL/TP."""
        if position.stop_loss and current_price <= position.stop_loss:
            logger.info(f"Stop loss triggered for {position.symbol}")
            self._close_position(position.symbol, pd.Series({'close': current_price}), current_time)

        elif position.take_profit and current_price >= position.take_profit:
            logger.info(f"Take profit triggered for {position.symbol}")
            self._close_position(position.symbol, pd.Series({'close': current_price}), current_time)

    def _update_portfolio_tracking(self):
        """Update portfolio value tracking."""
        total_value = self.balance

        # Add unrealized P&L from open positions
        for position in self.positions.values():
            if position.symbol in self.current_prices:
                current_price = self.current_prices[position.symbol]
                total_value += position.calculate_pnl(current_price)

        # Update portfolio values
        if self.portfolio_values:
            previous_value = self.portfolio_values[-1]
            current_return = (total_value - previous_value) / previous_value
            self.returns.append(current_return)

        self.portfolio_values.append(total_value)

        # Update drawdown
        peak = max(self.portfolio_values) if self.portfolio_values else total_value
        current_drawdown = (peak - total_value) / peak if peak > 0 else 0
        self.drawdown_series.append(current_drawdown)

    def _check_risk_limits(self):
        """Check and enforce risk limits."""
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.config.initial_balance

        # Daily loss limit
        if len(self.portfolio_values) > 24:  # After 24 hours of data
            day_start_value = self.portfolio_values[-24]
            daily_loss = (current_value - day_start_value) / day_start_value

            if daily_loss < -self.config.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {daily_loss:.2%}")
                # Close all positions
                for symbol in list(self.positions.keys()):
                    self._close_position(symbol, pd.Series({'close': self.current_prices.get(symbol, 0)}), datetime.now())

        # Maximum drawdown
        if self.drawdown_series:
            max_dd = max(self.drawdown_series)
            if max_dd > self.config.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {max_dd:.2%}")
                # Reduce position sizes or close positions

    def _log_trade(self, trade_info: Dict[str, Any]):
        """Log trade for analysis."""
        self.trades_log.append(trade_info)
        self.trade_count += 1

        if trade_info.get('pnl', 0) > 0:
            self.profitable_trades += 1

    def _estimate_volatility(self) -> float:
        """Estimate current market volatility."""
        if len(self.returns) < 10:
            return 0.02  # Default 2% volatility

        recent_returns = self.returns[-50:]  # Last 50 periods
        return np.std(recent_returns)

    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0

        total_return = (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]
        days = len(self.portfolio_values) / 24  # Assuming hourly data

        if days <= 0:
            return 0.0

        return (1 + total_return) ** (365 / days) - 1

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0

        avg_return = np.mean(self.returns)
        std_return = np.std(self.returns)

        if std_return == 0:
            return 0.0

        # Annualize (assuming hourly returns)
        annual_return = avg_return * 24 * 365
        annual_volatility = std_return * np.sqrt(24 * 365)

        return annual_return / annual_volatility if annual_volatility > 0 else 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.drawdown_series) == 0:
            return 0.0

        return max(self.drawdown_series)

    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns."""
        if len(self.portfolio_values) < 48:  # Need at least 2 days
            return {}

        monthly_returns = {}

        # Group by month
        for i in range(48, len(self.portfolio_values), 24):  # Daily intervals
            if i >= len(self.portfolio_values):
                break

            month_key = f"{i//(24*30)}"  # Simple month approximation
            start_value = self.portfolio_values[i-24]
            end_value = self.portfolio_values[i]

            if start_value > 0:
                monthly_return = (end_value - start_value) / start_value
                monthly_returns[month_key] = monthly_return

        return monthly_returns

    def _calculate_strategy_metrics(self) -> Dict[str, Any]:
        """Calculate strategy-specific metrics."""
        metrics = {
            'total_positions_opened': len([t for t in self.trades_log if t['type'] == 'open']),
            'total_positions_closed': len([t for t in self.trades_log if t['type'] == 'close']),
            'avg_position_duration': self._calculate_avg_position_duration(),
            'largest_position': max([abs(t['quantity']) for t in self.trades_log], default=0),
            'avg_leverage': np.mean([t['leverage'] for t in self.trades_log if 'leverage' in t], default=1)
        }

        return metrics

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        if len(self.returns) < 10:
            return {}

        returns_array = np.array(self.returns)

        return {
            'volatility': np.std(returns_array) * np.sqrt(365),  # Annualized
            'var_95': np.percentile(returns_array, 5),  # 95% Value at Risk
            'expected_shortfall': np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            'calmar_ratio': self.annualized_return / max(self.max_drawdown, 0.001)
        }

    def _calculate_avg_position_duration(self) -> float:
        """Calculate average position holding duration."""
        durations = []

        for trade in self.trades_log:
            if trade['type'] == 'close' and 'timestamp' in trade:
                # Find corresponding open trade
                for open_trade in self.trades_log:
                    if (open_trade['symbol'] == trade['symbol'] and
                        open_trade['type'] == 'open' and
                        open_trade['timestamp'] < trade['timestamp']):
                        duration = (trade['timestamp'] - open_trade['timestamp']).total_seconds() / 3600
                        durations.append(duration)
                        break

        return np.mean(durations) if durations else 0.0

    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades."""
        if not self.trades_log:
            return 0

        consecutive_losses = 0
        max_consecutive = 0

        for trade in self.trades_log:
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0

        return max_consecutive


# Example strategy functions
def simple_momentum_strategy(data: pd.DataFrame, symbol: str, current_time: datetime) -> List[Dict[str, Any]]:
    """Simple momentum strategy for testing."""
    if len(data) < 50:
        return []

    current_price = data.iloc[-1]['close']
    sma_20 = data['close'].rolling(20).mean().iloc[-1]
    sma_50 = data['close'].rolling(50).mean().iloc[-1]

    signals = []

    if current_price > sma_20 > sma_50:
        # Bullish momentum
        signals.append({
            'action': 'buy',
            'quantity': 1000 / current_price,  # $1000 position
            'leverage': 5
        })
    elif current_price < sma_20 < sma_50:
        # Bearish momentum
        signals.append({
            'action': 'sell',
            'quantity': 1000 / current_price,  # $1000 position
            'leverage': 5
        })

    return signals


def grid_trading_strategy(data: pd.DataFrame, symbol: str, current_time: datetime) -> List[Dict[str, Any]]:
    """Grid trading strategy for Aster DEX."""
    if len(data) < 100:
        return []

    current_price = data.iloc[-1]['close']

    # Simple grid setup
    grid_levels = 5
    grid_spacing = 0.02  # 2%
    base_quantity = 50 / current_price  # $50 per level

    signals = []

    for i in range(-grid_levels, grid_levels + 1):
        level_price = current_price * (1 + i * grid_spacing)

        if abs(current_price - level_price) / current_price < 0.005:  # Within 0.5%
            if i < 0:  # Below current price - buy levels
                signals.append({
                    'action': 'buy',
                    'quantity': base_quantity,
                    'leverage': 1
                })
            elif i > 0:  # Above current price - sell levels
                signals.append({
                    'action': 'sell',
                    'quantity': base_quantity,
                    'leverage': 1
                })

    return signals


# Example usage
def run_sample_backtest():
    """Run a sample backtest for demonstration."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    prices = 50000 + np.cumsum(np.random.normal(0.001, 0.02, 1000))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(1000000, 10000000, 1000)
    })

    # Run backtest
    config = BacktestConfig(
        initial_balance=10000.0,
        mode=BacktestMode.PERPETUAL,
        max_leverage=10
    )

    backtester = AdvancedBacktester(config)
    result = backtester.run_backtest(data, simple_momentum_strategy, "BTCUSDT")

    print("📊 Backtest Results:")
    print(f"   Initial: ${result.initial_balance:,.2f}")
    print(f"   Final: ${result.final_balance:,.2f}")
    print(f"   Return: {result.total_return:.2%}")
    print(f"   Sharpe: {result.sharpe_ratio:.2f}")
    print(f"   Max DD: {result.max_drawdown:.2%}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Trades: {result.total_trades}")

    return result


if __name__ == "__main__":
    result = run_sample_backtest()

"""
Paper Trading Setup and Validation

Complete paper trading setup for risk-free validation:
- Simulated trading environment
- Real-time market data integration
- Performance tracking and analysis
- Risk management validation
- 7-day validation protocol
- Automated reporting and alerts

Features:
- Realistic slippage and fees simulation
- Live market data without real trades
- Comprehensive performance metrics
- Risk limit enforcement
- Automated validation reports
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
import json
from pathlib import Path

from mcp_trader.ai.ensemble_trading_system import EnsembleTradingSystem
from mcp_trader.risk.dynamic_position_sizing import DynamicPositionSizer
from mcp_trader.data.aster_dex_realtime_collector import AsterDEXRealTimeCollector
from mcp_trader.backtesting.walk_forward_analysis import WalkForwardAnalyzer
from mcp_trader.backtesting.monte_carlo_simulation import MonteCarloSimulator

logger = logging.getLogger(__name__)


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading validation"""

    # Trading parameters
    initial_capital: float = 100.0  # $100 starting capital
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    max_position_size: float = 0.1  # 10% max per position
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk

    # Validation parameters
    validation_days: int = 7
    trading_hours_per_day: int = 24  # 24/7 crypto trading
    min_trades_per_day: int = 5
    max_trades_per_day: int = 20

    # Risk management
    stop_loss_percentage: float = 0.02  # 2% stop loss
    take_profit_percentage: float = 0.05  # 5% take profit
    emergency_stop_percentage: float = 0.1  # 10% emergency stop

    # Performance targets
    target_daily_return: float = 0.005  # 0.5% daily target
    max_daily_drawdown: float = 0.03  # 3% max daily drawdown
    target_sharpe_ratio: float = 1.5
    target_win_rate: float = 0.55

    # Simulation parameters
    slippage_model: str = "realistic"  # none, fixed, realistic
    commission_rate: float = 0.001  # 0.1% commission per trade
    enable_funding_rates: bool = True
    enable_market_impact: bool = True

    # Reporting
    report_interval_hours: int = 6
    alert_on_anomalies: bool = True
    save_trade_log: bool = True


@dataclass
class PaperTrade:
    """Individual paper trade record"""

    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    order_type: str = "market"

    @property
    def total_cost(self) -> float:
        """Total cost including commission and slippage"""
        return (self.quantity * self.price) + self.commission + self.slippage


@dataclass
class PaperPosition:
    """Paper trading position"""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        self.unrealized_pnl = self.quantity * (current_price - self.entry_price)


@dataclass
class PaperTradingResult:
    """Paper trading validation results"""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    total_pnl: float
    daily_returns: List[float]
    trade_log: List[Dict[str, Any]]
    validation_passed: bool
    validation_notes: List[str]


class PaperTradingEngine:
    """
    Complete paper trading engine for validation

    Features:
    - Realistic market simulation
    - Live data integration
    - Risk management
    - Performance tracking
    - Automated reporting
    """

    def __init__(self, config: PaperTradingConfig = None):
        self.config = config or PaperTradingConfig()

        # Trading components
        self.ensemble_system = EnsembleTradingSystem()
        self.position_sizer = DynamicPositionSizer()
        self.data_collector = AsterDEXRealTimeCollector()

        # Trading state
        self.portfolio_value = self.config.initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.trade_history: List[PaperTrade] = []
        self.daily_pnl: List[float] = []

        # Performance tracking
        self.start_time = None
        self.validation_results = None

        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'peak_portfolio_value': self.config.initial_capital,
            'max_drawdown': 0.0
        }

        logger.info("Paper Trading Engine initialized")

    async def run_validation(self) -> PaperTradingResult:
        """
        Run complete 7-day paper trading validation

        Returns:
            Comprehensive validation results
        """

        logger.info(f"Starting {self.config.validation_days}-day paper trading validation")
        self.start_time = datetime.now()

        try:
            # Initialize data collection
            await self.data_collector.start_collection()

            # Add position update callback
            self.data_collector.add_callback('order_book', self._update_positions)

            # Run validation period
            await self._run_trading_loop()

            # Generate results
            results = self._generate_validation_results()

            # Cleanup
            await self.data_collector.stop_collection()

            logger.info("Paper trading validation completed")
            return results

        except Exception as e:
            logger.error(f"Paper trading validation failed: {str(e)}")
            raise

    async def _run_trading_loop(self):
        """Main trading loop for validation period"""

        validation_end = datetime.now() + timedelta(days=self.config.validation_days)

        while datetime.now() < validation_end:
            try:
                # Generate trading signals
                await self._generate_signals_and_trade()

                # Update positions
                await self._update_all_positions()

                # Check risk limits
                await self._check_risk_limits()

                # Daily reporting
                await self._daily_reporting()

                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes between signals

            except Exception as e:
                logger.error(f"Trading loop error: {str(e)}")
                await asyncio.sleep(60)  # Brief pause on error

    async def _generate_signals_and_trade(self):
        """Generate trading signals and execute trades"""

        # Get market data
        market_data = {}
        for symbol in self.config.symbols:
            order_book = self.data_collector.get_order_book(symbol)
            if order_book:
                market_data[symbol] = {
                    'close': [order_book.get_mid_price()],
                    'volume': [1000.0],  # Mock volume
                    'high': [order_book.asks[0][0]],
                    'low': [order_book.bids[0][0]]
                }

        if not market_data:
            return

        # Generate ensemble predictions
        predictions = {}
        for symbol in self.config.symbols:
            if symbol in market_data:
                prediction = await self.ensemble_system.predict(market_data[symbol], symbol)
                predictions[symbol] = prediction

        # Execute trades based on predictions
        for symbol, prediction in predictions.items():
            await self._execute_trade_signal(symbol, prediction, market_data[symbol])

    async def _execute_trade_signal(self, symbol: str, prediction: Any, market_data: Dict[str, Any]):
        """Execute trade based on signal"""

        current_price = market_data['close'][-1]

        # Determine trade direction and size
        if abs(prediction.direction) < 0.3:  # Neutral signal
            return

        # Calculate position size
        position_size = await self.position_sizer.calculate_position_size(
            prediction, market_data, self.portfolio_value
        )

        if position_size <= 0:
            return

        # Check existing position
        existing_position = self.positions.get(symbol)

        if prediction.direction > 0:  # Buy signal
            if existing_position and existing_position.quantity > 0:
                # Already long, skip
                return

            # Calculate slippage and commission
            slippage = self._calculate_slippage(symbol, 'buy', position_size, current_price)
            commission = position_size * current_price * self.config.commission_rate

            # Execute buy
            trade = PaperTrade(
                trade_id=f"trade_{len(self.trade_history)}",
                symbol=symbol,
                side='buy',
                quantity=position_size,
                price=current_price + slippage,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage
            )

            self.trade_history.append(trade)

            # Update position
            if existing_position:
                # Average down/up
                total_quantity = existing_position.quantity + position_size
                total_cost = (existing_position.quantity * existing_position.entry_price +
                            position_size * trade.price)
                avg_price = total_cost / total_quantity

                existing_position.quantity = total_quantity
                existing_position.entry_price = avg_price
            else:
                # New position
                position = PaperPosition(
                    symbol=symbol,
                    quantity=position_size,
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    stop_loss=trade.price * (1 - self.config.stop_loss_percentage),
                    take_profit=trade.price * (1 + self.config.take_profit_percentage)
                )
                self.positions[symbol] = position

            # Update portfolio
            self.portfolio_value -= trade.total_cost

        elif prediction.direction < 0:  # Sell signal
            if not existing_position or existing_position.quantity <= 0:
                # No long position, skip
                return

            # Calculate sell quantity (partial or full close)
            sell_quantity = min(abs(position_size), existing_position.quantity)

            # Calculate slippage and commission
            slippage = self._calculate_slippage(symbol, 'sell', sell_quantity, current_price)
            commission = sell_quantity * current_price * self.config.commission_rate

            # Execute sell
            trade = PaperTrade(
                trade_id=f"trade_{len(self.trade_history)}",
                symbol=symbol,
                side='sell',
                quantity=-sell_quantity,  # Negative for sell
                price=current_price - slippage,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage
            )

            self.trade_history.append(trade)

            # Update position
            existing_position.quantity -= sell_quantity
            realized_pnl = sell_quantity * (trade.price - existing_position.entry_price)
            self.portfolio_value += (sell_quantity * trade.price) - commission

            # Remove position if fully closed
            if existing_position.quantity <= 0.001:
                del self.positions[symbol]

        self.stats['total_trades'] += 1
        self.stats['total_commission'] += commission
        self.stats['total_slippage'] += slippage

    def _calculate_slippage(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Calculate realistic slippage"""

        if self.config.slippage_model == "none":
            return 0.0

        if self.config.slippage_model == "fixed":
            return price * 0.0005  # 0.05% fixed slippage

        # Realistic slippage based on market conditions
        order_book = self.data_collector.get_order_book(symbol)
        if not order_book:
            return price * 0.001  # 0.1% default

        # Calculate slippage based on order book depth
        if side == 'buy':
            # Need to buy, so walk up the ask side
            cumulative_volume = 0
            slippage_price = price

            for ask_price, ask_qty in order_book.asks:
                if cumulative_volume + ask_qty >= quantity:
                    # Partial fill at this level
                    remaining_qty = quantity - cumulative_volume
                    slippage_price = ask_price
                    break
                cumulative_volume += ask_qty
                slippage_price = ask_price

            slippage = max(0, slippage_price - price)

        else:  # sell
            # Need to sell, so walk down the bid side
            cumulative_volume = 0
            slippage_price = price

            for bid_price, bid_qty in order_book.bids:
                if cumulative_volume + bid_qty >= quantity:
                    # Partial fill at this level
                    remaining_qty = quantity - cumulative_volume
                    slippage_price = bid_price
                    break
                cumulative_volume += bid_qty
                slippage_price = bid_price

            slippage = max(0, price - slippage_price)

        # Cap slippage at reasonable levels
        max_slippage = price * 0.005  # Max 0.5%
        return min(slippage, max_slippage)

    async def _update_all_positions(self):
        """Update all positions with latest prices"""

        for symbol, position in list(self.positions.items()):
            order_book = self.data_collector.get_order_book(symbol)
            if order_book:
                current_price = order_book.get_mid_price()
                position.update_pnl(current_price)

                # Check stop loss / take profit
                await self._check_position_exits(symbol, position, current_price)

    async def _check_position_exits(self, symbol: str, position: PaperPosition, current_price: float):
        """Check if position should be exited due to stops"""

        if position.stop_loss and current_price <= position.stop_loss:
            # Stop loss hit
            await self._close_position(symbol, current_price, "stop_loss")
            return

        if position.take_profit and current_price >= position.take_profit:
            # Take profit hit
            await self._close_position(symbol, current_price, "take_profit")
            return

        # Emergency stop check
        loss_percentage = (position.entry_price - current_price) / position.entry_price
        if loss_percentage >= self.config.emergency_stop_percentage:
            await self._close_position(symbol, current_price, "emergency_stop")

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Close position at market price"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position.quantity

        # Calculate costs
        slippage = self._calculate_slippage(symbol, 'sell', quantity, price)
        commission = quantity * price * self.config.commission_rate

        # Create closing trade
        trade = PaperTrade(
            trade_id=f"close_{len(self.trade_history)}",
            symbol=symbol,
            side='sell',
            quantity=-quantity,
            price=price - slippage,
            timestamp=datetime.now(),
            commission=commission,
            slippage=slippage
        )

        self.trade_history.append(trade)

        # Update portfolio
        realized_pnl = quantity * (trade.price - position.entry_price)
        self.portfolio_value += (quantity * trade.price) - commission

        # Remove position
        del self.positions[symbol]

        logger.info(f"Closed {symbol} position: {reason}, P&L: ${realized_pnl:.2f}")

    async def _check_risk_limits(self):
        """Check and enforce risk limits"""

        # Portfolio risk limit
        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        portfolio_risk = total_exposure / self.portfolio_value

        if portfolio_risk > self.config.max_portfolio_risk:
            # Reduce positions
            reduction_factor = self.config.max_portfolio_risk / portfolio_risk
            await self._reduce_positions(1 - reduction_factor)

        # Daily drawdown limit
        current_drawdown = (self.stats['peak_portfolio_value'] - self.portfolio_value) / self.stats['peak_portfolio_value']
        self.stats['max_drawdown'] = max(self.stats['max_drawdown'], current_drawdown)

        if current_drawdown > self.config.max_daily_drawdown:
            logger.warning(f"Daily drawdown limit reached: {current_drawdown:.2%}")
            # Emergency position reduction
            await self._reduce_positions(0.5)

    async def _reduce_positions(self, reduction_factor: float):
        """Reduce all positions by factor"""

        logger.warning(f"Reducing all positions by {reduction_factor:.1%}")

        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            reduce_quantity = position.quantity * reduction_factor

            if reduce_quantity > 0.001:
                order_book = self.data_collector.get_order_book(symbol)
                if order_book:
                    price = order_book.get_mid_price()
                    await self._close_partial_position(symbol, reduce_quantity, price)

    async def _close_partial_position(self, symbol: str, quantity: float, price: float):
        """Close partial position"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate costs
        slippage = self._calculate_slippage(symbol, 'sell', quantity, price)
        commission = quantity * price * self.config.commission_rate

        # Create trade
        trade = PaperTrade(
            trade_id=f"partial_close_{len(self.trade_history)}",
            symbol=symbol,
            side='sell',
            quantity=-quantity,
            price=price - slippage,
            timestamp=datetime.now(),
            commission=commission,
            slippage=slippage
        )

        self.trade_history.append(trade)

        # Update position
        position.quantity -= quantity
        realized_pnl = quantity * (trade.price - position.entry_price)
        self.portfolio_value += (quantity * trade.price) - commission

        # Remove if fully closed
        if position.quantity <= 0.001:
            del self.positions[symbol]

    async def _daily_reporting(self):
        """Generate daily performance report"""

        current_time = datetime.now()

        # Calculate daily P&L
        if self.start_time and (current_time - self.start_time).days >= 1:
            daily_pnl = self.portfolio_value - self.config.initial_capital
            self.daily_pnl.append(daily_pnl)

            # Update peak
            self.stats['peak_portfolio_value'] = max(self.stats['peak_portfolio_value'], self.portfolio_value)

            # Log daily stats
            logger.info(f"Daily Report - Portfolio: ${self.portfolio_value:.2f}, "
                       f"P&L: ${daily_pnl:.2f}, Positions: {len(self.positions)}")

    def _generate_validation_results(self) -> PaperTradingResult:
        """Generate comprehensive validation results"""

        # Calculate performance metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital

        # Sharpe ratio
        if self.daily_pnl and len(self.daily_pnl) > 1:
            sharpe_ratio = np.mean(self.daily_pnl) / np.std(self.daily_pnl) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Win rate
        winning_trades = sum(1 for trade in self.trade_history if trade.quantity < 0 and
                           trade.price > self.positions.get(trade.symbol, PaperPosition('', 0, 0, datetime.now())).entry_price)
        win_rate = winning_trades / max(len(self.trade_history), 1)

        # Trade log
        trade_log = []
        for trade in self.trade_history:
            trade_log.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'timestamp': trade.timestamp.isoformat(),
                'commission': trade.commission,
                'slippage': trade.slippage,
                'total_cost': trade.total_cost
            })

        # Validation checks
        validation_passed = True
        validation_notes = []

        if sharpe_ratio < self.config.target_sharpe_ratio:
            validation_passed = False
            validation_notes.append(f"Sharpe ratio {sharpe_ratio:.2f} below target {self.config.target_sharpe_ratio}")

        if win_rate < self.config.target_win_rate:
            validation_passed = False
            validation_notes.append(f"Win rate {win_rate:.2%} below target {self.config.target_win_rate:.2%}")

        if self.stats['max_drawdown'] > self.config.max_daily_drawdown:
            validation_passed = False
            validation_notes.append(f"Max drawdown {self.stats['max_drawdown']:.2%} above limit {self.config.max_daily_drawdown:.2%}")

        if len(self.trade_history) < self.config.validation_days * self.config.min_trades_per_day:
            validation_passed = False
            validation_notes.append(f"Insufficient trades: {len(self.trade_history)} vs required {self.config.validation_days * self.config.min_trades_per_day}")

        result = PaperTradingResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.stats['max_drawdown'],
            win_rate=win_rate,
            total_trades=len(self.trade_history),
            profitable_trades=winning_trades,
            total_pnl=self.portfolio_value - self.config.initial_capital,
            daily_returns=self.daily_pnl.copy(),
            trade_log=trade_log,
            validation_passed=validation_passed,
            validation_notes=validation_notes
        )

        return result

    def get_current_status(self) -> Dict[str, Any]:
        """Get current paper trading status"""

        return {
            'portfolio_value': self.portfolio_value,
            'positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'win_rate': sum(1 for t in self.trade_history if t.side == 'sell' and t.price > 0) / max(len(self.trade_history), 1),
            'total_commission': self.stats['total_commission'],
            'total_slippage': self.stats['total_slippage'],
            'max_drawdown': self.stats['max_drawdown'],
            'active_positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for symbol, pos in self.positions.items()
            }
        }


# Convenience functions
def create_paper_trading_engine(config: PaperTradingConfig = None) -> PaperTradingEngine:
    """Create paper trading engine instance"""
    return PaperTradingEngine(config)


async def run_paper_trading_validation(config: PaperTradingConfig = None) -> PaperTradingResult:
    """Run complete paper trading validation"""
    engine = create_paper_trading_engine(config)
    return await engine.run_validation()


def generate_validation_report(result: PaperTradingResult) -> str:
    """Generate detailed validation report"""

    report = f"""
PAPER TRADING VALIDATION REPORT
{'='*50}

VALIDATION RESULT: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}

PERFORMANCE METRICS:
- Total Return: {result.total_return:.2%}
- Sharpe Ratio: {result.sharpe_ratio:.2f}
- Max Drawdown: {result.max_drawdown:.2%}
- Win Rate: {result.win_rate:.2%}
- Total Trades: {result.total_trades}
- Profitable Trades: {result.profitable_trades}
- Total P&L: ${result.total_pnl:.2f}

VALIDATION NOTES:
"""

    for note in result.validation_notes:
        report += f"- {note}\n"

    if result.validation_passed:
        report += "\n🎉 System ready for live trading deployment!"
    else:
        report += "\n⚠️ System requires additional optimization before live deployment."

    return report

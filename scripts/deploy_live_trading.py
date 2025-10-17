"""
Live Trading Deployment with $100 Capital Scaling

Complete live trading deployment system:
- Progressive capital scaling from $100
- Risk management with position limits
- Performance-based capital allocation
- Emergency stop mechanisms
- Real-time monitoring and alerts
- Profit taking and reinvestment strategy

Features:
- Start with $100, scale based on performance
- Maximum 1% risk per trade initially
- Scale up position sizes with profits
- Automatic profit taking at targets
- Emergency stops and risk controls
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path
import sys

from mcp_trader.ai.ensemble_trading_system import EnsembleTradingSystem
from mcp_trader.risk.dynamic_position_sizing import DynamicPositionSizer
from mcp_trader.data.aster_dex_realtime_collector import AsterDEXRealTimeCollector
from mcp_trader.ai.adaptive_retraining_system import AdaptiveRetrainingSystem
from scripts.setup_paper_trading import PaperTradingEngine

logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading deployment"""

    # Capital management
    initial_capital: float = 100.0  # Start with $100
    max_capital: float = 10000.0  # Scale up to $10K
    profit_target_levels: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0])  # 10%, 25%, 50%, 100%, 200%

    # Risk management
    initial_risk_per_trade: float = 0.005  # 0.5% initial risk
    max_risk_per_trade: float = 0.02  # 2% max risk
    portfolio_risk_limit: float = 0.05  # 5% max portfolio risk
    daily_loss_limit: float = 0.03  # 3% daily loss limit
    emergency_stop_loss: float = 0.1  # 10% emergency stop

    # Scaling rules
    scale_up_threshold: float = 0.05  # Scale up after 5% profit
    scale_down_threshold: float = -0.03  # Scale down after 3% loss
    max_position_size: float = 0.2  # 20% max position size
    min_position_size: float = 0.001  # 0.1% min position size

    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT', 'ETHUSDT'])
    min_trade_interval: int = 300  # 5 minutes between trades
    max_trades_per_day: int = 10
    trading_enabled: bool = False  # Start disabled for safety

    # Profit taking
    profit_taking_levels: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.1, 0.2])  # 2%, 5%, 10%, 20%
    trailing_stop_activation: float = 0.02  # Activate trailing stop after 2% profit

    # Monitoring
    status_update_interval: int = 60  # 1 minute
    alert_email: Optional[str] = None
    webhook_url: Optional[str] = None

    # Safety
    require_paper_trading_validation: bool = True
    emergency_kill_switch: bool = False


@dataclass
class CapitalLevel:
    """Capital scaling level configuration"""

    level: int
    min_capital: float
    max_capital: float
    risk_per_trade: float
    max_position_size: float
    max_portfolio_risk: float
    description: str


@dataclass
class LiveTradingState:
    """Current live trading state"""

    current_capital: float
    available_capital: float
    invested_capital: float
    total_pnl: float
    daily_pnl: float
    capital_level: int
    trading_enabled: bool
    emergency_stop: bool
    last_trade_time: Optional[datetime] = None
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    best_equity: float = 0.0


class LiveTradingDeployment:
    """
    Complete live trading deployment with capital scaling

    Features:
    - Progressive scaling from $100
    - Performance-based risk management
    - Automated profit taking
    - Emergency risk controls
    """

    def __init__(self, config: LiveTradingConfig = None):
        self.config = config or LiveTradingConfig()

        # Initialize components
        self.ensemble_system = EnsembleTradingSystem()
        self.position_sizer = DynamicPositionSizer()
        self.data_collector = AsterDEXRealTimeCollector()
        self.retraining_system = AdaptiveRetrainingSystem()

        # Trading state
        self.state = LiveTradingState(
            current_capital=self.config.initial_capital,
            available_capital=self.config.initial_capital,
            invested_capital=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            capital_level=1,
            trading_enabled=False,
            emergency_stop=False,
            best_equity=self.config.initial_capital
        )

        # Capital scaling levels
        self.capital_levels = self._define_capital_levels()

        # Active positions and orders
        self.positions = {}
        self.pending_orders = {}

        # Performance tracking
        self.trade_history = []
        self.daily_performance = {}

        # Callbacks
        self.status_callbacks = []
        self.alert_callbacks = []

        logger.info("Live Trading Deployment initialized")

    def _define_capital_levels(self) -> List[CapitalLevel]:
        """Define capital scaling levels"""

        levels = [
            CapitalLevel(
                level=1,
                min_capital=0,
                max_capital=500,
                risk_per_trade=0.005,  # 0.5%
                max_position_size=0.1,  # 10%
                max_portfolio_risk=0.03,  # 3%
                description="$0-$500: Conservative risk management"
            ),
            CapitalLevel(
                level=2,
                min_capital=500,
                max_capital=2000,
                risk_per_trade=0.01,  # 1%
                max_position_size=0.15,  # 15%
                max_portfolio_risk=0.04,  # 4%
                description="$500-$2K: Moderate risk increase"
            ),
            CapitalLevel(
                level=3,
                min_capital=2000,
                max_capital=5000,
                risk_per_trade=0.015,  # 1.5%
                max_position_size=0.2,  # 20%
                max_portfolio_risk=0.05,  # 5%
                description="$2K-$5K: Balanced risk-reward"
            ),
            CapitalLevel(
                level=4,
                min_capital=5000,
                max_capital=10000,
                risk_per_trade=0.02,  # 2%
                max_position_size=0.25,  # 25%
                max_portfolio_risk=0.06,  # 6%
                description="$5K-$10K: Aggressive scaling"
            )
        ]

        return levels

    async def start_live_trading(self):
        """Start live trading deployment"""

        logger.info("🚀 Starting live trading deployment with $100 capital")

        # Pre-deployment validation
        if self.config.require_paper_trading_validation:
            await self._run_pre_deployment_validation()

        # Initialize systems
        await self._initialize_trading_systems()

        # Enable trading
        self.state.trading_enabled = True

        logger.info("✅ Live trading deployment started successfully")

        # Start main trading loop
        await self._run_live_trading_loop()

    async def _run_pre_deployment_validation(self):
        """Run paper trading validation before live deployment"""

        logger.info("📊 Running pre-deployment paper trading validation...")

        from scripts.setup_paper_trading import create_paper_trading_engine, PaperTradingConfig

        # Run 7-day validation
        paper_config = PaperTradingConfig(
            initial_capital=100.0,
            validation_days=7,
            target_sharpe_ratio=1.0,
            target_win_rate=0.5
        )

        engine = create_paper_trading_engine(paper_config)
        results = await engine.run_validation()

        if not results.validation_passed:
            raise ValueError(f"Paper trading validation failed: {results.validation_notes}")

        logger.info("✅ Paper trading validation passed")

    async def _initialize_trading_systems(self):
        """Initialize all trading systems"""

        logger.info("🔧 Initializing trading systems...")

        # Start data collection
        await self.data_collector.start_collection()

        # Start adaptive retraining
        await self.retraining_system.start_monitoring()

        # Add callbacks
        self.data_collector.add_callback('order_book', self._on_market_data)
        self.retraining_system.add_emergency_callback(self._on_emergency_event)

        logger.info("✅ Trading systems initialized")

    async def _run_live_trading_loop(self):
        """Main live trading loop"""

        logger.info("🔄 Starting live trading loop")

        while not self.state.emergency_stop:
            try:
                # Check if trading is enabled
                if not self.state.trading_enabled:
                    await asyncio.sleep(60)
                    continue

                # Update positions and P&L
                await self._update_positions()

                # Check for trading opportunities
                await self._check_trading_opportunities()

                # Update capital scaling
                self._update_capital_scaling()

                # Risk management checks
                await self._risk_management_checks()

                # Status updates
                await self._status_update()

                # Wait before next iteration
                await asyncio.sleep(30)  # 30 second cycles

            except Exception as e:
                logger.error(f"Live trading loop error: {str(e)}")
                await self._handle_error(e)

    async def _check_trading_opportunities(self):
        """Check for trading opportunities"""

        # Rate limiting
        if self.state.last_trade_time:
            time_since_last_trade = datetime.now() - self.state.last_trade_time
            if time_since_last_trade.seconds < self.config.min_trade_interval:
                return

        # Daily trade limit
        if self.state.daily_trade_count >= self.config.max_trades_per_day:
            return

        # Get market data
        market_data = {}
        for symbol in self.config.symbols:
            order_book = self.data_collector.get_order_book(symbol)
            if order_book:
                market_data[symbol] = {
                    'close': [order_book.get_mid_price()],
                    'volume': [1000.0],  # Estimated volume
                    'high': [order_book.asks[0][0]],
                    'low': [order_book.bids[0][0]]
                }

        if not market_data:
            return

        # Generate predictions
        for symbol in self.config.symbols:
            if symbol in market_data:
                prediction = await self.ensemble_system.predict(market_data[symbol], symbol)

                # Check if signal is strong enough
                if abs(prediction.direction) >= 0.4:  # 40% confidence threshold
                    await self._execute_live_trade(symbol, prediction, market_data[symbol])

    async def _execute_live_trade(self, symbol: str, prediction: Any, market_data: Dict[str, Any]):
        """Execute live trade"""

        try:
            # Calculate position size
            position_size = await self._calculate_live_position_size(prediction, market_data)

            if position_size <= 0:
                return

            current_price = market_data['close'][-1]

            # Execute trade (in live system, this would connect to Aster DEX API)
            if prediction.direction > 0:  # Buy
                success = await self._place_buy_order(symbol, position_size, current_price)
            else:  # Sell
                success = await self._place_sell_order(symbol, position_size, current_price)

            if success:
                self.state.last_trade_time = datetime.now()
                self.state.daily_trade_count += 1

                # Update state
                trade_value = position_size * current_price
                self.state.available_capital -= trade_value
                self.state.invested_capital += trade_value

                logger.info(f"✅ Executed {prediction.direction > 0 and 'BUY' or 'SELL'} {symbol}: "
                           f"{position_size:.4f} @ ${current_price:.2f}")

                # Send alert
                await self._send_alert("TRADE_EXECUTED", {
                    'symbol': symbol,
                    'side': 'buy' if prediction.direction > 0 else 'sell',
                    'size': position_size,
                    'price': current_price,
                    'capital_remaining': self.state.available_capital
                })

        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            await self._send_alert("TRADE_FAILED", {'error': str(e)})

    async def _calculate_live_position_size(self, prediction: Any, market_data: Dict[str, Any]) -> float:
        """Calculate position size for live trading"""

        # Get current capital level
        capital_level = self.capital_levels[self.state.capital_level - 1]

        # Base position size
        max_position_value = self.state.available_capital * capital_level.max_position_size
        risk_amount = self.state.current_capital * capital_level.risk_per_trade

        # Calculate stop loss distance
        current_price = market_data['close'][-1]
        stop_loss_distance = current_price * prediction.stop_loss

        # Position size based on risk
        position_size = risk_amount / stop_loss_distance

        # Limit by available capital
        max_size_by_capital = max_position_value / current_price
        position_size = min(position_size, max_size_by_capital)

        # Apply conviction scaling
        conviction_multiplier = min(abs(prediction.direction) * 2.5, 2.0)
        position_size *= conviction_multiplier

        # Final limits
        position_size = max(position_size, self.config.min_position_size)

        return position_size

    async def _place_buy_order(self, symbol: str, size: float, price: float) -> bool:
        """Place buy order (mock implementation)"""

        # In real implementation, this would connect to Aster DEX API
        logger.info(f"📈 Placing BUY order: {symbol} {size:.4f} @ ~${price:.2f}")

        # Simulate order placement
        await asyncio.sleep(0.1)  # Simulate API call

        # Mock success/failure
        success = True  # In real system, check API response

        if success:
            # Record position
            self.positions[symbol] = {
                'size': size,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop_loss': price * 0.95,  # 5% stop loss
                'take_profit': price * 1.10  # 10% take profit
            }

        return success

    async def _place_sell_order(self, symbol: str, size: float, price: float) -> bool:
        """Place sell order (mock implementation)"""

        # Check if we have position to sell
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        sell_size = min(size, position['size'])

        logger.info(f"📉 Placing SELL order: {symbol} {sell_size:.4f} @ ~${price:.2f}")

        # Simulate order placement
        await asyncio.sleep(0.1)

        # Calculate P&L
        pnl = sell_size * (price - position['entry_price'])
        self.state.total_pnl += pnl
        self.state.available_capital += (sell_size * price)
        self.state.invested_capital -= (sell_size * price)

        # Update or remove position
        position['size'] -= sell_size
        if position['size'] <= 0.001:
            del self.positions[symbol]

        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'side': 'sell',
            'size': sell_size,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.now()
        })

        return True

    async def _update_positions(self):
        """Update all positions with latest prices"""

        for symbol, position in list(self.positions.items()):
            order_book = self.data_collector.get_order_book(symbol)
            if order_book:
                current_price = order_book.get_mid_price()

                # Update P&L
                unrealized_pnl = position['size'] * (current_price - position['entry_price'])
                position['unrealized_pnl'] = unrealized_pnl

                # Check stops
                if current_price <= position['stop_loss']:
                    await self._close_position(symbol, current_price, "stop_loss")
                elif current_price >= position['take_profit']:
                    await self._close_position(symbol, current_price, "take_profit")

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Close position"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        size = position['size']

        # Execute sell
        success = await self._place_sell_order(symbol, size, price)

        if success:
            logger.info(f"🔒 Closed {symbol} position: {reason} @ ${price:.2f}")
            await self._send_alert("POSITION_CLOSED", {
                'symbol': symbol,
                'reason': reason,
                'price': price,
                'pnl': position.get('unrealized_pnl', 0)
            })

    def _update_capital_scaling(self):
        """Update capital scaling based on performance"""

        current_return = self.state.total_pnl / self.config.initial_capital

        # Find appropriate capital level
        new_level = 1
        for level in self.capital_levels:
            if self.state.current_capital >= level.min_capital:
                new_level = level.level
            else:
                break

        if new_level != self.state.capital_level:
            old_level = self.state.capital_level
            self.state.capital_level = new_level
            logger.info(f"📈 Capital level upgraded: {old_level} → {new_level}")
            await self._send_alert("CAPITAL_LEVEL_UP", {
                'old_level': old_level,
                'new_level': new_level,
                'current_capital': self.state.current_capital
            })

    async def _risk_management_checks(self):
        """Perform risk management checks"""

        # Daily loss limit
        if self.state.daily_pnl <= -self.state.current_capital * self.config.daily_loss_limit:
            logger.warning("🚨 Daily loss limit reached")
            self.state.trading_enabled = False
            await self._send_alert("DAILY_LOSS_LIMIT", {
                'daily_pnl': self.state.daily_pnl,
                'limit': self.state.current_capital * self.config.daily_loss_limit
            })

        # Emergency stop
        drawdown = (self.state.best_equity - self.state.current_capital) / self.state.best_equity
        if drawdown >= self.config.emergency_stop_loss:
            logger.critical("🚨 EMERGENCY STOP ACTIVATED")
            self.state.emergency_stop = True
            await self._emergency_shutdown()

        # Update best equity
        self.state.best_equity = max(self.state.best_equity, self.state.current_capital)

    async def _emergency_shutdown(self):
        """Emergency shutdown - close all positions"""

        logger.critical("🔥 Emergency shutdown initiated")

        # Close all positions at market
        for symbol in list(self.positions.keys()):
            order_book = self.data_collector.get_order_book(symbol)
            if order_book:
                price = order_book.get_mid_price()
                await self._close_position(symbol, price, "emergency_shutdown")

        self.state.trading_enabled = False

        await self._send_alert("EMERGENCY_SHUTDOWN", {
            'remaining_capital': self.state.available_capital,
            'total_pnl': self.state.total_pnl
        })

    async def _status_update(self):
        """Send status update"""

        status = {
            'capital': self.state.current_capital,
            'available': self.state.available_capital,
            'invested': self.state.invested_capital,
            'total_pnl': self.state.total_pnl,
            'daily_pnl': self.state.daily_pnl,
            'capital_level': self.state.capital_level,
            'active_positions': len(self.positions),
            'trading_enabled': self.state.trading_enabled,
            'emergency_stop': self.state.emergency_stop
        }

        # Send to callbacks
        for callback in self.status_callbacks:
            await callback(status)

    async def _send_alert(self, alert_type: str, data: Dict[str, Any]):
        """Send alert notification"""

        alert = {
            'type': alert_type,
            'timestamp': datetime.now(),
            'data': data
        }

        # Send to callbacks
        for callback in self.alert_callbacks:
            await callback(alert)

    async def _on_market_data(self, order_book):
        """Handle market data updates"""
        # Update position valuations
        pass

    async def _on_emergency_event(self, alerts, risk_metrics):
        """Handle emergency events from retraining system"""
        if "KILL_SWITCH" in str(alerts):
            await self._emergency_shutdown()

    async def _handle_error(self, error: Exception):
        """Handle trading errors"""
        logger.error(f"Trading error: {str(error)}")
        await self._send_alert("TRADING_ERROR", {'error': str(error)})

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""

        return {
            'current_capital': self.state.current_capital,
            'available_capital': self.state.available_capital,
            'invested_capital': self.state.invested_capital,
            'total_pnl': self.state.total_pnl,
            'capital_level': self.state.capital_level,
            'trading_enabled': self.state.trading_enabled,
            'emergency_stop': self.state.emergency_stop,
            'active_positions': len(self.positions),
            'daily_trade_count': self.state.daily_trade_count,
            'total_trades': len(self.trade_history),
            'win_rate': len([t for t in self.trade_history if t['pnl'] > 0]) / max(len(self.trade_history), 1)
        }


# Convenience functions
def create_live_trading_deployment(config: LiveTradingConfig = None) -> LiveTradingDeployment:
    """Create live trading deployment instance"""
    return LiveTradingDeployment(config)


async def start_live_trading_deployment(config: LiveTradingConfig = None):
    """Start live trading deployment"""
    deployment = create_live_trading_deployment(config)
    await deployment.start_live_trading()
    return deployment


def get_deployment_status(deployment: LiveTradingDeployment) -> Dict[str, Any]:
    """Get deployment status"""
    return deployment.get_deployment_status()


def generate_live_trading_report(deployment: LiveTradingDeployment) -> str:
    """Generate comprehensive live trading report"""

    status = deployment.get_deployment_status()

    report = f"""
LIVE TRADING DEPLOYMENT REPORT
{'='*50}

CAPITAL STATUS:
- Current Capital: ${status['current_capital']:.2f}
- Available Capital: ${status['available_capital']:.2f}
- Invested Capital: ${status['invested_capital']:.2f}
- Total P&L: ${status['total_pnl']:.2f}
- Capital Level: {status['capital_level']}

TRADING STATUS:
- Trading Enabled: {status['trading_enabled']}
- Emergency Stop: {status['emergency_stop']}
- Active Positions: {status['active_positions']}
- Daily Trades: {status['daily_trade_count']}
- Total Trades: {status['total_trades']}
- Win Rate: {status['win_rate']:.2%}

PERFORMANCE:
- Return on Initial Capital: {status['total_pnl']/100:.2%}
- Daily P&L: ${status['daily_pnl']:.2f}

SYSTEM STATUS: {'🟢 ACTIVE' if status['trading_enabled'] and not status['emergency_stop'] else '🔴 INACTIVE'}
"""

    return report

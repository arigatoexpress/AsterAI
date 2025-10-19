"""
Autonomous Trading Engine for Aster DEX
Handles real-time trading decisions, risk management, and profit optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

from ..execution.aster_client import (
    AsterClient, OrderSide, OrderType, KlineInterval,
    AccountBalance, AccountPosition, TickerPrice
)
from ..config import get_settings, PRIORITY_SYMBOLS
from .types import TradingMode, MarketRegime, TradingDecision, PortfolioState
from .strategies.grid_strategy import GridStrategy
from .strategies.volatility_strategy import VolatilityStrategy
from ..risk.volatility.risk_manager import VolatilityRiskManager
from ..data.aster_feed import AsterDataFeed
from ..backtesting.cost_model import OnChainCostModel
from ..monitoring.metrics import Metrics

logger = logging.getLogger(__name__)  # symbol -> grid config


class AutonomousTrader:
    """
    Main autonomous trading engine for Aster DEX.
    Manages multiple strategies, risk controls, and profit optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()

        # Core components
        self.aster_client = AsterClient(
            self.settings.aster_api_key,
            self.settings.aster_api_secret
        )
        self.data_feed = AsterDataFeed()
        self.risk_manager = VolatilityRiskManager(self.config.get('risk_config', {}))

        # Trading strategies
        self.grid_strategy = GridStrategy(self.config.get('grid_config', {}))
        self.volatility_strategy = VolatilityStrategy(self.config.get('volatility_config', {}))

        # State management
        self.portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            total_balance=0.0,
            available_balance=0.0,
            total_positions_value=0.0,
            unrealized_pnl=0.0,
            active_positions={},
            active_grids={}
        )

        # Trading configuration
        self.trading_mode = TradingMode(self.config.get('trading_mode', 'hybrid'))
        self.max_concurrent_positions = self.config.get('max_concurrent_positions', 3)
        self.min_position_size = self.config.get('min_position_size', 10.0)  # USD
        self.max_position_size = self.config.get('max_position_size', 1000.0)  # USD

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.start_time = datetime.now()
        self.start_of_day_value = 0.0
        self._last_pnl_reset_date = datetime.utcnow().date()

        # Control flags
        self.is_running = False
        self.emergency_stop = False
        
        # Observability and cost model
        self.metrics = Metrics()
        self.cost_model = OnChainCostModel(client=self.aster_client)

        logger.info(f"AutonomousTrader initialized in {self.trading_mode.value} mode")

    async def start(self):
        """Start the autonomous trading system."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return

        logger.info("Starting Autonomous Trading System...")
        self.is_running = True

        try:
            # Initialize connections
            await self._initialize_connections()

            # Start main trading loop
            await self._trading_loop()

        except Exception as e:
            logger.error(f"Critical error in trading system: {e}")
            await self._emergency_shutdown()
        finally:
            self.is_running = False

    async def stop(self):
        """Stop the autonomous trading system."""
        logger.info("Stopping Autonomous Trading System...")
        self.is_running = False

        # Close all positions if configured
        if self.config.get('close_positions_on_stop', True):
            await self._close_all_positions()

        # Disconnect from APIs
        await self.aster_client.disconnect()

    async def _initialize_connections(self):
        """Initialize all API connections and data feeds."""
        try:
            # Connect to Aster DEX
            await self.aster_client.connect()
            logger.info("Connected to Aster DEX")

            # Start data feed subscriptions
            await self.data_feed.start()
            await self._subscribe_to_market_data()

            # Initialize portfolio state
            await self._update_portfolio_state()

            # Validate configuration
            await self._validate_trading_setup()

            logger.info("All connections initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise

    async def _trading_loop(self):
        """Main trading loop with decision making."""
        decision_interval = self.config.get('decision_interval_seconds', 60)

        while self.is_running and not self.emergency_stop:
            try:
                start_time = time.time()

                # Update market data and portfolio state
                await self._update_market_data()
                await self._update_portfolio_state()

                # Assess market conditions
                market_regime = await self._assess_market_regime()

                # Make trading decisions
                decisions = await self._make_trading_decisions(market_regime)

                # Execute decisions with risk checks
                await self._execute_decisions(decisions)

                # Update performance metrics
                self._update_performance_metrics()

                # Log status
                await self._log_status_update(market_regime)

                # Wait for next decision cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, decision_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _assess_market_regime(self) -> MarketRegime:
        """Assess current market conditions across all traded assets."""
        try:
            regime_scores = {}

            for symbol in PRIORITY_SYMBOLS:
                # Get recent price data
                klines = await self.data_feed.get_klines(symbol, KlineInterval.ONE_HOUR, limit=24)

                if klines:
                    # Calculate trend and volatility metrics
                    prices = [kline.close for kline in klines]
                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

                    # Trend assessment (24h return)
                    trend_24h = (prices[-1] - prices[0]) / prices[0]

                    # Volatility assessment
                    volatility = sum(abs(r) for r in returns) / len(returns) if returns else 0

                    # Classify regime
                    if volatility > 0.05:  # 5% daily volatility
                        regime_scores[symbol] = MarketRegime.HIGH_VOLATILITY
                    elif abs(trend_24h) > 0.03:  # 3% trend
                        regime_scores[symbol] = MarketRegime.BULL_TREND if trend_24h > 0 else MarketRegime.BEAR_TREND
                    else:
                        regime_scores[symbol] = MarketRegime.SIDEWAYS

            # Determine overall market regime (majority vote)
            regime_counts = {}
            for regime in regime_scores.values():
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            overall_regime = max(regime_counts, key=regime_counts.get)
            return overall_regime

        except Exception as e:
            logger.error(f"Error assessing market regime: {e}")
            return MarketRegime.SIDEWAYS  # Safe default

    async def _make_trading_decisions(self, market_regime: MarketRegime) -> List[TradingDecision]:
        """Generate trading decisions based on current state and market conditions."""
        decisions = []

        try:
            for symbol in PRIORITY_SYMBOLS:
                # Skip if we already have too many positions
                if len(self.portfolio_state.active_positions) >= self.max_concurrent_positions:
                    break

                # Get current market data
                ticker = await self.data_feed.get_ticker(symbol)
                if not ticker:
                    continue

                # Check existing position
                existing_position = self.portfolio_state.active_positions.get(symbol)

                # Strategy-specific decision making
                if self.trading_mode in [TradingMode.GRID_TRADING, TradingMode.HYBRID]:
                    grid_decisions = await self.grid_strategy.make_decisions(
                        symbol, ticker, existing_position, market_regime
                    )
                    decisions.extend(grid_decisions)

                if self.trading_mode in [TradingMode.VOLATILITY_TRADING, TradingMode.HYBRID]:
                    vol_decisions = await self.volatility_strategy.make_decisions(
                        symbol, ticker, existing_position, market_regime
                    )
                    decisions.extend(vol_decisions)

        except Exception as e:
            logger.error(f"Error making trading decisions: {e}")

        return decisions

    async def _execute_decisions(self, decisions: List[TradingDecision]):
        """Execute trading decisions with risk management."""
        # Global risk guard: daily loss limit and exposure cap
        try:
            equity = self.portfolio_state.total_balance + self.portfolio_state.unrealized_pnl
            # Reset daily baseline at UTC midnight
            if datetime.utcnow().date() != self._last_pnl_reset_date:
                self._last_pnl_reset_date = datetime.utcnow().date()
                self.start_of_day_value = equity
                self.daily_pnl = 0.0

            if self.start_of_day_value == 0.0:
                self.start_of_day_value = equity

            self.daily_pnl = equity - self.start_of_day_value
            self.metrics.observe('daily_pnl', self.daily_pnl, {'mode': self.trading_mode.value})

            max_daily_loss_pct = getattr(self.settings, 'max_daily_loss', 0.15)
            if self.daily_pnl < -max_daily_loss_pct * self.start_of_day_value:
                logger.warning("Daily loss limit reached - pausing trading and triggering emergency stop")
                self.metrics.inc('kill_switch_daily_loss', 1)
                self.emergency_stop = True
                return

            # Exposure cap (portfolio-level)
            total_exposure = self.portfolio_state.total_positions_value
            max_portfolio_risk = getattr(self.settings, 'max_portfolio_risk', 0.1)
            if self.portfolio_state.total_balance > 0 and \
               total_exposure > max_portfolio_risk * self.portfolio_state.total_balance:
                logger.warning("Exposure limit exceeded - skipping new trades this cycle")
                self.metrics.inc('exposure_skipped', 1)
                return
        except Exception as risk_e:
            logger.error(f"Global risk guard failed: {risk_e}")

        for decision in decisions:
            try:
                # Risk assessment
                risk_check = await self.risk_manager.assess_trade_risk(decision, self.portfolio_state)

                if not risk_check['approved']:
                    logger.warning(f"Trade rejected by risk manager: {risk_check['reason']}")
                    continue

                # Execute the trade
                await self._execute_single_decision(decision)

                # Update position tracking
                await self._update_portfolio_state()

            except Exception as e:
                logger.error(f"Error executing decision {decision}: {e}")

    async def _execute_single_decision(self, decision: TradingDecision):
        """Execute a single trading decision."""
        try:
            # MEV/slippage guard: estimate execution price vs current to cap price impact
            current_ticker = await self.data_feed.get_ticker(decision.symbol)
            base_price = current_ticker['last'] if current_ticker and 'last' in current_ticker else decision.price or 0.0
            side = 'buy' if decision.action == 'BUY' else 'sell'
            est_price = await self.cost_model.estimate_execution_price(
                symbol=decision.symbol,
                base_price=base_price,
                quantity=abs(decision.quantity),
                side=side
            )
            price_impact_pct = abs(est_price - (base_price or est_price)) / (base_price or est_price or 1.0)
            max_price_impact = self.config.get('mev', {}).get('max_price_impact_pct', 0.08)
            if price_impact_pct > max_price_impact:
                self.metrics.inc('trade_rejected_price_impact', 1, {'symbol': decision.symbol})
                logger.warning(f"Trade rejected due to price impact {price_impact_pct:.2%} > {max_price_impact:.2%}")
                return

            # Correlation ID for tracing
            corr_id = f"corr-{uuid.uuid4().hex[:12]}"

            if decision.action == "BUY":
                await self.aster_client.place_order(
                    symbol=decision.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=decision.quantity,
                    new_client_order_id=corr_id
                )

            elif decision.action == "SELL":
                await self.aster_client.place_order(
                    symbol=decision.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=decision.quantity,
                    new_client_order_id=corr_id
                )

            elif decision.action == "ADJUST_GRID":
                # Grid-specific adjustments
                await self._adjust_grid_position(decision)

            logger.info(f"Executed {decision.action} order for {decision.symbol}: {decision.quantity} id={corr_id}")
            self.metrics.inc('trade_executed', 1, {'symbol': decision.symbol, 'action': decision.action, 'corr_id': corr_id})

        except Exception as e:
            logger.error(f"Failed to execute {decision.action} for {decision.symbol}: {e}")
            self.metrics.inc('trade_execute_error', 1, {'symbol': decision.symbol, 'action': decision.action})
            raise

    async def _adjust_grid_position(self, decision: TradingDecision):
        """Adjust grid trading positions."""
        # Implementation depends on grid strategy requirements
        pass

    async def _update_portfolio_state(self):
        """Update current portfolio state from Aster API."""
        try:
            # Get account balance
            balance = await self.aster_client.get_account_balance_v2()

            # Get positions
            positions = await self.aster_client.get_position_info_v2()

            # Calculate totals
            total_balance = sum(b.balance for b in balance if b.asset == 'USDT')
            available_balance = sum(b.available_balance for b in balance if b.asset == 'USDT')

            total_positions_value = sum(pos.position_amt * pos.entry_price for pos in positions)
            unrealized_pnl = sum(pos.unrealized_profit for pos in positions)

            # Update state
            self.portfolio_state = PortfolioState(
                timestamp=datetime.now(),
                total_balance=total_balance,
                available_balance=available_balance,
                total_positions_value=total_positions_value,
                unrealized_pnl=unrealized_pnl,
                active_positions={pos.symbol: pos for pos in positions},
                active_grids=self.portfolio_state.active_grids  # Preserve grid state
            )

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    async def _update_market_data(self):
        """Update market data cache."""
        try:
            await self.data_feed.update_cache()
        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    async def _subscribe_to_market_data(self):
        """Subscribe to real-time market data streams."""
        try:
            # Subscribe to ticker updates for priority symbols
            for symbol in PRIORITY_SYMBOLS:
                await self.aster_client.subscribe_ticker(symbol, self._handle_ticker_update)
                await self.aster_client.subscribe_trades(symbol, self._handle_trade_update)

            # Start WebSocket listening
            asyncio.create_task(self.aster_client.listen())

        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")

    async def _handle_ticker_update(self, data: Dict):
        """Handle real-time ticker updates."""
        try:
            # Update data feed cache
            self.data_feed.update_ticker(data)
        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")

    async def _handle_trade_update(self, data: Dict):
        """Handle real-time trade updates."""
        try:
            # Update trade data for strategy analysis
            self.data_feed.update_trades(data)
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def _validate_trading_setup(self):
        """Validate trading setup and configuration."""
        try:
            # Check API connectivity
            await self.aster_client.test_connectivity()

            # Check account balance
            balance = await self.aster_client.get_account_balance_v2()
            if not balance:
                raise ValueError("No account balance available")

            # Check available assets
            exchange_info = await self.aster_client.get_exchange_info()
            available_symbols = [s['symbol'] for s in exchange_info.get('symbols', [])]

            for symbol in PRIORITY_SYMBOLS:
                if symbol not in available_symbols:
                    logger.warning(f"Symbol {symbol} not available on Aster DEX")

            logger.info("Trading setup validation completed")

        except Exception as e:
            logger.error(f"Trading setup validation failed: {e}")
            raise

    async def _close_all_positions(self):
        """Emergency close all positions."""
        try:
            logger.warning("Emergency closing all positions...")

            for symbol, position in self.portfolio_state.active_positions.items():
                if position.position_amt != 0:
                    side = OrderSide.SELL if position.position_amt > 0 else OrderSide.BUY
                    quantity = abs(position.position_amt)

                    await self.aster_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        reduce_only=True
                    )

                    logger.info(f"Closed position: {symbol} {quantity}")

        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    async def _emergency_shutdown(self):
        """Emergency shutdown procedure."""
        logger.critical("Emergency shutdown initiated")
        self.emergency_stop = True

        try:
            await self._close_all_positions()
            await self.aster_client.disconnect()
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")

    def _update_performance_metrics(self):
        """Update trading performance metrics."""
        try:
            equity = self.portfolio_state.total_balance + self.portfolio_state.unrealized_pnl
            # Reset daily baseline at UTC midnight
            if datetime.utcnow().date() != self._last_pnl_reset_date:
                self._last_pnl_reset_date = datetime.utcnow().date()
                self.start_of_day_value = equity
                self.daily_pnl = 0.0

            if self.start_of_day_value == 0.0:
                self.start_of_day_value = equity

            self.daily_pnl = equity - self.start_of_day_value
            self.metrics.observe('equity', equity, {'mode': self.trading_mode.value})
            self.metrics.observe('daily_pnl', self.daily_pnl, {'mode': self.trading_mode.value})
        except Exception as e:
            logger.error(f"Performance metric update failed: {e}")

    async def _log_status_update(self, market_regime: MarketRegime):
        """Log periodic status updates."""
        if logger.isEnabledFor(logging.INFO):
            status = {
                'portfolio_value': self.portfolio_state.total_balance + self.portfolio_state.unrealized_pnl,
                'active_positions': len(self.portfolio_state.active_positions),
                'market_regime': market_regime.value,
                'total_trades': self.total_trades,
                'daily_pnl': self.daily_pnl,
                'uptime': str(datetime.now() - self.start_time)
            }
            logger.info(f"Status Update: {status}")

    # Public interface methods

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return {
            'total_balance': self.portfolio_state.total_balance,
            'available_balance': self.portfolio_state.available_balance,
            'total_positions_value': self.portfolio_state.total_positions_value,
            'unrealized_pnl': self.portfolio_state.unrealized_pnl,
            'active_positions': len(self.portfolio_state.active_positions),
            'active_grids': len(self.portfolio_state.active_grids),
            'is_running': self.is_running,
            'trading_mode': self.trading_mode.value
        }

    def set_emergency_stop(self, stop: bool = True):
        """Set emergency stop flag."""
        self.emergency_stop = stop
        if stop:
            logger.warning("Emergency stop activated")

    async def manual_trade(self, symbol: str, side: OrderSide, quantity: float,
                          order_type: OrderType = OrderType.MARKET, price: float = None):
        """Execute manual trade (for testing/debugging)."""
        if not self.config.get('allow_manual_trades', False):
            raise ValueError("Manual trades not allowed in current configuration")

        # Risk check
        decision = TradingDecision(
            timestamp=datetime.now(),
            symbol=symbol,
            action=side.value,
            quantity=quantity,
            price=price,
            reason="Manual trade",
            confidence=1.0,
            risk_score=0.5,
            expected_profit=None
        )

        risk_check = await self.risk_manager.assess_trade_risk(decision, self.portfolio_state)
        if not risk_check['approved']:
            raise ValueError(f"Manual trade rejected: {risk_check['reason']}")

        # Execute
        await self._execute_single_decision(decision)
        logger.info(f"Manual {side.value} executed: {symbol} {quantity}")


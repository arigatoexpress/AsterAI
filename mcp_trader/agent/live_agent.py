"""
Live Trading Agent for Rari Trade AI.
Executes trades on Aster DEX based on strategy signals.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from mcp_trader.execution.aster_client import AsterClient, OrderRequest, OrderResponse
from mcp_trader.execution.trade_executor import TradeExecutor
from mcp_trader.strategies.dmark_strategy import DMarkStrategy
from mcp_trader.strategies.rules import generate_positions_sma_crossover
from mcp_trader.models.base import TradingSignal
from mcp_trader.config import get_settings
from mcp_trader.logging_utils import setup_logger

logger = setup_logger(__name__)


class LiveTradingAgent:
    """
    Live trading agent that monitors market data and executes trades
    based on configured strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_type = config.get('strategy_type', 'sma_crossover')
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.position_size = config.get('position_size', 0.1)  # 10% of portfolio
        self.max_daily_trades = config.get('max_daily_trades', 5)
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()

        # Initialize clients
        self.aster_client = None
        self.trade_executor = None
        self.strategy = None

        # Trading state
        self.current_position = 0.0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        logger.info(f"LiveTradingAgent initialized for {self.symbol} with {self.strategy_type} strategy")

    async def initialize(self):
        """Initialize connections and strategies."""
        try:
            # Get API credentials
            settings = get_settings()
            api_key = settings.aster_api_key
            api_secret = settings.aster_api_secret

            if not api_key or not api_secret:
                raise ValueError("Aster API credentials not found. Run setup_credentials.py first.")

            # Initialize Aster client
            self.aster_client = AsterClient(api_key, api_secret)
            self.trade_executor = TradeExecutor(self.aster_client)

            # Initialize strategy
            if self.strategy_type == 'dmark':
                strategy_config = {
                    'dmark_config': {'mode': 'balanced'},
                    'max_daily_trades': self.max_daily_trades,
                    'min_confidence': 0.6,
                    'max_position_size': self.position_size,
                    'stop_loss_threshold': 0.02,
                    'take_profit_threshold': 0.04
                }
                self.strategy = DMarkStrategy(strategy_config)
            elif self.strategy_type == 'sma_crossover':
                # SMA crossover handled in generate_positions_sma_crossover
                pass
            else:
                raise ValueError(f"Unknown strategy type: {self.strategy_type}")

            logger.info("LiveTradingAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LiveTradingAgent: {e}")
            raise

    async def start_trading(self):
        """Start the live trading loop."""
        logger.info(f"Starting live trading for {self.symbol}")

        try:
            async with self.aster_client:
                # Get initial account info
                account_info = await self.aster_client.get_account_info()
                logger.info(f"Account balance: ${account_info.total_balance:.2f}")

                # Main trading loop
                while True:
                    try:
                        await self._trading_cycle()
                        await asyncio.sleep(60)  # Check every minute

                    except Exception as e:
                        logger.error(f"Error in trading cycle: {e}")
                        await asyncio.sleep(30)  # Wait before retry

        except Exception as e:
            logger.error(f"Critical error in live trading: {e}")
            raise

    async def _trading_cycle(self):
        """Execute one trading cycle."""
        # Reset daily counters if needed
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.last_reset_date = today

        # Check daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            logger.info(f"Daily trade limit ({self.max_daily_trades}) reached. Skipping cycle.")
            return

        # Get current market data
        market_data = await self.aster_client.get_market_data(self.symbol)
        logger.debug(f"Market data for {self.symbol}: {market_data}")

        # Get recent klines for strategy
        klines = await self.aster_client.get_klines(self.symbol, interval='1m', limit=100)
        if klines.empty:
            logger.warning("No klines data available")
            return

        # Generate trading signals
        signal = await self._generate_signal(klines)

        if signal:
            await self._execute_signal(signal)

    async def _generate_signal(self, klines) -> Optional[TradingSignal]:
        """Generate trading signal based on strategy."""
        try:
            close_prices = klines['close']

            if self.strategy_type == 'sma_crossover':
                # Simple SMA crossover
                short_win, long_win = 20, 50
                positions = generate_positions_sma_crossover(
                    pd.DataFrame({"close": close_prices}),
                    short_win, long_win
                )

                latest_position = positions.iloc[-1] if not positions.empty else 0

                if latest_position != self.current_position:
                    # Position change detected
                    if latest_position > 0 and self.current_position <= 0:
                        return TradingSignal(
                            timestamp=datetime.now(),
                            symbol=self.symbol,
                            signal=TradingSignal.BUY,
                            quantity=self.position_size,
                            confidence=0.7
                        )
                    elif latest_position < 0 and self.current_position >= 0:
                        return TradingSignal(
                            timestamp=datetime.now(),
                            symbol=self.symbol,
                            signal=TradingSignal.SELL,
                            quantity=self.position_size,
                            confidence=0.7
                        )

            elif self.strategy_type == 'dmark' and self.strategy:
                # Use DMark strategy
                signals = self.strategy.generate_signals(pd.DataFrame({
                    'timestamp': klines.index,
                    'open': klines['open'],
                    'high': klines['high'],
                    'low': klines['low'],
                    'close': klines['close'],
                    'volume': klines['volume'],
                    'symbol': self.symbol
                }))

                if signals:
                    return signals[-1]  # Return latest signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")

        return None

    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal."""
        try:
            logger.info(f"Executing signal: {signal.signal} {signal.quantity} {signal.symbol}")

            if signal.signal == TradingSignal.BUY:
                # Close any existing short position first
                if self.current_position < 0:
                    await self._close_position()

                # Open long position
                result = await self.trade_executor.market_buy(
                    symbol=signal.symbol,
                    quantity=signal.quantity
                )
                self.current_position = signal.quantity
                self.entry_price = result.get('price', 0)
                self._set_stop_loss_take_profit(self.entry_price, 'long')

            elif signal.signal == TradingSignal.SELL:
                # Close any existing long position first
                if self.current_position > 0:
                    await self._close_position()

                # Open short position
                result = await self.trade_executor.market_sell(
                    symbol=signal.symbol,
                    quantity=signal.quantity
                )
                self.current_position = -signal.quantity
                self.entry_price = result.get('price', 0)
                self._set_stop_loss_take_profit(self.entry_price, 'short')

            self.daily_trade_count += 1
            logger.info(f"Trade executed successfully. Daily count: {self.daily_trade_count}")

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")

    async def _close_position(self):
        """Close current position."""
        try:
            if self.current_position > 0:
                # Close long position
                await self.trade_executor.market_sell(
                    symbol=self.symbol,
                    quantity=abs(self.current_position)
                )
            elif self.current_position < 0:
                # Close short position
                await self.trade_executor.market_buy(
                    symbol=self.symbol,
                    quantity=abs(self.current_position)
                )

            logger.info("Position closed successfully")
            self.current_position = 0
            self.entry_price = None
            self.stop_loss_price = None
            self.take_profit_price = None

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def _set_stop_loss_take_profit(self, entry_price: float, position_type: str):
        """Set stop loss and take profit levels."""
        stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # 2%
        take_profit_pct = self.config.get('take_profit_pct', 0.04)  # 4%

        if position_type == 'long':
            self.stop_loss_price = entry_price * (1 - stop_loss_pct)
            self.take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # short
            self.stop_loss_price = entry_price * (1 + stop_loss_pct)
            self.take_profit_price = entry_price * (1 - take_profit_pct)

    async def stop_trading(self):
        """Stop trading and close all positions."""
        logger.info("Stopping live trading agent")

        try:
            if self.current_position != 0:
                await self._close_position()

            if self.aster_client:
                await self.aster_client.disconnect()

        except Exception as e:
            logger.error(f"Error stopping trading agent: {e}")

        logger.info("Trading agent stopped")


# Convenience functions for running the agent
async def run_live_agent(config: Dict[str, Any]):
    """Run the live trading agent with given configuration."""
    agent = LiveTradingAgent(config)

    try:
        await agent.initialize()
        await agent.start_trading()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Live agent failed: {e}")
    finally:
        await agent.stop_trading()


def start_live_trading(symbol: str = 'BTCUSDT',
                       strategy: str = 'sma_crossover',
                       position_size: float = 0.1):
    """Convenience function to start live trading."""
    config = {
        'symbol': symbol,
        'strategy_type': strategy,
        'position_size': position_size,
        'max_daily_trades': 5,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }

    logger.info(f"Starting live trading with config: {config}")
    asyncio.run(run_live_agent(config))


if __name__ == "__main__":
    # Example usage
    start_live_trading()

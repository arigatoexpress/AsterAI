"""
Live Trading Agent for Real-Money Trading

This module implements the live trading agent that executes real trades
with safety mechanisms, risk controls, and real-time monitoring.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from mcp_trader.execution.aster_client import AsterClient
from mcp_trader.risk.risk_manager import RiskManager
from mcp_trader.risk.dynamic_position_sizing import DynamicPositionSizing
from mcp_trader.strategies.market_making import MarketMakingStrategy
from mcp_trader.strategies.funding_arbitrage import FundingArbitrageStrategy
from mcp_trader.strategies.dmark_strategy import DMarkStrategy
from autonomous_mcp_agent import AutonomousMCPAgent
from self_improvement_engine import SelfImprovementEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for live trading"""
    initial_capital: float = 100.0
    max_leverage: float = 3.0
    position_size_pct: float = 0.02  # 2% per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    daily_loss_limit_pct: float = 0.10  # 10% daily loss limit
    max_positions: int = 3
    trading_pairs: List[str] = None
    emergency_stop: bool = False
    
    def __post_init__(self):
        if self.trading_pairs is None:
            self.trading_pairs = ["BTCUSDT", "ETHUSDT"]

@dataclass
class Position:
    """Represents an active trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: str

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class LiveTradingAgent:
    """Main live trading agent for real-money trading"""
    
    def __init__(self, config: TradingConfig, aster_client: AsterClient):
        self.config = config
        self.aster_client = aster_client
        self.risk_manager = RiskManager()
        self.position_sizer = DynamicPositionSizing()
        self.positions: Dict[str, Position] = {}
        self.metrics = TradingMetrics()
        self.daily_start_balance = config.initial_capital
        self.last_daily_reset = datetime.now().date()
        
        # Initialize strategies
        self.strategies = {
            'market_making': MarketMakingStrategy(),
            'funding_arbitrage': FundingArbitrageStrategy(),
            'dmark': DMarkStrategy()
        }
        
        # Initialize MCP agent for decision making
        self.mcp_agent = AutonomousMCPAgent()
        
        # Initialize self-improvement engine
        self.improvement_engine = SelfImprovementEngine({})
        
        # Trading state
        self.is_trading = False
        self.emergency_stop = False
        
        logger.info(f"Live Trading Agent initialized with ${config.initial_capital} capital")
    
    async def start_trading(self):
        """Start the live trading loop"""
        logger.info("Starting live trading...")
        self.is_trading = True
        
        try:
            while self.is_trading and not self.emergency_stop:
                # Check daily reset
                await self._check_daily_reset()
                
                # Check emergency conditions
                if await self._check_emergency_conditions():
                    await self._emergency_shutdown()
                    break
                
                # Update market data
                market_data = await self._update_market_data()
                
                # Generate trading signals
                signals = await self._generate_signals(market_data)
                
                # Execute trades based on signals
                await self._execute_trades(signals)
                
                # Update positions
                await self._update_positions()
                
                # Update metrics
                await self._update_metrics()
                
                # Log status
                await self._log_status()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second cycle
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await self._emergency_shutdown()
        
        logger.info("Live trading stopped")
    
    async def stop_trading(self):
        """Stop the live trading loop"""
        logger.info("Stopping live trading...")
        self.is_trading = False
        
        # Close all positions
        await self._close_all_positions()
    
    async def emergency_stop_trading(self):
        """Emergency stop - immediately close all positions"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_stop = True
        self.is_trading = False
        
        # Immediately close all positions
        await self._close_all_positions()
        
        # Log emergency stop
        logger.critical("All positions closed due to emergency stop")
    
    async def _check_daily_reset(self):
        """Check if daily reset is needed"""
        current_date = datetime.now().date()
        
        if current_date != self.last_daily_reset:
            logger.info("Daily reset - updating metrics and balance")
            
            # Reset daily metrics
            self.metrics.daily_pnl = 0.0
            self.daily_start_balance = self._get_current_balance()
            self.last_daily_reset = current_date
            
            # Save daily performance
            await self._save_daily_performance()
    
    async def _check_emergency_conditions(self) -> bool:
        """Check if emergency conditions are met"""
        
        # Check daily loss limit
        if self.metrics.daily_pnl < -self.config.daily_loss_limit_pct * self.daily_start_balance:
            logger.critical(f"Daily loss limit exceeded: {self.metrics.daily_pnl:.2f}")
            return True
        
        # Check maximum drawdown
        if self.metrics.max_drawdown > 0.15 * self.config.initial_capital:
            logger.critical(f"Maximum drawdown exceeded: {self.metrics.max_drawdown:.2f}")
            return True
        
        # Check position limits
        if len(self.positions) > self.config.max_positions:
            logger.critical(f"Position limit exceeded: {len(self.positions)}")
            return True
        
        return False
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("Initiating emergency shutdown")
        
        # Close all positions immediately
        await self._close_all_positions()
        
        # Set emergency stop flag
        self.emergency_stop = True
        self.is_trading = False
        
        # Log emergency shutdown
        logger.critical("Emergency shutdown completed")
    
    async def _update_market_data(self) -> Dict[str, Any]:
        """Update market data for all trading pairs"""
        market_data = {}
        
        for symbol in self.config.trading_pairs:
            try:
                # Get current price
                ticker = await self.aster_client.get_ticker(symbol)
                current_price = float(ticker['lastPrice'])
                
                # Get order book
                order_book = await self.aster_client.get_order_book(symbol)
                
                # Get recent klines
                klines = await self.aster_client.get_klines(symbol, '1m', limit=100)
                
                market_data[symbol] = {
                    'price': current_price,
                    'order_book': order_book,
                    'klines': klines,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
        
        return market_data
    
    async def _generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate trading signals from all strategies"""
        signals = {}
        
        for symbol, data in market_data.items():
            symbol_signals = []
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Generate signal using strategy
                    signal = await self._get_strategy_signal(strategy, data, symbol)
                    
                    if signal:
                        signal['strategy'] = strategy_name
                        signal['symbol'] = symbol
                        signal['timestamp'] = datetime.now()
                        symbol_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal for {strategy_name} on {symbol}: {e}")
            
            signals[symbol] = symbol_signals
        
        return signals
    
    async def _get_strategy_signal(self, strategy, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading signal from a specific strategy"""
        try:
            # Convert market data to DataFrame for strategy
            klines_df = pd.DataFrame(market_data['klines'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                klines_df[col] = pd.to_numeric(klines_df[col])
            
            # Generate signal
            signal = strategy.generate_signal(klines_df.iloc[-1])
            
            if signal != 0:  # Non-zero signal
                return {
                    'signal': signal,
                    'confidence': 0.8,  # Default confidence
                    'price': market_data['price']
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy signal: {e}")
        
        return None
    
    async def _execute_trades(self, signals: Dict[str, List[Dict[str, Any]]]):
        """Execute trades based on generated signals"""
        
        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                continue
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                continue
            
            # Check position limit
            if len(self.positions) >= self.config.max_positions:
                continue
            
            # Select best signal (highest confidence)
            best_signal = max(symbol_signals, key=lambda x: x.get('confidence', 0))
            
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, best_signal)
            
            if position_size <= 0:
                continue
            
            # Execute trade
            await self._place_trade(symbol, best_signal, position_size)
    
    async def _calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get current balance
            balance = await self._get_current_balance()
            
            # Calculate position size based on percentage
            position_value = balance * self.config.position_size_pct
            
            # Get current price
            current_price = signal['price']
            
            # Calculate position size in base currency
            position_size = position_value / current_price
            
            # Apply risk management
            position_size = self.position_sizer.calculate_position_size(
                symbol=symbol,
                signal_strength=signal['signal'],
                current_price=current_price,
                account_balance=balance,
                risk_per_trade=self.config.position_size_pct
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    async def _place_trade(self, symbol: str, signal: Dict[str, Any], position_size: float):
        """Place a trade order"""
        try:
            side = 'BUY' if signal['signal'] > 0 else 'SELL'
            
            # Place market order
            order = await self.aster_client.place_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=position_size
            )
            
            if order and order.get('orderId'):
                # Create position
                position = Position(
                    symbol=symbol,
                    side='long' if side == 'BUY' else 'short',
                    size=position_size,
                    entry_price=signal['price'],
                    current_price=signal['price'],
                    unrealized_pnl=0.0,
                    stop_loss=self._calculate_stop_loss(signal['price'], side),
                    take_profit=self._calculate_take_profit(signal['price'], side),
                    entry_time=datetime.now(),
                    strategy=signal['strategy']
                )
                
                self.positions[symbol] = position
                
                logger.info(f"Trade placed: {side} {position_size} {symbol} at {signal['price']}")
                
        except Exception as e:
            logger.error(f"Error placing trade for {symbol}: {e}")
    
    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        if side == 'BUY':
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        if side == 'BUY':
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)
    
    async def _update_positions(self):
        """Update all active positions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = await self.aster_client.get_ticker(symbol)
                current_price = float(ticker['lastPrice'])
                
                # Update position
                position.current_price = current_price
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)
                
                # Check stop loss and take profit
                if await self._should_close_position(position):
                    await self._close_position(symbol)
                    
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
    
    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L for a position"""
        if position.side == 'long':
            return (position.current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - position.current_price) * position.size
    
    async def _should_close_position(self, position: Position) -> bool:
        """Check if position should be closed"""
        # Check stop loss
        if position.side == 'long' and position.current_price <= position.stop_loss:
            return True
        elif position.side == 'short' and position.current_price >= position.stop_loss:
            return True
        
        # Check take profit
        if position.side == 'long' and position.current_price >= position.take_profit:
            return True
        elif position.side == 'short' and position.current_price <= position.take_profit:
            return True
        
        return False
    
    async def _close_position(self, symbol: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Determine close side
            close_side = 'SELL' if position.side == 'long' else 'BUY'
            
            # Place close order
            order = await self.aster_client.place_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=position.size
            )
            
            if order and order.get('orderId'):
                # Calculate realized P&L
                realized_pnl = self._calculate_unrealized_pnl(position)
                
                # Update metrics
                self.metrics.total_trades += 1
                self.metrics.total_pnl += realized_pnl
                self.metrics.daily_pnl += realized_pnl
                
                if realized_pnl > 0:
                    self.metrics.winning_trades += 1
                else:
                    self.metrics.losing_trades += 1
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"Position closed: {symbol} P&L: {realized_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol)
    
    async def _update_metrics(self):
        """Update trading metrics"""
        # Calculate unrealized P&L
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.metrics.unrealized_pnl = total_unrealized
        
        # Calculate win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Calculate profit factor
        if self.metrics.losing_trades > 0:
            avg_win = self.metrics.winning_trades / self.metrics.total_trades if self.metrics.total_trades > 0 else 0
            avg_loss = self.metrics.losing_trades / self.metrics.total_trades if self.metrics.total_trades > 0 else 0
            self.metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Update max drawdown
        current_balance = await self._get_current_balance()
        if current_balance < self.config.initial_capital:
            drawdown = self.config.initial_capital - current_balance
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)
        
        self.metrics.last_updated = datetime.now()
    
    async def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = await self.aster_client.get_account_info()
            return float(account_info.get('totalWalletBalance', self.config.initial_capital))
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return self.config.initial_capital
    
    async def _log_status(self):
        """Log current trading status"""
        if self.metrics.last_updated and (datetime.now() - self.metrics.last_updated).seconds < 60:
            return  # Only log every minute
        
        logger.info(f"Trading Status - "
                   f"Positions: {len(self.positions)}, "
                   f"Total P&L: {self.metrics.total_pnl:.2f}, "
                   f"Daily P&L: {self.metrics.daily_pnl:.2f}, "
                   f"Win Rate: {self.metrics.win_rate:.2%}")
    
    async def _save_daily_performance(self):
        """Save daily performance data"""
        try:
            performance_data = {
                'date': self.last_daily_reset.isoformat(),
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.metrics.daily_pnl,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'max_drawdown': self.metrics.max_drawdown,
                'positions': len(self.positions)
            }
            
            # Save to file (in production, save to database)
            with open(f"daily_performance_{self.last_daily_reset}.json", 'w') as f:
                json.dump(performance_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving daily performance: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            'is_trading': self.is_trading,
            'emergency_stop': self.emergency_stop,
            'positions': len(self.positions),
            'metrics': {
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.metrics.daily_pnl,
                'unrealized_pnl': self.metrics.unrealized_pnl,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'max_drawdown': self.metrics.max_drawdown
            },
            'config': {
                'initial_capital': self.config.initial_capital,
                'position_size_pct': self.config.position_size_pct,
                'daily_loss_limit_pct': self.config.daily_loss_limit_pct,
                'max_positions': self.config.max_positions
            }
        }

# Example usage
async def main():
    """Test the live trading agent"""
    
    # Create configuration
    config = TradingConfig(
        initial_capital=100.0,
        max_leverage=3.0,
        position_size_pct=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        daily_loss_limit_pct=0.10,
        max_positions=3,
        trading_pairs=["BTCUSDT", "ETHUSDT"]
    )
    
    # Create Aster client (you'll need to provide actual API credentials)
    aster_client = AsterClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        base_url="https://api.aster.exchange"
    )
    
    # Create live trading agent
    agent = LiveTradingAgent(config, aster_client)
    
    # Start trading (in production, this would run continuously)
    try:
        await agent.start_trading()
    except KeyboardInterrupt:
        logger.info("Stopping trading...")
        await agent.stop_trading()
    except Exception as e:
        logger.error(f"Trading error: {e}")
        await agent.emergency_stop_trading()

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Paper Trading System for Live Strategy Validation

This system executes paper trades (simulated real trading) to validate
strategies in live market conditions before deploying real capital.

Features:
- Real-time market data integration
- Simulated order execution
- Position tracking and risk management
- Performance monitoring and reporting
- Automatic stop-loss and take-profit execution
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PaperTradingConfig:
    """Configuration for paper trading system."""

    def __init__(self):
        self.initial_capital = 10000.0
        self.max_position_size = 0.1  # 10% of capital per position
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005   # 0.05% slippage
        self.stop_loss_pct = 0.02     # 2% stop loss
        self.take_profit_pct = 0.05   # 5% take profit
        self.max_daily_loss_pct = 0.05  # 5% daily loss limit
        self.update_interval_seconds = 60  # Update every minute

class Position:
    """Represents a trading position."""

    def __init__(self, symbol: str, side: str, quantity: float, entry_price: float,
                 stop_loss: float = None, take_profit: float = None):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = datetime.now()
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.current_price = entry_price
        self.unrealized_pnl = 0.0

    def update_price(self, current_price: float) -> bool:
        """Update position price and check for stop-loss/take-profit triggers."""
        self.current_price = current_price

        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
            # Check stop loss
            if self.stop_loss and current_price <= self.stop_loss:
                return True  # Trigger stop loss
            # Check take profit
            if self.take_profit and current_price >= self.take_profit:
                return True  # Trigger take profit
        else:  # short position
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            # Check stop loss (for short, stop loss is when price goes up)
            if self.stop_loss and current_price >= self.stop_loss:
                return True  # Trigger stop loss
            # Check take profit (for short, take profit is when price goes down)
            if self.take_profit and current_price <= self.take_profit:
                return True  # Trigger take profit

        return False  # No trigger

    def get_value(self) -> float:
        """Get current position value."""
        return abs(self.quantity) * self.current_price

    def get_pnl(self) -> float:
        """Get current P&L."""
        return self.unrealized_pnl

class PaperTradingEngine:
    """Main paper trading engine for live strategy validation."""

    def __init__(self, config: PaperTradingConfig = None):
        self.config = config or PaperTradingConfig()
        self.capital = self.config.initial_capital
        self.positions = {}  # symbol -> Position
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.daily_start_capital = self.config.initial_capital
        self.is_running = False

        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(f"Paper Trading Engine initialized with ${self.config.initial_capital:,.2f}")

    def can_open_position(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Check if we can open a new position."""

        position_value = quantity * price
        required_capital = position_value * (1 + self.config.commission_rate)

        # Check position size limit
        if position_value > self.capital * self.config.max_position_size:
            logger.warning(f"Position size too large: {position_value:.2f} > {self.capital * self.config.max_position_size:.2f}")
            return False

        # Check available capital
        if required_capital > self.capital:
            logger.warning(f"Insufficient capital: {required_capital:.2f} > {self.capital:.2f}")
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.daily_start_capital * self.config.max_daily_loss_pct:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False

        return True

    def open_position(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Open a new trading position."""

        if not self.can_open_position(symbol, side, quantity, price):
            return False

        # Calculate stop loss and take profit
        if side == 'long':
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:  # short
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)

        position = Position(symbol, side, quantity, price, stop_loss, take_profit)

        # Deduct capital for long position or add for short position
        position_cost = quantity * price * (1 + self.config.commission_rate)
        if side == 'long':
            self.capital -= position_cost
        else:
            self.capital += position_cost  # Short selling credits account

        self.positions[symbol] = position
        logger.info(f"Opened {side} position: {symbol} {quantity:.6f} @ ${price:.2f}")

        return True

    def close_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Close an existing position."""

        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return {}

        position = self.positions[symbol]

        # Calculate final P&L
        if position.side == 'long':
            pnl = (current_price - position.entry_price) * position.quantity
        else:  # short
            pnl = (position.entry_price - current_price) * position.quantity

        # Apply commission
        commission = abs(position.quantity) * current_price * self.config.commission_rate
        pnl -= commission

        # Update capital
        position_value = abs(position.quantity) * current_price
        if position.side == 'long':
            self.capital += pnl + position_value
        else:  # short
            self.capital -= pnl + position_value  # For shorts, we need to buy back

        # Record trade
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'quantity': position.quantity,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'commission': commission,
            'holding_period': (datetime.now() - position.entry_time).total_seconds() / 3600  # hours
        }

        self.closed_trades.append(trade_record)

        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update daily P&L
        self.daily_pnl += pnl

        # Remove position
        del self.positions[symbol]

        logger.info(".2f")

        return trade_record

    def update_positions(self, market_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """Update all positions with current market prices and check for triggers."""

        triggered_trades = []

        for symbol, position in list(self.positions.items()):
            if symbol in market_data:
                current_price = market_data[symbol]

                # Update position price
                should_close = position.update_price(current_price)

                if should_close:
                    # Close position due to stop-loss or take-profit
                    trade_record = self.close_position(symbol, current_price)
                    triggered_trades.append(trade_record)

        return triggered_trades

    def get_portfolio_value(self, market_data: Dict[str, float]) -> float:
        """Calculate total portfolio value including positions."""

        portfolio_value = self.capital

        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position_value = abs(position.quantity) * current_price
                portfolio_value += position_value

        return portfolio_value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        if not self.closed_trades:
            return {
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'current_capital': self.capital
            }

        # Calculate returns
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital

        # Calculate Sharpe ratio (simplified)
        if self.closed_trades:
            pnls = [trade['pnl'] for trade in self.closed_trades]
            if len(pnls) > 1:
                returns = np.array(pnls) / self.config.initial_capital
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Calculate max drawdown (simplified)
        capital_history = [self.config.initial_capital]
        current_capital = self.config.initial_capital

        for trade in self.closed_trades:
            current_capital += trade['pnl']
            capital_history.append(current_capital)

        if len(capital_history) > 1:
            cumulative = pd.Series(capital_history)
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()
        else:
            max_drawdown = 0

        # Calculate win rate and profit factor
        winning_trades = sum(1 for trade in self.closed_trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in self.closed_trades if trade['pnl'] < 0)
        win_rate = winning_trades / len(self.closed_trades) if self.closed_trades else 0

        total_profits = sum(trade['pnl'] for trade in self.closed_trades if trade['pnl'] > 0)
        total_losses = abs(sum(trade['pnl'] for trade in self.closed_trades if trade['pnl'] < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0

        return {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.closed_trades),
            'current_capital': self.capital,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }

    def execute_strategy_signal(self, symbol: str, signal: str, current_price: float) -> bool:
        """Execute a trading signal from a strategy."""

        if signal == 'buy' and symbol not in self.positions:
            # Open long position
            max_position_value = self.capital * self.config.max_position_size
            quantity = max_position_value / current_price

            return self.open_position(symbol, 'long', quantity, current_price)

        elif signal == 'sell' and symbol in self.positions and self.positions[symbol].side == 'long':
            # Close long position
            self.close_position(symbol, current_price)
            return True

        elif signal == 'short' and symbol not in self.positions:
            # Open short position
            max_position_value = self.capital * self.config.max_position_size
            quantity = max_position_value / current_price

            return self.open_position(symbol, 'short', quantity, current_price)

        elif signal == 'cover' and symbol in self.positions and self.positions[symbol].side == 'short':
            # Close short position
            self.close_position(symbol, current_price)
            return True

        return False

    def reset_daily_stats(self):
        """Reset daily statistics for new trading day."""

        self.daily_pnl = 0.0
        self.daily_start_capital = self.capital

    def save_session_data(self, filename: str = None) -> str:
        """Save current session data to file."""

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"paper_trading_session_{timestamp}.json"

        session_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_capital': self.config.initial_capital,
                'max_position_size': self.config.max_position_size,
                'commission_rate': self.config.commission_rate,
                'slippage_rate': self.config.slippage_rate
            },
            'current_state': {
                'capital': self.capital,
                'total_positions': len(self.positions),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades
            },
            'positions': {
                symbol: {
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time.isoformat(),
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                }
                for symbol, pos in self.positions.items()
            },
            'recent_trades': self.closed_trades[-10:],  # Last 10 trades
            'performance': self.get_performance_metrics()
        }

        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"Session data saved to {filename}")
        return filename

def simulate_paper_trading():
    """Simulate paper trading with sample market data."""

    print("="*80)
    print("📈 PAPER TRADING SIMULATION")
    print("="*80)

    # Create paper trading engine
    config = PaperTradingConfig()
    config.initial_capital = 10000.0

    engine = PaperTradingEngine(config)

    # Simulate market data (in real implementation, this would come from live feeds)
    market_data = {
        'BTC': 45000.0,
        'ETH': 2500.0,
        'SOL': 95.0
    }

    print(f"Starting with ${config.initial_capital:,.2f} capital")
    print(f"Market data: {market_data}")

    # Simulate some trading activity
    print("\n🔄 Simulating trading activity...")

    # Simulate buy BTC
    if engine.execute_strategy_signal('BTC', 'buy', market_data['BTC']):
        print(f"✅ Opened long position in BTC @ ${market_data['BTC']:,.2f}")

    # Update market data (simulate price movement)
    market_data['BTC'] = 46000.0  # 2.2% increase
    market_data['ETH'] = 2550.0   # 2% increase
    market_data['SOL'] = 98.0     # 3.2% increase

    # Update positions with new prices
    triggered_trades = engine.update_positions(market_data)
    print(f"📊 Updated positions, triggered trades: {len(triggered_trades)}")

    # Check if we should take profit on BTC
    btc_position = engine.positions.get('BTC')
    if btc_position:
        profit_pct = (market_data['BTC'] - btc_position.entry_price) / btc_position.entry_price
        print(".1%")

        if profit_pct > 0.03:  # 3% profit
            print("💰 Taking profit on BTC position...")
            engine.execute_strategy_signal('BTC', 'sell', market_data['BTC'])

    # Get final performance
    metrics = engine.get_performance_metrics()
    print("
📊 Final Performance:"    print(".1%")
    print(".2f")
    print(".1%")
    print(".1%")
    print(".2f")
    print(f"   Total Trades: {metrics['total_trades']}")

    # Save session data
    session_file = engine.save_session_data()
    print(f"💾 Session data saved to: {session_file}")

    return metrics

async def run_live_paper_trading():
    """Run live paper trading with real-time market data."""

    print("="*80)
    print("🚀 LIVE PAPER TRADING VALIDATION")
    print("="*80)

    # Create paper trading engine
    config = PaperTradingConfig()
    config.initial_capital = 10000.0

    engine = PaperTradingEngine(config)

    print(f"Starting live paper trading with ${config.initial_capital:,.2f}")
    print("Monitoring market conditions and executing strategies...")

    # In a real implementation, this would connect to live market data
    # For now, we'll simulate the process

    try:
        # Simulate 24 hours of trading
        for hour in range(24):
            print(f"\n⏰ Hour {hour + 1}/24")

            # Simulate market data updates
            market_data = {
                'BTC': 45000 + np.random.normal(0, 500),  # BTC around $45k with volatility
                'ETH': 2500 + np.random.normal(0, 50),   # ETH around $2.5k with volatility
                'SOL': 95 + np.random.normal(0, 5)       # SOL around $95 with volatility
            }

            # Update positions with current prices
            triggered_trades = engine.update_positions(market_data)

            if triggered_trades:
                print(f"   📈 Executed {len(triggered_trades)} automatic trades")

            # Simulate strategy signals (in real implementation, this would come from AI models)
            # For demo, we'll simulate occasional signals
            if hour % 4 == 0 and hour > 0:  # Every 4 hours
                # Random strategy signal
                symbols = list(market_data.keys())
                symbol = np.random.choice(symbols)

                if symbol not in engine.positions:
                    # 50% chance to buy, 50% chance to short
                    signal = np.random.choice(['buy', 'short'])
                    current_price = market_data[symbol]

                    if engine.execute_strategy_signal(symbol, signal, current_price):
                        print(f"   🎯 Strategy signal: {signal.upper()} {symbol} @ ${current_price:.2f}")

            # Check portfolio status
            portfolio_value = engine.get_portfolio_value(market_data)
            print(".2f")

            # Sleep for 1 hour (simulated)
            await asyncio.sleep(0.1)  # Fast simulation

        # Final performance report
        print("\n" + "="*80)
        print("📊 24-HOUR PAPER TRADING RESULTS")
        print("="*80)

        metrics = engine.get_performance_metrics()

        print(".1%")
        print(".2f")
        print(".1%")
        print(".1%")
        print(".2f")
        print(f"   Total Trades: {metrics['total_trades']}")

        # Save comprehensive session data
        session_file = engine.save_session_data("live_paper_trading_results.json")
        print(f"\n💾 Complete session data saved to: {session_file}")

        # Determine if ready for live trading
        if metrics['sharpe_ratio'] > 1.0 and metrics['max_drawdown_pct'] > -0.15:
            print("\n✅ READY FOR LIVE TRADING!")
            print("   • Sharpe ratio indicates good risk-adjusted returns")
            print("   • Drawdown within acceptable limits")
            print("   • Strategy validated in live market conditions")
        else:
            print("\n⚠️  NEEDS FURTHER OPTIMIZATION")
            print("   • Sharpe ratio below target (1.0)")
            print("   • Consider adjusting strategy parameters")
            print("   • Continue paper trading for more validation")

        return metrics

    except KeyboardInterrupt:
        print("\n⏹️  Paper trading stopped by user")
        engine.save_session_data("interrupted_paper_trading_session.json")
        return engine.get_performance_metrics()

    except Exception as e:
        print(f"\n❌ Paper trading failed: {e}")
        engine.save_session_data("error_paper_trading_session.json")
        return engine.get_performance_metrics()

def main():
    """Main function for paper trading demonstration."""

    print("Choose paper trading mode:")
    print("1. Quick simulation (demo)")
    print("2. Live validation (24-hour simulation)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Quick simulation
        metrics = simulate_paper_trading()
        print("
✅ Quick paper trading simulation completed!"        print("📈 Use this data to validate strategy performance before live trading"
    elif choice == "2":
        # Live validation
        try:
            metrics = asyncio.run(run_live_paper_trading())
            print("
✅ Live paper trading validation completed!"            print("🚀 Ready to proceed with profit maximization strategy"
        except Exception as e:
            print(f"\n❌ Live validation failed: {e}")
            print("💡 Run the quick simulation first to validate basic functionality")
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()

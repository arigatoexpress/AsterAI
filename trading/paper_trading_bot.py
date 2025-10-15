#!/usr/bin/env python3
"""
Paper Trading Bot for Aster DEX
Tests trading infrastructure with simulated trades (no real money).
"""

import sys
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.baseline_strategy import BaselineMomentumStrategy
from local_training.aster_dex_data_collector import AsterDEXDataCollector
from mcp_trader.data.api_manager import APIKeyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    signal_strength: float
    reason: str


@dataclass
class PaperTrade:
    """Represents a completed paper trade"""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    reason: str


class PaperTradingBot:
    """
    Paper trading bot that simulates trades on Aster DEX.
    No real money - just testing infrastructure and strategy.
    """
    
    def __init__(self, initial_capital: float = 10000, symbols: List[str] = None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Trading components
        self.strategy = BaselineMomentumStrategy()
        self.data_collector = None
        self.api_manager = APIKeyManager()
        
        # Portfolio tracking
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        
        # Performance tracking
        self.equity_curve = []
        self.start_time = datetime.now()
        
        # Output directory
        self.output_dir = Path("trading/paper_trading_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Paper Trading Bot initialized with ${initial_capital:,.2f}")
    
    async def initialize(self):
        """Initialize data collector and API connections"""
        logger.info("Initializing Aster DEX connection...")
        
        # Load API credentials
        self.api_manager.load_credentials()
        
        # Initialize data collector
        self.data_collector = AsterDEXDataCollector()
        await self.data_collector.initialize()
        
        logger.info("✅ Paper trading bot ready")
    
    async def get_latest_data(self, symbol: str, lookback_hours: int = 200) -> pd.DataFrame:
        """Get latest market data for a symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            df = await self.data_collector.collect_historical_data(
                symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            
            if df is not None and not df.empty:
                return df
            else:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, signal_strength: float, price: float) -> float:
        """Calculate position size based on available capital and signal"""
        # Use 10% of capital per trade, adjusted by signal strength
        max_position_value = self.capital * 0.1 * signal_strength
        quantity = max_position_value / price
        return quantity
    
    def open_position(self, symbol: str, signal: Dict, current_price: float):
        """Open a new paper position"""
        if symbol in self.positions:
            logger.info(f"Already have position in {symbol}, skipping")
            return
        
        # Calculate position size
        quantity = self.calculate_position_size(signal['signal_strength'], current_price)
        position_value = quantity * current_price
        
        if position_value > self.capital:
            logger.warning(f"Insufficient capital for {symbol} trade")
            return
        
        # Calculate stop loss and take profit
        stop_loss = current_price * 0.98  # 2% stop loss
        take_profit = current_price * 1.04  # 4% take profit
        
        # Create position
        position = PaperPosition(
            symbol=symbol,
            entry_price=current_price,
            quantity=quantity,
            entry_time=datetime.now(),
            side='long',
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=signal['signal_strength'],
            reason=signal['reason']
        )
        
        self.positions[symbol] = position
        self.capital -= position_value
        
        logger.info(f"📈 OPENED LONG: {symbol} @ ${current_price:,.2f} | "
                   f"Qty: {quantity:.6f} | Value: ${position_value:,.2f} | "
                   f"Reason: {signal['reason']}")
    
    def close_position(self, symbol: str, current_price: float, reason: str):
        """Close an existing paper position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L
        exit_value = position.quantity * current_price
        entry_value = position.quantity * position.entry_price
        pnl = exit_value - entry_value
        pnl_pct = (current_price / position.entry_price - 1) * 100
        
        # Update capital
        self.capital += exit_value
        
        # Record trade
        trade = PaperTrade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=current_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason
        )
        
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"{pnl_emoji} CLOSED: {symbol} @ ${current_price:,.2f} | "
                   f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
    
    def check_exit_conditions(self, symbol: str, current_price: float):
        """Check if position should be closed"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Check stop loss
        if current_price <= position.stop_loss:
            self.close_position(symbol, current_price, "STOP_LOSS")
            return
        
        # Check take profit
        if current_price >= position.take_profit:
            self.close_position(symbol, current_price, "TAKE_PROFIT")
            return
    
    async def trading_cycle(self):
        """Execute one trading cycle for all symbols"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        for symbol in self.symbols:
            try:
                # Get latest data
                df = await self.get_latest_data(symbol)
                
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                current_price = float(df['close'].iloc[-1])
                
                # Check exit conditions for existing positions
                self.check_exit_conditions(symbol, current_price)
                
                # Get signal from strategy
                signal = self.strategy.get_latest_signal(df)
                
                logger.info(f"{symbol}: Price=${current_price:,.2f} | "
                           f"Signal={'BUY' if signal['signal'] == 1 else 'SELL' if signal['signal'] == -1 else 'HOLD'} | "
                           f"Strength={signal['signal_strength']:.2f}")
                
                # Execute trades based on signals
                if signal['signal'] == 1 and symbol not in self.positions:
                    # Buy signal and no position
                    self.open_position(symbol, signal, current_price)
                
                elif signal['signal'] == -1 and symbol in self.positions:
                    # Sell signal and have position
                    self.close_position(symbol, current_price, "SELL_SIGNAL")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update equity curve
        total_equity = self.capital + sum(
            pos.quantity * df['close'].iloc[-1] 
            for symbol, pos in self.positions.items()
            if not df.empty
        )
        
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'capital': self.capital,
            'total_equity': total_equity,
            'open_positions': len(self.positions)
        })
        
        # Print status
        self.print_status()
    
    def print_status(self):
        """Print current portfolio status"""
        total_position_value = sum(
            pos.quantity * pos.entry_price  # Using entry price for now
            for pos in self.positions.values()
        )
        total_equity = self.capital + total_position_value
        total_return = (total_equity / self.initial_capital - 1) * 100
        
        logger.info(f"\n📊 Portfolio Status:")
        logger.info(f"   Cash: ${self.capital:,.2f}")
        logger.info(f"   Positions Value: ${total_position_value:,.2f}")
        logger.info(f"   Total Equity: ${total_equity:,.2f}")
        logger.info(f"   Total Return: {total_return:+.2f}%")
        logger.info(f"   Open Positions: {len(self.positions)}")
        logger.info(f"   Closed Trades: {len(self.closed_trades)}")
        
        if self.closed_trades:
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(self.closed_trades) * 100
            avg_pnl = np.mean([t.pnl for t in self.closed_trades])
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Avg P&L: ${avg_pnl:,.2f}")
    
    def save_results(self):
        """Save trading results to file"""
        results = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': (self.capital / self.initial_capital - 1) * 100,
            'total_trades': len(self.closed_trades),
            'open_positions': len(self.positions),
            'closed_trades': [asdict(t) for t in self.closed_trades],
            'equity_curve': self.equity_curve
        }
        
        # Save to JSON
        output_file = self.output_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {output_file}")
    
    async def run(self, cycles: int = 10, interval_minutes: int = 60):
        """Run paper trading bot for specified cycles"""
        logger.info(f"\n{'='*60}")
        logger.info("🤖 PAPER TRADING BOT STARTED")
        logger.info(f"{'='*60}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Cycles: {cycles}")
        logger.info(f"Interval: {interval_minutes} minutes")
        logger.info(f"{'='*60}\n")
        
        try:
            for cycle in range(cycles):
                logger.info(f"\n🔄 Cycle {cycle + 1}/{cycles}")
                await self.trading_cycle()
                
                if cycle < cycles - 1:
                    logger.info(f"⏳ Waiting {interval_minutes} minutes until next cycle...")
                    await asyncio.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user")
        
        finally:
            # Close all positions
            logger.info("\n🔒 Closing all open positions...")
            for symbol in list(self.positions.keys()):
                df = await self.get_latest_data(symbol, lookback_hours=1)
                if not df.empty:
                    current_price = float(df['close'].iloc[-1])
                    self.close_position(symbol, current_price, "FINAL_CLOSE")
            
            # Save results
            self.save_results()
            
            # Final summary
            logger.info(f"\n{'='*60}")
            logger.info("📊 FINAL SUMMARY")
            logger.info(f"{'='*60}")
            self.print_status()
            logger.info(f"\n✅ Paper trading session complete!")
            
            # Cleanup
            if self.data_collector:
                await self.data_collector.close()


async def main():
    """Main entry point"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Aster DEX Paper Trading Bot                       ║
║                  (No Real Money - Testing Only)                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize bot
    bot = PaperTradingBot(
        initial_capital=10000,
        symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    )
    
    await bot.initialize()
    
    # Run for 5 cycles (5 hours if interval is 60 minutes)
    # For testing, we'll use shorter intervals
    await bot.run(cycles=5, interval_minutes=5)  # 5 cycles, 5 min intervals = 25 min test


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""
Baseline Momentum Strategy for Immediate Deployment
Simple but effective strategy to test Aster DEX trading infrastructure.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineMomentumStrategy:
    """
    Simple momentum-based strategy combining multiple timeframes.
    
    Strategy Logic:
    1. Calculate momentum across multiple timeframes (1h, 4h, 1d)
    2. Use RSI for overbought/oversold conditions
    3. Confirm with volume
    4. Generate buy/sell signals
    
    This is a baseline to test infrastructure - will be replaced with AI models.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        logger.info("Baseline Momentum Strategy initialized")
    
    def _default_config(self) -> Dict:
        """Default strategy configuration"""
        return {
            # Momentum parameters
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            
            # RSI parameters
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            
            # Volume confirmation
            'volume_ma_period': 20,
            'volume_threshold': 1.5,  # 1.5x average volume
            
            # Risk management
            'max_position_size': 0.1,  # 10% of capital per trade
            'stop_loss_pct': 0.02,  # 2% stop loss
            'take_profit_pct': 0.04,  # 4% take profit
            
            # Signal thresholds
            'min_signal_strength': 0.6,  # 0-1 scale
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # MACD
        exp1 = df['close'].ewm(span=self.config['fast_period'], adjust=False).mean()
        exp2 = df['close'].ewm(span=self.config['slow_period'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.config['signal_period'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=self.config['volume_ma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum
        df['momentum_1h'] = df['close'].pct_change(periods=1)
        df['momentum_4h'] = df['close'].pct_change(periods=4)
        df['momentum_24h'] = df['close'].pct_change(periods=24)
        
        # Volatility (ATR approximation)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals"""
        df = self.calculate_indicators(df)
        
        # Initialize signal columns
        df['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
        df['signal_strength'] = 0.0  # 0-1 scale
        df['reason'] = ''
        
        for i in range(200, len(df)):  # Start after enough data for indicators
            signals = []
            reasons = []
            
            # 1. MACD Crossover
            if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and \
               df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]:
                signals.append(0.3)
                reasons.append('MACD_BULLISH')
            elif df['macd'].iloc[i] < df['macd_signal'].iloc[i] and \
                 df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]:
                signals.append(-0.3)
                reasons.append('MACD_BEARISH')
            
            # 2. RSI Conditions
            rsi = df['rsi'].iloc[i]
            if rsi < self.config['rsi_oversold']:
                signals.append(0.4)
                reasons.append('RSI_OVERSOLD')
            elif rsi > self.config['rsi_overbought']:
                signals.append(-0.4)
                reasons.append('RSI_OVERBOUGHT')
            
            # 3. Moving Average Trend
            close = df['close'].iloc[i]
            sma_20 = df['sma_20'].iloc[i]
            sma_50 = df['sma_50'].iloc[i]
            sma_200 = df['sma_200'].iloc[i]
            
            if close > sma_20 > sma_50 > sma_200:
                signals.append(0.3)
                reasons.append('STRONG_UPTREND')
            elif close < sma_20 < sma_50 < sma_200:
                signals.append(-0.3)
                reasons.append('STRONG_DOWNTREND')
            
            # 4. Volume Confirmation
            if df['volume_ratio'].iloc[i] > self.config['volume_threshold']:
                # Amplify signal if high volume
                if signals and signals[-1] > 0:
                    signals.append(0.2)
                    reasons.append('HIGH_VOLUME_CONFIRM')
                elif signals and signals[-1] < 0:
                    signals.append(-0.2)
                    reasons.append('HIGH_VOLUME_CONFIRM')
            
            # 5. Multi-timeframe Momentum
            mom_1h = df['momentum_1h'].iloc[i]
            mom_4h = df['momentum_4h'].iloc[i]
            mom_24h = df['momentum_24h'].iloc[i]
            
            if mom_1h > 0 and mom_4h > 0 and mom_24h > 0:
                signals.append(0.2)
                reasons.append('MULTI_TF_MOMENTUM_UP')
            elif mom_1h < 0 and mom_4h < 0 and mom_24h < 0:
                signals.append(-0.2)
                reasons.append('MULTI_TF_MOMENTUM_DOWN')
            
            # Aggregate signals
            if signals:
                total_signal = sum(signals)
                signal_strength = abs(total_signal)
                
                # Only generate signal if strength exceeds threshold
                if signal_strength >= self.config['min_signal_strength']:
                    if total_signal > 0:
                        df.loc[df.index[i], 'signal'] = 1  # BUY
                    elif total_signal < 0:
                        df.loc[df.index[i], 'signal'] = -1  # SELL
                    
                    df.loc[df.index[i], 'signal_strength'] = min(signal_strength, 1.0)
                    df.loc[df.index[i], 'reason'] = ', '.join(reasons)
        
        return df
    
    def calculate_position_size(self, signal_strength: float, capital: float, 
                               current_price: float, atr: float) -> float:
        """Calculate position size based on signal strength and risk"""
        # Base position size
        base_size = capital * self.config['max_position_size']
        
        # Adjust by signal strength
        adjusted_size = base_size * signal_strength
        
        # Risk-based sizing using ATR
        if atr > 0:
            risk_amount = capital * self.config['stop_loss_pct']
            shares_by_risk = risk_amount / (atr * 2)  # 2x ATR stop
            shares_by_capital = adjusted_size / current_price
            
            # Take minimum for safety
            shares = min(shares_by_risk, shares_by_capital)
        else:
            shares = adjusted_size / current_price
        
        return shares
    
    def backtest(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """Simple backtest to validate strategy"""
        df = self.generate_signals(df)
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(len(df)):
            signal = df['signal'].iloc[i]
            price = df['close'].iloc[i]
            signal_strength = df['signal_strength'].iloc[i]
            
            # Entry logic
            if signal == 1 and position == 0:  # BUY signal, no position
                shares = self.calculate_position_size(
                    signal_strength, capital, price, df['atr'].iloc[i]
                )
                position = shares
                entry_price = price
                capital -= shares * price
                
                trades.append({
                    'timestamp': df.index[i],
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'signal_strength': signal_strength,
                    'reason': df['reason'].iloc[i]
                })
            
            # Exit logic
            elif position > 0:
                exit_signal = False
                exit_reason = ''
                
                # Take profit
                if price >= entry_price * (1 + self.config['take_profit_pct']):
                    exit_signal = True
                    exit_reason = 'TAKE_PROFIT'
                
                # Stop loss
                elif price <= entry_price * (1 - self.config['stop_loss_pct']):
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                
                # Sell signal
                elif signal == -1:
                    exit_signal = True
                    exit_reason = 'SELL_SIGNAL'
                
                if exit_signal:
                    capital += position * price
                    pnl = (price - entry_price) * position
                    
                    trades.append({
                        'timestamp': df.index[i],
                        'type': 'SELL',
                        'price': price,
                        'shares': position,
                        'pnl': pnl,
                        'return_pct': (price / entry_price - 1) * 100,
                        'reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
        
        # Close any open position
        if position > 0:
            final_price = df['close'].iloc[-1]
            capital += position * final_price
            pnl = (final_price - entry_price) * position
            trades.append({
                'timestamp': df.index[-1],
                'type': 'SELL',
                'price': final_price,
                'shares': position,
                'pnl': pnl,
                'return_pct': (final_price / entry_price - 1) * 100,
                'reason': 'FINAL_CLOSE'
            })
        
        # Calculate metrics
        final_capital = capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': len([t for t in trades if t['type'] == 'BUY']),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1) * 100,
            'trades': trades
        }
        
        return results
    
    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """Get the latest trading signal"""
        df = self.generate_signals(df)
        latest = df.iloc[-1]
        
        return {
            'timestamp': latest.name,
            'signal': int(latest['signal']),
            'signal_strength': float(latest['signal_strength']),
            'reason': latest['reason'],
            'price': float(latest['close']),
            'rsi': float(latest['rsi']),
            'macd': float(latest['macd']),
            'volume_ratio': float(latest['volume_ratio'])
        }


def main():
    """Test the baseline strategy"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║           Baseline Momentum Strategy - Quick Test             ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Load some test data
    data_file = Path("data/historical/real_aster_only/BTCUSDT_1h.parquet")
    
    if not data_file.exists():
        print("❌ No test data found. Please run data collection first.")
        return
    
    df = pd.read_parquet(data_file)
    print(f"✓ Loaded {len(df)} records for BTCUSDT")
    
    # Initialize strategy
    strategy = BaselineMomentumStrategy()
    
    # Run backtest
    print("\n📊 Running backtest...")
    results = strategy.backtest(df)
    
    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Capital:   ${results['final_capital']:,.2f}")
    print(f"Total Return:    {results['total_return_pct']:.2f}%")
    print(f"\nTotal Trades:    {results['total_trades']}")
    print(f"Winning Trades:  {results['winning_trades']}")
    print(f"Losing Trades:   {results['losing_trades']}")
    print(f"Win Rate:        {results['win_rate']:.1f}%")
    
    # Show recent trades
    print(f"\n{'='*60}")
    print("RECENT TRADES (Last 5)")
    print(f"{'='*60}")
    for trade in results['trades'][-5:]:
        print(f"{trade['timestamp']}: {trade['type']:4s} @ ${trade['price']:.2f} - {trade.get('reason', '')}")
    
    # Get latest signal
    print(f"\n{'='*60}")
    print("CURRENT SIGNAL")
    print(f"{'='*60}")
    signal = strategy.get_latest_signal(df)
    signal_text = {1: '🟢 BUY', 0: '⚪ HOLD', -1: '🔴 SELL'}[signal['signal']]
    print(f"Signal: {signal_text}")
    print(f"Strength: {signal['signal_strength']:.2f}")
    print(f"Reason: {signal['reason']}")
    print(f"Price: ${signal['price']:.2f}")
    print(f"RSI: {signal['rsi']:.1f}")
    
    print("\n✅ Baseline strategy validated and ready for deployment!")


if __name__ == "__main__":
    main()


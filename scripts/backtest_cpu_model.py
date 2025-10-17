#!/usr/bin/env python3
"""
Backtest CPU-trained model on historical data
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPUBacktester:
    """
    Backtest trading strategy using CPU-trained model.
    """
    
    def __init__(self):
        self.model_path = Path("models/random_forest_cpu.pkl")
        self.metadata_path = Path("models/cpu_models_metadata.json")
        self.data_dir = Path("data/historical/real_aster_only")
        
        self.model = None
        self.metadata = None
        self.data = None
        
        # Trading parameters
        self.initial_capital = 10000
        self.risk_per_trade = 0.01  # 1%
        self.max_position_size = 0.1  # 10% of capital
        self.trading_fee = 0.001  # 0.1%
        
        # Results
        self.trades = []
        self.equity_curve = []
        
        logger.info("Backtester initialized")
    
    def load_model(self) -> bool:
        """Load trained model."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Model loaded: {self.metadata['model_type']}")
            logger.info(f"Accuracy: {self.metadata['accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_data(self) -> bool:
        """Load historical data."""
        try:
            logger.info("Loading historical data...")
            
            data_files = [f for f in self.data_dir.glob("*.parquet") if "collection_summary" not in f.name]
            if not data_files:
                logger.error("No data files found")
                return False
            
            all_data = []
            for file in data_files:
                df = pd.read_parquet(file)
                df['symbol'] = file.stem.replace('_1h', '').replace('_4h', '').replace('_1d', '')
                # Add timestamp index as column if not present
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.date_range(start='2024-04-01', periods=len(df), freq='1H')
                all_data.append(df)
            
            self.data = pd.concat(all_data, ignore_index=True)
            if 'timestamp' in self.data.columns:
                self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            else:
                self.data = self.data.reset_index(drop=True)
            
            logger.info(f"Loaded {len(self.data)} data points from {len(data_files)} assets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Price features
        df['price_change'] = df.groupby('symbol')['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_price_ratio'] = df['volume'] / df['close']
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        
        # Volatility
        df['volatility'] = df.groupby('symbol')['price_change'].rolling(20).std().reset_index(0, drop=True)
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Actual returns for validation
        df['actual_return'] = df.groupby('symbol')['close'].pct_change(1).shift(-1)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_backtest(self) -> bool:
        """Run backtest simulation."""
        try:
            logger.info("Running backtest...")
            
            # Prepare data
            df = self.prepare_features(self.data.copy())
            df = df.dropna()
            
            # Feature columns
            feature_cols = [
                'price_change', 'high_low_ratio', 'volume_price_ratio',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                'volatility', 'rsi'
            ]
            
            # Generate predictions
            X = df[feature_cols].values
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            df['signal'] = predictions
            df['confidence'] = probabilities.max(axis=1)
            
            # Simulate trading
            capital = self.initial_capital
            position = 0
            entry_price = 0
            entry_capital = 0
            
            self.equity_curve = [capital]
            
            for idx, row in df.iterrows():
                # Exit position if open
                if position != 0:
                    # Calculate P&L
                    if position > 0:  # Long position
                        pnl = position * (row['close'] - entry_price)
                        pnl -= entry_capital * self.trading_fee  # Exit fee
                        capital += entry_capital + pnl
                    
                    # Record trade
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['timestamp'],
                        'symbol': row['symbol'],
                        'direction': 'LONG' if position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'position_size': abs(position),
                        'capital': entry_capital,
                        'pnl': pnl,
                        'pnl_pct': (pnl / entry_capital) * 100 if entry_capital > 0 else 0,
                        'confidence': entry_confidence
                    })
                    
                    position = 0
                
                # Enter new position on signal
                if row['signal'] == True and row['confidence'] > 0.6:  # High confidence trades only
                    position_size = min(
                        capital * self.risk_per_trade,  # Risk management
                        capital * self.max_position_size  # Position limit
                    )
                    
                    # Long entry
                    position = position_size / row['close']
                    entry_price = row['close']
                    entry_capital = position_size
                    entry_time = row['timestamp']
                    entry_confidence = row['confidence']
                    capital -= position_size
                    capital -= position_size * self.trading_fee  # Entry fee
                
                self.equity_curve.append(capital + (position * row['close'] if position > 0 else 0))
            
            logger.info(f"Backtest complete: {len(self.trades)} trades executed")
            return True
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        try:
            if not self.trades:
                logger.warning("No trades to analyze")
                return {}
            
            trades_df = pd.DataFrame(self.trades)
            
            # Win rate
            winning_trades = trades_df[trades_df['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades_df) * 100
            
            # Profit metrics
            total_pnl = trades_df['pnl'].sum()
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            
            # Sharpe ratio
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Max drawdown
            equity_array = np.array(self.equity_curve)
            cummax = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            # Final equity
            final_equity = self.equity_curve[-1]
            total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
            
            metrics = {
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(trades_df) - len(winning_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'initial_capital': self.initial_capital,
                'final_equity': final_equity
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}
    
    def print_results(self, metrics: dict):
        """Print backtest results."""
        print("\n" + "="*80)
        print("                         BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n📊 Trading Summary:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Winning Trades: {metrics['winning_trades']}")
        print(f"   Losing Trades: {metrics['losing_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.2f}%")
        
        print(f"\n💰 Profitability:")
        print(f"   Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"   Final Equity: ${metrics['final_equity']:,.2f}")
        print(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Average Win: ${metrics['avg_win']:,.2f}")
        print(f"   Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        print("\n" + "="*80)
        
        # Verdict
        if metrics['win_rate'] >= 60 and metrics['sharpe_ratio'] >= 1.0 and metrics['max_drawdown_pct'] > -20:
            print("✅ VERDICT: Strategy is PROFITABLE and READY for paper trading!")
            print("   Next step: python trading/ai_trading_bot.py --mode paper")
        elif metrics['win_rate'] >= 50:
            print("⚠️  VERDICT: Strategy shows promise but needs optimization")
            print("   Recommendation: Adjust confidence threshold or risk parameters")
        else:
            print("❌ VERDICT: Strategy needs improvement before live trading")
            print("   Recommendation: Collect more data or retrain model")
        
        print("="*80 + "\n")
    
    def run(self) -> bool:
        """Run complete backtest pipeline."""
        try:
            logger.info("Starting backtest pipeline...")
            
            if not self.load_model():
                return False
            
            if not self.load_data():
                return False
            
            if not self.run_backtest():
                return False
            
            metrics = self.calculate_metrics()
            if not metrics:
                return False
            
            self.print_results(metrics)
            
            # Save results
            results_dir = Path("backtest_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"backtest_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Backtest pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution."""
    print("""
================================================================================
                        CPU Model Backtesting
                  Validating Strategy Performance
================================================================================
    """)
    
    backtester = CPUBacktester()
    
    try:
        success = backtester.run()
        if not success:
            print("\n❌ Backtest failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Backtest stopped by user")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


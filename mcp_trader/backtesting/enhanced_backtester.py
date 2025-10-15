import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage
    min_order_size: float = 10.0
    max_leverage: float = 10.0
    enable_shorting: bool = True
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: List[float]
    trade_log: List[Dict[str, Any]]
    final_balance: float
    total_pnl: float
    avg_trade_pnl: float
    max_consecutive_losses: int
    profit_factor: float


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name."""
        pass


class MovingAverageCrossoverStrategy(TradingStrategy):
    """Simple moving average crossover strategy."""

    def get_strategy_name(self) -> str:
        return "MA_Crossover"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover."""
        df = data.copy()

        # Calculate moving averages
        fast_period = self.config.get('fast_ma', 10)
        slow_period = self.config.get('slow_ma', 30)

        df['fast_ma'] = df['close'].rolling(fast_period).mean()
        df['slow_ma'] = df['close'].rolling(slow_period).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Buy
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Sell

        # Only signal on crossovers
        df['signal_change'] = df['signal'].diff()
        df['final_signal'] = df['signal_change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        return df


class RSIStrategy(TradingStrategy):
    """RSI-based mean reversion strategy."""

    def get_strategy_name(self) -> str:
        return "RSI_Strategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI."""
        df = data.copy()

        # Calculate RSI
        period = self.config.get('rsi_period', 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Generate signals
        oversold = self.config.get('oversold', 30)
        overbought = self.config.get('overbought', 70)

        df['signal'] = 0
        df.loc[df['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
        df.loc[df['rsi'] > overbought, 'signal'] = -1  # Sell when overbought

        return df


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands mean reversion strategy."""

    def get_strategy_name(self) -> str:
        return "Bollinger_Bands"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Bands."""
        df = data.copy()

        # Calculate Bollinger Bands
        period = self.config.get('bb_period', 20)
        std_dev = self.config.get('bb_std', 2)

        df['sma'] = df['close'].rolling(period).mean()
        df['std'] = df['close'].rolling(period).std()
        df['upper_band'] = df['sma'] + (df['std'] * std_dev)
        df['lower_band'] = df['sma'] - (df['std'] * std_dev)

        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy when below lower band
        df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell when above upper band

        return df


class EnhancedBacktester:
    """Enhanced backtesting engine with multiple strategies and advanced analytics."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results: Dict[str, BacktestResult] = {}

    async def run_backtest(self,
                          strategy_class: Callable,
                          strategy_config: Dict[str, Any],
                          data: pd.DataFrame,
                          symbols: List[str],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run a backtest for a given strategy and data.

        Args:
            strategy_class: Strategy class to instantiate
            strategy_config: Configuration for the strategy
            data: Historical market data
            symbols: List of symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            BacktestResult with comprehensive performance metrics
        """
        try:
            # Initialize strategy
            strategy = strategy_class(strategy_config)

            # Filter data by date range if specified
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]

            if data.empty:
                raise ValueError("No data available for the specified date range")

            # Generate signals
            signals_data = strategy.generate_signals(data)

            # Run simulation
            result = await self._simulate_trades(signals_data, symbols)

            logger.info(f"Backtest completed for {strategy.get_strategy_name()}: "
                       f"Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            raise

    async def _simulate_trades(self, data: pd.DataFrame, symbols: List[str]) -> BacktestResult:
        """Simulate trades based on signals."""
        balance = self.config.initial_balance
        position = 0  # Current position size
        entry_price = 0
        equity_curve = [balance]
        trade_log = []
        consecutive_losses = 0
        max_consecutive_losses = 0
        total_wins = 0
        total_losses = 0
        gross_profit = 0
        gross_loss = 0

        # Process each row in chronological order
        for idx, row in data.iterrows():
            current_price = row['close']
            signal = row.get('final_signal', row.get('signal', 0))

            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size (simple fixed percentage)
                position_size = balance * 0.1  # 10% of balance
                commission = position_size * self.config.commission_rate
                slippage = position_size * self.config.slippage_rate

                actual_price = current_price * (1 + self.config.slippage_rate)
                position = position_size / actual_price
                entry_price = actual_price
                balance -= (position_size + commission + slippage)

                trade_log.append({
                    'timestamp': idx,
                    'type': 'BUY',
                    'price': actual_price,
                    'size': position,
                    'commission': commission,
                    'balance': balance
                })

                logger.debug(f"BUY at {actual_price:.4f}, position: {position:.6f}")

            elif signal == -1 and position > 0:  # Sell signal
                # Calculate exit
                exit_price = current_price * (1 - self.config.slippage_rate)
                exit_value = position * exit_price
                commission = exit_value * self.config.commission_rate

                pnl = exit_value - (position * entry_price) - commission
                balance += exit_value - commission

                # Track wins/losses
                if pnl > 0:
                    total_wins += 1
                    gross_profit += pnl
                    consecutive_losses = 0
                else:
                    total_losses += 1
                    gross_loss += abs(pnl)
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

                trade_log.append({
                    'timestamp': idx,
                    'type': 'SELL',
                    'price': exit_price,
                    'size': position,
                    'pnl': pnl,
                    'commission': commission,
                    'balance': balance
                })

                position = 0
                entry_price = 0

                logger.debug(f"SELL at {exit_price:.4f}, PnL: {pnl:.2f}")

            # Update equity curve
            current_equity = balance + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)

        # Calculate performance metrics
        total_return = (equity_curve[-1] - self.config.initial_balance) / self.config.initial_balance
        total_trades = len([t for t in trade_log if t['type'] == 'SELL'])

        # Calculate Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 0:
            sharpe_ratio = (returns.mean() - self.config.risk_free_rate/252) / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Calculate maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate win rate
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        # Calculate profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate average trade PnL
        total_pnl = sum([t.get('pnl', 0) for t in trade_log if 'pnl' in t])
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            equity_curve=equity_curve,
            trade_log=trade_log,
            final_balance=equity_curve[-1],
            total_pnl=total_pnl,
            avg_trade_pnl=avg_trade_pnl,
            max_consecutive_losses=max_consecutive_losses,
            profit_factor=profit_factor
        )

    async def compare_strategies(self,
                                strategies: List[Tuple[Callable, Dict[str, Any]]],
                                data: pd.DataFrame,
                                symbols: List[str]) -> Dict[str, BacktestResult]:
        """Compare multiple strategies on the same data."""
        results = {}

        for strategy_class, config in strategies:
            strategy_name = strategy_class(config).get_strategy_name()
            logger.info(f"Running backtest for {strategy_name}")

            try:
                result = await self.run_backtest(strategy_class, config, data, symbols)
                results[strategy_name] = result
            except Exception as e:
                logger.error(f"Error running {strategy_name}: {e}")
                results[strategy_name] = None

        return results

    def get_best_strategy(self, results: Dict[str, BacktestResult]) -> Tuple[str, BacktestResult]:
        """Find the best performing strategy based on Sharpe ratio."""
        valid_results = {k: v for k, v in results.items() if v is not None}

        if not valid_results:
            raise ValueError("No valid strategy results to compare")

        best_strategy = max(valid_results.items(),
                          key=lambda x: x[1].sharpe_ratio)

        return best_strategy

    def print_performance_report(self, result: BacktestResult, strategy_name: str):
        """Print a detailed performance report."""
        print(f"\n{'='*60}")
        print(f"STRATEGY PERFORMANCE REPORT: {strategy_name}")
        print(f"{'='*60}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Final Balance: ${result.final_balance:.2f}")
        print(f"Average Trade PnL: ${result.avg_trade_pnl:.2f}")
        print(f"Max Consecutive Losses: {result.max_consecutive_losses}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"{'='*60}")


async def main():
    """Main function for testing the backtester."""
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    np.random.seed(42)

    # Generate realistic price data
    initial_price = 50000
    prices = [initial_price]
    for i in range(999):
        # Random walk with slight upward trend
        change = np.random.normal(0.0001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Floor price

    # Create DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in prices]
    }, index=dates)

    # Setup backtester
    config = BacktestConfig(initial_balance=10000.0)
    backtester = EnhancedBacktester(config)

    # Test different strategies
    strategies = [
        (MovingAverageCrossoverStrategy, {'fast_ma': 10, 'slow_ma': 30}),
        (RSIStrategy, {'rsi_period': 14, 'oversold': 30, 'overbought': 70}),
        (BollingerBandsStrategy, {'bb_period': 20, 'bb_std': 2})
    ]

    print("Running strategy comparison...")
    results = await backtester.compare_strategies(strategies, data, ['BTCUSDT'])

    # Print results
    for strategy_name, result in results.items():
        if result:
            backtester.print_performance_report(result, strategy_name)
        else:
            print(f"Strategy {strategy_name} failed to run")

    # Find best strategy
    try:
        best_name, best_result = backtester.get_best_strategy(results)
        print(f"\n🏆 BEST STRATEGY: {best_name} (Sharpe: {best_result.sharpe_ratio:.2f})")
    except ValueError as e:
        print(f"Error finding best strategy: {e}")


if __name__ == "__main__":
    asyncio.run(main())


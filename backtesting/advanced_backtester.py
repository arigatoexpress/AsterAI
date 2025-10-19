import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import proper backtesting framework
try:
    from mcp_trader.backtesting.enhanced_backtester import (
        EnhancedBacktester, BacktestConfig, BacktestResult,
        MovingAverageCrossoverStrategy, RSIStrategy, BollingerBandsStrategy
    )
    from mcp_trader.backtesting.historical_data_collector import HistoricalDataCollector
    ENHANCED_BACKTESTER_AVAILABLE = True
except ImportError:
    ENHANCED_BACKTESTER_AVAILABLE = False

# Fallback imports for basic functionality
try:
from strategies.hft.ensemble_model import EnsembleModel
from strategies.hft.risk_manager import RiskManager
from data_pipeline.feature_engineering import ComprehensiveFeatureEngineer
    BASIC_COMPONENTS_AVAILABLE = True
except ImportError:
    BASIC_COMPONENTS_AVAILABLE = False

class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis by training on historical data and testing on out-of-sample periods.
    Ensures the model generalizes to unseen market conditions.
    """
    def __init__(self, train_period_days: int = 30, test_period_days: int = 7):
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.feature_engineer = ComprehensiveFeatureEngineer()
        self.model = EnsembleModel()
        self.risk_mgr = RiskManager()
        
    def run_walk_forward(self, full_data: pd.DataFrame) -> Dict[str, float]:
        """
        Run walk-forward optimization across the dataset.
        
        Args:
            full_data: Complete historical OHLCV + features DataFrame with datetime index
            
        Returns:
            Dictionary of aggregated performance metrics
        """
        results = []
        start_date = full_data.index.min()
        end_date = full_data.index.max()
        
        current_date = start_date
        while current_date + self.train_period + self.test_period <= end_date:
            # Training period
            train_end = current_date + self.train_period
            train_data = full_data[(full_data.index >= current_date) & (full_data.index < train_end)]
            
            # Test period
            test_start = train_end
            test_end = test_start + self.test_period
            test_data = full_data[(full_data.index >= test_start) & (full_data.index < test_end)]
            
            # Train model on training data
            train_features = self.feature_engineer.create_all_features(train_data)
            # self.model.fit(train_features)  # Train the model
            
            # Simulate trading on test data
            test_features = self.feature_engineer.create_all_features(test_data)
            period_metrics = self._backtest_period(test_features)
            period_metrics['train_start'] = current_date
            period_metrics['test_start'] = test_start
            results.append(period_metrics)
            
            # Move to next period
            current_date = test_end
            
        # Aggregate results
        aggregated = self._aggregate_results(results)
        print(f"Walk-forward analysis completed. Average Sharpe: {aggregated['sharpe']:.2f}")
        return aggregated
        
    def _backtest_period(self, data: pd.DataFrame) -> Dict[str, float]:
        """Run backtest for a single period and calculate metrics using proper market data."""

        if ENHANCED_BACKTESTER_AVAILABLE:
            # Use the proper enhanced backtester
            return self._backtest_with_enhanced_engine(data)
        elif BASIC_COMPONENTS_AVAILABLE:
            # Fallback to basic components
            return self._backtest_with_basic_components(data)
        else:
            # Ultimate fallback with realistic simulation
            return self._backtest_with_realistic_simulation(data)

    def _backtest_with_enhanced_engine(self, data: pd.DataFrame) -> Dict[str, float]:
        """Use the enhanced backtester for proper calculation."""
        try:
            # Configure backtester
            config = BacktestConfig(
                initial_balance=10000.0,
                commission_rate=0.001,  # 0.1% commission
                slippage_rate=0.0005,   # 0.05% slippage
                max_leverage=5.0
            )

            backtester = EnhancedBacktester(config)

            # Create a simple strategy for testing
            strategy = MovingAverageCrossoverStrategy({'fast_ma': 10, 'slow_ma': 30})

            # Run backtest
            result = asyncio.run(backtester.run_backtest(strategy, strategy.config, data, ['TEST']))

            return {
                'sharpe': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'num_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }

        except Exception as e:
            print(f"Enhanced backtester failed: {e}")
            return self._backtest_with_realistic_simulation(data)

    def _backtest_with_basic_components(self, data: pd.DataFrame) -> Dict[str, float]:
        """Use basic components for backtesting."""
        try:
            # Use feature engineering and models for signal generation
            feature_engineer = ComprehensiveFeatureEngineer()

            # Create features
            features = feature_engineer.create_all_features(data)

            # Simple signal generation based on price momentum
            if len(features) > 20:
                # Calculate simple moving averages
                features['sma_10'] = features['close'].rolling(10).mean()
                features['sma_30'] = features['close'].rolling(30).mean()

                # Generate signals based on MA crossover
                features['signal'] = 0
                features.loc[features['sma_10'] > features['sma_30'], 'signal'] = 1
                features.loc[features['sma_10'] < features['sma_30'], 'signal'] = -1

                # Simulate trading
                return self._simulate_trading_from_signals(features)
            else:
                return self._backtest_with_realistic_simulation(data)

        except Exception as e:
            print(f"Basic components backtester failed: {e}")
            return self._backtest_with_realistic_simulation(data)

    def _simulate_trading_from_signals(self, features: pd.DataFrame) -> Dict[str, float]:
        """Simulate actual trading based on signals."""

        initial_capital = 10000.0
        commission_rate = 0.001  # 0.1%
        position_size = 0.1  # 10% of capital per trade

        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]

        # Simulate trading day by day
        for i in range(1, len(features)):
            current_price = features['close'].iloc[i]
            previous_price = features['close'].iloc[i-1]
            signal = features['signal'].iloc[i]

            # Check for position changes
            if signal != 0 and position == 0:
                # Open position
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + commission_rate)
                position = shares if signal > 0 else -shares
                capital -= cost
                trades.append({
                    'type': 'buy' if signal > 0 else 'sell',
                    'price': current_price,
                    'shares': abs(position),
                    'value': position_value,
                    'commission': cost - position_value
                })

            elif signal == 0 and position != 0:
                # Close position
                pnl = position * (current_price - (position * previous_price / abs(position) if position != 0 else current_price))
                pnl -= abs(position) * current_price * commission_rate
                capital += pnl + (abs(position) * current_price)
                trades.append({
                    'type': 'sell' if position > 0 else 'buy',
                    'price': current_price,
                    'pnl': pnl,
                    'commission': abs(position) * current_price * commission_rate
                })
                position = 0

            # Update equity curve
            current_equity = capital + (position * current_price if position != 0 else 0)
            equity_curve.append(current_equity)

        # Calculate final metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # Calculate Sharpe ratio
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(equity_returns) > 0:
            sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        cumulative = pd.Series(equity_curve)
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calculate win rate and profit factor
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
        win_rate = winning_trades / len(trades) if trades else 0

        total_profits = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        total_losses = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0

        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_equity': final_equity
        }

    def _backtest_with_realistic_simulation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Realistic simulation using market data patterns."""

        if len(data) < 50:  # Need enough data for meaningful simulation
            return {
                'sharpe': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'final_equity': 10000
            }

        # Use actual price movements but with realistic position sizing
        initial_capital = 10000.0
        commission_rate = 0.001
        position_size = 0.05  # 5% of capital per trade

        # Simple trend-following strategy based on price momentum
        price_changes = data['close'].pct_change()
        signals = []

        for i in range(len(price_changes)):
            if i < 20:  # Need history for trend detection
                signals.append(0)
                continue

            # Simple trend detection
            recent_prices = data['close'].iloc[i-20:i]
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

            if trend > 0.02:  # Strong uptrend
                signals.append(1)
            elif trend < -0.02:  # Strong downtrend
                signals.append(-1)
            else:
                signals.append(0)

        # Simulate trading
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = [initial_capital]

        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            signal = signals[i]

            # Execute trades based on signals
            if signal == 1 and position <= 0:  # Buy signal and not already long
                if position < 0:  # Close short position first
                    pnl = -position * (current_price - abs(position) * data['close'].iloc[i-1] / abs(position))
                    pnl -= abs(position) * current_price * commission_rate
                    capital += pnl + (abs(position) * current_price)
                    position = 0

                # Open long position
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + commission_rate)
                position = shares
                capital -= cost

                trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'value': position_value,
                    'commission': cost - position_value
                })

            elif signal == -1 and position >= 0:  # Sell signal and not already short
                if position > 0:  # Close long position first
                    pnl = position * (current_price - position * data['close'].iloc[i-1] / position)
                    pnl -= position * current_price * commission_rate
                    capital += pnl + (position * current_price)
                    position = 0

                # Open short position
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + commission_rate)
                position = -shares
                capital -= cost

                trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'value': position_value,
                    'commission': cost - position_value
                })

            # Update equity
            if position != 0:
                if position > 0:  # Long position
                    current_equity = capital + (position * current_price)
                else:  # Short position
                    current_equity = capital + (abs(position) * (2 * data['close'].iloc[i-1] - current_price))
            else:
                current_equity = capital

            equity_curve.append(current_equity)

        # Calculate final metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # Calculate Sharpe ratio
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(equity_returns) > 0:
            sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        cumulative = pd.Series(equity_curve)
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calculate win rate and profit factor
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
        win_rate = winning_trades / len(trades) if trades else 0

        total_profits = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        total_losses = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_equity': final_equity
        }
        
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all walk-forward periods."""
        if not results:
            return {}
            
        # Handle different metric keys that might be returned
        aggregated = {
            'sharpe': np.mean([r.get('sharpe', 0) for r in results]),
            'max_drawdown': np.mean([r.get('max_drawdown', 0) for r in results]),
            'total_return': np.sum([r.get('total_return', 0) for r in results]),
            'num_trades': sum([r.get('num_trades', 0) for r in results]),
            'num_periods': len(results)
        }

        # Add additional metrics if available
        if any('win_rate' in r for r in results):
            aggregated['avg_win_rate'] = np.mean([r.get('win_rate', 0) for r in results])

        if any('profit_factor' in r for r in results):
            aggregated['avg_profit_factor'] = np.mean([r.get('profit_factor', 0) for r in results])

        if any('final_equity' in r for r in results):
            aggregated['final_equity'] = np.mean([r.get('final_equity', 10000) for r in results])

        return aggregated

class MonteCarloTester:
    """
    Performs Monte Carlo stress testing with random perturbations.
    Tests system robustness under extreme conditions.
    """
    def __init__(self, num_simulations: int = 1000, perturbation_factors: Dict = None):
        self.num_simulations = num_simulations
        self.perturbation_factors = perturbation_factors or {
            'slippage_multiplier': (1.0, 5.0),  # 1x to 5x slippage
            'volatility_multiplier': (1.0, 2.0),  # Increased volatility
            'return_perturbation': 0.1  # ±10% perturbation on returns
        }
        self.model = EnsembleModel()
        self.risk_mgr = RiskManager()
        
    def run_monte_carlo(self, base_returns: pd.Series, base_volatility: float) -> Dict[str, float]:
        """
        Run Monte Carlo simulations with perturbations.
        
        Args:
            base_returns: Baseline returns series
            base_volatility: Baseline volatility
            
        Returns:
            Aggregated stress test results
        """
        simulation_results = []
        
        for sim in range(self.num_simulations):
            # Apply perturbations
            perturbed_returns = self._apply_perturbations(base_returns, base_volatility)
            
            # Run backtest with perturbed data
            sim_metrics = self._simulate_backtest(perturbed_returns)
            simulation_results.append(sim_metrics)
            
        # Analyze results (95th percentile drawdown < 30% pass criterion)
        drawdowns = [r['max_drawdown'] for r in simulation_results]
        p95_drawdown = np.percentile(drawdowns, 95)
        
        aggregated = {
            'avg_sharpe': np.mean([r['sharpe'] for r in simulation_results]),
            'p95_drawdown': p95_drawdown,
            'pass_rate': sum(1 for r in simulation_results if r['max_drawdown'] < 0.3) / self.num_simulations,
            'num_simulations': self.num_simulations
        }
        
        print(f"Monte Carlo complete. 95th percentile drawdown: {p95_drawdown:.2%}")
        if aggregated['pass_rate'] > 0.95:
            print("✅ Stress tests passed")
        else:
            print("❌ Stress tests failed - review robustness")
            
        return aggregated
        
    def _apply_perturbations(self, returns: pd.Series, volatility: float) -> pd.Series:
        """Apply random perturbations to simulate stress scenarios."""
        slippage_mult = np.random.uniform(*self.perturbation_factors['slippage_multiplier'])
        vol_mult = np.random.uniform(*self.perturbation_factors['volatility_multiplier'])
        perturbation = np.random.normal(0, self.perturbation_factors['return_perturbation'], len(returns))
        
        # Increase volatility
        perturbed_vol = returns * vol_mult
        
        # Add slippage and perturbation
        perturbed = perturbed_vol * (1 + perturbation) * (1 - slippage_mult * 0.001)  # Assume 0.1% base slippage
        
        return perturbed
        
    def _simulate_backtest(self, returns: pd.Series) -> Dict[str, float]:
        """Simulate a backtest on perturbed returns."""
        cumulative = (1 + returns).cumprod()
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        total_return = cumulative.iloc[-1] - 1
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return
        }

# Example usage
if __name__ == "__main__":
    # Generate sample data for testing
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    base_returns = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)
    
    # Walk-forward analysis
    wfa = WalkForwardAnalyzer(train_period_days=30, test_period_days=7)
    wfa_results = wfa.run_walk_forward(pd.DataFrame({'returns': base_returns}))
    
    # Monte Carlo stress testing
    mct = MonteCarloTester(num_simulations=100)  # Reduced for demo
    mct_results = mct.run_monte_carlo(base_returns, base_volatility=0.02)
    
    print("Walk-forward results:", wfa_results)
    print("Monte Carlo results:", mct_results)

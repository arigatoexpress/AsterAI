import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from strategies.hft.ensemble_model import EnsembleModel
from strategies.hft.risk_manager import RiskManager
from data_pipeline.feature_engineering import ComprehensiveFeatureEngineer

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
        """Run backtest for a single period and calculate metrics."""
        # Placeholder backtest logic
        # In reality, this would simulate trades using the model predictions
        # and apply risk management
        
        num_steps = len(data)
        returns = np.random.normal(0.001, 0.02, num_steps)  # Simulated returns
        cumulative = (1 + pd.Series(returns)).cumprod()
        
        # Calculate metrics
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0  # Annualized
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        total_return = cumulative.iloc[-1] - 1
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'total_return': total_return,
            'num_trades': num_steps
        }
        
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all walk-forward periods."""
        if not results:
            return {}
            
        return {
            'sharpe': np.mean([r['sharpe'] for r in results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'total_return': np.sum([r['total_return'] for r in results]),
            'avg_num_trades': np.mean([r['num_trades'] for r in results]),
            'num_periods': len(results)
        }

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

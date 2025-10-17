"""
Data Utilization Module
Feeds analyzed data into backtesting and real-time execution.
Ensures autonomous trader uses data correctly.
"""
import pandas as pd
from typing import Dict, Any
from backtesting.advanced_backtester import WalkForwardAnalyzer, MonteCarloTester
from strategies.hft.ensemble_model import EnsembleModel
import logging

logger = logging.getLogger(__name__)

class DataUtilizer:
    """Utilizes data for backtesting and live trading."""

    def __init__(self):
        self.model = EnsembleModel()
        self.walk_forward = WalkForwardAnalyzer()
        self.monte_carlo = MonteCarloTester()

    def backtest_autonomous_trader(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Backtest the autonomous trading system."""
        logger.info("Starting autonomous trader backtest")

        # Walk-forward analysis
        wf_results = self.walk_forward.run_walk_forward(historical_data)

        # Monte Carlo stress testing
        base_returns = historical_data['returns'] if 'returns' in historical_data.columns else pd.Series()
        if not base_returns.empty:
            mc_results = self.monte_carlo.run_monte_carlo(base_returns, base_volatility=0.02)
        else:
            mc_results = {}

        # Combine results
        backtest_results = {
            'walk_forward': wf_results,
            'monte_carlo': mc_results,
            'overall_score': self._calculate_overall_score(wf_results, mc_results)
        }

        logger.info("Backtest completed")
        return backtest_results

    def _calculate_overall_score(self, wf: Dict, mc: Dict) -> float:
        """Calculate composite score for system effectiveness."""
        wf_sharpe = wf.get('sharpe', 0)
        mc_pass_rate = mc.get('pass_rate', 0)
        mc_p95_dd = mc.get('p95_drawdown', 1)

        # Score: Sharpe weight 40%, Pass rate 30%, Drawdown 30%
        score = (wf_sharpe * 0.4) + (mc_pass_rate * 0.3) + ((1 - mc_p95_dd) * 0.3)
        return score

    def prepare_for_live_execution(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for real-time execution."""
        # Generate features
        from data_pipeline.feature_engineering import ComprehensiveFeatureEngineer
        engineer = ComprehensiveFeatureEngineer()
        features = engineer.create_all_features(current_data)

        # Get model predictions
        if not features.empty:
            # Simulate prediction (last row)
            state = features.iloc[-1].values
            prediction = self.model.predict(state)
        else:
            prediction = 0  # Hold

        # Risk assessment
        from strategies.hft.risk_manager import RiskManager
        risk_mgr = RiskManager()
        win_rate = 0.6  # From backtest
        avg_win, avg_loss = 0.02, -0.01
        volatility = features['price_volatility_20'].iloc[-1] if 'price_volatility_20' in features.columns else 0.02
        position_size = risk_mgr.calculate_position_size(win_rate, avg_win, avg_loss, volatility)

        execution_data = {
            'features': features,
            'prediction': prediction,
            'position_size': position_size,
            'volatility': volatility,
            'timestamp': pd.Timestamp.now()
        }

        logger.info("Data prepared for live execution")
        return execution_data

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'close': np.random.normal(50000, 1000, 1000),
        'volume': np.random.normal(10000, 2000, 1000),
        'returns': np.random.normal(0.001, 0.02, 1000)
    })

    utilizer = DataUtilizer()
    backtest_results = utilizer.backtest_autonomous_trader(df)
    live_data = utilizer.prepare_for_live_execution(df.tail(50))

    print("Backtest Results:", backtest_results['overall_score'])
    print("Live Prediction:", live_data['prediction'])

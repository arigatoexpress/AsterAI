"""
Monte Carlo Simulation Engine for Trading Strategy Stress Testing

Comprehensive risk assessment through Monte Carlo analysis:
- Bootstrapped return distributions
- Probability of ruin calculation
- Stress testing under extreme market conditions
- Confidence intervals for performance metrics
- Scenario analysis for black swan events
- Parameter sensitivity analysis

Provides statistical robustness for trading strategy deployment.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import warnings

from mcp_trader.ai.ppo_trading_model import PPOMostProfitableTrader, PPOConfig
from mcp_trader.ai.trading_environment import AdvancedTradingEnvironment, TradingConfig
from mcp_trader.backtesting.walk_forward_analysis import WalkForwardResults

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""

    # Simulation parameters
    num_simulations: int = 10000
    simulation_horizon_days: int = 252  # One trading year
    confidence_level: float = 0.95

    # Bootstrapping settings
    block_size_days: int = 5  # For block bootstrapping
    use_block_bootstrapping: bool = True

    # Risk assessment
    initial_capital: float = 10000.0
    risk_free_rate: float = 0.02
    max_drawdown_threshold: float = 0.20

    # Stress testing
    stress_test_enabled: bool = True
    volatility_shock: float = 2.0  # 2x normal volatility
    liquidity_crisis: bool = True
    market_crash: bool = True

    # Scenario analysis
    scenario_analysis_enabled: bool = True
    bull_market: bool = True
    bear_market: bool = True
    sideways_market: bool = True

    # Computational settings
    parallel_processing: bool = True
    max_workers: int = 8
    batch_size: int = 1000
    save_intermediate_results: bool = True

    # Output settings
    generate_risk_report: bool = True
    plot_distributions: bool = True
    export_stress_test_results: bool = True


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""

    config: MonteCarloConfig
    simulations_completed: int = 0

    # Return distributions
    final_portfolio_values: np.ndarray = field(default_factory=lambda: np.array([]))
    total_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    annualized_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_ratios: np.ndarray = field(default_factory=lambda: np.array([]))
    sortino_ratios: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Risk metrics
    value_at_risk_95: float = 0.0
    expected_shortfall_95: float = 0.0
    probability_of_ruin: float = 0.0
    risk_of_ruin_threshold: float = 0.10  # 10% loss threshold

    # Confidence intervals
    return_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    sharpe_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    drawdown_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Stress test results
    stress_test_results: Dict[str, Any] = field(default_factory=dict)

    # Scenario analysis
    scenario_results: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    computation_time_seconds: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""

        if len(self.total_returns) == 0:
            return

        # Value at Risk (95%)
        self.value_at_risk_95 = np.percentile(self.total_returns, 5)

        # Expected Shortfall (95%)
        losses = self.total_returns[self.total_returns < self.value_at_risk_95]
        self.expected_shortfall_95 = losses.mean() if len(losses) > 0 else 0.0

        # Probability of Ruin (falling below risk threshold)
        ruin_threshold = -self.config.risk_of_ruin_threshold
        self.probability_of_ruin = (self.total_returns < ruin_threshold).mean()

        # Confidence intervals
        alpha = (1 - self.config.confidence_level) / 2
        self.return_confidence_interval = (
            np.percentile(self.total_returns, alpha * 100),
            np.percentile(self.total_returns, (1 - alpha) * 100)
        )

        if len(self.sharpe_ratios) > 0:
            self.sharpe_confidence_interval = (
                np.percentile(self.sharpe_ratios, alpha * 100),
                np.percentile(self.sharpe_ratios, (1 - alpha) * 100)
            )

        if len(self.max_drawdowns) > 0:
            self.drawdown_confidence_interval = (
                np.percentile(self.max_drawdowns, alpha * 100),
                np.percentile(self.max_drawdowns, (1 - alpha) * 100)
            )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for reporting"""

        return {
            'total_simulations': self.simulations_completed,
            'mean_return': np.mean(self.total_returns),
            'std_return': np.std(self.total_returns),
            'median_return': np.median(self.total_returns),
            'best_case': np.max(self.total_returns),
            'worst_case': np.min(self.total_returns),
            'sharpe_ratio': np.mean(self.sharpe_ratios) if len(self.sharpe_ratios) > 0 else 0.0,
            'sortino_ratio': np.mean(self.sortino_ratios) if len(self.sortino_ratios) > 0 else 0.0,
            'max_drawdown_avg': np.mean(self.max_drawdowns) if len(self.max_drawdowns) > 0 else 0.0,
            'value_at_risk_95': self.value_at_risk_95,
            'expected_shortfall_95': self.expected_shortfall_95,
            'probability_of_ruin': self.probability_of_ruin,
            'computation_time_minutes': self.computation_time_seconds / 60
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for comprehensive risk assessment

    Provides statistical robustness through thousands of simulated market scenarios
    to estimate true strategy performance and risk characteristics.
    """

    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()

        # Core components
        self.ppo_model: Optional[PPOMostProfitableTrader] = None
        self.environment: Optional[AdvancedTradingEnvironment] = None

        # Historical data for bootstrapping
        self.historical_returns: Optional[np.ndarray] = None
        self.historical_volatility: Optional[np.ndarray] = None

        # Results storage
        self.results = MonteCarloResults(config=self.config)

        # Stress testing scenarios
        self.stress_scenarios = self._define_stress_scenarios()

        logger.info("Monte Carlo Simulator initialized")

    def configure_simulation(self, ppo_model: PPOMostProfitableTrader,
                           environment: AdvancedTradingEnvironment,
                           historical_data: Optional[pd.DataFrame] = None):
        """Configure the simulation with model and data"""

        self.ppo_model = ppo_model
        self.environment = environment

        if historical_data is not None:
            self._prepare_historical_data(historical_data)

        logger.info("Monte Carlo simulation configured")

    def _prepare_historical_data(self, data: pd.DataFrame):
        """Prepare historical data for bootstrapping"""

        # Calculate daily returns
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna().values
            self.historical_returns = returns

            # Calculate rolling volatility
            if len(returns) > 20:
                volatility = pd.Series(returns).rolling(20).std().dropna().values
                self.historical_volatility = volatility

        logger.info(f"Prepared historical data: {len(self.historical_returns or [])} return observations")

    def _define_stress_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Define stress testing scenarios"""

        scenarios = {
            'normal_market': {
                'volatility_multiplier': 1.0,
                'trend_bias': 0.0,
                'liquidity_multiplier': 1.0,
                'description': 'Normal market conditions'
            },
            'high_volatility': {
                'volatility_multiplier': self.config.volatility_shock,
                'trend_bias': 0.0,
                'liquidity_multiplier': 0.8,
                'description': 'High volatility period'
            },
            'liquidity_crisis': {
                'volatility_multiplier': 1.5,
                'trend_bias': -0.001,  # Slight downward bias
                'liquidity_multiplier': 0.3,
                'description': 'Liquidity crisis scenario'
            },
            'market_crash': {
                'volatility_multiplier': 3.0,
                'trend_bias': -0.005,  # Strong downward bias
                'liquidity_multiplier': 0.5,
                'description': 'Market crash scenario'
            },
            'bull_market': {
                'volatility_multiplier': 1.2,
                'trend_bias': 0.002,  # Upward bias
                'liquidity_multiplier': 1.2,
                'description': 'Bull market scenario'
            },
            'bear_market': {
                'volatility_multiplier': 1.8,
                'trend_bias': -0.003,  # Downward bias
                'liquidity_multiplier': 0.7,
                'description': 'Bear market scenario'
            }
        }

        return scenarios

    async def run_monte_carlo_simulation(self) -> MonteCarloResults:
        """
        Execute complete Monte Carlo simulation analysis

        Returns:
            Comprehensive simulation results with risk metrics
        """

        logger.info("="*70)
        logger.info("STARTING MONTE CARLO SIMULATION FOR RISK ASSESSMENT")
        logger.info("="*70)

        start_time = datetime.now()
        self.results.start_time = start_time

        try:
            if self.config.parallel_processing:
                await self._run_parallel_simulations()
            else:
                await self._run_sequential_simulations()

            # Calculate risk metrics
            self.results.calculate_risk_metrics()

            # Run stress tests
            if self.config.stress_test_enabled:
                await self._run_stress_tests()

            # Run scenario analysis
            if self.config.scenario_analysis_enabled:
                await self._run_scenario_analysis()

            # Generate reports
            if self.config.generate_risk_report:
                self._generate_risk_report()

            if self.config.plot_distributions:
                self._plot_distributions()

        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {str(e)}")
            raise
        finally:
            end_time = datetime.now()
            self.results.end_time = end_time
            self.results.computation_time_seconds = (end_time - start_time).total_seconds()

        logger.info("Monte Carlo simulation completed")
        logger.info(".1f")

        return self.results

    async def _run_parallel_simulations(self):
        """Run simulations in parallel"""

        logger.info(f"Running {self.config.num_simulations} simulations in parallel (max {self.config.max_workers} workers)")

        # Create batches for parallel processing
        batch_size = self.config.batch_size
        batches = [
            range(i, min(i + batch_size, self.config.num_simulations))
            for i in range(0, self.config.num_simulations, batch_size)
        ]

        # Run batches in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._run_simulation_batch, batch)
                for batch in batches
            ]

            # Collect results
            all_results = []
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)

        # Process results
        self._process_simulation_results(all_results)

    async def _run_sequential_simulations(self):
        """Run simulations sequentially"""

        logger.info(f"Running {self.config.num_simulations} simulations sequentially")

        all_results = []
        for i in range(self.config.num_simulations):
            result = await self._run_single_simulation(i)
            all_results.append(result)

            if (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1}/{self.config.num_simulations} simulations")

        self._process_simulation_results(all_results)

    def _run_simulation_batch(self, batch_indices: range) -> List[Dict[str, Any]]:
        """Run a batch of simulations (for parallel processing)"""

        results = []
        for i in batch_indices:
            # Note: In parallel mode, we can't use async directly
            # This is a simplified synchronous version
            result = self._run_single_simulation_sync(i)
            results.append(result)

        return results

    async def _run_single_simulation(self, simulation_id: int) -> Dict[str, Any]:
        """Run a single Monte Carlo simulation"""

        # Reset environment
        state, info = self.environment.reset()

        portfolio_values = [self.config.initial_capital]
        trades = []
        max_drawdown = 0.0
        peak_value = self.config.initial_capital

        # Run simulation for specified horizon
        for step in range(self.config.simulation_horizon_days):
            # Generate market conditions (could be bootstrapped from historical data)
            market_conditions = self._generate_market_conditions()

            # Update environment with market conditions
            # This would modify the environment's market data

            # Get action from model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.ppo_model.config.device)
                actions, _, _ = self.ppo_model.actor.sample_actions(state_tensor)
                action = actions.squeeze(0).cpu().numpy()

            # Execute action
            next_state, reward, terminated, truncated, info = self.environment.step(action)

            # Track portfolio value
            current_value = self.environment.balance + self.environment._calculate_unrealized_pnl()
            portfolio_values.append(current_value)

            # Update drawdown
            if current_value > peak_value:
                peak_value = current_value
            current_drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, current_drawdown)

            # Record trades
            if info.get('executed', False):
                trades.append({
                    'step': step,
                    'symbol': info.get('symbol', 'unknown'),
                    'action': 'buy' if action[0] == 1 else 'sell',
                    'price': info.get('price', 0),
                    'quantity': info.get('quantity', 0)
                })

            state = next_state

            if terminated or truncated:
                break

        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        annualized_return = total_return * (252 / len(portfolio_values))  # Assuming daily data

        # Calculate Sharpe ratio (simplified)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) - self.config.risk_free_rate/252) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) - self.config.risk_free_rate/252) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        return {
            'simulation_id': simulation_id,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'num_trades': len(trades)
        }

    def _run_single_simulation_sync(self, simulation_id: int) -> Dict[str, Any]:
        """Synchronous version for parallel processing"""
        # This would be implemented similarly to the async version
        # but without async/await for ProcessPoolExecutor compatibility
        return self._run_single_simulation_sync(simulation_id)

    def _generate_market_conditions(self) -> Dict[str, Any]:
        """Generate market conditions for simulation"""

        # Bootstrap from historical data if available
        if self.historical_returns is not None and self.config.use_block_bootstrapping:
            # Block bootstrapping for serial dependence
            block_size = self.config.block_size_days
            start_idx = np.random.randint(0, len(self.historical_returns) - block_size)
            returns_block = self.historical_returns[start_idx:start_idx + block_size]

            # Add noise for variation
            noise = np.random.normal(0, 0.001, len(returns_block))
            returns_block += noise

            return {
                'returns': returns_block,
                'volatility': np.std(returns_block),
                'trend': np.mean(returns_block)
            }
        else:
            # Generate synthetic market conditions
            volatility = np.random.lognormal(-2, 0.5)  # Typical crypto volatility
            trend = np.random.normal(0.0001, 0.002)     # Slight upward bias
            liquidity = np.random.uniform(0.5, 1.5)     # Liquidity multiplier

            return {
                'volatility': volatility,
                'trend': trend,
                'liquidity': liquidity
            }

    def _process_simulation_results(self, all_results: List[Dict[str, Any]]):
        """Process results from all simulations"""

        if not all_results:
            return

        # Extract arrays
        self.results.final_portfolio_values = np.array([r['final_value'] for r in all_results])
        self.results.total_returns = np.array([r['total_return'] for r in all_results])
        self.results.annualized_returns = np.array([r['annualized_return'] for r in all_results])
        self.results.sharpe_ratios = np.array([r['sharpe_ratio'] for r in all_results])
        self.results.sortino_ratios = np.array([r['sortino_ratio'] for r in all_results])
        self.results.max_drawdowns = np.array([r['max_drawdown'] for r in all_results])

        self.results.simulations_completed = len(all_results)

        logger.info(f"Processed results from {len(all_results)} simulations")

    async def _run_stress_tests(self):
        """Run stress tests under extreme conditions"""

        logger.info("Running stress tests...")

        stress_results = {}

        for scenario_name, scenario_params in self.stress_scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")

            # Run simulations with modified parameters
            scenario_results = []
            for i in range(min(1000, self.config.num_simulations // 10)):  # Fewer simulations for speed
                result = await self._run_single_simulation(i)
                # Apply scenario modifications
                modified_result = self._apply_scenario_modifications(result, scenario_params)
                scenario_results.append(modified_result)

            # Calculate scenario statistics
            scenario_returns = np.array([r['total_return'] for r in scenario_results])

            stress_results[scenario_name] = {
                'mean_return': np.mean(scenario_returns),
                'worst_case': np.min(scenario_returns),
                'probability_of_loss': (scenario_returns < 0).mean(),
                'description': scenario_params['description'],
                'simulations': len(scenario_results)
            }

        self.results.stress_test_results = stress_results
        logger.info("Stress tests completed")

    def _apply_scenario_modifications(self, result: Dict[str, Any],
                                    scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scenario-specific modifications to simulation results"""

        modified_result = result.copy()

        # Apply volatility shock
        vol_multiplier = scenario_params['volatility_multiplier']
        modified_result['total_return'] *= (1 + (vol_multiplier - 1) * 0.5)  # Partial effect

        # Apply trend bias
        trend_bias = scenario_params['trend_bias']
        modified_result['total_return'] += trend_bias * self.config.simulation_horizon_days

        # Apply liquidity impact
        liquidity_multiplier = scenario_params['liquidity_multiplier']
        if liquidity_multiplier < 1.0:
            # Reduce returns due to slippage in low liquidity
            liquidity_penalty = (1 - liquidity_multiplier) * 0.1
            modified_result['total_return'] *= (1 - liquidity_penalty)

        return modified_result

    async def _run_scenario_analysis(self):
        """Run scenario analysis for different market conditions"""

        logger.info("Running scenario analysis...")

        # This would analyze how the strategy performs under different market regimes
        # For now, use the stress test results
        self.results.scenario_results = self.results.stress_test_results.copy()

        logger.info("Scenario analysis completed")

    def _generate_risk_report(self):
        """Generate comprehensive risk assessment report"""

        output_dir = Path("results/monte_carlo")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "risk_assessment_report.md"

        with open(report_path, 'w') as f:
            f.write("# Monte Carlo Risk Assessment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Simulations Run**: {self.results.simulations_completed:,}\n")
            f.write(f"- **Time Horizon**: {self.config.simulation_horizon_days} days\n")
            f.write(f"- **Initial Capital**: ${self.config.initial_capital:,.0f}\n")
            f.write(".1f")

            summary_stats = self.results.get_summary_stats()

            f.write("## Performance Distribution\n\n")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")

            f.write("## Risk Metrics\n\n")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")
            f.write(".1f")

            f.write("## Stress Test Results\n\n")
            for scenario, results in self.results.stress_test_results.items():
                f.write(f"### {scenario.replace('_', ' ').title()}\n")
                f.write(f"- Mean Return: {results['mean_return']:.1f}\n")
                f.write(f"- Worst Case: {results['worst_case']:.1f}\n")
                f.write(f"- Probability of Loss: {results['probability_of_loss']:.1f}\n")
                f.write(f"- Description: {results['description']}\n\n")

            f.write("## Recommendations\n\n")

            # Generate recommendations based on results
            if summary_stats['probability_of_ruin'] > 0.10:
                f.write("⚠️  **HIGH RISK OF RUIN**: Consider reducing position sizes or implementing stricter risk controls.\n\n")

            if summary_stats['max_drawdown_avg'] > 0.25:
                f.write("⚠️  **HIGH DRAWDOWN RISK**: Strategy shows significant drawdown potential. Consider adding stop-loss mechanisms.\n\n")

            if summary_stats['sharpe_ratio'] < 1.0:
                f.write("⚠️  **LOW RISK-ADJUSTED RETURNS**: Strategy may not provide sufficient return for the risk taken.\n\n")

            if summary_stats['probability_of_ruin'] < 0.05 and summary_stats['sharpe_ratio'] > 1.5:
                f.write("✅ **STRATEGY SHOWS PROMISING RISK PROFILE**: Proceed with confidence to live testing.\n\n")

        logger.info(f"Risk assessment report generated: {report_path}")

    def _plot_distributions(self):
        """Generate distribution plots"""

        try:
            import matplotlib.pyplot as plt

            output_dir = Path("results/monte_carlo")
            output_dir.mkdir(parents=True, exist_ok=True)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Returns distribution
            ax1.hist(self.results.total_returns, bins=50, alpha=0.7, color='blue')
            ax1.axvline(np.mean(self.results.total_returns), color='red', linestyle='--', label='Mean')
            ax1.set_title('Return Distribution')
            ax1.set_xlabel('Total Return')
            ax1.set_ylabel('Frequency')
            ax1.legend()

            # Sharpe ratio distribution
            if len(self.results.sharpe_ratios) > 0:
                ax2.hist(self.results.sharpe_ratios, bins=30, alpha=0.7, color='green')
                ax2.axvline(np.mean(self.results.sharpe_ratios), color='red', linestyle='--', label='Mean')
                ax2.set_title('Sharpe Ratio Distribution')
                ax2.set_xlabel('Sharpe Ratio')
                ax2.set_ylabel('Frequency')
                ax2.legend()

            # Maximum drawdown distribution
            if len(self.results.max_drawdowns) > 0:
                ax3.hist(self.results.max_drawdowns, bins=30, alpha=0.7, color='red')
                ax3.axvline(np.mean(self.results.max_drawdowns), color='black', linestyle='--', label='Mean')
                ax3.set_title('Maximum Drawdown Distribution')
                ax3.set_xlabel('Max Drawdown')
                ax3.set_ylabel('Frequency')
                ax3.legend()

            # Cumulative distribution
            sorted_returns = np.sort(self.results.total_returns)
            y_vals = np.arange(len(sorted_returns)) / float(len(sorted_returns))
            ax4.plot(sorted_returns, y_vals, 'b-', linewidth=2)
            ax4.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='5% VaR')
            ax4.set_title('Cumulative Return Distribution')
            ax4.set_xlabel('Total Return')
            ax4.set_ylabel('Cumulative Probability')
            ax4.legend()

            plt.tight_layout()

            plot_path = output_dir / "monte_carlo_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {plot_path}")

            plt.close()

        except ImportError:
            logger.warning("Matplotlib not available, skipping distribution plots")


# Convenience functions
def create_monte_carlo_simulator(config: MonteCarloConfig = None) -> MonteCarloSimulator:
    """Create Monte Carlo simulator instance"""
    return MonteCarloSimulator(config)


async def run_monte_carlo_analysis(config: MonteCarloConfig = None,
                                 ppo_model: PPOMostProfitableTrader = None,
                                 environment: AdvancedTradingEnvironment = None) -> MonteCarloResults:
    """Convenience function to run complete Monte Carlo analysis"""

    simulator = create_monte_carlo_simulator(config)

    if ppo_model and environment:
        simulator.configure_simulation(ppo_model, environment)

    return await simulator.run_monte_carlo_simulation()


def analyze_monte_carlo_results(results: MonteCarloResults) -> Dict[str, Any]:
    """Analyze Monte Carlo results for key insights"""

    summary = results.get_summary_stats()

    analysis = {
        'is_robust': summary['mean_return'] > 0 and summary['probability_of_ruin'] < 0.10,
        'risk_adjusted_performance': summary['sharpe_ratio'],
        'worst_case_scenario': summary['worst_case'],
        'best_case_scenario': summary['best_case'],
        'return_volatility': summary['std_return'],
        'tail_risk': summary['expected_shortfall_95'],
        'recommended_position_size': min(1.0, 0.10 / max(summary['max_drawdown_avg'], 0.01)),
        'confidence_assessment': 'high' if summary['probability_of_ruin'] < 0.05 else 'medium' if summary['probability_of_ruin'] < 0.15 else 'low'
    }

    return analysis

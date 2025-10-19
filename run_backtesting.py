#!/usr/bin/env python3
"""
Advanced Backtesting Framework for Aster DEX Trading Strategies

This script runs comprehensive backtests on historical Aster DEX data using
multiple trading strategies to identify the most profitable approaches for
autonomous trading.

Goal: Find optimal strategies that can generate $1M+ in profits by 2026.
"""

import asyncio
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _viz_available = True
except Exception:
    _viz_available = False
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_trader.backtesting.enhanced_backtester import (
    EnhancedBacktester, BacktestConfig, BacktestResult,
    MovingAverageCrossoverStrategy, RSIStrategy, BollingerBandsStrategy
)
from mcp_trader.backtesting.historical_data_collector import HistoricalDataCollector
from mcp_trader.config import get_settings
from mcp_trader.security.secrets import get_secret_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestingManager:
    """Manages comprehensive backtesting operations."""

    def __init__(self):
        self.settings = get_settings()
        self.backtester = EnhancedBacktester(BacktestConfig())
        self.data_collector = HistoricalDataCollector()
        self.results_dir = Path("results/backtesting")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_all_strategies(self) -> List[Tuple]:
        """Get all available trading strategies with configurations."""
        strategies = []

        # Moving Average strategies
        for fast_ma in [5, 10, 20]:
            for slow_ma in [20, 30, 50]:
                if fast_ma < slow_ma:
                    strategies.append((
                        MovingAverageCrossoverStrategy,
                        {'fast_ma': fast_ma, 'slow_ma': slow_ma}
                    ))

        # RSI strategies
        for rsi_period in [7, 14, 21]:
            for oversold in [25, 30, 35]:
                for overbought in [65, 70, 75]:
                    strategies.append((
                        RSIStrategy,
                        {
                            'rsi_period': rsi_period,
                            'oversold': oversold,
                            'overbought': overbought
                        }
                    ))

        # Bollinger Band strategies
        for bb_period in [10, 20, 30]:
            for bb_std in [1.5, 2.0, 2.5]:
                strategies.append((
                    BollingerBandsStrategy,
                    {
                        'bb_period': bb_period,
                        'bb_std': bb_std
                    }
                ))

        logger.info(f"Generated {len(strategies)} strategy configurations")
        return strategies

    async def load_historical_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting."""
        data = {}

        if symbols is None:
            # Get available symbols from data directory
            data_dir = Path("data/historical")
            if data_dir.exists():
                symbols = [d.name for d in data_dir.iterdir() if d.is_dir()]
            else:
                logger.error("No historical data directory found")
                return {}

        for symbol in symbols:
            # Load 1h data for backtesting
            df = self.data_collector.load_historical_data(symbol, '1h')
            if df is not None and not df.empty:
                data[symbol] = df
                logger.info(f"Loaded {len(df)} hours of data for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")

        return data

    async def run_strategy_comparison(self, symbols: List[str] = None,
                                    max_strategies: int = 50) -> Dict[str, Dict[str, BacktestResult]]:
        """Run comprehensive strategy comparison."""
        logger.info("Starting comprehensive strategy comparison...")

        # Load historical data
        data = await self.load_historical_data(symbols)
        if not data:
            logger.error("No historical data available for backtesting")
            return {}

        # Get strategies
        all_strategies = self.get_all_strategies()
        selected_strategies = all_strategies[:max_strategies]  # Limit for performance

        results = {}

        for symbol, df in data.items():
            logger.info(f"Running backtests for {symbol}...")

            # Run comparison for this symbol
            symbol_results = await self.backtester.compare_strategies(
                selected_strategies, df, [symbol]
            )

            results[symbol] = symbol_results

            # Print top 5 strategies for this symbol
            valid_results = {k: v for k, v in symbol_results.items() if v is not None}
            if valid_results:
                top_5 = sorted(
                    valid_results.items(),
                    key=lambda x: x[1].sharpe_ratio,
                    reverse=True
                )[:5]

                print(f"\n🏆 TOP 5 STRATEGIES FOR {symbol}:")
                for i, (strategy_name, result) in enumerate(top_5, 1):
                    print(
                        f"   {i}. {strategy_name}: "
                        f"Sharpe={result.sharpe_ratio:.2f}, "
                        f"Return={result.total_return:.1%}, "
                        f"WinRate={result.win_rate:.1%}, "
                        f"MaxDD={result.max_drawdown:.1%}, "
                        f"Trades={result.total_trades}"
                    )
        return results

    async def find_optimal_parameters(self, strategy_class, symbol: str,
                                    param_ranges: Dict) -> Dict:
        """Find optimal parameters for a strategy using grid search."""
        logger.info(f"Optimizing parameters for {strategy_class.__name__} on {symbol}")

        # Load data
        data = await self.load_historical_data([symbol])
        if symbol not in data:
            return {}

        df = data[symbol]

        best_result = None
        best_params = {}
        best_sharpe = -float('inf')

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)

        for params in param_combinations:
            try:
                result = await self.backtester.run_backtest(
                    strategy_class, params, df, [symbol]
                )

                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_result = result
                    best_params = params

            except Exception as e:
                logger.debug(f"Parameter combination failed: {params} - {e}")

        logger.info(f"Best parameters for {strategy_class.__name__}: {best_params} "
                   f"(Sharpe: {best_sharpe:.2f})")

        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_combinations_tested': len(param_combinations)
        }

    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools

        keys = list(param_ranges.keys())
        values = [param_ranges[key] for key in keys]

        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))

        return combinations

    def generate_performance_report(self, results: Dict[str, Dict[str, BacktestResult]],
                                  output_file: str = "backtesting_report.html"):
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Aster DEX Backtesting Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <div class="container-fluid mt-4">
                <div class="row">
                    <div class="col-12">
                        <h1 class="text-center mb-4">🚀 Aster DEX Backtesting Report</h1>
                        <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>

                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>🎯 Executive Summary</h2>
                            </div>
                            <div class="card-body">
        """

        # Calculate overall statistics
        all_results = []
        for symbol_results in results.values():
            for result in symbol_results.values():
                if result is not None:
                    all_results.append(result)

        if all_results:
            avg_return = np.mean([r.total_return for r in all_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
            avg_max_dd = np.mean([r.max_drawdown for r in all_results])
            avg_win_rate = np.mean([r.win_rate for r in all_results])

            html_content += f"""
                                <div class="row">
                                    <div class="col-md-3">
                                        <div class="card bg-primary text-white">
                                            <div class="card-body text-center">
                                                <h4>Avg Return</h4>
                                                <h2>{avg_return:.1%}</h2>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card bg-success text-white">
                                            <div class="card-body text-center">
                                                <h4>Avg Sharpe</h4>
                                                <h2>{avg_sharpe:.2f}</h2>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card bg-warning text-white">
                                            <div class="card-body text-center">
                                                <h4>Avg Max DD</h4>
                                                <h2>{avg_max_dd:.1%}</h2>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card bg-info text-white">
                                            <div class="card-body text-center">
                                                <h4>Avg Win Rate</h4>
                                                <h2>{avg_win_rate:.1%}</h2>
                                            </div>
                                        </div>
                                    </div>
                                </div>
            """

        html_content += """
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row mt-4">
        """

        # Add results for each symbol
        for symbol, symbol_results in results.items():
            valid_results = {k: v for k, v in symbol_results.items() if v is not None}

            if valid_results:
                # Sort by Sharpe ratio
                top_strategies = sorted(valid_results.items(),
                                      key=lambda x: x[1].sharpe_ratio,
                                      reverse=True)[:10]

                html_content += f"""
                    <div class="col-md-6 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h3>📊 {symbol} - Top Strategies</h3>
                            </div>
                            <div class="card-body">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Strategy</th>
                                            <th>Return</th>
                                            <th>Sharpe</th>
                                            <th>Max DD</th>
                                            <th>Win Rate</th>
                                            <th>Trades</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                """

                for strategy_name, result in top_strategies:
                    html_content += f"""
                        <tr>
                            <td>{strategy_name}</td>
                            <td class="{'text-success' if result.total_return > 0 else 'text-danger'}">{result.total_return:.1%}</td>
                            <td>{result.sharpe_ratio:.2f}</td>
                            <td class="text-danger">{result.max_drawdown:.1%}</td>
                            <td>{result.win_rate:.1%}</td>
                            <td>{result.total_trades}</td>
                        </tr>
                    """

                html_content += """
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                """

        html_content += """
                </div>

                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h2>🎯 Path to $1M Goal</h2>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-4">
                                        <h4>✅ Completed</h4>
                                        <ul>
                                            <li>Historical data collection</li>
                                            <li>Strategy backtesting framework</li>
                                            <li>Performance analysis</li>
                                        </ul>
                                    </div>
                                    <div class="col-md-4">
                                        <h4>🔄 In Progress</h4>
                                        <ul>
                                            <li>AI model training</li>
                                            <li>Strategy optimization</li>
                                            <li>Risk management</li>
                                        </ul>
                                    </div>
                                    <div class="col-md-4">
                                        <h4>🎯 Next Steps</h4>
                                        <ul>
                                            <li>GCP deployment</li>
                                            <li>Autonomous trading</li>
                                            <li>Profit maximization</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

        # Save report
        with open(self.results_dir / output_file, 'w') as f:
            f.write(html_content)

        logger.info(f"Performance report saved to {self.results_dir / output_file}")


async def main():
    """Main backtesting function."""
    parser = argparse.ArgumentParser(description="Run comprehensive backtesting on Aster DEX data")
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to test')
    parser.add_argument('--max-strategies', type=int, default=20,
                       help='Maximum number of strategies to test')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report from existing results')

    args = parser.parse_args()

    manager = BacktestingManager()

    try:
        if args.report_only:
            # Load existing results and generate report
            logger.info("Generating report from existing results...")
            # This would load saved results
            print("Report generation not yet implemented for saved results")
        else:
            # Run comprehensive backtesting
            results = await manager.run_strategy_comparison(
                symbols=args.symbols,
                max_strategies=args.max_strategies
            )

            if results:
                # Generate performance report
                manager.generate_performance_report(results)

                # Find overall best strategy
                all_results = []
                for symbol_results in results.values():
                    for result in symbol_results.values():
                        if result is not None:
                            all_results.append(result)

                if all_results:
                    best_result = max(all_results, key=lambda x: x.sharpe_ratio)
                    print(f"\n🎉 OVERALL BEST STRATEGY PERFORMANCE:")
                    print(f"   Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
                    print(f"   Total Return: {best_result.total_return:.1%}")
                    print(f"   Win Rate: {best_result.win_rate:.1%}")
                    print(f"   Max Drawdown: {best_result.max_drawdown:.1%}")
                    print(f"   Total Trades: {best_result.total_trades}")

                    print(f"\n💰 PROJECTION TO $1M GOAL:")
                    initial_balance = 10000
                    final_balance = initial_balance * (1 + best_result.total_return)
                    years_to_1m = np.log(1000000 / initial_balance) / np.log(1 + best_result.total_return)
                    print(f"   Current balance: ${initial_balance:,.0f}")
                    print(f"   Best strategy result: ${final_balance:,.0f}")
                    print(f"   Years to $1M at this rate: {years_to_1m:.1f}")

                    if years_to_1m <= 2.5:  # By end of 2026
                        print("   🎯 STATUS: ON TRACK FOR $1M GOAL!")
                    else:
                        print("   ⚠️  STATUS: NEED BETTER STRATEGIES")

        print("\n📊 Backtesting completed!")
        print("📈 Check results/backtesting/backtesting_report.html for detailed analysis")
        print("🤖 Next: Train AI models with the best strategies")
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        print(f"\n❌ Backtesting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

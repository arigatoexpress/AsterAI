#!/usr/bin/env python3
"""
Walk-Forward Analysis for Realistic Trading Strategy Validation

Executes comprehensive walk-forward analysis to provide unbiased performance estimates:
- Rolling training windows with out-of-sample testing
- Statistical significance testing vs benchmarks
- Risk-adjusted performance metrics
- Model adaptation tracking
- Performance decay detection

This is the gold standard for quantitative trading validation.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.backtesting.walk_forward_analysis import (
    create_walk_forward_analyzer,
    run_walk_forward_analysis,
    analyze_walk_forward_results,
    WalkForwardConfig
)
from mcp_trader.ai.ppo_trading_model import PPOConfig
from mcp_trader.ai.trading_environment import TradingConfig
from mcp_trader.ai.ml_training_data_structure import prepare_ml_training_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/walk_forward_analysis.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run walk-forward analysis for trading strategy validation")

    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to walk-forward configuration JSON file'
    )

    parser.add_argument(
        '--model-config',
        type=str,
        default='volatile',
        choices=['standard', 'volatile', 'hft'],
        help='PPO model configuration (default: volatile)'
    )

    parser.add_argument(
        '--training-window-days',
        type=int,
        default=365,
        help='Initial training window in days (default: 365)'
    )

    parser.add_argument(
        '--walk-step-days',
        type=int,
        default=30,
        help='Walk-forward step size in days (default: 30)'
    )

    parser.add_argument(
        '--test-window-days',
        type=int,
        default=30,
        help='Test window size in days (default: 30)'
    )

    parser.add_argument(
        '--analysis-years',
        type=float,
        default=2.0,
        help='Total analysis period in years (default: 2.0)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/walk_forward',
        help='Output directory for results (default: results/walk_forward)'
    )

    parser.add_argument(
        '--benchmark-comparison',
        action='store_true',
        default=True,
        help='Compare against buy-and-hold benchmark (default: True)'
    )

    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Statistical confidence level (default: 0.95)'
    )

    return parser.parse_args()


def create_walk_forward_config(args) -> WalkForwardConfig:
    """Create walk-forward configuration from arguments"""

    config = WalkForwardConfig(
        initial_training_window_days=args.training_window_days,
        walk_forward_step_days=args.walk_step_days,
        test_window_days=args.test_window_days,
        total_analysis_period_years=args.analysis_years,
        parallel_processing=args.parallel,
        max_workers=args.max_workers,
        benchmark_comparison=args.benchmark_comparison,
        confidence_level=args.confidence_level
    )

    return config


def create_model_configs(model_type: str) -> Tuple[PPOConfig, TradingConfig]:
    """Create model and environment configurations"""

    if model_type == 'volatile':
        # Optimized for volatile bull market downturns
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            gamma=0.95,  # Shorter horizon for volatile markets
            gae_lambda=0.90,
            clip_ratio=0.15,  # Tighter clipping for stability
            value_loss_coef=0.7,
            entropy_coef=0.02,
            adaptive_lr=True,
            risk_aware_updates=True,
            num_episodes=50  # Shorter training for walk-forward
        )

        env_config = TradingConfig(
            reward_function='sortino',
            max_drawdown_limit=0.10,
            stop_loss_threshold=0.02,
            take_profit_threshold=0.03,
            adaptive_reward=True,
            symbols=['BTC', 'ETH', 'ADA', 'SOL']
        )

    elif model_type == 'hft':
        # Optimized for high-frequency trading
        ppo_config = PPOConfig(
            learning_rate=5e-4,
            gamma=0.99,  # Longer horizon for HFT
            gae_lambda=0.95,
            clip_ratio=0.25,
            value_loss_coef=0.5,
            entropy_coef=0.05,  # Higher exploration
            epochs_per_update=5,
            num_episodes=30
        )

        env_config = TradingConfig(
            step_size_minutes=1,
            enable_hft=True,
            max_orders_per_step=10,
            enable_market_microstructure=True,
            order_book_depth=20,
            maker_fee=0.0001,
            taker_fee=0.0003,
            symbols=['BTC', 'ETH']
        )

    else:  # standard
        ppo_config = PPOConfig(num_episodes=50)
        env_config = TradingConfig()

    return ppo_config, env_config


async def prepare_walk_forward_data() -> Dict[str, Any]:
    """Prepare data for walk-forward analysis"""

    logger.info("Preparing walk-forward analysis data...")

    # This would prepare historical data segmented by time periods
    # For now, return placeholder
    training_data = {
        'train': None,  # Would be MLTrainingDataset
        'val': None,
        'test': None
    }

    logger.info("Walk-forward data preparation completed")

    return training_data


async def run_complete_walk_forward_analysis(args) -> Dict[str, Any]:
    """Run complete walk-forward analysis"""

    logger.info("="*70)
    logger.info("STARTING WALK-FORWARD ANALYSIS FOR TRADING STRATEGY VALIDATION")
    logger.info("="*70)

    # Create configurations
    wf_config = create_walk_forward_config(args)
    ppo_config, env_config = create_model_configs(args.model_config)

    logger.info(f"Model Type: {args.model_config}")
    logger.info(f"Training Window: {wf_config.initial_training_window_days} days")
    logger.info(f"Walk Step: {wf_config.walk_forward_step_days} days")
    logger.info(f"Test Window: {wf_config.test_window_days} days")
    logger.info(f"Analysis Period: {wf_config.total_analysis_period_years} years")
    logger.info(f"Parallel Processing: {wf_config.parallel_processing}")
    logger.info("")

    # Prepare training data
    training_data = await prepare_walk_forward_data()

    # Create and run analyzer
    analyzer = create_walk_forward_analyzer(wf_config)
    analyzer.configure_analysis(ppo_config, env_config, training_data)

    # Execute walk-forward analysis
    results = await analyzer.run_walk_forward_analysis()

    # Analyze results
    analysis = analyze_walk_forward_results(results)

    # Save comprehensive results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis summary
    analysis_path = output_dir / "analysis_summary.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            'configuration': {
                'model_type': args.model_config,
                'walk_forward_config': wf_config.__dict__,
                'ppo_config': ppo_config.__dict__,
                'env_config': env_config.__dict__
            },
            'results': {
                'total_steps': results.total_steps,
                'average_test_score': results.average_test_score,
                'average_benchmark_score': results.average_benchmark_score,
                'overall_sharpe_ratio': results.overall_sharpe_ratio,
                'overall_max_drawdown': results.overall_max_drawdown,
                'p_value_vs_benchmark': results.p_value_vs_benchmark,
                'computation_time_seconds': results.total_computation_time
            },
            'analysis': analysis,
            'timestamp': str(results.end_time)
        }, f, indent=2, default=str)

    logger.info(f"Analysis summary saved to {analysis_path}")

    return {
        'results': results,
        'analysis': analysis,
        'config': wf_config
    }


def print_walk_forward_summary(result: Dict[str, Any]):
    """Print comprehensive walk-forward analysis summary"""

    results = result['results']
    analysis = result['analysis']

    print("\n" + "="*70)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("="*70)

    print("📊 PERFORMANCE SUMMARY")
    print(f"   Total Steps: {results.total_steps}")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("
🎯 STATISTICAL SIGNIFICANCE")
    print(".4f")
    print(f"   Confidence Level: {results.config.confidence_level:.1%}")
    print(f"   Significant vs Benchmark: {'✅' if analysis['is_statistically_significant'] else '❌'}")

    print("
📈 RISK METRICS")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".1f")

    print("
🔍 STRATEGY ANALYSIS")
    print(f"   Is Profitable: {'✅' if analysis['is_profitable'] else '❌'}")
    print(f"   Outperforms Benchmark: {'✅' if analysis['outperforms_benchmark'] else '❌'}")
    print(".3f")
    print(".1f")

    print("
📋 RECOMMENDATIONS")

    if analysis['is_profitable'] and analysis['is_statistically_significant']:
        print("   ✅ STRATEGY SHOWS PROMISING RESULTS")
        print("      - Proceed to Monte Carlo stress testing")
        print("      - Consider paper trading validation")
    elif analysis['is_profitable'] and not analysis['is_statistically_significant']:
        print("   ⚠️  STRATEGY IS PROFITABLE BUT NOT STATISTICALLY SIGNIFICANT")
        print("      - Increase sample size or analysis period")
        print("      - Review parameter sensitivity")
    else:
        print("   ❌ STRATEGY DOES NOT SHOW PROFITABLE RESULTS")
        print("      - Review strategy logic and parameters")
        print("      - Consider alternative approaches")

    if analysis['maximum_drawdown'] > 0.20:
        print("   ⚠️  HIGH DRAWDOWN DETECTED - CONSIDER RISK MANAGEMENT IMPROVEMENTS")

    print("
🚀 NEXT STEPS")
    print("   1. Review detailed results in results/walk_forward/")
    print("   2. Run Monte Carlo simulation: python scripts/monte_carlo_simulation.py")
    print("   3. Analyze parameter sensitivity")
    print("   4. Optimize strategy parameters")
    print("   5. Proceed to paper trading if results are satisfactory")

    print("\n" + "="*70)


async def main():
    """Main walk-forward analysis function"""
    args = parse_arguments()

    try:
        # Run complete analysis
        result = await run_complete_walk_forward_analysis(args)

        # Print summary
        print_walk_forward_summary(result)

        # Determine exit code based on results
        analysis = result['analysis']
        if analysis['is_profitable'] and analysis['is_statistically_significant']:
            logger.info("Walk-forward analysis completed successfully")
            return 0
        else:
            logger.warning("Walk-forward analysis shows concerning results")
            return 1

    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

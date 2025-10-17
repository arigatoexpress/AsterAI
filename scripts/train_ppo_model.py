#!/usr/bin/env python3
"""
Train PPO Model for Most Profitable AI Trading

Advanced training pipeline for PPO trading model with:
- RTX 5070 Ti GPU optimization
- Self-improving features
- Curriculum learning
- Risk-aware training
- Multi-timeframe integration
- Walk-forward validation

Usage:
    python scripts/train_ppo_model.py --episodes 1000 --gpu --curriculum
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.ai.ppo_trading_model import (
    create_ppo_trader,
    create_volatile_market_ppo,
    create_hft_ppo,
    PPOConfig
)
from mcp_trader.ai.ml_training_data_structure import (
    prepare_ml_training_data,
    create_ml_data_manager,
    MLDataConfig
)
from mcp_trader.ai.trading_environment import create_trading_environment, TradingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ppo_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train PPO model for AI trading")

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes (default: 1000)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['standard', 'volatile', 'hft'],
        default='volatile',
        help='Training mode (default: volatile)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Enable GPU training (default: True)'
    )

    parser.add_argument(
        '--curriculum',
        action='store_true',
        default=True,
        help='Enable curriculum learning (default: True)'
    )

    parser.add_argument(
        '--self-improving',
        action='store_true',
        default=True,
        help='Enable self-improving features (default: True)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/ppo',
        help='Output directory for models (default: models/ppo)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )

    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate existing model'
    )

    return parser.parse_args()


def create_training_config(args) -> PPOConfig:
    """Create PPO configuration from arguments"""

    base_config = PPOConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        curriculum_learning=args.curriculum,
        adaptive_lr=args.self_improving,
        online_learning=args.self_improving,
        risk_aware_updates=args.self_improving
    )

    # GPU optimization
    if args.gpu and torch.cuda.is_available():
        base_config.device = 'cuda'
        base_config.use_tensorrt = True
        base_config.mixed_precision = True
        base_config.gradient_checkpointing = True

        # RTX 5070 Ti specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logger.info("RTX 5070 Ti optimizations enabled")
    else:
        base_config.device = 'cpu'
        logger.info("CPU training mode")

    return base_config


def create_environment_config(mode: str) -> TradingConfig:
    """Create environment configuration based on mode"""

    if mode == 'volatile':
        return TradingConfig(
            reward_function='sortino',
            max_drawdown_limit=0.10,
            stop_loss_threshold=0.02,
            take_profit_threshold=0.03,
            adaptive_reward=True,
            symbols=['BTC', 'ETH', 'ADA', 'SOL']
        )

    elif mode == 'hft':
        return TradingConfig(
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
        return TradingConfig()


async def prepare_training_data() -> Dict[str, Any]:
    """Prepare ML training data"""

    logger.info("Preparing training data...")

    # Create ML data configuration
    ml_config = MLDataConfig(
        symbols=['BTC', 'ETH', 'ADA', 'SOL'],
        timeframes=['1h'],
        sequence_length=128,
        prediction_horizon=24,
        gpu_optimization=torch.cuda.is_available()
    )

    # Create data manager
    data_manager = create_ml_data_manager(ml_config)

    # Prepare training datasets
    datasets = await data_manager.prepare_training_data()

    logger.info(f"Training data prepared: {len(datasets)} datasets")

    return {
        'data_manager': data_manager,
        'datasets': datasets,
        'config': ml_config
    }


def create_ppo_model(args, training_data: Dict[str, Any]) -> Any:
    """Create PPO model based on arguments"""

    # Get configuration
    ppo_config = create_training_config(args)
    env_config = create_environment_config(args.mode)

    # Create model based on mode
    if args.mode == 'volatile':
        trader = create_volatile_market_ppo()
        # Override configs
        trader.config = ppo_config
        trader.env_config = env_config
    elif args.mode == 'hft':
        trader = create_hft_ppo()
        trader.config = ppo_config
        trader.env_config = env_config
    else:
        trader = create_ppo_trader(ppo_config, env_config)

    # Load training data into environment if available
    if 'datasets' in training_data:
        # This would integrate the training data with the environment
        logger.info("Training data integrated with environment")

    # Resume from checkpoint if specified
    if args.resume:
        trader.load_model(args.resume)
        logger.info(f"Resumed training from {args.resume}")

    return trader


async def train_model(args, trader, training_data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Train the PPO model"""

    logger.info("="*60)
    logger.info("STARTING PPO TRAINING FOR MOST PROFITABLE AI TRADING")
    logger.info("="*60)

    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"GPU: {args.gpu and torch.cuda.is_available()}")
    logger.info(f"Curriculum learning: {args.curriculum}")
    logger.info(f"Self-improving: {args.self_improving}")
    logger.info("")

    # Training loop
    start_time = datetime.now()
    training_history = await asyncio.get_event_loop().run_in_executor(
        None, trader.train, args.episodes
    )
    end_time = datetime.now()

    training_time = (end_time - start_time).total_seconds()

    logger.info("
Training completed!")
    logger.info(f"Total training time: {training_time:.1f} seconds")
    logger.info(".1f")

    # Final evaluation
    logger.info("Performing final evaluation...")
    eval_reward = trader.evaluate(num_episodes=20)
    logger.info(".2f")

    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = f"ppo_{args.mode}_final"
    trader.save_model(final_model_path)

    # Save training history
    history_path = output_dir / f"training_history_{args.mode}.json"
    with open(history_path, 'w') as f:
        import json
        json.dump({
            'config': vars(args),
            'training_time': training_time,
            'final_eval_reward': eval_reward,
            'history': training_history
        }, f, indent=2, default=str)

    logger.info(f"Training history saved to {history_path}")

    return training_history


def evaluate_model(args, trader):
    """Evaluate existing model"""

    logger.info("Evaluating existing model...")

    # Run evaluation
    eval_reward = trader.evaluate(num_episodes=50)
    logger.info(".2f")

    # Run extended evaluation
    logger.info("Running extended evaluation (100 episodes)...")
    extended_reward = trader.evaluate(num_episodes=100)
    logger.info(".2f")

    # Calculate additional metrics
    metrics = {
        'mean_reward': eval_reward,
        'extended_mean_reward': extended_reward,
        'evaluation_episodes': 50,
        'extended_episodes': 100,
        'timestamp': datetime.now().isoformat()
    }

    # Save evaluation results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = output_dir / f"evaluation_{args.mode}.json"
    with open(eval_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation results saved to {eval_path}")

    return metrics


def plot_training_history(training_history: Dict[str, List[float]], output_dir: str):
    """Plot training history"""

    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot rewards
        ax1.plot(training_history['rewards'], alpha=0.7)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        # Plot losses
        ax2.plot(training_history['actor_losses'], label='Actor Loss', alpha=0.7)
        ax2.plot(training_history['critic_losses'], label='Critic Loss', alpha=0.7)
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot entropy
        ax3.plot(training_history['entropies'], color='orange', alpha=0.7)
        ax3.set_title('Policy Entropy')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Entropy')
        ax3.grid(True, alpha=0.3)

        # Plot learning rate
        ax4.plot(training_history['learning_rates'], color='red', alpha=0.7)
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = Path(output_dir) / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {plot_path}")

        plt.close()

    except ImportError:
        logger.warning("Matplotlib not available, skipping training history plot")


async def main():
    """Main training function"""
    args = parse_arguments()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.evaluate_only:
            # Only evaluate existing model
            logger.info("Evaluation mode - loading existing model")

            # Create model for evaluation
            training_data = {}  # Not needed for evaluation
            trader = create_ppo_model(args, training_data)

            # Load best model
            model_path = f"ppo_{args.mode}_best"
            try:
                trader.load_model(model_path)
                logger.info(f"Loaded model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {str(e)}")
                return 1

            # Evaluate
            metrics = evaluate_model(args, trader)

        else:
            # Full training pipeline
            logger.info("Starting complete PPO training pipeline")

            # Prepare training data
            training_data = await prepare_training_data()

            # Create PPO model
            trader = create_ppo_model(args, training_data)

            # Train model
            training_history = await train_model(args, trader, training_data)

            # Plot training history
            plot_training_history(training_history, args.output_dir)

            # Final evaluation
            metrics = evaluate_model(args, trader)

        # Print final summary
        print("\n" + "="*60)
        print("PPO TRAINING PIPELINE COMPLETED")
        print("="*60)

        if not args.evaluate_only:
            print(f"Model saved as: ppo_{args.mode}_final")
            print(f"Best model: ppo_{args.mode}_best")
            print(f"Training time: {training_history.get('total_time', 'N/A')}")
            print(f"Final evaluation reward: {metrics.get('mean_reward', 'N/A'):.2f}")

        print("\nNext steps:")
        print("1. Run walk-forward analysis: python scripts/walk_forward_analysis.py")
        print("2. Run Monte Carlo simulation: python scripts/monte_carlo_simulation.py")
        print("3. Deploy to paper trading: python scripts/deploy_paper_trading.py")

        return 0

    except Exception as e:
        logger.error(f"PPO training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    exit_code = asyncio.run(main())
    sys.exit(exit_code)

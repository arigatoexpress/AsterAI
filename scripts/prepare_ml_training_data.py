#!/usr/bin/env python3
"""
Prepare ML Training Data for Most Profitable AI Trading System

This script prepares comprehensive training data using the self-improving
ML data structure, integrating multi-modal data sources for optimal AI training.

Features:
- Multi-modal data integration (price, technical, sentiment, macro, alternative)
- Self-improving feature engineering
- GPU-optimized data loading
- Walk-forward validation support
- Real-time data streaming capabilities
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

from mcp_trader.ai.ml_training_data_structure import (
    prepare_ml_training_data,
    create_ml_data_manager,
    MLDataConfig,
    SelfImprovingMLDataManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/ml_data_preparation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare ML training data for AI trading")

    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to custom configuration JSON file'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Specific symbols to prepare data for'
    )

    parser.add_argument(
        '--timeframes',
        type=str,
        nargs='+',
        default=['1h'],
        help='Timeframes to prepare (default: 1h)'
    )

    parser.add_argument(
        '--sequence-length',
        type=int,
        default=128,
        help='Sequence length for training samples (default: 128)'
    )

    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=24,
        help='Prediction horizon in hours (default: 24)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ml_training',
        help='Output directory for prepared data'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for data loading (default: 32)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing data without preparing new datasets'
    )

    parser.add_argument(
        '--gpu-optimization',
        action='store_true',
        default=True,
        help='Enable GPU optimizations (default: True)'
    )

    return parser.parse_args()


def load_custom_config(config_file: str) -> Optional[MLDataConfig]:
    """Load custom configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        config = MLDataConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        logger.info(f"Loaded custom configuration from {config_file}")
        return config

    except Exception as e:
        logger.warning(f"Failed to load custom config {config_file}: {str(e)}")
        return None


def create_config_from_args(args) -> MLDataConfig:
    """Create configuration from command line arguments"""
    config = MLDataConfig()

    if args.symbols:
        config.symbols = args.symbols

    if args.timeframes:
        config.timeframes = args.timeframes

    config.sequence_length = args.sequence_length
    config.prediction_horizon = args.prediction_horizon
    config.gpu_optimization = args.gpu_optimization

    return config


async def prepare_and_validate_data(config: MLDataConfig, output_dir: str) -> Dict[str, Any]:
    """Prepare ML training data and validate quality"""

    logger.info("="*70)
    logger.info("PREPARING ML TRAINING DATA FOR MOST PROFITABLE AI TRADING")
    logger.info("="*70)

    # Create data manager
    data_manager = create_ml_data_manager(config)

    try:
        # Prepare training datasets
        logger.info("Starting data preparation...")
        datasets = await data_manager.prepare_training_data()

        if not datasets:
            logger.error("No datasets were created")
            return {'success': False, 'error': 'No datasets created'}

        # Create data loaders
        data_loaders = {}
        for split_name, dataset in datasets.items():
            data_loader = data_manager.create_data_loader(dataset, batch_size=32)
            data_loaders[split_name] = data_loader

        # Validate data quality
        quality_report = data_manager.get_data_quality_report()

        # Log summary
        logger.info("Data Preparation Summary:")
        logger.info(f"  Overall Quality Score: {quality_report['overall_quality']:.3f}")
        logger.info(f"  Assets Processed: {len(quality_report['asset_quality'])}")
        logger.info(f"  Training Samples: {len(datasets.get('train', []))}")
        logger.info(f"  Validation Samples: {len(datasets.get('val', []))}")
        logger.info(f"  Test Samples: {len(datasets.get('test', []))}")

        # Save datasets and metadata
        await save_prepared_data(datasets, data_loaders, quality_report, output_dir, config)

        return {
            'success': True,
            'datasets': datasets,
            'data_loaders': data_loaders,
            'quality_report': quality_report,
            'config': config.__dict__
        }

    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}


async def save_prepared_data(datasets: Dict, data_loaders: Dict, quality_report: Dict,
                           output_dir: str, config: MLDataConfig):
    """Save prepared data to disk"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save metadata
        metadata = {
            'creation_timestamp': asyncio.get_event_loop().time(),
            'config': config.__dict__,
            'quality_report': quality_report,
            'dataset_sizes': {name: len(dataset) for name, dataset in datasets.items()},
            'feature_info': {}
        }

        # Extract feature information from first sample
        if datasets:
            first_dataset = next(iter(datasets.values()))
            if len(first_dataset) > 0:
                sample = first_dataset[0]
                metadata['feature_info'] = {
                    'feature_shape': sample['features'].shape,
                    'target_keys': list(sample.keys()),
                    'sample_keys': list(sample.keys())
                }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save quality report separately
        quality_path = output_path / "quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Saved quality report to {quality_path}")

        # Note: In production, you might want to save the actual datasets
        # but for now, we keep them in memory for training

    except Exception as e:
        logger.error(f"Failed to save prepared data: {str(e)}")


async def validate_existing_data(output_dir: str) -> Dict[str, Any]:
    """Validate existing prepared data"""

    output_path = Path(output_dir)

    if not output_path.exists():
        return {'success': False, 'error': 'Data directory does not exist'}

    try:
        # Load metadata
        metadata_path = output_path / "metadata.json"
        if not metadata_path.exists():
            return {'success': False, 'error': 'Metadata file not found'}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load quality report
        quality_path = output_path / "quality_report.json"
        quality_report = {}
        if quality_path.exists():
            with open(quality_path, 'r') as f:
                quality_report = json.load(f)

        # Validate data integrity
        validation_results = {
            'metadata_loaded': True,
            'quality_report_loaded': bool(quality_report),
            'overall_quality': quality_report.get('overall_quality', 0),
            'assets_count': len(quality_report.get('asset_quality', {})),
            'config_valid': bool(metadata.get('config'))
        }

        logger.info("Existing data validation:")
        logger.info(f"  Overall Quality: {validation_results['overall_quality']:.3f}")
        logger.info(f"  Assets: {validation_results['assets_count']}")
        logger.info(f"  Metadata: {'✓' if validation_results['metadata_loaded'] else '✗'}")
        logger.info(f"  Quality Report: {'✓' if validation_results['quality_report_loaded'] else '✗'}")

        return {
            'success': True,
            'validation_results': validation_results,
            'metadata': metadata,
            'quality_report': quality_report
        }

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return {'success': False, 'error': str(e)}


def print_data_summary(result: Dict[str, Any]):
    """Print comprehensive data summary"""

    if not result.get('success', False):
        logger.error(f"Data preparation failed: {result.get('error', 'Unknown error')}")
        return

    print("\n" + "="*70)
    print("ML TRAINING DATA PREPARATION COMPLETE")
    print("="*70)

    quality_report = result.get('quality_report', {})

    print("📊 DATA QUALITY SUMMARY")
    print(f"   Overall Quality Score: {quality_report.get('overall_quality', 0):.3f}")
    print(f"   Assets Processed: {len(quality_report.get('asset_quality', {}))}")

    print("\n🎯 TRAINING DATASETS")
    datasets = result.get('datasets', {})
    for split_name, dataset in datasets.items():
        print(f"   {split_name.upper()}: {len(dataset)} samples")

    print("\n🔧 FEATURES & CONFIGURATION")
    config = result.get('config', {})
    print(f"   Sequence Length: {config.get('sequence_length', 'N/A')}")
    print(f"   Prediction Horizon: {config.get('prediction_horizon', 'N/A')} hours")
    print(f"   GPU Optimization: {'✓' if config.get('gpu_optimization', False) else '✗'}")
    print(f"   Online Learning: {'✓' if config.get('online_learning_enabled', False) else '✗'}")

    print("\n💡 SELF-IMPROVING FEATURES")
    feature_engineering = quality_report.get('feature_engineering', {})
    print(f"   Adaptive Feature Selection: {'✓' if feature_engineering.get('adaptive_selection_enabled', False) else '✗'}")
    online_learning = quality_report.get('online_learning', {})
    print(f"   Online Learning Buffer: {online_learning.get('buffer_size', 0)} samples")
    print(f"   Performance History: {online_learning.get('performance_history_length', 0)} entries")

    print("\n✅ NEXT STEPS")
    print("   1. Run PPO model training: python scripts/train_ppo_model.py")
    print("   2. Implement walk-forward analysis: python scripts/walk_forward_analysis.py")
    print("   3. Run Monte Carlo simulations: python scripts/monte_carlo_simulation.py")
    print("   4. Integrate VPIN calculation: python scripts/integrate_vpin.py")

    print("\n🚀 SYSTEM READY FOR MOST PROFITABLE AI TRADING!")


async def main():
    """Main data preparation function"""
    args = parse_arguments()

    # Setup logging directory
    Path('logs').mkdir(exist_ok=True)

    try:
        if args.validate_only:
            # Only validate existing data
            logger.info("Validating existing ML training data...")
            result = await validate_existing_data(args.output_dir)
        else:
            # Prepare new data
            # Load configuration
            config = None
            if args.config_file:
                config = load_custom_config(args.config_file)

            if config is None:
                config = create_config_from_args(args)

            # Prepare data
            result = await prepare_and_validate_data(config, args.output_dir)

        # Print summary
        print_data_summary(result)

        return 0 if result.get('success', False) else 1

    except Exception as e:
        logger.error(f"ML data preparation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

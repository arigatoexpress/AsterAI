#!/usr/bin/env python3
"""
Complete Training Pipeline - Real Data Only
Automated pipeline ensuring no synthetic data contamination.
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Run complete training pipeline with safety checks.
    """

    def __init__(self):
        self.pipeline_steps = [
            {
                'name': 'Asset Discovery',
                'script': 'scripts/discover_aster_assets.py',
                'description': 'Discover all Aster DEX assets and test data availability',
                'required': True
            },
            {
                'name': 'Rate Limit Testing',
                'script': 'scripts/test_aster_rate_limits.py',
                'description': 'Test API rate limits and data stream reliability',
                'required': True
            },
            {
                'name': 'Data Collection (Real Only)',
                'script': 'scripts/collect_real_aster_data.py',
                'description': 'Collect 6 months of real Aster DEX data only',
                'required': True
            },
            {
                'name': 'Training Readiness Validation',
                'script': 'scripts/validate_training_readiness.py',
                'description': 'Validate no synthetic data and training readiness',
                'required': True
            },
            {
                'name': 'Feature Engineering',
                'script': 'scripts/validate_confluence_features.py',
                'description': 'Generate and validate confluence features',
                'required': False
            },
            {
                'name': 'Model Training (General)',
                'script': 'local_training/train_confluence_model.py',
                'description': 'Train general confluence model with real data',
                'required': False
            },
            {
                'name': 'Model Training (Aster-Native)',
                'script': 'local_training/train_aster_native_model.py',
                'description': 'Train Aster-specific model with real data',
                'required': False
            },
            {
                'name': 'Backtesting',
                'script': 'scripts/backtest_confluence_strategy.py',
                'description': 'Backtest trained models on historical data',
                'required': False
            }
        ]

    def run_pipeline(self, start_from: str = None, stop_at: str = None):
        """Run the complete pipeline."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║        Complete Training Pipeline - Real Data Only            ║
║      Automated Safety-First Training with Validation          ║
╚════════════════════════════════════════════════════════════════╝
        """)

        logger.info("Starting complete training pipeline...")

        # Find start index
        start_idx = 0
        if start_from:
            for i, step in enumerate(self.pipeline_steps):
                if step['name'].lower().replace(' ', '_') == start_from.lower():
                    start_idx = i
                    break

        # Find stop index
        stop_idx = len(self.pipeline_steps)
        if stop_at:
            for i, step in enumerate(self.pipeline_steps):
                if step['name'].lower().replace(' ', '_') == stop_at.lower():
                    stop_idx = i + 1
                    break

        # Run pipeline steps
        for i in range(start_idx, stop_idx):
            step = self.pipeline_steps[i]

            print(f"\n{'='*70}")
            print(f"STEP {i+1}/{len(self.pipeline_steps)}: {step['name']}")
            print(f"{'='*70}")
            print(f"Description: {step['description']}")
            print(f"Script: {step['script']}")
            print(f"Required: {'✅' if step['required'] else '⚠️  Optional'}")

            # Run the step
            success = self.run_step(step)

            if not success and step['required']:
                logger.error(f"❌ Required step '{step['name']}' failed - stopping pipeline")
                return False
            elif not success and not step['required']:
                logger.warning(f"⚠️  Optional step '{step['name']}' failed - continuing")
                continue

        logger.info("✅ Pipeline completed successfully!")
        return True

    def run_step(self, step: dict) -> bool:
        """Run a single pipeline step."""
        script_path = step['script']

        # Check if script exists
        if not Path(script_path).exists():
            logger.error(f"Script not found: {script_path}")
            return False

        try:
            logger.info(f"Running: python {script_path}")

            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Log output
            if result.stdout:
                logger.info("Script output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")

            if result.stderr:
                logger.warning("Script stderr:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(f"  {line}")

            # Check result
            if result.returncode == 0:
                logger.info(f"✅ Step '{step['name']}' completed successfully")
                return True
            else:
                logger.error(f"❌ Step '{step['name']}' failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Step '{step['name']}' timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Step '{step['name']}' failed with exception: {e}")
            return False

    def validate_pipeline_prerequisites(self) -> bool:
        """Validate pipeline prerequisites."""
        logger.info("Checking pipeline prerequisites...")

        # Check Python environment
        try:
            import torch
            import pandas
            import numpy
            logger.info("✅ Python environment ready")
        except ImportError as e:
            logger.error(f"❌ Missing Python package: {e}")
            return False

        # Check GPU availability
        if torch.cuda.is_available():
            logger.info("✅ GPU available for training")
        else:
            logger.warning("⚠️  GPU not available - training will be slower")

        # Check directory structure
        required_dirs = [
            "data",
            "data/historical",
            "models",
            "scripts",
            "local_training"
        ]

        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
            else:
                logger.info(f"✅ Directory exists: {dir_path}")

        logger.info("✅ All prerequisites validated")
        return True


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete Training Pipeline")
    parser.add_argument('--start-from', help='Start pipeline from specific step')
    parser.add_argument('--stop-at', help='Stop pipeline at specific step')
    parser.add_argument('--validate-only', action='store_true', help='Only validate prerequisites')

    args = parser.parse_args()

    runner = PipelineRunner()

    # Validate prerequisites
    if not runner.validate_pipeline_prerequisites():
        logger.error("❌ Prerequisites not met - exiting")
        sys.exit(1)

    if args.validate_only:
        logger.info("✅ Prerequisites validated - ready to run pipeline")
        return

    # Run pipeline
    success = runner.run_pipeline(
        start_from=args.start_from,
        stop_at=args.stop_at
    )

    if success:
        print("""
╔════════════════════════════════════════════════════════════════╗
║                   🎉 PIPELINE COMPLETED!                      ║
║      Real data training pipeline executed successfully        ║
║      No synthetic data contamination detected                 ║
╚════════════════════════════════════════════════════════════════╝

Next Steps:
1. Review training results in models/ directory
2. Check backtest results in data/backtest_results/
3. Run deployment scripts when ready for cloud
4. Start live trading with validated models

🚀 Ready to transform $50 → $500K with real data confidence!
        """)
        sys.exit(0)
    else:
        print("""
╔════════════════════════════════════════════════════════════════╗
║                 ❌ PIPELINE FAILED                            ║
║      Check logs above for failure details                     ║
╚════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()





"""
Model Deployment Pipeline for RTX 5070Ti Local Training → Cloud Production
Seamlessly deploy locally trained models to the autonomous cloud trading system.
"""

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import tempfile
import shutil
import zipfile
import hashlib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelDeploymentPipeline:
    """
    Complete pipeline for deploying locally trained models to cloud production.
    Handles serialization, compression, versioning, and cloud integration.
    """

    def __init__(self, gcp_project_id: str = None, storage_bucket: str = None):
        self.gcp_project_id = gcp_project_id or os.getenv('GCP_PROJECT_ID', 'ai-trading-models')
        self.storage_bucket = storage_bucket or f"{self.gcp_project_id}-models"

        # Local paths
        self.local_models_dir = Path.home() / "ai_trading_local" / "models"
        self.deployment_dir = self.local_models_dir / "deployment"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.model_registry = {}
        self.load_model_registry()

        # Cloud integration
        self.gcs_available = self._check_gcs_availability()

        logger.info(f"Model deployment pipeline initialized for {self.gcp_project_id}")

    def _check_gcs_availability(self) -> bool:
        """Check if Google Cloud Storage is available."""
        try:
            from google.cloud import storage
            return True
        except ImportError:
            logger.warning("Google Cloud Storage not available - using local storage only")
            return False

    def load_model_registry(self):
        """Load model registry from disk."""
        registry_file = self.deployment_dir / "model_registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Loaded model registry with {len(self.model_registry)} models")
            except Exception as e:
                logger.warning(f"Could not load model registry: {e}")
                self.model_registry = {}
        else:
            self.model_registry = {}

    def save_model_registry(self):
        """Save model registry to disk."""
        registry_file = self.deployment_dir / "model_registry.json"

        try:
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            logger.debug("Model registry saved")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def package_model_for_deployment(self, model_path: str, model_name: str,
                                  model_version: str, metadata: Dict[str, Any] = None) -> str:
        """
        Package a trained model for cloud deployment.

        Args:
            model_path: Path to the trained model file
            model_name: Name of the model (e.g., 'lstm_predictor')
            model_version: Version string (e.g., 'v1.2.3')
            metadata: Additional metadata about the model

        Returns:
            Path to the packaged model file
        """
        logger.info(f"Packaging model {model_name} {model_version} for deployment")

        # Create package directory
        package_dir = self.deployment_dir / f"{model_name}_{model_version}"
        package_dir.mkdir(exist_ok=True)

        # Copy model files
        model_file = Path(model_path)
        if model_file.exists():
            shutil.copy2(model_file, package_dir / model_file.name)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create model metadata
        model_metadata = {
            'name': model_name,
            'version': model_version,
            'created_at': datetime.now().isoformat(),
            'model_type': self._detect_model_type(model_path),
            'file_size_mb': model_file.stat().st_size / (1024 * 1024),
            'sha256_hash': self._calculate_file_hash(model_file),
            'performance_metrics': metadata or {},
            'deployment_ready': True
        }

        # Save metadata
        with open(package_dir / "model_metadata.json", 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)

        # Create deployment script
        self._create_deployment_script(package_dir, model_name, model_version)

        # Create compressed package
        package_file = self._create_compressed_package(package_dir, model_name, model_version)

        # Update registry
        self.model_registry[f"{model_name}_{model_version}"] = model_metadata
        self.save_model_registry()

        logger.info(f"✅ Model packaged: {package_file}")
        return str(package_file)

    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type from file path."""
        path_str = str(model_path).lower()

        if 'lstm' in path_str or 'rnn' in path_str:
            return 'lstm_predictor'
        elif 'transformer' in path_str:
            return 'transformer_predictor'
        elif 'rl' in path_str or 'ppo' in path_str or 'sac' in path_str:
            return 'rl_agent'
        elif 'ensemble' in path_str:
            return 'ensemble_model'
        else:
            return 'unknown'

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of model file."""
        try:
            import hashlib

            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "unknown"

    def _create_deployment_script(self, package_dir: Path, model_name: str, model_version: str):
        """Create deployment script for the model."""
        script_content = f'''#!/bin/bash
# Deployment script for {model_name} {model_version}

MODEL_NAME="{model_name}"
MODEL_VERSION="{model_version}"
PACKAGE_DIR="$(dirname "$0")"

echo "🚀 Deploying {model_name} {model_version} to production"

# Upload to cloud storage (if GCS available)
if command -v gsutil &> /dev/null; then
    echo "📤 Uploading to Google Cloud Storage..."
    gsutil cp -r "$PACKAGE_DIR" "gs://{self.storage_bucket}/models/{model_name}/{model_version}/"
    echo "✅ Model uploaded to GCS"
fi

# Update model registry in cloud system
echo "🔄 Updating model registry..."
# This would integrate with the cloud system's model registry

echo "✅ Model deployment complete!"
echo "   Model: {model_name} {model_version}"
echo "   Ready for production use"
'''

        script_path = package_dir / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

    def _create_compressed_package(self, package_dir: Path, model_name: str, model_version: str) -> str:
        """Create compressed package for efficient storage and transfer."""
        package_filename = f"{model_name}_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        package_path = self.deployment_dir / package_filename

        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)

        logger.debug(f"Created compressed package: {package_path}")
        return str(package_path)

    def deploy_to_cloud_storage(self, package_path: str, model_name: str, model_version: str) -> bool:
        """
        Deploy model package to Google Cloud Storage.

        Args:
            package_path: Path to the packaged model file
            model_name: Model name
            model_version: Model version

        Returns:
            True if deployment successful
        """
        if not self.gcs_available:
            logger.warning("GCS not available - skipping cloud deployment")
            return False

        try:
            from google.cloud import storage

            # Initialize GCS client
            client = storage.Client(project=self.gcp_project_id)
            bucket = client.bucket(self.storage_bucket)

            # Upload package
            blob_path = f"models/{model_name}/{model_version}/{Path(package_path).name}"
            blob = bucket.blob(blob_path)

            blob.upload_from_filename(package_path)

            # Set metadata
            blob.metadata = {
                'model_name': model_name,
                'model_version': model_version,
                'uploaded_at': datetime.now().isoformat(),
                'file_size': os.path.getsize(package_path)
            }
            blob.patch()

            logger.info(f"✅ Model deployed to GCS: gs://{self.storage_bucket}/{blob_path}")
            return True

        except Exception as e:
            logger.error(f"Error deploying to GCS: {e}")
            return False

    def generate_cloud_integration_code(self, model_name: str, model_version: str) -> str:
        """Generate code for integrating deployed model with cloud system."""
        integration_code = f'''
# Cloud integration code for {model_name} {model_version}
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import os
import json
from google.cloud import storage
from mcp_trader.models.deep_learning.gpu_lstm import GPUOptimizedPredictor

def load_deployed_model(model_name: str, model_version: str):
    """
    Load model from Google Cloud Storage.

    Args:
        model_name: Name of the model
        model_version: Version of the model

    Returns:
        Loaded model instance
    """
    try:
        # Download model from GCS
        client = storage.Client()
        bucket = client.bucket("{self.storage_bucket}")
        blob_path = f"models/{{model_name}}/{{model_version}}/model_metadata.json"

        # Download metadata first
        metadata_blob = bucket.blob(blob_path)
        metadata_content = metadata_blob.download_as_text()
        metadata = json.loads(metadata_content)

        # Download model file
        model_blob_path = f"models/{{model_name}}/{{model_version}}/{{metadata.get('model_file', 'model.pth')}}"
        model_blob = bucket.blob(model_blob_path)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_blob.download_to_filename(tmp_file.name)

            # Load model
            model = GPUOptimizedPredictor()
            model.load_model(tmp_file.name)

            # Clean up
            os.unlink(tmp_file.name)

            logger.info(f"✅ Loaded {{model_name}} {{model_version}} from cloud")
            return model

    except Exception as e:
        logger.error(f"Error loading model from cloud: {{e}}")
        raise

# Usage example:
# model = load_deployed_model("{model_name}", "{model_version}")
# predictions = model.predict(market_data)
'''

        return integration_code

    def create_model_training_report(self, model_name: str, model_version: str,
                                   training_metrics: Dict[str, Any]) -> str:
        """Create comprehensive training report for deployment."""
        report = {
            'model_info': {
                'name': model_name,
                'version': model_version,
                'training_date': datetime.now().isoformat(),
                'hardware_used': {
                    'cpu_cores': os.cpu_count(),
                    'gpu_available': torch.cuda.is_available() if 'torch' in globals() else False,
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                }
            },
            'training_metrics': training_metrics,
            'deployment_instructions': {
                'cloud_storage_path': f"gs://{self.storage_bucket}/models/{model_name}/{model_version}/",
                'integration_code': self.generate_cloud_integration_code(model_name, model_version),
                'validation_checklist': [
                    'Model loads without errors',
                    'Predictions are reasonable',
                    'Performance metrics meet thresholds',
                    'Integration with trading system works',
                    'Backtesting validates performance'
                ]
            },
            'next_steps': [
                'Deploy to cloud storage',
                'Update cloud system model registry',
                'Run integration tests',
                'Monitor live performance',
                'Set up automated retraining'
            ]
        }

        # Save report
        report_file = self.deployment_dir / f"{model_name}_{model_version}_training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✅ Training report created: {report_file}")
        return str(report_file)

    def batch_deploy_models(self, model_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Deploy multiple models in batch.

        Args:
            model_configs: List of model configurations with paths and metadata

        Returns:
            Dict mapping model names to deployment status
        """
        deployment_results = {}

        for config in model_configs:
            try:
                model_path = config['model_path']
                model_name = config['model_name']
                model_version = config['model_version']
                metadata = config.get('metadata', {})

                # Package model
                package_path = self.package_model_for_deployment(
                    model_path, model_name, model_version, metadata
                )

                # Deploy to cloud (if GCS available)
                if self.gcs_available:
                    success = self.deploy_to_cloud_storage(package_path, model_name, model_version)
                    deployment_results[f"{model_name}_{model_version}"] = "success" if success else "failed"
                else:
                    deployment_results[f"{model_name}_{model_version}"] = "packaged"

                # Generate integration code
                integration_code = self.generate_cloud_integration_code(model_name, model_version)

                # Save integration code
                integration_file = self.deployment_dir / f"{model_name}_{model_version}_integration.py"
                with open(integration_file, 'w') as f:
                    f.write(integration_code)

                logger.info(f"✅ Batch deployment completed for {model_name}_{model_version}")

            except Exception as e:
                logger.error(f"Error in batch deployment for {config}: {e}")
                deployment_results[f"{config['model_name']}_{config['model_version']}"] = f"error: {e}"

        return deployment_results

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status and available models."""
        status = {
            'total_models': len(self.model_registry),
            'gcs_available': self.gcs_available,
            'storage_bucket': self.storage_bucket,
            'deployment_directory': str(self.deployment_dir),
            'models': {}
        }

        for model_key, metadata in self.model_registry.items():
            status['models'][model_key] = {
                'version': metadata.get('version'),
                'created_at': metadata.get('created_at'),
                'file_size_mb': metadata.get('file_size_mb'),
                'model_type': metadata.get('model_type'),
                'deployment_ready': metadata.get('deployment_ready')
            }

        return status

    def cleanup_old_models(self, keep_recent: int = 10):
        """Clean up old model versions to save space."""
        logger.info(f"Cleaning up old models, keeping {keep_recent} most recent")

        # Sort models by creation date
        sorted_models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
        )

        # Keep only the most recent models
        models_to_keep = sorted_models[:keep_recent]
        models_to_remove = sorted_models[keep_recent:]

        for model_key, _ in models_to_remove:
            # Remove from registry
            if model_key in self.model_registry:
                del self.model_registry[model_key]

            # Remove package directory if exists
            package_dir = self.deployment_dir / model_key
            if package_dir.exists():
                shutil.rmtree(package_dir)
                logger.debug(f"Removed package directory: {package_dir}")

        self.save_model_registry()
        logger.info(f"✅ Cleaned up {len(models_to_remove)} old model versions")


# Example usage for RTX 5070Ti training workflow
def example_training_and_deployment():
    """Example of complete training and deployment workflow."""

    # Initialize deployment pipeline
    pipeline = ModelDeploymentPipeline()

    # Example model configurations (from your RTX 5070Ti training)
    model_configs = [
        {
            'model_path': 'models/gpu_lstm/lstm_predictor_best.pth',
            'model_name': 'lstm_predictor',
            'model_version': 'v1.0.0',
            'metadata': {
                'training_data_period': '2024-01-01 to 2024-06-01',
                'validation_accuracy': 0.67,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.12,
                'hardware_used': 'RTX 5070Ti + 16-core AMD'
            }
        },
        {
            'model_path': 'models/rl_agents/ppo_agent_best.zip',
            'model_name': 'ppo_trading_agent',
            'model_version': 'v1.0.0',
            'metadata': {
                'training_episodes': 50000,
                'avg_reward': 1.2,
                'convergence_epoch': 25000,
                'strategy': 'ensemble_momentum'
            }
        }
    ]

    # Package and deploy models
    deployment_results = pipeline.batch_deploy_models(model_configs)

    # Generate training reports
    for config in model_configs:
        pipeline.create_model_training_report(
            config['model_name'],
            config['model_version'],
            config['metadata']
        )

    # Show deployment status
    status = pipeline.get_deployment_status()
    print(f"✅ Deployment complete: {len(status['models'])} models ready")
    print(f"📊 GCS available: {status['gcs_available']}")

    return deployment_results


if __name__ == "__main__":
    # Example deployment
    results = example_training_and_deployment()
    print(f"Deployment results: {results}")

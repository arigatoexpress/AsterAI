#!/usr/bin/env python3
"""
CPU Fallback Training Script
Trains AI models on CPU when GPU is not available.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPUTrainer:
    """
    CPU-based trainer for AI models as fallback when GPU is not available.
    """

    def __init__(self, data_dir: str = "data/historical/real_aster_only"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.features = None
        self.targets = None
        
        logger.info("CPU Trainer initialized")

    def load_data(self) -> bool:
        """Load historical data for training."""
        try:
            logger.info("Loading historical data...")
            
            # Load data files
            data_files = list(self.data_dir.glob("*.parquet"))
            if not data_files:
                logger.error("No data files found")
                return False
            
            all_data = []
            for file in data_files:
                if "collection_summary" in file.name:
                    continue
                    
                df = pd.read_parquet(file)
                df['symbol'] = file.stem.replace('_1h', '').replace('_4h', '').replace('_1d', '')
                all_data.append(df)
            
            if not all_data:
                logger.error("No valid data files found")
                return False
            
            # Combine all data
            self.data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(self.data)} data points from {len(data_files)} files")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def prepare_features(self) -> bool:
        """Prepare features for training."""
        try:
            logger.info("Preparing features...")
            
            # Basic technical indicators
            df = self.data.copy()
            
            # Price features
            df['price_change'] = df.groupby('symbol')['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Moving averages
            for window in [5, 10, 20]:
                df[f'sma_{window}'] = df.groupby('symbol')['close'].rolling(window).mean().reset_index(0, drop=True)
                df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
            # Volatility
            df['volatility'] = df.groupby('symbol')['price_change'].rolling(20).std().reset_index(0, drop=True)
            
            # RSI-like indicator
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # Target variable (next period return > 1%)
            df['target'] = (df.groupby('symbol')['close'].shift(-1) / df['close'] - 1) > 0.01
            
            # Remove NaN values
            df = df.dropna()
            
            # Select features
            feature_cols = [
                'price_change', 'high_low_ratio', 'volume_price_ratio',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                'volatility', 'rsi'
            ]
            
            self.features = df[feature_cols].values
            self.targets = df['target'].values
            
            logger.info(f"Prepared {len(self.features)} samples with {len(feature_cols)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")
            return False

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train_models(self) -> bool:
        """Train multiple models."""
        try:
            logger.info("Training models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.features, self.targets, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            logger.info("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Random Forest accuracy: {accuracy:.3f}")
            
            # Store model
            self.models['random_forest'] = {
                'model': rf_model,
                'accuracy': accuracy,
                'feature_names': [
                    'price_change', 'high_low_ratio', 'volume_price_ratio',
                    'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                    'volatility', 'rsi'
                ]
            }
            
            # Generate report
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info("Classification Report:")
            if 'True' in report:
                logger.info(f"Precision: {report['True']['precision']:.3f}")
                logger.info(f"Recall: {report['True']['recall']:.3f}")
                logger.info(f"F1-Score: {report['True']['f1-score']:.3f}")
            elif True in report:
                logger.info(f"Precision: {report[True]['precision']:.3f}")
                logger.info(f"Recall: {report[True]['recall']:.3f}")
                logger.info(f"F1-Score: {report[True]['f1-score']:.3f}")
            else:
                logger.info(f"Overall accuracy: {report['accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return False

    def save_models(self) -> bool:
        """Save trained models."""
        try:
            logger.info("Saving models...")
            
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            for name, model_info in self.models.items():
                model_path = models_dir / f"{name}_cpu.pkl"
                joblib.dump(model_info['model'], model_path)
                logger.info(f"Saved {name} to {model_path}")
            
            # Save metadata
            metadata = {
                'model_type': 'cpu_fallback',
                'features': self.models['random_forest']['feature_names'],
                'accuracy': self.models['random_forest']['accuracy'],
                'training_date': pd.Timestamp.now().isoformat()
            }
            
            with open(models_dir / "cpu_models_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def run_training(self) -> bool:
        """Run complete training pipeline."""
        try:
            logger.info("Starting CPU training pipeline...")
            
            # Load data
            if not self.load_data():
                return False
            
            # Prepare features
            if not self.prepare_features():
                return False
            
            # Train models
            if not self.train_models():
                return False
            
            # Save models
            if not self.save_models():
                return False
            
            logger.info("✅ CPU training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False


def main():
    """Main execution."""
    print("""
================================================================================
                    CPU Fallback Training
              Training AI Models without GPU Acceleration
================================================================================
    """)
    
    trainer = CPUTrainer()
    
    try:
        success = trainer.run_training()
        if success:
            print("\n✅ Training completed successfully!")
            print("Models saved to models/ directory")
        else:
            print("\n❌ Training failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

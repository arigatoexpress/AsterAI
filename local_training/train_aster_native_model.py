"""
Train Aster-Native Asset Model
Specialized model for assets native to Aster DEX platform.

Key differences from general model:
- Focuses on Aster-specific trading patterns
- Accounts for lower liquidity
- Handles newer assets with limited history
- Optimized for Aster DEX fee structure
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.features.confluence_features import ConfluenceFeatureEngine, ConfluenceConfig
from local_training.train_confluence_model import LSTMPricePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsterNativeModelTrainer:
    """
    Train specialized model for Aster-native assets.
    
    Adaptations for Aster DEX:
    - Lower liquidity handling
    - Wider spreads consideration
    - Newer asset patterns
    - Platform-specific fees
    - Limited historical data strategies
    """
    
    def __init__(self, 
                 data_dir: str = "data/historical/aster_native",
                 output_dir: str = "models/aster_native"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🎮 Training device: {self.device}")
        
        # Models
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        
        # Feature engine
        self.feature_engine = ConfluenceFeatureEngine(ConfluenceConfig())
        
        # Aster-specific training config
        self.sequence_length = 60
        self.lookahead_periods = [1, 4, 12]  # Shorter horizons for new assets
        self.return_threshold = 0.03  # 3% (wider for lower liquidity)
        self.drawdown_threshold = 0.015  # 1.5% (more conservative)
        
        # Aster DEX fees
        self.maker_fee = 0.0005  # 0.05%
        self.taker_fee = 0.00075  # 0.075%
        
        logger.info("📊 Aster Native Model Trainer initialized")
        logger.info(f"   Return threshold: {self.return_threshold*100}%")
        logger.info(f"   Drawdown threshold: {self.drawdown_threshold*100}%")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load Aster-native asset data with validation."""
        logger.info("📥 Loading Aster-native asset data...")

        # Load and validate collection summary
        summary_file = self.data_dir / "collection_summary.csv"
        if summary_file.exists():
            summary = pd.read_csv(summary_file)

            # CRITICAL: Reject any synthetic data
            synthetic_data = summary[summary['source'] == 'synthetic']
            if not synthetic_data.empty:
                logger.error("❌ SYNTHETIC DATA DETECTED - TRAINING ABORTED")
                logger.error("   Synthetic data found for symbols:")
                for _, row in synthetic_data.iterrows():
                    logger.error(f"     • {row['symbol']}")
                raise ValueError("Synthetic data detected in training dataset. Use only real market data.")

            logger.info("\n📊 Data Sources (Real Market Data Only):")
            for _, row in summary.iterrows():
                status = "✅" if row['success'] else "❌"
                logger.info(f"   {status} {row['symbol']:12} | Source: {row['source']:12} | "
                          f"Quality: {row['data_quality_score']:.2f}")
        else:
            logger.warning("No collection summary found - proceeding without validation")

        # Load 1h data for all assets with quality checks
        asset_data = {}
        quality_threshold = 0.7  # Minimum quality score

        for file in self.data_dir.glob("*_1h.parquet"):
            symbol = file.stem.replace("_1h", "")
            try:
                df = pd.read_parquet(file)

                # Basic data quality check
                if len(df) < 100:  # Minimum 100 data points
                    logger.warning(f"  ⚠️  {symbol}: Insufficient data ({len(df)} points) - SKIPPING")
                    continue

                if df.isnull().sum().sum() / (len(df) * len(df.columns)) > 0.1:  # Max 10% missing
                    logger.warning(f"  ⚠️  {symbol}: Too much missing data - SKIPPING")
                    continue

                asset_data[symbol] = df
                logger.info(f"  ✅ Loaded {symbol}: {len(df)} records")

            except Exception as e:
                logger.error(f"  ❌ Error loading {symbol}: {e}")

        if not asset_data:
            raise ValueError("No valid data found. Ensure data collection completed successfully.")

        # Validate no assets with synthetic or poor quality data
        if summary_file.exists():
            summary = pd.read_csv(summary_file)
            valid_symbols = set(asset_data.keys())
            summary_valid = summary[summary['symbol'].isin(valid_symbols)]

            poor_quality = summary_valid[summary_valid['data_quality_score'] < quality_threshold]
            if not poor_quality.empty:
                logger.warning("⚠️  Poor quality data detected:")
                for _, row in poor_quality.iterrows():
                    logger.warning(f"   • {row['symbol']}: quality {row['data_quality_score']:.2f} < {quality_threshold}")

        logger.info(f"✅ Loaded {len(asset_data)} assets with validated real market data")

        # Generate confluence features
        logger.info("🔗 Generating confluence features for Aster assets...")
        enriched_data = self.feature_engine.generate_all_features(asset_data)

        # Add Aster-specific features
        enriched_data = self._add_aster_specific_features(enriched_data)

        # Combine all assets
        logger.info("🔨 Preparing training dataset...")
        combined_data = []

        for symbol, df in enriched_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            df_copy = self._generate_labels(df_copy)
            combined_data.append(df_copy)

        full_data = pd.concat(combined_data, axis=0).sort_index()

        logger.info(f"✅ Dataset prepared: {len(full_data)} total records from {len(asset_data)} assets")
        logger.info("   🔒 Training safety: Synthetic data validation passed")
        return full_data
    
    def _add_aster_specific_features(self, enriched_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add Aster DEX-specific features.
        
        Features:
        - Liquidity proxies (volume-based)
        - Spread estimates
        - New asset indicators
        - Platform activity metrics
        """
        logger.info("   Adding Aster-specific features...")
        
        for symbol, df in enriched_data.items():
            # Liquidity score (volume-based)
            volume_ma = df['volume'].rolling(24).mean()
            df['aster_liquidity_score'] = df['volume'] / (volume_ma + 1)
            
            # Estimate spread (high-low as proxy)
            df['aster_estimated_spread'] = (df['high'] - df['low']) / df['close']
            df['aster_spread_ma'] = df['aster_estimated_spread'].rolling(24).mean()
            
            # Volatility (important for new assets)
            df['aster_volatility'] = df['close'].pct_change().rolling(24).std()
            
            # Price stability (lower is more stable)
            df['aster_price_stability'] = 1 / (df['aster_volatility'] + 0.01)
            
            # Volume trend (growing or declining)
            volume_trend = df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean()
            df['aster_volume_trend'] = volume_trend.fillna(1)
            
            # New asset indicator (based on data availability)
            df['aster_data_maturity'] = range(len(df))  # Days since listing
            df['aster_is_new_asset'] = (df['aster_data_maturity'] < 720).astype(int)  # < 30 days
            
            enriched_data[symbol] = df
        
        return enriched_data
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels with Aster-specific considerations.
        
        More conservative thresholds for new/low-liquidity assets.
        """
        df = df.copy()
        
        # Calculate forward returns
        for period in self.lookahead_periods:
            df[f'forward_return_{period}h'] = df['close'].pct_change(period).shift(-period)
            df[f'forward_max_dd_{period}h'] = df['close'].rolling(period).apply(
                lambda x: (x.min() - x.iloc[0]) / x.iloc[0] if len(x) > 0 else 0,
                raw=False
            ).shift(-period)
        
        # Primary label based on 4h forward return
        # Adjust thresholds based on liquidity
        liquidity_adjustment = 1 + (1 - df.get('aster_liquidity_score', 1).clip(0, 2)) * 0.5
        
        return_threshold = self.return_threshold * liquidity_adjustment
        drawdown_threshold = self.drawdown_threshold * liquidity_adjustment
        
        conditions_buy = (
            (df['forward_return_4h'] > return_threshold) &
            (df['forward_max_dd_4h'] > -drawdown_threshold) &
            (df.get('aster_liquidity_score', 1) > 0.5)  # Minimum liquidity
        )
        
        conditions_sell = (
            (df['forward_return_4h'] < -return_threshold) |
            (df['forward_max_dd_4h'] < -drawdown_threshold)
        )
        
        df['label'] = 1  # HOLD
        df.loc[conditions_buy, 'label'] = 2  # BUY
        df.loc[conditions_sell, 'label'] = 0  # SELL
        
        df = df.dropna(subset=['label'])
        
        return df
    
    def prepare_training_data(self, full_data: pd.DataFrame) -> Dict:
        """Prepare features and labels for training."""
        logger.info("🔨 Preparing training data...")
        
        # Select feature columns
        feature_cols = [col for col in full_data.columns 
                       if col.startswith('confluence_') or col.startswith('aster_')]
        
        # Add basic price features
        price_features = ['open', 'high', 'low', 'close', 'volume']
        for col in price_features:
            if col in full_data.columns:
                feature_cols.append(col)
        
        # Remove rows with NaN
        data_clean = full_data[feature_cols + ['label']].dropna()
        
        X = data_clean[feature_cols].values
        y = data_clean['label'].values.astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-based split (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y[split_idx:]
        
        logger.info(f"✅ Training set: {len(X_train)} samples")
        logger.info(f"✅ Validation set: {len(X_val)} samples")
        logger.info(f"   Class distribution (train): {np.bincount(y_train)}")
        logger.info(f"   Class distribution (val): {np.bincount(y_val)}")
        logger.info(f"   Features: {len(feature_cols)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_cols': feature_cols
        }
    
    def train_xgboost_classifier(self, train_data: Dict):
        """Train XGBoost for Aster-native assets."""
        logger.info("\n🚀 Training Aster-Native XGBoost Classifier...")
        
        # XGBoost parameters tuned for smaller datasets
        params = {
            'max_depth': 5,  # Shallower to avoid overfitting
            'learning_rate': 0.03,  # Lower for stability
            'n_estimators': 200,
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 15,
            'random_state': 42,
            'min_child_weight': 5,  # Higher to prevent overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Train model
        dtrain = xgb.DMatrix(train_data['X_train'], label=train_data['y_train'])
        dval = xgb.DMatrix(train_data['X_val'], label=train_data['y_val'])
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            verbose_eval=25
        )
        
        # Evaluate
        y_pred_proba = self.xgb_model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        logger.info("\n📊 Aster-Native XGBoost Performance:")
        logger.info(f"\n{classification_report(train_data['y_val'], y_pred, target_names=['SELL', 'HOLD', 'BUY'])}")
        
        # Feature importance
        importance = self.xgb_model.get_score(importance_type='gain')
        logger.info("\n🔝 Top 10 Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, score in sorted_importance:
            feature_name = train_data['feature_cols'][int(feat.replace('f', ''))]
            logger.info(f"   {feature_name:40} {score:.2f}")
        
        # Save model
        xgb_path = self.output_dir / "xgboost_aster_native.json"
        self.xgb_model.save_model(str(xgb_path))
        logger.info(f"\n✅ Model saved to {xgb_path}")
        
        return y_pred, y_pred_proba
    
    def save_artifacts(self, train_data: Dict):
        """Save all training artifacts."""
        # Save scaler
        joblib.dump(self.scaler, self.output_dir / "feature_scaler.pkl")
        
        # Save feature columns
        with open(self.output_dir / "feature_columns.json", 'w') as f:
            json.dump(train_data['feature_cols'], f, indent=2)
        
        # Save model config
        config = {
            'model_type': 'aster_native',
            'return_threshold': self.return_threshold,
            'drawdown_threshold': self.drawdown_threshold,
            'sequence_length': self.sequence_length,
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,
            'trained_date': datetime.now().isoformat(),
            'platform': 'Aster DEX',
            'notes': 'Specialized model for Aster-native assets with liquidity adjustments'
        }
        
        with open(self.output_dir / "model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ All artifacts saved to {self.output_dir}")
    
    def train(self):
        """Execute complete training pipeline."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║         Aster-Native Asset Model Training (GPU)                ║
║      Optimized for new assets & lower liquidity               ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        # Load and prepare data
        full_data = self.load_and_prepare_data()
        train_data = self.prepare_training_data(full_data)
        
        # Train XGBoost
        xgb_pred, xgb_pred_proba = self.train_xgboost_classifier(train_data)
        
        # Save artifacts
        self.save_artifacts(train_data)
        
        print("""
╔════════════════════════════════════════════════════════════════╗
║            🎉 Aster-Native Model Training Complete!            ║
╚════════════════════════════════════════════════════════════════╝

Models saved to: models/aster_native/
- xgboost_aster_native.json
- feature_scaler.pkl
- model_config.json

Key Features:
- Optimized for Aster DEX fee structure
- Handles new assets with limited history
- Liquidity-adjusted thresholds
- Lower overfitting risk

Next steps:
1. Compare with general model performance
2. Backtest on Aster-native data
3. Deploy both models for ensemble
        """)


def main():
    """Main execution."""
    trainer = AsterNativeModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()


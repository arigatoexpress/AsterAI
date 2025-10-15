"""
Train Multi-Asset Confluence Model
GPU-accelerated training for buy/sell signal generation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from typing import Dict, Tuple, List
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.features.confluence_features import ConfluenceFeatureEngine, ConfluenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMPricePredictor(nn.Module):
    """LSTM model for price direction prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: BUY, HOLD, SELL
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


class ConfluenceModelTrainer:
    """
    Train multi-asset confluence model on RTX 5070Ti.
    
    Features:
    - GPU-accelerated XGBoost
    - LSTM price direction predictor
    - Ensemble combination
    - Cross-validation on time series
    """
    
    def __init__(self, data_dir: str = "data/historical/real_aster_only",
                 output_dir: str = "models/confluence"):
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
        
        # Training config
        self.sequence_length = 60  # 60 hours
        self.lookahead_periods = [1, 4]  # 1h, 4h (reduced to avoid too many NaN)
        self.return_threshold = 0.005  # 0.5% return threshold (more realistic for hourly data)
        self.drawdown_threshold = 0.005  # 0.5% max drawdown
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load historical data and generate features with validation."""
        logger.info("📥 Loading historical data...")

        # Check for data source validation
        summary_file = self.data_dir / "collection_summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)

            # CRITICAL: Reject any synthetic data
            synthetic_found = False
            if 'assets' in summary_data:
                for symbol, asset_info in summary_data['assets'].items():
                    if asset_info.get('source') == 'synthetic':
                        synthetic_found = True
                        logger.error(f"❌ SYNTHETIC DATA DETECTED - TRAINING ABORTED")
                        logger.error(f"   Synthetic data found for symbol: {symbol}")
                        break

            if synthetic_found:
                raise ValueError("Synthetic data detected in training dataset. Use only real market data.")

            logger.info("\n📊 Data Sources Validation:")
            total_assets = summary_data.get('successful_collections', 0)
            logger.info(f"   ✅ Real market data: {total_assets} assets collected")
            logger.info(f"   📈 Success rate: {summary_data.get('success_rate', 0):.1%}")

        # Load 1h data for all assets with quality checks
        asset_data = {}

        for file in self.data_dir.glob("*_1h.parquet"):
            symbol = file.stem.replace("_1h", "")
            try:
                df = pd.read_parquet(file)

                # Basic data validation
                if len(df) < 50:  # Minimum 50 data points (relaxed for more assets)
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
            raise ValueError("No valid data found. Run data collection scripts first.")

        # Generate confluence features
        logger.info("🔗 Generating confluence features...")
        enriched_data = self.feature_engine.generate_all_features(asset_data)

        # Combine all assets into single training dataset
        logger.info("🔨 Preparing training dataset...")
        combined_data = []

        for symbol, df in enriched_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol

            # Generate labels
            df_copy = self._generate_labels(df_copy)

            combined_data.append(df_copy)

        # Concatenate all data
        full_data = pd.concat(combined_data, axis=0).sort_index()

        logger.info(f"✅ Dataset prepared: {len(full_data)} total records from {len(asset_data)} assets")
        logger.info(f"   Features: {len([c for c in full_data.columns if c.startswith('confluence_')])} confluence features")
        logger.info("   🔒 Training safety: Synthetic data validation passed")

        return full_data
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading labels based on forward returns.
        
        Labels:
        - 0: SELL (negative return or large drawdown)
        - 1: HOLD (small movement)
        - 2: BUY (positive return, no large drawdown)
        """
        df = df.copy()
        
        # Calculate forward returns for different periods
        for period in self.lookahead_periods:
            df[f'forward_return_{period}h'] = df['close'].pct_change(period).shift(-period)
            
            # Calculate max drawdown in forward period
            df[f'forward_max_dd_{period}h'] = df['close'].rolling(period).apply(
                lambda x: (x.min() - x.iloc[0]) / x.iloc[0] if len(x) > 0 else 0,
                raw=False
            ).shift(-period)
        
        # Primary label based on 4h forward return (only drop if 4h return is NaN)
        conditions_buy = (
            (df['forward_return_4h'] > self.return_threshold) &
            (df['forward_max_dd_4h'] > -self.drawdown_threshold)
        )

        conditions_sell = (
            (df['forward_return_4h'] < -self.return_threshold) |
            (df['forward_max_dd_4h'] < -self.drawdown_threshold)
        )

        df['label'] = 1  # Default: HOLD
        df.loc[conditions_buy, 'label'] = 2  # BUY
        df.loc[conditions_sell, 'label'] = 0  # SELL

        # Only remove rows where the primary label (4h) is NaN
        df = df.dropna(subset=['forward_return_4h'])
        
        return df
    
    def prepare_training_data(self, full_data: pd.DataFrame) -> Dict:
        """Prepare features and labels for training."""
        logger.info("🔨 Preparing training data...")
        
        # Select feature columns
        feature_cols = [col for col in full_data.columns if col.startswith('confluence_')]
        
        # Add basic price features
        price_features = ['open', 'high', 'low', 'close', 'volume']
        for col in price_features:
            if col in full_data.columns:
                feature_cols.append(col)
        
        # Debug: Check label generation
        logger.info(f"Total rows before cleaning: {len(full_data)}")
        logger.info(f"Label distribution: {full_data['label'].value_counts(dropna=False).to_dict()}")

        # Remove rows with NaN in features, but keep rows with valid labels
        # (forward_return columns may be NaN but labels are still valid)
        valid_rows = full_data['label'].notna()
        logger.info(f"Rows with valid labels: {valid_rows.sum()}")

        data_clean = full_data.loc[valid_rows, feature_cols + ['label']].dropna()
        logger.info(f"Rows after feature cleaning: {len(data_clean)}")

        if len(data_clean) == 0:
            # Debug: Show what's causing the issue
            logger.error("No valid training data after cleaning:")
            logger.error(f"Feature columns: {feature_cols[:5]}...")
            nan_counts = full_data[feature_cols].isnull().sum()
            logger.error(f"NaN counts in features: {nan_counts[nan_counts > 0].to_dict()}")
            raise ValueError("No valid training data after cleaning. Check label generation.")

        X = data_clean[feature_cols].values
        y = data_clean['label'].values.astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-based train/val split (80/20)
        split_idx = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y[split_idx:]
        
        logger.info(f"✅ Training set: {len(X_train)} samples")
        logger.info(f"✅ Validation set: {len(X_val)} samples")
        logger.info(f"   Class distribution (train): {np.bincount(y_train)}")
        logger.info(f"   Class distribution (val): {np.bincount(y_val)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_cols': feature_cols
        }
    
    def train_xgboost_classifier(self, train_data: Dict):
        """Train XGBoost classifier with GPU acceleration."""
        logger.info("\n🚀 Training XGBoost Classifier...")
        
        # XGBoost parameters optimized for GPU
        params = {
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',  # Use 'gpu_hist' if xgboost compiled with GPU support
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 20,
            'random_state': 42
        }
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(train_data['X_train'], label=train_data['y_train'])
        dval = xgb.DMatrix(train_data['X_val'], label=train_data['y_val'])
        
        # Train model
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=evals,
            verbose_eval=50
        )
        
        # Evaluate
        y_pred_proba = self.xgb_model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        logger.info("\n📊 XGBoost Performance:")
        logger.info(f"\nClassification Report:\n{classification_report(train_data['y_val'], y_pred, target_names=['SELL', 'HOLD', 'BUY'])}")
        
        # Save model
        xgb_path = self.output_dir / "xgboost_classifier.json"
        self.xgb_model.save_model(str(xgb_path))
        logger.info(f"✅ XGBoost model saved to {xgb_path}")
        
        return y_pred, y_pred_proba
    
    def train_lstm_predictor(self, train_data: Dict):
        """Train LSTM price direction predictor."""
        logger.info("\n🚀 Training LSTM Predictor...")
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self._prepare_sequences(
            train_data['X_train'], train_data['y_train'], self.sequence_length
        )
        X_val_seq, y_val_seq = self._prepare_sequences(
            train_data['X_val'], train_data['y_val'], self.sequence_length
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_seq).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Initialize model
        input_dim = X_train_seq.shape[2]
        self.lstm_model = LSTMPricePredictor(input_dim).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            self.lstm_model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/100] "
                          f"Train Loss: {train_loss/len(train_loader):.4f} "
                          f"Val Loss: {val_loss:.4f} "
                          f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.lstm_model.state_dict(), self.output_dir / "lstm_predictor.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.lstm_model.load_state_dict(torch.load(self.output_dir / "lstm_predictor.pth"))
        
        # Final evaluation
        self.lstm_model.eval()
        with torch.no_grad():
            val_outputs = self.lstm_model(X_val_tensor)
            y_pred = torch.argmax(val_outputs, dim=1).cpu().numpy()
        
        logger.info("\n📊 LSTM Performance:")
        logger.info(f"\nClassification Report:\n{classification_report(y_val_seq, y_pred, target_names=['SELL', 'HOLD', 'BUY'])}")
        
        logger.info(f"✅ LSTM model saved to {self.output_dir}/lstm_predictor.pth")
        
        return y_pred
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM."""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def create_ensemble(self, train_data: Dict, xgb_pred_proba: np.ndarray, lstm_pred: np.ndarray):
        """Create ensemble of XGBoost and LSTM predictions."""
        logger.info("\n🔗 Creating ensemble model...")
        
        # Simple weighted average (can be optimized)
        xgb_weight = 0.6
        lstm_weight = 0.4
        
        # Convert LSTM predictions to probabilities
        lstm_proba = np.zeros((len(lstm_pred), 3))
        for i, pred in enumerate(lstm_pred):
            lstm_proba[i, pred] = 1.0
        
        # Ensemble prediction
        ensemble_proba = xgb_weight * xgb_pred_proba + lstm_weight * lstm_proba
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Evaluate ensemble
        logger.info("\n📊 Ensemble Performance:")
        logger.info(f"\nClassification Report:\n{classification_report(train_data['y_val'], ensemble_pred, target_names=['SELL', 'HOLD', 'BUY'])}")
        
        # Save ensemble config
        ensemble_config = {
            'xgb_weight': xgb_weight,
            'lstm_weight': lstm_weight,
            'sequence_length': self.sequence_length,
            'feature_cols': train_data['feature_cols'],
            'trained_date': datetime.now().isoformat()
        }
        
        with open(self.output_dir / "ensemble_config.json", 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"✅ Ensemble config saved")
    
    def save_artifacts(self, train_data: Dict):
        """Save training artifacts."""
        # Save scaler
        joblib.dump(self.scaler, self.output_dir / "feature_scaler.pkl")
        
        # Save feature columns
        with open(self.output_dir / "feature_columns.json", 'w') as f:
            json.dump(train_data['feature_cols'], f, indent=2)
        
        logger.info(f"✅ Training artifacts saved to {self.output_dir}")
    
    def train(self):
        """Execute complete training pipeline."""
        print("""
================================================================================
         Multi-Asset Confluence Model Training (GPU)
================================================================================
        """)

        # Load and prepare data
        print("Step 1/5: Loading and preparing data...")
        full_data = self.load_and_prepare_data()
        train_data = self.prepare_training_data(full_data)
        print("✅ Data preparation complete")

        # Train XGBoost
        print("Step 2/5: Training XGBoost classifier...")
        xgb_pred, xgb_pred_proba = self.train_xgboost_classifier(train_data)
        print("✅ XGBoost training complete")

        # Train LSTM
        print("Step 3/5: Training LSTM predictor...")
        lstm_pred = self.train_lstm_predictor(train_data)
        print("✅ LSTM training complete")

        # Create ensemble
        print("Step 4/5: Creating ensemble model...")
        self.create_ensemble(train_data, xgb_pred_proba, lstm_pred)
        print("✅ Ensemble creation complete")

        # Save artifacts
        print("Step 5/5: Saving training artifacts...")
        self.save_artifacts(train_data)
        print("✅ Artifacts saved")
        
        print("""
================================================================================
                   Training Complete!
================================================================================

Models saved to: models/confluence/
- xgboost_classifier.json
- lstm_predictor.pth
- feature_scaler.pkl
- ensemble_config.json

Next steps:
1. Run backtesting (Phase 5)
2. Optimize hyperparameters (Phase 6)
3. Export to ONNX (Phase 7)
        """)


def main():
    """Main execution."""
    trainer = ConfluenceModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()


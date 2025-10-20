"""
Advanced LSTM/RNN Price Prediction Models for Cryptocurrency Trading
Implements multi-step forecasting with attention mechanisms and uncertainty estimation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from ..base import BaseMLModel, ModelPrediction, TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    input_size: int = 50  # Number of features
    hidden_size: int = 128
    num_layers: int = 3
    output_size: int = 1
    dropout: float = 0.2
    bidirectional: bool = True
    attention: bool = True
    forecast_horizon: int = 24  # Hours ahead to predict
    sequence_length: int = 168  # 7 days of hourly data
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10


class AttentionMechanism(nn.Module):
    """Multi-head attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, hidden_size = x.size()

        # Linear transformations and reshape
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Output projection
        output = self.out_linear(attended)
        return output


class LSTMPredictor(nn.Module):
    """
    Advanced LSTM model with attention mechanism for price prediction.
    Includes uncertainty estimation and multi-step forecasting.
    """

    def __init__(self, config: LSTMConfig):
        super(LSTMPredictor, self).__init__()
        self.config = config
        
        # Device selection with CPU fallback for stability
        use_cpu = os.getenv('ASTERAI_FORCE_CPU', '1') == '1'  # Default to CPU
        enable_gpu = os.getenv('ASTERAI_ENABLE_GPU', '0') == '1'
        
        if enable_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("LSTM using GPU acceleration")
        else:
            self.device = torch.device('cpu')
            logger.info("LSTM using CPU (for stability and compatibility)")

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        ).to(self.device)

        # Adjust hidden size for bidirectional
        lstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size

        # Attention mechanism
        self.attention = AttentionMechanism(lstm_output_size).to(self.device) if config.attention else None

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64).to(self.device)
        self.fc2 = nn.Linear(64, config.output_size).to(self.device)

        # Uncertainty estimation (variance prediction)
        self.uncertainty_head = nn.Linear(lstm_output_size, config.output_size).to(self.device)

        # Multi-step forecasting head
        if config.forecast_horizon > 1:
            self.multi_step_head = nn.Linear(lstm_output_size, config.forecast_horizon).to(self.device)
        else:
            self.multi_step_head = None

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(lstm_output_size).to(self.device)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            prediction: Main prediction
            uncertainty: Uncertainty estimate (variance)
            multi_step: Multi-step forecasts if configured
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention if enabled
        if self.attention is not None:
            lstm_out = self.attention(lstm_out)

        # Use last timestep output or apply attention
        if self.attention is not None:
            # Global average pooling after attention
            context = torch.mean(lstm_out, dim=1)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        # Batch normalization
        context = self.batch_norm(context)

        # Dropout
        context = self.dropout(context)

        # Main prediction head
        fc_out = F.relu(self.fc1(context))
        prediction = self.fc2(fc_out)

        # Uncertainty estimation
        uncertainty = torch.exp(self.uncertainty_head(context))  # Ensure positive variance

        # Multi-step forecasting
        multi_step = None
        if self.multi_step_head is not None:
            multi_step = self.multi_step_head(context)

        return prediction, uncertainty, multi_step


class LSTMPredictorModel(BaseMLModel):
    """
    LSTM-based price prediction model with advanced features.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__("LSTM_Predictor", **kwargs)

        # Default configuration
        default_config = {
            'input_size': 50,
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'bidirectional': True,
            'attention': True,
            'forecast_horizon': 24,
            'sequence_length': 168,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100,
            'patience': 10,
            'target_column': 'close',
            'lookback_window': 168
        }

        # Update with provided config
        if config:
            default_config.update(config)

        # Filter keys not present in LSTMConfig to avoid TypeError from extra kwargs
        allowed_keys = {
            'input_size',
            'hidden_size',
            'num_layers',
            'output_size',
            'dropout',
            'bidirectional',
            'attention',
            'forecast_horizon',
            'sequence_length',
            'learning_rate',
            'batch_size',
            'num_epochs',
            'patience',
        }
        filtered = {k: v for k, v in default_config.items() if k in allowed_keys}
        # Preserve non-config keys for model logic
        self._target_column = default_config.get('target_column', 'close')
        self._lookback_window = default_config.get('lookback_window', 168)

        self.config = LSTMConfig(**filtered)
        self.model = LSTMPredictor(self.config)
        self.scaler = None
        self.feature_columns = []
        self.is_fitted = False

        # Training state
        self.optimizer = None
        self.criterion = None
        self.best_loss = float('inf')
        self.patience_counter = 0

        # Use device from model (already set in LSTMPredictor.__init__)
        self.device = self.model.device

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare technical features for LSTM input.
        Creates comprehensive feature set for price prediction.
        """
        df = data.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Volatility features
        df['volatility_24h'] = df['returns'].rolling(24).std()
        df['volatility_168h'] = df['returns'].rolling(168).std()
        df['atr'] = self._calculate_atr(df, 14)

        # RSI
        df['rsi'] = self._calculate_rsi(df, 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df, 20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']

        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = df['close'].pct_change(period)

        # Lagged features
        for lag in range(1, 25):  # Last 24 hours
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Drop NaN values
        df = df.dropna()

        # Select feature columns (exclude target and non-feature columns)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.feature_columns = feature_cols

        return df[feature_cols].values

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        """
        X, y = [], []

        for i in range(len(features) - self.config.sequence_length - self.config.forecast_horizon + 1):
            # Input sequence
            X_seq = features[i:i + self.config.sequence_length]
            X.append(X_seq)

            # Target (next forecast_horizon values)
            if self.config.forecast_horizon == 1:
                y_seq = targets[i + self.config.sequence_length - 1]
            else:
                y_seq = targets[i + self.config.sequence_length:i + self.config.sequence_length + self.config.forecast_horizon]
                if len(y_seq) == self.config.forecast_horizon:
                    y_seq = y_seq[-1]  # Use last value for single prediction

            y.append(y_seq)

        return np.array(X), np.array(y)

    def fit(self, data: pd.DataFrame, **kwargs) -> 'LSTMPredictorModel':
        """
        Train the LSTM model.
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from torch.utils.data import DataLoader, TensorDataset

            # Prepare features and targets
            features = self.prepare_features(data)
            targets = self.prepare_targets(data)

            # Align data
            min_length = min(len(features), len(targets))
            features = features[:min_length]
            targets = targets[:min_length]

            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

            # Create sequences
            X, y = self.prepare_sequences(features_scaled, targets)

            if len(X) == 0:
                raise ValueError("Not enough data to create sequences")

            # Train/validation split
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            # Convert to tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

            # Initialize optimizer and loss
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            self.criterion = nn.MSELoss()

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()

                    # Forward pass
                    pred, uncertainty, _ = self.model(batch_X)
                    loss = self.criterion(pred.squeeze(), batch_y)

                    # Add uncertainty regularization
                    uncertainty_loss = torch.mean(uncertainty)
                    total_loss = loss + 0.1 * uncertainty_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    train_loss += total_loss.item()

                train_loss /= len(train_loader)

                # Validation
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        pred, _, _ = self.model(batch_X)
                        loss = self.criterion(pred.squeeze(), batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), f"{self.name}_best.pth")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Load best model
            self.model.load_state_dict(torch.load(f"{self.name}_best.pth"))
            self.is_fitted = True

            logger.info("LSTM model training completed")
            return self

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """
        Generate predictions using the trained LSTM model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            self.model.eval()

            # Prepare features
            features = self.prepare_features(data)
            features_scaled = self.scaler.transform(features)

            # Create sequences (use last available sequence)
            if len(features_scaled) < self.config.sequence_length:
                # Pad with zeros if not enough data
                padding = np.zeros((self.config.sequence_length - len(features_scaled), features_scaled.shape[1]))
                features_scaled = np.vstack([padding, features_scaled])

            # Take last sequence
            X = features_scaled[-self.config.sequence_length:].reshape(1, self.config.sequence_length, -1)
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                pred, uncertainty, multi_step = self.model(X_tensor)

            # Convert to predictions
            predictions = []

            for i, (_, row) in enumerate(data.iterrows()):
                if i >= len(data) - 1:  # Only predict for the last available data point
                    prediction_value = pred.item()
                    uncertainty_value = uncertainty.item()

                    predictions.append(ModelPrediction(
                        timestamp=row['timestamp'],
                        symbol=row.get('symbol', 'UNKNOWN'),
                        prediction=prediction_value,
                        confidence=max(0.1, min(1.0, 1.0 / (1.0 + uncertainty_value))),  # Convert uncertainty to confidence
                        features={col: row.get(col, 0) for col in self.feature_columns[:10]},  # First 10 features
                        metadata={
                            'model_type': 'deep_learning',
                            'model_name': self.name,
                            'uncertainty': uncertainty_value,
                            'forecast_horizon': self.config.forecast_horizon,
                            'multi_step_available': multi_step is not None
                        }
                    ))

            return predictions

        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            return []

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals from LSTM predictions.
        """
        predictions = self.predict(data)
        signals = []

        for pred in predictions:
            # Convert prediction to signal
            threshold = 0.005  # 0.5% threshold

            if pred.prediction > threshold:
                signal = 1  # BUY
            elif pred.prediction < -threshold:
                signal = -1  # SELL
            else:
                signal = 0  # HOLD

            # Adjust confidence based on uncertainty
            uncertainty_penalty = pred.metadata.get('uncertainty', 0) * 0.1
            adjusted_confidence = max(0.1, pred.confidence - uncertainty_penalty)

            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=signal,
                confidence=adjusted_confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0] if 'close' in data.columns else 0,
                metadata={
                    'prediction': pred.prediction,
                    'uncertainty': pred.metadata.get('uncertainty', 0),
                    'model': self.name,
                    'forecast_horizon': self.config.forecast_horizon
                }
            ))

        return signals

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower

    def save_model(self, filepath: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, filepath)

    def load_model(self, filepath: str):
        """Load model from disk."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        self.is_fitted = checkpoint['is_fitted']
        self.model.to(self.device)


class TransformerPredictor(nn.Module):
    """
    Transformer-based price prediction model.
    Uses self-attention mechanisms for long-range dependencies.
    """

    def __init__(self, config: LSTMConfig):
        super(TransformerPredictor, self).__init__()
        self.config = config

        # Input embedding
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(config.sequence_length, config.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output heads
        self.dropout = nn.Dropout(config.dropout)
        self.output_head = nn.Linear(config.hidden_size, config.output_size)
        self.uncertainty_head = nn.Linear(config.hidden_size, config.output_size)

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Create positional encoding matrix."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Input embedding
        x = self.input_embedding(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Dropout
        x = self.dropout(x)

        # Output predictions
        prediction = self.output_head(x)
        uncertainty = torch.exp(self.uncertainty_head(x))

        return prediction, uncertainty, None


class TransformerPredictorModel(LSTMPredictorModel):
    """Transformer-based price prediction model."""

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.name = "Transformer_Predictor"

        # Override model with transformer
        self.model = TransformerPredictor(self.config)

        # Move to device
        self.model.to(self.device)


class EnsembleDLPredictor(BaseMLModel):
    """
    Ensemble of deep learning models for robust price prediction.
    Combines LSTM, Transformer, and other DL models.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__("Ensemble_DL_Predictor", **kwargs)

        # Initialize multiple models
        self.models = {
            'lstm': LSTMPredictorModel(config, **kwargs),
            'transformer': TransformerPredictorModel(config, **kwargs)
        }

        # Ensemble weights (learned or equal)
        self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}

        self.is_fitted = False

    def fit(self, data: pd.DataFrame, **kwargs) -> 'EnsembleDLPredictor':
        """Fit all models in the ensemble."""
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(data, **kwargs)

        self.is_fitted = True
        logger.info("Ensemble deep learning models trained")
        return self

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Get predictions from all models
        all_predictions = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(data)
                all_predictions[name] = preds
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                continue

        # Ensemble predictions
        ensemble_predictions = []

        if not all_predictions:
            return ensemble_predictions

        # Use first model's predictions as template
        template_preds = list(all_predictions.values())[0]

        for i, template_pred in enumerate(template_preds):
            # Weighted average of predictions
            weighted_pred = 0.0
            weighted_uncertainty = 0.0
            total_weight = 0.0

            for model_name, preds in all_predictions.items():
                if i < len(preds):
                    weight = self.model_weights[model_name]
                    weighted_pred += weight * preds[i].prediction
                    weighted_uncertainty += weight * preds[i].metadata.get('uncertainty', 0.1)
                    total_weight += weight

            if total_weight > 0:
                final_pred = weighted_pred / total_weight
                final_uncertainty = weighted_uncertainty / total_weight
                final_confidence = max(0.1, min(1.0, 1.0 / (1.0 + final_uncertainty)))

                ensemble_predictions.append(ModelPrediction(
                    timestamp=template_pred.timestamp,
                    symbol=template_pred.symbol,
                    prediction=final_pred,
                    confidence=final_confidence,
                    features=template_pred.features,
                    metadata={
                        'model_type': 'ensemble_dl',
                        'model_name': self.name,
                        'uncertainty': final_uncertainty,
                        'constituent_models': list(self.models.keys()),
                        'model_weights': self.model_weights
                    }
                ))

        return ensemble_predictions


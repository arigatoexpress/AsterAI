"""
GPU-Optimized LSTM Models for RTX 5070Ti
High-performance deep learning models with CUDA acceleration and multi-core optimization.
"""

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
class GPUConfig:
    """GPU-optimized configuration for RTX 5070Ti."""
    input_size: int = 100  # Increased features for better accuracy
    hidden_size: int = 256  # Larger hidden state for RTX 5070Ti
    num_layers: int = 4     # Deeper network for GPU
    output_size: int = 1
    dropout: float = 0.1    # Lower dropout for GPU training
    bidirectional: bool = True
    attention_heads: int = 16  # Multi-head attention
    forecast_horizon: int = 24
    sequence_length: int = 336  # 2 weeks of hourly data
    learning_rate: float = 0.001
    batch_size: int = 128   # Larger batches for GPU
    num_epochs: int = 200   # More epochs for GPU training
    patience: int = 15      # More patience for complex models

    # GPU-specific optimizations
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_fused_kernels: bool = True
    enable_flash_attention: bool = True


class FlashAttention(nn.Module):
    """Flash Attention implementation for RTX 5070Ti."""

    def __init__(self, hidden_size: int, num_heads: int = 16):
        super(FlashAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.05)  # Lower dropout

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flash attention forward pass."""
        batch_size, seq_len, hidden_size = x.size()

        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (Flash Attention algorithm)
        scale = self.head_dim ** 0.5

        # Use PyTorch's built-in attention for RTX 5070Ti optimization
        attention_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.05 if self.training else 0.0,
            scale=scale
        )

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_linear(attention_output)

        return output


class OptimizedLSTM(nn.Module):
    """
    GPU-optimized LSTM with Flash Attention for RTX 5070Ti.
    Designed for maximum performance on high-end consumer GPUs.
    """

    def __init__(self, config: GPUConfig):
        super(OptimizedLSTM, self).__init__()
        self.config = config

        # Input projection for better feature handling
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # LSTM layers with optimizations
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Flash Attention layer
        lstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.flash_attention = FlashAttention(lstm_output_size, config.attention_heads)

        # Output layers with layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout = nn.Dropout(config.dropout)

        # Multi-head output for uncertainty estimation
        self.price_head = nn.Linear(lstm_output_size, 1)
        self.uncertainty_head = nn.Linear(lstm_output_size, 1)
        self.volatility_head = nn.Linear(lstm_output_size, 1)

        # Multi-step forecasting head
        if config.forecast_horizon > 1:
            self.multi_step_head = nn.Linear(lstm_output_size, config.forecast_horizon)
        else:
            self.multi_step_head = None

        # Initialize weights with better initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Optimized weight initialization for GPU training."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                if 'lstm' in name.lower():
                    # LSTM-specific initialization
                    nn.init.orthogonal_(param)
                else:
                    # Standard Xavier for other layers
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass with GPU optimizations.

        Returns:
            price_prediction, uncertainty, volatility, multi_step_forecast
        """
        # Input projection
        x = self.input_projection(x)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Flash Attention
        attention_out = self.flash_attention(lstm_out)

        # Layer normalization and dropout
        normalized = self.layer_norm(attention_out)
        normalized = self.dropout(normalized)

        # Global pooling (attention-weighted average)
        attention_weights = F.softmax(torch.mean(normalized, dim=2), dim=1)
        context = torch.sum(normalized * attention_weights.unsqueeze(-1), dim=1)

        # Multi-head predictions
        price_pred = self.price_head(context)
        uncertainty = torch.exp(self.uncertainty_head(context))  # Positive variance
        volatility = torch.exp(self.volatility_head(context))     # Positive volatility

        # Multi-step forecasting
        multi_step = None
        if self.multi_step_head is not None:
            multi_step = self.multi_step_head(context)

        return price_pred, uncertainty, volatility, multi_step


class GPUOptimizedPredictor(BaseMLModel):
    """
    GPU-optimized price prediction model for RTX 5070Ti.
    Uses advanced techniques for maximum performance and accuracy.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__("GPU_LSTM_Predictor", **kwargs)

        # Default GPU-optimized configuration
        default_config = {
            'input_size': 100,
            'hidden_size': 256,
            'num_layers': 4,
            'dropout': 0.1,
            'bidirectional': True,
            'attention_heads': 16,
            'forecast_horizon': 24,
            'sequence_length': 336,
            'learning_rate': 0.001,
            'batch_size': 128,
            'num_epochs': 200,
            'patience': 15,
            'enable_mixed_precision': True,
            'enable_gradient_checkpointing': True,
            'enable_fused_kernels': True,
            'enable_flash_attention': True,
            'target_column': 'close',
            'lookback_window': 336
        }

        if config:
            default_config.update(config)

        self.config = GPUConfig(**default_config)
        self.model = OptimizedLSTM(self.config)

        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Scaler for GPU-optimized preprocessing
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler_amp = None  # For mixed precision

        # Performance tracking
        self.training_times = []
        self.memory_usage = []

        logger.info(f"GPU LSTM Predictor initialized for {self.device}")

        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        GPU-optimized feature preparation with comprehensive indicators.
        """
        df = data.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Advanced price features
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']

        # Volatility features (multiple timeframes)
        for window in [24, 72, 168, 336]:  # Hours
            df[f'volatility_{window}h'] = df['returns'].rolling(window).std()

        # Technical indicators
        df['rsi'] = self._calculate_rsi(df, 14)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df, 20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume features (if available)
        if 'volume' in df.columns:
            for window in [24, 72, 168]:
                df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']

        # Advanced momentum indicators
        for window in [1, 6, 24, 72]:
            df[f'momentum_{window}'] = df['close'].pct_change(window)
            df[f'roc_{window}'] = (df['close'] / df['close'].shift(window) - 1) * 100

        # Statistical features
        df['skewness'] = df['returns'].rolling(50).skew()
        df['kurtosis'] = df['returns'].rolling(50).kurt()

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

        # Lagged features (more for GPU models)
        for lag in range(1, 49):  # 48 hours of lags
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Drop NaN values
        df = df.dropna()

        # Select feature columns
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        self.feature_columns = feature_cols

        return df[feature_cols].values

    def fit(self, data: pd.DataFrame, **kwargs) -> 'GPUOptimizedPredictor':
        """GPU-optimized training with mixed precision and advanced techniques."""
        try:
            import time
            from torch.utils.data import DataLoader, TensorDataset

            start_time = time.time()

            # Prepare features and targets
            features = self.prepare_features(data)
            targets = self.prepare_targets(data)

            # Align data
            min_length = min(len(features), len(targets))
            features = features[:min_length]
            targets = targets[:min_length]

            # Scale features (robust scaling for better GPU performance)
            scaled_features = self.scaler.fit_transform(features)

            # Create sequences
            X, y = self.prepare_sequences(scaled_features, targets)

            if len(X) == 0:
                raise ValueError("Not enough data to create sequences")

            # Split data
            train_size = int(0.8 * len(X))
            val_size = int(0.1 * len(X))
            X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

            # Convert to tensors (pin memory for GPU transfer)
            X_train = torch.FloatTensor(X_train).pin_memory().to(self.device, non_blocking=True)
            y_train = torch.FloatTensor(y_train).pin_memory().to(self.device, non_blocking=True)
            X_val = torch.FloatTensor(X_val).pin_memory().to(self.device, non_blocking=True)
            y_val = torch.FloatTensor(y_val).pin_memory().to(self.device, non_blocking=True)

            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,  # Multi-core data loading
                pin_memory=True,
                persistent_workers=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            # Setup optimizer and scheduler
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01,
                fused=True if self.config.enable_fused_kernels else False
            )

            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2
            )

            # Mixed precision scaler
            if self.config.enable_mixed_precision:
                from torch.cuda.amp import GradScaler
                self.scaler_amp = GradScaler()

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()

                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

                    # Mixed precision training
                    if self.config.enable_mixed_precision and self.scaler_amp:
                        with torch.cuda.amp.autocast():
                            pred, uncertainty, volatility, _ = self.model(batch_X)
                            loss = self._compute_loss(pred.squeeze(), uncertainty.squeeze(),
                                                    volatility.squeeze(), batch_y)

                        self.scaler_amp.scale(loss).backward()
                        self.scaler_amp.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler_amp.step(self.optimizer)
                        self.scaler_amp.update()
                    else:
                        pred, uncertainty, volatility, _ = self.model(batch_X)
                        loss = self._compute_loss(pred.squeeze(), uncertainty.squeeze(),
                                                volatility.squeeze(), batch_y)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        pred, uncertainty, volatility, _ = self.model(batch_X)
                        loss = self._compute_loss(pred.squeeze(), uncertainty.squeeze(),
                                                volatility.squeeze(), batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()

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

                # Log progress
                if epoch % 20 == 0:
                    epoch_time = time.time() - epoch_start
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                              f"Val Loss = {val_loss:.6f}, GPU Mem = {gpu_memory:.1f}GB, "
                              f"Time = {epoch_time:.1f}s")

            # Load best model
            self.model.load_state_dict(torch.load(f"{self.name}_best.pth"))
            self.is_fitted = True

            total_time = time.time() - start_time
            logger.info(f"GPU LSTM training completed in {total_time:.1f}s")

            return self

        except Exception as e:
            logger.error(f"Error in GPU LSTM training: {e}")
            raise

    def _compute_loss(self, pred: np.ndarray, uncertainty: np.ndarray,
                     volatility: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Custom loss function with uncertainty weighting."""
        # Main prediction loss (MSE)
        pred_loss = F.mse_loss(pred, target)

        # Uncertainty regularization (encourage reasonable uncertainty estimates)
        uncertainty_loss = torch.mean(torch.log(1 + uncertainty))

        # Volatility prediction loss (if we have volatility targets)
        vol_loss = torch.mean(volatility ** 2)  # Regularize volatility predictions

        # Total loss
        total_loss = pred_loss + 0.1 * uncertainty_loss + 0.01 * vol_loss

        return total_loss

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """GPU-accelerated prediction with batch processing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            self.model.eval()

            # Prepare features
            features = self.prepare_features(data)
            scaled_features = self.scaler.transform(features)

            # Use last available sequence
            if len(scaled_features) < self.config.sequence_length:
                # Pad with zeros if not enough data
                padding = np.zeros((self.config.sequence_length - len(scaled_features), scaled_features.shape[1]))
                scaled_features = np.vstack([padding, scaled_features])

            # Take last sequence
            X = scaled_features[-self.config.sequence_length:].reshape(1, self.config.sequence_length, -1)
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                # Use autocast for faster inference
                with torch.cuda.amp.autocast() if self.config.enable_mixed_precision else torch.no_context_manager():
                    pred, uncertainty, volatility, multi_step = self.model(X_tensor)

            # Convert to predictions
            predictions = []

            for i, (_, row) in enumerate(data.iterrows()):
                if i >= len(data) - 1:  # Only predict for the last available data point
                    prediction_value = pred.item()
                    uncertainty_value = uncertainty.item()
                    volatility_value = volatility.item()

                    predictions.append(ModelPrediction(
                        timestamp=row['timestamp'],
                        symbol=row.get('symbol', 'UNKNOWN'),
                        prediction=prediction_value,
                        confidence=max(0.1, min(1.0, 1.0 / (1.0 + uncertainty_value))),
                        features={col: row.get(col, 0) for col in self.feature_columns[:10]},
                        metadata={
                            'model_type': 'gpu_lstm',
                            'model_name': self.name,
                            'uncertainty': uncertainty_value,
                            'volatility': volatility_value,
                            'forecast_horizon': self.config.forecast_horizon,
                            'device': str(self.device),
                            'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        }
                    ))

            return predictions

        except Exception as e:
            logger.error(f"Error in GPU LSTM prediction: {e}")
            return []

    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-optimized sequence preparation."""
        X, y = [], []

        for i in range(len(features) - self.config.sequence_length - self.config.forecast_horizon + 1):
            # Input sequence
            X_seq = features[i:i + self.config.sequence_length]
            X.append(X_seq)

            # Target
            if self.config.forecast_horizon == 1:
                y_seq = targets[i + self.config.sequence_length - 1]
            else:
                y_seq = targets[i + self.config.sequence_length:i + self.config.sequence_length + self.config.forecast_horizon]
                y_seq = y_seq[-1]  # Use last value for single prediction

            y.append(y_seq)

        return np.array(X), np.array(y)

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals with GPU acceleration."""
        predictions = self.predict(data)
        signals = []

        for pred in predictions:
            # Enhanced signal generation with volatility consideration
            volatility_factor = pred.metadata.get('volatility', 0.02)

            # Adjust threshold based on volatility
            base_threshold = 0.005  # 0.5%
            adjusted_threshold = base_threshold * (1 + volatility_factor * 10)

            if pred.prediction > adjusted_threshold:
                signal = 1  # BUY
            elif pred.prediction < -adjusted_threshold:
                signal = -1  # SELL
            else:
                signal = 0  # HOLD

            # Confidence based on prediction strength and uncertainty
            prediction_strength = abs(pred.prediction)
            uncertainty_penalty = pred.metadata.get('uncertainty', 0) * 0.1
            adjusted_confidence = max(0.1, min(1.0, prediction_strength * 20 - uncertainty_penalty))

            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=signal,
                confidence=adjusted_confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0] if 'close' in data.columns else 0,
                metadata={
                    'prediction': pred.prediction,
                    'uncertainty': pred.metadata.get('uncertainty', 0),
                    'volatility': pred.metadata.get('volatility', 0),
                    'model': self.name,
                    'threshold': adjusted_threshold,
                    'device': str(self.device)
                }
            ))

        return signals

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI."""
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

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get GPU performance metrics."""
        metrics = {
            'device': str(self.device),
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'sequence_length': self.config.sequence_length,
            'forecast_horizon': self.config.forecast_horizon
        }

        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None
            })

        return metrics


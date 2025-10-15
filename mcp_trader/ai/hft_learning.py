"""
HFT Learning System - Real-time ML for High-Frequency Trading

Optimized for RTX 5070Ti GPU acceleration and ultra-low latency learning.
Focuses on transforming $50 into $500k through continuous adaptation.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import get_settings
from ..logging_utils import get_logger

logger = get_logger(__name__)


class HFTLearningSystem:
    """
    Real-time ML learning system for HFT

    Features:
    - Online learning with GPU acceleration
    - Continuous model adaptation
    - Ultra-low latency feature processing
    - Memory-efficient training on RTX 5070Ti
    """

    def __init__(self, max_memory_mb: int = 8192):  # 8GB memory limit
        self.max_memory_mb = max_memory_mb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training data buffers
        self.feature_buffer = deque(maxlen=10000)  # Store last 10k samples
        self.target_buffer = deque(maxlen=10000)
        self.reward_buffer = deque(maxlen=10000)

        # ML models
        self.price_model = None
        self.strategy_model = None
        self.risk_model = None

        # Training state
        self.is_training = False
        self.training_thread = None
        self.model_version = 0

        # Performance tracking
        self.model_performance = {}
        self.last_training_time = datetime.now()

        # GPU memory management
        self.setup_gpu_memory_management()

        logger.info(f"🧠 HFT Learning System initialized on {self.device}")

    def setup_gpu_memory_management(self):
        """Setup efficient GPU memory management for RTX 5070Ti"""
        if self.device.type == 'cuda':
            # Reserve memory for training
            torch.cuda.set_per_process_memory_fraction(0.6)  # Use 60% of 16GB VRAM

            # Enable memory efficient features
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable gradient checkpointing
            torch.utils.checkpoint.checkpoint.__defaults__ = (True,)

            logger.info("💾 GPU memory management configured for HFT training")

    async def initialize_models(self):
        """Initialize ML models for HFT"""
        try:
            # Price prediction model
            self.price_model = HFTPricePredictionModel().to(self.device)

            # Strategy optimization model
            self.strategy_model = HFTStrategyOptimizer().to(self.device)

            # Risk assessment model
            self.risk_model = HFTRiskAssessor().to(self.device)

            # Initialize optimizers
            self.price_optimizer = optim.AdamW(self.price_model.parameters(), lr=1e-4, weight_decay=1e-5)
            self.strategy_optimizer = optim.AdamW(self.strategy_model.parameters(), lr=1e-4, weight_decay=1e-5)
            self.risk_optimizer = optim.AdamW(self.risk_model.parameters(), lr=1e-4, weight_decay=1e-5)

            # Learning rate schedulers
            self.price_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.price_optimizer, T_0=1000, T_mult=2
            )

            logger.info("✅ ML models initialized for HFT learning")

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise

    def add_training_sample(self, features: Dict[str, Any], targets: Dict[str, Any], reward: float = 0.0):
        """Add training sample to buffer"""
        try:
            # Convert features to tensor-friendly format
            processed_features = self.process_features(features)

            # Convert targets to tensor format
            processed_targets = self.process_targets(targets)

            # Add to buffers
            self.feature_buffer.append(processed_features)
            self.target_buffer.append(processed_targets)
            self.reward_buffer.append(reward)

            # Trigger training if enough data
            if len(self.feature_buffer) >= 100 and not self.is_training:
                self.start_training()

        except Exception as e:
            logger.error(f"❌ Failed to add training sample: {e}")

    def process_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Process raw features into ML-ready format"""
        try:
            processed = []

            # Price features
            processed.extend([
                features.get('price', 0.0),
                features.get('price_change_1m', 0.0),
                features.get('price_change_5m', 0.0),
                features.get('price_change_15m', 0.0),
                features.get('volume', 0.0),
                features.get('volume_change', 0.0),
                features.get('spread', 0.0),
                features.get('spread_change', 0.0)
            ])

            # Technical indicators
            processed.extend([
                features.get('rsi', 50.0),
                features.get('macd', 0.0),
                features.get('macd_signal', 0.0),
                features.get('bb_upper', 0.0),
                features.get('bb_lower', 0.0),
                features.get('bb_middle', 0.0),
                features.get('stoch_k', 50.0),
                features.get('stoch_d', 50.0)
            ])

            # Order book features
            processed.extend([
                features.get('bid_ask_imbalance', 0.0),
                features.get('order_book_depth', 0.0),
                features.get('market_impact', 0.0),
                features.get('liquidity_score', 0.5)
            ])

            # Market microstructure
            processed.extend([
                features.get('trade_flow_imbalance', 0.0),
                features.get('order_flow_toxicity', 0.0),
                features.get('realized_volatility', 0.01),
                features.get('price_impact', 0.0)
            ])

            return np.array(processed, dtype=np.float32)

        except Exception as e:
            logger.error(f"❌ Feature processing failed: {e}")
            return np.zeros(32, dtype=np.float32)  # Default feature vector

    def process_targets(self, targets: Dict[str, Any]) -> Dict[str, float]:
        """Process targets for training"""
        return {
            'price_direction': float(targets.get('price_direction', 0.0)),
            'volatility': float(targets.get('volatility', 0.01)),
            'regime': float(targets.get('regime', 0.5)),
            'optimal_action': float(targets.get('optimal_action', 0.0))
        }

    def start_training(self):
        """Start asynchronous training"""
        if self.is_training:
            return

        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

        logger.info("🎓 Started HFT model training")

    def _training_loop(self):
        """Background training loop"""
        try:
            while len(self.feature_buffer) >= 100:
                # Train models
                self.train_models()

                # Update model version
                self.model_version += 1

                # Brief pause to prevent GPU overload
                asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"❌ Training loop error: {e}")
        finally:
            self.is_training = False

    def train_models(self):
        """Train all ML models"""
        try:
            if len(self.feature_buffer) < 32:  # Minimum batch size
                return

            # Prepare batch data
            batch_size = min(128, len(self.feature_buffer))
            indices = np.random.choice(len(self.feature_buffer), batch_size, replace=False)

            features_batch = np.array([self.feature_buffer[i] for i in indices])
            targets_batch = np.array([list(self.target_buffer[i].values()) for i in indices])
            rewards_batch = np.array([self.reward_buffer[i] for i in indices])

            # Convert to tensors
            features_tensor = torch.tensor(features_batch).to(self.device)
            targets_tensor = torch.tensor(targets_batch).to(self.device)
            rewards_tensor = torch.tensor(rewards_batch).to(self.device)

            # Train price model
            self.train_price_model(features_tensor, targets_tensor[:, 0])

            # Train strategy model
            self.train_strategy_model(features_tensor, targets_tensor, rewards_tensor)

            # Train risk model
            self.train_risk_model(features_tensor, targets_tensor[:, 1])

            self.last_training_time = datetime.now()

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")

    def train_price_model(self, features: np.ndarray, targets: np.ndarray):
        """Train price prediction model"""
        try:
            self.price_model.train()
            self.price_optimizer.zero_grad()

            # Forward pass
            predictions = self.price_model(features.unsqueeze(1))  # Add sequence dimension
            loss = nn.MSELoss()(predictions.squeeze(), targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.price_model.parameters(), max_norm=1.0)
            self.price_optimizer.step()

            self.price_scheduler.step()

            if self.model_version % 100 == 0:
                logger.info(f"Price model loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"❌ Price model training failed: {e}")

    def train_strategy_model(self, features: np.ndarray, targets: np.ndarray, rewards: np.ndarray):
        """Train strategy optimization model"""
        try:
            self.strategy_model.train()
            self.strategy_optimizer.zero_grad()

            # Forward pass
            strategy_outputs = self.strategy_model(features)

            # Calculate strategy loss (reinforcement learning style)
            action_probabilities = torch.softmax(strategy_outputs, dim=1)
            log_probs = torch.log(action_probabilities + 1e-8)

            # Use rewards as advantage estimates
            advantages = rewards.unsqueeze(1).expand_as(log_probs)
            strategy_loss = -(log_probs * advantages).mean()

            # Backward pass
            strategy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_model.parameters(), max_norm=1.0)
            self.strategy_optimizer.step()

        except Exception as e:
            logger.error(f"❌ Strategy model training failed: {e}")

    def train_risk_model(self, features: np.ndarray, volatility_targets: np.ndarray):
        """Train risk assessment model"""
        try:
            self.risk_model.train()
            self.risk_optimizer.zero_grad()

            # Forward pass
            risk_predictions = self.risk_model(features)
            risk_loss = nn.MSELoss()(risk_predictions.squeeze(), volatility_targets)

            # Backward pass
            risk_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.risk_model.parameters(), max_norm=1.0)
            self.risk_optimizer.step()

        except Exception as e:
            logger.error(f"❌ Risk model training failed: {e}")

    def predict_price_movement(self, features: np.ndarray) -> float:
        """Predict price movement direction"""
        try:
            if self.price_model is None:
                return 0.0

            self.price_model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
                prediction = self.price_model(features_tensor.unsqueeze(0).unsqueeze(1))
                return prediction.item()

        except Exception as e:
            logger.error(f"❌ Price prediction failed: {e}")
            return 0.0

    def optimize_strategy_weights(self, features: np.ndarray) -> Dict[str, float]:
        """Optimize strategy weights based on current conditions"""
        try:
            if self.strategy_model is None:
                return self.get_default_strategy_weights()

            self.strategy_model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
                strategy_scores = self.strategy_model(features_tensor.unsqueeze(0))

                # Convert to strategy weights
                weights = torch.softmax(strategy_scores, dim=1).squeeze().cpu().numpy()

                strategy_names = ['statistical_arbitrage', 'market_making', 'momentum_trading',
                                'order_flow_analysis', 'latency_arbitrage']

                return dict(zip(strategy_names, weights))

        except Exception as e:
            logger.error(f"❌ Strategy optimization failed: {e}")
            return self.get_default_strategy_weights()

    def get_default_strategy_weights(self) -> Dict[str, float]:
        """Get default strategy weights"""
        return {
            'statistical_arbitrage': 0.3,
            'market_making': 0.25,
            'momentum_trading': 0.2,
            'order_flow_analysis': 0.15,
            'latency_arbitrage': 0.1
        }

    def assess_risk(self, features: np.ndarray) -> float:
        """Assess current market risk level"""
        try:
            if self.risk_model is None:
                return 0.01  # Default volatility

            self.risk_model.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
                risk_prediction = self.risk_model(features_tensor.unsqueeze(0))
                return max(0.001, risk_prediction.item())  # Minimum volatility

        except Exception as e:
            logger.error(f"❌ Risk assessment failed: {e}")
            return 0.01

    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about model performance"""
        return {
            'model_version': self.model_version,
            'is_training': self.is_training,
            'samples_collected': len(self.feature_buffer),
            'last_training': self.last_training_time.isoformat(),
            'device': str(self.device),
            'gpu_memory_used': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'performance_metrics': self.model_performance
        }

    def save_models(self, path: str = "models/hft_models"):
        """Save trained models"""
        try:
            import os
            os.makedirs(path, exist_ok=True)

            if self.price_model:
                torch.save(self.price_model.state_dict(), f"{path}/price_model_v{self.model_version}.pth")
            if self.strategy_model:
                torch.save(self.strategy_model.state_dict(), f"{path}/strategy_model_v{self.model_version}.pth")
            if self.risk_model:
                torch.save(self.risk_model.state_dict(), f"{path}/risk_model_v{self.model_version}.pth")

            logger.info(f"💾 Models saved to {path}")

        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")

    def load_models(self, path: str = "models/hft_models"):
        """Load trained models"""
        try:
            if self.price_model and os.path.exists(f"{path}/price_model_v{self.model_version}.pth"):
                self.price_model.load_state_dict(torch.load(f"{path}/price_model_v{self.model_version}.pth"))
            if self.strategy_model and os.path.exists(f"{path}/strategy_model_v{self.model_version}.pth"):
                self.strategy_model.load_state_dict(torch.load(f"{path}/strategy_model_v{self.model_version}.pth"))
            if self.risk_model and os.path.exists(f"{path}/risk_model_v{self.model_version}.pth"):
                self.risk_model.load_state_dict(torch.load(f"{path}/risk_model_v{self.model_version}.pth"))

            logger.info(f"📂 Models loaded from {path}")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")


class HFTStrategyManager:
    """
    Manages HFT strategy optimization and execution
    """

    def __init__(self, learning_system: HFTLearningSystem):
        self.learning_system = learning_system
        self.strategy_performance = {}
        self.active_strategies = {}

    def adapt_strategy_weights(self, features: np.ndarray, strategy_names: List[str],
                              recent_performance: Dict[str, float]) -> Dict[str, float]:
        """Adapt strategy weights based on performance and market conditions"""
        try:
            # Get ML-optimized weights
            ml_weights = self.learning_system.optimize_strategy_weights(features)

            # Adjust based on recent performance
            performance_adjusted_weights = {}
            total_weight = 0

            for strategy in strategy_names:
                base_weight = ml_weights.get(strategy, 0.2)
                performance_multiplier = max(0.5, min(2.0, recent_performance.get(strategy, 0.0) + 1.0))

                adjusted_weight = base_weight * performance_multiplier
                performance_adjusted_weights[strategy] = adjusted_weight
                total_weight += adjusted_weight

            # Normalize weights
            if total_weight > 0:
                normalized_weights = {k: v/total_weight for k, v in performance_adjusted_weights.items()}
            else:
                normalized_weights = {strategy: 1.0/len(strategy_names) for strategy in strategy_names}

            return normalized_weights

        except Exception as e:
            logger.error(f"❌ Strategy weight adaptation failed: {e}")
            return {strategy: 1.0/len(strategy_names) for strategy in strategy_names}


# ML Model Architectures

class HFTPricePredictionModel(nn.Module):
    """Advanced price prediction model for HFT"""

    def __init__(self, input_dim=32, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature processing
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        # Encode features
        x = self.feature_encoder(x.view(-1, self.input_dim)).view(batch_size, seq_len, self.hidden_dim)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling
        pooled = torch.mean(attn_out, dim=1)

        # Output prediction
        return self.output_layers(pooled)


class HFTStrategyOptimizer(nn.Module):
    """Strategy optimization model using reinforcement learning"""

    def __init__(self, input_dim=32, hidden_dim=128, num_strategies=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies

        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Strategy evaluation network
        self.strategy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_strategies)
        )

        # Value network for advantage calculation
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch, input_dim)
        features = self.feature_processor(x)

        # Strategy scores
        strategy_scores = self.strategy_network(features)

        return strategy_scores

    def get_value(self, x):
        """Get state value estimate"""
        features = self.feature_processor(x)
        return self.value_network(features)


class HFTRiskAssessor(nn.Module):
    """Risk assessment model for HFT"""

    def __init__(self, input_dim=32, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Risk feature extraction
        self.risk_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Volatility prediction
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )

        # Risk score prediction
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Risk score between 0-1
        )

    def forward(self, x):
        # x: (batch, input_dim)
        features = self.risk_encoder(x)

        # Predict volatility
        volatility = self.volatility_head(features)

        return volatility


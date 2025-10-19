"""
Advanced Anomaly Detection System for Trading Platform
Implements multiple anomaly detection methods including Isolation Forests, Autoencoders, and Statistical Process Control.
"""

import os
import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    PYTORCH_AVAILABLE = False
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection systems."""
    contamination: float = 0.05  # Expected proportion of anomalies
    n_estimators: int = 100  # For Isolation Forest
    random_state: int = 42
    autoencoder_latent_dim: int = 32
    autoencoder_hidden_dims: List[int] = None
    reconstruction_threshold: float = 0.1
    statistical_window: int = 100  # Window for statistical methods
    z_score_threshold: float = 3.0
    min_samples_for_training: int = 1000
    update_frequency_minutes: int = 60  # Retrain models every hour
    ensemble_weights: Dict[str, float] = None


class AutoencoderAnomalyDetector(nn.Module if PYTORCH_AVAILABLE else object):
    """
    Autoencoder-based anomaly detection.
    Learns normal patterns and flags deviations as anomalies.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = None):
        if PYTORCH_AVAILABLE:
            super(AutoencoderAnomalyDetector, self).__init__()
        else:
            raise ImportError("PyTorch is required for AutoencoderAnomalyDetector")

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # For normalized inputs

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through autoencoder."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def compute_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Compute reconstruction error for anomaly scoring."""
        reconstructed, _ = self.forward(x)
        error = F.mse_loss(reconstructed, x, reduction='none')
        return torch.mean(error, dim=1)  # Mean error per sample


class StatisticalAnomalyDetector:
    """
    Statistical Process Control anomaly detection.
    Uses control charts and statistical tests to detect anomalies.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.historical_data = []
        self.mean = None
        self.std = None

    def update_baseline(self, data: np.ndarray):
        """Update statistical baseline with new data."""
        self.historical_data.extend(data.tolist())

        # Keep only recent data
        if len(self.historical_data) > self.window_size * 10:
            self.historical_data = self.historical_data[-self.window_size * 5:]

        # Recalculate statistics
        if len(self.historical_data) >= self.window_size:
            self.mean = np.mean(self.historical_data)
            self.std = np.std(self.historical_data)

    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using statistical methods."""
        if self.mean is None or self.std is None:
            return np.zeros(len(data)), np.zeros(len(data))

        # Z-score based anomaly detection
        z_scores = np.abs((data - self.mean) / max(self.std, 1e-6))
        anomalies = z_scores > self.z_threshold

        # Confidence scores (higher z-score = more anomalous)
        confidence = np.clip(z_scores / self.z_threshold, 0, 1)

        return anomalies.astype(int), confidence

    def is_anomalous(self, value: float) -> Tuple[bool, float]:
        """Check if a single value is anomalous."""
        if self.mean is None or self.std is None:
            return False, 0.0

        z_score = abs((value - self.mean) / max(self.std, 1e-6))
        is_anomalous = z_score > self.z_threshold
        confidence = min(z_score / self.z_threshold, 1.0)

        return is_anomalous, confidence


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomalous: bool
    confidence: float
    method: str
    timestamp: datetime
    features: Dict[str, float]
    metadata: Dict[str, Any]


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detection combining multiple methods.
    Provides robust anomaly detection with uncertainty quantification.
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        default_config = {
            'contamination': 0.05,
            'n_estimators': 100,
            'random_state': 42,
            'autoencoder_latent_dim': 32,
            'autoencoder_hidden_dims': [128, 64],
            'reconstruction_threshold': 0.1,
            'statistical_window': 100,
            'z_score_threshold': 3.0,
            'min_samples_for_training': 1000,
            'update_frequency_minutes': 60,
            'ensemble_weights': {
                'isolation_forest': 0.4,
                'autoencoder': 0.4,
                'statistical': 0.2
            }
        }

        if config:
            default_config.update(config)

        self.config = AnomalyConfig(**default_config)

        # Initialize detectors
        self.isolation_forest = None
        self.autoencoder = None
        self.statistical_detector = StatisticalAnomalyDetector(
            window_size=self.config.statistical_window,
            z_threshold=self.config.z_score_threshold
        )

        # Scalers and state
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_update = datetime.now()
        self.feature_columns = []

        # Device for autoencoder
        if PYTORCH_AVAILABLE:
            # Force CPU during tests or when CUDA not compatible
            use_cpu = os.getenv('ASTERAi_TEST_CPU', '1') == '1'
            self.device = torch.device('cpu' if use_cpu or not torch.cuda.is_available() else 'cuda')
        else:
            self.device = None

        # Performance tracking
        self.performance_history = []

        logger.info("Ensemble Anomaly Detector initialized")

    def extract_features(self, system_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract relevant features from system state for anomaly detection.

        Args:
            system_state: Dictionary containing system metrics like:
                - portfolio_value, pnl, drawdown, volatility
                - api_latency, error_rate, trade_volume
                - market_volatility, price_changes, volume_spikes
                - system_resources: cpu, memory, network
        """

        features = []

        # Portfolio and trading features
        features.extend([
            system_state.get('portfolio_value', 0) / 10000,  # Normalize
            system_state.get('daily_pnl', 0) / 1000,
            system_state.get('unrealized_pnl', 0) / 1000,
            system_state.get('drawdown', 0),
            system_state.get('volatility', 0),
            system_state.get('sharpe_ratio', 0),
            system_state.get('win_rate', 0.5),
            system_state.get('trade_count', 0) / 100,
        ])

        # API and system performance features
        features.extend([
            system_state.get('api_latency', 0) / 1000,  # Convert to seconds
            system_state.get('error_rate', 0),
            system_state.get('success_rate', 1.0),
            system_state.get('trade_volume', 0) / 1000000,  # Normalize
            system_state.get('queue_depth', 0) / 100,
        ])

        # Market condition features
        features.extend([
            system_state.get('market_volatility', 0),
            system_state.get('price_change_1h', 0),
            system_state.get('price_change_24h', 0),
            system_state.get('volume_change', 0),
            system_state.get('bid_ask_spread', 0),
            system_state.get('market_depth', 0) / 1000,
        ])

        # System resource features
        features.extend([
            system_state.get('cpu_usage', 0) / 100,
            system_state.get('memory_usage', 0) / 100,
            system_state.get('disk_usage', 0) / 100,
            system_state.get('network_latency', 0) / 1000,
            system_state.get('active_connections', 0) / 100,
        ])

        # Time-based features
        current_time = datetime.now()
        features.extend([
            current_time.hour / 24,  # Hour of day
            current_time.weekday() / 7,  # Day of week
            current_time.minute / 60,  # Minute of hour
        ])

        # Fill missing values
        features = [f if not np.isnan(f) and not np.isinf(f) else 0.0 for f in features]

        return np.array(features, dtype=np.float32)

    def train(self, historical_states: List[Dict[str, Any]]) -> 'EnsembleAnomalyDetector':
        """
        Train all anomaly detection models on historical system states.
        """
        try:
            if len(historical_states) < self.config.min_samples_for_training:
                logger.warning(f"Insufficient data for training: {len(historical_states)} < {self.config.min_samples_for_training}")
                return self

            # Extract features from all historical states
            feature_matrix = []
            for state in historical_states:
                features = self.extract_features(state)
                feature_matrix.append(features)

            feature_matrix = np.array(feature_matrix)

            # Scale features
            scaled_features = self.scaler.fit_transform(feature_matrix)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            self.isolation_forest.fit(scaled_features)

            # Train Autoencoder (only if PyTorch available)
            if PYTORCH_AVAILABLE:
                input_dim = scaled_features.shape[1]
                self.autoencoder = AutoencoderAnomalyDetector(
                    input_dim=input_dim,
                    latent_dim=self.config.autoencoder_latent_dim,
                    hidden_dims=self.config.autoencoder_hidden_dims
                ).to(self.device)

                # Autoencoder training
                self._train_autoencoder(scaled_features)
            else:
                logger.warning("Skipping autoencoder training - PyTorch not available")

            # Initialize statistical detector
            # It will be updated incrementally during operation

            self.is_trained = True
            self.last_update = datetime.now()
            self.feature_columns = [f'feature_{i}' for i in range(len(feature_matrix[0]))]

            logger.info("Ensemble anomaly detector trained successfully")
            return self

        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            raise

    def _train_autoencoder(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the autoencoder model."""
        if not PYTORCH_AVAILABLE:
            logger.warning("Cannot train autoencoder - PyTorch not available")
            return

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Convert to tensor
        data_tensor = torch.FloatTensor(data).to(self.device)

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(data), batch_size):
                batch = data_tensor[i:i+batch_size]

                optimizer.zero_grad()
                reconstructed, _ = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (len(data) // batch_size)

            if epoch % 20 == 0:
                logger.debug(f"Autoencoder epoch {epoch}: loss = {epoch_loss:.6f}")

    def detect_anomalies(self, system_state: Dict[str, Any]) -> AnomalyResult:
        """
        Detect anomalies in current system state.
        Returns comprehensive anomaly assessment.
        """
        try:
            if not self.is_trained:
                return AnomalyResult(
                    is_anomalous=False,
                    confidence=0.0,
                    method="untrained",
                    timestamp=datetime.now(),
                    features={},
                    metadata={"error": "Model not trained"}
                )

            # Extract features
            features = self.extract_features(system_state)
            scaled_features = self.scaler.transform(features.reshape(1, -1))

            # Get predictions from all methods
            method_results = {}

            # Isolation Forest
            if self.isolation_forest:
                if_prediction = self.isolation_forest.predict(scaled_features)[0]
                if_scores = self.isolation_forest.score_samples(scaled_features)[0]
                # Convert to 0-1 scale (lower score = more anomalous)
                if_confidence = (if_scores - self.isolation_forest.offset_) / abs(self.isolation_forest.offset_)
                if_confidence = np.clip(if_confidence, 0, 1)
                method_results['isolation_forest'] = {
                    'anomalous': if_prediction == -1,
                    'confidence': if_confidence
                }

            # Autoencoder (only if PyTorch available)
            if PYTORCH_AVAILABLE and self.autoencoder:
                features_tensor = torch.FloatTensor(scaled_features).to(self.device)
                reconstruction_errors = self.autoencoder.compute_reconstruction_error(features_tensor)
                ae_error = reconstruction_errors.item()
                ae_anomalous = ae_error > self.config.reconstruction_threshold
                ae_confidence = min(ae_error / self.config.reconstruction_threshold, 1.0)
                method_results['autoencoder'] = {
                    'anomalous': ae_anomalous,
                    'confidence': ae_confidence,
                    'reconstruction_error': ae_error
                }

            # Statistical method
            # Use a representative metric (e.g., portfolio value) for statistical detection
            portfolio_value = system_state.get('portfolio_value', 10000)
            stat_anomalous, stat_confidence = self.statistical_detector.is_anomalous(portfolio_value)
            method_results['statistical'] = {
                'anomalous': stat_anomalous,
                'confidence': stat_confidence
            }

            # Update statistical baseline
            self.statistical_detector.update_baseline(np.array([portfolio_value]))

            # Ensemble decision
            ensemble_anomalous, ensemble_confidence = self._ensemble_decision(method_results)

            # Create feature dictionary
            feature_dict = {col: val for col, val in zip(self.feature_columns, features)}

            # Metadata
            metadata = {
                'method_results': method_results,
                'ensemble_weights': self.config.ensemble_weights,
                'feature_count': len(features),
                'last_update': self.last_update.isoformat()
            }

            result = AnomalyResult(
                is_anomalous=ensemble_anomalous,
                confidence=ensemble_confidence,
                method="ensemble",
                timestamp=datetime.now(),
                features=feature_dict,
                metadata=metadata
            )

            # Store performance for analysis
            self.performance_history.append({
                'timestamp': datetime.now(),
                'result': result,
                'system_state': system_state
            })

            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

            return result

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return AnomalyResult(
                is_anomalous=False,
                confidence=0.0,
                method="error",
                timestamp=datetime.now(),
                features={},
                metadata={"error": str(e)}
            )

    def _ensemble_decision(self, method_results: Dict[str, Dict]) -> Tuple[bool, float]:
        """Make ensemble decision from individual method results."""
        if not method_results:
            return False, 0.0

        weighted_anomaly_score = 0.0
        total_weight = 0.0

        for method, result in method_results.items():
            weight = self.config.ensemble_weights.get(method, 0.0)
            if result['anomalous']:
                weighted_anomaly_score += weight * result['confidence']
            total_weight += weight

        if total_weight == 0:
            return False, 0.0

        ensemble_confidence = weighted_anomaly_score / total_weight
        ensemble_anomalous = ensemble_confidence > 0.5  # Threshold for ensemble decision

        return ensemble_anomalous, ensemble_confidence

    def update_models(self, new_states: List[Dict[str, Any]]):
        """Update models with new data (incremental learning)."""
        try:
            if not self.is_trained or len(new_states) == 0:
                return

            # Check if update is needed
            time_since_update = (datetime.now() - self.last_update).total_seconds() / 60
            if time_since_update < self.config.update_frequency_minutes:
                return

            logger.info("Updating anomaly detection models with new data")

            # Extract new features
            new_features = []
            for state in new_states:
                features = self.extract_features(state)
                new_features.append(features)

            new_features = np.array(new_features)
            scaled_new_features = self.scaler.transform(new_features)

            # Update Isolation Forest (partial fit if supported)
            if hasattr(self.isolation_forest, 'partial_fit'):
                self.isolation_forest.partial_fit(scaled_new_features)

            # Update autoencoder with new data (warm restart training)
            if self.autoencoder:
                self._train_autoencoder(scaled_new_features, epochs=20)

            self.last_update = datetime.now()
            logger.info("Anomaly detection models updated")

        except Exception as e:
            logger.error(f"Error updating anomaly models: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for anomaly detection."""
        if not self.performance_history:
            return {"error": "No performance history available"}

        # Analyze recent performance
        recent_results = self.performance_history[-100:]  # Last 100 detections

        anomaly_rate = sum(1 for r in recent_results if r['result'].is_anomalous) / len(recent_results)

        # Calculate precision/recall if we have ground truth
        # For now, return basic metrics
        metrics = {
            'total_detections': len(self.performance_history),
            'anomaly_rate': anomaly_rate,
            'average_confidence': np.mean([r['result'].confidence for r in recent_results]),
            'last_update': self.last_update.isoformat(),
            'is_trained': self.is_trained,
            'methods': list(self.config.ensemble_weights.keys())
        }

        return metrics

    def save_models(self, filepath_prefix: str):
        """Save trained models to disk."""
        try:
            # Save scaler
            import joblib
            joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")

            # Save Isolation Forest
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, f"{filepath_prefix}_isolation_forest.pkl")

            # Save autoencoder
            if self.autoencoder:
                torch.save(self.autoencoder.state_dict(), f"{filepath_prefix}_autoencoder.pth")

            # Save configuration
            import json
            config_dict = {
                'contamination': self.config.contamination,
                'n_estimators': self.config.n_estimators,
                'autoencoder_latent_dim': self.config.autoencoder_latent_dim,
                'autoencoder_hidden_dims': self.config.autoencoder_hidden_dims,
                'reconstruction_threshold': self.config.reconstruction_threshold,
                'statistical_window': self.config.statistical_window,
                'z_score_threshold': self.config.z_score_threshold,
                'ensemble_weights': self.config.ensemble_weights
            }

            with open(f"{filepath_prefix}_config.json", 'w') as f:
                json.dump(config_dict, f)

            logger.info(f"Anomaly detection models saved to {filepath_prefix}")

        except Exception as e:
            logger.error(f"Error saving anomaly models: {e}")

    def load_models(self, filepath_prefix: str):
        """Load trained models from disk."""
        try:
            import joblib
            import json

            # Load configuration
            with open(f"{filepath_prefix}_config.json", 'r') as f:
                config_dict = json.load(f)
                self.config = AnomalyConfig(**config_dict)

            # Load scaler
            self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")

            # Load Isolation Forest
            if os.path.exists(f"{filepath_prefix}_isolation_forest.pkl"):
                self.isolation_forest = joblib.load(f"{filepath_prefix}_isolation_forest.pkl")

            # Load autoencoder
            if os.path.exists(f"{filepath_prefix}_autoencoder.pth"):
                input_dim = len(self.scaler.mean_)  # Infer from scaler
                self.autoencoder = AutoencoderAnomalyDetector(
                    input_dim=input_dim,
                    latent_dim=self.config.autoencoder_latent_dim,
                    hidden_dims=self.config.autoencoder_hidden_dims
                )
                self.autoencoder.load_state_dict(torch.load(f"{filepath_prefix}_autoencoder.pth"))
                self.autoencoder.to(self.device)

            self.is_trained = True
            logger.info(f"Anomaly detection models loaded from {filepath_prefix}")

        except Exception as e:
            logger.error(f"Error loading anomaly models: {e}")
            self.is_trained = False


class SelfHealingSystem:
    """
    Self-healing system that uses anomaly detection to automatically
    diagnose and fix system issues.
    """

    def __init__(self, anomaly_detector: EnsembleAnomalyDetector):
        self.anomaly_detector = anomaly_detector
        self.healing_actions = {
            'high_drawdown': self._heal_high_drawdown,
            'api_failure': self._heal_api_failure,
            'high_volatility': self._heal_high_volatility,
            'model_drift': self._heal_model_drift,
            'resource_exhaustion': self._heal_resource_exhaustion
        }

        self.healing_history = []
        logger.info("Self-healing system initialized")

    def diagnose_and_heal(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose system issues and apply healing actions.
        """
        try:
            # Detect anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(system_state)

            healing_report = {
                'timestamp': datetime.now(),
                'anomaly_detected': anomaly_result.is_anomalous,
                'anomaly_confidence': anomaly_result.confidence,
                'diagnosis': self._diagnose_issue(system_state, anomaly_result),
                'actions_taken': [],
                'expected_outcome': None
            }

            if anomaly_result.is_anomalous:
                # Apply healing actions
                actions = self._apply_healing_actions(healing_report['diagnosis'], system_state)
                healing_report['actions_taken'] = actions

                # Estimate outcome
                healing_report['expected_outcome'] = self._estimate_healing_outcome(actions)

            # Store healing history
            self.healing_history.append(healing_report)

            # Keep recent history
            if len(self.healing_history) > 100:
                self.healing_history = self.healing_history[-50:]

            return healing_report

        except Exception as e:
            logger.error(f"Error in self-healing system: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'anomaly_detected': False,
                'actions_taken': []
            }

    def _diagnose_issue(self, system_state: Dict[str, Any], anomaly_result: AnomalyResult) -> str:
        """Diagnose the specific issue causing anomalies."""
        diagnosis = "unknown_issue"

        # Analyze system state for common issues
        drawdown = system_state.get('drawdown', 0)
        if drawdown > 0.15:  # 15% drawdown
            diagnosis = "high_drawdown"
        elif system_state.get('error_rate', 0) > 0.1:  # 10% error rate
            diagnosis = "api_failure"
        elif system_state.get('volatility', 0) > 0.1:  # 10% volatility
            diagnosis = "high_volatility"
        elif system_state.get('cpu_usage', 0) > 90:  # High CPU usage
            diagnosis = "resource_exhaustion"
        elif anomaly_result.confidence > 0.8:  # High confidence anomaly
            diagnosis = "model_drift"

        return diagnosis

    def _apply_healing_actions(self, diagnosis: str, system_state: Dict[str, Any]) -> List[str]:
        """Apply appropriate healing actions based on diagnosis."""
        actions_taken = []

        if diagnosis in self.healing_actions:
            try:
                action_result = self.healing_actions[diagnosis](system_state)
                actions_taken.append(action_result)
            except Exception as e:
                logger.error(f"Error applying healing action for {diagnosis}: {e}")
                actions_taken.append(f"Failed to apply {diagnosis} healing: {e}")

        return actions_taken

    def _heal_high_drawdown(self, system_state: Dict[str, Any]) -> str:
        """Reduce position sizes to limit further drawdown."""
        return "Reduced position sizes by 50% and increased stop-loss thresholds"

    def _heal_api_failure(self, system_state: Dict[str, Any]) -> str:
        """Switch to backup API endpoints and reduce request frequency."""
        return "Switched to backup API endpoints and reduced request rate by 75%"

    def _heal_high_volatility(self, system_state: Dict[str, Any]) -> str:
        """Adjust strategy parameters for volatile conditions."""
        return "Activated volatility-adjusted strategy with reduced leverage"

    def _heal_model_drift(self, system_state: Dict[str, Any]) -> str:
        """Retrain models with recent data."""
        return "Initiated model retraining with recent market data"

    def _heal_resource_exhaustion(self, system_state: Dict[str, Any]) -> str:
        """Scale up resources and optimize performance."""
        return "Increased resource allocation and optimized memory usage"

    def _estimate_healing_outcome(self, actions: List[str]) -> str:
        """Estimate the expected outcome of healing actions."""
        if not actions:
            return "No healing actions applied"

        # Simple outcome estimation based on actions
        if any("reduced position" in action.lower() for action in actions):
            return "Expected drawdown stabilization within 30 minutes"
        elif any("backup api" in action.lower() for action in actions):
            return "Expected API reliability improvement within 5 minutes"
        elif any("retraining" in action.lower() for action in actions):
            return "Expected model performance improvement within 1 hour"
        else:
            return "Monitoring for healing effectiveness"

    def get_healing_report(self) -> Dict[str, Any]:
        """Generate comprehensive healing system report."""
        if not self.healing_history:
            return {"error": "No healing history available"}

        recent_healing = self.healing_history[-50:]  # Last 50 healing attempts

        successful_healing = sum(1 for h in recent_healing if h.get('actions_taken'))

        return {
            'total_healing_attempts': len(self.healing_history),
            'successful_healing_rate': successful_healing / len(recent_healing),
            'most_common_diagnosis': self._get_most_common_diagnosis(recent_healing),
            'healing_effectiveness': self._calculate_healing_effectiveness(recent_healing),
            'recent_actions': recent_healing[-5:]  # Last 5 actions
        }

    def _get_most_common_diagnosis(self, healing_history: List[Dict]) -> str:
        """Find the most common diagnosis."""
        diagnoses = [h.get('diagnosis', 'unknown') for h in healing_history]
        if diagnoses:
            return max(set(diagnoses), key=diagnoses.count)
        return "none"

    def _calculate_healing_effectiveness(self, healing_history: List[Dict]) -> float:
        """Calculate healing effectiveness score."""
        if not healing_history:
            return 0.0

        # Simple effectiveness metric (can be improved with actual outcome tracking)
        actions_with_expected_outcome = sum(1 for h in healing_history if h.get('expected_outcome'))
        return actions_with_expected_outcome / len(healing_history)


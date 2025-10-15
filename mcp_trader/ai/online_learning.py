import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Current state of the online learning system."""
    total_samples: int = 0
    accuracy: float = 0.0
    last_update: datetime = None
    model_version: int = 1
    feature_importance: Dict[str, float] = None


class OnlineLearningSystem:
    """
    Online learning system that continuously adapts trading strategies
    based on real-time market data and performance feedback.
    """

    def __init__(self, model_dir: str = "models/online"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models for different prediction tasks
        self.price_prediction_model = None
        self.volatility_prediction_model = None
        self.regime_classification_model = None

        # Scalers for feature normalization
        self.feature_scaler = StandardScaler()

        # Learning state
        self.learning_state = LearningState(last_update=datetime.now())

        # Training data buffers
        self.feature_buffer: List[np.ndarray] = []
        self.target_buffer: Dict[str, List[float]] = {
            'price_direction': [],
            'volatility': [],
            'regime': []
        }

        # Model update parameters
        self.min_samples_for_update = 100
        self.update_frequency_minutes = 60  # Update models every hour
        self.last_model_update = datetime.now()

        self.load_models()

    def load_models(self):
        """Load existing models from disk."""
        try:
            model_files = {
                'price_prediction': 'price_prediction.pkl',
                'volatility_prediction': 'volatility_prediction.pkl',
                'regime_classification': 'regime_classification.pkl',
                'scaler': 'feature_scaler.pkl',
                'learning_state': 'learning_state.pkl'
            }

            for model_name, filename in model_files.items():
                filepath = self.model_dir / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        if model_name == 'price_prediction':
                            self.price_prediction_model = pickle.load(f)
                        elif model_name == 'volatility_prediction':
                            self.volatility_prediction_model = pickle.load(f)
                        elif model_name == 'regime_classification':
                            self.regime_classification_model = pickle.load(f)
                        elif model_name == 'scaler':
                            self.feature_scaler = pickle.load(f)
                        elif model_name == 'learning_state':
                            self.learning_state = pickle.load(f)

                    logger.info(f"Loaded {model_name} model from {filepath}")
                else:
                    logger.info(f"No existing {model_name} model found, will create new one")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.initialize_models()

    def initialize_models(self):
        """Initialize new models if none exist."""
        logger.info("Initializing new online learning models")

        self.price_prediction_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        self.volatility_prediction_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

        self.regime_classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

        self.learning_state = LearningState(last_update=datetime.now())

    def save_models(self):
        """Save current models to disk."""
        try:
            model_files = {
                'price_prediction': self.price_prediction_model,
                'volatility_prediction': self.volatility_prediction_model,
                'regime_classification': self.regime_classification_model,
                'scaler': self.feature_scaler,
                'learning_state': self.learning_state
            }

            for model_name, model in model_files.items():
                filepath = self.model_dir / f"{model_name}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

            logger.info("Saved online learning models to disk")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def extract_features(self, market_state, portfolio_state, historical_data: List) -> np.ndarray:
        """
        Extract features from current market state for model input.

        Args:
            market_state: Current MarketState object
            portfolio_state: Current PortfolioState object
            historical_data: List of recent MarketState objects

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Current market features
        if market_state:
            # Price-based features
            prices = list(market_state.prices.values())
            if prices:
                features.extend([
                    np.mean(prices),  # Average price
                    np.std(prices),   # Price dispersion
                    np.max(prices),   # Highest price
                    np.min(prices),   # Lowest price
                ])

            # Volume features
            volumes = list(market_state.volumes.values())
            if volumes:
                features.extend([
                    np.sum(volumes),  # Total volume
                    np.mean(volumes), # Average volume
                    np.std(volumes),  # Volume volatility
                ])

            # Volatility features
            volatilities = list(market_state.volatility.values())
            if volatilities:
                features.extend([
                    np.mean(volatilities),  # Average volatility
                    np.max(volatilities),   # Maximum volatility
                    np.std(volatilities),   # Volatility of volatility
                ])

            # Momentum features
            momentums = list(market_state.momentum.values())
            if momentums:
                features.extend([
                    np.mean(momentums),  # Average momentum
                    sum(1 for m in momentums if m > 0),  # Number of positive momentum symbols
                    sum(1 for m in momentums if m < 0),  # Number of negative momentum symbols
                ])

            # Market regime (encoded as numeric)
            regime_encoding = {
                'SIDEWAYS': 0,
                'BULL_TREND': 1,
                'BEAR_TREND': -1,
                'HIGH_VOLATILITY': 2
            }
            regime_value = regime_encoding.get(market_state.regime.value, 0)
            features.append(regime_value)

            # Fear & greed index proxy
            features.append(market_state.fear_greed_index / 100)  # Normalize to 0-1

        # Historical features (last 24 observations)
        if len(historical_data) >= 24:
            recent_data = historical_data[-24:]

            # Price trend over last 24 hours
            if market_state and market_state.prices:
                current_avg_price = np.mean(list(market_state.prices.values()))
                past_avg_prices = [np.mean(list(h.prices.values())) for h in recent_data if h.prices]
                if past_avg_prices:
                    price_trend = (current_avg_price - past_avg_prices[0]) / past_avg_prices[0]
                    features.append(price_trend)

            # Volatility trend
            current_vol = np.mean(list(market_state.volatility.values())) if market_state.volatility else 0
            past_vols = [np.mean(list(h.volatility.values())) for h in recent_data if h.volatility]
            if past_vols:
                vol_trend = current_vol - past_vols[0]
                features.append(vol_trend)

        # Portfolio features
        if portfolio_state:
            features.extend([
                portfolio_state.total_balance,
                portfolio_state.available_balance,
                portfolio_state.total_positions_value,
                portfolio_state.unrealized_pnl,
                len(portfolio_state.active_positions) if portfolio_state.active_positions else 0,
            ])

        # Time-based features
        current_time = datetime.now()
        features.extend([
            current_time.hour / 24,  # Hour of day (normalized)
            current_time.weekday() / 7,  # Day of week (normalized)
        ])

        return np.array(features)

    def add_training_sample(self, features: np.ndarray, targets: Dict[str, float]):
        """
        Add a new training sample to the learning system.

        Args:
            features: Feature vector
            targets: Dictionary of target values
        """
        # Store features
        self.feature_buffer.append(features)

        # Store targets
        for target_name, value in targets.items():
            if target_name in self.target_buffer:
                self.target_buffer[target_name].append(value)

        self.learning_state.total_samples += 1

        # Periodic model updates
        if (len(self.feature_buffer) >= self.min_samples_for_update and
            (datetime.now() - self.last_model_update).total_seconds() / 60 >= self.update_frequency_minutes):
            self.update_models()

    def update_models(self):
        """Update all models with accumulated training data."""
        try:
            if len(self.feature_buffer) < self.min_samples_for_update:
                return

            logger.info(f"Updating online learning models with {len(self.feature_buffer)} samples")

            # Prepare training data
            X = np.array(self.feature_buffer)
            X_scaled = self.feature_scaler.fit_transform(X)

            # Update price direction prediction model
            if self.target_buffer['price_direction']:
                y_price = np.array(self.target_buffer['price_direction'])
                self.price_prediction_model.fit(X_scaled, y_price)

            # Update volatility prediction model
            if self.target_buffer['volatility']:
                y_vol = np.array(self.target_buffer['volatility'])
                self.volatility_prediction_model.fit(X_scaled, y_vol)

            # Update regime classification model
            if self.target_buffer['regime']:
                y_regime = np.array(self.target_buffer['regime'])
                self.regime_classification_model.fit(X_scaled, y_regime)

                # Calculate accuracy
                y_pred = self.regime_classification_model.predict(X_scaled)
                self.learning_state.accuracy = accuracy_score(y_regime, y_pred)

            # Update feature importance
            if hasattr(self.regime_classification_model, 'feature_importances_'):
                self.learning_state.feature_importance = dict(zip(
                    [f'feature_{i}' for i in range(X.shape[1])],
                    self.regime_classification_model.feature_importances_
                ))

            # Update learning state
            self.learning_state.last_update = datetime.now()
            self.learning_state.model_version += 1

            # Clear buffers to prevent memory issues
            self.feature_buffer = self.feature_buffer[-self.min_samples_for_update//2:]
            for target_name in self.target_buffer:
                self.target_buffer[target_name] = self.target_buffer[target_name][-self.min_samples_for_update//2:]

            # Save updated models
            self.save_models()

            logger.info(f"Models updated successfully. Accuracy: {self.learning_state.accuracy:.3f}")

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    def predict_price_direction(self, features: np.ndarray) -> float:
        """
        Predict future price direction (-1 to 1 scale).

        Returns:
            Prediction value: negative for down, positive for up
        """
        if self.price_prediction_model is None:
            return 0.0

        try:
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            prediction = self.price_prediction_model.predict(features_scaled)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Error predicting price direction: {e}")
            return 0.0

    def predict_volatility(self, features: np.ndarray) -> float:
        """
        Predict future volatility.

        Returns:
            Predicted volatility (0-1 scale)
        """
        if self.volatility_prediction_model is None:
            return 0.02  # Default volatility

        try:
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            prediction = self.volatility_prediction_model.predict(features_scaled)[0]
            return max(0.001, min(0.5, prediction))  # Clamp to reasonable range
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return 0.02

    def predict_market_regime(self, features: np.ndarray) -> int:
        """
        Predict market regime.

        Returns:
            Regime class (0=sideways, 1=bull, -1=bear, 2=high_vol)
        """
        if self.regime_classification_model is None:
            return 0  # Default to sideways

        try:
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            prediction = self.regime_classification_model.predict(features_scaled)[0]
            return int(prediction)
        except Exception as e:
            logger.error(f"Error predicting market regime: {e}")
            return 0

    def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about current model performance and feature importance."""
        return {
            'total_samples': self.learning_state.total_samples,
            'accuracy': self.learning_state.accuracy,
            'model_version': self.learning_state.model_version,
            'last_update': self.learning_state.last_update.isoformat() if self.learning_state.last_update else None,
            'feature_importance': self.learning_state.feature_importance,
            'samples_in_buffer': len(self.feature_buffer),
            'models_trained': {
                'price_prediction': self.price_prediction_model is not None,
                'volatility_prediction': self.volatility_prediction_model is not None,
                'regime_classification': self.regime_classification_model is not None
            }
        }


class AdaptiveStrategyManager:
    """
    Manages strategy adaptation based on online learning predictions.
    Continuously optimizes strategy weights and parameters.
    """

    def __init__(self, learning_system: OnlineLearningSystem):
        self.learning_system = learning_system
        self.strategy_performance: Dict[str, List[float]] = {}
        self.optimal_parameters: Dict[str, Dict[str, float]] = {}

    def adapt_strategy_weights(self, current_features: np.ndarray,
                             strategy_names: List[str],
                             recent_performance: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt strategy weights based on predictions and performance.

        Args:
            current_features: Current market features
            strategy_names: List of available strategies
            recent_performance: Recent performance by strategy

        Returns:
            Updated strategy weights
        """
        # Get predictions from learning system
        price_direction = self.learning_system.predict_price_direction(current_features)
        volatility = self.learning_system.predict_volatility(current_features)
        regime = self.learning_system.predict_market_regime(current_features)

        # Base weights from market conditions
        base_weights = {}

        for strategy in strategy_names:
            weight = 0.0

            if strategy == 'barbell':
                # Barbell works best in trending markets with moderate volatility
                if abs(price_direction) > 0.1 and volatility < 0.1:
                    weight = 0.6
                elif regime in [1, 2]:  # Bull or high vol
                    weight = 0.4

            elif strategy == 'asymmetric':
                # Asymmetric bets work in high volatility with momentum
                if volatility > 0.05 and abs(price_direction) > 0.05:
                    weight = 0.5
                elif regime == 2:  # High volatility
                    weight = 0.3

            elif strategy == 'tail_risk':
                # Tail risk hedging in extreme conditions
                if volatility > 0.08 or abs(price_direction) > 0.15:
                    weight = 0.4

            base_weights[strategy] = weight

        # Adjust weights based on recent performance
        performance_factor = 0.3  # How much performance influences weights

        for strategy in strategy_names:
            perf_score = recent_performance.get(strategy, 0)
            base_weight = base_weights[strategy]

            # Performance adjustment (positive performance increases weight)
            perf_adjustment = perf_score * performance_factor
            adjusted_weight = base_weight * (1 + perf_adjustment)

            # Ensure minimum weight for exploration
            base_weights[strategy] = max(0.05, min(0.8, adjusted_weight))

        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for strategy in base_weights:
                base_weights[strategy] /= total_weight

        return base_weights

    def optimize_strategy_parameters(self, strategy_name: str,
                                   historical_performance: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize strategy parameters based on historical performance.

        Args:
            strategy_name: Name of strategy to optimize
            historical_performance: Historical performance data

        Returns:
            Optimized parameters
        """
        # This would implement parameter optimization using techniques like
        # Bayesian optimization, genetic algorithms, or grid search
        # For now, return default parameters with slight random adjustments

        base_params = {
            'barbell': {
                'safe_allocation': 0.6,
                'risk_allocation': 0.4,
                'momentum_threshold': 0.02,
                'volatility_filter': 0.03
            },
            'asymmetric': {
                'kelly_fraction': 0.5,
                'min_volatility': 0.05,
                'max_position_size': 0.15,
                'momentum_window': 6
            },
            'tail_risk': {
                'hedge_threshold': 0.08,
                'max_hedge_size': 0.3,
                'volatility_lookback': 20
            }
        }

        params = base_params.get(strategy_name, {})

        # Add small random variations for exploration
        for param_name in params:
            params[param_name] *= (0.9 + np.random.random() * 0.2)  # ±10% variation

        self.optimal_parameters[strategy_name] = params
        return params


def create_training_targets(market_state, future_market_states: List) -> Dict[str, float]:
    """
    Create training targets from future market observations.

    Args:
        market_state: Current market state
        future_market_states: Future market states (1, 4, 24 hours ahead)

    Returns:
        Dictionary of target values
    """
    targets = {}

    if not future_market_states:
        return targets

    # Price direction target (1 hour ahead)
    if len(future_market_states) >= 1:
        future_state = future_market_states[0]
        current_avg_price = np.mean(list(market_state.prices.values())) if market_state.prices else 0
        future_avg_price = np.mean(list(future_state.prices.values())) if future_state.prices else 0

        if current_avg_price > 0:
            price_change = (future_avg_price - current_avg_price) / current_avg_price
            targets['price_direction'] = price_change  # Can be negative

    # Volatility target (4 hours ahead)
    if len(future_market_states) >= 4:
        future_states = future_market_states[:4]
        price_changes = []

        for i in range(1, len(future_states)):
            prev_prices = list(future_states[i-1].prices.values()) if future_states[i-1].prices else []
            curr_prices = list(future_states[i].prices.values()) if future_states[i].prices else []

            if prev_prices and curr_prices:
                avg_prev = np.mean(prev_prices)
                avg_curr = np.mean(curr_prices)
                if avg_prev > 0:
                    price_changes.append((avg_curr - avg_prev) / avg_prev)

        if price_changes:
            targets['volatility'] = np.std(price_changes)
        else:
            targets['volatility'] = 0.02  # Default

    # Regime target (24 hours ahead)
    if len(future_market_states) >= 24:
        future_state = future_market_states[23]

        # Classify future regime based on price movement and volatility
        future_prices = list(future_state.prices.values())
        current_prices = list(market_state.prices.values())

        if future_prices and current_prices:
            price_change = (np.mean(future_prices) - np.mean(current_prices)) / np.mean(current_prices)

            if price_change > 0.05:
                targets['regime'] = 1  # Bull
            elif price_change < -0.05:
                targets['regime'] = -1  # Bear
            else:
                targets['regime'] = 0  # Sideways
        else:
            targets['regime'] = 0

    return targets


# Example usage and testing
if __name__ == "__main__":
    # Create learning system
    learning_system = OnlineLearningSystem()

    # Example feature extraction
    from mcp_trader.ai.adaptive_trading_agent import MarketState, PortfolioState

    # Mock market state
    market_state = MarketState(
        timestamp=datetime.now(),
        prices={'BTCUSDT': 50000, 'ETHUSDT': 3000},
        volumes={'BTCUSDT': 1000000, 'ETHUSDT': 500000},
        volatility={'BTCUSDT': 0.03, 'ETHUSDT': 0.04},
        momentum={'BTCUSDT': 0.02, 'ETHUSDT': -0.01}
    )

    portfolio_state = PortfolioState(
        timestamp=datetime.now(),
        total_balance=10000,
        available_balance=8000,
        total_positions_value=2000
    )

    # Extract features
    features = learning_system.extract_features(market_state, portfolio_state, [])

    print(f"Extracted {len(features)} features")
    print(f"Feature vector: {features[:10]}...")

    # Mock training targets
    targets = {
        'price_direction': 0.02,  # 2% up
        'volatility': 0.025,
        'regime': 1  # Bull market
    }

    # Add training sample
    learning_system.add_training_sample(features, targets)

    # Get insights
    insights = learning_system.get_model_insights()
    print(f"Learning system insights: {insights}")

    print("Online learning system initialized and ready for continuous adaptation!")


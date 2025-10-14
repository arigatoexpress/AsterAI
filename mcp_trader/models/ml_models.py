"""
Machine learning models for trading.
Includes various ML approaches for price prediction and signal generation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
from .base import BaseMLModel, ModelPrediction, TradingSignal


class PricePredictionModel(BaseMLModel):
    """Base class for price prediction models."""
    
    def __init__(self, name: str, model_class, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model_class(**kwargs.get('model_params', {}))
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.lookback_window = kwargs.get('lookback_window', 24)
        self.prediction_horizon = kwargs.get('prediction_horizon', 1)  # 1 hour ahead
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features for ML model."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(24).std()
        df['atr'] = self._calculate_atr(df, 14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df, 20)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df, 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['price_volume'] = df['close'] * df['volume']
        
        # Lagged features
        for lag in range(1, self.lookback_window + 1):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Drop NaN values
        df = df.dropna()
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        self.feature_columns = feature_cols
        
        return df[feature_cols]
    
    def prepare_targets(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable (future price change)."""
        if self.target_column == 'returns':
            return data['close'].pct_change().shift(-self.prediction_horizon)
        elif self.target_column == 'price':
            return data['close'].shift(-self.prediction_horizon)
        else:
            return data[self.target_column]
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'PricePredictionModel':
        """Train the model with scaling."""
        features = self.prepare_features(data)
        targets = self.prepare_targets(data)
        
        # Remove NaN values
        valid_idx = ~(features.isnull().any(axis=1) | targets.isnull())
        features = features[valid_idx]
        targets = targets[valid_idx]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, targets)
        self.is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions with scaling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Convert to ModelPrediction objects
        results = []
        for idx, (_, row) in enumerate(data.iterrows()):
            if idx < len(predictions):
                results.append(ModelPrediction(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=predictions[idx],
                    confidence=0.7,  # Default confidence
                    features=features.iloc[idx].to_dict(),
                    metadata={'model_type': 'ml', 'model_name': self.name}
                ))
        
        return results
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from predictions."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            # Convert prediction to signal
            if pred.prediction > 0.01:  # 1% threshold
                signal = 1  # BUY
                confidence = min(abs(pred.prediction) * 10, 1.0)
            elif pred.prediction < -0.01:  # -1% threshold
                signal = -1  # SELL
                confidence = min(abs(pred.prediction) * 10, 1.0)
            else:
                signal = 0  # HOLD
                confidence = 0.3
            
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=signal,
                confidence=confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata={'prediction': pred.prediction, 'model': self.name}
            ))
        
        return signals
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int) -> tuple:
        """Calculate Bollinger Bands."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, sma, lower
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, df: pd.DataFrame) -> tuple:
        """Calculate MACD."""
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        return macd, signal, histogram


class RandomForestModel(PricePredictionModel):
    """Random Forest for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 10),
            'min_samples_split': kwargs.get('min_samples_split', 5),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 2),
            'random_state': 42
        }
        super().__init__("RandomForest", RandomForestRegressor, **kwargs)
        self.model = RandomForestRegressor(**model_params)


class XGBoostModel(PricePredictionModel):
    """XGBoost for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42
        }
        super().__init__("XGBoost", xgb.XGBRegressor, **kwargs)
        self.model = xgb.XGBRegressor(**model_params)


class LightGBMModel(PricePredictionModel):
    """LightGBM for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42
        }
        super().__init__("LightGBM", lgb.LGBMRegressor, **kwargs)
        self.model = lgb.LGBMRegressor(**model_params)


class SVMModel(PricePredictionModel):
    """Support Vector Machine for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'kernel': kwargs.get('kernel', 'rbf'),
            'C': kwargs.get('C', 1.0),
            'gamma': kwargs.get('gamma', 'scale'),
            'epsilon': kwargs.get('epsilon', 0.1)
        }
        super().__init__("SVM", SVR, **kwargs)
        self.model = SVR(**model_params)


class RidgeModel(PricePredictionModel):
    """Ridge Regression for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'alpha': kwargs.get('alpha', 1.0),
            'random_state': 42
        }
        super().__init__("Ridge", Ridge, **kwargs)
        self.model = Ridge(**model_params)


class LassoModel(PricePredictionModel):
    """Lasso Regression for price prediction."""
    
    def __init__(self, **kwargs):
        model_params = {
            'alpha': kwargs.get('alpha', 0.1),
            'random_state': 42
        }
        super().__init__("Lasso", Lasso, **kwargs)
        self.model = Lasso(**model_params)

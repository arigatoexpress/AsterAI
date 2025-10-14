"""
Base classes for trading models and strategies.
Standardized interface for backtesting and live trading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class ModelMetadata:
    """Metadata for a trading model."""
    name: str
    version: str
    model_type: str  # 'grid', 'ml', 'rule_based', 'ensemble'
    description: str
    features_used: List[str]
    hyperparameters: Dict[str, Any]
    created_at: datetime
    performance: Optional[Dict[str, float]] = None


@dataclass
class TradingSignal:
    """Standardized trading signal."""
    timestamp: datetime
    symbol: str
    signal: int  # -1 (SELL), 0 (HOLD), 1 (BUY)
    confidence: float  # 0.0 to 1.0
    price: float
    metadata: Dict[str, Any]


@dataclass
class ModelPrediction:
    """Model prediction output."""
    timestamp: datetime
    symbol: str
    prediction: Union[float, int]  # Price prediction or signal
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class BaseTradingModel(ABC):
    """Base class for all trading models."""
    
    def __init__(self, name: str, model_type: str, **kwargs):
        self.name = name
        self.model_type = model_type
        self.is_fitted = False
        self.metadata = ModelMetadata(
            name=name,
            version="1.0.0",
            model_type=model_type,
            description="",
            features_used=[],
            hyperparameters=kwargs,
            created_at=datetime.now()
        )
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseTradingModel':
        """Train/fit the model on historical data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions for given data."""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from predictions."""
        pass
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self.metadata
    
    def set_metadata(self, **kwargs):
        """Update model metadata."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)


class BaseGridStrategy(BaseTradingModel):
    """Base class for grid trading strategies."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "grid", **kwargs)
        self.grid_levels: List[float] = []
        self.position_sizes: List[float] = []
        self.grid_spacing: float = kwargs.get('grid_spacing', 0.01)  # 1%
        self.max_levels: int = kwargs.get('max_levels', 10)
        self.base_price: Optional[float] = None
    
    @abstractmethod
    def calculate_grid_levels(self, current_price: float) -> List[float]:
        """Calculate grid levels based on current price."""
        pass
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate grid trading signals."""
        signals = []
        
        for idx, row in data.iterrows():
            current_price = row['close']
            
            if self.base_price is None:
                self.base_price = current_price
                self.grid_levels = self.calculate_grid_levels(current_price)
            
            # Check if price hit any grid level
            for level in self.grid_levels:
                if abs(current_price - level) / level < 0.001:  # Within 0.1%
                    signal = 1 if current_price < self.base_price else -1
                    confidence = 0.8  # High confidence for grid levels
                    
                    signals.append(TradingSignal(
                        timestamp=row['timestamp'],
                        symbol=row.get('symbol', 'UNKNOWN'),
                        signal=signal,
                        confidence=confidence,
                        price=current_price,
                        metadata={'grid_level': level, 'base_price': self.base_price}
                    ))
        
        return signals


class BaseMLModel(BaseTradingModel):
    """Base class for machine learning models."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "ml", **kwargs)
        self.model = None
        self.feature_columns: List[str] = []
        self.target_column: str = kwargs.get('target_column', 'returns')
        self.lookback_window: int = kwargs.get('lookback_window', 24)  # 24 hours
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        # This should be implemented by subclasses
        return data
    
    def prepare_targets(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable for training."""
        if self.target_column == 'returns':
            return data['close'].pct_change().shift(-1)  # Next period return
        elif self.target_column == 'direction':
            returns = data['close'].pct_change().shift(-1)
            return (returns > 0).astype(int)  # Binary classification
        else:
            return data[self.target_column]
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseMLModel':
        """Train the ML model."""
        features = self.prepare_features(data)
        targets = self.prepare_targets(data)
        
        # Remove NaN values
        valid_idx = ~(features.isnull().any(axis=1) | targets.isnull())
        features = features[valid_idx]
        targets = targets[valid_idx]
        
        self.model.fit(features, targets)
        self.is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate ML predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        features = self.prepare_features(data)
        predictions = self.model.predict(features)
        
        # Convert to ModelPrediction objects
        results = []
        for idx, (_, row) in enumerate(data.iterrows()):
            results.append(ModelPrediction(
                timestamp=row['timestamp'],
                symbol=row.get('symbol', 'UNKNOWN'),
                prediction=predictions[idx],
                confidence=0.7,  # Default confidence
                features=features.iloc[idx].to_dict(),
                metadata={'model_type': 'ml'}
            ))
        
        return results


class BaseEnsembleModel(BaseTradingModel):
    """Base class for ensemble models."""
    
    def __init__(self, name: str, base_models: List[BaseTradingModel], **kwargs):
        super().__init__(name, "ensemble", **kwargs)
        self.base_models = base_models
        self.weights: Optional[List[float]] = None
        self.ensemble_method: str = kwargs.get('ensemble_method', 'weighted_average')
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseEnsembleModel':
        """Fit all base models."""
        for model in self.base_models:
            model.fit(data, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from all base models
        all_predictions = []
        for model in self.base_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        # Combine predictions
        ensemble_predictions = self._combine_predictions(all_predictions)
        return ensemble_predictions
    
    def _combine_predictions(self, predictions_list: List[List[ModelPrediction]]) -> List[ModelPrediction]:
        """Combine predictions from multiple models."""
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_ensemble(predictions_list)
        elif self.ensemble_method == 'majority_vote':
            return self._majority_vote_ensemble(predictions_list)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_average_ensemble(self, predictions_list: List[List[ModelPrediction]]) -> List[ModelPrediction]:
        """Weighted average ensemble."""
        if self.weights is None:
            self.weights = [1.0 / len(predictions_list)] * len(predictions_list)
        
        # Get first model's predictions as template
        template = predictions_list[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            weighted_pred = sum(
                w * pred_list[i].prediction 
                for w, pred_list in zip(self.weights, predictions_list)
            )
            weighted_confidence = sum(
                w * pred_list[i].confidence 
                for w, pred_list in zip(self.weights, predictions_list)
            )
            
            ensemble_predictions.append(ModelPrediction(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                prediction=weighted_pred,
                confidence=weighted_confidence,
                features=pred.features,
                metadata={'ensemble_method': 'weighted_average', 'weights': self.weights}
            ))
        
        return ensemble_predictions
    
    def _majority_vote_ensemble(self, predictions_list: List[List[ModelPrediction]]) -> List[ModelPrediction]:
        """Majority vote ensemble."""
        template = predictions_list[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            # Convert predictions to signals for voting
            signals = []
            for pred_list in predictions_list:
                signal = 1 if pred_list[i].prediction > 0 else -1
                signals.append(signal)
            
            # Majority vote
            majority_signal = 1 if sum(signals) > 0 else -1
            confidence = abs(sum(signals)) / len(signals)  # Agreement level
            
            ensemble_predictions.append(ModelPrediction(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                prediction=majority_signal,
                confidence=confidence,
                features=pred.features,
                metadata={'ensemble_method': 'majority_vote'}
            ))
        
        return ensemble_predictions

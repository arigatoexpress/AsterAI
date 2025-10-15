"""
Ensemble strategies combining multiple models.
Includes stacking, voting, and adaptive weighting methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from ..models.base import BaseEnsembleModel, ModelPrediction, TradingSignal


class StackingEnsemble(BaseEnsembleModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, name: str, base_models: List, meta_learner=None, **kwargs):
        super().__init__(name, base_models, **kwargs)
        self.meta_learner = meta_learner or LogisticRegression(random_state=42)
        self.meta_features = None
        self.is_meta_fitted = False
    
    def _prepare_meta_features(self, predictions_list: List[List[ModelPrediction]]) -> np.ndarray:
        """Prepare features for meta-learner from base model predictions."""
        # Get first model's predictions as template
        template = predictions_list[0]
        meta_features = []
        
        for i, pred in enumerate(template):
            # Extract prediction and confidence from each base model
            features = []
            for pred_list in predictions_list:
                if i < len(pred_list):
                    features.extend([
                        pred_list[i].prediction,
                        pred_list[i].confidence
                    ])
                else:
                    features.extend([0.0, 0.0])  # Padding for missing predictions
            
            meta_features.append(features)
        
        return np.array(meta_features)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'StackingEnsemble':
        """Fit base models and meta-learner."""
        # Fit base models
        super().fit(data, **kwargs)
        
        # Prepare meta-features for training
        all_predictions = []
        for model in self.base_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        # Create training targets (next period returns > 0)
        returns = data['close'].pct_change().shift(-1)
        targets = (returns > 0).astype(int)
        
        # Align predictions with targets
        aligned_predictions = []
        for pred_list in all_predictions:
            aligned_preds = []
            for pred in pred_list:
                # Find matching timestamp in data
                matching_idx = data[data['timestamp'] == pred.timestamp].index
                if len(matching_idx) > 0:
                    aligned_preds.append(pred)
            aligned_predictions.append(aligned_preds)
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(aligned_predictions)
        
        # Align targets
        aligned_targets = []
        for pred in aligned_predictions[0]:
            matching_idx = data[data['timestamp'] == pred.timestamp].index
            if len(matching_idx) > 0:
                target = targets.iloc[matching_idx[0]]
                aligned_targets.append(target)
        
        # Remove NaN values
        valid_idx = ~(np.isnan(meta_features).any(axis=1) | np.isnan(aligned_targets))
        meta_features = meta_features[valid_idx]
        aligned_targets = np.array(aligned_targets)[valid_idx]
        
        # Fit meta-learner
        self.meta_learner.fit(meta_features, aligned_targets)
        self.is_meta_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate stacking predictions."""
        if not self.is_meta_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from base models
        all_predictions = []
        for model in self.base_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        # Prepare meta-features
        meta_features = self._prepare_meta_features(all_predictions)
        
        # Get meta-learner predictions
        meta_predictions = self.meta_learner.predict_proba(meta_features)
        
        # Convert to ModelPrediction objects
        template = all_predictions[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            if i < len(meta_predictions):
                # Convert probability to signal
                prob_positive = meta_predictions[i][1] if len(meta_predictions[i]) > 1 else meta_predictions[i][0]
                signal = 1 if prob_positive > 0.5 else -1
                confidence = abs(prob_positive - 0.5) * 2  # Scale to 0-1
                
                ensemble_predictions.append(ModelPrediction(
                    timestamp=pred.timestamp,
                    symbol=pred.symbol,
                    prediction=signal,
                    confidence=confidence,
                    features=pred.features,
                    metadata={
                        'ensemble_method': 'stacking',
                        'meta_learner': type(self.meta_learner).__name__,
                        'base_models': [model.name for model in self.base_models],
                        'meta_probability': prob_positive
                    }
                ))
        
        return ensemble_predictions


class AdaptiveEnsemble(BaseEnsembleModel):
    """Adaptive ensemble that adjusts weights based on recent performance."""
    
    def __init__(self, name: str, base_models: List, **kwargs):
        super().__init__(name, base_models, **kwargs)
        self.performance_window = kwargs.get('performance_window', 24)  # 24 hours
        self.adaptation_rate = kwargs.get('adaptation_rate', 0.1)  # 10% adaptation rate
        self.min_weight = kwargs.get('min_weight', 0.05)  # 5% minimum weight
        self.max_weight = kwargs.get('max_weight', 0.5)  # 50% maximum weight
        self.performance_history = {model.name: [] for model in base_models}
        self.current_weights = [1.0 / len(base_models)] * len(base_models)
    
    def _calculate_performance(self, model: BaseEnsembleModel, data: pd.DataFrame) -> float:
        """Calculate recent performance of a model."""
        # Get recent predictions
        predictions = model.predict(data.tail(self.performance_window))
        
        if not predictions:
            return 0.0
        
        # Calculate accuracy based on next period returns
        correct_predictions = 0
        total_predictions = 0
        
        for pred in predictions:
            # Find matching data point
            matching_data = data[data['timestamp'] == pred.timestamp]
            if len(matching_data) > 0:
                # Get next period return
                next_idx = matching_data.index[0] + 1
                if next_idx < len(data):
                    next_return = data.iloc[next_idx]['close'] / data.iloc[next_idx - 1]['close'] - 1
                    
                    # Check if prediction was correct
                    if (pred.prediction > 0 and next_return > 0) or (pred.prediction < 0 and next_return < 0):
                        correct_predictions += 1
                    total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _update_weights(self, data: pd.DataFrame):
        """Update model weights based on recent performance."""
        performances = []
        
        for i, model in enumerate(self.base_models):
            performance = self._calculate_performance(model, data)
            performances.append(performance)
            self.performance_history[model.name].append(performance)
        
        # Normalize performances
        if sum(performances) > 0:
            normalized_performances = np.array(performances) / sum(performances)
        else:
            normalized_performances = np.array([1.0 / len(performances)] * len(performances))
        
        # Update weights with adaptation rate
        for i in range(len(self.current_weights)):
            new_weight = (1 - self.adaptation_rate) * self.current_weights[i] + \
                        self.adaptation_rate * normalized_performances[i]
            
            # Apply min/max constraints
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            self.current_weights[i] = new_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(self.current_weights)
        self.current_weights = [w / total_weight for w in self.current_weights]
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate adaptive ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Update weights based on recent performance
        self._update_weights(data)
        
        # Get predictions from base models
        all_predictions = []
        for model in self.base_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        # Weighted combination
        template = all_predictions[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            weighted_pred = sum(
                w * pred_list[i].prediction 
                for w, pred_list in zip(self.current_weights, all_predictions)
            )
            weighted_confidence = sum(
                w * pred_list[i].confidence 
                for w, pred_list in zip(self.current_weights, all_predictions)
            )
            
            # Convert to signal
            signal = 1 if weighted_pred > 0.5 else -1 if weighted_pred < -0.5 else 0
            
            ensemble_predictions.append(ModelPrediction(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                prediction=signal,
                confidence=weighted_confidence,
                features=pred.features,
                metadata={
                    'ensemble_method': 'adaptive',
                    'weights': self.current_weights,
                    'base_models': [model.name for model in self.base_models],
                    'weighted_prediction': weighted_pred
                }
            ))
        
        return ensemble_predictions


class RegimeBasedEnsemble(BaseEnsembleModel):
    """Ensemble that selects different models based on market regime."""
    
    def __init__(self, name: str, base_models: List, regime_models: Dict[str, List], **kwargs):
        super().__init__(name, base_models, **kwargs)
        self.regime_models = regime_models  # {'trending': [model1, model2], 'ranging': [model3, model4]}
        self.regime_classifier = None
        self.regime_features = ['volatility', 'trend_strength', 'volume_ratio']
        self.current_regime = 'normal'
    
    def _detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime."""
        if len(data) < 24:
            return 'normal'
        
        recent_data = data.tail(24)
        
        # Calculate regime features
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(24)  # Daily volatility
        
        # Trend strength
        sma_short = recent_data['close'].rolling(8).mean().iloc[-1]
        sma_long = recent_data['close'].rolling(24).mean().iloc[-1]
        trend_strength = abs(sma_short - sma_long) / sma_long
        
        # Volume ratio (if available)
        if 'volume' in recent_data.columns:
            volume_ratio = recent_data['volume'].tail(8).mean() / recent_data['volume'].tail(24).mean()
        else:
            volume_ratio = 1.0
        
        # Simple regime classification
        if volatility > 0.05 and trend_strength > 0.02:
            return 'trending'
        elif volatility < 0.02 and trend_strength < 0.01:
            return 'ranging'
        else:
            return 'normal'
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'RegimeBasedEnsemble':
        """Fit all models and regime classifier."""
        # Fit base models
        super().fit(data, **kwargs)
        
        # Fit regime-specific models
        for regime, models in self.regime_models.items():
            for model in models:
                model.fit(data, **kwargs)
        
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate regime-based predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Detect current regime
        current_regime = self._detect_regime(data)
        self.current_regime = current_regime
        
        # Select appropriate models for current regime
        if current_regime in self.regime_models:
            active_models = self.regime_models[current_regime]
        else:
            active_models = self.base_models
        
        # Get predictions from active models
        all_predictions = []
        for model in active_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        if not all_predictions:
            return []
        
        # Combine predictions from active models
        template = all_predictions[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            # Simple average of active models
            active_predictions = [pred_list[i] for pred_list in all_predictions if i < len(pred_list)]
            
            if active_predictions:
                avg_prediction = sum(p.prediction for p in active_predictions) / len(active_predictions)
                avg_confidence = sum(p.confidence for p in active_predictions) / len(active_predictions)
                
                signal = 1 if avg_prediction > 0.5 else -1 if avg_prediction < -0.5 else 0
                
                ensemble_predictions.append(ModelPrediction(
                    timestamp=pred.timestamp,
                    symbol=pred.symbol,
                    prediction=signal,
                    confidence=avg_confidence,
                    features=pred.features,
                    metadata={
                        'ensemble_method': 'regime_based',
                        'current_regime': current_regime,
                        'active_models': [model.name for model in active_models],
                        'num_active_models': len(active_models)
                    }
                ))
        
        return ensemble_predictions


class ConfidenceWeightedEnsemble(BaseEnsembleModel):
    """Ensemble that weights models based on their confidence scores."""
    
    def __init__(self, name: str, base_models: List, **kwargs):
        super().__init__(name, base_models, **kwargs)
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.6)
        self.min_models = kwargs.get('min_models', 2)  # Minimum models required for prediction
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate confidence-weighted predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from base models
        all_predictions = []
        for model in self.base_models:
            predictions = model.predict(data)
            all_predictions.append(predictions)
        
        # Combine predictions with confidence weighting
        template = all_predictions[0]
        ensemble_predictions = []
        
        for i, pred in enumerate(template):
            # Get predictions for this timestamp
            timestamp_predictions = []
            for pred_list in all_predictions:
                if i < len(pred_list):
                    timestamp_predictions.append(pred_list[i])
            
            # Filter by confidence threshold
            high_confidence_preds = [
                p for p in timestamp_predictions 
                if p.confidence >= self.confidence_threshold
            ]
            
            # Use high confidence predictions if enough available
            if len(high_confidence_preds) >= self.min_models:
                active_predictions = high_confidence_preds
            else:
                active_predictions = timestamp_predictions
            
            if len(active_predictions) >= self.min_models:
                # Calculate confidence-weighted average
                total_confidence = sum(p.confidence for p in active_predictions)
                
                if total_confidence > 0:
                    weighted_pred = sum(
                        p.prediction * p.confidence for p in active_predictions
                    ) / total_confidence
                    
                    weighted_confidence = sum(
                        p.confidence for p in active_predictions
                    ) / len(active_predictions)
                    
                    signal = 1 if weighted_pred > 0.5 else -1 if weighted_pred < -0.5 else 0
                    
                    ensemble_predictions.append(ModelPrediction(
                        timestamp=pred.timestamp,
                        symbol=pred.symbol,
                        prediction=signal,
                        confidence=weighted_confidence,
                        features=pred.features,
                        metadata={
                            'ensemble_method': 'confidence_weighted',
                            'confidence_threshold': self.confidence_threshold,
                            'active_models': len(active_predictions),
                            'total_models': len(timestamp_predictions),
                            'weighted_prediction': weighted_pred
                        }
                    ))
        
        return ensemble_predictions


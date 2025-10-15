"""
DMark Strategy - Trading strategy based on the proprietary DMark indicator.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from mcp_trader.models.base import BaseTradingModel, ModelPrediction, TradingSignal
from mcp_trader.indicators.dmark import DMarkIndicator, DMarkConfig

logger = logging.getLogger(__name__)


class DMarkStrategy(BaseTradingModel):
    """
    DMark Strategy - Uses the proprietary DMark indicator for trading decisions.
    
    This strategy combines the DMark indicator with risk management rules
    to generate high-quality trading signals.
    """
    
    def __init__(self, **kwargs):
        super().__init__("DMarkStrategy", "rule_based", **kwargs)
        
        # DMark indicator configuration
        dmark_config = DMarkConfig(
            lookback_period=kwargs.get('lookback_period', 20),
            momentum_period=kwargs.get('momentum_period', 10),
            volatility_period=kwargs.get('volatility_period', 14),
            volume_period=kwargs.get('volume_period', 10),
            strong_signal_threshold=kwargs.get('strong_signal_threshold', 0.7),
            moderate_signal_threshold=kwargs.get('moderate_signal_threshold', 0.4),
            weak_signal_threshold=kwargs.get('weak_signal_threshold', 0.2)
        )
        
        self.dmark_indicator = DMarkIndicator(dmark_config)
        
        # Strategy parameters
        self.min_confidence = kwargs.get('min_confidence', 0.6)
        self.position_size_multiplier = kwargs.get('position_size_multiplier', 1.0)
        self.max_position_size = kwargs.get('max_position_size', 0.25)  # 25% of capital
        self.stop_loss_threshold = kwargs.get('stop_loss_threshold', 0.02)  # 2% stop loss
        self.take_profit_threshold = kwargs.get('take_profit_threshold', 0.04)  # 4% take profit
        
        # Risk management
        self.max_daily_trades = kwargs.get('max_daily_trades', 10)
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Position tracking
        self.current_position = 0.0
        self.entry_price = 0.0
        self.position_side = 0  # 1 for long, -1 for short, 0 for no position
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'DMarkStrategy':
        """Fit the strategy (no training needed for rule-based strategy)."""
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions using DMark indicator."""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before making predictions")
        
        # Calculate DMark indicator
        dmark_results = self.dmark_indicator.calculate(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            open_price=data.get('open')
        )
        
        # Generate predictions
        predictions = []
        for idx, row in data.iterrows():
            # Handle timestamp column - use index if no timestamp column
            if 'timestamp' in row.index:
                timestamp = row['timestamp']
            else:
                timestamp = idx

            dmark_signal = dmark_results['dmark_signal'].iloc[idx]
            confidence = dmark_results['confidence'].iloc[idx]
            regime = dmark_results['regime'].iloc[idx]
            
            # Apply risk management filters
            filtered_signal = self._apply_risk_filters(
                dmark_signal, confidence, timestamp, row['close']
            )
            
            # Calculate position size based on confidence and signal strength
            position_size = self._calculate_position_size(filtered_signal, confidence)
            
            predictions.append(ModelPrediction(
                timestamp=timestamp,
                symbol=row.get('symbol', 'UNKNOWN'),
                prediction=filtered_signal,
                confidence=confidence,
                features={
                    'dmark_raw': dmark_results['dmark_raw'].iloc[idx],
                    'dmark_filtered': dmark_results['dmark_filtered'].iloc[idx],
                    'regime': regime,
                    'momentum': dmark_results['momentum_component'].iloc[idx],
                    'volatility': dmark_results['volatility_component'].iloc[idx],
                    'volume': dmark_results['volume_component'].iloc[idx],
                    'microstructure': dmark_results['microstructure_component'].iloc[idx],
                    'trend': dmark_results['trend_component'].iloc[idx],
                    'position_size': position_size
                },
                metadata={
                    'strategy': 'dmark',
                    'signal_strength': self.dmark_indicator.get_signal_strength(dmark_signal),
                    'signal_direction': self.dmark_indicator.get_signal_direction(dmark_signal),
                    'regime_description': self.dmark_indicator.get_regime_description(regime),
                    'current_position': self.current_position,
                    'position_side': self.position_side
                }
            ))
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals with position management."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            # Update position tracking
            self._update_position_tracking(pred.prediction, pred.features['position_size'])
            
            # Generate trading signal
            if pred.prediction != 0 and pred.confidence >= self.min_confidence:
                signal = self._create_trading_signal(pred)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _apply_risk_filters(self, signal: float, confidence: float,
                           timestamp, price: float) -> float:
        """Apply risk management filters to the signal."""

        # Check daily trade limit - handle both datetime and int types
        if isinstance(timestamp, datetime):
            current_date = timestamp.date()
        else:
            # If timestamp is an int (index), skip date-based filtering
            current_date = None

        if current_date is not None and self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trade_count >= self.max_daily_trades:
            return 0.0
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return 0.0
        
        # Check for stop loss
        if self.current_position != 0:
            price_change = (price - self.entry_price) / self.entry_price
            
            if self.position_side == 1:  # Long position
                if price_change <= -self.stop_loss_threshold:
                    return -1.0  # Force close long position
                elif price_change >= self.take_profit_threshold:
                    return -0.5  # Partial profit taking
            elif self.position_side == -1:  # Short position
                if price_change >= self.stop_loss_threshold:
                    return 1.0  # Force close short position
                elif price_change <= -self.take_profit_threshold:
                    return 0.5  # Partial profit taking
        
        # Check position size limits
        if abs(signal) > 0 and self.current_position >= self.max_position_size:
            return 0.0
        
        return signal
    
    def _calculate_position_size(self, signal: float, confidence: float) -> float:
        """Calculate position size based on signal strength and confidence."""
        if signal == 0:
            return 0.0
        
        # Base position size from signal strength
        base_size = abs(signal) * self.position_size_multiplier
        
        # Adjust by confidence
        confidence_adjusted_size = base_size * confidence
        
        # Apply maximum position size limit
        final_size = min(confidence_adjusted_size, self.max_position_size)
        
        return final_size
    
    def _update_position_tracking(self, signal: float, position_size: float):
        """Update internal position tracking."""
        if signal > 0:  # Buy signal
            self.current_position = min(self.current_position + position_size, self.max_position_size)
            self.position_side = 1
        elif signal < 0:  # Sell signal
            self.current_position = max(self.current_position - position_size, -self.max_position_size)
            self.position_side = -1
        
        # Update entry price (simplified)
        if self.current_position != 0 and self.entry_price == 0:
            # This would need actual price data in a real implementation
            pass
    
    def _create_trading_signal(self, prediction: ModelPrediction) -> Optional[TradingSignal]:
        """Create a trading signal from prediction."""
        if prediction.prediction == 0:
            return None
        
        # Determine signal direction
        if prediction.prediction > 0:
            signal_direction = 1  # BUY
        else:
            signal_direction = -1  # SELL
        
        # Calculate signal strength
        signal_strength = abs(prediction.prediction)
        
        # Create trading signal
        return TradingSignal(
            timestamp=prediction.timestamp,
            symbol=prediction.symbol,
            signal=signal_direction,
            confidence=prediction.confidence,
            price=0.0,  # Would need actual price in real implementation
            metadata={
                'strategy': 'dmark',
                'signal_strength': signal_strength,
                'position_size': prediction.features['position_size'],
                'regime': prediction.features['regime'],
                'dmark_components': {
                    'momentum': prediction.features['momentum'],
                    'volatility': prediction.features['volatility'],
                    'volume': prediction.features['volume'],
                    'microstructure': prediction.features['microstructure'],
                    'trend': prediction.features['trend']
                }
            }
        )
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status."""
        return {
            'current_position': self.current_position,
            'position_side': self.position_side,
            'daily_trade_count': self.daily_trade_count,
            'max_daily_trades': self.max_daily_trades,
            'min_confidence': self.min_confidence,
            'max_position_size': self.max_position_size,
            'stop_loss_threshold': self.stop_loss_threshold,
            'take_profit_threshold': self.take_profit_threshold
        }
    
    def reset_position_tracking(self):
        """Reset position tracking (useful for backtesting)."""
        self.current_position = 0.0
        self.entry_price = 0.0
        self.position_side = 0
        self.daily_trade_count = 0
        self.last_trade_date = None


class DMarkEnsembleStrategy(BaseTradingModel):
    """
    DMark Ensemble Strategy - Combines multiple DMark configurations.
    """
    
    def __init__(self, **kwargs):
        super().__init__("DMarkEnsemble", "ensemble", **kwargs)
        
        # Create multiple DMark strategies with different configurations
        self.strategies = []
        
        # Conservative strategy
        conservative_config = DMarkConfig(
            lookback_period=30,
            strong_signal_threshold=0.8,
            moderate_signal_threshold=0.5,
            momentum_weight=0.4,
            volatility_weight=0.3,
            volume_weight=0.3
        )
        self.strategies.append(DMarkStrategy(
            dmark_config=conservative_config,
            min_confidence=0.7,
            max_position_size=0.15
        ))
        
        # Aggressive strategy
        aggressive_config = DMarkConfig(
            lookback_period=10,
            strong_signal_threshold=0.5,
            moderate_signal_threshold=0.3,
            momentum_weight=0.5,
            microstructure_weight=0.3,
            trend_weight=0.2
        )
        self.strategies.append(DMarkStrategy(
            dmark_config=aggressive_config,
            min_confidence=0.5,
            max_position_size=0.3
        ))
        
        # Balanced strategy
        balanced_config = DMarkConfig(
            lookback_period=20,
            strong_signal_threshold=0.6,
            moderate_signal_threshold=0.4,
            momentum_weight=0.3,
            volatility_weight=0.25,
            volume_weight=0.2,
            microstructure_weight=0.15,
            trend_weight=0.1
        )
        self.strategies.append(DMarkStrategy(
            dmark_config=balanced_config,
            min_confidence=0.6,
            max_position_size=0.2
        ))
        
        # Ensemble weights
        self.strategy_weights = [0.3, 0.2, 0.5]  # Conservative, Aggressive, Balanced
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'DMarkEnsembleStrategy':
        """Fit all sub-strategies."""
        for strategy in self.strategies:
            strategy.fit(data, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get predictions from all strategies
        all_predictions = []
        for strategy in self.strategies:
            predictions = strategy.predict(data)
            all_predictions.append(predictions)
        
        # Combine predictions
        ensemble_predictions = []
        for i in range(len(data)):
            # Get predictions for this timestamp
            timestamp_predictions = []
            for pred_list in all_predictions:
                if i < len(pred_list):
                    timestamp_predictions.append(pred_list[i])
            
            if timestamp_predictions:
                # Weighted average of predictions
                weighted_pred = sum(
                    pred.prediction * weight 
                    for pred, weight in zip(timestamp_predictions, self.strategy_weights)
                )
                
                weighted_confidence = sum(
                    pred.confidence * weight 
                    for pred, weight in zip(timestamp_predictions, self.strategy_weights)
                )
                
                # Combine features
                combined_features = {}
                for pred in timestamp_predictions:
                    combined_features.update(pred.features)
                
                # Combine metadata
                combined_metadata = {
                    'strategy': 'dmark_ensemble',
                    'sub_strategies': len(timestamp_predictions),
                    'weights': self.strategy_weights
                }
                
                ensemble_predictions.append(ModelPrediction(
                    timestamp=timestamp_predictions[0].timestamp,
                    symbol=timestamp_predictions[0].symbol,
                    prediction=weighted_pred,
                    confidence=weighted_confidence,
                    features=combined_features,
                    metadata=combined_metadata
                ))
        
        return ensemble_predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate ensemble trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            if pred.prediction != 0 and pred.confidence >= 0.6:  # Higher threshold for ensemble
                signal = TradingSignal(
                    timestamp=pred.timestamp,
                    symbol=pred.symbol,
                    signal=1 if pred.prediction > 0 else -1,
                    confidence=pred.confidence,
                    price=0.0,
                    metadata=pred.metadata
                )
                signals.append(signal)
        
        return signals


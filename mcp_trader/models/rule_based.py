"""
Rule-based trading strategies.
Simple but effective strategies based on technical indicators.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from ..models.base import BaseTradingModel, ModelPrediction, TradingSignal


class SMACrossoverStrategy(BaseTradingModel):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, **kwargs):
        super().__init__("SMACrossover", "rule_based", **kwargs)
        self.short_window = kwargs.get('short_window', 10)
        self.long_window = kwargs.get('long_window', 20)
        self.threshold = kwargs.get('threshold', 0.001)  # 0.1% threshold
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'SMACrossoverStrategy':
        """No training needed for rule-based strategy."""
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions based on SMA crossover."""
        df = data.copy()
        
        # Calculate SMAs
        df['sma_short'] = df['close'].rolling(self.short_window).mean()
        df['sma_long'] = df['close'].rolling(self.long_window).mean()
        
        # Calculate crossover signal
        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'] * (1 + self.threshold), 'signal'] = 1
        df.loc[df['sma_short'] < df['sma_long'] * (1 - self.threshold), 'signal'] = -1
        
        # Convert to ModelPrediction objects
        predictions = []
        for idx, row in df.iterrows():
            if not pd.isna(row['signal']):
                predictions.append(ModelPrediction(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=row['signal'],
                    confidence=0.6,  # Medium confidence
                    features={
                        'sma_short': row['sma_short'],
                        'sma_long': row['sma_long'],
                        'crossover_ratio': row['sma_short'] / row['sma_long'] if row['sma_long'] != 0 else 1
                    },
                    metadata={'strategy': 'sma_crossover', 'short_window': self.short_window, 'long_window': self.long_window}
                ))
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=int(pred.prediction),
                confidence=pred.confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata=pred.metadata
            ))
        
        return signals


class RSIStrategy(BaseTradingModel):
    """RSI-based trading strategy."""
    
    def __init__(self, **kwargs):
        super().__init__("RSIStrategy", "rule_based", **kwargs)
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.oversold_threshold = kwargs.get('oversold_threshold', 30)
        self.overbought_threshold = kwargs.get('overbought_threshold', 70)
        self.confirmation_periods = kwargs.get('confirmation_periods', 2)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'RSIStrategy':
        """No training needed for rule-based strategy."""
        self.is_fitted = True
        return self
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions based on RSI."""
        df = data.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # Generate signals
        df['signal'] = 0
        
        # Oversold condition
        oversold = df['rsi'] < self.oversold_threshold
        df.loc[oversold, 'signal'] = 1
        
        # Overbought condition
        overbought = df['rsi'] > self.overbought_threshold
        df.loc[overbought, 'signal'] = -1
        
        # Add confirmation requirement
        if self.confirmation_periods > 1:
            df['signal_confirmed'] = df['signal'].rolling(self.confirmation_periods).sum()
            df.loc[abs(df['signal_confirmed']) < self.confirmation_periods, 'signal'] = 0
        
        # Convert to ModelPrediction objects
        predictions = []
        for idx, row in df.iterrows():
            if not pd.isna(row['signal']) and row['signal'] != 0:
                confidence = min(abs(row['rsi'] - 50) / 50, 1.0)  # Higher confidence for extreme RSI
                
                predictions.append(ModelPrediction(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=row['signal'],
                    confidence=confidence,
                    features={
                        'rsi': row['rsi'],
                        'oversold': row['rsi'] < self.oversold_threshold,
                        'overbought': row['rsi'] > self.overbought_threshold
                    },
                    metadata={
                        'strategy': 'rsi',
                        'rsi_period': self.rsi_period,
                        'oversold_threshold': self.oversold_threshold,
                        'overbought_threshold': self.overbought_threshold
                    }
                ))
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=int(pred.prediction),
                confidence=pred.confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata=pred.metadata
            ))
        
        return signals


class BollingerBandsStrategy(BaseTradingModel):
    """Bollinger Bands mean reversion strategy."""
    
    def __init__(self, **kwargs):
        super().__init__("BollingerBands", "rule_based", **kwargs)
        self.period = kwargs.get('period', 20)
        self.std_dev = kwargs.get('std_dev', 2)
        self.threshold = kwargs.get('threshold', 0.1)  # 10% threshold for extreme positions
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BollingerBandsStrategy':
        """No training needed for rule-based strategy."""
        self.is_fitted = True
        return self
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions based on Bollinger Bands."""
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], self.period, self.std_dev
        )
        
        # Calculate position within bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when price touches lower band
        df.loc[df['close'] <= df['bb_lower'], 'signal'] = 1
        
        # Sell when price touches upper band
        df.loc[df['close'] >= df['bb_upper'], 'signal'] = -1
        
        # Convert to ModelPrediction objects
        predictions = []
        for idx, row in df.iterrows():
            if not pd.isna(row['signal']) and row['signal'] != 0:
                # Higher confidence for more extreme positions
                confidence = min(abs(row['bb_position'] - 0.5) * 2, 1.0)
                
                predictions.append(ModelPrediction(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=row['signal'],
                    confidence=confidence,
                    features={
                        'bb_upper': row['bb_upper'],
                        'bb_middle': row['bb_middle'],
                        'bb_lower': row['bb_lower'],
                        'bb_position': row['bb_position'],
                        'price': row['close']
                    },
                    metadata={
                        'strategy': 'bollinger_bands',
                        'period': self.period,
                        'std_dev': self.std_dev
                    }
                ))
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=int(pred.prediction),
                confidence=pred.confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata=pred.metadata
            ))
        
        return signals


class MACDStrategy(BaseTradingModel):
    """MACD crossover strategy."""
    
    def __init__(self, **kwargs):
        super().__init__("MACDStrategy", "rule_based", **kwargs)
        self.fast_period = kwargs.get('fast_period', 12)
        self.slow_period = kwargs.get('slow_period', 26)
        self.signal_period = kwargs.get('signal_period', 9)
        self.threshold = kwargs.get('threshold', 0.001)  # 0.1% threshold
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MACDStrategy':
        """No training needed for rule-based strategy."""
        self.is_fitted = True
        return self
    
    def calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD."""
        exp1 = prices.ewm(span=self.fast_period).mean()
        exp2 = prices.ewm(span=self.slow_period).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions based on MACD."""
        df = data.copy()
        
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self.calculate_macd(df['close'])
        
        # Generate signals
        df['signal'] = 0
        
        # MACD crosses above signal line
        df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'signal'] = 1
        
        # MACD crosses below signal line
        df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'signal'] = -1
        
        # Convert to ModelPrediction objects
        predictions = []
        for idx, row in df.iterrows():
            if not pd.isna(row['signal']) and row['signal'] != 0:
                # Confidence based on histogram strength
                confidence = min(abs(row['macd_histogram']) * 100, 1.0)
                
                predictions.append(ModelPrediction(
                    timestamp=row['timestamp'],
                    symbol=row.get('symbol', 'UNKNOWN'),
                    prediction=row['signal'],
                    confidence=confidence,
                    features={
                        'macd': row['macd'],
                        'macd_signal': row['macd_signal'],
                        'macd_histogram': row['macd_histogram']
                    },
                    metadata={
                        'strategy': 'macd',
                        'fast_period': self.fast_period,
                        'slow_period': self.slow_period,
                        'signal_period': self.signal_period
                    }
                ))
        
        return predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=int(pred.prediction),
                confidence=pred.confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata=pred.metadata
            ))
        
        return signals


class MultiIndicatorStrategy(BaseTradingModel):
    """Combines multiple indicators for more robust signals."""
    
    def __init__(self, **kwargs):
        super().__init__("MultiIndicator", "rule_based", **kwargs)
        self.strategies = [
            SMACrossoverStrategy(**kwargs.get('sma_params', {})),
            RSIStrategy(**kwargs.get('rsi_params', {})),
            BollingerBandsStrategy(**kwargs.get('bb_params', {})),
            MACDStrategy(**kwargs.get('macd_params', {}))
        ]
        self.vote_threshold = kwargs.get('vote_threshold', 2)  # Need 2+ strategies to agree
        self.weighted_voting = kwargs.get('weighted_voting', True)
        self.strategy_weights = kwargs.get('strategy_weights', [0.3, 0.2, 0.3, 0.2])  # SMA, RSI, BB, MACD
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MultiIndicatorStrategy':
        """Fit all sub-strategies."""
        for strategy in self.strategies:
            strategy.fit(data, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate predictions by combining multiple strategies."""
        all_predictions = []
        
        # Get predictions from all strategies
        for strategy in self.strategies:
            predictions = strategy.predict(data)
            all_predictions.append(predictions)
        
        # Combine predictions
        combined_predictions = []
        
        # Group by timestamp
        timestamps = set()
        for pred_list in all_predictions:
            for pred in pred_list:
                timestamps.add(pred.timestamp)
        
        for timestamp in sorted(timestamps):
            # Get predictions for this timestamp
            timestamp_predictions = []
            for pred_list in all_predictions:
                for pred in pred_list:
                    if pred.timestamp == timestamp:
                        timestamp_predictions.append(pred)
                        break
            
            if len(timestamp_predictions) >= self.vote_threshold:
                # Calculate weighted vote
                if self.weighted_voting:
                    weighted_signal = sum(
                        pred.prediction * weight 
                        for pred, weight in zip(timestamp_predictions, self.strategy_weights[:len(timestamp_predictions)])
                    )
                    final_signal = 1 if weighted_signal > 0.5 else -1 if weighted_signal < -0.5 else 0
                else:
                    # Simple majority vote
                    signals = [pred.prediction for pred in timestamp_predictions]
                    final_signal = 1 if sum(signals) > 0 else -1 if sum(signals) < 0 else 0
                
                if final_signal != 0:
                    # Calculate confidence based on agreement
                    agreement = sum(1 for pred in timestamp_predictions if pred.prediction == final_signal)
                    confidence = agreement / len(timestamp_predictions)
                    
                    # Get symbol and price
                    symbol = timestamp_predictions[0].symbol
                    price = data[data['timestamp'] == timestamp]['close'].iloc[0] if len(data[data['timestamp'] == timestamp]) > 0 else 0
                    
                    combined_predictions.append(ModelPrediction(
                        timestamp=timestamp,
                        symbol=symbol,
                        prediction=final_signal,
                        confidence=confidence,
                        features={f'strategy_{i}': pred.prediction for i, pred in enumerate(timestamp_predictions)},
                        metadata={
                            'strategy': 'multi_indicator',
                            'num_strategies': len(timestamp_predictions),
                            'agreement': agreement,
                            'individual_predictions': [pred.prediction for pred in timestamp_predictions]
                        }
                    ))
        
        return combined_predictions
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals."""
        predictions = self.predict(data)
        signals = []
        
        for pred in predictions:
            signals.append(TradingSignal(
                timestamp=pred.timestamp,
                symbol=pred.symbol,
                signal=int(pred.prediction),
                confidence=pred.confidence,
                price=data[data['timestamp'] == pred.timestamp]['close'].iloc[0],
                metadata=pred.metadata
            ))
        
        return signals

"""
DMark Indicator - Proprietary trading indicator for Aster Perps.
Combines multiple market dynamics into a single comprehensive signal.

The DMark (Dynamic Market) indicator is designed to capture:
1. Price momentum and trend strength
2. Volatility regime changes
3. Volume confirmation
4. Market microstructure signals
5. Cross-timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DMarkConfig:
    """Configuration for DMark indicator."""
    # Core parameters
    lookback_period: int = 20
    momentum_period: int = 10
    volatility_period: int = 14
    volume_period: int = 10
    
    # Signal thresholds
    strong_signal_threshold: float = 0.7
    moderate_signal_threshold: float = 0.4
    weak_signal_threshold: float = 0.2
    
    # Weighting factors
    momentum_weight: float = 0.3
    volatility_weight: float = 0.25
    volume_weight: float = 0.2
    microstructure_weight: float = 0.15
    trend_weight: float = 0.1
    
    # Advanced parameters
    regime_detection_period: int = 50
    noise_filter_period: int = 5
    adaptive_threshold: bool = True


class DMarkIndicator:
    """
    DMark Indicator - Proprietary trading signal generator.
    
    The DMark indicator combines multiple market dynamics to generate
    comprehensive trading signals with confidence levels.
    """
    
    def __init__(self, config: DMarkConfig = None):
        self.config = config or DMarkConfig()
        self.signal_history = []
        self.regime_history = []
        
    def calculate(self, 
                  high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  volume: pd.Series,
                  open_price: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
        """
        Calculate DMark indicator values.
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            open_price: Open prices (optional)
            
        Returns:
            Dictionary containing DMark signals and components
        """
        logger.info("Calculating DMark indicator...")
        
        # Ensure all series have the same length
        min_length = min(len(high), len(low), len(close), len(volume))
        high = high.iloc[:min_length]
        low = low.iloc[:min_length]
        close = close.iloc[:min_length]
        volume = volume.iloc[:min_length]
        
        if open_price is not None:
            open_price = open_price.iloc[:min_length]
        
        # Calculate individual components
        momentum_signal = self._calculate_momentum_signal(close)
        volatility_signal = self._calculate_volatility_signal(high, low, close)
        volume_signal = self._calculate_volume_signal(close, volume)
        microstructure_signal = self._calculate_microstructure_signal(high, low, close, volume)
        trend_signal = self._calculate_trend_signal(close)
        
        # Detect market regime
        regime = self._detect_market_regime(close, volume)
        
        # Calculate adaptive weights based on regime
        weights = self._calculate_adaptive_weights(regime)
        
        # Combine signals with weights
        dmark_raw = (
            momentum_signal * weights['momentum'] +
            volatility_signal * weights['volatility'] +
            volume_signal * weights['volume'] +
            microstructure_signal * weights['microstructure'] +
            trend_signal * weights['trend']
        )
        
        # Apply noise filtering
        dmark_filtered = self._apply_noise_filter(dmark_raw)
        
        # Generate final signals
        dmark_signal = self._generate_signals(dmark_filtered, regime)
        
        # Calculate confidence levels
        confidence = self._calculate_confidence(dmark_filtered, regime)
        
        # Store history
        self.signal_history.append(dmark_signal.iloc[-1] if len(dmark_signal) > 0 else 0)
        self.regime_history.append(regime.iloc[-1] if len(regime) > 0 else 0)
        
        return {
            'dmark_signal': dmark_signal,
            'dmark_raw': dmark_raw,
            'dmark_filtered': dmark_filtered,
            'confidence': confidence,
            'regime': regime,
            'momentum_component': momentum_signal,
            'volatility_component': volatility_signal,
            'volume_component': volume_signal,
            'microstructure_component': microstructure_signal,
            'trend_component': trend_signal,
            'weights': pd.Series(weights, index=close.index)
        }
    
    def _calculate_momentum_signal(self, close: pd.Series) -> pd.Series:
        """Calculate momentum component using multiple timeframes."""
        # Short-term momentum
        short_momentum = close.pct_change(self.config.momentum_period)
        
        # Medium-term momentum
        medium_momentum = close.pct_change(self.config.lookback_period)
        
        # Price acceleration
        returns = close.pct_change()
        acceleration = returns.diff()
        
        # RSI-based momentum
        rsi = self._calculate_rsi(close, 14)
        rsi_momentum = (rsi - 50) / 50  # Normalize to -1 to 1
        
        # MACD momentum
        macd, signal, histogram = self._calculate_macd(close)
        macd_momentum = histogram / close  # Normalize by price
        
        # Combine momentum signals
        momentum_signal = (
            short_momentum * 0.4 +
            medium_momentum * 0.3 +
            acceleration * 0.1 +
            rsi_momentum * 0.1 +
            macd_momentum * 0.1
        )
        
        return momentum_signal.fillna(0)
    
    def _calculate_volatility_signal(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate volatility component with regime awareness."""
        # True Range
        tr = self._calculate_true_range(high, low, close)
        atr = tr.rolling(self.config.volatility_period).mean()
        
        # Volatility ratio (current vs historical)
        vol_ratio = atr / atr.rolling(self.config.regime_detection_period).mean()
        
        # Bollinger Band squeeze
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_squeeze = bb_width < bb_width.rolling(20).mean() * 0.8
        
        # Volatility momentum
        vol_momentum = atr.pct_change()
        
        # Volatility regime signal
        high_vol_threshold = atr.rolling(50).quantile(0.8)
        low_vol_threshold = atr.rolling(50).quantile(0.2)
        
        vol_regime = pd.Series(0, index=close.index)
        vol_regime[atr > high_vol_threshold] = 1  # High volatility
        vol_regime[atr < low_vol_threshold] = -1  # Low volatility
        
        # Combine volatility signals
        volatility_signal = (
            vol_ratio * 0.4 +
            bb_squeeze.astype(float) * 0.3 +
            vol_momentum * 0.2 +
            vol_regime * 0.1
        )
        
        return volatility_signal.fillna(0)
    
    def _calculate_volume_signal(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate volume confirmation signal."""
        # Volume moving average
        vol_sma = volume.rolling(self.config.volume_period).mean()
        vol_ratio = volume / vol_sma
        
        # Price-volume relationship
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        price_volume_corr = price_change.rolling(10).corr(volume_change)
        
        # On-Balance Volume
        obv = self._calculate_obv(close, volume)
        obv_signal = obv.pct_change()
        
        # Volume momentum
        vol_momentum = volume.pct_change()
        
        # Volume profile analysis
        vol_profile = volume.rolling(20).quantile(0.8)
        high_volume = volume > vol_profile
        
        # Combine volume signals
        volume_signal = (
            vol_ratio * 0.3 +
            price_volume_corr * 0.25 +
            obv_signal * 0.2 +
            vol_momentum * 0.15 +
            high_volume.astype(float) * 0.1
        )
        
        return volume_signal.fillna(0)
    
    def _calculate_microstructure_signal(self, high: pd.Series, low: pd.Series, 
                                       close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate market microstructure signals."""
        # Price impact (high-low range relative to close)
        price_range = (high - low) / close
        price_impact = price_range.rolling(5).mean()
        
        # Order flow imbalance (simplified)
        close_open_ratio = close / close.shift(1)
        order_flow = np.where(close_open_ratio > 1, volume, -volume)
        order_flow_signal = pd.Series(order_flow).rolling(10).sum()
        
        # Tick-by-tick momentum
        tick_momentum = close.diff().rolling(5).sum()
        
        # Spread proxy (high-low range)
        spread_proxy = (high - low) / close
        spread_signal = spread_proxy.rolling(10).mean()
        
        # Market depth proxy
        depth_proxy = volume / (high - low + 1e-8)  # Avoid division by zero
        depth_signal = depth_proxy.rolling(10).mean()
        
        # Combine microstructure signals
        microstructure_signal = (
            price_impact * 0.3 +
            order_flow_signal / close * 0.25 +
            tick_momentum / close * 0.2 +
            spread_signal * 0.15 +
            depth_signal * 0.1
        )
        
        return microstructure_signal.fillna(0)
    
    def _calculate_trend_signal(self, close: pd.Series) -> pd.Series:
        """Calculate trend strength signal."""
        # Multiple timeframe trend analysis
        short_trend = close.rolling(5).mean() / close.rolling(20).mean() - 1
        medium_trend = close.rolling(10).mean() / close.rolling(50).mean() - 1
        long_trend = close.rolling(20).mean() / close.rolling(100).mean() - 1
        
        # Trend consistency
        trend_direction = np.sign(close.diff())
        trend_consistency = trend_direction.rolling(10).mean()
        
        # ADX (Average Directional Index) approximation
        adx = self._calculate_adx(close, close, close)  # Simplified ADX
        
        # Trend momentum
        trend_momentum = close.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Combine trend signals
        trend_signal = (
            short_trend * 0.3 +
            medium_trend * 0.25 +
            long_trend * 0.2 +
            trend_consistency * 0.15 +
            adx * 0.1
        )
        
        return trend_signal.fillna(0)
    
    def _detect_market_regime(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Detect current market regime."""
        # Volatility regime
        returns = close.pct_change()
        volatility = returns.rolling(self.config.regime_detection_period).std()
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        # Trend regime
        trend_strength = abs(close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        trend_percentile = trend_strength.rolling(100).rank(pct=True)
        
        # Volume regime
        vol_ratio = volume / volume.rolling(50).mean()
        volume_percentile = vol_ratio.rolling(100).rank(pct=True)
        
        # Combine regime indicators
        regime_score = (
            vol_percentile * 0.4 +
            trend_percentile * 0.3 +
            volume_percentile * 0.3
        )
        
        # Classify regimes
        regime = pd.Series(0, index=close.index)  # Normal
        regime[regime_score > 0.7] = 1  # High volatility/trend
        regime[regime_score < 0.3] = -1  # Low volatility/trend
        
        return regime.fillna(0)
    
    def _calculate_adaptive_weights(self, regime: pd.Series) -> Dict[str, float]:
        """Calculate adaptive weights based on market regime."""
        current_regime = regime.iloc[-1] if len(regime) > 0 else 0
        
        if current_regime == 1:  # High volatility/trend regime
            return {
                'momentum': 0.4,
                'volatility': 0.3,
                'volume': 0.15,
                'microstructure': 0.1,
                'trend': 0.05
            }
        elif current_regime == -1:  # Low volatility/trend regime
            return {
                'momentum': 0.2,
                'volatility': 0.2,
                'volume': 0.2,
                'microstructure': 0.25,
                'trend': 0.15
            }
        else:  # Normal regime
            return {
                'momentum': self.config.momentum_weight,
                'volatility': self.config.volatility_weight,
                'volume': self.config.volume_weight,
                'microstructure': self.config.microstructure_weight,
                'trend': self.config.trend_weight
            }
    
    def _apply_noise_filter(self, signal: pd.Series) -> pd.Series:
        """Apply noise filtering to reduce false signals."""
        # Simple moving average filter
        filtered = signal.rolling(self.config.noise_filter_period).mean()
        
        # Outlier removal
        median = filtered.rolling(20).median()
        mad = (filtered - median).abs().rolling(20).median()
        threshold = 3 * mad
        filtered = np.where(
            (filtered - median).abs() > threshold,
            median,
            filtered
        )
        
        return pd.Series(filtered, index=signal.index)
    
    def _generate_signals(self, dmark_filtered: pd.Series, regime: pd.Series) -> pd.Series:
        """Generate final trading signals from filtered DMark values."""
        signals = pd.Series(0, index=dmark_filtered.index)
        
        # Adaptive thresholds based on regime
        current_regime = regime.iloc[-1] if len(regime) > 0 else 0
        
        if current_regime == 1:  # High volatility regime - use higher thresholds
            strong_threshold = self.config.strong_signal_threshold * 1.2
            moderate_threshold = self.config.moderate_signal_threshold * 1.1
        elif current_regime == -1:  # Low volatility regime - use lower thresholds
            strong_threshold = self.config.strong_signal_threshold * 0.8
            moderate_threshold = self.config.moderate_signal_threshold * 0.9
        else:
            strong_threshold = self.config.strong_signal_threshold
            moderate_threshold = self.config.moderate_signal_threshold
        
        # Generate signals
        signals[dmark_filtered > strong_threshold] = 1  # Strong buy
        signals[dmark_filtered < -strong_threshold] = -1  # Strong sell
        signals[(dmark_filtered > moderate_threshold) & (dmark_filtered <= strong_threshold)] = 0.5  # Moderate buy
        signals[(dmark_filtered < -moderate_threshold) & (dmark_filtered >= -strong_threshold)] = -0.5  # Moderate sell
        
        return signals
    
    def _calculate_confidence(self, dmark_filtered: pd.Series, regime: pd.Series) -> pd.Series:
        """Calculate confidence level for signals."""
        # Base confidence from signal strength
        base_confidence = np.abs(dmark_filtered)
        
        # Regime consistency bonus
        regime_consistency = 1.0
        if len(self.regime_history) > 5:
            recent_regimes = self.regime_history[-5:]
            regime_consistency = 1.0 + 0.2 * (len(set(recent_regimes)) == 1)  # Bonus for consistent regime
        
        # Signal consistency bonus
        signal_consistency = 1.0
        if len(self.signal_history) > 3:
            recent_signals = self.signal_history[-3:]
            signal_consistency = 1.0 + 0.1 * (len(set(np.sign(recent_signals))) == 1)  # Bonus for consistent direction
        
        # Calculate final confidence
        confidence = base_confidence * regime_consistency * signal_consistency
        confidence = np.clip(confidence, 0, 1)  # Clamp to [0, 1]
        
        return pd.Series(confidence, index=dmark_filtered.index)
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, close: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    def _calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        return np.maximum(high_low, np.maximum(high_close, low_close))
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        return obv
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate simplified ADX."""
        # Simplified ADX calculation
        tr = self._calculate_true_range(high, low, close)
        atr = tr.rolling(period).mean()
        
        # Directional movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / atr)
        di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / atr)
        
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def get_signal_strength(self, dmark_value: float) -> str:
        """Get signal strength description."""
        abs_value = abs(dmark_value)
        
        if abs_value >= self.config.strong_signal_threshold:
            return "STRONG"
        elif abs_value >= self.config.moderate_signal_threshold:
            return "MODERATE"
        elif abs_value >= self.config.weak_signal_threshold:
            return "WEAK"
        else:
            return "NEUTRAL"
    
    def get_signal_direction(self, dmark_value: float) -> str:
        """Get signal direction description."""
        if dmark_value > self.config.moderate_signal_threshold:
            return "BUY"
        elif dmark_value < -self.config.moderate_signal_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def get_regime_description(self, regime_value: int) -> str:
        """Get regime description."""
        if regime_value == 1:
            return "HIGH_VOLATILITY"
        elif regime_value == -1:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL"


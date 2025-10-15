"""
Feature engineering pipeline for trading models.
Creates comprehensive features from market data and sentiment.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
try:
    import talib
except ImportError:
    talib = None
    import warnings
    warnings.warn("TA-Lib not available. Some technical indicators will use fallback implementations.")
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Technical indicators
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Volatility features
    volatility_periods: List[int] = None
    garch_lags: int = 5
    
    # Volume features
    volume_sma_periods: List[int] = None
    obv_period: int = 10
    
    # Time-based features
    include_time_features: bool = True
    include_cyclical_time: bool = True
    
    # Sentiment features
    sentiment_lookback: int = 24  # hours
    sentiment_aggregation: str = 'mean'  # 'mean', 'max', 'min', 'std'
    
    # Cross-asset features
    include_cross_asset: bool = True
    reference_assets: List[str] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]
        if self.volatility_periods is None:
            self.volatility_periods = [5, 10, 20, 30]
        if self.volume_sma_periods is None:
            self.volume_sma_periods = [5, 10, 20]
        if self.reference_assets is None:
            self.reference_assets = ['BTCUSDT', 'ETHUSDT']


class TechnicalIndicators:
    """Technical indicator calculations."""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD indicator."""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std: float = 2.0) -> tuple:
        """Bollinger Bands."""
        sma = TechnicalIndicators.sma(prices, period)
        std_dev = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mean_deviation)


class VolatilityFeatures:
    """Volatility-based features."""
    
    @staticmethod
    def realized_volatility(returns: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate realized volatility for different periods."""
        features = {}
        for period in periods:
            features[f'realized_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(24)  # Daily vol
        return features
    
    @staticmethod
    def garch_volatility(returns: pd.Series, lags: int = 5) -> pd.Series:
        """Simple GARCH-like volatility estimation."""
        # Simplified GARCH(1,1) approximation
        squared_returns = returns ** 2
        vol = squared_returns.ewm(alpha=0.1).mean()  # Simple exponential smoothing
        return np.sqrt(vol)
    
    @staticmethod
    def volatility_ratio(returns: pd.Series, short_period: int = 5, long_period: int = 20) -> pd.Series:
        """Volatility ratio (short/long term)."""
        short_vol = returns.rolling(window=short_period).std()
        long_vol = returns.rolling(window=long_period).std()
        return short_vol / long_vol
    
    @staticmethod
    def volatility_percentile(returns: pd.Series, period: int = 252) -> pd.Series:
        """Volatility percentile ranking."""
        rolling_vol = returns.rolling(window=period).std()
        return rolling_vol.rolling(window=period).rank(pct=True)


class VolumeFeatures:
    """Volume-based features."""
    
    @staticmethod
    def volume_sma(volume: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Volume moving averages."""
        features = {}
        for period in periods:
            features[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
        return features
    
    @staticmethod
    def volume_ratio(volume: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Volume ratios."""
        features = {}
        for period in periods:
            sma = volume.rolling(window=period).mean()
            features[f'volume_ratio_{period}'] = volume / sma
        return features
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series, period: int = 10) -> pd.Series:
        """On-Balance Volume."""
        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        return obv.rolling(window=period).mean()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwap


class TimeFeatures:
    """Time-based features."""
    
    @staticmethod
    def extract_time_features(timestamps: pd.Series) -> Dict[str, pd.Series]:
        """Extract time-based features."""
        features = {}
        dt = pd.to_datetime(timestamps)
        
        # Basic time features
        features['hour'] = dt.dt.hour
        features['day_of_week'] = dt.dt.dayofweek
        features['day_of_month'] = dt.dt.day
        features['month'] = dt.dt.month
        features['quarter'] = dt.dt.quarter
        
        # Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features


class SentimentFeatures:
    """Sentiment-based features."""
    
    @staticmethod
    def aggregate_sentiment(sentiment_data: pd.DataFrame, 
                          lookback_hours: int = 24,
                          aggregation: str = 'mean') -> Dict[str, pd.Series]:
        """Aggregate sentiment data over time windows."""
        features = {}
        
        if sentiment_data.empty:
            return features
        
        # Set timestamp as index
        sentiment_data = sentiment_data.set_index('timestamp')
        
        # Resample to hourly data
        hourly_sentiment = sentiment_data.resample('1H').agg({
            'sentiment_score': aggregation,
            'confidence': 'mean',
            'symbols': 'count'
        }).fillna(0)
        
        # Create rolling features
        for hours in [1, 6, 12, 24]:
            features[f'sentiment_score_{hours}h'] = hourly_sentiment['sentiment_score'].rolling(window=hours).agg(aggregation)
            features[f'sentiment_confidence_{hours}h'] = hourly_sentiment['confidence'].rolling(window=hours).agg(aggregation)
            features[f'sentiment_volume_{hours}h'] = hourly_sentiment['symbols'].rolling(window=hours).sum()
        
        return features


class CrossAssetFeatures:
    """Cross-asset correlation and momentum features."""
    
    @staticmethod
    def correlation_features(price_data: Dict[str, pd.Series], 
                           reference_assets: List[str],
                           periods: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """Calculate correlation features with reference assets."""
        features = {}
        
        for ref_asset in reference_assets:
            if ref_asset not in price_data:
                continue
            
            ref_returns = price_data[ref_asset].pct_change()
            
            for period in periods:
                # Rolling correlation
                for asset, prices in price_data.items():
                    if asset == ref_asset:
                        continue
                    
                    asset_returns = prices.pct_change()
                    corr = ref_returns.rolling(window=period).corr(asset_returns)
                    features[f'corr_{asset}_{ref_asset}_{period}'] = corr
        
        return features
    
    @staticmethod
    def momentum_features(price_data: Dict[str, pd.Series], 
                         periods: List[int] = [5, 10, 20]) -> Dict[str, pd.Series]:
        """Calculate momentum features across assets."""
        features = {}
        
        for asset, prices in price_data.items():
            for period in periods:
                # Price momentum
                momentum = prices / prices.shift(period) - 1
                features[f'momentum_{asset}_{period}'] = momentum
                
                # Volatility momentum
                returns = prices.pct_change()
                vol_momentum = returns.rolling(window=period).std() / returns.rolling(window=period*2).std()
                features[f'vol_momentum_{asset}_{period}'] = vol_momentum
        
        return features


class FeatureEngine:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_columns = []
        self.scaler = None
    
    def create_features(self, 
                       market_data: pd.DataFrame,
                       sentiment_data: pd.DataFrame = None,
                       cross_asset_data: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """Create comprehensive feature set."""
        
        logger.info("Starting feature engineering...")
        
        # Start with market data
        features_df = market_data.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate returns
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # Technical indicators
        logger.info("Creating technical indicators...")
        tech_features = self._create_technical_features(features_df)
        features_df = pd.concat([features_df, tech_features], axis=1)
        
        # Volatility features
        logger.info("Creating volatility features...")
        vol_features = self._create_volatility_features(features_df)
        features_df = pd.concat([features_df, vol_features], axis=1)
        
        # Volume features
        logger.info("Creating volume features...")
        volume_features = self._create_volume_features(features_df)
        features_df = pd.concat([features_df, volume_features], axis=1)
        
        # Time features
        if self.config.include_time_features:
            logger.info("Creating time features...")
            time_features = self._create_time_features(features_df)
            features_df = pd.concat([features_df, time_features], axis=1)
        
        # Sentiment features
        if sentiment_data is not None and not sentiment_data.empty:
            logger.info("Creating sentiment features...")
            sentiment_features = self._create_sentiment_features(sentiment_data)
            features_df = pd.concat([features_df, sentiment_features], axis=1)
        
        # Cross-asset features
        if self.config.include_cross_asset and cross_asset_data:
            logger.info("Creating cross-asset features...")
            cross_asset_features = self._create_cross_asset_features(cross_asset_data)
            features_df = pd.concat([features_df, cross_asset_features], axis=1)
        
        # Store feature columns
        self.feature_columns = [col for col in features_df.columns if col not in market_data.columns]
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return features_df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        features = {}
        
        # Moving averages
        for period in self.config.sma_periods:
            features[f'sma_{period}'] = TechnicalIndicators.sma(df['close'], period)
            features[f'sma_ratio_{period}'] = df['close'] / features[f'sma_{period}']
        
        for period in self.config.ema_periods:
            features[f'ema_{period}'] = TechnicalIndicators.ema(df['close'], period)
            features[f'ema_ratio_{period}'] = df['close'] / features[f'ema_{period}']
        
        # RSI
        features['rsi'] = TechnicalIndicators.rsi(df['close'], self.config.rsi_period)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(
            df['close'], 
            self.config.macd_fast, 
            self.config.macd_slow, 
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_crossover'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            df['close'], 
            self.config.bb_period, 
            self.config.bb_std
        )
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean() * 0.8).astype(int)
        
        # ATR
        features['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], self.config.atr_period)
        features['atr_ratio'] = features['atr'] / df['close']
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_overbought'] = (stoch_k > 80).astype(int)
        features['stoch_oversold'] = (stoch_k < 20).astype(int)
        
        # Williams %R
        features['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        
        # CCI
        features['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        
        return pd.DataFrame(features, index=df.index)
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features."""
        features = {}
        
        # Realized volatility
        vol_features = VolatilityFeatures.realized_volatility(df['returns'], self.config.volatility_periods)
        features.update(vol_features)
        
        # GARCH volatility
        features['garch_vol'] = VolatilityFeatures.garch_volatility(df['returns'], self.config.garch_lags)
        
        # Volatility ratios
        features['vol_ratio_5_20'] = VolatilityFeatures.volatility_ratio(df['returns'], 5, 20)
        features['vol_ratio_10_30'] = VolatilityFeatures.volatility_ratio(df['returns'], 10, 30)
        
        # Volatility percentiles
        features['vol_percentile'] = VolatilityFeatures.volatility_percentile(df['returns'])
        
        return pd.DataFrame(features, index=df.index)
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume features."""
        features = {}
        
        # Volume SMAs
        vol_sma_features = VolumeFeatures.volume_sma(df['volume'], self.config.volume_sma_periods)
        features.update(vol_sma_features)
        
        # Volume ratios
        vol_ratio_features = VolumeFeatures.volume_ratio(df['volume'], self.config.volume_sma_periods)
        features.update(vol_ratio_features)
        
        # OBV
        features['obv'] = VolumeFeatures.obv(df['close'], df['volume'], self.config.obv_period)
        
        # VWAP
        features['vwap'] = VolumeFeatures.vwap(df['high'], df['low'], df['close'], df['volume'])
        features['vwap_ratio'] = df['close'] / features['vwap']
        
        return pd.DataFrame(features, index=df.index)
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        time_features = TimeFeatures.extract_time_features(df['timestamp'])
        return pd.DataFrame(time_features, index=df.index)
    
    def _create_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features."""
        sentiment_features = SentimentFeatures.aggregate_sentiment(
            sentiment_data, 
            self.config.sentiment_lookback,
            self.config.sentiment_aggregation
        )
        return pd.DataFrame(sentiment_features, index=sentiment_data.index)
    
    def _create_cross_asset_features(self, cross_asset_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Create cross-asset features."""
        features = {}
        
        # Correlation features
        corr_features = CrossAssetFeatures.correlation_features(
            cross_asset_data, 
            self.config.reference_assets
        )
        features.update(corr_features)
        
        # Momentum features
        momentum_features = CrossAssetFeatures.momentum_features(cross_asset_data)
        features.update(momentum_features)
        
        return pd.DataFrame(features, index=list(cross_asset_data.values())[0].index)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(self.feature_columns, model.coef_))
        else:
            logger.warning("Model does not support feature importance")
            return {}
    
    def select_features(self, 
                       features_df: pd.DataFrame, 
                       target: pd.Series,
                       method: str = 'correlation',
                       top_k: int = 50) -> List[str]:
        """Select top features using various methods."""
        
        if method == 'correlation':
            # Select features with highest correlation to target
            correlations = features_df[self.feature_columns].corrwith(target).abs()
            selected_features = correlations.nlargest(top_k).index.tolist()
        
        elif method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression
                from sklearn.impute import SimpleImputer
                
                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                features_imputed = imputer.fit_transform(features_df[self.feature_columns])
                
                # Calculate mutual information
                mi_scores = mutual_info_regression(features_imputed, target, random_state=42)
                mi_scores = pd.Series(mi_scores, index=self.feature_columns)
                selected_features = mi_scores.nlargest(top_k).index.tolist()
            except ImportError:
                logger.warning("scikit-learn not available for mutual information feature selection")
                return self.feature_columns[:top_k]
        
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        logger.info(f"Selected {len(selected_features)} features using {method}")
        return selected_features


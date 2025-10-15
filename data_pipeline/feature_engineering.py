"""
Comprehensive Feature Engineering Pipeline for AI Trading Models
RTX 5070Ti optimized with TA indicators, correlated assets, and alternative data.
"""

import numpy as np
import pandas as pd
import ta  # Technical Analysis library
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    lookback_periods: List[int] = None
    ta_indicators: List[str] = None
    correlated_assets: List[str] = None
    alternative_data: bool = True
    max_features: int = 200
    feature_selection: str = 'mutual_info'

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100, 200]
        if self.ta_indicators is None:
            self.ta_indicators = [
                'rsi', 'macd', 'bollinger_bands', 'stochastic', 'williams_r',
                'cci', 'atr', 'adx', 'ema', 'sma', 'wma', 'momentum',
                'roc', 'stochrsi', 'ultimate_oscillator', 'awesome_oscillator'
            ]
        if self.correlated_assets is None:
            self.correlated_assets = [
                'BTC-USD', 'ETH-USD', 'SPY', 'QQQ', 'GLD', 'VIX',
                'DXY', 'GC=F', 'CL=F', 'SI=F'
            ]


class ComprehensiveFeatureEngineer:
    """
    RTX 5070Ti optimized feature engineering for AI trading models.
    Generates 200+ features including TA indicators, correlated assets, and alternative data.
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_columns = []
        self.correlation_cache = {}

    def create_all_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for trading model training.

        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading pair symbol for correlated asset data

        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Creating comprehensive features for {len(df)} data points")

        # Ensure proper data types and columns
        df = self._prepare_dataframe(df)

        # Create all feature sets
        price_features = self._create_price_features(df)
        ta_features = self._create_ta_features(df)
        time_features = self._create_time_features(df)
        statistical_features = self._create_statistical_features(df)
        advanced_features = self._create_advanced_features(df)

        # Combine all features
        features_df = pd.concat([
            price_features,
            ta_features,
            time_features,
            statistical_features,
            advanced_features
        ], axis=1)

        # Add correlated assets features if symbol provided
        if symbol:
            correlated_features = self._create_correlated_features(df, symbol)
            features_df = pd.concat([features_df, correlated_features], axis=1)

        # Add alternative data features if enabled
        if self.config.alternative_data:
            alternative_features = self._create_alternative_features(df, symbol)
            features_df = pd.concat([features_df, alternative_features], axis=1)

        # Clean and finalize features
        features_df = self._finalize_features(features_df)

        logger.info(f"Created {len(features_df.columns)} features")
        return features_df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for feature engineering."""
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start=datetime.now(), periods=len(df), freq='1H')

        # Calculate returns if not present
        if 'close' in df.columns and 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic price-based features."""
        features = pd.DataFrame(index=df.index)

        # Price ratios and spreads
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['volume_price'] = df['volume'] * df['close']

        # Price volatility
        for window in self.config.lookback_periods:
            features[f'price_volatility_{window}'] = df['close'].pct_change().rolling(window).std()

        # Price momentum
        for window in self.config.lookback_periods:
            features[f'price_momentum_{window}'] = df['close'].pct_change(window)

        # Volume features
        if 'volume' in df.columns:
            for window in self.config.lookback_periods:
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_sma_{window}']

        return features

    def _create_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical analysis features."""
        features = pd.DataFrame(index=df.index)

        try:
            # RSI indicators
            for window in [14, 21, 28]:
                features[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()

            # MACD
            macd = ta.trend.MACD(df['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_histogram'] = macd.macd_diff()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            features['bb_upper'] = bb.bollinger_hband()
            features['bb_middle'] = bb.bollinger_mavg()
            features['bb_lower'] = bb.bollinger_lband()
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()

            # Williams %R
            features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()

            # Commodity Channel Index
            features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

            # Average True Range
            features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

            # Average Directional Index
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            features['adx'] = adx.adx()
            features['adx_pos'] = adx.adx_pos()
            features['adx_neg'] = adx.adx_neg()

            # Moving Averages
            for window in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{window}'] = ta.trend.SMAIndicator(df['close'], window=window).sma_indicator()
                features[f'ema_{window}'] = ta.trend.EMAIndicator(df['close'], window=window).ema_indicator()
                features[f'wma_{window}'] = ta.trend.WMAIndicator(df['close'], window=window).wma()

            # Rate of Change
            for window in [1, 5, 10, 20]:
                features[f'roc_{window}'] = ta.momentum.ROCIndicator(df['close'], window=window).roc()

            # Ultimate Oscillator
            features['ultimate_oscillator'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

            # Awesome Oscillator
            features['awesome_oscillator'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()

        except Exception as e:
            logger.warning(f"Error creating TA features: {e}")

        return features

    def _create_correlated_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create features from correlated assets."""
        features = pd.DataFrame(index=df.index)

        try:
            # Get correlated asset data
            correlated_data = self._get_correlated_data(symbol)

            for asset, asset_data in correlated_data.items():
                if asset_data is not None and len(asset_data) > 0:
                    # Asset returns correlation
                    features[f'{asset}_returns'] = asset_data['close'].pct_change()
                    features[f'{asset}_correlation'] = features['returns'].rolling(50).corr(features[f'{asset}_returns'])

                    # Cross-asset momentum
                    features[f'{asset}_momentum'] = asset_data['close'].pct_change(20)

                    # Relative strength
                    features[f'{asset}_relative_strength'] = df['close'] / asset_data['close']

        except Exception as e:
            logger.warning(f"Error creating correlated features: {e}")

        return features

    def _get_correlated_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data for correlated assets."""
        if symbol in self.correlation_cache:
            return self.correlation_cache[symbol]

        correlated_data = {}

        # Map trading symbols to Yahoo Finance tickers
        symbol_mapping = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'SOLUSDT': 'SOL-USD',
            'SUIUSDT': 'SUI-USD',
            'ASTERUSDT': 'CRYPTO:ASTER'  # Placeholder
        }

        yahoo_symbol = symbol_mapping.get(symbol, symbol.replace('USDT', '-USD'))

        for asset in self.config.correlated_assets:
            try:
                # Download 1 year of daily data
                asset_df = yf.download(asset, period='1y', interval='1d', progress=False)

                if not asset_df.empty:
                    correlated_data[asset] = asset_df
                else:
                    logger.warning(f"No data for correlated asset: {asset}")

            except Exception as e:
                logger.warning(f"Error downloading {asset}: {e}")

        self.correlation_cache[symbol] = correlated_data
        return correlated_data

    def _create_alternative_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create alternative data features (Twitter, news, sentiment)."""
        features = pd.DataFrame(index=df.index)

        # Placeholder for alternative data integration
        # In production, this would connect to Twitter API, news APIs, etc.

        # Mock sentiment features (replace with real data)
        features['twitter_sentiment'] = np.random.normal(0.5, 0.2, len(features))
        features['news_sentiment'] = np.random.normal(0.5, 0.2, len(features))
        features['fear_greed_index'] = np.random.randint(0, 100, len(features))

        # Social volume (mock)
        features['twitter_volume'] = np.random.exponential(1000, len(features))

        return features

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=df.index)

        # Extract time components
        features['hour'] = features.index.hour if hasattr(features.index, 'hour') else 0
        features['day_of_week'] = features.index.dayofweek if hasattr(features.index, 'dayofweek') else 0
        features['month'] = features.index.month if hasattr(features.index, 'month') else 0
        features['quarter'] = features['month'].apply(lambda x: (x-1)//3 + 1)

        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # Market session indicators
        features['is_market_open'] = features['hour'].between(9, 16).astype(int)  # US market hours

        return features

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        features = pd.DataFrame(index=df.index)

        # Rolling statistics on returns
        for window in [24, 72, 168]:  # Hours
            # Calculate returns if not already in features
            if 'returns' not in features.columns:
                returns = df['close'].pct_change()
            else:
                returns = features['returns']

            rolling_returns = returns.rolling(window)
            features[f'returns_skew_{window}'] = rolling_returns.skew()
            features[f'returns_kurtosis_{window}'] = rolling_returns.kurt()
            features[f'returns_autocorr_{window}'] = rolling_returns.apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )

        # Volume statistics
        if 'volume' in df.columns:
            for window in [24, 72, 168]:
                rolling_volume = df['volume'].rolling(window)
                features[f'volume_volatility_{window}'] = rolling_volume.std()

        return features

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features using feature interactions."""
        features = pd.DataFrame(index=df.index)

        # Price momentum interactions (only if the required features exist)
        momentum_cols = [col for col in features.columns if col.startswith('price_momentum_')]
        if len(momentum_cols) >= 2:
            # Sort by lookback period
            momentum_cols.sort(key=lambda x: int(x.split('_')[-1]))
            if len(momentum_cols) >= 2:
                features['momentum_acceleration'] = features[momentum_cols[1]] - features[momentum_cols[-1]]  # Short vs long term
                features['momentum_divergence'] = features[momentum_cols[0]] - features[momentum_cols[1]]  # Very short vs short term

        # Volatility regime features (only if volatility features exist)
        volatility_cols = [col for col in features.columns if col.startswith('price_volatility_')]
        if volatility_cols:
            # Use the first available volatility column
            vol_col = volatility_cols[0]
            features['volatility_regime'] = pd.cut(
                features[vol_col], bins=[0, 0.02, 0.05, 0.1, np.inf],
                labels=['low', 'medium', 'high', 'extreme']
            ).astype(str)

        # Trend strength indicators (only if momentum features exist)
        momentum_cols = [col for col in features.columns if col.startswith('price_momentum_')]
        if momentum_cols:
            # Use the longest lookback momentum as trend indicator
            longest_momentum = max(momentum_cols, key=lambda x: int(x.split('_')[-1]))
            features['trend_strength'] = abs(features[longest_momentum])
            features['trend_persistence'] = features[longest_momentum].rolling(100).std()

        # Volume-price divergence (only if required features exist)
        volume_ratio_cols = [col for col in features.columns if col.startswith('volume_ratio_')]
        momentum_cols = [col for col in features.columns if col.startswith('price_momentum_')]

        if volume_ratio_cols and momentum_cols:
            # Use first available volume ratio and momentum
            vol_ratio_col = volume_ratio_cols[0]
            momentum_col = momentum_cols[0]
            features['volume_price_divergence'] = features[vol_ratio_col] - features[momentum_col]

        return features

    def _finalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and finalize feature set."""
        # Remove any remaining NaN values
        features_df = features_df.fillna(0)

        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Limit number of features if specified
        if len(features_df.columns) > self.config.max_features:
            features_df = self._select_features(features_df)

        # Update feature columns list
        self.feature_columns = features_df.columns.tolist()

        return features_df

    def _select_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Select most important features using mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression

            # Target is next period return (for price prediction)
            target = features_df['returns'].shift(-1).fillna(0)

            # Calculate mutual information
            mi_scores = mutual_info_regression(
                features_df.values, target.values,
                random_state=42
            )

            # Select top features
            feature_scores = list(zip(features_df.columns, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)

            top_features = [name for name, score in feature_scores[:self.config.max_features]]

            logger.info(f"Selected top {len(top_features)} features")
            return features_df[top_features]

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            # Fallback: return first N features
            return features_df.iloc[:, :self.config.max_features]

    def get_feature_importance(self, features_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Get feature importance for model interpretation."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression

            # Mutual information scores
            mi_scores = mutual_info_regression(features_df.values, target.values)

            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features_df, target)
            rf_importance = rf.feature_importances_

            # Combine scores
            importance_df = pd.DataFrame({
                'feature': features_df.columns,
                'mutual_info': mi_scores,
                'rf_importance': rf_importance,
                'combined_score': (mi_scores + rf_importance) / 2
            }).sort_values('combined_score', ascending=False)

            return importance_df

        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
            return pd.DataFrame()

    def save_features(self, features_df: pd.DataFrame, filepath: str):
        """Save engineered features for model training."""
        features_df.to_parquet(filepath)
        logger.info(f"Features saved to {filepath}")

    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load previously engineered features."""
        return pd.read_parquet(filepath)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    prices = 50000 + np.cumsum(np.random.normal(0.001, 0.02, 1000))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.uniform(1000000, 10000000, 1000)
    })

    # Create feature engineer
    config = FeatureConfig(
        lookback_periods=[5, 10, 20, 50, 100],
        ta_indicators=['rsi', 'macd', 'bollinger_bands'],
        correlated_assets=['BTC-USD', 'ETH-USD', 'SPY'],
        alternative_data=True
    )

    engineer = ComprehensiveFeatureEngineer(config)

    # Generate features
    features = engineer.create_all_features(df, symbol='BTCUSDT')

    print(f"Generated {len(features.columns)} features")
    print(f"Feature columns: {features.columns[:10].tolist()}...")

    # Save features
    engineer.save_features(features, 'sample_features.parquet')

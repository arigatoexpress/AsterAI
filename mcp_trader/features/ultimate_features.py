"""
Ultimate Feature Engineering System
Advanced cross-market features with AI-generated feature discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import torch
import torch.nn as nn
from scipy import stats
from statsmodels.tsa.stattools import coint
import talib
import logging

from .engineering import FeatureEngine, FeatureConfig
from .confluence_features import ConfluenceFeatureEngine

logger = logging.getLogger(__name__)


class MarketRegime:
    """Market regime detection"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RECOVERY = "recovery"


class UltimateFeatureEngine:
    """
    Advanced feature engineering with cross-market analysis,
    regime detection, and AI-generated features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize base feature engines
        self.base_feature_engine = FeatureEngine()
        self.confluence_engine = ConfluenceFeatureEngine()
        
        # Feature discovery network
        self.feature_discovery_net = None
        self._init_feature_discovery()
        
        # Scalers for different feature types
        self.scalers = {
            'price': RobustScaler(),
            'volume': StandardScaler(),
            'technical': StandardScaler(),
            'sentiment': StandardScaler()
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        self.dynamic_features = []
        
    def generate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive feature set from multi-source data.
        
        Args:
            data: Dictionary with keys:
                - 'crypto': DataFrame with crypto OHLCV data
                - 'stocks': DataFrame with stock market data
                - 'commodities': DataFrame with commodity data
                - 'economic': DataFrame with economic indicators
                - 'sentiment': DataFrame with sentiment data
                - 'onchain': DataFrame with on-chain metrics
        
        Returns:
            Dictionary of enhanced DataFrames with all features
        """
        logger.info("Generating ultimate feature set...")
        
        features = {}
        
        # 1. Market Regime Detection
        regime_features = self._detect_market_regime(data)
        features['regime'] = regime_features
        
        # 2. Cross-Asset Momentum
        momentum_features = self._calculate_cross_asset_momentum(data)
        features.update(momentum_features)
        
        # 3. Macro Indicators
        macro_features = self._calculate_macro_indicators(data)
        features['macro'] = macro_features
        
        # 4. Market Microstructure
        if 'crypto' in data:
            microstructure = self._calculate_market_microstructure(data['crypto'])
            features['microstructure'] = microstructure
        
        # 5. Smart Money Flow
        smart_money = self._calculate_smart_money_flow(data)
        features['smart_money'] = smart_money
        
        # 6. AI-Generated Features
        ai_features = self._generate_ai_features(data)
        features['ai_generated'] = ai_features
        
        # 7. Correlation & Divergence Features
        correlation_features = self._calculate_correlations_divergences(data)
        features.update(correlation_features)
        
        # 8. Technical Confluence
        technical_features = self._calculate_technical_confluence(data)
        features['technical_confluence'] = technical_features
        
        # 9. Sentiment Integration
        if 'sentiment' in data:
            sentiment_features = self._process_sentiment_features(data['sentiment'])
            features['sentiment_processed'] = sentiment_features
        
        # 10. Risk Indicators
        risk_features = self._calculate_risk_indicators(data)
        features['risk'] = risk_features
        
        logger.info(f"Generated {sum(len(f) if isinstance(f, pd.DataFrame) else len(f.columns) for f in features.values())} total features")
        
        return features
    
    def _detect_market_regime(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Detect current market regime using multiple indicators"""
        logger.info("Detecting market regime...")
        
        regime_signals = []
        
        # Use S&P 500 as primary market indicator
        if 'stocks' in data and 'SPY' in data['stocks'].columns:
            spy = data['stocks']['SPY']
            
            # Moving average regime
            sma_20 = spy['close'].rolling(20).mean()
            sma_50 = spy['close'].rolling(50).mean()
            sma_200 = spy['close'].rolling(200).mean()
            
            # Trend strength
            trend_strength = (spy['close'] - sma_200) / sma_200
            
            # Volatility regime
            returns = spy['close'].pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(252)
            
            # Create regime features
            regime_df = pd.DataFrame(index=spy.index)
            
            # Bull market indicators
            regime_df['bull_market'] = (
                (spy['close'] > sma_20) & 
                (sma_20 > sma_50) & 
                (sma_50 > sma_200)
            ).astype(int)
            
            # Bear market indicators
            regime_df['bear_market'] = (
                (spy['close'] < sma_20) & 
                (sma_20 < sma_50) & 
                (sma_50 < sma_200)
            ).astype(int)
            
            # High volatility regime
            regime_df['high_volatility'] = (volatility > volatility.rolling(252).mean() + 2 * volatility.rolling(252).std()).astype(int)
            
            # Market crash detection (>3% daily drops)
            regime_df['crash_signal'] = (returns < -0.03).astype(int)
            
            # Recovery detection
            regime_df['recovery_signal'] = (
                (returns.rolling(5).mean() > 0.01) & 
                (regime_df['crash_signal'].rolling(20).sum() > 0)
            ).astype(int)
            
            # Regime classification
            regime_df['regime'] = MarketRegime.SIDEWAYS  # Default
            regime_df.loc[regime_df['bull_market'] == 1, 'regime'] = MarketRegime.BULL
            regime_df.loc[regime_df['bear_market'] == 1, 'regime'] = MarketRegime.BEAR
            regime_df.loc[regime_df['high_volatility'] == 1, 'regime'] = MarketRegime.VOLATILE
            regime_df.loc[regime_df['crash_signal'] == 1, 'regime'] = MarketRegime.CRASH
            regime_df.loc[regime_df['recovery_signal'] == 1, 'regime'] = MarketRegime.RECOVERY
            
            # One-hot encode regime
            for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, 
                          MarketRegime.VOLATILE, MarketRegime.CRASH, MarketRegime.RECOVERY]:
                regime_df[f'regime_{regime}'] = (regime_df['regime'] == regime).astype(int)
            
            regime_signals.append(regime_df)
        
        # Crypto market regime
        if 'crypto' in data and 'BTC' in data['crypto'].columns:
            btc = data['crypto']['BTC']
            
            # Bitcoin dominance and altcoin season detection
            btc_returns = btc['close'].pct_change()
            
            crypto_regime = pd.DataFrame(index=btc.index)
            crypto_regime['btc_bull'] = (btc_returns.rolling(30).mean() > 0.002).astype(int)
            crypto_regime['btc_bear'] = (btc_returns.rolling(30).mean() < -0.002).astype(int)
            
            # Altcoin season indicator (would need alt data)
            crypto_regime['altcoin_season'] = 0  # Placeholder
            
            regime_signals.append(crypto_regime)
        
        # Combine all regime signals
        if regime_signals:
            combined_regime = pd.concat(regime_signals, axis=1)
            return combined_regime
        
        return pd.DataFrame()
    
    def _calculate_cross_asset_momentum(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate momentum across different asset classes"""
        momentum_features = {}
        
        # BTC-S&P500 divergence
        if 'crypto' in data and 'stocks' in data:
            if 'BTC' in data['crypto'].columns and 'SPY' in data['stocks'].columns:
                btc_returns = data['crypto']['BTC']['close'].pct_change()
                spy_returns = data['stocks']['SPY']['close'].pct_change()
                
                # Align indices
                common_idx = btc_returns.index.intersection(spy_returns.index)
                btc_returns = btc_returns.loc[common_idx]
                spy_returns = spy_returns.loc[common_idx]
                
                divergence_df = pd.DataFrame(index=common_idx)
                
                # Rolling correlation
                divergence_df['btc_spy_corr'] = btc_returns.rolling(30).corr(spy_returns)
                
                # Momentum divergence
                btc_momentum = btc_returns.rolling(20).mean()
                spy_momentum = spy_returns.rolling(20).mean()
                divergence_df['momentum_divergence'] = btc_momentum - spy_momentum
                
                # Relative strength
                divergence_df['btc_spy_rs'] = (1 + btc_returns).rolling(30).apply(np.prod) / (1 + spy_returns).rolling(30).apply(np.prod)
                
                momentum_features['btc_sp500_divergence'] = divergence_df
        
        # Crypto-Equity correlation
        if 'crypto' in data and 'stocks' in data:
            correlation_df = self._calculate_rolling_correlations(
                data['crypto'], 
                data['stocks'],
                window=30
            )
            momentum_features['crypto_equity_correlation'] = correlation_df
        
        # Cross-asset momentum scores
        momentum_scores = self._calculate_momentum_scores(data)
        momentum_features['momentum_scores'] = momentum_scores
        
        return momentum_features
    
    def _calculate_macro_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate macroeconomic indicators"""
        macro_features = pd.DataFrame()
        
        if 'economic' in data:
            econ = data['economic']
            
            # Risk on/off indicator
            if 'VIX' in econ.columns:
                vix = econ['VIX']['close']
                macro_features['risk_off'] = (vix > vix.rolling(20).mean() + vix.rolling(20).std()).astype(int)
                macro_features['extreme_fear'] = (vix > 30).astype(int)
            
            # Yield curve
            if 'DGS10' in econ.columns and 'DGS2' in econ.columns:
                yield_spread = econ['DGS10']['close'] - econ['DGS2']['close']
                macro_features['yield_curve_inverted'] = (yield_spread < 0).astype(int)
                macro_features['yield_spread'] = yield_spread
            
            # Global liquidity conditions
            if 'M2SL' in econ.columns:
                m2 = econ['M2SL']['close']
                m2_growth = m2.pct_change(periods=252)  # YoY growth
                macro_features['liquidity_expanding'] = (m2_growth > 0.05).astype(int)
                macro_features['liquidity_contracting'] = (m2_growth < 0).astype(int)
            
            # Fed policy stance
            if 'DFF' in econ.columns:
                fed_rate = econ['DFF']['close']
                rate_change = fed_rate.diff()
                macro_features['fed_hiking'] = (rate_change > 0).rolling(5).sum() > 2
                macro_features['fed_cutting'] = (rate_change < 0).rolling(5).sum() > 2
                macro_features['fed_pause'] = (rate_change == 0).rolling(5).sum() > 3
            
            # Inflation regime
            if 'CPIAUCSL' in econ.columns:
                cpi = econ['CPIAUCSL']['close']
                inflation = cpi.pct_change(periods=12) * 100  # YoY inflation
                macro_features['high_inflation'] = (inflation > 3).astype(int)
                macro_features['deflation_risk'] = (inflation < 0).astype(int)
        
        return macro_features
    
    def _calculate_market_microstructure(self, crypto_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        microstructure = pd.DataFrame(index=crypto_data.index)
        
        # Order flow imbalance (using volume as proxy)
        if 'volume' in crypto_data.columns:
            volume = crypto_data['volume']
            
            # Volume momentum
            microstructure['volume_momentum'] = volume / volume.rolling(20).mean() - 1
            
            # Volume spikes
            volume_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
            microstructure['volume_spike'] = (volume_zscore > 2).astype(int)
            
            # Accumulation/Distribution
            high = crypto_data.get('high', crypto_data['close'])
            low = crypto_data.get('low', crypto_data['close'])
            close = crypto_data['close']
            
            clv = ((close - low) - (high - close)) / (high - low + 1e-10)
            microstructure['accumulation'] = (clv * volume).rolling(20).sum()
        
        # Price efficiency (using returns autocorrelation)
        returns = crypto_data['close'].pct_change()
        microstructure['return_autocorr'] = returns.rolling(30).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 2 else 0
        )
        
        # Microstructure noise (high-frequency volatility)
        microstructure['hf_volatility'] = returns.rolling(5).std() * np.sqrt(252 * 24)  # Hourly to annual
        
        return microstructure
    
    def _calculate_smart_money_flow(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate smart money flow indicators"""
        smart_money = pd.DataFrame()
        
        # Whale accumulation index (using on-chain data if available)
        if 'onchain' in data:
            onchain = data['onchain']
            
            if 'btc_transactions' in onchain.columns:
                # Large transaction indicator
                tx_volume = onchain['btc_transactions']
                tx_ma = tx_volume.rolling(7).mean()
                smart_money['whale_activity'] = (tx_volume > tx_ma * 1.5).astype(int)
            
            if 'btc_hashrate' in onchain.columns:
                # Miner confidence (hashrate trend)
                hashrate = onchain['btc_hashrate']
                hashrate_ma = hashrate.rolling(30).mean()
                smart_money['miner_confidence'] = (hashrate > hashrate_ma).astype(int)
        
        # Institutional flow indicators (using traditional markets)
        if 'stocks' in data:
            # Use sector rotation as smart money indicator
            sectors = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU']
            available_sectors = [s for s in sectors if s in data['stocks'].columns]
            
            if available_sectors:
                sector_returns = pd.DataFrame({
                    s: data['stocks'][s]['close'].pct_change()
                    for s in available_sectors
                })
                
                # Sector momentum
                sector_momentum = sector_returns.rolling(20).mean()
                
                # Risk-on sectors (Tech, Consumer Discretionary)
                if 'XLK' in sector_momentum.columns and 'XLY' in sector_momentum.columns:
                    smart_money['risk_on_flow'] = (
                        sector_momentum[['XLK', 'XLY']].mean(axis=1) > 
                        sector_momentum.mean(axis=1)
                    ).astype(int)
                
                # Defensive sectors (Utilities, Consumer Staples)
                if 'XLU' in sector_momentum.columns and 'XLP' in sector_momentum.columns:
                    smart_money['defensive_flow'] = (
                        sector_momentum[['XLU', 'XLP']].mean(axis=1) > 
                        sector_momentum.mean(axis=1)
                    ).astype(int)
        
        return smart_money
    
    def _init_feature_discovery(self):
        """Initialize neural network for feature discovery"""
        
        class FeatureDiscoveryNet(nn.Module):
            def __init__(self, input_dim: int = 50, hidden_dim: int = 128, output_dim: int = 20):
                super().__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
                
                # Decoder (for autoencoder training)
                self.decoder = nn.Sequential(
                    nn.Linear(output_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        self.feature_discovery_net = FeatureDiscoveryNet()
        
        # Initialize with pretrained weights if available
        # self.feature_discovery_net.load_state_dict(torch.load('models/feature_discovery.pth'))
    
    def _generate_ai_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate features using neural network"""
        ai_features = pd.DataFrame()
        
        # Prepare input data
        combined_data = []
        
        # Combine key features from different sources
        if 'crypto' in data and 'BTC' in data['crypto'].columns:
            btc_features = self._extract_basic_features(data['crypto']['BTC'])
            combined_data.append(btc_features)
        
        if 'stocks' in data and 'SPY' in data['stocks'].columns:
            spy_features = self._extract_basic_features(data['stocks']['SPY'])
            combined_data.append(spy_features)
        
        if combined_data:
            # Align and combine
            combined_df = pd.concat(combined_data, axis=1).fillna(method='ffill').dropna()
            
            # Convert to tensor
            if len(combined_df) > 0:
                X = torch.FloatTensor(combined_df.values)
                
                # Generate features
                self.feature_discovery_net.eval()
                with torch.no_grad():
                    encoded_features, _ = self.feature_discovery_net(X)
                
                # Convert back to DataFrame
                feature_names = [f'ai_feature_{i}' for i in range(encoded_features.shape[1])]
                ai_features = pd.DataFrame(
                    encoded_features.numpy(),
                    index=combined_df.index,
                    columns=feature_names
                )
        
        return ai_features
    
    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features for AI processing"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        
        # Price features
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Moving averages
        for period in [5, 10, 20]:
            features[f'sma_{period}'] = close.rolling(period).mean() / close - 1
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # RSI
        features['rsi'] = self._calculate_rsi(close, 14)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_correlations_divergences(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate correlation and divergence features"""
        features = {}
        
        # Inter-market correlations
        correlation_pairs = [
            ('crypto', 'BTC', 'stocks', 'SPY'),
            ('crypto', 'ETH', 'crypto', 'BTC'),
            ('stocks', 'QQQ', 'stocks', 'SPY'),
            ('commodities', 'GLD', 'stocks', 'SPY'),
            ('commodities', 'GLD', 'crypto', 'BTC')
        ]
        
        correlation_features = []
        
        for market1, asset1, market2, asset2 in correlation_pairs:
            if market1 in data and market2 in data:
                if asset1 in data[market1].columns and asset2 in data[market2].columns:
                    series1 = data[market1][asset1]['close'].pct_change()
                    series2 = data[market2][asset2]['close'].pct_change()
                    
                    # Align indices
                    common_idx = series1.index.intersection(series2.index)
                    if len(common_idx) > 0:
                        series1 = series1.loc[common_idx]
                        series2 = series2.loc[common_idx]
                        
                        corr_df = pd.DataFrame(index=common_idx)
                        
                        # Rolling correlation
                        for window in [10, 30, 60]:
                            corr_df[f'{asset1}_{asset2}_corr_{window}'] = series1.rolling(window).corr(series2)
                        
                        # Divergence indicator
                        corr_df[f'{asset1}_{asset2}_divergence'] = (
                            series1.rolling(20).mean() - series2.rolling(20).mean()
                        )
                        
                        correlation_features.append(corr_df)
        
        if correlation_features:
            features['correlations'] = pd.concat(correlation_features, axis=1)
        
        return features
    
    def _calculate_technical_confluence(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate technical indicator confluence"""
        confluence_features = pd.DataFrame()
        
        # Focus on major assets
        key_assets = [
            ('crypto', 'BTC'),
            ('stocks', 'SPY'),
            ('commodities', 'GLD')
        ]
        
        for market, asset in key_assets:
            if market in data and asset in data[market].columns:
                asset_data = data[market][asset]
                
                # Calculate multiple technical indicators
                close = asset_data['close']
                high = asset_data.get('high', close)
                low = asset_data.get('low', close)
                volume = asset_data.get('volume', pd.Series(1, index=close.index))
                
                # Trend indicators
                sma_20 = close.rolling(20).mean()
                sma_50 = close.rolling(50).mean()
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                
                # Momentum indicators
                rsi = self._calculate_rsi(close, 14)
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9).mean()
                
                # Volatility indicators
                bb_middle = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                bb_upper = bb_middle + 2 * bb_std
                bb_lower = bb_middle - 2 * bb_std
                
                # Confluence scoring
                prefix = f'{asset}_'
                
                # Trend confluence
                confluence_features[f'{prefix}trend_confluence'] = (
                    (close > sma_20).astype(int) +
                    (close > sma_50).astype(int) +
                    (sma_20 > sma_50).astype(int) +
                    (ema_12 > ema_26).astype(int)
                ) / 4
                
                # Momentum confluence
                confluence_features[f'{prefix}momentum_confluence'] = (
                    (rsi > 50).astype(int) +
                    (rsi < 70).astype(int) +
                    (macd > 0).astype(int) +
                    (macd > macd_signal).astype(int)
                ) / 4
                
                # Oversold/Overbought confluence
                confluence_features[f'{prefix}oversold_confluence'] = (
                    (rsi < 30).astype(int) +
                    (close < bb_lower).astype(int) +
                    ((close - bb_lower) / (bb_upper - bb_lower) < 0.2).astype(int)
                ) / 3
                
                confluence_features[f'{prefix}overbought_confluence'] = (
                    (rsi > 70).astype(int) +
                    (close > bb_upper).astype(int) +
                    ((close - bb_lower) / (bb_upper - bb_lower) > 0.8).astype(int)
                ) / 3
        
        return confluence_features
    
    def _process_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Process and enhance sentiment features"""
        processed = pd.DataFrame(index=sentiment_data.index)
        
        if 'polarity' in sentiment_data.columns:
            # Sentiment momentum
            processed['sentiment_momentum'] = sentiment_data['polarity'].rolling(24).mean()
            
            # Sentiment volatility
            processed['sentiment_volatility'] = sentiment_data['polarity'].rolling(24).std()
            
            # Extreme sentiment
            sentiment_zscore = (
                (sentiment_data['polarity'] - sentiment_data['polarity'].rolling(168).mean()) /
                sentiment_data['polarity'].rolling(168).std()
            )
            processed['extreme_positive_sentiment'] = (sentiment_zscore > 2).astype(int)
            processed['extreme_negative_sentiment'] = (sentiment_zscore < -2).astype(int)
            
            # Sentiment divergence from price
            # This would need price data aligned with sentiment
        
        if 'fear_greed' in sentiment_data.columns:
            # Fear & Greed extremes
            processed['extreme_fear'] = (sentiment_data['fear_greed'] < 20).astype(int)
            processed['extreme_greed'] = (sentiment_data['fear_greed'] > 80).astype(int)
            
            # Fear & Greed momentum
            processed['fear_greed_momentum'] = sentiment_data['fear_greed'].diff(7)
        
        return processed
    
    def _calculate_risk_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate comprehensive risk indicators"""
        risk_features = pd.DataFrame()
        
        # Correlation risk (high correlation = systemic risk)
        if 'crypto' in data and 'stocks' in data:
            correlations = []
            
            # Calculate pairwise correlations
            crypto_assets = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
            stock_assets = ['SPY', 'QQQ', 'IWM']
            
            for crypto in crypto_assets:
                for stock in stock_assets:
                    if crypto in data['crypto'].columns and stock in data['stocks'].columns:
                        corr = (
                            data['crypto'][crypto]['close'].pct_change()
                            .rolling(30)
                            .corr(data['stocks'][stock]['close'].pct_change())
                        )
                        correlations.append(corr)
            
            if correlations:
                # Average correlation as systemic risk indicator
                avg_correlation = pd.concat(correlations, axis=1).mean(axis=1)
                risk_features['systemic_risk'] = avg_correlation
                risk_features['high_correlation_risk'] = (avg_correlation > 0.7).astype(int)
        
        # Volatility clustering
        if 'crypto' in data and 'BTC' in data['crypto'].columns:
            btc_returns = data['crypto']['BTC']['close'].pct_change()
            btc_vol = btc_returns.rolling(20).std() * np.sqrt(365 * 24)  # Annualized
            
            # GARCH-like volatility clustering
            risk_features['vol_clustering'] = btc_vol.rolling(20).apply(lambda x: x.autocorr() if len(x) > 1 else 0)
            
            # Volatility regime
            vol_percentile = btc_vol.rolling(252).rank(pct=True)
            risk_features['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            risk_features['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
        
        # Drawdown risk
        for market, asset in [('crypto', 'BTC'), ('stocks', 'SPY')]:
            if market in data and asset in data[market].columns:
                close = data[market][asset]['close']
                
                # Calculate drawdown
                rolling_max = close.expanding().max()
                drawdown = (close - rolling_max) / rolling_max
                
                risk_features[f'{asset}_drawdown'] = drawdown
                risk_features[f'{asset}_max_drawdown_20d'] = drawdown.rolling(20).min()
                risk_features[f'{asset}_severe_drawdown'] = (drawdown < -0.2).astype(int)
        
        # Liquidity risk (using volume)
        if 'crypto' in data and 'BTC' in data['crypto'].columns:
            if 'volume' in data['crypto']['BTC'].columns:
                volume = data['crypto']['BTC']['volume']
                volume_ma = volume.rolling(30).mean()
                
                risk_features['low_liquidity'] = (volume < volume_ma * 0.5).astype(int)
                risk_features['liquidity_shock'] = (volume < volume_ma * 0.3).astype(int)
        
        return risk_features
    
    def _calculate_rolling_correlations(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                                       window: int = 30) -> pd.DataFrame:
        """Calculate rolling correlations between two datasets"""
        correlations = pd.DataFrame()
        
        # Get common assets
        assets1 = [col for col in data1.columns if isinstance(data1[col], pd.DataFrame) and 'close' in data1[col].columns]
        assets2 = [col for col in data2.columns if isinstance(data2[col], pd.DataFrame) and 'close' in data2[col].columns]
        
        for asset1 in assets1[:5]:  # Limit to top 5 for performance
            for asset2 in assets2[:5]:
                if asset1 != asset2:
                    series1 = data1[asset1]['close'].pct_change()
                    series2 = data2[asset2]['close'].pct_change()
                    
                    # Align indices
                    common_idx = series1.index.intersection(series2.index)
                    if len(common_idx) > window:
                        series1 = series1.loc[common_idx]
                        series2 = series2.loc[common_idx]
                        
                        corr = series1.rolling(window).corr(series2)
                        correlations[f'{asset1}_{asset2}_corr'] = corr
        
        return correlations
    
    def _calculate_momentum_scores(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate momentum scores across assets"""
        momentum_scores = pd.DataFrame()
        
        # Define assets to track
        assets_to_track = [
            ('crypto', ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']),
            ('stocks', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']),
            ('commodities', ['GLD', 'SLV', 'USO', 'UNG'])
        ]
        
        all_momentum = {}
        
        for market, assets in assets_to_track:
            if market in data:
                for asset in assets:
                    if asset in data[market].columns:
                        close = data[market][asset]['close']
                        
                        # Multi-timeframe momentum
                        momentum_1w = close.pct_change(periods=5)
                        momentum_1m = close.pct_change(periods=20)
                        momentum_3m = close.pct_change(periods=60)
                        
                        # Composite momentum score
                        composite_momentum = (
                            momentum_1w * 0.5 +
                            momentum_1m * 0.3 +
                            momentum_3m * 0.2
                        )
                        
                        all_momentum[f'{market}_{asset}'] = composite_momentum
        
        if all_momentum:
            # Create DataFrame
            momentum_df = pd.DataFrame(all_momentum)
            
            # Rank momentum across assets
            momentum_scores['top_momentum_rank'] = momentum_df.rank(axis=1, pct=True).max(axis=1)
            momentum_scores['bottom_momentum_rank'] = momentum_df.rank(axis=1, pct=True).min(axis=1)
            
            # Momentum dispersion (high = rotation)
            momentum_scores['momentum_dispersion'] = momentum_df.std(axis=1)
            
            # Sector/asset rotation signals
            momentum_scores['rotation_signal'] = momentum_scores['momentum_dispersion'].rolling(20).mean()
        
        return momentum_scores
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       method: str = 'mutual_info', k: int = 50) -> List[str]:
        """Select most important features"""
        
        # Remove non-numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_features = numeric_features.fillna(method='ffill').fillna(0)
        
        if method == 'mutual_info':
            # Convert target to classification
            target_class = pd.qcut(target, q=3, labels=[0, 1, 2])
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        # Fit selector
        selector.fit(numeric_features, target_class)
        
        # Get selected features
        selected_features = numeric_features.columns[selector.get_support()].tolist()
        
        # Store feature importance
        self.feature_importance = dict(zip(
            numeric_features.columns,
            selector.scores_
        ))
        
        return selected_features
    
    def create_interaction_features(self, features: pd.DataFrame, 
                                  max_interactions: int = 20) -> pd.DataFrame:
        """Create interaction features between most important features"""
        
        # Get top features by importance
        if self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            top_feature_names = [f[0] for f in top_features]
        else:
            # Use variance as proxy for importance
            variances = features.var()
            top_feature_names = variances.nlargest(10).index.tolist()
        
        interaction_features = pd.DataFrame(index=features.index)
        count = 0
        
        # Create multiplicative interactions
        for i, feat1 in enumerate(top_feature_names):
            for feat2 in top_feature_names[i+1:]:
                if count >= max_interactions:
                    break
                
                if feat1 in features.columns and feat2 in features.columns:
                    interaction_features[f'{feat1}_X_{feat2}'] = features[feat1] * features[feat2]
                    count += 1
        
        # Create ratio features
        for i, feat1 in enumerate(top_feature_names[:5]):
            for feat2 in top_feature_names[i+1:6]:
                if count >= max_interactions:
                    break
                
                if feat1 in features.columns and feat2 in features.columns:
                    # Avoid division by zero
                    denominator = features[feat2].replace(0, np.nan)
                    interaction_features[f'{feat1}_DIV_{feat2}'] = features[feat1] / denominator
                    count += 1
        
        return interaction_features
    
    def generate_temporal_features(self, features: pd.DataFrame, 
                                 lookback_periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Generate temporal features with various lookback periods"""
        temporal_features = pd.DataFrame(index=features.index)
        
        # Select numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:20]:  # Limit to prevent explosion
            for period in lookback_periods:
                # Lagged values
                temporal_features[f'{col}_lag_{period}'] = features[col].shift(period)
                
                # Rolling statistics
                temporal_features[f'{col}_roll_mean_{period}'] = features[col].rolling(period).mean()
                temporal_features[f'{col}_roll_std_{period}'] = features[col].rolling(period).std()
                
                # Rate of change
                temporal_features[f'{col}_roc_{period}'] = features[col].pct_change(period)
        
        return temporal_features



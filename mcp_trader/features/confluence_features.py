"""
Confluence Feature Engineering for Multi-Asset Trading

Identifies confluence signals across multiple assets and technical indicators.
Detects coordinated movements, correlations, and alignment patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceConfig:
    """Configuration for confluence feature generation."""
    
    # Correlation windows
    correlation_windows: List[int] = None
    min_correlation: float = 0.5
    
    # Volume confluence
    volume_spike_threshold: float = 2.0  # x times average
    min_assets_for_confluence: int = 2
    
    # Technical indicator alignment
    rsi_alignment_threshold: float = 10.0  # RSI difference
    macd_crossover_window: int = 3  # bars
    bb_squeeze_threshold: float = 0.8  # % of mean
    
    # Momentum alignment
    momentum_window: int = 14
    trend_agreement_threshold: float = 0.6  # 60% of assets
    
    def __post_init__(self):
        if self.correlation_windows is None:
            self.correlation_windows = [24, 168, 720]  # 1d, 1w, 1m in hours


class ConfluenceFeatureEngine:
    """
    Generate confluence signals across multiple assets.
    
    Features:
    - Cross-asset price correlations
    - Simultaneous volume spikes
    - Technical indicator alignment
    - Momentum confluence
    - Support/resistance level alignment
    """
    
    def __init__(self, config: ConfluenceConfig = None):
        self.config = config or ConfluenceConfig()
        logger.info(f"🔗 Confluence Feature Engine initialized")
    
    def generate_all_features(self, 
                             asset_data: Dict[str, pd.DataFrame],
                             reference_asset: str = "BTCUSDT") -> Dict[str, pd.DataFrame]:
        """
        Generate all confluence features for each asset.
        
        Args:
            asset_data: Dictionary of {symbol: DataFrame} with OHLCV data
            reference_asset: Primary reference asset (usually BTC)
            
        Returns:
            Dictionary of {symbol: DataFrame} with confluence features added
        """
        logger.info(f"🔗 Generating confluence features for {len(asset_data)} assets")
        
        # Calculate cross-asset correlations
        correlations = self.calculate_cross_asset_correlation(asset_data)
        
        # Detect volume confluence
        volume_confluence = self.detect_volume_confluence(asset_data)
        
        # Detect indicator confluence
        indicator_confluence = self.detect_indicator_confluence(asset_data)
        
        # Calculate momentum alignment
        momentum_alignment = self.calculate_momentum_alignment(asset_data)
        
        # Merge features into each asset's DataFrame
        enriched_data = {}
        for symbol, df in asset_data.items():
            enriched_df = df.copy()
            
            # Add correlation features
            if symbol in correlations:
                for col, values in correlations[symbol].items():
                    enriched_df[f'confluence_{col}'] = values
            
            # Add volume confluence
            if symbol in volume_confluence:
                enriched_df['confluence_volume_spike'] = volume_confluence[symbol]
            
            # Add indicator confluence
            if symbol in indicator_confluence:
                for col, values in indicator_confluence[symbol].items():
                    enriched_df[f'confluence_{col}'] = values
            
            # Add momentum alignment
            if symbol in momentum_alignment:
                for col, values in momentum_alignment[symbol].items():
                    enriched_df[f'confluence_{col}'] = values
            
            enriched_data[symbol] = enriched_df
        
        logger.info(f"✅ Confluence features generated")
        return enriched_data
    
    def calculate_cross_asset_correlation(self, 
                                         asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate rolling correlations between assets.
        
        Returns:
            Dict of {symbol: {correlation_feature: Series}}
        """
        logger.info("   📊 Calculating cross-asset correlations...")
        
        correlations = {}
        symbols = list(asset_data.keys())
        
        # Get price series for all assets
        price_series = {
            symbol: df['close'] for symbol, df in asset_data.items()
        }
        
        # Align all series to common index
        aligned_prices = pd.DataFrame(price_series)
        aligned_prices = aligned_prices.fillna(method='ffill').fillna(method='bfill')
        
        for symbol in symbols:
            correlations[symbol] = {}
            
            # Calculate correlations with other major assets
            for other_symbol in symbols:
                if symbol == other_symbol:
                    continue
                
                for window in self.config.correlation_windows:
                    # Rolling correlation
                    corr = aligned_prices[symbol].rolling(window).corr(aligned_prices[other_symbol])
                    corr_name = f'corr_{other_symbol}_{window}h'
                    correlations[symbol][corr_name] = corr
            
            # Calculate average correlation (market-wide alignment)
            all_corrs = []
            for other_symbol in symbols:
                if symbol != other_symbol:
                    corr = aligned_prices[symbol].rolling(24).corr(aligned_prices[other_symbol])
                    all_corrs.append(corr)
            
            if all_corrs:
                correlations[symbol]['avg_correlation'] = pd.concat(all_corrs, axis=1).mean(axis=1)
                correlations[symbol]['correlation_strength'] = pd.concat(all_corrs, axis=1).abs().mean(axis=1)
        
        return correlations
    
    def detect_volume_confluence(self, 
                                asset_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Detect simultaneous volume spikes across multiple assets.
        
        Returns:
            Dict of {symbol: Series} with volume confluence scores
        """
        logger.info("   📊 Detecting volume confluence...")
        
        volume_confluence = {}
        
        # Calculate volume spikes for each asset
        volume_spikes = {}
        for symbol, df in asset_data.items():
            volume_sma = df['volume'].rolling(20).mean()
            is_spike = (df['volume'] > volume_sma * self.config.volume_spike_threshold).astype(int)
            volume_spikes[symbol] = is_spike
        
        # Align all spike series
        aligned_spikes = pd.DataFrame(volume_spikes)
        aligned_spikes = aligned_spikes.fillna(0)
        
        # Count simultaneous spikes
        for symbol in asset_data.keys():
            # Sum of other assets spiking at same time
            other_spikes = aligned_spikes.drop(columns=[symbol]).sum(axis=1)
            
            # Confluence score: how many other assets are spiking
            confluence_score = other_spikes / (len(asset_data) - 1)
            
            # Only consider it confluence if minimum threshold met
            is_confluence = (other_spikes >= self.config.min_assets_for_confluence).astype(float)
            
            volume_confluence[symbol] = is_confluence * confluence_score
        
        return volume_confluence
    
    def detect_indicator_confluence(self, 
                                   asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
        """
        Detect alignment in technical indicators across assets.
        
        Returns:
            Dict of {symbol: {indicator_feature: Series}}
        """
        logger.info("   📊 Detecting indicator confluence...")
        
        indicator_confluence = {}
        
        # Calculate technical indicators for all assets if not present
        enriched_data = {}
        for symbol, df in asset_data.items():
            enriched_df = df.copy()
            
            # Calculate RSI if not present
            if 'rsi' not in enriched_df.columns:
                enriched_df['rsi'] = self._calculate_rsi(enriched_df['close'])
            
            # Calculate MACD if not present
            if 'macd' not in enriched_df.columns:
                macd, signal, _ = self._calculate_macd(enriched_df['close'])
                enriched_df['macd'] = macd
                enriched_df['macd_signal'] = signal
            
            # Calculate Bollinger Bands if not present
            if 'bb_width' not in enriched_df.columns:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(enriched_df['close'])
                enriched_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            enriched_data[symbol] = enriched_df
        
        # Detect RSI alignment (overbought/oversold confluence)
        for symbol, df in enriched_data.items():
            indicator_confluence[symbol] = {}
            
            # RSI confluence
            rsi_values = {sym: data['rsi'] for sym, data in enriched_data.items()}
            aligned_rsi = pd.DataFrame(rsi_values).fillna(method='ffill')
            
            # Count assets in overbought/oversold
            overbought_count = (aligned_rsi > 70).sum(axis=1)
            oversold_count = (aligned_rsi < 30).sum(axis=1)
            
            indicator_confluence[symbol]['rsi_overbought_confluence'] = overbought_count / len(enriched_data)
            indicator_confluence[symbol]['rsi_oversold_confluence'] = oversold_count / len(enriched_data)
            
            # MACD crossover confluence
            macd_bullish = {}
            for sym, data in enriched_data.items():
                macd_bullish[sym] = (data['macd'] > data['macd_signal']).astype(int)
            
            aligned_macd = pd.DataFrame(macd_bullish).fillna(0)
            bullish_count = aligned_macd.sum(axis=1)
            
            indicator_confluence[symbol]['macd_bullish_confluence'] = bullish_count / len(enriched_data)
            
            # Bollinger Band squeeze confluence
            bb_squeeze = {}
            for sym, data in enriched_data.items():
                squeeze_threshold = data['bb_width'].rolling(20).mean() * self.config.bb_squeeze_threshold
                bb_squeeze[sym] = (data['bb_width'] < squeeze_threshold).astype(int)
            
            aligned_squeeze = pd.DataFrame(bb_squeeze).fillna(0)
            squeeze_count = aligned_squeeze.sum(axis=1)
            
            indicator_confluence[symbol]['bb_squeeze_confluence'] = squeeze_count / len(enriched_data)
        
        return indicator_confluence
    
    def calculate_momentum_alignment(self, 
                                    asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate momentum alignment across assets.
        
        Returns:
            Dict of {symbol: {momentum_feature: Series}}
        """
        logger.info("   📊 Calculating momentum alignment...")
        
        momentum_alignment = {}
        
        # Calculate returns for all assets
        returns = {}
        for symbol, df in asset_data.items():
            returns[symbol] = df['close'].pct_change(self.config.momentum_window)
        
        aligned_returns = pd.DataFrame(returns).fillna(0)
        
        for symbol in asset_data.keys():
            momentum_alignment[symbol] = {}
            
            # Directional alignment (how many assets moving in same direction)
            symbol_direction = np.sign(aligned_returns[symbol])
            other_directions = aligned_returns.drop(columns=[symbol]).apply(np.sign)
            
            # Count assets moving in same direction
            same_direction = (other_directions.T == symbol_direction).T.sum(axis=1)
            momentum_alignment[symbol]['directional_alignment'] = same_direction / (len(asset_data) - 1)
            
            # Momentum strength alignment
            symbol_momentum = aligned_returns[symbol].abs()
            other_momentum = aligned_returns.drop(columns=[symbol]).abs().mean(axis=1)
            
            # Relative momentum strength
            momentum_alignment[symbol]['relative_momentum'] = symbol_momentum / (other_momentum + 1e-8)
            
            # Trend strength (percentage of assets in strong trend)
            strong_trend = (aligned_returns.abs() > 0.02).sum(axis=1)  # > 2% move
            momentum_alignment[symbol]['trend_strength'] = strong_trend / len(asset_data)
        
        return momentum_alignment
    
    # Helper methods for technical indicators
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, 
                                   period: int = 20, 
                                   std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    def get_confluence_score(self, 
                            features_df: pd.DataFrame, 
                            symbol: str) -> pd.Series:
        """
        Calculate overall confluence score for trading signals.
        
        Args:
            features_df: DataFrame with all confluence features
            symbol: Asset symbol
            
        Returns:
            Series with confluence scores (0-1)
        """
        confluence_cols = [col for col in features_df.columns if col.startswith('confluence_')]
        
        if not confluence_cols:
            return pd.Series(0, index=features_df.index)
        
        # Average of all confluence features
        confluence_score = features_df[confluence_cols].mean(axis=1)
        
        # Normalize to 0-1 range
        confluence_score = (confluence_score - confluence_score.min()) / (
            confluence_score.max() - confluence_score.min() + 1e-8
        )
        
        return confluence_score





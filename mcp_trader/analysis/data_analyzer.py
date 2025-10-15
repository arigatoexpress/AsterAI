"""
Comprehensive Data Analysis Module
Provides statistical analysis, correlations, and insights for trading data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    name: str
    value: Any
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'metadata': self.metadata
        }


class DataAnalyzer:
    """
    Comprehensive data analysis for trading systems
    - Statistical summaries
    - Correlation analysis
    - Pattern detection
    - Feature importance
    - Market regime detection
    """
    
    def __init__(self):
        logger.info("DataAnalyzer initialized")
    
    def basic_statistics(self, df: pd.DataFrame, price_col: str = 'price') -> Dict[str, Any]:
        """Calculate basic statistical measures"""
        if price_col not in df.columns:
            logger.error(f"Column {price_col} not found")
            return {}
        
        prices = df[price_col].dropna()
        
        if len(prices) == 0:
            return {}
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        stats_dict = {
            'count': len(prices),
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'range': float(prices.max() - prices.min()),
            
            # Returns
            'mean_return': float(returns.mean()),
            'std_return': float(returns.std()),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            
            # Log returns
            'mean_log_return': float(log_returns.mean()),
            'std_log_return': float(log_returns.std()),
            
            # Distribution
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            
            # Percentiles
            'p25': float(prices.quantile(0.25)),
            'p50': float(prices.quantile(0.50)),
            'p75': float(prices.quantile(0.75)),
            'p95': float(prices.quantile(0.95)),
            'p99': float(prices.quantile(0.99)),
        }
        
        # Add volume stats if available
        if 'volume' in df.columns:
            volume = df['volume'].dropna()
            stats_dict['volume_mean'] = float(volume.mean())
            stats_dict['volume_std'] = float(volume.std())
            stats_dict['volume_median'] = float(volume.median())
        
        return stats_dict
    
    def calculate_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20
    ) -> pd.Series:
        """Calculate rolling volatility"""
        returns = df[price_col].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def correlation_matrix(
        self,
        data_dict: Dict[str, pd.DataFrame],
        price_col: str = 'price',
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """Calculate correlation matrix across multiple assets"""
        # Align all data by timestamp
        aligned_data = {}
        
        for name, df in data_dict.items():
            if price_col in df.columns and 'timestamp' in df.columns:
                temp = df[['timestamp', price_col]].copy()
                temp = temp.set_index('timestamp')
                temp.columns = [name]
                aligned_data[name] = temp
        
        if not aligned_data:
            logger.warning("No valid data for correlation analysis")
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(aligned_data.values(), axis=1, join='outer')
        
        # Calculate returns
        returns = combined.pct_change().dropna()
        
        # Calculate correlation
        if method == 'pearson':
            corr = returns.corr(method='pearson')
        elif method == 'spearman':
            corr = returns.corr(method='spearman')
        else:
            corr = returns.corr()
        
        return corr
    
    def find_correlations(
        self,
        data_dict: Dict[str, pd.DataFrame],
        threshold: float = 0.7,
        price_col: str = 'price'
    ) -> List[Tuple[str, str, float]]:
        """Find highly correlated asset pairs"""
        corr_matrix = self.correlation_matrix(data_dict, price_col=price_col)
        
        if corr_matrix.empty:
            return []
        
        # Find pairs above threshold
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    correlations.append((asset1, asset2, float(corr_value)))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return correlations
    
    def detect_regime_changes(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 50
    ) -> pd.DataFrame:
        """Detect market regime changes (bull/bear/sideways)"""
        if price_col not in df.columns:
            return df
        
        df = df.copy()
        
        # Calculate returns and volatility
        df['returns'] = df[price_col].pct_change()
        df['volatility'] = df['returns'].rolling(window=window).std()
        df['trend'] = df[price_col].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        # Classify regime
        def classify_regime(row):
            if pd.isna(row['trend']) or pd.isna(row['volatility']):
                return 'unknown'
            
            # Normalize trend by price
            trend_normalized = row['trend'] / row[price_col] if row[price_col] != 0 else 0
            vol_threshold = df['volatility'].median()
            
            if trend_normalized > 0.001:  # Positive trend
                if row['volatility'] > vol_threshold:
                    return 'bull_volatile'
                else:
                    return 'bull_stable'
            elif trend_normalized < -0.001:  # Negative trend
                if row['volatility'] > vol_threshold:
                    return 'bear_volatile'
                else:
                    return 'bear_decline'
            else:  # Sideways
                if row['volatility'] > vol_threshold:
                    return 'sideways_volatile'
                else:
                    return 'sideways_stable'
        
        df['regime'] = df.apply(classify_regime, axis=1)
        
        return df
    
    def calculate_drawdown(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """Calculate drawdown metrics"""
        df = df.copy()
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df[price_col].pct_change()).cumprod()
        df['running_max'] = df['cumulative_returns'].cummax()
        df['drawdown'] = (df['cumulative_returns'] - df['running_max']) / df['running_max']
        
        return df
    
    def find_support_resistance(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 20,
        num_levels: int = 5
    ) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        if price_col not in df.columns or len(df) < window:
            return {'support': [], 'resistance': []}
        
        prices = df[price_col].values
        
        # Find local minima (support)
        support_indices = []
        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i-window:i+window+1]):
                support_indices.append(i)
        
        # Find local maxima (resistance)
        resistance_indices = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                resistance_indices.append(i)
        
        # Cluster nearby levels
        def cluster_levels(indices, num_clusters):
            if len(indices) == 0:
                return []
            
            levels = prices[indices]
            
            # Simple clustering by sorting and grouping
            sorted_levels = sorted(levels)
            clusters = []
            
            if len(sorted_levels) <= num_clusters:
                return sorted_levels.tolist()
            
            # Divide into clusters
            step = len(sorted_levels) // num_clusters
            for i in range(0, len(sorted_levels), step):
                cluster = sorted_levels[i:i+step]
                if cluster:
                    clusters.append(float(np.median(cluster)))
            
            return clusters[:num_clusters]
        
        support_levels = cluster_levels(support_indices, num_levels)
        resistance_levels = cluster_levels(resistance_indices, num_levels)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def calculate_momentum_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'price'
    ) -> pd.DataFrame:
        """Calculate various momentum indicators"""
        df = df.copy()
        
        # Simple momentum
        df['momentum_5'] = df[price_col].pct_change(periods=5)
        df['momentum_10'] = df[price_col].pct_change(periods=10)
        df['momentum_20'] = df[price_col].pct_change(periods=20)
        
        # Rate of change
        df['roc_10'] = ((df[price_col] - df[price_col].shift(10)) / df[price_col].shift(10)) * 100
        
        # Moving averages
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        df['sma_200'] = df[price_col].rolling(window=200).mean()
        
        # EMA
        df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def analyze_returns_distribution(
        self,
        df: pd.DataFrame,
        price_col: str = 'price'
    ) -> Dict[str, Any]:
        """Analyze the distribution of returns"""
        returns = df[price_col].pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Test for normality
        _, p_value_norm = stats.normaltest(returns)
        
        # Calculate tail risk
        var_95 = float(returns.quantile(0.05))  # 5% VaR
        var_99 = float(returns.quantile(0.01))  # 1% VaR
        cvar_95 = float(returns[returns <= var_95].mean())  # CVaR (Expected Shortfall)
        
        analysis = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'is_normal': bool(p_value_norm > 0.05),
            'p_value_normality': float(p_value_norm),
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'positive_returns_pct': float((returns > 0).sum() / len(returns) * 100),
            'max_gain': float(returns.max()),
            'max_loss': float(returns.min()),
        }
        
        return analysis
    
    def calculate_rolling_metrics(
        self,
        df: pd.DataFrame,
        price_col: str = 'price',
        window: int = 30
    ) -> pd.DataFrame:
        """Calculate rolling window metrics"""
        df = df.copy()
        returns = df[price_col].pct_change()
        
        df['rolling_return'] = returns.rolling(window=window).mean()
        df['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        df['rolling_sharpe'] = df['rolling_return'] / df['rolling_volatility'] * np.sqrt(252)
        df['rolling_max'] = df[price_col].rolling(window=window).max()
        df['rolling_min'] = df[price_col].rolling(window=window).min()
        df['rolling_range'] = df['rolling_max'] - df['rolling_min']
        
        return df
    
    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        asset_name: str,
        price_col: str = 'price'
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report for an asset"""
        logger.info(f"Generating comprehensive report for {asset_name}")
        
        report = {
            'asset': asset_name,
            'period': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
                'days': len(df)
            },
            'statistics': self.basic_statistics(df, price_col),
            'returns_distribution': self.analyze_returns_distribution(df, price_col),
            'support_resistance': self.find_support_resistance(df, price_col),
        }
        
        # Add regime analysis
        df_with_regime = self.detect_regime_changes(df, price_col)
        if 'regime' in df_with_regime.columns:
            regime_counts = df_with_regime['regime'].value_counts().to_dict()
            report['regime_distribution'] = regime_counts
        
        # Add drawdown analysis
        df_with_dd = self.calculate_drawdown(df, price_col)
        if 'drawdown' in df_with_dd.columns:
            report['max_drawdown'] = float(df_with_dd['drawdown'].min())
            report['avg_drawdown'] = float(df_with_dd[df_with_dd['drawdown'] < 0]['drawdown'].mean())
        
        logger.info(f"Report generated for {asset_name}")
        return report





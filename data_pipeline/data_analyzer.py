"""
Data Analysis Module
Performs statistical analysis and trains ML models on historical data.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Statistical and ML analysis of trading data."""

    def __init__(self):
        pass

    def statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform statistical analysis on data."""
        results = {}

        if 'returns' in df.columns:
            returns = df['returns'].dropna()
            results['mean_return'] = returns.mean()
            results['volatility'] = returns.std()
            results['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            results['max_drawdown'] = self._calculate_max_drawdown(df)

        if 'volume' in df.columns:
            results['avg_volume'] = df['volume'].mean()
            results['volume_volatility'] = df['volume'].std()

        # Correlation analysis
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = df.corr()
            results['top_correlations'] = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10)

        logger.info("Statistical analysis completed")
        return results

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if 'close' not in df.columns:
            return 0.0

        cumulative = (df['close'] / df['close'].iloc[0]) - 1
        return (cumulative / cumulative.cummax() - 1).min()

    def ml_analysis(self, df: pd.DataFrame, target_col: str = 'returns') -> Dict:
        """Train ML models for pattern recognition."""
        results = {}

        # Prepare features
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
        if not feature_cols:
            logger.warning("No features available for ML analysis")
            return results

        X = df[feature_cols].dropna()
        y = df[target_col].dropna()

        # Align indices
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        if len(X) < 10:
            logger.warning("Insufficient data for ML analysis")
            return results

        # Linear regression for trend analysis
        lr = LinearRegression()
        lr.fit(X, y)
        results['linear_regression_score'] = lr.score(X, y)
        results['feature_importance'] = dict(zip(feature_cols, lr.coef_))

        # Clustering for market regimes
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        results['market_regimes'] = pd.Series(clusters, index=X.index).value_counts().to_dict()

        logger.info("ML analysis completed")
        return results

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(10000, 2000, 100),
        'returns': np.random.normal(0.001, 0.02, 100)
    })

    analyzer = DataAnalyzer()
    stats = analyzer.statistical_analysis(df)
    ml_results = analyzer.ml_analysis(df)

    print("Statistical Results:", stats)
    print("ML Results:", ml_results)

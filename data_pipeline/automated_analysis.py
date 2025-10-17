"""
Automated Analysis Module
ML-driven insights and predictive analytics for trading.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AutomatedAnalyzer:
    """ML-powered automated analysis for market insights."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_predictor = RandomForestClassifier(n_estimators=100, random_state=42)

    def detect_market_anomalies(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Detect anomalous market behavior."""
        # Prepare features
        X = data[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        # Fit anomaly detector
        self.anomaly_detector.fit(X_scaled)

        # Predict anomalies
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        predictions = self.anomaly_detector.predict(X_scaled)

        # Add to dataframe
        data = data.copy()
        data['anomaly_score'] = anomaly_scores
        data['is_anomaly'] = predictions == -1  # -1 indicates anomaly

        logger.info(f"Detected {data['is_anomaly'].sum()} anomalies in {len(data)} data points")
        return data

    def predict_market_regime(self, data: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """Predict current market regime (bull, bear, sideways)."""
        data = data.copy()

        # Create regime labels (simplified)
        returns = data['close'].pct_change()
        data['regime'] = 'sideways'
        data.loc[returns > 0.02, 'regime'] = 'bull'
        data.loc[returns < -0.02, 'regime'] = 'bear'

        # Prepare features for prediction
        features = []
        for col in ['close', 'volume']:
            if col in data.columns:
                for lag in range(1, 6):
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                    features.append(f'{col}_lag_{lag}')

        # Remove NaN rows
        data = data.dropna()
        if len(data) < lookback:
            logger.warning("Insufficient data for regime prediction")
            return data

        # Train on historical data
        train_data = data.iloc[:-lookback]
        if len(train_data) > 0:
            X_train = train_data[features]
            y_train = train_data['regime']

            self.trend_predictor.fit(X_train, y_train)

            # Predict current regime
            X_current = data[features].iloc[-lookback:]
            predictions = self.trend_predictor.predict_proba(X_current)

            data.loc[X_current.index, 'bull_prob'] = predictions[:, 0]
            data.loc[X_current.index, 'bear_prob'] = predictions[:, 1]
            data.loc[X_current.index, 'sideways_prob'] = predictions[:, 2]

        logger.info("Market regime prediction completed")
        return data

    def generate_trading_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate automated trading insights."""
        insights = []

        # Volume analysis
        if 'volume' in data.columns:
            recent_vol = data['volume'].tail(20).mean()
            historical_vol = data['volume'].tail(100).mean()
            if recent_vol > historical_vol * 1.5:
                insights.append("⚠️ High volume spike detected - potential market move")

        # Price momentum
        if 'close' in data.columns:
            returns = data['close'].pct_change().tail(10)
            if returns.mean() > 0.01:
                insights.append("📈 Strong upward momentum in recent trades")
            elif returns.mean() < -0.01:
                insights.append("📉 Strong downward momentum in recent trades")

        # VPIN analysis
        if 'vpin' in data.columns:
            current_vpin = data['vpin'].iloc[-1]
            if current_vpin > 0.7:
                insights.append("🚨 High VPIN detected - toxic order flow, reduce position sizes")

        # Anomaly alerts
        if 'is_anomaly' in data.columns and data['is_anomaly'].iloc[-1]:
            insights.append("🔍 Anomalous market behavior detected - review positions")

        logger.info(f"Generated {len(insights)} trading insights")
        return insights

    def analyze_correlations(self, data: pd.DataFrame, threshold: float = 0.7) -> Dict[str, List[str]]:
        """Analyze correlations between features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()

        strong_correlations = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    direction = "positive" if corr_val > 0 else "negative"
                    strong_correlations[f"{col1}_{col2}"] = [direction, corr_val]

        logger.info(f"Found {len(strong_correlations)} strong correlations")
        return strong_correlations

# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=500, freq='1H'),
        'close': np.random.normal(50000, 1000, 500),
        'volume': np.random.normal(10000, 2000, 500),
        'vpin': np.random.uniform(0, 1, 500)
    })

    analyzer = AutomatedAnalyzer()

    # Detect anomalies
    features = ['close', 'volume', 'vpin']
    data_with_anomalies = analyzer.detect_market_anomalies(data, features)

    # Predict regimes
    data_with_regime = analyzer.predict_market_regime(data)

    # Generate insights
    insights = analyzer.generate_trading_insights(data_with_regime)
    print("Trading Insights:")
    for insight in insights:
        print(f"  {insight}")

    # Analyze correlations
    correlations = analyzer.analyze_correlations(data)
    print(f"Strong correlations: {len(correlations)}")

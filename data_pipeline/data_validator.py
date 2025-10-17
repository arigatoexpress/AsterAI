"""
Data Validation Module
Ensures data accuracy, consistency, and completeness upon ingestion.
Includes anomaly detection for suspicious patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Automated validation for data accuracy and anomalies."""

    def __init__(self):
        self.anomaly_thresholds = {
            'price_change_pct': 0.5,  # 50% price change threshold
            'volume_spike': 5.0,     # 5x volume spike
            'timestamp_gap_hours': 1  # Max gap between data points
        }

    def validate_ingestion(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data upon ingestion."""
        errors = []

        # Check timestamps
        if 'timestamp' not in df.columns:
            errors.append("Missing timestamp column")
        else:
            gaps = self._check_timestamp_gaps(df)
            if gaps:
                errors.extend(gaps)

        # Check price anomalies
        if 'close' in df.columns:
            anomalies = self._detect_price_anomalies(df)
            errors.extend(anomalies)

        # Check volume consistency
        if 'volume' in df.columns:
            volume_issues = self._validate_volume(df)
            errors.extend(volume_issues)

        return len(errors) == 0, errors

    def _check_timestamp_gaps(self, df: pd.DataFrame) -> List[str]:
        """Check for missing data points."""
        df = df.sort_values('timestamp')
        gaps = df['timestamp'].diff().dt.total_seconds() / 3600  # Hours
        large_gaps = gaps[gaps > self.anomaly_thresholds['timestamp_gap_hours']]
        return [f"Large timestamp gap: {gap:.1f} hours at {ts}" for ts, gap in zip(df['timestamp'][large_gaps.index], large_gaps)]

    def _detect_price_anomalies(self, df: pd.DataFrame) -> List[str]:
        """Detect price anomalies using statistical methods."""
        returns = df['close'].pct_change().abs()
        anomalies = returns[returns > self.anomaly_thresholds['price_change_pct']]
        return [f"Price anomaly: {pct:.1%} at {ts}" for ts, pct in zip(df['timestamp'][anomalies.index], anomalies)]

    def _validate_volume(self, df: pd.DataFrame) -> List[str]:
        """Validate volume consistency."""
        rolling_mean = df['volume'].rolling(24).mean()
        spikes = df['volume'][df['volume'] > rolling_mean * self.anomaly_thresholds['volume_spike']]
        return [f"Volume spike: {vol:.0f} at {ts}" for ts, vol in zip(df['timestamp'][spikes.index], spikes)]

# Example usage
if __name__ == "__main__":
    # Simulate data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.normal(50000, 1000, 100),
        'volume': np.random.normal(10000, 2000, 100)
    })

    validator = DataValidator()
    valid, errors = validator.validate_ingestion(df)
    print(f"Data valid: {valid}")
    if errors:
        print("Errors:", errors)

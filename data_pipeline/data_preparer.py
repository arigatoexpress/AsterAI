"""
Data Preparation Module
Normalizes data formats, handles standardization, and feeds into feature engineering.
"""
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DataPreparer:
    """Standardizes and prepares data for analysis."""

    def __init__(self):
        self.standard_formats = {
            'timestamp': 'datetime64[ns]',
            'price': 'float64',
            'volume': 'int64'
        }

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data formats and timestamps."""
        df = df.copy()

        # Standardize column names
        column_mapping = {
            'time': 'timestamp',
            'close_price': 'close',
            'volume_traded': 'volume'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert data types
        for col, dtype in self.standard_formats.items():
            if col in df.columns:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Data normalized: {len(df)} rows")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill or remove missing values."""
        # Forward fill price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')

        # Interpolate volume
        if 'volume' in df.columns:
            df['volume'] = df['volume'].interpolate()

        # Drop rows with critical missing data
        df = df.dropna(subset=['timestamp'])

        return df

    def prepare_for_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for feature engineering."""
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")

        # Calculate basic features
        if all(col in df.columns for col in ['high', 'low', 'close']):
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(24).std()  # Daily vol

        logger.info("Data prepared for feature engineering")
        return df

# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10),
        'close_price': [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
        'volume_traded': [1000, 1100, None, 1200, 900, 1300, 800, 1400, 700, 1500]
    })

    preparer = DataPreparer()
    normalized = preparer.normalize_data(df)
    prepared = preparer.prepare_for_features(normalized)
    print(prepared.head())

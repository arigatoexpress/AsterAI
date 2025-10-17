#!/usr/bin/env python3
"""
Debug feature engineering to verify 41 features are generated.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trading.aster_perps_extreme_growth import AsterPerpsExtremeStrategy

def test_feature_count():
    """Test that we generate exactly 41 features."""

    # Create strategy instance
    strategy = AsterPerpsExtremeStrategy(total_capital=150.0)

    # Create sample data
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000
    })

    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Calculate features
    features_df = strategy.calculate_features(df.copy())

    print(f"\nFeatures calculated: {len(features_df.columns)}")
    print(f"Feature columns: {list(features_df.columns)}")

    # Check for the specific features that were missing
    required_features = [
        "quote_volume", "taker_buy_volume",  # The ones we added
        "price_change", "price_change_5", "price_change_20",
        "high_low_ratio", "close_open_ratio",
        "volume_change", "volume_price_ratio", "volume_ma_ratio",
        "sma_5", "price_sma_5_ratio", "sma_10", "price_sma_10_ratio",
        "sma_20", "price_sma_20_ratio", "sma_50", "price_sma_50_ratio",
        "ema_12", "ema_26", "macd", "macd_signal", "macd_histogram",
        "bb_middle", "bb_std", "bb_upper", "bb_lower", "bb_width", "bb_position",
        "volatility_20", "volatility_50",
        "rsi", "rsi_30",
        "stoch_k", "stoch_d",
        "atr", "obv", "mfi",
        "market_momentum", "relative_strength", "volume_rank"
    ]

    print(f"\nExpected features: {len(required_features)}")
    print(f"Expected: {required_features}")

    missing_features = [f for f in required_features if f not in features_df.columns]
    extra_features = [f for f in features_df.columns if f not in required_features]

    if missing_features:
        print(f"\n❌ MISSING FEATURES: {missing_features}")
    else:
        print("\n✅ All expected features present")

    if extra_features:
        print(f"⚠️ EXTRA FEATURES: {extra_features}")
    else:
        print("✅ No extra features")

    # Check final count
    final_count = len(features_df.columns)
    print(f"\nFinal feature count: {final_count}")
    if final_count == 41:
        print("✅ SUCCESS: Exactly 41 features generated!")
        return True
    else:
        print(f"❌ FAILURE: Expected 41 features, got {final_count}")
        return False

if __name__ == "__main__":
    test_feature_count()

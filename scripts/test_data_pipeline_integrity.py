"""
Test Data Pipeline Integrity
End-to-end test of collection, validation, sharing, preparation, analysis, utilization.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add data_pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_pipeline.data_validator import DataValidator
from data_pipeline.data_sharer import DataSharer
from data_pipeline.data_preparer import DataPreparer
from data_pipeline.data_analyzer import DataAnalyzer
from data_pipeline.data_utilizer import DataUtilizer

def test_full_pipeline():
    """Test complete data pipeline."""
    print("Testing Data Pipeline Integrity...")

    # Generate test data
    dates = pd.date_range('2024-01-01', periods=500, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(50000, 500, 500),
        'high': np.random.normal(50100, 600, 500),
        'low': np.random.normal(49900, 400, 500),
        'close': np.random.normal(50000, 1000, 500),
        'volume': np.random.normal(10000, 2000, 500)
    })

    # 1. Validation
    validator = DataValidator()
    valid, errors = validator.validate_ingestion(df)
    assert valid, f"Validation failed: {errors}"
    print("✓ Data validation passed")

    # 2. Sharing
    sharer = DataSharer()
    data_bytes = df.to_parquet()
    stored = sharer.store_data("test_data", data_bytes, role="collector")
    assert stored, "Data storage failed"
    retrieved = sharer.retrieve_data("test_data", role="analyzer")
    assert retrieved is not None, "Data retrieval failed"
    retrieved_df = pd.read_parquet(pd.io.common.BytesIO(retrieved))
    assert len(retrieved_df) == len(df), "Data integrity lost in sharing"
    print("✓ Data sharing passed")

    # 3. Preparation
    preparer = DataPreparer()
    normalized = preparer.normalize_data(retrieved_df)
    prepared = preparer.prepare_for_features(normalized)
    assert 'returns' in prepared.columns, "Preparation failed to add returns"
    print("✓ Data preparation passed")

    # 4. Analysis
    analyzer = DataAnalyzer()
    stats = analyzer.statistical_analysis(prepared)
    assert 'sharpe_ratio' in stats, "Statistical analysis failed"
    ml_results = analyzer.ml_analysis(prepared, target_col='returns')
    assert 'linear_regression_score' in ml_results, "ML analysis failed"
    print("✓ Data analysis passed")

    # 5. Utilization
    utilizer = DataUtilizer()
    backtest_results = utilizer.backtest_autonomous_trader(prepared)
    assert 'overall_score' in backtest_results, "Backtest failed"
    live_data = utilizer.prepare_for_live_execution(prepared.tail(50))
    assert 'prediction' in live_data, "Live preparation failed"
    print("✓ Data utilization passed")

    print("🎉 All data pipeline tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

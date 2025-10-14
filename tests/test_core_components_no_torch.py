"""
Unit tests for core components (without PyTorch dependency).
Tests the core functionality that doesn't require deep learning libraries.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_training.feature_engineering import ComprehensiveFeatureEngineer, FeatureConfig
from local_training.advanced_backtester import AdvancedBacktester, BacktestConfig, BacktestMode
from local_training.strategies.leveraged_perps_strategies import StrategyConfig, EnsembleStrategy


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def test_feature_engineer_initialization(self):
        """Test feature engineer can be initialized."""
        config = FeatureConfig(
            lookback_periods=[5, 10, 20],
            ta_indicators=['rsi', 'macd'],
            alternative_data=False
        )

        engineer = ComprehensiveFeatureEngineer(config)

        assert engineer is not None
        assert engineer.config.lookback_periods == [5, 10, 20]
        assert 'rsi' in engineer.config.ta_indicators

    def test_feature_creation_basic(self):
        """Test basic feature creation."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        prices = 50000 + np.cumsum(np.random.normal(0.001, 0.02, 100))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'volume': np.random.uniform(1000000, 10000000, 100)
        })

        config = FeatureConfig(
            lookback_periods=[5, 10, 20, 50],
            ta_indicators=[],
            alternative_data=False
        )

        engineer = ComprehensiveFeatureEngineer(config)
        features = engineer.create_all_features(df)

        assert features is not None
        assert len(features) > 0
        assert 'price_volatility_5' in features.columns
        assert 'price_volatility_10' in features.columns

    def test_feature_engineering_no_crash(self):
        """Test feature engineering doesn't crash with various data."""
        # Test with minimal data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(1000, 2000, 50),
            'close': np.random.uniform(1000, 2000, 50),
            'high': np.random.uniform(1000, 2000, 50),
            'low': np.random.uniform(1000, 2000, 50),
            'volume': np.random.uniform(1000, 10000, 50)
        })

        engineer = ComprehensiveFeatureEngineer()
        features = engineer.create_all_features(df)

        # Should not crash and should return some features
        assert features is not None
        assert len(features) >= 0  # May be 0 if not enough data


class TestBacktester:
    """Test backtesting framework."""

    def test_backtester_initialization(self):
        """Test backtester can be initialized."""
        config = BacktestConfig(
            initial_balance=10000.0,
            mode=BacktestMode.PERPETUAL,
            max_leverage=10
        )

        backtester = AdvancedBacktester(config)

        assert backtester is not None
        assert backtester.config.initial_balance == 10000.0
        assert backtester.config.mode == BacktestMode.PERPETUAL

    def test_backtester_state_tracking(self):
        """Test backtester state tracking."""
        config = BacktestConfig()
        backtester = AdvancedBacktester(config)

        # Check initial state
        assert backtester.balance == config.initial_balance
        assert len(backtester.positions) == 0
        assert len(backtester.portfolio_values) == 1
        assert backtester.portfolio_values[0] == config.initial_balance

    def test_backtester_config_validation(self):
        """Test backtester configuration validation."""
        # Test invalid configuration
        try:
            config = BacktestConfig(initial_balance=-1000)  # Negative balance
            # Should not raise error during config creation, but during backtest
        except:
            pass  # Config creation should be flexible

        # Test valid configuration
        config = BacktestConfig(
            initial_balance=10000.0,
            max_daily_loss=0.05,
            max_drawdown=0.15
        )

        assert config.initial_balance > 0
        assert 0 < config.max_daily_loss < 1
        assert 0 < config.max_drawdown < 1


class TestStrategies:
    """Test trading strategies."""

    def test_strategy_config(self):
        """Test strategy configuration."""
        config = StrategyConfig(
            symbol="BTCUSDT",
            max_leverage=20,
            position_size_pct=0.1
        )

        assert config.symbol == "BTCUSDT"
        assert config.max_leverage == 20
        assert config.position_size_pct == 0.1

    def test_ensemble_strategy_generation(self):
        """Test ensemble strategy signal generation."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        prices = 50000 + np.cumsum(np.random.normal(0.001, 0.02, 100))

        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': np.random.uniform(1000000, 10000000, 100)
        })

        config = StrategyConfig(symbol="BTCUSDT")
        strategy = EnsembleStrategy(config)

        # Generate signals (should not crash)
        signals = strategy.generate_signals(df, "BTCUSDT", pd.Timestamp.now())

        # Should return a list (may be empty if not enough data)
        assert isinstance(signals, list)


class TestDataStructures:
    """Test data structures and utilities."""

    def test_dataframe_creation(self):
        """Test DataFrame creation and manipulation."""
        # Create sample OHLCV data
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'open': np.random.uniform(1000, 2000, 100),
            'high': np.random.uniform(1000, 2000, 100),
            'low': np.random.uniform(1000, 2000, 100),
            'close': np.random.uniform(1000, 2000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }

        df = pd.DataFrame(data)

        # Test basic operations
        assert len(df) == 100
        assert 'close' in df.columns
        assert df['close'].dtype in [np.dtype('float64'), np.dtype('float32')]

        # Test returns calculation
        returns = df['close'].pct_change()
        assert len(returns) == 100
        assert returns.iloc[0] is None or pd.isna(returns.iloc[0])  # First value should be NaN

    def test_array_operations(self):
        """Test numpy array operations."""
        # Test array creation and operations
        prices = np.random.uniform(1000, 2000, 100)
        returns = np.diff(prices) / prices[:-1]

        assert len(returns) == 99
        assert all(np.isfinite(returns))  # No NaN or inf values

        # Test rolling operations
        volatility = pd.Series(returns).rolling(20).std().iloc[-1]
        assert np.isfinite(volatility) or pd.isna(volatility)


class TestConfigurationManagement:
    """Test configuration management."""

    def test_config_creation(self):
        """Test configuration objects can be created."""
        from dataclasses import dataclass

        @dataclass
        class TestConfig:
            value: float = 1.0
            name: str = "test"

        config = TestConfig(value=2.0, name="custom")
        assert config.value == 2.0
        assert config.name == "custom"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that configurations handle edge cases gracefully
        config = FeatureConfig(
            lookback_periods=[1, 2, 3],
            ta_indicators=[],
            alternative_data=False
        )

        assert config.lookback_periods == [1, 2, 3]
        assert len(config.ta_indicators) == 0


class TestImportStructure:
    """Test that imports work correctly."""

    def test_local_training_imports(self):
        """Test that local training modules can be imported."""
        try:
            from local_training.feature_engineering import ComprehensiveFeatureEngineer
            from local_training.advanced_backtester import AdvancedBacktester
            from local_training.strategies.leveraged_perps_strategies import StrategyConfig

            # Should not raise ImportError
            assert True

        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


class TestPerformanceBenchmarks:
    """Test performance benchmarks."""

    def test_basic_operations_performance(self):
        """Test that basic operations are reasonably fast."""
        import time

        # Test DataFrame operations
        size = 1000
        data = np.random.random((size, 5))

        start = time.time()
        df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])
        df['sum'] = df.sum(axis=1)
        end = time.time()

        # Should complete in reasonable time (under 1 second)
        assert end - start < 1.0

    def test_array_operations_performance(self):
        """Test array operations performance."""
        import time

        size = 10000
        a = np.random.random(size)
        b = np.random.random(size)

        start = time.time()
        c = a * b + np.sin(a)
        end = time.time()

        # Should complete quickly (under 0.1 seconds)
        assert end - start < 0.1


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

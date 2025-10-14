"""
Unit tests for risk management components.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_trader.risk.kelly import kelly_fraction, fractional_kelly
from mcp_trader.risk.risk_manager import RiskManager, RiskMetrics, PositionRisk
from mcp_trader.trading.types import PortfolioState, MarketRegime
from mcp_trader.config import get_settings


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""

    def test_kelly_fraction_basic(self):
        """Test basic Kelly Criterion calculation."""
        # Win prob 60%, win/loss ratio 2:1
        kelly = kelly_fraction(win_prob=0.6, win_loss_ratio=2.0)

        assert kelly > 0
        assert kelly < 1.0  # Should be reasonable fraction

    def test_kelly_fraction_edge_cases(self):
        """Test edge cases for Kelly Criterion."""
        # Zero win probability
        kelly_zero = kelly_fraction(win_prob=0.0, win_loss_ratio=2.0)
        assert kelly_zero == 0.0

        # Win probability = 1
        kelly_one = kelly_fraction(win_prob=1.0, win_loss_ratio=2.0)
        assert kelly_one == 0.0  # Invalid input

        # Negative ratio
        kelly_neg = kelly_fraction(win_prob=0.6, win_loss_ratio=-1.0)
        assert kelly_neg == 0.0

    def test_fractional_kelly(self):
        """Test fractional Kelly Criterion."""
        full_kelly = kelly_fraction(win_prob=0.6, win_loss_ratio=2.0)
        half_kelly = fractional_kelly(win_prob=0.6, win_loss_ratio=2.0, fraction=0.5)

        assert half_kelly == full_kelly * 0.5

    def test_kelly_with_realistic_trading_params(self):
        """Test Kelly with realistic trading parameters."""
        # Typical crypto trading: 55% win rate, 1.5:1 reward ratio
        kelly = fractional_kelly(win_prob=0.55, win_loss_ratio=1.5, fraction=0.5)

        assert kelly > 0
        assert kelly < 0.2  # Should be conservative


class TestRiskManager:
    """Test risk management system."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager for testing."""
        settings = get_settings()
        return RiskManager(settings)

    @pytest.fixture
    def mock_portfolio_state(self):
        """Create mock portfolio state."""
        return PortfolioState(
            timestamp=datetime.now(),
            total_balance=10000.0,
            available_balance=8000.0,
            total_positions_value=2000.0,
            unrealized_pnl=100.0,
            active_positions={},
            active_grids={}
        )

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        prices = 1000 + np.cumsum(np.random.normal(0.5, 10, 100))

        return {
            'BTCUSDT': pd.DataFrame({
                'timestamp': dates,
                'open': prices * 0.99,
                'high': prices * 1.01,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.uniform(1000, 10000, 100)
            })
        }

    def test_risk_manager_initialization(self, risk_manager):
        """Test risk manager initializes correctly."""
        assert risk_manager is not None
        assert risk_manager.max_portfolio_risk > 0
        assert risk_manager.kelly_fraction > 0

    @pytest.mark.asyncio
    async def test_portfolio_risk_assessment(self, risk_manager, mock_portfolio_state, mock_market_data):
        """Test portfolio risk assessment."""
        metrics = await risk_manager.assess_portfolio_risk(mock_portfolio_state, mock_market_data)

        assert isinstance(metrics, RiskMetrics)
        assert metrics.portfolio_value > 0
        assert metrics.volatility >= 0
        assert 0 <= metrics.concentration_risk <= 1.0

    @pytest.mark.asyncio
    async def test_position_size_calculation(self, risk_manager, mock_portfolio_state, mock_market_data):
        """Test position size calculation."""
        symbol = 'BTCUSDT'
        entry_price = 50000.0
        stop_loss_price = 48500.0  # 3% stop

        position_risk = await risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_data=mock_market_data['BTCUSDT'],
            portfolio_state=mock_portfolio_state,
            market_regime=MarketRegime.SIDEWAYS
        )

        assert isinstance(position_risk, PositionRisk)
        assert position_risk.final_position_size > 0
        assert position_risk.stop_loss_price < entry_price
        assert position_risk.take_profit_price > entry_price

    def test_volatility_adjustment(self, risk_manager):
        """Test volatility-based position adjustment."""
        # Create high volatility data
        high_vol_data = pd.DataFrame({
            'close': 1000 + np.cumsum(np.random.normal(0, 50, 100))  # High volatility
        })

        # Create low volatility data
        low_vol_data = pd.DataFrame({
            'close': 1000 + np.cumsum(np.random.normal(0, 5, 100))  # Low volatility
        })

        high_vol_adj = risk_manager._calculate_volatility_adjustment('BTCUSDT', high_vol_data)
        low_vol_adj = risk_manager._calculate_volatility_adjustment('BTCUSDT', low_vol_data)

        # High volatility should reduce position size
        assert high_vol_adj < low_vol_adj

    def test_daily_returns_calculation(self, risk_manager):
        """Test daily returns calculation."""
        # Add mock history
        for i in range(10):
            risk_manager.portfolio_history.append({
                'timestamp': datetime.now(),
                'metrics': RiskMetrics(
                    portfolio_value=10000 + i * 100,
                    total_risk=0.05,
                    max_drawdown=0.02,
                    sharpe_ratio=1.5,
                    volatility=0.03,
                    var_95=0.04,
                    expected_shortfall=0.05,
                    concentration_risk=0.15,
                    correlation_risk=0.5
                ),
                'portfolio_state': None
            })

        returns = risk_manager._calculate_daily_returns(10500)

        assert len(returns) > 0
        assert all(isinstance(r, float) for r in returns)

    def test_sharpe_ratio_calculation(self, risk_manager):
        """Test Sharpe ratio calculation."""
        returns = np.random.normal(0.001, 0.02, 100)  # Positive average return

        sharpe = risk_manager._calculate_sharpe_ratio(returns.tolist())

        assert isinstance(sharpe, float)
        # Sharpe should be reasonable for positive returns
        assert -5 < sharpe < 10


class TestDataValidation:
    """Test data validation and preprocessing."""

    def test_data_normalization(self):
        """Test data normalization doesn't introduce NaN."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'close': 1000 + np.cumsum(np.random.normal(0, 10, 100))
        })

        # Normalize
        normalized = (data['close'] - data['close'].mean()) / data['close'].std()

        assert not normalized.isna().any()
        assert np.isfinite(normalized).all()

    def test_feature_engineering_no_inf(self):
        """Test feature engineering doesn't create infinite values."""
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'close': 1000 + np.cumsum(np.random.normal(0, 10, 100)),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Calculate returns
        returns = data['close'].pct_change()

        assert np.isfinite(returns[1:]).all()  # First value will be NaN
        assert not np.isinf(returns).any()


class TestModelPersistence:
    """Test model saving and loading."""

    def test_lstm_save_load_config(self, tmp_path):
        """Test LSTM configuration can be saved and loaded."""
        config = {
            'input_size': 20,
            'hidden_size': 64,
            'num_layers': 2
        }

        model = LSTMPredictorModel(config)

        # Model should have save capability
        assert hasattr(model, 'save_model')
        assert hasattr(model, 'load_model')


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test performance of critical operations."""

    def test_feature_extraction_performance(self):
        """Test feature extraction is reasonably fast."""
        import time

        # Large dataset
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 1000 + np.cumsum(np.random.normal(0, 10, 1000)),
            'high': 1100 + np.cumsum(np.random.normal(0, 10, 1000)),
            'low': 900 + np.cumsum(np.random.normal(0, 10, 1000)),
            'close': 1000 + np.cumsum(np.random.normal(0, 10, 1000)),
            'volume': np.random.uniform(1000, 10000, 1000)
        })

        model = LSTMPredictorModel({'input_size': 10})

        start = time.time()
        features = model.prepare_features(df)
        elapsed = time.time() - start

        # Should complete in under 2 seconds
        assert elapsed < 2.0
        assert features is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


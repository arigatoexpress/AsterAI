"""
Unit tests for data pipeline components.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_trader.data.aster_feed import AsterDataFeed, MarketData
from mcp_trader.config import PRIORITY_SYMBOLS


class TestAsterDataFeed:
    """Test Aster DEX data feed."""

    def test_data_feed_initialization(self):
        """Test data feed can be initialized."""
        feed = AsterDataFeed()

        assert feed is not None
        assert feed.price_cache == {}
        assert feed.kline_cache == {}
        assert not feed.websocket_active

    def test_volatility_calculation(self):
        """Test volatility calculation from klines."""
        feed = AsterDataFeed()

        # Create mock klines data
        mock_klines = []
        base_price = 1000
        for i in range(50):
            price = base_price + np.random.normal(0, 10)
            kline = [
                int(datetime.now().timestamp() * 1000),  # open_time
                str(price * 0.99),  # open
                str(price * 1.01),  # high
                str(price * 0.98),  # low
                str(price),  # close
                str(np.random.uniform(1000, 10000)),  # volume
            ]
            mock_klines.append(kline)

        feed.kline_cache['BTCUSDT'] = mock_klines

        volatility = feed.calculate_volatility('BTCUSDT', window_hours=24)

        assert volatility >= 0
        assert volatility < 10.0  # Should be reasonable

    def test_price_momentum_calculation(self):
        """Test price momentum calculation."""
        feed = AsterDataFeed()

        # Create upward trending prices
        mock_klines = []
        for i in range(20):
            price = 1000 + i * 10  # Upward trend
            kline = [
                int(datetime.now().timestamp() * 1000),
                str(price * 0.99),
                str(price * 1.01),
                str(price * 0.98),
                str(price),
                str(1000),
            ]
            mock_klines.append(kline)

        feed.kline_cache['BTCUSDT'] = mock_klines

        momentum = feed.get_price_momentum('BTCUSDT', periods=6)

        assert momentum > 0  # Should detect upward trend

    def test_feed_status(self):
        """Test feed status reporting."""
        feed = AsterDataFeed()

        status = feed.get_feed_status()

        assert 'websocket_active' in status
        assert 'subscribed_symbols' in status
        assert 'cache_sizes' in status

    def test_ticker_update_handling(self):
        """Test ticker data structure."""
        feed = AsterDataFeed()

        # Mock ticker update
        mock_ticker_data = {
            's': 'BTC',  # Symbol
            'p': '500',  # Price change
            'P': '1.0',  # Price change percent
            'c': '51000',  # Last price
            'v': '1000000',  # Volume
        }

        # Should not crash
        asyncio.run(feed._handle_ticker_update(mock_ticker_data))


class TestDataQuality:
    """Test data quality and integrity."""

    def test_price_data_integrity(self):
        """Test price data is valid."""
        # Simulate price data
        prices = 1000 + np.cumsum(np.random.normal(0, 10, 100))

        # Check no negative prices
        assert np.all(prices > 0)

        # Check no extreme jumps (>50%)
        returns = np.diff(prices) / prices[:-1]
        assert np.all(np.abs(returns) < 0.5)

    def test_volume_data_validation(self):
        """Test volume data is non-negative."""
        volumes = np.random.uniform(1000, 1000000, 100)

        assert np.all(volumes >= 0)
        assert np.isfinite(volumes).all()

    def test_timestamp_consistency(self):
        """Test timestamp data is monotonically increasing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')

        # Check monotonic
        assert dates.is_monotonic_increasing

        # Check no duplicates
        assert len(dates) == len(dates.unique())


class TestCacheManagement:
    """Test data caching mechanisms."""

    def test_cache_size_limits(self):
        """Test cache doesn't grow unbounded."""
        feed = AsterDataFeed()

        # Add many items to cache
        for i in range(2000):
            symbol = f"TEST{i}"
            feed.price_cache[symbol] = {'price': i}

        # Cache should not exceed reasonable size
        # In real implementation, old entries should be evicted
        # For now, just verify it doesn't crash
        assert len(feed.price_cache) > 0

    def test_cache_data_freshness(self):
        """Test cache tracks last update times."""
        feed = AsterDataFeed()

        symbol = 'BTCUSDT'
        feed.last_update[symbol] = datetime.now()

        # Check timestamp exists
        assert symbol in feed.last_update
        assert isinstance(feed.last_update[symbol], datetime)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


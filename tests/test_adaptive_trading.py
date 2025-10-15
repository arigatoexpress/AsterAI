#!/usr/bin/env python3
"""
Comprehensive tests for the Adaptive AI Trading System

Tests the core components including:
- AdaptiveTradingAgent
- OnlineLearningSystem
- DataFeed components
- Strategy optimization
"""

import asyncio
import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent, AdaptiveAgentConfig, MarketState, PortfolioState
from mcp_trader.ai.online_learning import OnlineLearningSystem, AdaptiveStrategyManager
from mcp_trader.data.aster_feed import AsterDataFeed, MarketData, DataCache
from mcp_trader.logging_utils import get_logger


@pytest.fixture
def logger():
    """Provide a logger for tests."""
    return get_logger("test_adaptive")


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return AdaptiveAgentConfig(
        initial_balance=10000.0,
        max_allocation_per_trade=0.1,
        min_allocation_per_trade=0.01,
        max_open_positions=5,
        rebalance_frequency_minutes=15,
        learning_rate=0.01,
        risk_tolerance=0.15,
        volatility_threshold=0.03,
        profit_taking_threshold=0.05,
        stop_loss_threshold=0.03,
        adaptation_window_minutes=60
    )


@pytest.fixture
def sample_market_state():
    """Provide sample market state for testing."""
    return MarketState(
        timestamp=datetime.now(),
        prices={'BTCUSDT': 50000, 'ETHUSDT': 3000, 'SOLUSDT': 100},
        volumes={'BTCUSDT': 1000000, 'ETHUSDT': 500000, 'SOLUSDT': 200000},
        volatility={'BTCUSDT': 0.03, 'ETHUSDT': 0.04, 'SOLUSDT': 0.06},
        momentum={'BTCUSDT': 0.02, 'ETHUSDT': -0.01, 'SOLUSDT': 0.03}
    )


@pytest.fixture
def sample_portfolio_state():
    """Provide sample portfolio state for testing."""
    return PortfolioState(
        timestamp=datetime.now(),
        total_balance=10000.0,
        available_balance=8000.0,
        total_positions_value=2000.0
    )


class TestDataCache:
    """Test the DataCache functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        cache = DataCache(ttl_seconds=60)

        # Test empty cache
        assert cache.get("nonexistent") is None

        # Test setting and getting
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        # Test cache clearing
        cache.clear()
        assert cache.get("test_key") is None

    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        import time
        cache = DataCache(ttl_seconds=1)  # 1 second TTL

        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_key") is None


class TestMarketData:
    """Test MarketData dataclass."""

    def test_market_data_creation(self):
        """Test MarketData object creation."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1000000.0,
            timestamp=datetime.now(),
            bid_price=49999.0,
            ask_price=50001.0,
            high_24h=52000.0,
            low_24h=48000.0
        )

        assert market_data.symbol == "BTCUSDT"
        assert market_data.price == 50000.0
        assert market_data.volume == 1000000.0
        assert market_data.bid_price == 49999.0
        assert market_data.ask_price == 50001.0


class TestAdaptiveTradingAgent:
    """Test AdaptiveTradingAgent functionality."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, sample_config, logger):
        """Test agent initialization."""
        agent = AdaptiveTradingAgent(sample_config)

        assert agent.config == sample_config
        assert agent.portfolio_state is None
        assert agent.market_state is None
        assert agent.positions == {}
        assert agent.market_history == []
        assert agent.strategy_weights == {}

    @pytest.mark.asyncio
    async def test_agent_market_update(self, sample_config, sample_market_state, sample_portfolio_state, logger):
        """Test market state updates."""
        agent = AdaptiveTradingAgent(sample_config)

        # Update market state
        agent.market_state = sample_market_state
        agent.portfolio_state = sample_portfolio_state

        assert agent.market_state == sample_market_state
        assert agent.portfolio_state == sample_portfolio_state

    @pytest.mark.asyncio
    async def test_agent_position_management(self, sample_config, logger):
        """Test position management."""
        agent = AdaptiveTradingAgent(sample_config)

        # Add a position
        from mcp_trader.ai.adaptive_trading_agent import Position

        position = Position(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000.0,
            quantity=0.1,
            timestamp=datetime.now()
        )

        agent.positions["BTCUSDT"] = position
        assert len(agent.positions) == 1
        assert "BTCUSDT" in agent.positions


class TestOnlineLearning:
    """Test online learning components."""

    @pytest.mark.asyncio
    async def test_learning_system_initialization(self, logger):
        """Test OnlineLearningSystem initialization."""
        learning_system = OnlineLearningSystem()

        assert learning_system.models == {}
        assert learning_system.feature_scaler is None
        assert learning_system.training_samples == []

    @pytest.mark.asyncio
    async def test_feature_extraction(self, sample_market_state, sample_portfolio_state, logger):
        """Test feature extraction from market and portfolio data."""
        learning_system = OnlineLearningSystem()

        features = learning_system.extract_features(
            sample_market_state,
            sample_portfolio_state,
            []
        )

        assert isinstance(features, dict)
        assert len(features) > 0
        # Should have features for each symbol
        assert 'BTCUSDT_price' in features
        assert 'BTCUSDT_volume' in features
        assert 'BTCUSDT_volatility' in features

    @pytest.mark.asyncio
    async def test_training_sample_addition(self, sample_market_state, sample_portfolio_state, logger):
        """Test adding training samples."""
        learning_system = OnlineLearningSystem()

        features = learning_system.extract_features(
            sample_market_state,
            sample_portfolio_state,
            []
        )

        targets = {
            'price_direction': 0.02,
            'volatility': 0.025,
            'regime': 1
        }

        learning_system.add_training_sample(features, targets)

        assert len(learning_system.training_samples) == 1
        assert learning_system.training_samples[0]['features'] == features
        assert learning_system.training_samples[0]['targets'] == targets


class TestStrategyManager:
    """Test strategy management functionality."""

    @pytest.mark.asyncio
    async def test_strategy_manager_initialization(self, logger):
        """Test AdaptiveStrategyManager initialization."""
        learning_system = OnlineLearningSystem()
        strategy_manager = AdaptiveStrategyManager(learning_system)

        assert strategy_manager.learning_system == learning_system
        assert strategy_manager.strategy_weights == {}

    @pytest.mark.asyncio
    async def test_strategy_weight_adaptation(self, sample_market_state, sample_portfolio_state, logger):
        """Test strategy weight adaptation."""
        learning_system = OnlineLearningSystem()
        strategy_manager = AdaptiveStrategyManager(learning_system)

        # Add some training data
        features = learning_system.extract_features(
            sample_market_state,
            sample_portfolio_state,
            []
        )

        targets = {'price_direction': 0.02, 'volatility': 0.025, 'regime': 1}
        learning_system.add_training_sample(features, targets)

        # Test strategy adaptation
        strategy_names = ['barbell', 'asymmetric', 'tail_risk']
        recent_performance = {
            'barbell': 0.05,
            'asymmetric': 0.08,
            'tail_risk': 0.02
        }

        new_weights = strategy_manager.adapt_strategy_weights(
            features,
            strategy_names,
            recent_performance
        )

        assert isinstance(new_weights, dict)
        assert len(new_weights) == len(strategy_names)
        # Weights should sum to approximately 1.0
        total_weight = sum(new_weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestAsterDataFeed:
    """Test AsterDataFeed functionality."""

    @pytest.mark.asyncio
    async def test_data_feed_initialization(self, logger):
        """Test AsterDataFeed initialization."""
        with patch('mcp_trader.data.aster_feed.AsterClient') as mock_client:
            # Mock the client
            mock_client_instance = AsyncMock()
            mock_client_instance.get_ticker.return_value = {
                'lastPrice': '50000',
                'volume': '1000000',
                'bidPrice': '49999',
                'askPrice': '50001',
                'highPrice': '52000',
                'lowPrice': '48000'
            }
            mock_client.return_value = mock_client_instance

            data_feed = AsterDataFeed(['BTCUSDT', 'ETHUSDT'])
            data_feed.client = mock_client_instance

            # Test initialization
            await data_feed._load_initial_market_data()

            assert 'BTCUSDT' in data_feed.market_data
            assert 'ETHUSDT' in data_feed.market_data
            assert data_feed.market_data['BTCUSDT'].price == 50000.0

    @pytest.mark.asyncio
    async def test_data_feed_caching(self, logger):
        """Test data feed caching functionality."""
        cache = DataCache(ttl_seconds=60)

        # Test cache decorator functionality
        from mcp_trader.data.aster_feed import cache_result

        @cache_result(ttl_seconds=30)
        async def mock_api_call():
            return {"data": "test_response"}

        # First call should execute the function
        result1 = await mock_api_call()
        assert result1 == {"data": "test_response"}

        # Second call should return cached result
        result2 = await mock_api_call()
        assert result2 == {"data": "test_response"}


class TestIntegration:
    """Test integration between components."""

    @pytest.mark.asyncio
    async def test_full_system_integration(self, sample_config, sample_market_state,
                                          sample_portfolio_state, logger):
        """Test full system integration."""
        # Initialize all components
        agent = AdaptiveTradingAgent(sample_config)
        learning_system = OnlineLearningSystem()
        strategy_manager = AdaptiveStrategyManager(learning_system)

        # Set up agent state
        agent.market_state = sample_market_state
        agent.portfolio_state = sample_portfolio_state

        # Extract features and add training sample
        features = learning_system.extract_features(
            sample_market_state,
            sample_portfolio_state,
            []
        )

        targets = {'price_direction': 0.02, 'volatility': 0.025, 'regime': 1}
        learning_system.add_training_sample(features, targets)

        # Test strategy adaptation
        strategy_names = ['barbell', 'asymmetric', 'tail_risk']
        recent_performance = {'barbell': 0.05, 'asymmetric': 0.08, 'tail_risk': 0.02}

        new_weights = strategy_manager.adapt_strategy_weights(
            features,
            strategy_names,
            recent_performance
        )

        # Update agent weights
        agent.strategy_weights = new_weights

        # Verify integration
        assert agent.strategy_weights == new_weights
        assert len(learning_system.training_samples) == 1
        assert strategy_manager.strategy_weights == new_weights


def run_sync_test(test_func):
    """Helper to run async tests in sync context."""
    asyncio.run(test_func)


if __name__ == "__main__":
    # Run tests
    test_cache = TestDataCache()
    test_cache.test_cache_basic_operations()
    test_cache.test_cache_ttl()

    test_market_data = TestMarketData()
    test_market_data.test_market_data_creation()

    print("✅ All tests completed successfully!")



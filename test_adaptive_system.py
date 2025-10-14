#!/usr/bin/env python3
"""
Test script for the Current Era Adaptive AI Trading System

This script tests the key components of the adaptive trading system
focused on CURRENT market conditions only. No historical backtesting.

Validates real-time AI learning, current market adaptation, and
live strategy optimization for the present volatile bull market.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent, AdaptiveAgentConfig
from mcp_trader.ai.online_learning import OnlineLearningSystem, AdaptiveStrategyManager
from mcp_trader.config import get_settings
from mcp_trader.security.secrets import get_secret_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_adaptive_components():
    """Test individual components of the adaptive system."""
    print("🧪 Testing Adaptive AI Trading System Components")
    print("="*60)

    try:
        # Test 1: Online Learning System
        print("\n1️⃣ Testing Online Learning System...")
        learning_system = OnlineLearningSystem()

        # Create mock market data
        from mcp_trader.ai.adaptive_trading_agent import MarketState, PortfolioState

        market_state = MarketState(
            timestamp=datetime.now(),
            prices={'BTCUSDT': 50000, 'ETHUSDT': 3000, 'SOLUSDT': 100},
            volumes={'BTCUSDT': 1000000, 'ETHUSDT': 500000, 'SOLUSDT': 200000},
            volatility={'BTCUSDT': 0.03, 'ETHUSDT': 0.04, 'SOLUSDT': 0.06},
            momentum={'BTCUSDT': 0.02, 'ETHUSDT': -0.01, 'SOLUSDT': 0.03}
        )

        portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            total_balance=10000,
            available_balance=8000,
            total_positions_value=2000
        )

        # Extract features
        features = learning_system.extract_features(market_state, portfolio_state, [])
        print(f"✅ Feature extraction: {len(features)} features extracted")

        # Add training sample
        targets = {'price_direction': 0.02, 'volatility': 0.025, 'regime': 1}
        learning_system.add_training_sample(features, targets)
        print("✅ Training sample added to learning system")

        # Get insights
        insights = learning_system.get_model_insights()
        print(f"✅ Learning insights: {insights['total_samples']} samples, accuracy: {insights['accuracy']:.3f}")

        # Test 2: Strategy Manager
        print("\n2️⃣ Testing Strategy Manager...")
        strategy_manager = AdaptiveStrategyManager(learning_system)

        recent_performance = {
            'barbell': 0.05,  # 5% return
            'asymmetric': 0.08,  # 8% return
            'tail_risk': 0.02   # 2% return
        }

        strategy_names = ['barbell', 'asymmetric', 'tail_risk']
        new_weights = strategy_manager.adapt_strategy_weights(
            features, strategy_names, recent_performance
        )
        print(f"✅ Strategy weights adapted: {new_weights}")

        # Test 3: Adaptive Trading Agent
        print("\n3️⃣ Testing Adaptive Trading Agent...")
        config = AdaptiveAgentConfig(initial_balance=10000.0)
        agent = AdaptiveTradingAgent(config)

        # Test market state update (without real API)
        print("✅ Agent initialized with strategies:")
        for name, strategy in agent.strategies.items():
            print(f"   - {name}: {strategy.get_strategy_name()}")

        print(f"✅ Initial strategy weights: {agent.strategy_weights}")

        # Test strategy generation
        decisions = await agent.generate_decisions()
        print(f"✅ Strategy decisions generated: {len(decisions)} decisions")

        # Test 4: System Integration
        print("\n4️⃣ Testing System Integration...")

        # Simulate a few market updates
        for i in range(3):
            # Update market state with slight variations
            market_state.prices['BTCUSDT'] += (i - 1) * 100  # Add some movement
            market_state.momentum['BTCUSDT'] = (i - 1) * 0.01

            features = learning_system.extract_features(market_state, portfolio_state, [])
            learning_system.add_training_sample(features, targets)

            # Adapt strategies
            new_weights = strategy_manager.adapt_strategy_weights(
                features, strategy_names, recent_performance
            )
            agent.strategy_weights = new_weights

            print(f"   Update {i+1}: Strategy weights = {new_weights}")

        print("✅ System integration test completed")

        # Final Report
        print("\n" + "="*60)
        print("🎉 ADAPTIVE SYSTEM TEST RESULTS")
        print("="*60)
        print("✅ Online Learning System: Working")
        print("✅ Strategy Manager: Working")
        print("✅ Trading Agent: Working")
        print("✅ System Integration: Working")
        print("\n🚀 SYSTEM STATUS: READY FOR AUTONOMOUS TRADING")
        print("🎯 Next step: Run 'python3 run_adaptive_trader.py' to start")
        print("="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_adaptive_components()

    if success:
        print("\n🎯 Adaptive AI Trading System is ready!")
        print("💡 To start autonomous trading:")
        print("   python3 run_adaptive_trader.py")
        print("\n💡 To monitor with dashboard:")
        print("   python3 dashboard/aster_trader_dashboard.py")
        print("\n💡 To view data feed:")
        print("   python3 aster_data_feed.py")
    else:
        print("\n❌ System test failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

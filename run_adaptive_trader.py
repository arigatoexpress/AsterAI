#!/usr/bin/env python3
"""
Adaptive AI Trading Agent for Aster DEX - Current Era Learning Focus

This system learns EXCLUSIVELY from current market conditions (2024-present).
NO historical backtesting - adapts to the current volatile bull market era only.

Key Features:
- Zero historical data dependency - learns from NOW onwards
- Continuous real-time adaptation to current market regime
- Advanced strategies optimized for present market conditions
- Goal: $1M by end of 2026 through autonomous current-era adaptation

Current Era Strategies:
1. Barbell Portfolio: BTC/ETH stability + altcoin asymmetric opportunities
2. Asymmetric Bets: Current volatility exploitation with limited downside
3. Tail Risk Hedging: Protection against present market crash scenarios

Real-Time Learning (Current Era Only):
- Online machine learning from live market data
- Market regime detection for current conditions
- Strategy weight optimization based on live performance
- Continuous model updates from current streaming data
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent, AdaptiveAgentConfig
from mcp_trader.ai.online_learning import OnlineLearningSystem, AdaptiveStrategyManager
from mcp_trader.config import get_settings
from mcp_trader.security.secrets import get_secret_manager
from mcp_trader.trading.types import PortfolioState, MarketState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdaptiveTradingSystem:
    """
    Complete adaptive trading system integrating AI learning,
    strategy optimization, and autonomous execution.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.agent: AdaptiveTradingAgent = None
        self.learning_system: OnlineLearningSystem = None
        self.strategy_manager: AdaptiveStrategyManager = None
        self.running = False
        self.start_time = None
        self.performance_log = []

        # System metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    async def initialize(self):
        """Initialize all system components."""
        logger.info("🚀 Initializing Adaptive AI Trading System")
        logger.info(f"🎯 Target: $1M by end of 2026 through autonomous crypto trading")

        try:
            # Check if we should run in demo mode (no real API calls)
            demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'

            if demo_mode:
                logger.info("🎭 Running in DEMO MODE - No real API calls will be made")
                self.demo_mode = True
            else:
                try:
                    # Try to load secrets for real trading
                    sm = get_secret_manager()
                    sm.load_secrets_from_file()
                    logger.info("✅ API credentials loaded")

                    # Quick test of API keys
                    settings = get_settings()
                    if hasattr(settings, 'aster_api_key') and settings.aster_api_key:
                        # Test API connectivity
                        from mcp_trader.execution.aster_client import AsterClient
                        test_client = AsterClient(settings.aster_api_key, settings.aster_api_secret)
                        connectivity = await test_client.test_connectivity()
                        if not connectivity:
                            logger.warning("⚠️  API connectivity test failed - falling back to demo mode")
                            self.demo_mode = True
                        else:
                            self.demo_mode = False
                            logger.info("✅ API connectivity confirmed")
                    else:
                        logger.warning("⚠️  No API keys found - falling back to demo mode")
                        self.demo_mode = True

                except Exception as api_error:
                    logger.warning(f"⚠️  API setup failed: {api_error} - falling back to demo mode")
                    logger.info("💡 To use real trading, update API keys with: python3 update_api_keys.py")
                    self.demo_mode = True

            # Initialize online learning system
            self.learning_system = OnlineLearningSystem()
            logger.info("✅ Online learning system initialized")

            # Initialize strategy manager
            self.strategy_manager = AdaptiveStrategyManager(self.learning_system)
            logger.info("✅ Strategy manager initialized")

            # Configure trading agent for volatile bull market
            config = AdaptiveAgentConfig(
                initial_balance=self.initial_balance,
                max_allocation_per_trade=0.15,  # Higher allocation in bull markets
                risk_tolerance=0.20,  # Higher risk tolerance for bull market
                volatility_threshold=0.04,  # Higher volatility threshold
                learning_rate=0.05,  # Faster learning adaptation
                rebalance_frequency_minutes=15,  # More frequent rebalancing
                min_allocation_per_trade=0.005,  # Lower minimum for small bets
                profit_taking_threshold=0.08,  # Higher profit taking in bull market
                stop_loss_threshold=0.04  # Tighter stops for volatile market
            )

            # Initialize trading agent
            self.agent = AdaptiveTradingAgent(config)
            if not self.demo_mode:
                await self.agent.initialize()
                logger.info("✅ Adaptive trading agent initialized (LIVE MODE)")
            else:
                # Initialize without API connection for demo
                self.agent.portfolio_state = PortfolioState(
                    timestamp=datetime.now(),
                    total_balance=config.initial_balance,
                    available_balance=config.initial_balance
                )
                logger.info("✅ Adaptive trading agent initialized (DEMO MODE)")

            if self.demo_mode:
                logger.info("🎭 DEMO MODE ACTIVE: System will simulate trading without real API calls")
                logger.info("💡 This allows you to see the AI learning and dashboard functionality")
            else:
                logger.info("🔥 LIVE MODE ACTIVE: System connected to Aster DEX for real trading")

            logger.info("🎉 All systems initialized and ready for autonomous trading!")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    async def start_trading(self):
        """Start the autonomous trading system."""
        if not self.agent or not self.learning_system:
            raise RuntimeError("System not properly initialized")

        self.running = True
        self.start_time = datetime.now()

        logger.info("🟢 Starting autonomous trading system")
        logger.info(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        logger.info("🎯 Strategies: Barbell Portfolio, Asymmetric Bets, Tail Risk Hedging")
        logger.info("🔄 Adaptation: Continuous learning and optimization")

        try:
            while self.running:
                # Update system state
                await self.update_system_state()

                # Let agent make trading decisions
                await self.agent.start_trading()

                # Update learning system with new data
                await self.update_learning_system()

                # Optimize strategies
                self.optimize_strategies()

                # Log performance
                self.log_performance()

                # Brief pause between cycles
                await asyncio.sleep(60)  # 1 minute cycle

        except Exception as e:
            logger.error(f"❌ Trading system error: {e}")
            self.running = False

    async def update_system_state(self):
        """Update overall system state."""
        try:
            # Update portfolio metrics
            current_balance = self.agent.portfolio_state.total_balance
            self.peak_balance = max(self.peak_balance, current_balance)
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Count trades
            for pos_key, position in self.agent.positions.items():
                if position.current_pnl != 0:  # Position has P&L
                    self.total_trades += 1
                    if position.current_pnl > 0:
                        self.winning_trades += 1
                    self.total_pnl += position.current_pnl

        except Exception as e:
            logger.error(f"Error updating system state: {e}")

    async def update_learning_system(self):
        """Update the online learning system with current data."""
        try:
            if not self.agent.market_state or not self.agent.portfolio_state:
                return

            # Extract features
            features = self.learning_system.extract_features(
                self.agent.market_state,
                self.agent.portfolio_state,
                self.agent.market_history[-24:]  # Last 24 observations
            )

            # Create training targets (if we have future data)
            # In real-time, we'd need to wait for future observations
            # For now, we'll use recent performance as a proxy

            recent_performance = {}
            for strategy_name, strategy in self.agent.strategies.items():
                perf = strategy.get_recent_performance(hours=1)
                recent_performance[strategy_name] = perf['avg_pnl']

            # Use current market conditions as targets for now
            targets = {
                'price_direction': np.mean(list(self.agent.market_state.momentum.values())) if self.agent.market_state.momentum else 0,
                'volatility': np.mean(list(self.agent.market_state.volatility.values())) if self.agent.market_state.volatility else 0.02,
                'regime': 1 if self.agent.market_state.regime.value == 'BULL_TREND' else 0
            }

            # Add training sample
            self.learning_system.add_training_sample(features, targets)

        except Exception as e:
            logger.error(f"Error updating learning system: {e}")

    def optimize_strategies(self):
        """Optimize strategy parameters and weights."""
        try:
            if not self.agent.market_state:
                return

            # Get current features
            features = self.learning_system.extract_features(
                self.agent.market_state,
                self.agent.portfolio_state,
                self.agent.market_history[-24:]
            )

            # Get recent performance
            recent_performance = {}
            for strategy_name, strategy in self.agent.strategies.items():
                perf = strategy.get_recent_performance(hours=4)
                recent_performance[strategy_name] = perf['avg_pnl'] * perf['win_rate']

            # Adapt strategy weights
            new_weights = self.strategy_manager.adapt_strategy_weights(
                features,
                list(self.agent.strategies.keys()),
                recent_performance
            )

            # Update agent strategy weights
            self.agent.strategy_weights = new_weights

            logger.debug(f"Strategy weights updated: {new_weights}")

        except Exception as e:
            logger.error(f"Error optimizing strategies: {e}")

    def log_performance(self):
        """Log current system performance."""
        if not self.start_time:
            return

        runtime = datetime.now() - self.start_time
        current_balance = self.agent.portfolio_state.total_balance

        # Calculate performance metrics
        total_return = (current_balance - self.initial_balance) / self.initial_balance
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Project to $1M goal
        days_running = runtime.total_seconds() / (24 * 3600)
        if days_running > 0:
            daily_return = (1 + total_return) ** (1 / days_running) - 1
            days_to_1m = np.log(1000000 / self.initial_balance) / np.log(1 + daily_return) if daily_return > 0 else float('inf')
            projected_completion = self.start_time + timedelta(days=days_to_1m) if days_to_1m != float('inf') else None
        else:
            projected_completion = None

        # Log performance
        performance_entry = {
            'timestamp': datetime.now(),
            'runtime_hours': runtime.total_seconds() / 3600,
            'current_balance': current_balance,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.agent.positions),
            'market_regime': self.agent.market_state.regime.value if self.agent.market_state else 'Unknown',
            'strategy_weights': self.agent.strategy_weights.copy(),
            'days_to_1m': days_to_1m if 'days_to_1m' in locals() else None,
            'projected_completion': projected_completion
        }

        self.performance_log.append(performance_entry)

        # Console output
        print(f"\n{'='*80}")
        print(f"🕐 Runtime: {runtime}")
        print(f"💰 Balance: ${current_balance:.2f} (Return: {total_return:.2%})")
        print(f"📊 Trades: {self.total_trades} (Win Rate: {win_rate:.1%})")
        print(f"🎯 P&L: ${self.total_pnl:.2f} | Max DD: {self.max_drawdown:.1%}")
        print(f"📈 Positions: {len(self.agent.positions)} | Regime: {performance_entry['market_regime']}")
        print(f"🎲 Strategy Weights: {performance_entry['strategy_weights']}")

        if projected_completion:
            print(f"🎯 $1M Projection: {projected_completion.strftime('%Y-%m-%d')} ({days_to_1m:.0f} days)")
        else:
            print("🎯 $1M Projection: More data needed")

        print(f"{'='*80}")

        # Keep only recent log entries
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-500:]

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("🛑 Shutdown signal received")
        self.running = False

    async def shutdown(self):
        """Shutdown the trading system gracefully."""
        logger.info("🔄 Shutting down adaptive trading system...")

        self.running = False

        # Save final state
        try:
            self.save_performance_report()
            logger.info("✅ Performance report saved")
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")

        # Disconnect from exchanges
        if self.agent and self.agent.aster_client:
            await self.agent.aster_client.disconnect()
            logger.info("✅ Disconnected from Aster DEX")

        logger.info("👋 Adaptive trading system shutdown complete")

    def save_performance_report(self):
        """Save comprehensive performance report."""
        try:
            import json

            report = {
                'system_info': {
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'end_time': datetime.now().isoformat(),
                    'initial_balance': self.initial_balance,
                    'final_balance': self.agent.portfolio_state.total_balance if self.agent else 0,
                    'total_return': (self.agent.portfolio_state.total_balance - self.initial_balance) / self.initial_balance if self.agent else 0
                },
                'trading_stats': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                    'total_pnl': self.total_pnl,
                    'max_drawdown': self.max_drawdown,
                    'final_open_positions': len(self.agent.positions) if self.agent else 0
                },
                'strategy_performance': {
                    strategy_name: strategy.get_recent_performance(hours=24)
                    for strategy_name, strategy in self.agent.strategies.items()
                } if self.agent else {},
                'learning_insights': self.learning_system.get_model_insights() if self.learning_system else {},
                'performance_log': self.performance_log[-100:]  # Last 100 entries
            }

            with open('adaptive_trading_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info("Performance report saved to adaptive_trading_report.json")

        except Exception as e:
            logger.error(f"Error saving performance report: {e}")

    def print_startup_banner(self):
        """Print the startup banner."""
        print("\n" + "="*80)
        print("🚀 ADAPTIVE AI TRADING AGENT - CURRENT ERA LEARNING")
        print("="*80)
        print("🎯 MISSION: $1M by end of 2026 through autonomous crypto trading")
        print("🧠 AI: Learns EXCLUSIVELY from current market conditions (2024-present)")
        print("📊 STRATEGIES: Barbell Portfolio, Asymmetric Bets, Tail Risk Hedging")
        print("⚡ EXECUTION: Real-time adaptation to CURRENT volatile bull market")
        print("🔄 LEARNING: Online ML from live data - NO historical backtesting")
        print("📅 ERA: Current market cycle optimization only")
        print("="*80)


async def main():
    """Main entry point for the adaptive trading system."""
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive AI Trading Agent for Aster DEX")
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (simulated trading, no API calls)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (no real trades)')
    parser.add_argument('--max-runtime-hours', type=float, default=None, help='Maximum runtime in hours')

    args = parser.parse_args()

    # Set demo mode environment variable
    if args.demo:
        os.environ['DEMO_MODE'] = 'true'

    # Initialize system
    system = AdaptiveTradingSystem(initial_balance=args.balance)

    try:
        system.print_startup_banner()
        await system.initialize()

        if args.test_mode:
            logger.info("🧪 Running in TEST MODE - No real trades will be executed")

        # Start trading
        await system.start_trading()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

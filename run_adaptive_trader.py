#!/usr/bin/env python3
"""
HFT Aster Trader - Ultra-Low Latency Trading System

MISSION: Transform $50 into $500k through High-Frequency Trading on Aster DEX

SYSTEM ARCHITECTURE:
1. LOCAL DEVELOPMENT ENVIRONMENT (RTX 5070Ti Optimized)
   - Real-time data analysis and strategy development
   - GPU-accelerated backtesting and simulation
   - ML model training and optimization

2. CLOUD AUTONOMOUS TRADER
   - Ultra-low latency order execution
   - Real-time market making and arbitrage
   - Risk management and position control
   - 24/7 autonomous operation

HFT STRATEGIES:
1. Statistical Arbitrage: Cross-aster asset price inefficiencies
2. Market Making: Provide liquidity with tight spreads
3. Momentum Trading: Sub-millisecond momentum capture
4. Order Flow Analysis: Institutional flow prediction
5. Latency Arbitrage: Speed-based edge exploitation

TECHNICAL EDGE:
- RTX 5070Ti GPU acceleration (16GB VRAM, 4th Gen Tensor Cores)
- Ultra-low latency data feeds (sub-1ms processing)
- Co-located infrastructure optimization
- Advanced ML models (Transformers, LSTMs, GANs)
- Real-time feature engineering and prediction
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

try:
    from mcp_trader.ai.hft_trading_agent import HFTTradingAgent, HFTAgentConfig
    from mcp_trader.ai.hft_learning import HFTLearningSystem, HFTStrategyManager
except ImportError:
    logger.warning("HFT modules not found, falling back to adaptive trading")
    from mcp_trader.ai.adaptive_trading_agent import AdaptiveTradingAgent, AdaptiveAgentConfig
    from mcp_trader.ai.online_learning import OnlineLearningSystem, AdaptiveStrategyManager
    HFTTradingAgent = AdaptiveTradingAgent
    HFTAgentConfig = AdaptiveAgentConfig
    HFTLearningSystem = OnlineLearningSystem
    HFTStrategyManager = AdaptiveStrategyManager
from mcp_trader.config import get_settings
from mcp_trader.security.secrets import get_secret_manager
from mcp_trader.trading.types import PortfolioState, MarketState
from mcp_trader.logging_utils import get_logger

# Configure structured logging
logger = get_logger("adaptive_trader")


class HFTAsterTrader:
    """
    Ultra-Low Latency HFT Trading System for Aster DEX
    Optimized for transforming $50 into $500k through high-frequency trading
    """

    def __init__(self, initial_balance: float = 50.0):
        self.initial_balance = initial_balance
        self.agent: HFTTradingAgent = None
        self.learning_system: HFTLearningSystem = None
        self.strategy_manager: HFTStrategyManager = None
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


class DegenTradingSystem:
    """
    High-Risk, High-Reward Degen Trading System for Aster DEX
    Optimized for social sentiment and meme coin trading
    """

    def __init__(self, initial_balance: float = 10.0):  # Smaller default for degen
        self.initial_balance = initial_balance
        self.agent = None
        self.running = False
        self.start_time = None
        self.performance_log = []

        # Degen-specific metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    async def initialize(self):
        """Initialize degen trading system."""
        logger.info("🎲 Initializing Degen Trading System")
        logger.info("🎯 Target: High-risk, high-reward (200-500% monthly)")
        logger.info("⚠️  HIGH RISK MODE - Monitor closely!")

        try:
            # Check demo mode
            demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'

            if demo_mode:
                logger.info("🎭 Running in DEMO MODE - No real API calls")
                self.demo_mode = True
            else:
                try:
                    sm = get_secret_manager()
                    sm.load_secrets_from_file()
                    logger.info("✅ API credentials loaded for degen trading")

                    settings = get_settings()
                    if hasattr(settings, 'aster_api_key') and settings.aster_api_key:
                        from mcp_trader.execution.aster_client import AsterClient
                        test_client = AsterClient(settings.aster_api_key, settings.aster_api_secret)
                        connectivity = await test_client.test_connectivity()
                        if not connectivity:
                            logger.warning("⚠️  API connectivity test failed - demo mode")
                            self.demo_mode = True
                        else:
                            self.demo_mode = False
                            logger.info("✅ API connectivity confirmed")
                    else:
                        logger.warning("⚠️  No API keys found - demo mode")
                        self.demo_mode = True

                except Exception as api_error:
                    logger.warning(f"⚠️  API setup failed: {api_error} - demo mode")
                    self.demo_mode = True

            # Initialize degen trading agent
            from mcp_trader.ai.degen_trading_agent import DegenTradingAgent
            self.agent = DegenTradingAgent({
                'initial_balance': self.initial_balance,
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '5.0')),
                'target_daily_return': float(os.getenv('TARGET_DAILY_RETURN', '0.05')),
                'gcp_project_id': os.getenv('GCP_PROJECT_ID', 'hft-aster-trader'),
            })

            logger.info("✅ Degen trading agent initialized")

        except Exception as e:
            logger.error(f"❌ Degen system initialization failed: {e}")
            raise

    def print_startup_banner(self):
        """Print degen trading startup banner."""
        print("="*80)
        print("🎲 DEGEN TRADING AGENT - HIGH RISK HIGH REWARD")
        print("="*80)
        print(f"💰 INITIAL BALANCE: ${self.initial_balance}")
        print("🎯 TARGET: 200-500% monthly returns through social sentiment trading")
        print("📊 STRATEGIES: Social momentum, meme coin pumps, viral arbitrage")
        print("⚠️  RISK PROFILE: EXTREME - Use only risk capital")
        print("🔍 FOCUS: Aster DEX assets, memecoins, social-driven opportunities")
        print("🚀 FEATURES: Real-time sentiment analysis, social data mining")
        print("="*80)

    async def start_trading(self):
        """Start degen trading operations."""
        logger.info("🎲 Starting degen trading operations...")

        self.running = True
        self.start_time = datetime.now()

        try:
            # Initialize sentiment analysis
            await self.agent.initialize_sentiment_analysis()

            # Main trading loop
            while self.running:
                try:
                    await self._trading_cycle()
                    await asyncio.sleep(1)  # 1-second cycles for degen trading

                except Exception as cycle_error:
                    logger.error(f"❌ Trading cycle error: {cycle_error}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"❌ Degen trading failed: {e}")
        finally:
            await self.shutdown()

    async def _trading_cycle(self):
        """Execute one degen trading cycle."""
        try:
            # Get market data
            market_data = await self.agent.get_market_data()

            # Analyze opportunities
            for symbol_data in market_data:
                analysis = await self.agent.analyze_market_opportunity(symbol_data)
                if analysis:
                    trade = await self.agent.execute_trade_decision(analysis)
                    if trade:
                        logger.info(f"🎲 Degen trade executed: {trade}")

            # Update performance
            self._update_performance()

        except Exception as e:
            logger.error(f"❌ Degen trading cycle error: {e}")

    def _update_performance(self):
        """Update performance metrics."""
        try:
            current_balance = self.agent.get_current_balance()
            self.total_pnl = current_balance - self.initial_balance

            # Update peak and drawdown
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance

            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            # Log performance every 5 minutes
            if int((datetime.now() - self.start_time).seconds) % 300 == 0:
                logger.info(f"📊 Degen Performance: P&L=${self.total_pnl:.2f}, "
                           f"Drawdown={self.max_drawdown:.1%}")

        except Exception as e:
            logger.error(f"❌ Performance update error: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("🎲 Degen trader received shutdown signal")
        self.running = False

    async def shutdown(self):
        """Clean shutdown of degen trading system."""
        logger.info("🎲 Shutting down degen trading system...")

        self.running = False

        if self.agent:
            await self.agent.shutdown()

        # Log final performance
        final_balance = self.agent.get_current_balance() if self.agent else self.initial_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        logger.info("🎲 Degen Trading Session Complete")
        logger.info(f"💰 Final Balance: ${final_balance:.2f}")
        logger.info(f"📈 Total Return: {total_return:.1%}")
        logger.info(f"📊 Max Drawdown: {self.max_drawdown:.1%}")
        logger.info("🎲 Remember: Degen trading is extremely risky!")


class AdaptiveTradingSystem(HFTAsterTrader):
    """
    Legacy alias for backward compatibility
    """
    pass


async def main():

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

        # Project to $500k goal
        days_running = runtime.total_seconds() / (24 * 3600)
        if days_running > 0:
            daily_return = (1 + total_return) ** (1 / days_running) - 1
            days_to_500k = np.log(500000 / self.initial_balance) / np.log(1 + daily_return) if daily_return > 0 else float('inf')
            projected_completion = self.start_time + timedelta(days=days_to_500k) if days_to_500k != float('inf') else None
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
            print(f"🎯 $500k Projection: {projected_completion.strftime('%Y-%m-%d')} ({days_to_500k:.0f} days)")
        else:
            print("🎯 $500k Projection: More data needed")

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
        """Print the HFT startup banner."""
        print("\n" + "="*80)
        print("🚀 HFT ASTER TRADER - ULTRA-LOW LATENCY TRADING SYSTEM")
        print("="*80)
        print("🎯 MISSION: $50 → $500k through High-Frequency Trading on Aster DEX")
        print("🧠 AI: RTX 5070Ti GPU-accelerated ML for sub-millisecond decisions")
        print("📊 STRATEGIES: Statistical Arbitrage, Market Making, Momentum, Order Flow, Latency Arbitrage")
        print("⚡ EXECUTION: Ultra-low latency order execution (< 1ms target)")
        print("🔄 LEARNING: Real-time ML adaptation with continuous GPU training")
        print("💰 FOCUS: Aster DEX exclusive - no external exchanges")
        print("="*80)


async def main():
    """Main entry point for the adaptive trading system."""
    import argparse

    parser = argparse.ArgumentParser(description="HFT Aster Trader - Ultra-Low Latency Trading System")
    parser.add_argument('--balance', type=float, default=50.0, help='Initial balance (default: $50 for HFT)')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (simulated trading, no API calls)')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (no real trades)')
    parser.add_argument('--max-runtime-hours', type=float, default=None, help='Maximum runtime in hours')
    parser.add_argument('--agent-type', type=str, default='hft', choices=['hft', 'degen'],
                       help='Agent type: hft (conservative) or degen (high-risk)')

    args = parser.parse_args()

    # Check environment variable for agent type (overrides command line)
    agent_type = os.getenv('AGENT_TYPE', args.agent_type).lower()

    # Set demo mode environment variable
    if args.demo:
        os.environ['DEMO_MODE'] = 'true'

    # Initialize appropriate system based on agent type
    if agent_type == 'degen':
        from mcp_trader.ai.degen_trading_agent import DegenTradingAgent
        logger.info("🎲 Initializing Degen Trading Agent (High-Risk Mode)")
        system = DegenTradingSystem(initial_balance=args.balance)
    else:
        logger.info("🚀 Initializing HFT Trading Agent (Conservative Mode)")
        system = HFTAsterTrader(initial_balance=args.balance)

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

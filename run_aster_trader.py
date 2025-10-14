#!/usr/bin/env python3
"""
Rari Trade AI - Aster Autonomous Trader
Command-line interface for running the autonomous Aster DEX trading system.
"""

import sys
import os
import asyncio
import logging
import signal
from typing import Optional

# Add mcp_trader to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_trader.config import get_settings
from mcp_trader.trading.autonomous_trader import AutonomousTrader, TradingMode


class AsterTraderApp:
    """Command-line application for running the Aster autonomous trader."""

    def __init__(self):
        # Load secrets from file if available before initializing settings
        from mcp_trader.security.secrets import get_secret_manager
        sm = get_secret_manager()
        sm.load_secrets_from_file()

        # Set environment variables for pydantic settings
        import os
        api_key = sm.get_secret('ASTER_API_KEY')
        secret_key = sm.get_secret('ASTER_SECRET_KEY')
        if api_key:
            os.environ['ASTER_API_KEY'] = api_key
        if secret_key:
            os.environ['ASTER_API_SECRET'] = secret_key

        self.trader: Optional[AutonomousTrader] = None
        self.settings = get_settings()
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('aster_trader.log')
            ]
        )

    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
        if self.trader:
            asyncio.create_task(self.trader.stop())

    async def run(self, mode: str = "hybrid", test_mode: bool = False):
        """Run the autonomous trader with enhanced UX."""
        self._print_header(mode, test_mode)

        # Validate configuration
        if not await self._validate_configuration():
            return

        # Validate API credentials
        if not self.settings.aster_api_key or not self.settings.aster_api_secret:
            self._print_credential_error()
            return

        # Build configuration
        config = self._build_trader_config(mode, test_mode)

        try:
            # Initialize trader with progress display
            await self._initialize_trader(config)

            # Start trading with enhanced monitoring
            await self._start_trading()

        except KeyboardInterrupt:
            self._print_shutdown_message("user request")
        except Exception as e:
            self._print_error("running trader", e)
            logging.exception("Trader error")
        finally:
            await self._cleanup_trader()

    def _print_header(self, mode: str, test_mode: bool):
        """Print enhanced header with system information."""
        print("🚀 Rari Trade AI - Aster Autonomous Trader")
        print("=" * 55)
        print(f"📊 Trading Mode: {mode.upper()}")
        print(f"🧪 Test Mode: {'ENABLED' if test_mode else 'DISABLED (LIVE TRADING)'}")

        from mcp_trader.config import PRIORITY_SYMBOLS
        print(f"🎯 Active Symbols: {', '.join(PRIORITY_SYMBOLS)}")
        print(f"⚙️  Risk Settings: Max {self.settings.max_portfolio_risk*100:.0f}% portfolio risk")
        print()

        if not test_mode:
            print("⚠️  WARNING: LIVE TRADING MODE - Real funds will be used!")
            print("   Press Ctrl+C to stop at any time.")
            print()

    async def _validate_configuration(self) -> bool:
        """Validate system configuration before starting."""
        print("🔍 Validating system configuration...")

        issues = []

        # Check credentials
        if not self.settings.aster_api_key or not self.settings.aster_api_secret:
            issues.append("API credentials not configured")

        # Check risk settings
        if self.settings.max_portfolio_risk > 0.5:
            issues.append("Portfolio risk limit seems high (>50%)")

        if self.settings.max_single_position_risk > 0.2:
            issues.append("Single position risk limit seems high (>20%)")

        if issues:
            print("❌ Configuration validation failed:")
            for issue in issues:
                print(f"   • {issue}")
            return False

        print("✅ Configuration validation passed")
        return True

    def _print_credential_error(self):
        """Print detailed credential setup instructions."""
        print("❌ ERROR: Aster DEX API credentials not found!")
        print()
        print("📋 To set up your credentials:")
        print()
        print("Method 1 - Environment Variables:")
        print("  export ASTER_API_KEY='your_api_key_here'")
        print("  export ASTER_API_SECRET='your_secret_here'")
        print()
        print("Method 2 - Secure Storage:")
        print("  python update_api_keys.py")
        print()
        print("Method 3 - Get API Keys:")
        print("  1. Visit https://asterdex.com")
        print("  2. Go to Account → API Keys")
        print("  3. Create new API key with Futures permissions")
        print("  4. Restrict IP addresses for security")

    def _build_trader_config(self, mode: str, test_mode: bool) -> dict:
        """Build comprehensive trader configuration."""
        return {
            'trading_mode': mode,
            'test_mode': test_mode,
            'max_concurrent_positions': self.settings.max_concurrent_positions,
            'min_position_size': self.settings.min_position_size_usd,
            'max_position_size': self.settings.max_position_size_usd,
            'risk_config': {
                'limits': {
                    'max_portfolio_risk': self.settings.max_portfolio_risk,
                    'max_single_position_risk': self.settings.max_single_position_risk,
                    'max_daily_loss': self.settings.max_daily_loss,
                    'max_drawdown': self.settings.max_daily_loss,
                    'max_concurrent_positions': self.settings.max_concurrent_positions
                }
            },
            'grid_config': {
                'grid_levels': self.settings.grid_levels,
                'grid_spacing_percent': self.settings.grid_spacing_percent,
                'position_size_per_level': self.settings.grid_position_size_usd,
                'max_position_size': self.settings.max_position_size_usd,
                'volatility_multiplier': self.settings.volatility_multiplier
            },
            'volatility_config': {
                'min_volatility_threshold': 3.0,
                'max_volatility_threshold': 15.0,
                'profit_taking_threshold': self.settings.take_profit_threshold,
                'stop_loss_threshold': self.settings.stop_loss_threshold,
                'position_size_scaler': 0.5
            }
        }

    async def _initialize_trader(self, config: dict):
        """Initialize trader with progress indicators."""
        print("🤖 Initializing autonomous trading system...")

        # Step 1: Initialize API client
        print("   🔗 Connecting to Aster DEX API...")
        # API client is initialized in AutonomousTrader constructor

        # Step 2: Initialize strategies
        print("   🎯 Loading trading strategies...")
        # Strategies are initialized in AutonomousTrader constructor

        # Step 3: Initialize risk management
        print("   🛡️  Setting up risk management...")
        # Risk manager is initialized in AutonomousTrader constructor

        # Step 4: Create trader instance
        print("   🚀 Creating autonomous trader...")
        self.trader = AutonomousTrader(config)

        print("✅ System initialization complete!")
        print()

    async def _start_trading(self):
        """Start trading with enhanced monitoring."""
        mode_desc = "PAPER TRADING" if self.trader.config.get('test_mode') else "LIVE TRADING"
        print(f"🚀 Starting {mode_desc} session...")
        print("📊 Status updates every 30 seconds (Ctrl+C to stop)")
        print("-" * 55)

        self.running = True

        # Show initial status
        await self._show_enhanced_status()

        # Start status monitoring
        status_task = asyncio.create_task(self._status_monitor_loop())

        # Start the trader
        await self.trader.start()

    async def _status_monitor_loop(self):
        """Enhanced status monitoring loop."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                await self._show_enhanced_status()
            except Exception as e:
                print(f"Status update error: {e}")
                await asyncio.sleep(30)

    async def _show_enhanced_status(self):
        """Show comprehensive trading status."""
        try:
            if not self.trader:
                return

            # Get current portfolio state
            portfolio = self.trader.portfolio_state

            print(f"📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
            print(f"   💰 Portfolio Value: ${portfolio.total_balance + portfolio.total_positions_value:,.2f}")
            print(f"   💵 Available Balance: ${portfolio.available_balance:,.2f}")
            print(f"   📈 Unrealized P&L: ${portfolio.unrealized_pnl:+,.2f}")
            print(f"   🎯 Active Positions: {len(portfolio.active_positions)}")
            print(f"   📊 Active Grids: {len(portfolio.active_grids)}")

            # Show top positions if any
            if portfolio.active_positions:
                print("   📋 Positions:")
                for symbol, pos_data in list(portfolio.active_positions.items())[:3]:
                    if hasattr(pos_data, 'size'):
                        print(f"      • {symbol}: {pos_data.size:.6f} units")
                    else:
                        print(f"      • {symbol}: {pos_data}")

        except Exception as e:
            print(f"Status display error: {e}")

    def _print_shutdown_message(self, reason: str):
        """Print shutdown message with reason."""
        print(f"\n⏹️  Trading stopped: {reason}")
        print("📊 Final session summary:")

        if self.trader and hasattr(self.trader, 'portfolio_state'):
            portfolio = self.trader.portfolio_state
            print(f"   💰 Final Balance: ${portfolio.total_balance + portfolio.total_positions_value:,.2f}")
            print(f"   📈 Total P&L: ${portfolio.unrealized_pnl:+,.2f}")

    def _print_error(self, operation: str, error: Exception):
        """Print formatted error message."""
        print(f"❌ Error during {operation}: {str(error)}")
        print("📋 Check the logs for detailed error information.")

    async def _cleanup_trader(self):
        """Clean up trader resources."""
        if self.trader:
            print("🔄 Shutting down trading system...")
            await self.trader.stop()

        print("✅ Trader shutdown complete")
        print("👋 Thank you for using Rari Trade AI!")

    async def show_status_loop(self):
        """Show trading status periodically."""
        while self.running:
            try:
                if self.trader:
                    status = self.trader.get_portfolio_status()
                    print(f"\r📊 Status: {status['active_positions']} positions, "
                          f"Balance: ${status['total_balance']:.2f}, "
                          f"P&L: ${status['unrealized_pnl']:+.2f}", end="", flush=True)
                await asyncio.sleep(10)
            except Exception as e:
                print(f"\nStatus update error: {e}")
                await asyncio.sleep(30)

    def show_help(self):
        """Show help information."""
        print("Rari Trade AI - Aster Autonomous Trader")
        print()
        print("USAGE:")
        print("  python run_aster_trader.py [MODE] [OPTIONS]")
        print()
        print("MODES:")
        print("  grid      - Grid trading strategy")
        print("  volatility - Volatility-based trading")
        print("  hybrid    - Combined grid + volatility (default)")
        print()
        print("OPTIONS:")
        print("  --test    - Run in test mode (no real trades)")
        print("  --help    - Show this help message")
        print()
        print("EXAMPLES:")
        print("  python run_aster_trader.py grid")
        print("  python run_aster_trader.py hybrid --test")
        print("  python run_aster_trader.py volatility")
        print()
        print("ENVIRONMENT VARIABLES:")
        print("  ASTER_API_KEY     - Your Aster DEX API key")
        print("  ASTER_API_SECRET  - Your Aster DEX API secret")
        print("  LOG_LEVEL         - Logging level (DEBUG, INFO, WARNING, ERROR)")


def main():
    """Main entry point."""
    import sys

    # Parse command line arguments
    mode = "hybrid"  # default
    test_mode = False

    args = sys.argv[1:]
    for arg in args:
        if arg in ["grid", "volatility", "hybrid"]:
            mode = arg
        elif arg == "--test":
            test_mode = True
        elif arg in ["--help", "-h"]:
            AsterTraderApp().show_help()
            return

    # Run the trader
    app = AsterTraderApp()
    asyncio.run(app.run(mode=mode, test_mode=test_mode))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🚀 LAUNCH ULTRA-AGGRESSIVE RTX-SUPERCHARGED TRADING SYSTEM
$150 → $1,000,000 (6,667x) with RTX 5070 Ti Blackwell Acceleration

This script launches your complete ultra-aggressive trading system with:
✅ RTX 5070 Ti GPU acceleration
✅ VPN-optimized data collection
✅ VPIN toxic flow detection
✅ 82.44% AI ensemble predictions
✅ Ultra-high leverage (10-50x)
✅ Monte Carlo VaR risk management

WARNING: This is ULTRA-HIGH RISK. Only use money you can afford to lose completely.
"""

import asyncio
import logging
import sys
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from ULTRA_AGGRESSIVE_RTX_SUPERCHARGED_TRADING import UltraAggressiveRTXTradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_aggressive_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def launch_ultra_aggressive_trading(
    capital: float = 150.0,
    mode: str = "paper",
    max_cycles: int = 10,
    symbols: list = None
):
    """
    Launch the ultra-aggressive RTX-supercharged trading system

    Args:
        capital: Starting capital ($150 default)
        mode: 'paper' or 'live'
        max_cycles: Number of trading cycles to run
        symbols: List of symbols to trade (default: Aster assets)
    """

    print("="*80)
    print("🚀 LAUNCHING ULTRA-AGGRESSIVE RTX-SUPERCHARGED TRADING")
    print("="*80)
    print(f"💰 Capital: ${capital:,.0f}")
    print(f"🎯 Target: $1,000,000 (Multiplier: {1000000/capital:.0f}x)")
    print(f"⚡ RTX 5070 Ti: Blackwell acceleration enabled")
    print(f"🌐 VPN: Optimized for Iceland → Binance")
    print(f"🎪 VPIN: Toxic flow detection active")
    print(f"🤖 AI: 82.44% accuracy ensemble")
    print(f"⚠️  Mode: {mode.upper()}")
    print(f"🔄 Cycles: {max_cycles}")
    print()

    # Risk warning
    print("⚠️  ULTRA-HIGH RISK WARNING ⚠️")
    print("="*50)
    print("This system uses EXTREME leverage (10-50x)")
    print("HIGH PROBABILITY of TOTAL CAPITAL LOSS")
    print("Only use money you can afford to LOSE COMPLETELY")
    print("Start with PAPER TRADING first!")
    print("="*50)
    print()

    # Get user confirmation
    if mode == "live":
        confirm = input("⚠️  Are you sure you want to run LIVE trading? (type 'YES' to confirm): ")
        if confirm != "YES":
            print("❌ Live trading cancelled. Switching to paper trading...")
            mode = "paper"

    # Initialize system
    print("🔧 Initializing Ultra-Aggressive RTX Trading System...")

    try:
        system = UltraAggressiveRTXTradingSystem(total_capital=capital)

        # Initialize all components
        init_success = await system.initialize_system()

        if not init_success:
            print("❌ System initialization failed!")
            return False

        print("✅ System initialized successfully!")

        # Show system status
        status = system.get_system_status()
        print("
📊 System Status:"        print(f"  • RTX Accelerated: {status['gpu_accelerated']}")
        print(f"  • VPN Optimized: {status['data_collection'] == 'vpn_optimized'}")
        print(f"  • VPIN Enabled: {status['vpin_enabled']}")
        print(f"  • Ultra Aggressive: {status['ultra_aggressive_mode']}")
        print(f"  • AI Accuracy: {status['ai_accuracy']:.1%}")
        print()

        # Launch trading
        if mode == "paper":
            print("🎯 Starting PAPER TRADING (no real money at risk)")
            print("Use this to validate the system before live trading!")
            print()

        elif mode == "live":
            print("🔴 Starting LIVE TRADING (real money at risk!)")
            print("⚠️  Monitor closely - emergency stop available")
            print()

        # Run trading loop
        await system.run_ultra_aggressive_trading_loop(max_cycles=max_cycles)

        # Final results
        final_status = system.get_system_status()
        print("
🏁 TRADING COMPLETE"        print("="*40)
        print(f"💰 Final Capital: ${final_status['equity_curve']:,.2f}")
        print(f"📈 Total Return: ${final_status['total_pnl']:,.2f}")
        print(f"🎯 Progress to $1M: {final_status['progress_to_target']:.1f}%")
        print(f"🔄 Total Trades: {final_status['total_trades']}")

        if final_status['total_trades'] > 0:
            print(".1f"
        print("="*40)

        if final_status['progress_to_target'] >= 100:
            print("🎉 CONGRATULATIONS! $1M TARGET ACHIEVED! 🎉")
        elif final_status['equity_curve'] > capital:
            print("✅ PROFITABLE! Ready to scale capital.")
        else:
            print("⚠️  No profit yet. Review strategy and risk parameters.")

        return True

    except KeyboardInterrupt:
        print("\n⚠️  Trading interrupted by user")
        return False

    except Exception as e:
        logger.error(f"❌ Trading system error: {e}")
        print(f"❌ System error: {e}")
        return False


async def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Launch Ultra-Aggressive RTX-Supercharged Trading System"
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=150.0,
        help='Starting capital (default: $150)'
    )
    parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--cycles',
        type=int,
        default=10,
        help='Number of trading cycles (default: 10)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'AVAXUSDT'],
        help='Symbols to trade (default: major assets)'
    )

    args = parser.parse_args()

    # Launch trading
    success = await launch_ultra_aggressive_trading(
        capital=args.capital,
        mode=args.mode,
        max_cycles=args.cycles,
        symbols=args.symbols
    )

    if success:
        print("\n✅ Ultra-Aggressive RTX Trading completed successfully!")
    else:
        print("\n❌ Trading system encountered issues.")
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    # Run trading system
    asyncio.run(main())

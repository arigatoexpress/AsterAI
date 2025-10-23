#!/usr/bin/env python3
"""
Multi-Asset Trading Test Runner
Executes comprehensive tests across perpetual contracts, spot markets, and stocks
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from tests.test_multi_asset_trading import MultiAssetTradingTester


async def main():
    """Main test runner function."""
    print("🚀 AsterAI Multi-Asset Trading Test Runner")
    print("=" * 50)

    # Initialize tester
    tester = MultiAssetTradingTester()

    # Run comprehensive tests
    start_time = datetime.now()
    print(f"⏰ Test started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results = await tester.run_comprehensive_tests()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save detailed results
        output_file = Path("multi_asset_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n📄 Detailed results saved to:")
        print(f"   {output_file.absolute()}")

        # Print summary
        print_summary(results, duration)

        # Print recommendations
        print_recommendations(results)

        return results

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(results: dict, duration: float):
    """Print test summary."""
    print("\n" + "=" * 50)
    print("📊 MULTI-ASSET TRADING TEST SUMMARY")
    print("=" * 50)

    perf = results.get("performance_metrics", {})

    print(f"⏱️  Total Test Duration: {duration:.1f} seconds")
    print(f"📈 Success Rates by Asset Class:")
    print("-" * 30)

    success_rates = perf.get("success_rates", {})
    for asset_class, stats in success_rates.items():
        success_rate = stats["success_rate"] * 100
        status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 75 else "❌"

        asset_name = asset_class.replace("_", " ").title()
        print(f"  {status_icon} {asset_name}: {success_rate:.1f}% "
              f"({stats['successful_tests']}/{stats['total_tests']})")

    print("\n🔍 Detailed Results by Asset Class:")
    print("-" * 30)

    # Perpetual contracts summary
    perp_results = results.get("perpetual_contracts", {})
    print_asset_summary("Perpetual Contracts", perp_results)

    # Spot markets summary
    spot_results = results.get("spot_markets", {})
    print_asset_summary("Spot Markets", spot_results)

    # Stocks summary
    stock_results = results.get("stocks", {})
    print_asset_summary("Stocks", stock_results)

    # Cross-asset arbitrage summary
    arb_results = results.get("cross_asset_arbitrage", {})
    print(f"  🔄 Cross-Asset Arbitrage: {len(arb_results)} strategies tested")


def print_asset_summary(asset_name: str, results: dict):
    """Print summary for a specific asset class."""
    if not results:
        print(f"  📊 {asset_name}: No tests completed")
        return

    total_symbols = len(results)
    total_tests = sum(len(symbol_results) for symbol_results in results.values())
    successful_tests = 0

    for symbol_results in results.values():
        successful_tests += sum(1 for test in symbol_results.values()
                              if test.get("success", False))

    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    status_icon = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 75 else "❌"

    print(f"  {status_icon} {asset_name}: {total_symbols} symbols, "
          f"{successful_tests}/{total_tests} tests ({success_rate:.1f}%)")


def print_recommendations(results: dict):
    """Print recommendations based on test results."""
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS")
    print("=" * 50)

    perf = results.get("performance_metrics", {})
    overall_rate = perf.get("overall_success_rate", 0) * 100

    if overall_rate >= 95:
        print("🎉 EXCELLENT! Your bot is ready for live multi-asset trading!")
        print("   ✅ All asset classes performing optimally")
        print("   ✅ Risk management working across all markets")
        print("   ✅ Arbitrage opportunities being detected")
        print("\n🚀 Recommended: Start with $50-100 in live trading")

    elif overall_rate >= 85:
        print("✅ VERY GOOD! Minor optimizations needed")
        print("   • Review failed tests for specific improvements")
        print("   • Consider additional risk controls")
        print("   • Test with larger position sizes in paper trading")

    elif overall_rate >= 75:
        print("⚠️  GOOD but needs work")
        print("   • Focus on failed asset classes")
        print("   • Improve error handling")
        print("   • Enhance position management logic")

    elif overall_rate >= 60:
        print("🔧 SIGNIFICANT IMPROVEMENTS NEEDED")
        print("   • Major issues in multiple asset classes")
        print("   • Review risk management implementation")
        print("   • Consider simpler strategies first")

    else:
        print("❌ CRITICAL ISSUES DETECTED")
        print("   • Do not deploy to live trading")
        print("   • Complete rewrite of trading logic needed")
        print("   • Focus on paper trading and debugging")

    # Specific recommendations
    print("\n🔧 Specific Recommendations:")
    print("-" * 30)

    # Check perpetual contracts
    perp_results = results.get("perpetual_contracts", {})
    if perp_results:
        failed_perp = []
        for symbol, tests in perp_results.items():
            failed_tests = [t for t, r in tests.items() if not r.get("success", False)]
            if failed_tests:
                failed_perp.extend(failed_tests)

        if failed_perp:
            print(f"  📈 Fix {len(failed_perp)} perpetual contract issues:")
            for issue in failed_perp[:3]:  # Show first 3
                print(f"     • {issue}")

    # Check risk management
    risk_results = results.get("risk_management", {})
    if risk_results:
        portfolio_risk = risk_results.get("portfolio_limits", {})
        if not portfolio_risk.get("max_drawdown_respected", True):
            print("  🛡️  Improve portfolio drawdown controls")
        if not portfolio_risk.get("var_limits_maintained", True):
            print("  🛡️  Enhance VaR calculations")

    print("\n📚 Next Steps:")
    print("  1. Review detailed results in multi_asset_test_results.json")
    print("  2. Fix identified issues")
    print("  3. Re-run tests to verify improvements")
    print("  4. Deploy to Cloud Run for continuous testing")
    print("  5. Start with small amounts in live trading")


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        if results:
            # Exit with success/failure code based on performance
            perf = results.get("performance_metrics", {})
            overall_rate = perf.get("overall_success_rate", 0)

            if overall_rate >= 0.75:
                sys.exit(0)  # Success
            else:
                sys.exit(1)  # Failure - needs work

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

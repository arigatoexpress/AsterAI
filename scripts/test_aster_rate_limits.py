#!/usr/bin/env python3
"""
Test Aster DEX Rate Limits and Data Streams
Ensures we can collect data without hitting limits or using synthetic data.
"""

import asyncio
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_training.aster_dex_data_collector import AsterDEXDataCollector
from mcp_trader.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimitTester:
    """
    Test Aster DEX API rate limits and data stream reliability.
    Ensures safe data collection without synthetic fallbacks.
    """

    def __init__(self):
        self.collector = None
        self.settings = get_settings()

        # Test parameters
        self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ASTERUSDT"]
        self.test_delays = [0.1, 0.2, 0.5, 1.0, 2.0]  # seconds between requests
        self.test_batch_sizes = [1, 5, 10, 20]
        self.test_duration = 60  # seconds per test

        # Rate limiting thresholds (conservative)
        self.max_success_rate = 0.95  # 95% success rate required
        self.max_avg_response_time = 2.0  # 2 seconds max response time
        self.min_data_points_per_request = 10

    async def initialize(self):
        """Initialize the collector."""
        self.collector = AsterDEXDataCollector()
        await self.collector.initialize()
        logger.info("✅ Rate limit tester initialized")

    async def run_comprehensive_test(self) -> Dict:
        """
        Run comprehensive rate limit and reliability testing.
        Returns detailed test results.
        """
        logger.info(f"\n{'='*70}")
        logger.info("COMPREHENSIVE ASTER DEX RATE LIMIT & RELIABILITY TEST")
        logger.info(f"{'='*70}\n")

        # Test connectivity first
        if not await self.test_basic_connectivity():
            return {"error": "Basic connectivity test failed"}

        # Test different rate limits
        rate_limit_results = await self.test_rate_limits()

        # Test data stream reliability
        reliability_results = await self.test_data_reliability()

        # Test concurrent requests
        concurrent_results = await self.test_concurrent_requests()

        # Analyze results and provide recommendations
        analysis = self.analyze_results({
            'rate_limits': rate_limit_results,
            'reliability': reliability_results,
            'concurrent': concurrent_results
        })

        return {
            'timestamp': datetime.now().isoformat(),
            'test_parameters': {
                'symbols': self.test_symbols,
                'delays_tested': self.test_delays,
                'batch_sizes_tested': self.test_batch_sizes,
                'test_duration_seconds': self.test_duration
            },
            'results': {
                'rate_limits': rate_limit_results,
                'reliability': reliability_results,
                'concurrent': concurrent_results
            },
            'analysis': analysis
        }

    async def test_basic_connectivity(self) -> bool:
        """Test basic API connectivity."""
        logger.info("🔗 Testing basic connectivity...")

        try:
            # Test server time
            connected = await self.collector._test_connectivity()
            if connected:
                logger.info("✅ Basic connectivity confirmed")
                return True
            else:
                logger.error("❌ Basic connectivity failed")
                return False
        except Exception as e:
            logger.error(f"❌ Connectivity test error: {e}")
            return False

    async def test_rate_limits(self) -> Dict:
        """Test different request rates to find optimal rate."""
        logger.info("\n⏱️  Testing rate limits...")

        results = {}

        for delay in self.test_delays:
            logger.info(f"   Testing {delay}s delay between requests...")

            # Test for 30 seconds
            start_time = time.time()
            requests = 0
            successes = 0
            response_times = []

            while time.time() - start_time < 30:  # 30 second test
                for symbol in self.test_symbols[:2]:  # Test with 2 symbols
                    try:
                        request_start = time.time()

                        # Test orderbook (lightweight request)
                        orderbook = await self.collector.collect_orderbook_data(symbol)

                        request_time = time.time() - request_start
                        response_times.append(request_time)

                        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                            successes += 1

                        requests += 1

                        # Rate limiting
                        await asyncio.sleep(delay)

                    except Exception as e:
                        logger.debug(f"Request failed: {e}")
                        requests += 1
                        await asyncio.sleep(delay)

            # Calculate metrics
            duration = time.time() - start_time
            requests_per_second = requests / duration
            success_rate = successes / requests if requests > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            results[f"{delay}s"] = {
                'delay_seconds': delay,
                'total_requests': requests,
                'successful_requests': successes,
                'success_rate': success_rate,
                'requests_per_second': requests_per_second,
                'avg_response_time': avg_response_time,
                'duration_seconds': duration
            }

            logger.info(f"     {delay}s: {successes}/{requests} success ({success_rate:.1%}), "
                       f"{requests_per_second:.2f} req/s, {avg_response_time:.2f}s avg response")

        return results

    async def test_data_reliability(self) -> Dict:
        """Test data stream reliability over time."""
        logger.info("\n🔄 Testing data stream reliability...")

        symbol = self.test_symbols[0]  # Test with first symbol
        reliability_data = []

        logger.info(f"   Monitoring {symbol} data streams for {self.test_duration} seconds...")

        start_time = time.time()
        interval = 5  # Check every 5 seconds

        while time.time() - start_time < self.test_duration:
            try:
                # Test orderbook
                orderbook_start = time.time()
                orderbook = await self.collector.collect_orderbook_data(symbol)
                orderbook_time = time.time() - orderbook_start

                # Test recent trades
                trades_start = time.time()
                trades = await self.collector.collect_recent_trades(symbol, limit=5)
                trades_time = time.time() - trades_start

                # Test historical data (1 hour)
                hist_start = time.time()
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                today = datetime.now().strftime("%Y-%m-%d")
                hist_data = await self.collector.collect_historical_data(
                    symbol=symbol,
                    start_date=yesterday,
                    end_date=today,
                    interval="1h"
                )
                hist_time = time.time() - hist_start

                # Record results
                reliability_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'orderbook_success': orderbook is not None and 'bids' in orderbook,
                    'orderbook_response_time': orderbook_time,
                    'trades_success': trades is not None and len(trades) > 0,
                    'trades_response_time': trades_time,
                    'trades_count': len(trades) if trades else 0,
                    'historical_success': not hist_data.empty,
                    'historical_response_time': hist_time,
                    'historical_data_points': len(hist_data)
                })

                logger.debug(f"   ✓ Data check: OB={orderbook is not None}, "
                           f"Trades={len(trades) if trades else 0}, "
                           f"Hist={len(hist_data)} points")

            except Exception as e:
                logger.warning(f"   ⚠️  Data check failed: {e}")
                reliability_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'orderbook_success': False,
                    'orderbook_response_time': 0,
                    'trades_success': False,
                    'trades_response_time': 0,
                    'trades_count': 0,
                    'historical_success': False,
                    'historical_response_time': 0,
                    'historical_data_points': 0
                })

            await asyncio.sleep(interval)

        # Analyze reliability
        df = pd.DataFrame(reliability_data)

        analysis = {
            'total_checks': len(df),
            'orderbook_success_rate': df['orderbook_success'].mean(),
            'trades_success_rate': df['trades_success'].mean(),
            'historical_success_rate': df['historical_success'].mean(),
            'avg_orderbook_response_time': df['orderbook_response_time'].mean(),
            'avg_trades_response_time': df['trades_response_time'].mean(),
            'avg_historical_response_time': df['historical_response_time'].mean(),
            'avg_trades_per_check': df['trades_count'].mean(),
            'avg_historical_points_per_check': df['historical_data_points'].mean()
        }

        logger.info(f"   📊 Reliability Results:")
        logger.info(f"     Orderbook success: {analysis['orderbook_success_rate']:.1%}")
        logger.info(f"     Trades success: {analysis['trades_success_rate']:.1%}")
        logger.info(f"     Historical success: {analysis['historical_success_rate']:.1%}")
        logger.info(f"     Avg response times: OB={analysis['avg_orderbook_response_time']:.2f}s, "
                   f"Trades={analysis['avg_trades_response_time']:.2f}s, "
                   f"Hist={analysis['avg_historical_response_time']:.2f}s")

        return analysis

    async def test_concurrent_requests(self) -> Dict:
        """Test concurrent request handling."""
        logger.info("\n🔄 Testing concurrent request handling...")

        concurrent_results = {}

        for batch_size in self.test_batch_sizes:
            logger.info(f"   Testing {batch_size} concurrent requests...")

            # Create tasks
            tasks = []
            for i in range(batch_size):
                symbol = self.test_symbols[i % len(self.test_symbols)]
                tasks.append(self._single_request_test(symbol))

            # Execute concurrently
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception) and r.get('success'))
            response_times = [r.get('response_time', 0) for r in results
                            if not isinstance(r, Exception) and 'response_time' in r]

            concurrent_results[f"batch_{batch_size}"] = {
                'batch_size': batch_size,
                'total_requests': batch_size,
                'successful_requests': successful,
                'success_rate': successful / batch_size,
                'total_time': total_time,
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0
            }

            logger.info(f"     {batch_size} concurrent: {successful}/{batch_size} success "
                       f"({concurrent_results[f'batch_{batch_size}']['success_rate']:.1%}) in {total_time:.2f}s")

        return concurrent_results

    async def _single_request_test(self, symbol: str) -> Dict:
        """Single request for concurrent testing."""
        try:
            start_time = time.time()
            orderbook = await self.collector.collect_orderbook_data(symbol)
            response_time = time.time() - start_time

            return {
                'symbol': symbol,
                'success': orderbook is not None and 'bids' in orderbook,
                'response_time': response_time
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'success': False,
                'response_time': time.time() - time.time(),  # 0
                'error': str(e)
            }

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze test results and provide recommendations."""
        logger.info(f"\n{'='*70}")
        logger.info("ANALYSIS & RECOMMENDATIONS")
        logger.info(f"{'='*70}\n")

        analysis = {
            'recommended_settings': {},
            'warnings': [],
            'optimal_strategy': {}
        }

        # Analyze rate limits
        rate_limits = results['rate_limits']
        best_delay = None
        best_success_rate = 0

        for delay_key, delay_results in rate_limits.items():
            success_rate = delay_results['success_rate']
            avg_response_time = delay_results['avg_response_time']

            if (success_rate >= self.max_success_rate and
                avg_response_time <= self.max_avg_response_time):
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_delay = float(delay_key.replace('s', ''))

        if best_delay:
            analysis['recommended_settings']['request_delay_seconds'] = best_delay
            analysis['recommended_settings']['max_requests_per_minute'] = 60 / best_delay
            logger.info(f"✅ Recommended delay: {best_delay}s between requests")
            logger.info(f"   Max requests/min: {analysis['recommended_settings']['max_requests_per_minute']:.0f}")
        else:
            analysis['warnings'].append("No delay setting met success criteria")
            logger.warning("⚠️  No delay setting met success criteria (>95% success, <2s response)")

        # Analyze reliability
        reliability = results['reliability']
        if reliability['orderbook_success_rate'] < 0.9:
            analysis['warnings'].append(f"Orderbook reliability low: {reliability['orderbook_success_rate']:.1%}")
        if reliability['historical_success_rate'] < 0.8:
            analysis['warnings'].append(f"Historical data reliability low: {reliability['historical_success_rate']:.1%}")

        # Analyze concurrent performance
        concurrent = results['concurrent']
        best_batch_size = 1
        best_throughput = 0

        for batch_key, batch_results in concurrent.items():
            success_rate = batch_results['success_rate']
            total_time = batch_results['total_time']
            batch_size = batch_results['batch_size']

            if success_rate >= 0.9:  # Only consider reliable batch sizes
                throughput = batch_size / total_time
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size

        analysis['recommended_settings']['optimal_batch_size'] = best_batch_size
        analysis['recommended_settings']['expected_throughput_req_per_sec'] = best_throughput

        logger.info(f"✅ Recommended batch size: {best_batch_size} concurrent requests")
        logger.info(f"   Expected throughput: {best_throughput:.2f} requests/second")

        # Overall assessment
        if not analysis['warnings']:
            analysis['overall_status'] = 'EXCELLENT'
            analysis['optimal_strategy']['approach'] = 'aggressive'
            logger.info("🎉 Overall Status: EXCELLENT - Ready for full data collection")
        elif len(analysis['warnings']) <= 2:
            analysis['overall_status'] = 'GOOD'
            analysis['optimal_strategy']['approach'] = 'moderate'
            logger.info("👍 Overall Status: GOOD - Proceed with recommended settings")
        else:
            analysis['overall_status'] = 'CAUTION'
            analysis['optimal_strategy']['approach'] = 'conservative'
            logger.warning("⚠️  Overall Status: CAUTION - Review warnings before proceeding")

        if analysis['warnings']:
            logger.info("\n⚠️  Warnings:")
            for warning in analysis['warnings']:
                logger.info(f"   • {warning}")

        return analysis

    async def save_test_report(self, results: Dict, output_file: str = "aster_rate_limit_test.json"):
        """Save comprehensive test report."""
        output_path = Path("data") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Test report saved to {output_path}")
        return str(output_path)

    async def close(self):
        """Close connections."""
        if self.collector:
            await self.collector.close()


async def main():
    """Main execution."""
    print("""
================================================================================
          Aster DEX Rate Limit & Data Stream Testing
      Ensuring Reliable Data Collection (No Synthetic Data)
================================================================================
    """)

    tester = RateLimitTester()

    try:
        await tester.initialize()

        # Run comprehensive testing
        results = await tester.run_comprehensive_test()

        if 'error' in results:
            logger.error(f"❌ Testing failed: {results['error']}")
            return

        # Save report
        report_file = await tester.save_test_report(results)

        # Display key findings
        analysis = results['analysis']

        print(f"\n{'='*70}")
        print("RATE LIMIT TEST RESULTS")
        print(f"{'='*70}\n")

        print(f"🎯 Overall Status: {analysis['overall_status']}")
        print(f"📊 Recommended Settings:")

        settings = analysis['recommended_settings']
        if 'request_delay_seconds' in settings:
            print(f"   • Delay between requests: {settings['request_delay_seconds']}s")
            print(f"   • Max requests per minute: {settings['max_requests_per_minute']:.0f}")

        if 'optimal_batch_size' in settings:
            print(f"   • Optimal batch size: {settings['optimal_batch_size']} concurrent requests")
            print(f"   • Expected throughput: {settings['expected_throughput_req_per_sec']:.2f} req/s")

        if analysis['warnings']:
            print(f"\n⚠️  Warnings ({len(analysis['warnings'])}):")
            for warning in analysis['warnings']:
                print(f"   • {warning}")

        print(f"\n📋 Full report saved to: {report_file}")

        print("""
================================================================================
                   🎉 Rate Limit Testing Complete!
    Ready to collect real Aster DEX data safely and reliably
================================================================================

Next steps:
1. Review aster_rate_limit_test.json for detailed results
2. Run scripts/discover_aster_assets.py to find all assets
3. Run scripts/collect_real_aster_data.py for validated collection
4. Train models on real data only (no synthetic data)
        """)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Testing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main())


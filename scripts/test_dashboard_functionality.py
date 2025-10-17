#!/usr/bin/env python3
"""
Comprehensive Dashboard Functionality Test
Tests all dashboard components, data sources, and matrix theme.
"""

import sys
import logging
from pathlib import Path
import time
import requests
import subprocess
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.unified_trading_dashboard import DashboardDataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardTester:
    """Comprehensive dashboard testing suite."""

    def __init__(self):
        self.data_manager = DashboardDataManager()
        self.streamlit_process = None
        self.dashboard_url = "http://localhost:8501"
        self.test_results = {
            "data_sources": {},
            "dashboard_components": {},
            "theme_elements": {},
            "functionality": {}
        }

    def test_data_sources(self):
        """Test all data sources."""
        logger.info("🔍 Testing data sources...")

        try:
            # Test system info
            sys_info = self.data_manager.get_system_info()
            self.test_results["data_sources"]["system_info"] = bool(sys_info)
            logger.info(f"✅ System info: {bool(sys_info)}")

            # Test trading bot status
            bot_status = self.data_manager.get_trading_bot_status()
            self.test_results["data_sources"]["bot_status"] = bool(bot_status)
            logger.info(f"✅ Bot status: {bool(bot_status)}")

            # Test training status
            training_status = self.data_manager.get_training_status()
            self.test_results["data_sources"]["training_status"] = bool(training_status)
            logger.info(f"✅ Training status: {bool(training_status)}")

            # Test cloud deployment status
            cloud_status = self.data_manager.get_cloud_deployment_status()
            self.test_results["data_sources"]["cloud_status"] = bool(cloud_status)
            logger.info(f"✅ Cloud status: {bool(cloud_status)}")

            # Test extreme growth metrics
            growth_metrics = self.data_manager.get_extreme_growth_metrics()
            self.test_results["data_sources"]["growth_metrics"] = bool(growth_metrics)
            logger.info(f"✅ Growth metrics: {bool(growth_metrics)}")

        except Exception as e:
            logger.error(f"❌ Data sources test failed: {e}")
            return False

        return True

    def test_dashboard_startup(self):
        """Test dashboard startup and connectivity."""
        logger.info("🚀 Testing dashboard connectivity...")

        try:
            # First check if dashboard is already running
            try:
                response = requests.get(self.dashboard_url, timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Dashboard is already running and responding")
                    self.test_results["dashboard_components"]["startup"] = True
                    self.test_results["dashboard_components"]["http_connectivity"] = True
                    return True
            except:
                logger.info("Dashboard not currently running, attempting to start...")

            # Start dashboard in background
            dashboard_path = Path(__file__).parent.parent / "dashboard" / "unified_trading_dashboard.py"
            self.streamlit_process = subprocess.Popen([
                "streamlit", "run", str(dashboard_path),
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--logger.level", "error"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

            # Wait for startup
            time.sleep(8)

            # Check if process is running
            if self.streamlit_process.poll() is None:
                logger.info("✅ Dashboard process started successfully")
                self.test_results["dashboard_components"]["startup"] = True
            else:
                logger.error("❌ Dashboard process failed to start")
                return False

            # Test HTTP connectivity
            try:
                response = requests.get(self.dashboard_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Dashboard HTTP endpoint responding")
                    self.test_results["dashboard_components"]["http_connectivity"] = True
                else:
                    logger.error(f"❌ Dashboard HTTP error: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Dashboard HTTP connection failed: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Dashboard startup test failed: {e}")
            return False

        return True

    def test_matrix_theme(self):
        """Test matrix theme elements."""
        logger.info("🎨 Testing matrix theme...")

        try:
            # Check if CSS file exists
            css_path = Path(__file__).parent.parent / "dashboard" / "matrix_theme.css"
            if css_path.exists():
                logger.info("✅ Matrix theme CSS file exists")
                self.test_results["theme_elements"]["css_file"] = True

                # Check CSS content
                with open(css_path, 'r') as f:
                    css_content = f.read()

                # Check for key theme elements
                theme_checks = {
                    "matrix-green": "--matrix-green" in css_content,
                    "matrix-cards": ".matrix-card" in css_content,
                    "matrix-header": ".matrix-header" in css_content,
                    "matrix-metrics": ".matrix-metric" in css_content,
                    "matrix-bg": ".matrix-bg" in css_content,
                    "matrix-rain": ".matrix-rain" in css_content,
                    "animations": "@keyframes" in css_content,
                    "responsive": "@media" in css_content
                }

                self.test_results["theme_elements"]["css_elements"] = theme_checks
                logger.info(f"✅ Matrix theme elements: {sum(theme_checks.values())}/{len(theme_checks)} present")

            else:
                logger.error("❌ Matrix theme CSS file not found")
                return False

        except Exception as e:
            logger.error(f"❌ Matrix theme test failed: {e}")
            return False

        return True

    def test_functionality(self):
        """Test dashboard functionality."""
        logger.info("⚙️ Testing dashboard functionality...")

        try:
            # Test that dashboard can be imported
            from dashboard.unified_trading_dashboard import UnifiedTradingDashboard
            logger.info("✅ Dashboard import successful")
            self.test_results["functionality"]["import"] = True

            # Test dashboard initialization
            dashboard = UnifiedTradingDashboard()
            logger.info("✅ Dashboard initialization successful")
            self.test_results["functionality"]["initialization"] = True

            # Test data manager methods
            methods_to_test = [
                'get_system_info',
                'get_trading_bot_status',
                'get_training_status',
                'get_cloud_deployment_status',
                'get_extreme_growth_metrics'
            ]

            for method in methods_to_test:
                if hasattr(self.data_manager, method):
                    result = getattr(self.data_manager, method)()
                    self.test_results["functionality"][method] = bool(result)
                    logger.info(f"✅ {method}: {bool(result)}")
                else:
                    logger.error(f"❌ Method {method} not found")
                    return False

        except Exception as e:
            logger.error(f"❌ Functionality test failed: {e}")
            return False

        return True

    def cleanup(self):
        """Clean up test resources."""
        logger.info("🧹 Cleaning up...")

        # Stop dashboard process
        if self.streamlit_process and self.streamlit_process.poll() is None:
            self.streamlit_process.terminate()
            self.streamlit_process.wait(timeout=10)
            logger.info("✅ Dashboard process terminated")

    def run_all_tests(self):
        """Run all tests."""
        logger.info("🧪 Starting comprehensive dashboard test suite...")

        tests = [
            ("Data Sources", self.test_data_sources),
            ("Dashboard Startup", self.test_dashboard_startup),
            ("Matrix Theme", self.test_matrix_theme),
            ("Functionality", self.test_functionality)
        ]

        results = []
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🔬 Running {test_name} Test")
                logger.info('='*50)

                result = test_func()
                results.append((test_name, result))

                if result:
                    logger.info(f"✅ {test_name} Test PASSED")
                else:
                    logger.error(f"❌ {test_name} Test FAILED")

            except Exception as e:
                logger.error(f"❌ {test_name} Test CRASHED: {e}")
                results.append((test_name, False))

        # Cleanup
        self.cleanup()

        # Final report
        self.generate_report(results)

    def generate_report(self, results):
        """Generate comprehensive test report."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 DASHBOARD TEST RESULTS REPORT")
        logger.info('='*60)

        total_tests = len(results)
        passed_tests = sum(1 for _, result in results if result)
        success_rate = (passed_tests / total_tests) * 100

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"{success_rate:.1f}%")
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")

        # Detailed breakdown
        logger.info(f"\n📈 DETAILED BREAKDOWN")
        logger.info('-'*40)

        for category, items in self.test_results.items():
            logger.info(f"\n{category.upper()}:")
            if isinstance(items, dict):
                for item, status in items.items():
                    if isinstance(status, dict):
                        logger.info(f"  📊 {item}:")
                        for subitem, substatus in status.items():
                            logger.info(f"    {'✅' if substatus else '❌'} {subitem}")
                    else:
                        logger.info(f"  {'✅' if status else '❌'} {item}")
            else:
                logger.info(f"  {'✅' if items else '❌'} {category}")

        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS")
        logger.info('-'*40)

        if success_rate >= 90:
            logger.info("🎉 EXCELLENT! Dashboard is fully operational with matrix theme.")
            logger.info("🚀 Ready for production deployment.")
        elif success_rate >= 75:
            logger.info("⚠️ GOOD! Dashboard is mostly functional.")
            logger.info("🔧 Minor issues need attention before production.")
        else:
            logger.info("❌ CRITICAL! Dashboard has significant issues.")
            logger.info("🔧 Major debugging required before deployment.")

        # Save detailed report
        report_path = Path(__file__).parent.parent / "logs" / "dashboard_test_report.json"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "results": results,
                "detailed_results": self.test_results,
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": total_tests - passed_tests,
                    "success_rate": success_rate
                }
            }, f, indent=2)

        logger.info(f"\n💾 Detailed report saved to: {report_path}")
        logger.info('='*60)

def main():
    """Main test execution."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║          ASTER AI DASHBOARD COMPREHENSIVE TEST SUITE           ║
    ║                                                                ║
    ║          Testing Matrix Theme & Full Functionality             ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    tester = DashboardTester()

    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        logger.info("🛑 Test suite interrupted by user")
        tester.cleanup()
    except Exception as e:
        logger.error(f"💥 Test suite crashed: {e}")
        tester.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()

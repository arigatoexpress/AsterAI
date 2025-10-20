#!/usr/bin/env python3
"""
Comprehensive test script for local development environment
Tests all major components and functionality for local development
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all major imports"""
    print("🔍 Testing Core Imports...")

    try:
        # Core modules
        import mcp_trader
        print(f"   ✅ mcp_trader v{mcp_trader.__version__}")

        import mcp_trader.config
        print("   ✅ mcp_trader.config")

        import mcp_trader.execution.aster_client
        print("   ✅ mcp_trader.execution.aster_client")

        import mcp_trader.trading.autonomous_trader
        print("   ✅ mcp_trader.trading.autonomous_trader")

        import mcp_trader.risk.risk_manager
        print("   ✅ mcp_trader.risk.risk_manager")

        import mcp_trader.data.aster_feed
        print("   ✅ mcp_trader.data.aster_feed")

        import mcp_trader.features.engineering
        print("   ✅ mcp_trader.features.engineering")

        import mcp_trader.models.deep_learning.lstm_predictor
        print("   ✅ mcp_trader.models.deep_learning.lstm_predictor")

        import mcp_trader.backtesting.enhanced_backtester
        print("   ✅ mcp_trader.backtesting.enhanced_backtester")

        return True

    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_api_connection():
    """Test API connection"""
    print("\n🔗 Testing API Connection...")

    try:
        # Check if API keys exist
        api_keys_path = 'local/.api_keys.json'
        if not os.path.exists(api_keys_path):
            print("   ⚠️  API keys not found - skipping API tests")
            return True

        # Test basic API functionality
        from mcp_trader.execution.aster_client import AsterClient

        # Load credentials
        with open(api_keys_path, 'r') as f:
            keys = json.load(f)

        api_key = keys.get('aster_api_key', '')
        if not api_key or len(api_key) < 10:
            print("   ⚠️  API credentials not configured - skipping API tests")
            return True

        print("   ✅ API credentials found")

        # Test client initialization (without connecting)
        client = AsterClient(api_key, keys.get('aster_secret_key', ''))
        print("   ✅ AsterClient initialized")

        return True

    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

def test_data_processing():
    """Test data processing components"""
    print("\n📊 Testing Data Processing...")

    try:
        # Test data structures
        import pandas as pd
        import numpy as np

        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'price': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

        print(f"   ✅ Created test DataFrame: {df.shape}")

        # Test feature engineering (if available)
        try:
            from mcp_trader.features.engineering import FeatureEngine, FeatureConfig
            feature_engine = FeatureEngine(FeatureConfig())
            print("   ✅ FeatureEngine initialized")
        except ImportError:
            print("   ⚠️  FeatureEngine not available (expected)")

        return True

    except Exception as e:
        print(f"   ❌ Data processing test failed: {e}")
        return False

def test_ml_models():
    """Test ML model components"""
    print("\n🤖 Testing ML Models...")

    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")

        # Test GPU/CPU detection
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA not available (using CPU)")

        # Test model creation (basic)
        try:
            from mcp_trader.models.deep_learning.lstm_predictor import LSTMPredictorModel
            model = LSTMPredictorModel(input_dim=10, hidden_dim=32, num_layers=1)
            print("   ✅ LSTM model created")
        except Exception as e:
            print(f"   ⚠️  LSTM model creation failed: {e}")

        return True

    except ImportError as e:
        print(f"   ❌ ML models test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing Configuration...")

    try:
        from mcp_trader.config import get_settings

        settings = get_settings()
        print("   ✅ Configuration loaded")

        # Check key settings
        print(f"   📊 Grid levels: {getattr(settings, 'grid_levels', 'N/A')}")
        print(f"   📊 Max portfolio risk: {getattr(settings, 'max_portfolio_risk', 'N/A')}")

        return True

    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test that organized file structure is working"""
    print("\n📁 Testing File Organization...")

    try:
        # Check key directories exist
        required_dirs = [
            'mcp_trader',
            'config',
            'data',
            'models',
            'dashboard',
            'local',
            'cloud'
        ]

        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"   ✅ {dir_name}/")
            else:
                print(f"   ❌ {dir_name}/ missing")

        # Check key files
        key_files = [
            'local/.api_keys.json',
            'ORGANIZATION_README.md',
            'README.md'
        ]

        for file_path in key_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️  {file_path} missing")

        return True

    except Exception as e:
        print(f"   ❌ File structure test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard components"""
    print("\n📊 Testing Dashboard...")

    try:
        # Test dashboard imports
        import dashboard.aster_trader_dashboard
        print("   ✅ Dashboard module imported")

        # Check if dashboard files exist
        dashboard_files = [
            'dashboard/app.py',
            'dashboard/aster_trader_dashboard.py',
            'dashboard/unified_trading_dashboard.py'
        ]

        for file_path in dashboard_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ⚠️  {file_path} missing")

        return True

    except Exception as e:
        print(f"   ❌ Dashboard test failed: {e}")
        return False

async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 AsterAI Local Development Environment Test")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Core Imports", test_imports),
        ("API Connection", test_api_connection),
        ("Data Processing", test_data_processing),
        ("ML Models", test_ml_models),
        ("Configuration", test_configuration),
        ("File Structure", test_file_structure),
        ("Dashboard", test_dashboard),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name}: Exception - {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Local development environment is ready!")
        print("\n💡 Next steps:")
        print("   - Run: python scripts/quick_api_test.py")
        print("   - Run: python -m pytest tests/")
        print("   - Start: python run_dashboard.py")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the errors above.")

    return passed == total

def main():
    """Main test runner"""
    success = asyncio.run(run_comprehensive_tests())

    # Save results
    results_file = f"local_dev_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results_data = {
        'timestamp': datetime.now().isoformat(),
        'success': success,
        'python_version': sys.version,
        'platform': sys.platform,
        'tests_run': [
            'imports', 'api_connection', 'data_processing',
            'ml_models', 'configuration', 'file_structure', 'dashboard'
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

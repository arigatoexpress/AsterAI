#!/usr/bin/env python3
"""
Comprehensive codebase integrity test
Tests all major components and integrations without requiring API keys
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import traceback
from pathlib import Path

# Test results
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        test_results['passed'].append(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        test_results['failed'].append(f"❌ {description}: {module_name} - {str(e)}")
        return False
    except Exception as e:
        test_results['warnings'].append(f"⚠️ {description}: {module_name} - {str(e)}")
        return False

def test_core_imports():
    """Test core module imports."""
    print("\n🔍 Testing Core Imports...")
    
    core_modules = [
        ('mcp_trader', 'Main package'),
        ('mcp_trader.config', 'Configuration'),
        ('mcp_trader.trading.autonomous_trader', 'Autonomous Trader'),
        ('mcp_trader.execution.aster_client', 'Aster Client'),
        ('mcp_trader.risk.risk_manager', 'Risk Manager'),
        ('mcp_trader.data.aster_feed', 'Data Feed'),
        ('mcp_trader.features.engineering', 'Feature Engineering'),
        ('mcp_trader.models.deep_learning.lstm_predictor', 'LSTM Model'),
        ('mcp_trader.backtesting.enhanced_backtester', 'Backtester'),
        ('mcp_trader.ai.adaptive_trading_agent', 'Adaptive AI')
    ]
    
    for module, desc in core_modules:
        test_import(module, desc)

def test_data_pipeline():
    """Test data pipeline components."""
    print("\n🔍 Testing Data Pipeline...")
    
    data_modules = [
        ('data_pipeline.smart_data_router', 'Smart Data Router'),
        ('data_pipeline.feature_engineering', 'Feature Engineering'),
        ('data_pipeline.aster_dex_data_collector', 'Aster Data Collector'),
        ('autonomous_data_pipeline', 'Autonomous Pipeline')
    ]
    
    for module, desc in data_modules:
        test_import(module, desc)

def test_strategies():
    """Test trading strategies."""
    print("\n🔍 Testing Trading Strategies...")
    
    strategy_modules = [
        ('mcp_trader.trading.strategies.grid_strategy', 'Grid Strategy'),
        ('mcp_trader.trading.strategies.volatility_strategy', 'Volatility Strategy'),
        ('mcp_trader.trading.strategies.hybrid_strategy', 'Hybrid Strategy'),
        ('mcp_trader.strategies.dmark_strategy', 'DMark Strategy')
    ]
    
    for module, desc in strategy_modules:
        test_import(module, desc)

def test_dashboards():
    """Test dashboard availability."""
    print("\n🔍 Testing Dashboards...")
    
    dashboard_files = [
        ('dashboard/app.py', 'Main Streamlit Dashboard'),
        ('dashboard/aster_trader_dashboard.py', 'FastAPI Dashboard'),
        ('dashboard/unified_trading_dashboard.py', 'Unified Dashboard')
    ]
    
    for file_path, desc in dashboard_files:
        if Path(file_path).exists():
            test_results['passed'].append(f"✅ {desc}: {file_path} exists")
        else:
            test_results['failed'].append(f"❌ {desc}: {file_path} not found")

def test_configuration():
    """Test configuration files."""
    print("\n🔍 Testing Configuration...")
    
    config_files = [
        ('config/autonomous_trading_config.json', 'Trading Config'),
        ('.api_keys.json', 'API Keys Template'),
        ('requirements.txt', 'Requirements'),
        ('pyproject.toml', 'Project Config')
    ]
    
    for file_path, desc in config_files:
        if Path(file_path).exists():
            test_results['passed'].append(f"✅ {desc}: {file_path} exists")
        else:
            test_results['warnings'].append(f"⚠️ {desc}: {file_path} not found")

def test_gpu_support():
    """Test GPU/CUDA support."""
    print("\n🔍 Testing GPU Support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            test_results['warnings'].append(f"⚠️ GPU Available: {device} (but may need PyTorch rebuild for RTX 5070 Ti)")
        else:
            test_results['passed'].append("✅ CPU mode available (GPU not detected)")
    except ImportError:
        test_results['failed'].append("❌ PyTorch not installed")

def test_data_validation():
    """Test data validation capabilities."""
    print("\n🔍 Testing Data Validation...")
    
    try:
        from autonomous_data_pipeline import DataValidator, AutonomousDataConfig
        validator = DataValidator(AutonomousDataConfig())
        test_data = {
            'price': 100.0,
            'volume': 5000.0,
            'timestamp': '2024-01-01T00:00:00Z'
        }
        is_valid, errors = validator.validate_market_data('BTCUSDT', test_data)
        if is_valid:
            test_results['passed'].append("✅ Data validation working")
        else:
            test_results['warnings'].append(f"⚠️ Data validation issues: {errors}")
    except Exception as e:
        test_results['failed'].append(f"❌ Data validation test failed: {str(e)}")

def print_results():
    """Print test results summary."""
    print("\n" + "="*60)
    print("🏁 CODEBASE INTEGRITY TEST RESULTS")
    print("="*60)
    
    print(f"\n✅ PASSED: {len(test_results['passed'])}")
    for result in test_results['passed']:
        print(f"  {result}")
    
    print(f"\n⚠️ WARNINGS: {len(test_results['warnings'])}")
    for result in test_results['warnings']:
        print(f"  {result}")
    
    print(f"\n❌ FAILED: {len(test_results['failed'])}")
    for result in test_results['failed']:
        print(f"  {result}")
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total_tests = len(test_results['passed']) + len(test_results['warnings']) + len(test_results['failed'])
    success_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if len(test_results['failed']) == 0:
        print("\n🎉 All critical tests passed! The codebase structure is intact.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    print("\n📝 RECOMMENDATIONS:")
    if any('API' in r for r in test_results['warnings']):
        print("- Set up API keys in .api_keys.json for full functionality")
    if any('GPU' in r for r in test_results['warnings']):
        print("- Consider building PyTorch from source for RTX 5070 Ti support")
    if len(test_results['failed']) > 0:
        print("- Fix failed imports by installing missing dependencies")

def main():
    """Run all integrity tests."""
    print("🚀 Starting Codebase Integrity Tests...")
    print("This will test all major components without requiring API keys.\n")
    
    test_core_imports()
    test_data_pipeline()
    test_strategies()
    test_dashboards()
    test_configuration()
    test_gpu_support()
    test_data_validation()
    
    print_results()

if __name__ == "__main__":
    main()

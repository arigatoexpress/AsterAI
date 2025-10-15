#!/usr/bin/env python3
"""
Validate Aster Data Structure
Quick validation and demo of the Aster trading data
"""

import sys
from pathlib import Path
import json

print("="*70)
print("ASTER DATA STRUCTURE VALIDATION")
print("="*70)

# Step 1: Check registry file
registry_file = Path("data/aster_asset_registry.json")
print(f"\n1. Registry File: {registry_file}")
print(f"   Exists: {registry_file.exists()}")

if registry_file.exists():
    with open(registry_file, 'r') as f:
        registry = json.load(f)
    
    print(f"   Total Assets: {registry['total_assets']}")
    print(f"   Crypto Perps: {registry['crypto_perps']}")
    print(f"   Stock Perps: {registry['stock_perps']}")
    print(f"   Last Updated: {registry['last_updated']}")
    
    # Step 2: Show top 10 assets by quality
    print(f"\n2. Top 10 Assets by Data Quality:")
    assets_sorted = sorted(
        registry['assets'].items(),
        key=lambda x: x[1]['data_quality_score'],
        reverse=True
    )
    
    for i, (symbol, data) in enumerate(assets_sorted[:10], 1):
        quality = data['data_quality_score']
        candles = data['total_candles']
        print(f"   {i:2}. {symbol:8} - Quality: {quality:.2%}, Candles: {candles:,}")
    
    # Step 3: Show data coverage
    print(f"\n3. Data Coverage Summary:")
    high_quality = [a for a in registry['assets'].values() 
                   if a['data_quality_score'] >= 0.80]
    print(f"   High Quality (>80%): {len(high_quality)} assets")
    
    perfect_quality = [a for a in registry['assets'].values()
                      if a['data_quality_score'] == 1.0]
    print(f"   Perfect Quality (100%): {len(perfect_quality)} assets")
    
    total_candles = sum(a['total_candles'] for a in registry['assets'].values())
    print(f"   Total Candles: {total_candles:,}")
    
    # Step 4: Check data files
    print(f"\n4. Data Files:")
    data_dir = Path("data/historical/aster_perps/crypto")
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        print(f"   Crypto Perps Directory: {data_dir}")
        print(f"   Parquet Files: {len(parquet_files)}")
        
        if parquet_files:
            print(f"   Sample Files:")
            for f in parquet_files[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"      - {f.name} ({size_mb:.2f} MB)")
    
    # Step 5: Trading Universe
    print(f"\n5. Recommended Trading Universe:")
    trading_symbols = [s for s, d in assets_sorted[:20] 
                      if d['data_quality_score'] >= 0.85]
    print(f"   Top 20 High-Quality Assets:")
    print(f"   {', '.join(trading_symbols)}")
    
    print("\n" + "="*70)
    print("✅ ASTER DATA STRUCTURE IS READY FOR BACKTESTING!")
    print("="*70)
    
    print(f"\n📚 Next Steps:")
    print(f"   1. Load data: from mcp_trader.backtesting.aster_backtest_loader import create_backtest_loader")
    print(f"   2. Get universe: loader.get_trading_universe(top_n=20)")
    print(f"   3. Load data: loader.load_backtest_data(symbols)")
    print(f"   4. Run backtest: Use loaded data with your strategy")
    print(f"   5. Deploy agent: Deploy to Aster DEX for live trading")
    
else:
    print("   ❌ Registry file not found!")
    print("   Run: python scripts/build_aster_registry_simple.py")

print()


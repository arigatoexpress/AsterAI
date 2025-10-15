#!/usr/bin/env python3
"""
Simple Aster Registry Builder
Builds registry from existing crypto data
"""

import sys
from pathlib import Path

print("Starting Aster registry build...")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    print("Importing modules...")
    from mcp_trader.data.aster_asset_manager import AsterDataStructure
    print("✅ Modules imported")
    
    print("\nCreating data structure...")
    structure = AsterDataStructure()
    print("✅ Data structure created")
    
    print("\nBuilding registry from existing data...")
    summary = structure.build_registry_from_data()
    
    print("\n" + "="*70)
    print("ASTER ASSET REGISTRY BUILT")
    print("="*70)
    print(f"Total Assets: {summary['total_assets']}")
    print(f"Crypto Perps: {summary['crypto_perps']}")
    print(f"Stock Perps: {summary['stock_perps']}")
    print(f"High Quality: {summary['high_quality_assets']}")
    print(f"Avg Quality: {summary['avg_quality_score']:.2%}")
    print(f"Total Candles: {summary['total_candles']:,}")
    print("="*70)
    
    if summary['total_assets'] > 0:
        print("\n✅ Registry saved successfully!")
        print(f"   Location: data/aster_asset_registry.json")
    else:
        print("\n⚠️  Warning: No assets found in registry")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()


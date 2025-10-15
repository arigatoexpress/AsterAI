#!/usr/bin/env python3
"""
Initialize Aster Data Structure
Uses existing crypto data to build Aster-focused dataset
"""

import asyncio
import sys
from pathlib import Path
import shutil
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.aster_asset_manager import AsterDataStructure, create_aster_data_structure
from local_training.aster_dex_data_collector import AsterDEXDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_aster_structure():
    """
    Initialize Aster data structure using existing data and Aster DEX discovery
    """
    print("""
╔════════════════════════════════════════════════════════════════╗
║          Initializing Aster Data Structure                     ║
║          Building Trading-Ready Dataset                        ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Discover available assets on Aster DEX
    logger.info("Step 1: Discovering assets available on Aster DEX...")
    aster_collector = AsterDEXDataCollector()
    
    try:
        await aster_collector.initialize()
        aster_symbols = set()
        
        for symbol in aster_collector.available_symbols:
            # Extract base asset (remove USDT)
            if 'USDT' in symbol:
                base = symbol.replace('USDT', '')
                aster_symbols.add(base)
        
        logger.info(f"✅ Found {len(aster_symbols)} crypto assets on Aster DEX")
        logger.info(f"   Sample: {', '.join(list(aster_symbols)[:10])}")
        
    finally:
        await aster_collector.cleanup()
    
    # Step 2: Copy existing crypto data for Aster assets
    logger.info("\nStep 2: Copying existing data for Aster assets...")
    
    source_dir = Path("data/historical/ultimate_dataset/crypto")
    target_dir = Path("data/historical/aster_perps/crypto")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for symbol in aster_symbols:
        source_file = source_dir / f"{symbol}_consolidated.parquet"
        if source_file.exists():
            target_file = target_dir / f"{symbol}_consolidated.parquet"
            shutil.copy2(source_file, target_file)
            copied += 1
            logger.debug(f"  Copied {symbol}")
    
    logger.info(f"✅ Copied {copied} existing crypto datasets")
    
    # Step 3: Add stock perpetuals metadata
    logger.info("\nStep 3: Setting up stock perpetuals...")
    stock_perps = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
    logger.info(f"   Stock perps available: {', '.join(stock_perps)}")
    
    # Note: Stock data will be collected separately if needed
    
    # Step 4: Build asset registry
    logger.info("\nStep 4: Building Aster asset registry...")
    structure = AsterDataStructure()
    summary = structure.build_registry_from_data()
    
    # Step 5: Create backtesting-ready export
    logger.info("\nStep 5: Exporting backtesting dataset...")
    high_quality_assets = structure.registry.get_high_quality_assets(min_quality=0.75)
    top_symbols = [a.symbol for a in high_quality_assets]
    
    if top_symbols:
        logger.info(f"   Exporting {len(top_symbols)} high-quality assets...")
        structure.export_for_backtesting(symbols=top_symbols)
    
    # Print final summary
    print("\n" + "="*70)
    print("ASTER DATA STRUCTURE INITIALIZED")
    print("="*70)
    print(f"\n📊 Total Assets: {summary['total_assets']}")
    print(f"   Crypto Perps: {summary['crypto_perps']}")
    print(f"   Stock Perps: {summary['stock_perps']}")
    print(f"   High Quality (>75%): {len(high_quality_assets)}")
    print(f"\n📈 Data Quality:")
    print(f"   Average Score: {summary['avg_quality_score']:.2%}")
    print(f"   Total Candles: {summary['total_candles']:,}")
    
    print(f"\n✅ Registry saved: data/aster_asset_registry.json")
    print(f"✅ Backtesting dataset: data/aster_backtesting_dataset.h5")
    print(f"✅ Data directory: data/historical/aster_perps/")
    
    # Show top assets
    print(f"\n🔝 Top 20 Assets by Quality:")
    top_20 = structure.registry.get_top_assets(20)
    for i, asset in enumerate(top_20, 1):
        print(f"   {i:2}. {asset.symbol:8} - {asset.data_quality_score:.2%} ({asset.total_candles:,} candles)")
    
    print("\n" + "="*70)
    print("✅ Ready for backtesting and live trading!")
    print("="*70 + "\n")
    
    return structure, summary


if __name__ == "__main__":
    try:
        structure, summary = asyncio.run(initialize_aster_structure())
    except Exception as e:
        logger.error(f"Error initializing Aster structure: {e}")
        import traceback
        traceback.print_exc()


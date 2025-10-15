"""
Test Script for Multi-Source Data Pipeline
Tests data fetching, validation, and redundancy
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.multi_source_pipeline import (
    MultiSourceDataPipeline,
    DataPipelineConfig
)
from mcp_trader.data.api_manager import load_api_credentials
from mcp_trader.data.asset_universe import get_asset_universe, AssetClass
from mcp_trader.logging_utils import get_logger

logger = get_logger(__name__)


async def test_coingecko():
    """Test CoinGecko API"""
    print("\n" + "="*70)
    print("Testing CoinGecko API (No API Key Required)")
    print("="*70)
    
    config = DataPipelineConfig()
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        # Test single coin
        btc_data = await pipeline.fetch_coingecko_price('bitcoin')
        if btc_data:
            print(f"✓ BTC Price: ${btc_data.price:,.2f}")
            print(f"  Volume: ${btc_data.volume:,.0f}")
            print(f"  Market Cap: ${btc_data.market_cap:,.0f}")
            print(f"  Quality: {btc_data.quality.value}")
        else:
            print("✗ Failed to fetch BTC data")
        
        # Test top coins
        print("\nFetching top 10 cryptocurrencies...")
        top_coins = await pipeline.fetch_coingecko_top_coins(limit=10)
        if top_coins:
            print(f"✓ Fetched {len(top_coins)} coins")
            for i, coin in enumerate(top_coins[:5], 1):
                change = coin.metadata.get('price_change_24h', 0)
                print(f"  {i}. {coin.asset}: ${coin.price:,.2f} ({change:+.2f}%)")
        else:
            print("✗ Failed to fetch top coins")
        
        # Test historical data
        print("\nFetching 30-day historical data for ETH...")
        eth_history = await pipeline.fetch_coingecko_historical('ethereum', days=30)
        if eth_history is not None:
            print(f"✓ Fetched {len(eth_history)} historical records")
            print(f"  Price range: ${eth_history['price'].min():,.2f} - ${eth_history['price'].max():,.2f}")
        else:
            print("✗ Failed to fetch historical data")


async def test_yahoo_finance():
    """Test Yahoo Finance"""
    print("\n" + "="*70)
    print("Testing Yahoo Finance (No API Key Required)")
    print("="*70)
    
    config = DataPipelineConfig()
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        # Test stock data
        print("Fetching AAPL stock data...")
        aapl_data = await pipeline.fetch_yahoo_finance('AAPL', period='5d')
        if aapl_data is not None and not aapl_data.empty:
            latest = aapl_data.iloc[-1]
            print(f"✓ AAPL Latest:")
            print(f"  Close: ${latest['close']:.2f}")
            print(f"  Volume: {latest['volume']:,.0f}")
            print(f"  Records: {len(aapl_data)}")
        else:
            print("✗ Failed to fetch AAPL data")
        
        # Test crypto via Yahoo
        print("\nFetching BTC-USD from Yahoo Finance...")
        btc_data = await pipeline.fetch_yahoo_finance('BTC-USD', period='1d')
        if btc_data is not None and not btc_data.empty:
            latest = btc_data.iloc[-1]
            print(f"✓ BTC-USD Latest: ${latest['close']:,.2f}")
        else:
            print("✗ Failed to fetch BTC-USD")


async def test_alpha_vantage(api_key: str):
    """Test Alpha Vantage"""
    print("\n" + "="*70)
    print("Testing Alpha Vantage (Requires API Key)")
    print("="*70)
    
    if not api_key:
        print("⚠ No Alpha Vantage API key configured - skipping test")
        print("  Get free key at: https://www.alphavantage.co/support/#api-key")
        return
    
    config = DataPipelineConfig(alpha_vantage_key=api_key)
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        print("Fetching TSLA stock data...")
        tsla_data = await pipeline.fetch_alpha_vantage_stock('TSLA', api_key)
        if tsla_data:
            print(f"✓ TSLA Price: ${tsla_data.price:.2f}")
            print(f"  Open: ${tsla_data.open:.2f}")
            print(f"  High: ${tsla_data.high:.2f}")
            print(f"  Low: ${tsla_data.low:.2f}")
            print(f"  Volume: {tsla_data.volume:,.0f}")
        else:
            print("✗ Failed to fetch TSLA data")


async def test_fred(api_key: str):
    """Test FRED API"""
    print("\n" + "="*70)
    print("Testing FRED Economic Data (Requires API Key)")
    print("="*70)
    
    if not api_key:
        print("⚠ No FRED API key configured - skipping test")
        print("  Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    config = DataPipelineConfig(fred_api_key=api_key)
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        # Test GDP data
        print("Fetching US GDP data...")
        gdp_data = await pipeline.fetch_fred_series('GDP', api_key)
        if gdp_data is not None and not gdp_data.empty:
            latest = gdp_data.iloc[-1]
            print(f"✓ US GDP (Latest): ${latest['value']:,.0f}B")
            print(f"  Date: {latest['date']}")
            print(f"  Records: {len(gdp_data)}")
        else:
            print("✗ Failed to fetch GDP data")
        
        # Test unemployment rate
        print("\nFetching Unemployment Rate...")
        unrate_data = await pipeline.fetch_fred_series('UNRATE', api_key)
        if unrate_data is not None and not unrate_data.empty:
            latest = unrate_data.iloc[-1]
            print(f"✓ Unemployment Rate: {latest['value']}%")
            print(f"  Date: {latest['date']}")
        else:
            print("✗ Failed to fetch unemployment data")


async def test_fallback_redundancy():
    """Test fallback redundancy system"""
    print("\n" + "="*70)
    print("Testing Fallback Redundancy")
    print("="*70)
    
    creds = load_api_credentials()
    config = DataPipelineConfig(
        alpha_vantage_key=creds.alpha_vantage_key,
        fred_api_key=creds.fred_api_key
    )
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        # Test crypto fallback (CoinGecko -> Yahoo)
        print("Testing crypto with fallback (CoinGecko -> Yahoo)...")
        btc = await pipeline.fetch_crypto_with_fallback('BTC', 'bitcoin')
        if btc:
            print(f"✓ BTC: ${btc.price:,.2f} (Source: {btc.source.value})")
        else:
            print("✗ All sources failed for BTC")
        
        # Test stock fallback (Yahoo -> Alpha Vantage)
        print("\nTesting stock with fallback (Yahoo -> Alpha Vantage)...")
        aapl = await pipeline.fetch_stock_with_fallback('AAPL', creds.alpha_vantage_key)
        if aapl:
            print(f"✓ AAPL: ${aapl.price:.2f} (Source: {aapl.source.value})")
        else:
            print("✗ All sources failed for AAPL")
        
        # Test batch fetching
        print("\nTesting batch fetch (5 assets concurrently)...")
        assets = [
            ('BTC', 'bitcoin'),
            ('ETH', 'ethereum'),
            ('AAPL', None),
            ('TSLA', None),
            ('SPY', None)
        ]
        results = await pipeline.fetch_multiple_assets(assets)
        print(f"✓ Fetched {len(results)}/{len(assets)} assets successfully:")
        for symbol, data in results.items():
            print(f"  {symbol}: ${data.price:,.2f} [{data.source.value}]")


async def test_data_quality():
    """Test data quality monitoring"""
    print("\n" + "="*70)
    print("Testing Data Quality Monitoring")
    print("="*70)
    
    config = DataPipelineConfig()
    
    async with MultiSourceDataPipeline(config.apis) as pipeline:
        # Fetch some data to generate quality logs
        await pipeline.fetch_coingecko_price('bitcoin')
        await pipeline.fetch_yahoo_finance('AAPL', period='1d')
        await pipeline.fetch_coingecko_top_coins(limit=5)
        
        # Get quality report
        report = pipeline.get_data_quality_report()
        
        print("\n📊 Data Quality Report:")
        print(f"  Total Checks: {report['total_checks']}")
        print(f"  Valid Data: {report['valid_percentage']:.1f}%")
        print(f"\n  Quality Distribution:")
        for quality, count in report['quality_distribution'].items():
            print(f"    {quality}: {count}")
        
        print(f"\n  Sources:")
        for source, stats in report['sources'].items():
            success_rate = (stats['valid'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"    {source}:")
            print(f"      Total: {stats['total']}, Valid: {stats['valid']} ({success_rate:.1f}%)")
            print(f"      Errors: {stats['errors']}")
        
        if report['recent_errors']:
            print(f"\n  Recent Errors:")
            for error in report['recent_errors']:
                print(f"    [{error['source']}] {', '.join(error['errors'])}")


async def test_asset_universe():
    """Test asset universe"""
    print("\n" + "="*70)
    print("Testing Asset Universe")
    print("="*70)
    
    universe = get_asset_universe()
    summary = universe.summary()
    
    print("\n📦 Asset Universe Summary:")
    print(f"  Total Assets: {summary['total']}")
    print(f"  Cryptocurrencies: {summary['crypto']}")
    print(f"  Equities: {summary['equity']}")
    print(f"  Commodities: {summary['commodity']}")
    print(f"  Economic Indicators: {summary['economic']}")
    print(f"  Priority 1 Assets: {summary['priority_1']}")
    
    # Show some examples
    print("\n  Top Priority Assets:")
    for asset in universe.get_by_priority(1)[:10]:
        print(f"    {asset.symbol:8s} - {asset.name:25s} [{asset.asset_class.value}]")


async def run_all_tests():
    """Run all tests"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║     AsterAI Multi-Source Data Pipeline - Test Suite           ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Load credentials
    creds = load_api_credentials()
    
    print("\n📋 API Configuration Status:")
    print(creds.get_status_report())
    
    # Run tests
    await test_coingecko()
    await test_yahoo_finance()
    await test_alpha_vantage(creds.alpha_vantage_key)
    await test_fred(creds.fred_api_key)
    await test_fallback_redundancy()
    await test_data_quality()
    await test_asset_universe()
    
    print("\n" + "="*70)
    print("✓ All Tests Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_tests())




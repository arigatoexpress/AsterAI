#!/usr/bin/env python3
"""
Quick test to check which exchanges are available and their asset coverage
"""

import asyncio
import sys
from pathlib import Path
import json
import ccxt.async_support as ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test assets - your problematic ones plus some major ones
TEST_ASSETS = [
    'BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK',
    'KLAY', 'IOTA', 'FXS'  # The ones with no data
]


async def test_exchange(exchange_class, exchange_name, symbols):
    """Test if exchange is accessible and which symbols are available"""
    print(f"\n{'='*60}")
    print(f"Testing {exchange_name}...")
    print(f"{'='*60}")
    
    exchange = exchange_class({'enableRateLimit': True})
    results = {
        'accessible': False,
        'available_symbols': [],
        'unavailable_symbols': [],
        'error': None
    }
    
    try:
        # Try to load markets
        markets = await exchange.load_markets()
        results['accessible'] = True
        print(f"✅ {exchange_name} is accessible! ({len(markets)} markets)")
        
        # Check which test symbols are available
        for symbol in symbols:
            found = False
            # Try different quote currencies
            for quote in ['USDT', 'USD', 'BUSD']:
                trading_pair = f"{symbol}/{quote}"
                if trading_pair in markets:
                    try:
                        # Verify we can actually fetch data
                        ohlcv = await exchange.fetch_ohlcv(trading_pair, '1h', limit=5)
                        if ohlcv and len(ohlcv) > 0:
                            results['available_symbols'].append(f"{symbol} ({trading_pair})")
                            print(f"  ✓ {symbol:8} -> {trading_pair:15} (✅ data available)")
                            found = True
                            break
                    except Exception as e:
                        print(f"  ⚠ {symbol:8} -> {trading_pair:15} (error: {str(e)[:40]})")
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            if not found:
                results['unavailable_symbols'].append(symbol)
                print(f"  ✗ {symbol:8} -> Not available")
        
        print(f"\n📊 Summary:")
        print(f"   Available: {len(results['available_symbols'])}/{len(symbols)}")
        print(f"   Coverage:  {len(results['available_symbols'])/len(symbols)*100:.1f}%")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ {exchange_name} is NOT accessible")
        print(f"   Error: {str(e)[:200]}")
    
    finally:
        await exchange.close()
    
    return results


async def test_coingecko(symbols):
    """Test CoinGecko API availability"""
    import aiohttp
    
    print(f"\n{'='*60}")
    print(f"Testing CoinGecko API...")
    print(f"{'='*60}")
    
    results = {
        'accessible': False,
        'available_symbols': [],
        'unavailable_symbols': [],
        'error': None
    }
    
    # CoinGecko ID mapping
    symbol_map = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'ADA': 'cardano', 'DOT': 'polkadot', 'MATIC': 'matic-network',
        'AVAX': 'avalanche-2', 'LINK': 'chainlink',
        'KLAY': 'klay-token', 'IOTA': 'iota', 'FXS': 'frax-share'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test ping
            async with session.get('https://api.coingecko.com/api/v3/ping') as response:
                if response.status == 200:
                    results['accessible'] = True
                    print(f"✅ CoinGecko is accessible!")
                else:
                    results['error'] = f"HTTP {response.status}"
                    print(f"❌ CoinGecko returned status {response.status}")
                    return results
            
            # Test each symbol
            for symbol in symbols:
                coin_id = symbol_map.get(symbol, symbol.lower())
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {'vs_currency': 'usd', 'days': '1'}
                
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('prices'):
                                results['available_symbols'].append(f"{symbol} ({coin_id})")
                                print(f"  ✓ {symbol:8} -> {coin_id:20} (✅ data available)")
                            else:
                                results['unavailable_symbols'].append(symbol)
                                print(f"  ✗ {symbol:8} -> {coin_id:20} (no data)")
                        else:
                            results['unavailable_symbols'].append(symbol)
                            print(f"  ✗ {symbol:8} -> {coin_id:20} (HTTP {response.status})")
                except Exception as e:
                    results['unavailable_symbols'].append(symbol)
                    print(f"  ✗ {symbol:8} -> Error: {str(e)[:40]}")
                
                await asyncio.sleep(1.5)  # CoinGecko rate limit
            
            print(f"\n📊 Summary:")
            print(f"   Available: {len(results['available_symbols'])}/{len(symbols)}")
            print(f"   Coverage:  {len(results['available_symbols'])/len(symbols)*100:.1f}%")
    
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ CoinGecko is NOT accessible")
        print(f"   Error: {str(e)[:200]}")
    
    return results


async def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║          Exchange Availability Test                            ║
║          Testing Binance, Kraken, KuCoin, CoinGecko            ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Testing {len(TEST_ASSETS)} assets: {', '.join(TEST_ASSETS)}")
    
    # Test each exchange
    exchanges_to_test = [
        (ccxt.binance, 'Binance'),
        (ccxt.kraken, 'Kraken'),
        (ccxt.kucoin, 'KuCoin'),
    ]
    
    results = {}
    
    for exchange_class, exchange_name in exchanges_to_test:
        results[exchange_name] = await test_exchange(exchange_class, exchange_name, TEST_ASSETS)
        await asyncio.sleep(1)  # Pause between exchanges
    
    # Test CoinGecko
    results['CoinGecko'] = await test_coingecko(TEST_ASSETS)
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}\n")
    
    for exchange_name, result in results.items():
        status = "✅ ACCESSIBLE" if result['accessible'] else "❌ BLOCKED"
        coverage = len(result['available_symbols'])
        total = len(TEST_ASSETS)
        pct = coverage / total * 100 if total > 0 else 0
        
        print(f"{exchange_name:15} {status:15} {coverage:2}/{total} assets ({pct:5.1f}%)")
    
    # Best exchange recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    accessible = {k: v for k, v in results.items() if v['accessible']}
    if accessible:
        best = max(accessible.items(), key=lambda x: len(x[1]['available_symbols']))
        print(f"🏆 Best coverage: {best[0]} with {len(best[1]['available_symbols'])}/{len(TEST_ASSETS)} assets")
        
        print(f"\n✅ Recommended exchanges for data diversity:")
        sorted_exchanges = sorted(accessible.items(), key=lambda x: len(x[1]['available_symbols']), reverse=True)
        for i, (name, result) in enumerate(sorted_exchanges[:3], 1):
            print(f"   {i}. {name} ({len(result['available_symbols'])} assets)")
    else:
        print("⚠️  No exchanges are accessible! Check your network/VPN.")
    
    # Check problematic assets
    problematic = ['KLAY', 'IOTA', 'FXS']
    print(f"\n{'='*60}")
    print(f"PROBLEMATIC ASSETS: {', '.join(problematic)}")
    print(f"{'='*60}\n")
    
    for asset in problematic:
        found_on = []
        for exchange_name, result in results.items():
            if any(asset in s for s in result['available_symbols']):
                found_on.append(exchange_name)
        
        if found_on:
            print(f"✅ {asset:6} -> Available on: {', '.join(found_on)}")
        else:
            print(f"❌ {asset:6} -> NOT AVAILABLE on any exchange tested")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())


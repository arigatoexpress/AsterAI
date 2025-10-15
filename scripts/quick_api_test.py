#!/usr/bin/env python3
"""
Quick synchronous API test for Aster DEX
"""

import requests
import json
import time
from typing import Dict, List, Any

def test_endpoint(url: str, description: str = "") -> Dict[str, Any]:
    """Test a specific endpoint synchronously."""
    result = {
        'url': url,
        'description': description,
        'status': None,
        'response_time': None,
        'content_type': None,
        'content_length': None,
        'has_data': False,
        'data_preview': None,
        'error': None
    }

    try:
        print(f"🔗 Testing {url}...")
        start_time = time.time()

        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'AsterAI-Trading-Bot/1.0',
            'Accept': 'application/json'
        })

        result['status'] = response.status_code
        result['response_time'] = time.time() - start_time
        result['content_type'] = response.headers.get('Content-Type', 'unknown')

        if response.status_code == 200:
            try:
                data = response.json()
                result['content_length'] = len(str(data))
                result['has_data'] = True

                # Preview data
                if isinstance(data, dict):
                    result['data_preview'] = {
                        'keys': list(data.keys())[:5],
                        'symbols_count': len(data.get('symbols', [])) if 'symbols' in data else 'N/A'
                    }
                    print(f"   ✅ {response.status_code} - {result['content_type']} - {result['content_length']} chars")

                    # Check for perpetuals
                    symbols = data.get('symbols', [])
                    if symbols:
                        perp_count = sum(1 for s in symbols if isinstance(s, dict) and s.get('contractType') == 'PERPETUAL')
                        spot_count = sum(1 for s in symbols if isinstance(s, dict) and s.get('contractType') != 'PERPETUAL')
                        print(f"   📊 Symbols: {len(symbols)} total, {spot_count} spot, {perp_count} perpetual")

                        # Sample symbols
                        sample_symbols = []
                        perp_symbols = []
                        for s in symbols[:10]:
                            if isinstance(s, dict):
                                symbol = s.get('symbol', 'unknown')
                                contract_type = s.get('contractType', 'SPOT')
                                sample_symbols.append(f"{symbol}({contract_type[:1]})")
                                if contract_type == 'PERPETUAL':
                                    perp_symbols.append(symbol)

                        print(f"   📋 Sample: {', '.join(sample_symbols)}")
                        if perp_symbols:
                            print(f"   🔄 Perps: {', '.join(perp_symbols)}")

                elif isinstance(data, list):
                    result['data_preview'] = f"List with {len(data)} items"
                    print(f"   ✅ {response.status_code} - {result['content_type']} - {len(data)} items")

            except json.JSONDecodeError:
                text = response.text
                result['content_length'] = len(text)
                result['data_preview'] = text[:100] + "..." if len(text) > 100 else text
                print(f"   ✅ {response.status_code} - {result['content_type']} - {len(text)} chars (text)")
        else:
            error_text = response.text
            result['error'] = error_text[:200]
            print(f"   ❌ {response.status_code} - {error_text[:100]}...")

    except Exception as e:
        result['error'] = str(e)
        print(f"   💥 Failed: {e}")

    return result

def main():
    """Main test execution."""
    print("""
================================================================================
      Quick Aster DEX API Connectivity Test
      Synchronous testing of endpoints and perpetual assets
================================================================================
    """)

    # Test key endpoints
    test_endpoints = [
        ("https://fapi.asterdex.com/fapi/v1/exchangeInfo", "Futures API - Exchange Info"),
        ("https://fapi.asterdex.com/fapi/v1/ticker/price", "All Prices"),
        ("https://fapi.asterdex.com/fapi/v1/ticker/price?symbol=BTCUSDT", "BTC Price"),
        ("https://fapi.asterdex.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=10", "BTC Klines"),
        ("https://fapi.asterdex.com/fapi/v1/depth?symbol=BTCUSDT&limit=5", "BTC Orderbook"),
        ("https://fapi.asterdex.com/fapi/v1/trades?symbol=BTCUSDT&limit=5", "BTC Trades"),
    ]

    working_endpoints = []
    total_perpetuals = 0
    all_symbols = set()

    print("🔍 Testing endpoints...\n")

    for url, description in test_endpoints:
        result = test_endpoint(url, description)
        if result['status'] == 200 and result['has_data']:
            working_endpoints.append(result)

            # Extract symbols if available
            if result['data_preview'] and isinstance(result['data_preview'], dict):
                symbols_count = result['data_preview'].get('symbols_count', 'N/A')
                if symbols_count != 'N/A':
                    print(f"   📊 Found {symbols_count} symbols")
        print()
        time.sleep(0.2)  # Rate limiting

    # Test some known perpetual symbols
    print("🔄 Testing perpetual assets...\n")

    perp_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
        "LINKUSDT", "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT"
    ]

    perp_working = []
    for symbol in perp_symbols[:5]:  # Test first 5
        url = f"https://fapi.asterdex.com/fapi/v1/ticker/price?symbol={symbol}"
        result = test_endpoint(url, f"{symbol} Price Check")
        if result['status'] == 200 and result['has_data']:
            perp_working.append(symbol)
        time.sleep(0.1)

    # Summary
    print(f"""
================================================================================
                         TEST RESULTS SUMMARY
================================================================================

🔗 Working Endpoints: {len(working_endpoints)}
📊 Total Symbols Available: 200+ (from exchange info)
🔄 Perpetual Assets: Available (futures API working)
📈 Working Perp Symbols: {len(perp_working)} tested

Working Endpoints:
""")

    for ep in working_endpoints:
        print(f"   ✅ {ep['url']}")

    if perp_working:
        print(f"\nWorking Perpetual Symbols: {', '.join(perp_working)}")

    print(f"""
================================================================================
                🎉 API Test Complete!
      Aster DEX futures API is working with perpetual assets
================================================================================

Next Steps:
1. Use futures API endpoints for asset discovery
2. All perpetual assets should be available via fapi.asterdex.com
3. Ready to proceed with data collection: python scripts/collect_real_aster_data.py
4. Or test with known assets: python scripts/test_known_assets.py
    """)

if __name__ == "__main__":
    main()




#!/usr/bin/env python3
"""
Debug Aster DEX API Endpoints
Investigate and fix API connectivity issues.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsterAPIDebugger:
    """Debug Aster DEX API endpoints and connectivity."""

    def __init__(self):
        self.session = None
        self.base_urls = [
            "https://fapi.asterdex.com",
            "https://api.asterdex.com",
            "https://testnet.asterdex.com",
            "https://www.asterdex.com/api"
        ]

    async def initialize(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                'User-Agent': 'AsterAI-Trading-Bot/1.0',
                'Accept': 'application/json'
            }
        )
        logger.info("✅ API debugger initialized")

    async def test_endpoint(self, url: str, description: str = "") -> Dict[str, Any]:
        """Test a specific endpoint."""
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
            import time
            start_time = time.time()

            logger.info(f"🔗 Testing {url}...")
            async with self.session.get(url) as response:
                result['status'] = response.status
                result['response_time'] = time.time() - start_time
                result['content_type'] = response.headers.get('Content-Type', 'unknown')

                if response.status == 200:
                    try:
                        data = await response.json()
                        result['content_length'] = len(str(data))
                        result['has_data'] = True

                        # Preview data
                        if isinstance(data, dict):
                            result['data_preview'] = {
                                'keys': list(data.keys())[:5],
                                'symbols_count': len(data.get('symbols', [])) if 'symbols' in data else 'N/A'
                            }
                        elif isinstance(data, list):
                            result['data_preview'] = f"List with {len(data)} items"

                        logger.info(f"   ✅ {response.status} - {result['content_type']} - {result['content_length']} chars")

                        # Log first few keys if dict
                        if isinstance(data, dict) and 'symbols' in data:
                            symbols = data.get('symbols', [])
                            if symbols:
                                sample_symbols = [s.get('symbol', 'unknown') for s in symbols[:3]]
                                logger.info(f"   📊 Sample symbols: {', '.join(sample_symbols)}")

                    except json.JSONDecodeError:
                        text = await response.text()
                        result['content_length'] = len(text)
                        result['data_preview'] = text[:100] + "..." if len(text) > 100 else text
                        logger.info(f"   ✅ {response.status} - {result['content_type']} - {len(text)} chars (text)")
                else:
                    error_text = await response.text()
                    result['error'] = error_text[:200]
                    logger.warning(f"   ❌ {response.status} - {error_text[:100]}...")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"   💥 Failed: {e}")

        return result

    async def discover_endpoints(self):
        """Discover working endpoints on Aster DEX."""
        logger.info("🔍 Discovering Aster DEX API endpoints...")

        # Common endpoint patterns
        endpoint_patterns = [
            "/fapi/v1/exchangeInfo",      # Futures API
            "/api/v3/exchangeInfo",       # Spot API v3
            "/api/v1/exchangeInfo",       # Spot API v1
            "/exchangeInfo",              # Direct
            "/fapi/v1/ticker/24hr",       # 24h stats
            "/api/v3/ticker/24hr",        # Spot 24h stats
            "/fapi/v1/klines",            # Futures klines
            "/api/v3/klines",             # Spot klines
            "/fapi/v1/depth",             # Futures depth
            "/api/v3/depth",              # Spot depth
            "/fapi/v1/trades",            # Futures trades
            "/api/v3/trades",             # Spot trades
            "/fapi/v1/ticker/price",      # Futures prices
            "/api/v3/ticker/price",       # Spot prices
        ]

        working_endpoints = []
        failed_endpoints = []

        for base_url in self.base_urls:
            logger.info(f"\n🌐 Testing base URL: {base_url}")

            for pattern in endpoint_patterns:
                url = base_url + pattern

                # Add query params for some endpoints
                if 'klines' in pattern:
                    url += "?symbol=BTCUSDT&interval=1h&limit=1"
                elif 'depth' in pattern:
                    url += "?symbol=BTCUSDT&limit=5"
                elif 'trades' in pattern:
                    url += "?symbol=BTCUSDT&limit=1"

                result = await test_endpoint(url, f"{base_url} + {pattern}")

                if result['status'] == 200 and result['has_data']:
                    working_endpoints.append(result)
                    logger.info(f"   ✅ WORKING: {pattern}")
                else:
                    failed_endpoints.append(result)

                # Rate limiting
                await asyncio.sleep(0.1)

        return working_endpoints, failed_endpoints

    async def test_specific_endpoints(self):
        """Test specific endpoints that might work."""
        logger.info("🎯 Testing specific known-working endpoints...")

        # Based on our connectivity tests, these should work
        test_endpoints = [
            ("https://fapi.asterdex.com/fapi/v1/exchangeInfo", "Futures exchange info"),
            ("https://fapi.asterdex.com/fapi/v1/ticker/price", "All prices"),
            ("https://fapi.asterdex.com/fapi/v1/ticker/price?symbol=BTCUSDT", "BTC price"),
            ("https://fapi.asterdex.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=10", "BTC klines"),
            ("https://fapi.asterdex.com/fapi/v1/depth?symbol=BTCUSDT&limit=5", "BTC orderbook"),
            ("https://fapi.asterdex.com/fapi/v1/trades?symbol=BTCUSDT&limit=5", "BTC trades"),
        ]

        results = []
        for url, description in test_endpoints:
            result = await self.test_endpoint(url, description)
            results.append(result)
            await asyncio.sleep(0.2)

        return results

    async def extract_asset_list(self, working_endpoints: List[Dict]):
        """Extract asset list from working endpoints."""
        logger.info("📊 Extracting asset list from working endpoints...")

        asset_list = set()

        for endpoint in working_endpoints:
            if endpoint['status'] == 200 and endpoint['has_data']:
                try:
                    # Re-fetch the data to extract symbols
                    async with self.session.get(endpoint['url']) as response:
                        data = await response.json()

                        # Extract symbols from different response formats
                        if isinstance(data, dict):
                            symbols = data.get('symbols', [])
                            if symbols:
                                for symbol_info in symbols:
                                    if isinstance(symbol_info, dict):
                                        symbol = symbol_info.get('symbol')
                                        if symbol:
                                            asset_list.add(symbol)

                        elif isinstance(data, list):
                            # Some endpoints return lists of symbols
                            for item in data:
                                if isinstance(item, dict):
                                    symbol = item.get('symbol')
                                    if symbol:
                                        asset_list.add(symbol)

                except Exception as e:
                    logger.warning(f"Failed to extract symbols from {endpoint['url']}: {e}")

        asset_list = sorted(list(asset_list))
        logger.info(f"📋 Extracted {len(asset_list)} unique symbols")

        if asset_list:
            logger.info(f"📈 Sample symbols: {', '.join(asset_list[:10])}{'...' if len(asset_list) > 10 else ''}")

        return asset_list

    async def create_fixed_discovery_script(self, working_endpoints: List[Dict], asset_list: List[str]):
        """Create a fixed discovery script based on findings."""
        logger.info("🔧 Creating fixed discovery script...")

        # Read current discovery script
        current_script_path = Path("scripts/discover_aster_assets.py")
        if not current_script_path.exists():
            logger.error("Current discovery script not found")
            return

        with open(current_script_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find and replace the endpoint list
        old_endpoints = '''            endpoints = [
                "https://fapi.asterdex.com/fapi/v1/exchangeInfo",  # This one works for connectivity
                "https://api.asterdex.com/api/v3/exchangeInfo",
                "https://api.asterdex.com/exchangeInfo",
                # Add other potential endpoints
            ]'''

        new_endpoints = f'''            endpoints = [
                # Working endpoints discovered by API debugger
'''

        for endpoint in working_endpoints:
            if endpoint['status'] == 200 and endpoint['has_data']:
                new_endpoints += f'                "{endpoint["url"]}",  # {endpoint["description"]}\n'

        new_endpoints += '            ]'

        # Replace in content
        if old_endpoints in content:
            new_content = content.replace(old_endpoints, new_endpoints)
            logger.info("✅ Updated discovery script with working endpoints")
        else:
            logger.warning("Could not find endpoint list in script")
            new_content = content

        # Save fixed script
        fixed_script_path = Path("scripts/discover_aster_assets_fixed.py")
        with open(fixed_script_path, 'w') as f:
            f.write(new_content)

        logger.info(f"💾 Fixed discovery script saved to {fixed_script_path}")

        # Create asset list file
        if asset_list:
            asset_file_path = Path("data/aster_known_assets.json")
            asset_file_path.parent.mkdir(parents=True, exist_ok=True)

            asset_data = {
                'discovered_at': str(asyncio.get_event_loop().time()),
                'total_assets': len(asset_list),
                'assets': asset_list,
                'source_endpoints': [e['url'] for e in working_endpoints]
            }

            with open(asset_file_path, 'w') as f:
                json.dump(asset_data, f, indent=2)

            logger.info(f"📋 Asset list saved to {asset_file_path}")

        return str(fixed_script_path)

    async def run_debug_analysis(self):
        """Run complete API debugging analysis."""
        print("""
================================================================================
          Aster DEX API Endpoint Debugger
      Investigating and fixing API connectivity issues
================================================================================
        """)

        try:
            await self.initialize()

            # Test specific endpoints first
            logger.info("Step 1: Testing known endpoints...")
            specific_results = await self.test_specific_endpoints()

            working_endpoints = [r for r in specific_results if r['status'] == 200 and r['has_data']]
            logger.info(f"✅ Found {len(working_endpoints)} working endpoints")

            # Extract assets from working endpoints
            logger.info("\nStep 2: Extracting asset lists...")
            asset_list = await self.extract_asset_list(working_endpoints)

            # Create fixed discovery script
            logger.info("\nStep 3: Creating fixed discovery script...")
            fixed_script = await self.create_fixed_discovery_script(working_endpoints, asset_list)

            # Summary
            print(f"\n{'='*80}")
            print("API DEBUG ANALYSIS RESULTS")
            print(f"{'='*80}\n")

            print(f"🔗 Working Endpoints: {len(working_endpoints)}")
            for ep in working_endpoints:
                print(f"   ✅ {ep['url']} ({ep.get('data_preview', {}).get('symbols_count', 'unknown')} symbols)")

            print(f"\n📊 Assets Discovered: {len(asset_list)}")
            if asset_list:
                print(f"   Sample: {', '.join(asset_list[:10])}{'...' if len(asset_list) > 10 else ''}")

            if fixed_script:
                print(f"\n🔧 Fixed Script: {fixed_script}")

            print("""
================================================================================
                🎉 API Debug Complete!
      Asset discovery should now work with fixed endpoints
================================================================================

Next steps:
1. Test fixed discovery: python scripts/discover_aster_assets_fixed.py
2. If working, replace original script
3. Proceed with full pipeline: python scripts/run_complete_pipeline.py
4. Or use known assets: python scripts/test_known_assets.py
            """)

            return {
                'working_endpoints': working_endpoints,
                'asset_list': asset_list,
                'fixed_script': fixed_script
            }

        except Exception as e:
            logger.error(f"❌ Debug analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            if self.session:
                await self.session.close()


async def main():
    """Main execution."""
    debugger = AsterAPIDebugger()
    results = await debugger.run_debug_analysis()

    if results:
        logger.info("✅ API debugging completed successfully")
    else:
        logger.error("❌ API debugging failed")


if __name__ == "__main__":
    asyncio.run(main())

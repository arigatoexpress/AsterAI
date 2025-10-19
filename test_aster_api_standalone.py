"""
Standalone Aster DEX API Connectivity Test Script
Tests all critical API endpoints and provides detailed diagnostics.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_trader.execution.aster_client import AsterClient


class APITester:
    """Comprehensive API testing suite for Aster DEX."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
    
    def log_test(self, test_name: str, status: str, details: Any = None, error: str = None):
        """Log test result."""
        self.results["tests"][test_name] = {
            "status": status,
            "details": details,
            "error": error
        }
        self.results["summary"]["total"] += 1
        if status == "PASS":
            self.results["summary"]["passed"] += 1
            print(f"[PASS] {test_name}")
        else:
            self.results["summary"]["failed"] += 1
            print(f"[FAIL] {test_name} - {error}")
        
        if details:
            print(f"   Details: {json.dumps(details, indent=2)}")
    
    async def test_api_keys(self):
        """Test 1: Verify API keys are set."""
        test_name = "API Keys Configuration"
        try:
            if not self.api_key or self.api_key == "demo_key":
                self.log_test(test_name, "FAIL", error="API key not set or using demo key")
                return False
            
            if not self.secret_key or self.secret_key == "demo_secret":
                self.log_test(test_name, "FAIL", error="Secret key not set or using demo secret")
                return False
            
            self.log_test(test_name, "PASS", details={
                "api_key_length": len(self.api_key),
                "secret_key_length": len(self.secret_key)
            })
            return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def test_client_initialization(self):
        """Test 2: Initialize AsterClient."""
        test_name = "Client Initialization"
        try:
            client = AsterClient(api_key=self.api_key, secret_key=self.secret_key)
            self.log_test(test_name, "PASS", details={
                "base_url": client.config.base_url,
                "ws_url": client.config.ws_url
            })
            return client
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return None
    
    async def test_server_time(self, client: AsterClient):
        """Test 3: Get server time (lightweight endpoint)."""
        test_name = "Server Time Endpoint"
        try:
            async with client:
                server_time = await client.get_server_time()
                self.log_test(test_name, "PASS", details={
                    "server_time": server_time,
                    "local_time": datetime.now().isoformat()
                })
                return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def test_account_info(self, client: AsterClient):
        """Test 4: Get account information."""
        test_name = "Account Info Endpoint"
        try:
            async with client:
                account_info = await client.get_account_info()
                self.log_test(test_name, "PASS", details={
                    "total_balance": account_info.total_balance,
                    "available_balance": account_info.available_balance,
                    "used_margin": account_info.used_margin,
                    "positions_count": len(account_info.positions),
                    "open_orders_count": len(account_info.open_orders)
                })
                return account_info
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return None
    
    async def test_market_data_btc(self, client: AsterClient):
        """Test 5: Get BTC market data."""
        test_name = "Market Data - BTCUSDT"
        try:
            async with client:
                ticker = await client.get_24hr_ticker("BTCUSDT")
                # Handle both dict and dataclass responses
                if isinstance(ticker, dict):
                    details = {
                        "symbol": ticker.get("symbol"),
                        "last_price": ticker.get("lastPrice"),
                        "volume": ticker.get("volume"),
                        "price_change_pct": ticker.get("priceChangePercent")
                    }
                else:
                    # It's a dataclass (TickerPrice)
                    details = {
                        "symbol": getattr(ticker, "symbol", None),
                        "last_price": getattr(ticker, "last_price", None),
                        "volume": getattr(ticker, "volume", None),
                        "price_change_pct": getattr(ticker, "price_change_percent", None)
                    }
                self.log_test(test_name, "PASS", details=details)
                return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def test_market_data_eth(self, client: AsterClient):
        """Test 6: Get ETH market data."""
        test_name = "Market Data - ETHUSDT"
        try:
            async with client:
                ticker = await client.get_24hr_ticker("ETHUSDT")
                # Handle both dict and dataclass responses
                if isinstance(ticker, dict):
                    details = {
                        "symbol": ticker.get("symbol"),
                        "last_price": ticker.get("lastPrice"),
                        "volume": ticker.get("volume")
                    }
                else:
                    # It's a dataclass (TickerPrice)
                    details = {
                        "symbol": getattr(ticker, "symbol", None),
                        "last_price": getattr(ticker, "last_price", None),
                        "volume": getattr(ticker, "volume", None)
                    }
                self.log_test(test_name, "PASS", details=details)
                return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def test_order_book(self, client: AsterClient):
        """Test 7: Get order book data."""
        test_name = "Order Book - BTCUSDT"
        try:
            async with client:
                order_book = await client.get_order_book("BTCUSDT", limit=10)
                self.log_test(test_name, "PASS", details={
                    "bids_count": len(order_book.get("bids", [])),
                    "asks_count": len(order_book.get("asks", [])),
                    "best_bid": order_book.get("bids", [[0]])[0][0] if order_book.get("bids") else None,
                    "best_ask": order_book.get("asks", [[0]])[0][0] if order_book.get("asks") else None
                })
                return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def test_recent_trades(self, client: AsterClient):
        """Test 8: Get recent trades."""
        test_name = "Recent Trades - BTCUSDT"
        try:
            async with client:
                trades = await client.get_recent_trades("BTCUSDT", limit=5)
                self.log_test(test_name, "PASS", details={
                    "trades_count": len(trades) if isinstance(trades, list) else 0
                })
                return True
        except Exception as e:
            self.log_test(test_name, "FAIL", error=str(e))
            return False
    
    async def run_all_tests(self):
        """Run all API tests."""
        print("=" * 60)
        print("Aster DEX API Connectivity Test Suite")
        print("=" * 60)
        print()
        
        # Test 1: API Keys
        if not await self.test_api_keys():
            print("\nAPI keys not configured. Cannot proceed with tests.")
            return self.results
        
        # Test 2: Client Initialization
        client = await self.test_client_initialization()
        if not client:
            print("\nFailed to initialize client. Cannot proceed with tests.")
            return self.results
        
        # Test 3: Server Time
        await self.test_server_time(client)
        
        # Test 4: Account Info
        account_info = await self.test_account_info(client)
        
        # Test 5-6: Market Data
        await self.test_market_data_btc(client)
        await self.test_market_data_eth(client)
        
        # Test 7: Order Book
        await self.test_order_book(client)
        
        # Test 8: Recent Trades
        await self.test_recent_trades(client)
        
        # Print summary
        print()
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total Tests: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']}")
        print(f"Failed: {self.results['summary']['failed']}")
        
        if self.results['summary']['failed'] == 0:
            print("\nAll tests passed! API is fully functional.")
        else:
            print(f"\n{self.results['summary']['failed']} test(s) failed. Review errors above.")
        
        # Save results to file
        with open("api_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: api_test_results.json")
        
        return self.results


async def main():
    """Main entry point."""
    # Get API keys from environment
    api_key = os.getenv("ASTER_API_KEY")
    secret_key = os.getenv("ASTER_API_SECRET")
    
    if not api_key or not secret_key:
        print("Error: ASTER_API_KEY and ASTER_API_SECRET must be set in environment")
        print("\nUsage:")
        print("  Windows: ")
        print("    $env:ASTER_API_KEY='your_key'")
        print("    $env:ASTER_API_SECRET='your_secret'")
        print("    python test_aster_api_standalone.py")
        print("\n  Linux/Mac:")
        print("    export ASTER_API_KEY='your_key'")
        print("    export ASTER_API_SECRET='your_secret'")
        print("    python test_aster_api_standalone.py")
        sys.exit(1)
    
    # Run tests
    tester = APITester(api_key, secret_key)
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())


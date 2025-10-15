#!/usr/bin/env python3
"""
Aster DEX Data Feed Dashboard
Comprehensive real-time data display for Aster DEX markets and account information.
"""

import asyncio
import sys
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
import json

# Add project root to path
sys.path.insert(0, '.')

from mcp_trader.config import get_settings, PRIORITY_SYMBOLS
from mcp_trader.execution.aster_client import AsterClient
from mcp_trader.data.aster_feed import AsterDataFeed
from mcp_trader.trading.autonomous_trader import AutonomousTrader


class AsterDataDashboard:
    """Comprehensive Aster DEX data feed dashboard."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsterClient(
            self.settings.aster_api_key,
            self.settings.aster_api_secret
        )
        self.data_feed = AsterDataFeed()
        self.market_data: Dict[str, Any] = {}
        self.account_data: Dict[str, Any] = {}
        self.running = False

    async def run_dashboard(self, refresh_rate: int = 5):
        """Run the interactive data dashboard."""
        print("📊 Aster DEX Data Feed Dashboard")
        print("=" * 50)
        print(f"🔄 Refresh Rate: {refresh_rate} seconds")
        print(f"🎯 Monitored Symbols: {', '.join(PRIORITY_SYMBOLS)}")
        print("Press Ctrl+C to exit")
        print()

        self.running = True

        try:
            # Initial data load
            await self._load_initial_data()

            # Main dashboard loop
            while self.running:
                await self._display_dashboard()
                await asyncio.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
        finally:
            await self._cleanup()

    async def _load_initial_data(self):
        """Load initial market and account data."""
        print("🔄 Loading initial data...")

        async with self.client:
            # Test connectivity
            try:
                connected = await self.client.test_connectivity()
                print(f"✅ API Connectivity: {'OK' if connected else 'FAILED'}")
            except Exception as e:
                print(f"⚠️  API Connectivity: {e}")

            # Load market data for each symbol
            for symbol in PRIORITY_SYMBOLS:
                try:
                    # 24hr ticker
                    ticker = await self.client.get_24hr_ticker(symbol)
                    self.market_data[f"{symbol}_ticker"] = ticker

                    # Order book
                    orderbook = await self.client.get_order_book(symbol, 10)
                    self.market_data[f"{symbol}_orderbook"] = orderbook

                    print(f"✅ Loaded data for {symbol}")

                except Exception as e:
                    print(f"⚠️  Failed to load {symbol} data: {str(e)[:50]}...")

            # Load account data (may fail with demo keys)
            try:
                account_info = await self.client.get_account_info()
                self.account_data['account_info'] = account_info

                balance = await self.client.get_account_balance_v2()
                self.account_data['balance'] = balance

                positions = await self.client.get_positions()
                self.account_data['positions'] = positions

                print("✅ Account data loaded")

            except Exception as e:
                print(f"⚠️  Account data: Expected with real keys - {str(e)[:50]}...")
                # Set empty defaults
                self.account_data = {
                    'account_info': None,
                    'balance': [],
                    'positions': []
                }

        print("✅ Initial data load complete")
        print()

    async def _display_dashboard(self):
        """Display the comprehensive data dashboard."""
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")

        # Header
        print("📊 Aster DEX Live Data Feed")
        print("=" * 60)
        print(f"🕒 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Market Data Section
        await self._display_market_data()

        # Account Data Section
        await self._display_account_data()

        # System Status
        await self._display_system_status()

        print("-" * 60)

    async def _display_market_data(self):
        """Display market data for all symbols."""
        print("🌍 MARKET DATA")
        print("-" * 30)

        for symbol in PRIORITY_SYMBOLS:
            print(f"\n💎 {symbol}")
            print("-" * 20)

            # Ticker data
            ticker_key = f"{symbol}_ticker"
            if ticker_key in self.market_data:
                ticker = self.market_data[ticker_key]
                if ticker and isinstance(ticker, dict):
                    print("24hr Stats:")
                    print(".2f"                    print(".2f"                    print(".1f"                    print(".2f"                    print(".2f"                else:
                    print("Ticker data unavailable")
            else:
                print("Ticker data not loaded")

            # Order book
            ob_key = f"{symbol}_orderbook"
            if ob_key in self.market_data:
                orderbook = self.market_data[ob_key]
                if orderbook and isinstance(orderbook, dict):
                    bids = orderbook.get('bids', [])[:3]  # Top 3 bids
                    asks = orderbook.get('asks', [])[:3]  # Top 3 asks

                    print("Order Book (Top 3):")
                    if bids:
                        print("  Bids:")
                        for price, qty in bids:
                            print(".2f"
                    if asks:
                        print("  Asks:")
                        for price, qty in asks:
                            print(".2f"
                else:
                    print("Order book data unavailable")
            else:
                print("Order book data not loaded")

    async def _display_account_data(self):
        """Display account and position data."""
        print("\n💼 ACCOUNT DATA")
        print("-" * 30)

        # Account info
        if self.account_data.get('account_info'):
            account = self.account_data['account_info']
            print("Account Overview:")
            print(".2f"            print(".2f"            print(".2f"            print(".2f"            print(f"  Positions: {len(account.positions) if hasattr(account, 'positions') else 'N/A'}")
        else:
            print("Account information not available (requires valid API keys)")

        # Balance
        balance = self.account_data.get('balance', [])
        if balance:
            print("\nAsset Balances:")
            for asset in balance[:5]:  # Show top 5 assets
                if isinstance(asset, dict):
                    asset_name = asset.get('asset', 'Unknown')
                    free_balance = float(asset.get('free', asset.get('availableBalance', 0)))
                    locked_balance = float(asset.get('locked', 0))
                    total_balance = free_balance + locked_balance
                    print(f"  {asset_name}: {total_balance:.4f} (Free: {free_balance:.4f}, Locked: {locked_balance:.4f})")
        else:
            print("\nBalance data not available")

        # Positions
        positions = self.account_data.get('positions', [])
        if positions:
            print("
Open Positions:")
            for pos in positions[:5]:  # Show top 5 positions
                if isinstance(pos, dict):
                    symbol = pos.get('symbol', 'N/A')
                    size = pos.get('positionAmt', 0)
                    entry = pos.get('entryPrice', 0)
                    pnl = pos.get('unrealizedProfit', 0)
                    print(".6f"        else:
            print("
No open positions")
        else:
            print("
Position data not available")

    async def _display_system_status(self):
        """Display system status and connection info."""
        print("\n⚙️  SYSTEM STATUS")
        print("-" * 30)

        # API connectivity
        try:
            async with self.client:
                connectivity = await self.client.test_connectivity()
                status = "🟢 CONNECTED" if connectivity else "🔴 DISCONNECTED"
                print(f"API Status: {status}")
        except Exception as e:
            print(f"API Status: 🔴 ERROR - {str(e)[:30]}...")

        # WebSocket status
        ws_status = "🟢 ACTIVE" if hasattr(self, 'data_feed') and self.data_feed else "🟡 INACTIVE"
        print(f"WebSocket: {ws_status}")

        # Data freshness
        print(f"Data Points: {len(self.market_data)} market, {len(self.account_data)} account")

        # Memory usage (rough estimate)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(".1f"
    async def _cleanup(self):
        """Clean up resources."""
        self.running = False
        print("🔄 Cleaning up dashboard resources...")


class AsterDataExporter:
    """Export Aster DEX data to various formats."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsterClient(
            self.settings.aster_api_key,
            self.settings.aster_api_secret
        )

    async def export_market_data(self, format: str = 'json', filename: str = None) -> str:
        """Export comprehensive market data."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"aster_market_data_{timestamp}.{format}"

        data = {
            'timestamp': datetime.now().isoformat(),
            'symbols': {},
            'system_info': {
                'api_connected': False,
                'symbols_monitored': len(PRIORITY_SYMBOLS),
                'export_format': format
            }
        }

        async with self.client:
            # Test connectivity
            try:
                data['system_info']['api_connected'] = await self.client.test_connectivity()
            except:
                pass

            # Collect data for each symbol
            for symbol in PRIORITY_SYMBOLS:
                symbol_data = {}

                try:
                    ticker = await self.client.get_24hr_ticker(symbol)
                    symbol_data['ticker'] = ticker
                except Exception as e:
                    symbol_data['ticker_error'] = str(e)

                try:
                    orderbook = await self.client.get_order_book(symbol, 20)
                    symbol_data['orderbook'] = orderbook
                except Exception as e:
                    symbol_data['orderbook_error'] = str(e)

                data['symbols'][symbol] = symbol_data

            # Export to requested format
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == 'csv':
                # Convert to CSV format
                self._export_as_csv(data, filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

        return filename

    def _export_as_csv(self, data: Dict, filename: str):
        """Export data in CSV format."""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Symbol', 'Data_Type', 'Key', 'Value'])

            # Data rows
            for symbol, symbol_data in data['symbols'].items():
                for data_type, content in symbol_data.items():
                    if isinstance(content, dict):
                        for key, value in content.items():
                            writer.writerow([symbol, data_type, key, str(value)])
                    else:
                        writer.writerow([symbol, data_type, 'data', str(content)])


async def main():
    """Main entry point for data feed dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description='Aster DEX Data Feed Dashboard')
    parser.add_argument('--export', choices=['json', 'csv'],
                       help='Export market data and exit')
    parser.add_argument('--refresh-rate', type=int, default=5,
                       help='Dashboard refresh rate in seconds')
    parser.add_argument('--export-file', type=str,
                       help='Custom export filename')

    args = parser.parse_args()

    if args.export:
        # Export mode
        exporter = AsterDataExporter()
        try:
            filename = await exporter.export_market_data(args.export, args.export_file)
            print(f"✅ Data exported to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    else:
        # Dashboard mode
        dashboard = AsterDataDashboard()
        await dashboard.run_dashboard(args.refresh_rate)


if __name__ == "__main__":
    asyncio.run(main())

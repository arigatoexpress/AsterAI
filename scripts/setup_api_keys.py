#!/usr/bin/env python3
"""
Interactive API Key Setup Script
Helps users configure all required API keys for the Ultimate Trading System.
"""

import sys
import os
from pathlib import Path
import json
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.data.api_manager import APIKeyManager, APICredentials


def print_banner():
    """Display setup banner"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║               Ultimate AI Trading System - API Setup           ║
╚════════════════════════════════════════════════════════════════╝
    """)


def print_api_info():
    """Print information about each API"""
    api_info = {
        'Aster DEX': {
            'required': True,
            'free': False,
            'description': 'Required for live trading on Aster DEX',
            'get_key': 'Contact Aster DEX support for API access'
        },
        'Alpha Vantage': {
            'required': False,
            'free': True,
            'description': 'Stock market data (S&P 500, etc.)',
            'get_key': 'https://www.alphavantage.co/support/#api-key',
            'limits': '5 calls/min, 500/day (free tier)'
        },
        'FRED': {
            'required': False,
            'free': True,
            'description': 'Economic indicators (GDP, CPI, Fed rates)',
            'get_key': 'https://fred.stlouisfed.org/docs/api/api_key.html',
            'limits': 'Unlimited'
        },
        'NewsAPI': {
            'required': False,
            'free': True,
            'description': 'News sentiment analysis',
            'get_key': 'https://newsapi.org/register',
            'limits': '1000 requests/day (free tier)'
        },
        'Finnhub': {
            'required': False,
            'free': True,
            'description': 'Alternative financial data',
            'get_key': 'https://finnhub.io/register',
            'limits': '60 calls/min (free tier)'
        },
        'Metals-API': {
            'required': False,
            'free': True,
            'description': 'Commodity prices (Gold, Silver, etc.)',
            'get_key': 'https://metals-api.com/',
            'limits': '50 requests/month (free tier)'
        }
    }
    
    print("\n📋 API Information:")
    print("=" * 60)
    
    for api_name, info in api_info.items():
        status = "⚠️  REQUIRED" if info['required'] else "✓ Optional"
        free = "FREE" if info['free'] else "PAID"
        
        print(f"\n{api_name} [{free}] {status}")
        print(f"  Description: {info['description']}")
        print(f"  Get key at: {info['get_key']}")
        if 'limits' in info:
            print(f"  Rate limits: {info['limits']}")


def get_existing_keys(manager: APIKeyManager) -> APICredentials:
    """Load existing credentials"""
    return manager.load_credentials()


def prompt_for_key(api_name: str, current_value: Optional[str] = None) -> Optional[str]:
    """Prompt user for API key"""
    if current_value:
        masked = f"{current_value[:4]}...{current_value[-4:]}" if len(current_value) > 8 else current_value
        print(f"\nCurrent {api_name}: {masked}")
        choice = input("Keep current value? (Y/n): ").strip().lower()
        if choice != 'n':
            return current_value
    
    value = input(f"Enter {api_name} (or press Enter to skip): ").strip()
    return value if value else None


def interactive_setup():
    """Run interactive setup"""
    print_banner()
    
    # Initialize manager
    manager = APIKeyManager()
    credentials = get_existing_keys(manager)
    
    # Show current status
    print("\n" + credentials.get_status_report())
    
    # Ask if user wants to see API information
    show_info = input("\nWould you like to see API information? (y/N): ").strip().lower()
    if show_info == 'y':
        print_api_info()
    
    # Setup process
    print("\n\n🔧 API Key Configuration")
    print("=" * 60)
    print("Note: You can skip optional APIs and add them later\n")
    
    # Aster DEX (required)
    print("\n1️⃣  Aster DEX API (REQUIRED for trading)")
    credentials.aster_api_key = prompt_for_key("Aster API Key", credentials.aster_api_key)
    credentials.aster_secret_key = prompt_for_key("Aster Secret Key", credentials.aster_secret_key)
    
    # Financial APIs (optional)
    print("\n2️⃣  Financial Data APIs (Optional)")
    credentials.alpha_vantage_key = prompt_for_key("Alpha Vantage API Key", credentials.alpha_vantage_key)
    credentials.finnhub_key = prompt_for_key("Finnhub API Key", credentials.finnhub_key)
    
    # Economic data
    print("\n3️⃣  Economic Data APIs (Optional)")
    credentials.fred_api_key = prompt_for_key("FRED API Key", credentials.fred_api_key)
    
    # News & Sentiment
    print("\n4️⃣  News & Sentiment APIs (Optional)")
    credentials.newsapi_key = prompt_for_key("NewsAPI Key", credentials.newsapi_key)
    
    # Commodities
    print("\n5️⃣  Commodities APIs (Optional)")
    credentials.metals_api_key = prompt_for_key("Metals-API Key", credentials.metals_api_key)
    
    # Save configuration
    print("\n\n💾 Saving configuration...")
    
    # Ask where to save
    save_choice = input("\nSave to: [1] .env file, [2] .api_keys.json, [3] Both (default: 3): ").strip()
    
    if save_choice != '2':
        # Save to .env
        env_path = Path('.env')
        with open(env_path, 'w') as f:
            if credentials.aster_api_key:
                f.write(f"ASTER_API_KEY={credentials.aster_api_key}\n")
            if credentials.aster_secret_key:
                f.write(f"ASTER_SECRET_KEY={credentials.aster_secret_key}\n")
            if credentials.alpha_vantage_key:
                f.write(f"ALPHA_VANTAGE_API_KEY={credentials.alpha_vantage_key}\n")
            if credentials.finnhub_key:
                f.write(f"FINNHUB_API_KEY={credentials.finnhub_key}\n")
            if credentials.fred_api_key:
                f.write(f"FRED_API_KEY={credentials.fred_api_key}\n")
            if credentials.newsapi_key:
                f.write(f"NEWSAPI_KEY={credentials.newsapi_key}\n")
            if credentials.metals_api_key:
                f.write(f"METALS_API_KEY={credentials.metals_api_key}\n")
        print(f"✓ Saved to {env_path}")
    
    if save_choice != '1':
        # Save to JSON
        manager.save_credentials(credentials)
        print(f"✓ Saved to {manager.config_file}")
    
    # Final status
    print("\n\n✅ Configuration Complete!")
    print("\n" + credentials.get_status_report())
    
    # Next steps
    validation = credentials.validate()
    if not validation['aster']:
        print("\n⚠️  Warning: Aster DEX keys not configured. Live trading will not work!")
    
    optional_configured = sum(1 for k, v in validation.items() if k != 'aster' and v)
    print(f"\n📊 Optional APIs configured: {optional_configured}/5")
    
    if optional_configured < 3:
        print("\n💡 Tip: Configure more APIs for better data coverage and signals!")
    
    print("\n\n🚀 Ready to collect data! Run:")
    print("   python scripts/collect_all_ultimate_data.py")


def main():
    """Main entry point"""
    # Check for command line arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python setup_api_keys.py [--interactive] [--export-template]")
        print("\nOptions:")
        print("  --interactive    Run interactive setup (default)")
        print("  --export-template Export .env template file")
        return
    
    if '--export-template' in sys.argv:
        manager = APIKeyManager()
        manager.export_env_template()
        return
    
    # Default to interactive mode
    try:
        interactive_setup()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Asset configuration for Aster DEX and Jewster.fun trading pairs.
Focus data ingestion and analysis on assets actually tradeable on these platforms.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TradingPair:
    """Represents a trading pair available on Aster/Jewster."""
    symbol: str
    base_asset: str
    quote_asset: str
    venues: List[str]  # ["aster", "jewster"]
    binance_symbol: Optional[str] = None  # Binance futures symbol
    okx_symbol: Optional[str] = None      # OKX swap symbol
    min_trade_size: float = 0.001
    price_precision: int = 2
    quantity_precision: int = 6


# Core tokens available on Aster DEX
ASTER_PAIRS = [
    # Major crypto pairs
    TradingPair(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol="BTCUSDT",
        okx_symbol="BTC-USDT-SWAP",
        min_trade_size=0.001,
        price_precision=2,
        quantity_precision=6
    ),
    TradingPair(
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol="ETHUSDT",
        okx_symbol="ETH-USDT-SWAP",
        min_trade_size=0.01,
        price_precision=2,
        quantity_precision=4
    ),
    
    # Aster native token
    TradingPair(
        symbol="ASTERUSDT",
        base_asset="ASTER",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol=None,  # May not be on Binance
        okx_symbol=None,      # May not be on OKX
        min_trade_size=1.0,
        price_precision=4,
        quantity_precision=2
    ),
    
    # Solana ecosystem
    TradingPair(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol="SOLUSDT",
        okx_symbol="SOL-USDT-SWAP",
        min_trade_size=0.1,
        price_precision=2,
        quantity_precision=3
    ),
    
    # Sui ecosystem
    TradingPair(
        symbol="SUIUSDT",
        base_asset="SUI",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol="SUIUSDT",
        okx_symbol="SUI-USDT-SWAP",
        min_trade_size=1.0,
        price_precision=4,
        quantity_precision=2
    ),
    
    # Pengu token (meme/community token)
    TradingPair(
        symbol="PENGUUSDT",
        base_asset="PENGU",
        quote_asset="USDT",
        venues=["aster"],
        binance_symbol=None,  # Likely not on major exchanges
        okx_symbol=None,
        min_trade_size=1000.0,
        price_precision=6,
        quantity_precision=0
    ),
]


def get_pairs_by_venue(venue: str) -> List[TradingPair]:
    """Get all trading pairs available on a specific venue."""
    return [pair for pair in ASTER_PAIRS if venue in pair.venues]


def get_pairs_with_external_data() -> List[TradingPair]:
    """Get pairs that have external data sources (Binance/OKX)."""
    return [pair for pair in ASTER_PAIRS if pair.binance_symbol or pair.okx_symbol]


def get_symbol_mapping() -> Dict[str, TradingPair]:
    """Get mapping from symbol to TradingPair object."""
    return {pair.symbol: pair for pair in ASTER_PAIRS}


def get_binance_symbols() -> List[str]:
    """Get all Binance symbols for data ingestion."""
    return [pair.binance_symbol for pair in ASTER_PAIRS if pair.binance_symbol]


def get_okx_symbols() -> List[str]:
    """Get all OKX symbols for data ingestion."""
    return [pair.okx_symbol for pair in ASTER_PAIRS if pair.okx_symbol]


def get_aster_symbols() -> List[str]:
    """Get all symbols available on Aster DEX."""
    return [pair.symbol for pair in get_pairs_by_venue("aster")]


# Priority symbols for initial data ingestion (most liquid/important)
PRIORITY_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT", 
    "ASTERUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "PENGUUSDT"
]


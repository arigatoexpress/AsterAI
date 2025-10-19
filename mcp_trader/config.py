from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class TradingPair:
    """Represents a trading pair available on Aster DEX."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_trade_size: float = 0.001
    price_precision: int = 2
    quantity_precision: int = 6
    max_leverage: int = 25
    maintenance_margin: float = 0.005  # 0.5%
    is_active: bool = True


class Settings(BaseSettings):
    """Aster DEX focused configuration settings."""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="allow")
    aster_api_key: str | None = None
    aster_api_secret: str | None = None
    aster_base_url: str = "https://fapi.asterdex.com"
    aster_ws_url: str = "wss://fstream.asterdex.com"
    log_level: str = "INFO"

    # Trading configuration
    max_portfolio_risk: float = 0.1  # 10% max portfolio risk
    max_single_position_risk: float = 0.05  # 5% max per position
    max_concurrent_positions: int = 3
    min_position_size_usd: float = 10.0
    max_position_size_usd: float = 1000.0

    # Grid trading defaults
    grid_levels: int = 10
    grid_spacing_percent: float = 2.0
    grid_position_size_usd: float = 50.0

    # Risk management
    stop_loss_threshold: float = 0.03  # 3%
    take_profit_threshold: float = 0.05  # 5%
    max_daily_loss: float = 0.15  # 15%
    volatility_multiplier: float = 1.5

    def validate(self) -> None:
        """Fail-fast schema and range validation for critical settings."""
        # API configuration
        if self.aster_base_url and not re.match(r"^https?://", self.aster_base_url):
            raise ValueError("aster_base_url must start with http:// or https://")
        if self.aster_ws_url and not re.match(r"^wss?://", self.aster_ws_url):
            raise ValueError("aster_ws_url must start with ws:// or wss://")

        # Risk bounds
        if not (0 < self.max_portfolio_risk <= 0.5):
            raise ValueError("max_portfolio_risk must be in (0, 0.5]")
        if not (0 < self.max_single_position_risk <= self.max_portfolio_risk):
            raise ValueError("max_single_position_risk must be >0 and <= max_portfolio_risk")
        if not (1 <= self.max_concurrent_positions <= 50):
            raise ValueError("max_concurrent_positions must be between 1 and 50")
        if not (0 < self.stop_loss_threshold < 1):
            raise ValueError("stop_loss_threshold must be in (0,1)")
        if not (0 < self.take_profit_threshold < 1):
            raise ValueError("take_profit_threshold must be in (0,1)")
        if not (0 < self.max_daily_loss < 1):
            raise ValueError("max_daily_loss must be in (0,1)")

        # Position sizing bounds
        if not (0 < self.min_position_size_usd <= self.max_position_size_usd):
            raise ValueError("min_position_size_usd must be >0 and <= max_position_size_usd")
        if not (self.max_position_size_usd <= 1_000_000):
            raise ValueError("max_position_size_usd is unreasonably high")

        # Grid strategy parameters
        if not (1 <= self.grid_levels <= 200):
            raise ValueError("grid_levels must be between 1 and 200")
        if not (0 < self.grid_spacing_percent <= 50):
            raise ValueError("grid_spacing_percent must be in (0, 50]")
        if not (self.grid_position_size_usd >= self.min_position_size_usd):
            raise ValueError("grid_position_size_usd must be >= min_position_size_usd")

    


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings instance."""
    s = Settings()
    s.validate()
    return s


# Core tokens available on Aster DEX for autonomous trading
ASTER_PAIRS = [
    # High-volume established assets (primary focus)
    TradingPair(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        min_trade_size=0.001,
        price_precision=2,
        quantity_precision=6,
        max_leverage=50,
        maintenance_margin=0.004  # 0.4%
    ),
    TradingPair(
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT",
        min_trade_size=0.01,
        price_precision=2,
        quantity_precision=4,
        max_leverage=50,
        maintenance_margin=0.004
    ),

    # High-volatility assets (grid trading focus)
    TradingPair(
        symbol="SOLUSDT",
        base_asset="SOL",
        quote_asset="USDT",
        min_trade_size=0.1,
        price_precision=2,
        quantity_precision=3,
        max_leverage=25,
        maintenance_margin=0.005
    ),
    TradingPair(
        symbol="SUIUSDT",
        base_asset="SUI",
        quote_asset="USDT",
        min_trade_size=1.0,
        price_precision=4,
        quantity_precision=2,
        max_leverage=25,
        maintenance_margin=0.005
    ),

    # Aster native token
    TradingPair(
        symbol="ASTERUSDT",
        base_asset="ASTER",
        quote_asset="USDT",
        min_trade_size=1.0,
        price_precision=4,
        quantity_precision=2,
        max_leverage=10,
        maintenance_margin=0.01  # Higher margin for native token
    ),

    # High-volatility meme token (volatility trading)
    TradingPair(
        symbol="PENGUUSDT",
        base_asset="PENGU",
        quote_asset="USDT",
        min_trade_size=100.0,
        price_precision=6,
        quantity_precision=0,
        max_leverage=5,  # Lower leverage for high volatility
        maintenance_margin=0.02  # Higher margin requirement
    ),
]


def get_symbol_mapping() -> Dict[str, TradingPair]:
    """Get mapping from symbol to TradingPair object."""
    return {pair.symbol: pair for pair in ASTER_PAIRS}


def get_aster_symbols() -> List[str]:
    """Get all symbols available on Aster DEX."""
    return [pair.symbol for pair in ASTER_PAIRS if pair.is_active]


# Priority symbols for autonomous trading (most liquid/important)
# Ordered by trading volume and volatility potential
PRIORITY_SYMBOLS = [
    "BTCUSDT",    # Primary - most liquid
    "ETHUSDT",    # Primary - high volume
    "SOLUSDT",    # High volatility - grid trading focus
    "SUIUSDT",    # High volatility - grid trading focus
    "ASTERUSDT",  # Native token - medium volatility
    "PENGUUSDT"   # Meme token - extreme volatility
]


def get_high_volatility_symbols() -> List[str]:
    """Get symbols suitable for volatility-based strategies."""
    return ["SOLUSDT", "SUIUSDT", "PENGUUSDT"]


def get_stable_symbols() -> List[str]:
    """Get more stable symbols for conservative strategies."""
    return ["BTCUSDT", "ETHUSDT"]


def get_grid_trading_symbols() -> List[str]:
    """Get symbols optimal for grid trading."""
    return ["SOLUSDT", "SUIUSDT", "ASTERUSDT"]


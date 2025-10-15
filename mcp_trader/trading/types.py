"""
Shared types and enums for the trading system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any


class TradingMode(Enum):
    """Trading operation modes."""
    GRID_TRADING = "grid"
    VOLATILITY_TRADING = "volatility"
    HYBRID = "hybrid"
    MANUAL = "manual"


class MarketRegime(Enum):
    """Market condition assessment."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class TradingDecision:
    """Represents a trading decision made by the autonomous system."""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "HOLD", "ADJUST_GRID"
    quantity: float
    price: float = None
    reason: str = ""
    confidence: float = 0.0
    risk_score: float = 0.0
    expected_profit: float = None


@dataclass
class PortfolioState:
    """Current portfolio state snapshot."""
    timestamp: datetime
    total_balance: float = 0.0
    available_balance: float = 0.0
    total_positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    active_positions: Dict[str, Any] = None  # Forward reference to avoid circular import
    active_grids: Dict[str, Dict] = None

    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = {}
        if self.active_grids is None:
            self.active_grids = {}


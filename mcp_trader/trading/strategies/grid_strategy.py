"""
Grid Trading Strategy for Aster DEX
Implements intelligent grid trading optimized for volatile assets.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ...execution.aster_client import AccountPosition, TickerPrice
from ..types import TradingDecision, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """Configuration for grid trading."""
    grid_levels: int = 10  # Number of grid levels
    grid_spacing_percent: float = 2.0  # Percentage spacing between grid levels
    position_size_per_level: float = 50.0  # USD per grid level
    max_position_size: float = 500.0  # Maximum position size per symbol
    rebalancing_threshold: float = 1.0  # Percentage deviation before rebalancing
    volatility_multiplier: float = 1.5  # Adjust spacing based on volatility
    min_order_size: float = 5.0  # Minimum order size in USD
    max_slippage_percent: float = 0.5  # Maximum allowed slippage


@dataclass
class GridLevel:
    """Represents a single grid level."""
    price: float
    buy_quantity: float = 0.0
    sell_quantity: float = 0.0
    is_active: bool = True


@dataclass
class GridPosition:
    """Represents a complete grid position."""
    symbol: str
    base_price: float
    levels: List[GridLevel]
    total_invested: float = 0.0
    unrealized_pnl: float = 0.0
    created_at: datetime = None
    last_rebalance: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_rebalance is None:
            self.last_rebalance = datetime.now()


class GridStrategy:
    """
    Intelligent grid trading strategy for volatile assets on Aster DEX.
    Adapts grid spacing and position sizing based on market volatility.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = GridConfig(**config)
        self.active_grids: Dict[str, GridPosition] = {}
        self.grid_history: List[Dict] = []  # Track grid performance

        logger.info(f"GridStrategy initialized with {self.config.grid_levels} levels, "
                   f"{self.config.grid_spacing_percent}% spacing")

    async def make_decisions(self, symbol: str, ticker: TickerPrice,
                           existing_position: Optional[AccountPosition],
                           market_regime: MarketRegime) -> List[TradingDecision]:
        """
        Generate trading decisions for grid strategy.
        """
        decisions = []

        try:
            # Check if we should have a grid for this symbol
            if not self._should_trade_symbol(symbol, ticker, market_regime):
                return decisions

            # Get or create grid position
            grid = self.active_grids.get(symbol)
            current_price = ticker.last_price

            if grid is None:
                # Initialize new grid
                if self._should_create_grid(symbol, current_price, market_regime):
                    grid = self._create_grid(symbol, current_price, market_regime)
                    self.active_grids[symbol] = grid
                    decisions.extend(self._get_grid_initialization_decisions(grid))
            else:
                # Adjust existing grid
                decisions.extend(self._get_grid_adjustment_decisions(grid, current_price, ticker))

        except Exception as e:
            logger.error(f"Error making grid decisions for {symbol}: {e}")

        return decisions

    def _should_trade_symbol(self, symbol: str, ticker: TickerPrice,
                           market_regime: MarketRegime) -> bool:
        """Determine if we should trade this symbol with grid strategy."""
        # Only trade priority symbols
        from ...config import get_settings
        settings = get_settings()

        from ...config import PRIORITY_SYMBOLS
        if symbol not in PRIORITY_SYMBOLS:
            return False

        # Check minimum liquidity (24h volume)
        min_volume_threshold = 10000  # $10k daily volume
        if ticker.quote_volume < min_volume_threshold:
            return False

        # Adjust based on market regime
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            # More aggressive in high volatility
            return True
        elif market_regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # Conservative in trending markets (grid works better in sideways)
            return ticker.price_change_percent < 5.0  # Less than 5% daily change
        else:
            # Good for sideways markets
            return True

    def _should_create_grid(self, symbol: str, current_price: float,
                          market_regime: MarketRegime) -> bool:
        """Determine if we should create a new grid position."""
        # Check if we already have a grid
        if symbol in self.active_grids:
            return False

        # Check position size limits
        total_grid_value = sum(grid.total_invested for grid in self.active_grids.values())
        if total_grid_value >= self.config.max_position_size * 5:  # Allow up to 5 grids
            return False

        # Market regime considerations
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            # Create grids more aggressively in volatile markets
            return True
        elif market_regime == MarketRegime.SIDEWAYS:
            # Ideal for grid trading
            return True
        else:
            # Be more selective in trending markets
            return False

    def _create_grid(self, symbol: str, base_price: float,
                    market_regime: MarketRegime) -> GridPosition:
        """Create a new grid position."""
        # Adjust spacing based on market regime
        base_spacing = self.config.grid_spacing_percent / 100.0

        if market_regime == MarketRegime.HIGH_VOLATILITY:
            spacing = base_spacing * self.config.volatility_multiplier
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            spacing = base_spacing * 0.8  # Tighter grid in low vol
        else:
            spacing = base_spacing

        # Calculate grid levels
        levels = []
        for i in range(self.config.grid_levels):
            level_price = base_price * (1 + spacing * (i - self.config.grid_levels // 2))
            level = GridLevel(price=level_price)
            levels.append(level)

        grid = GridPosition(
            symbol=symbol,
            base_price=base_price,
            levels=levels
        )

        logger.info(f"Created grid for {symbol} at ${base_price:.4f} with {len(levels)} levels")
        return grid

    def _get_grid_initialization_decisions(self, grid: GridPosition) -> List[TradingDecision]:
        """Get decisions to initialize a grid position."""
        decisions = []
        current_price = grid.base_price

        # Find the current price level and initialize positions
        for level in grid.levels:
            if abs(level.price - current_price) / current_price < 0.01:  # Within 1%
                # We're at this level, don't place initial orders
                continue

            # Calculate position size for this level
            position_value = self.config.position_size_per_level
            quantity = position_value / level.price

            # Ensure minimum order size
            if position_value < self.config.min_order_size:
                continue

            # Decide buy or sell based on level position relative to current price
            if level.price < current_price:
                # Buy level (below current price)
                level.buy_quantity = quantity
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=grid.symbol,
                    action="BUY",
                    quantity=quantity,
                    price=None,  # Market order
                    reason=f"Grid initialization - buy level at ${level.price:.4f}",
                    confidence=0.8,
                    risk_score=self._calculate_level_risk(level, current_price),
                    expected_profit=None
                ))
            else:
                # Sell level (above current price) - only if we have inventory
                # For now, we'll place limit orders for sell levels
                pass

        return decisions

    def _get_grid_adjustment_decisions(self, grid: GridPosition, current_price: float,
                                     ticker: TickerPrice) -> List[TradingDecision]:
        """Get decisions to adjust existing grid positions."""
        decisions = []

        # Check if grid needs rebalancing
        price_deviation = abs(current_price - grid.base_price) / grid.base_price * 100

        if price_deviation > self.config.rebalancing_threshold:
            # Price has moved significantly, consider rebalancing
            decisions.extend(self._get_rebalancing_decisions(grid, current_price, ticker))

        # Check for grid level triggers
        for level in grid.levels:
            if not level.is_active:
                continue

            price_diff = abs(current_price - level.price) / level.price

            # Check if price hit this level
            if price_diff < 0.005:  # Within 0.5%
                decisions.extend(self._handle_level_trigger(grid, level, current_price))

        return decisions

    def _handle_level_trigger(self, grid: GridPosition, level: GridLevel,
                            current_price: float) -> List[TradingDecision]:
        """Handle when price triggers a grid level."""
        decisions = []

        # Determine if this is a buy or sell trigger
        if current_price < level.price and level.buy_quantity > 0:
            # Price dropped to buy level - we should have bought here
            # This means we can now sell at a profit
            if level.sell_quantity > 0:
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=grid.symbol,
                    action="SELL",
                    quantity=level.sell_quantity,
                    price=None,
                    reason=f"Grid profit taking - sell level triggered at ${level.price:.4f}",
                    confidence=0.9,
                    risk_score=0.2,  # Low risk profit taking
                    expected_profit=(current_price - level.price) * level.sell_quantity
                ))

        elif current_price > level.price:
            # Price rose to sell level - place buy order for next dip
            position_value = self.config.position_size_per_level
            quantity = position_value / current_price

            if position_value >= self.config.min_order_size:
                level.buy_quantity = quantity
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=grid.symbol,
                    action="BUY",
                    quantity=quantity,
                    price=None,
                    reason=f"Grid buy level triggered at ${level.price:.4f}",
                    confidence=0.7,
                    risk_score=self._calculate_level_risk(level, current_price),
                    expected_profit=None
                ))

        return decisions

    def _get_rebalancing_decisions(self, grid: GridPosition, current_price: float,
                                 ticker: TickerPrice) -> List[TradingDecision]:
        """Get decisions to rebalance the grid."""
        decisions = []

        # If price has moved significantly, consider closing the grid
        # or adjusting the base price
        price_deviation = abs(current_price - grid.base_price) / grid.base_price

        if price_deviation > 10.0:  # 10% deviation
            # Close the entire grid position
            total_quantity = sum(level.buy_quantity + level.sell_quantity for level in grid.levels)

            if total_quantity > 0:
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=grid.symbol,
                    action="ADJUST_GRID",
                    quantity=total_quantity,
                    price=None,
                    reason=f"Grid rebalancing - closing grid due to {price_deviation:.1f}% price move",
                    confidence=0.95,
                    risk_score=0.3,
                    expected_profit=None
                ))

                # Remove the grid
                del self.active_grids[grid.symbol]

        elif price_deviation > 5.0:  # 5% deviation
            # Adjust grid base price
            new_base_price = current_price
            grid.base_price = new_base_price
            grid.last_rebalance = datetime.now()

            logger.info(f"Rebalanced grid for {grid.symbol} to new base price ${new_base_price:.4f}")

        return decisions

    def _calculate_level_risk(self, level: GridLevel, current_price: float) -> float:
        """Calculate risk score for a grid level."""
        # Risk increases with distance from current price
        distance_percent = abs(level.price - current_price) / current_price

        # Base risk on distance
        risk_score = min(distance_percent * 2, 1.0)  # Max 100% risk score

        # Adjust for position size
        position_value = self.config.position_size_per_level
        if position_value > self.config.max_position_size * 0.1:  # Large position
            risk_score += 0.2

        return min(risk_score, 1.0)

    def get_grid_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific grid."""
        grid = self.active_grids.get(symbol)
        if not grid:
            return None

        return {
            'symbol': symbol,
            'base_price': grid.base_price,
            'levels': len(grid.levels),
            'total_invested': grid.total_invested,
            'unrealized_pnl': grid.unrealized_pnl,
            'created_at': grid.created_at.isoformat(),
            'last_rebalance': grid.last_rebalance.isoformat(),
            'active_levels': sum(1 for level in grid.levels if level.is_active)
        }

    def get_all_grids_status(self) -> Dict[str, Any]:
        """Get status of all active grids."""
        return {
            'total_grids': len(self.active_grids),
            'total_invested': sum(grid.total_invested for grid in self.active_grids.values()),
            'total_unrealized_pnl': sum(grid.unrealized_pnl for grid in self.active_grids.values()),
            'grids': {symbol: self.get_grid_status(symbol) for symbol in self.active_grids}
        }

    def close_grid(self, symbol: str) -> List[TradingDecision]:
        """Close a specific grid position."""
        grid = self.active_grids.get(symbol)
        if not grid:
            return []

        decisions = []

        # Close all positions
        for level in grid.levels:
            if level.buy_quantity > 0:
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="SELL",
                    quantity=level.buy_quantity,
                    price=None,
                    reason=f"Grid closure - selling remaining position",
                    confidence=0.95,
                    risk_score=0.1,
                    expected_profit=None
                ))

        # Remove grid
        del self.active_grids[symbol]

        logger.info(f"Closed grid for {symbol}")
        return decisions

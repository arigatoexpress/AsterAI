import math
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    # Import lazily to allow offline/backtest-only use
    from mcp_trader.execution.aster_client import AsterClient
except Exception:  # pragma: no cover - optional in offline unit tests
    AsterClient = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class CostModelConfig:
    max_slippage_pct: float = 0.02
    default_taker_fee: float = 0.0005   # 5 bps
    default_maker_fee: float = 0.0002   # 2 bps
    use_orderbook_walk: bool = True


class CostModel:
    """Abstract cost model interface."""

    def estimate_execution_price(self, symbol: str, base_price: float, quantity: float,
                                 side: str, timestamp: Optional[datetime] = None,
                                 orderbook_snapshot: Optional[Dict[str, Any]] = None) -> float:
        raise NotImplementedError

    def estimate_fee(self, symbol: str, notional: float, is_maker: bool,
                      timestamp: Optional[datetime] = None) -> float:
        raise NotImplementedError


class OnChainCostModel(CostModel):
    """On-chain single source of truth cost model.

    - Slippage: walks the current (or provided) order book to estimate impact.
    - Fees: queries user commission rate when available, otherwise uses sane defaults.
    """

    def __init__(self, client: Optional[AsterClient] = None, config: Optional[CostModelConfig] = None):
        self.client = client
        self.config = config or CostModelConfig()
        self._cached_commission: Dict[str, Dict[str, float]] = {}

    async def _get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            return await self.client.get_order_book(symbol, limit=200)
        except Exception as e:
            logger.debug(f"Orderbook fetch failed, fallback to model: {e}")
            return None

    async def _get_commission(self, symbol: str) -> Dict[str, float]:
        if symbol in self._cached_commission:
            return self._cached_commission[symbol]
        if not self.client:
            return {
                'maker': self.config.default_maker_fee,
                'taker': self.config.default_taker_fee,
            }
        try:
            data = await self.client.rest_client.get_user_commission_rate(symbol)
            maker = float(data.get('makerCommissionRate', self.config.default_maker_fee))
            taker = float(data.get('takerCommissionRate', self.config.default_taker_fee))
            self._cached_commission[symbol] = {'maker': maker, 'taker': taker}
            return self._cached_commission[symbol]
        except Exception as e:
            logger.debug(f"Commission fetch failed, using defaults: {e}")
            return {
                'maker': self.config.default_maker_fee,
                'taker': self.config.default_taker_fee,
            }

    @staticmethod
    def _walk_orderbook(orderbook: Dict[str, Any], side: str, quantity: float,
                        max_slippage_pct: float, base_price: float) -> float:
        """Walk the orderbook to estimate average execution price for a market order."""
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return base_price

        levels: List[List[str]] = orderbook['asks'] if side.lower() == 'buy' else orderbook['bids']
        # For sells, bids are descending; for buys, asks are ascending. Assume API provides correct order.

        remaining = quantity
        notional = 0.0
        accumulated_qty = 0.0

        for price_str, size_str in levels:
            level_price = float(price_str)
            level_size = float(size_str)

            take_qty = min(remaining, level_size)
            notional += take_qty * level_price
            accumulated_qty += take_qty
            remaining -= take_qty

            # Stop if slippage cap reached
            slippage = abs(level_price - base_price) / base_price
            if slippage > max_slippage_pct:
                break

            if remaining <= 0:
                break

        if accumulated_qty <= 0:
            return base_price

        # If we could not fill completely, assume rest at worst capped price
        if remaining > 0:
            capped_price = base_price * (1 + max_slippage_pct if side.lower() == 'buy' else 1 - max_slippage_pct)
            notional += remaining * capped_price
            accumulated_qty += remaining

        return notional / accumulated_qty

    async def estimate_execution_price(self, symbol: str, base_price: float, quantity: float,
                                       side: str, timestamp: Optional[datetime] = None,
                                       orderbook_snapshot: Optional[Dict[str, Any]] = None) -> float:
        if quantity <= 0:
            return base_price

        ob = orderbook_snapshot or await self._get_orderbook(symbol)
        if self.config.use_orderbook_walk and ob:
            return self._walk_orderbook(ob, side, quantity, self.config.max_slippage_pct, base_price)

        # Fallback: adaptive slippage ~ sqrt impact capped
        # Assume implicit volume from price level count if orderbook missing
        impact = min(self.config.max_slippage_pct, 0.001 * math.sqrt(max(quantity, 1e-9)))
        if side.lower() == 'buy':
            return base_price * (1 + impact)
        return base_price * (1 - impact)

    async def estimate_fee(self, symbol: str, notional: float, is_maker: bool,
                           timestamp: Optional[datetime] = None) -> float:
        commission = await self._get_commission(symbol)
        rate = commission['maker'] if is_maker else commission['taker']
        return notional * rate



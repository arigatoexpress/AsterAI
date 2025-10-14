from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .aster_client import AsterClient

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades on the Aster DEX with proper error handling and logging."""

    def __init__(self, client: AsterClient | None = None):
        """Initialize trade executor with Aster client."""
        self.client = client or AsterClient()

    def market_buy(self, symbol: str, quantity: float, leverage: int | None = None) -> Dict[str, Any]:
        """Execute a market buy order.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
            leverage: Optional leverage for perpetual futures

        Returns:
            Order response from API

        Raises:
            Exception: If order execution fails
        """
        try:
            from .aster_client import OrderRequest, OrderType, Side

            logger.info(f"Executing market buy: {symbol} qty={quantity} leverage={leverage}")
            order_request = OrderRequest(
                symbol=symbol,
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                leverage=leverage
            )
            response = self.client.place_order(order_request)
            logger.info(f"Market buy executed successfully: {response}")
            return response.__dict__ if hasattr(response, '__dict__') else response
        except Exception as e:
            logger.error(f"Market buy failed for {symbol}: {e}")
            raise

    def market_sell(self, symbol: str, quantity: float, leverage: int | None = None) -> Dict[str, Any]:
        """Execute a market sell order.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            quantity: Order quantity
            leverage: Optional leverage for perpetual futures

        Returns:
            Order response from API

        Raises:
            Exception: If order execution fails
        """
        try:
            from .aster_client import OrderRequest, OrderType, Side

            logger.info(f"Executing market sell: {symbol} qty={quantity} leverage={leverage}")
            order_request = OrderRequest(
                symbol=symbol,
                side=Side.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                leverage=leverage
            )
            response = self.client.place_order(order_request)
            logger.info(f"Market sell executed successfully: {response}")
            return response.__dict__ if hasattr(response, '__dict__') else response
        except Exception as e:
            logger.error(f"Market sell failed for {symbol}: {e}")
            raise

"""
Aster DEX API client for live trading.
Supports REST API and WebSocket connections.
Enhanced with full API capabilities from Aster DEX documentation.
"""

import asyncio
import aiohttp
import websockets
import json
import hmac
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    """Position side enumeration."""
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


class TimeInForce(Enum):
    """Time in force enumeration."""
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"
    HIDDEN = "HIDDEN"


class WorkingType(Enum):
    """Working type for stop orders."""
    MARK_PRICE = "MARK_PRICE"
    CONTRACT_PRICE = "CONTRACT_PRICE"


class MarginType(Enum):
    """Margin type enumeration."""
    ISOLATED = "ISOLATED"
    CROSSED = "CROSSED"


class KlineInterval(Enum):
    """Kline/Candlestick chart intervals."""
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


def handle_api_error(func):
    """Decorator to handle API errors consistently."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error in {func.__name__}: {e}")
            raise
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error in {func.__name__}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper
from datetime import datetime


@dataclass
class AsterConfig:
    """Configuration for Aster API client."""
    api_key: str
    secret_key: str
    base_url: str = "https://fapi.asterdex.com"  # Aster Futures API base URL
    ws_url: str = "wss://fstream.asterdex.com"  # Aster WebSocket streams URL
    timeout: int = 30
    max_retries: int = 3
    recv_window: int = 5000  # Default recvWindow for signed requests


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reduce_only: bool = False
    post_only: bool = False


@dataclass
class OrderResponse:
    """Order response structure."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str  # 'new', 'filled', 'partially_filled', 'canceled', 'rejected'
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    timestamp: datetime
    client_order_id: Optional[str] = None


@dataclass
class Position:
    """Position information."""
    symbol: str
    size: float
    side: str  # 'long' or 'short'
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin: float
    leverage: float


@dataclass
class AccountInfo:
    """Account information."""
    total_balance: float
    available_balance: float
    used_margin: float
    free_margin: float
    equity: float
    positions: List[Position]
    open_orders: List[OrderResponse]


@dataclass
class TickerPrice:
    """24hr ticker price change statistics."""
    symbol: str
    price_change: float
    price_change_percent: float
    weighted_avg_price: float
    prev_close_price: float
    last_price: float
    last_qty: float
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    quote_volume: float
    open_time: datetime
    close_time: datetime
    first_id: int
    last_id: int
    count: int


@dataclass
class FundingRate:
    """Funding rate information."""
    symbol: str
    funding_rate: float
    funding_time: datetime


@dataclass
class MarkPrice:
    """Mark price information."""
    symbol: str
    mark_price: float
    index_price: float
    estimated_settle_price: float
    last_funding_rate: float
    next_funding_time: datetime
    interest_rate: float
    time: datetime


@dataclass
class LeverageBracket:
    """Leverage bracket information."""
    bracket: int
    initial_leverage: int
    notional_cap: float
    notional_floor: float
    maint_margin_ratio: float
    cum: float


@dataclass
class SymbolInfo:
    """Symbol exchange information."""
    symbol: str
    pair: str
    contract_type: str
    delivery_date: datetime
    onboard_date: datetime
    status: str
    maint_margin_percent: float
    required_margin_percent: float
    base_asset: str
    quote_asset: str
    margin_asset: str
    price_precision: int
    quantity_precision: int
    base_asset_precision: int
    quote_precision: int
    underlying_type: str
    underlying_sub_type: List[str]
    settle_plan: int
    trigger_protect: float
    filters: List[Dict[str, Any]]
    order_type: List[str]
    time_in_force: List[str]
    liquidation_fee: float
    market_take_bound: float


@dataclass
class AccountBalance:
    """Account balance information."""
    account_alias: str
    asset: str
    balance: float
    cross_wallet_balance: float
    cross_unpnl: float
    available_balance: float
    max_withdraw_amount: float
    margin_available: bool
    update_time: datetime


@dataclass
class AccountPosition:
    """Account position information."""
    symbol: str
    initial_margin: float
    maint_margin: float
    unrealized_profit: float
    position_initial_margin: float
    open_order_initial_margin: float
    leverage: float
    isolated: bool
    entry_price: float
    max_notional_value: float
    position_side: str
    position_amt: float
    update_time: datetime


@dataclass
class IncomeHistory:
    """Income history record."""
    symbol: str
    income_type: str
    income: float
    asset: str
    info: str
    time: datetime
    tran_id: str
    trade_id: str


@dataclass
class UserTrade:
    """User trade information."""
    buyer: bool
    commission: float
    commission_asset: str
    id: int
    maker: bool
    order_id: int
    price: float
    qty: float
    quote_qty: float
    realized_pnl: float
    side: str
    position_side: str
    symbol: str
    time: datetime


@dataclass
class ForceOrder:
    """Force order (liquidation) information."""
    order_id: int
    symbol: str
    status: str
    client_order_id: str
    price: float
    avg_price: float
    orig_qty: float
    executed_qty: float
    cum_quote: float
    time_in_force: str
    type: str
    reduce_only: bool
    close_position: bool
    side: str
    position_side: str
    stop_price: float
    working_type: str
    orig_type: str
    time: datetime
    update_time: datetime


@dataclass
class ADLQuantile:
    """ADL quantile information."""
    symbol: str
    adl_quantile: Dict[str, int]  # LONG, SHORT, BOTH, HEDGE -> quantile value


class AsterRESTClient:
    """REST API client for Aster DEX."""
    
    def __init__(self, config: AsterConfig):
        self.config = config
        self.session = None
        self.time_offset = 0  # Offset between local and server time
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))
        # Sync time with server on initialization
        await self._sync_server_time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: Dict, timestamp: str) -> str:
        """Generate HMAC signature for Aster DEX authentication."""
        # Create the query string from all parameters including timestamp and recvWindow
        all_params = params.copy()
        all_params['timestamp'] = timestamp
        all_params['recvWindow'] = str(self.config.recv_window)

        # Sort parameters by key and create query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted(all_params.items())])

        # Generate signature using HMAC SHA256
        signature = hmac.new(
            self.config.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature
    
    def _get_headers(self, params: Dict = None, signed: bool = False) -> Dict[str, str]:
        """Get headers with optional authentication for Aster DEX."""
        headers = {'Content-Type': 'application/json'}

        if signed:
            # Add API key to headers
            headers['X-MBX-APIKEY'] = self.config.api_key

        return headers
    
    async def _sync_server_time(self):
        """Synchronize local time with server time."""
        try:
            server_time = await self.get_server_time()
            if server_time > 0:
                local_time = int(time.time() * 1000)
                self.time_offset = server_time - local_time
                logger.info(f"Time synchronized: offset = {self.time_offset}ms")
            else:
                logger.warning("Failed to sync server time, using local time")
                self.time_offset = 0
        except Exception as e:
            logger.warning(f"Time sync failed: {e}, using local time")
            self.time_offset = 0

    def _get_synced_timestamp(self) -> str:
        """Get synchronized timestamp for API requests."""
        return str(int(time.time() * 1000) + self.time_offset)

    def _add_signature(self, params: Dict = None) -> Dict:
        """Add timestamp, recvWindow, and signature to parameters for signed requests."""
        signed_params = params.copy() if params else {}

        # Add timestamp and recvWindow
        timestamp = self._get_synced_timestamp()

        # Generate and add signature
        signature = self._generate_signature(signed_params, timestamp)
        signed_params['timestamp'] = timestamp
        signed_params['recvWindow'] = str(self.config.recv_window)
        signed_params['signature'] = signature

        return signed_params

    async def _make_request(self, method: str, endpoint: str, params: Dict = None,
                           data: Dict = None, signed: bool = False) -> Dict:
        """Make API request with optional authentication."""
        # Prepare parameters
        request_params = params.copy() if params else {}
        headers = self._get_headers(request_params, signed)

        # Add signature for signed endpoints
        if signed:
            request_params = self._add_signature(request_params)

        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                if method == 'GET':
                    async with self.session.get(url, headers=headers, params=request_params) as response:
                        result = await response.json()
                elif method == 'POST':
                    async with self.session.post(url, headers=headers, params=request_params, json=data) as response:
                        result = await response.json()
                elif method == 'DELETE':
                    async with self.session.delete(url, headers=headers, params=request_params) as response:
                        result = await response.json()
                
                if response.status == 200:
                    return result
                else:
                    # Handle specific API error codes
                    error_msg = result.get('msg', 'Unknown error') if isinstance(result, dict) else str(result)
                    error_code = result.get('code', response.status) if isinstance(result, dict) else response.status

                    logger.error(f"API request failed: HTTP {response.status}, Code {error_code} - {error_msg}")

                    # Don't retry on client errors (4xx) except for rate limits
                    if 400 <= response.status < 500 and response.status != 429:
                        raise Exception(f"API Client Error {response.status}: {error_msg}")

                    if attempt == self.config.max_retries - 1:
                        raise Exception(f"API request failed after {self.config.max_retries} attempts: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("All request attempts failed"        )

    @handle_api_error
    async def ping(self) -> Dict:
        """Test connectivity with a simple ping."""
        return await self._make_request('GET', '/fapi/v1/ping')

    @handle_api_error
    async def get_server_time(self) -> int:
        """Get server time for timestamp synchronization."""
        result = await self._make_request('GET', '/fapi/v1/time')
        return result.get('serverTime', 0)

    @handle_api_error
    async def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics."""
        params = {'symbol': symbol}
        result = await self._make_request('GET', '/fapi/v1/ticker/24hr', params=params)

        # Handle different response formats from Aster DEX
        if isinstance(result, dict):
            # Clean up any problematic fields
            cleaned_result = {}
            for key, value in result.items():
                try:
                    # Convert string numbers to float where appropriate
                    if key in ['lastPrice', 'bidPrice', 'askPrice', 'highPrice', 'lowPrice', 'openPrice', 'prevClosePrice']:
                        cleaned_result[key] = float(value) if value else 0.0
                    elif key in ['volume', 'quoteVolume', 'priceChangePercent']:
                        cleaned_result[key] = float(value) if value else 0.0
                    else:
                        cleaned_result[key] = value
                except (ValueError, TypeError):
                    cleaned_result[key] = value

            return cleaned_result

        return result

    @handle_api_error
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth."""
        params = {'symbol': symbol, 'limit': str(limit)}
        return await self._make_request('GET', '/fapi/v1/depth', params=params)

    @handle_api_error
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        result = await self._make_request('GET', '/account')
        
        return AccountInfo(
            total_balance=result.get('total_balance', 0.0),
            available_balance=result.get('available_balance', 0.0),
            used_margin=result.get('used_margin', 0.0),
            free_margin=result.get('free_margin', 0.0),
            equity=result.get('equity', 0.0),
            positions=[self._parse_position(pos) for pos in result.get('positions', [])],
            open_orders=[self._parse_order(order) for order in result.get('open_orders', [])]
        )
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        result = await self._make_request('GET', '/positions')
        return [self._parse_position(pos) for pos in result.get('positions', [])]
    
    async def get_orders(self, symbol: str = None, status: str = None) -> List[OrderResponse]:
        """Get orders."""
        params = {}
        if symbol:
            params['symbol'] = symbol
        if status:
            params['status'] = status
        
        result = await self._make_request('GET', '/orders', params=params)
        return [self._parse_order(order) for order in result.get('orders', [])]
    
    @handle_api_error
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a new order."""
        data = {
            'symbol': order_request.symbol,
            'side': order_request.side,
            'type': order_request.order_type,
            'quantity': order_request.quantity,
            'time_in_force': order_request.time_in_force,
            'reduce_only': order_request.reduce_only,
            'post_only': order_request.post_only
        }
        
        if order_request.price:
            data['price'] = order_request.price
        if order_request.stop_price:
            data['stop_price'] = order_request.stop_price
        
        result = await self._make_request('POST', '/orders', data=data)
        return self._parse_order(result)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self._make_request('DELETE', f'/orders/{order_id}')
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all orders."""
        try:
            params = {'symbol': symbol} if symbol else {}
            await self._make_request('DELETE', '/orders', params=params)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for a symbol."""
        result = await self._make_request('GET', f'/market/{symbol}')
        return result
    
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Get kline/candlestick data."""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        result = await self._make_request('GET', '/klines', params=params)
        
        # Convert to DataFrame
        df = pd.DataFrame(result.get('klines', []))
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
        
        return df
    
    def _parse_position(self, pos_data: Dict) -> Position:
        """Parse position data."""
        return Position(
            symbol=pos_data.get('symbol', ''),
            size=float(pos_data.get('size', 0.0)),
            side=pos_data.get('side', 'long'),
            entry_price=float(pos_data.get('entry_price', 0.0)),
            mark_price=float(pos_data.get('mark_price', 0.0)),
            unrealized_pnl=float(pos_data.get('unrealized_pnl', 0.0)),
            realized_pnl=float(pos_data.get('realized_pnl', 0.0)),
            margin=float(pos_data.get('margin', 0.0)),
            leverage=float(pos_data.get('leverage', 1.0))
        )
    
    def _parse_order(self, order_data: Dict) -> OrderResponse:
        """Parse order data."""
        return OrderResponse(
            order_id=order_data.get('order_id', ''),
            symbol=order_data.get('symbol', ''),
            side=order_data.get('side', ''),
            order_type=order_data.get('type', ''),
            quantity=float(order_data.get('quantity', 0.0)),
            price=float(order_data.get('price', 0.0)) if order_data.get('price') else None,
            status=order_data.get('status', ''),
            filled_quantity=float(order_data.get('filled_quantity', 0.0)),
            remaining_quantity=float(order_data.get('remaining_quantity', 0.0)),
            average_price=float(order_data.get('average_price', 0.0)) if order_data.get('average_price') else None,
            timestamp=datetime.fromtimestamp(order_data.get('timestamp', 0) / 1000),
            client_order_id=order_data.get('client_order_id')
        )

    # ===== MARKET DATA ENDPOINTS =====

    @handle_api_error
    async def test_connectivity(self) -> Dict:
        """Test connectivity to the Rest API."""
        async with self.session.get(f"{self.config.base_url}/fapi/v1/ping") as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def check_server_time(self) -> Dict:
        """Check server time."""
        async with self.session.get(f"{self.config.base_url}/fapi/v1/time") as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_exchange_info(self) -> Dict:
        """Get current exchange trading rules and symbol information."""
        async with self.session.get(f"{self.config.base_url}/fapi/v1/exchangeInfo") as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_order_book(self, symbol: str, limit: int = 500) -> Dict:
        """Get order book for a symbol."""
        params = {"symbol": symbol, "limit": limit}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/depth", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get recent trades for a symbol."""
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/trades", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_historical_trades(self, symbol: str, limit: int = 500, from_id: int = None) -> List[Dict]:
        """Get older historical trades."""
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        if from_id:
            params["fromId"] = from_id

        # This endpoint requires API key
        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/historicalTrades", params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_agg_trades(self, symbol: str, from_id: int = None, start_time: int = None,
                           end_time: int = None, limit: int = 500) -> List[Dict]:
        """Get compressed/aggregate trades."""
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        if from_id:
            params["fromId"] = from_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        async with self.session.get(f"{self.config.base_url}/fapi/v1/aggTrades", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_klines(self, symbol: str, interval: KlineInterval, start_time: int = None,
                        end_time: int = None, limit: int = 500) -> List[List]:
        """Get kline/candlestick data."""
        params = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": min(limit, 1500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        async with self.session.get(f"{self.config.base_url}/fapi/v1/klines", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_index_price_klines(self, pair: str, interval: KlineInterval, start_time: int = None,
                                   end_time: int = None, limit: int = 500) -> List[List]:
        """Get index price kline data."""
        params = {
            "pair": pair,
            "interval": interval.value,
            "limit": min(limit, 1500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        async with self.session.get(f"{self.config.base_url}/fapi/v1/indexPriceKlines", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_mark_price_klines(self, symbol: str, interval: KlineInterval, start_time: int = None,
                                  end_time: int = None, limit: int = 500) -> List[List]:
        """Get mark price kline data."""
        params = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": min(limit, 1500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        async with self.session.get(f"{self.config.base_url}/fapi/v1/markPriceKlines", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_mark_price(self, symbol: str = None) -> Union[MarkPrice, List[MarkPrice]]:
        """Get mark price and funding rate."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with self.session.get(f"{self.config.base_url}/fapi/v1/premiumIndex", params=params) as response:
            response.raise_for_status()
            data = await response.json()

            if symbol:
                # Single symbol response
                return MarkPrice(
                    symbol=data["symbol"],
                    mark_price=float(data["markPrice"]),
                    index_price=float(data["indexPrice"]),
                    estimated_settle_price=float(data["estimatedSettlePrice"]),
                    last_funding_rate=float(data["lastFundingRate"]),
                    next_funding_time=datetime.fromtimestamp(data["nextFundingTime"] / 1000),
                    interest_rate=float(data["interestRate"]),
                    time=datetime.fromtimestamp(data["time"] / 1000)
                )
            else:
                # Multiple symbols response
                return [
                    MarkPrice(
                        symbol=item["symbol"],
                        mark_price=float(item["markPrice"]),
                        index_price=float(item["indexPrice"]),
                        estimated_settle_price=float(item["estimatedSettlePrice"]),
                        last_funding_rate=float(item["lastFundingRate"]),
                        next_funding_time=datetime.fromtimestamp(item["nextFundingTime"] / 1000),
                        interest_rate=float(item["interestRate"]),
                        time=datetime.fromtimestamp(item["time"] / 1000)
                    ) for item in data
                ]

    @handle_api_error
    async def get_funding_rate_history(self, symbol: str = None, start_time: int = None,
                                     end_time: int = None, limit: int = 100) -> List[FundingRate]:
        """Get funding rate history."""
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        async with self.session.get(f"{self.config.base_url}/fapi/v1/fundingRate", params=params) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                FundingRate(
                    symbol=item["symbol"],
                    funding_rate=float(item["fundingRate"]),
                    funding_time=datetime.fromtimestamp(item["fundingTime"] / 1000)
                ) for item in data
            ]

    @handle_api_error
    async def get_funding_info(self, symbol: str = None) -> List[Dict]:
        """Get funding rate configuration."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with self.session.get(f"{self.config.base_url}/fapi/v1/fundingInfo", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_24hr_ticker(self, symbol: str = None) -> Union[TickerPrice, List[TickerPrice]]:
        """Get 24hr ticker price change statistics."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with self.session.get(f"{self.config.base_url}/fapi/v1/ticker/24hr", params=params) as response:
            response.raise_for_status()
            data = await response.json()

            if symbol:
                # Single symbol response
                return TickerPrice(
                    symbol=data["symbol"],
                    price_change=float(data["priceChange"]),
                    price_change_percent=float(data["priceChangePercent"]),
                    weighted_avg_price=float(data["weightedAvgPrice"]),
                    prev_close_price=float(data["prevClosePrice"]),
                    last_price=float(data["lastPrice"]),
                    last_qty=float(data["lastQty"]),
                    bid_price=float(data.get("bidPrice", 0)),
                    bid_qty=float(data.get("bidQty", 0)),
                    ask_price=float(data.get("askPrice", 0)),
                    ask_qty=float(data.get("askQty", 0)),
                    open_price=float(data["openPrice"]),
                    high_price=float(data["highPrice"]),
                    low_price=float(data["lowPrice"]),
                    volume=float(data["volume"]),
                    quote_volume=float(data["quoteVolume"]),
                    open_time=datetime.fromtimestamp(data["openTime"] / 1000),
                    close_time=datetime.fromtimestamp(data["closeTime"] / 1000),
                    first_id=int(data["firstId"]),
                    last_id=int(data["lastId"]),
                    count=int(data["count"])
                )
            else:
                # Multiple symbols response
                return [
                    TickerPrice(
                        symbol=item["symbol"],
                        price_change=float(item["priceChange"]),
                        price_change_percent=float(item["priceChangePercent"]),
                        weighted_avg_price=float(item["weightedAvgPrice"]),
                        prev_close_price=float(item["prevClosePrice"]),
                        last_price=float(item["lastPrice"]),
                        last_qty=float(item["lastQty"]),
                        bid_price=float(item.get("bidPrice", 0)),
                        bid_qty=float(item.get("bidQty", 0)),
                        ask_price=float(item.get("askPrice", 0)),
                        ask_qty=float(item.get("askQty", 0)),
                        open_price=float(item["openPrice"]),
                        high_price=float(item["highPrice"]),
                        low_price=float(item["lowPrice"]),
                        volume=float(item["volume"]),
                        quote_volume=float(item["quoteVolume"]),
                        open_time=datetime.fromtimestamp(item["openTime"] / 1000),
                        close_time=datetime.fromtimestamp(item["closeTime"] / 1000),
                        first_id=int(item["firstId"]),
                        last_id=int(item["lastId"]),
                        count=int(item["count"])
                    ) for item in data
                ]

    @handle_api_error
    async def get_symbol_price_ticker(self, symbol: str = None) -> Union[Dict, List[Dict]]:
        """Get latest price for symbol(s)."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with self.session.get(f"{self.config.base_url}/fapi/v1/ticker/price", params=params) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_symbol_orderbook_ticker(self, symbol: str = None) -> Union[Dict, List[Dict]]:
        """Get best price/qty on the order book."""
        params = {}
        if symbol:
            params["symbol"] = symbol

        async with self.session.get(f"{self.config.base_url}/fapi/v1/ticker/bookTicker", params=params) as response:
            response.raise_for_status()
            return await response.json()

    # ===== ACCOUNT/TRADES ENDPOINTS =====

    @handle_api_error
    async def change_position_mode(self, dual_side_position: bool) -> Dict:
        """Change position mode (Hedge/One-way)."""
        timestamp = int(time.time() * 1000)
        params = {
            "dualSidePosition": "true" if dual_side_position else "false",
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/positionSide/dual", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/positionSide/dual",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_position_mode(self) -> Dict:
        """Get current position mode."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("GET", "/fapi/v1/positionSide/dual", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/positionSide/dual",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def change_multi_assets_mode(self, multi_assets_margin: bool) -> Dict:
        """Change Multi-Assets mode."""
        timestamp = int(time.time() * 1000)
        params = {
            "multiAssetsMargin": "true" if multi_assets_margin else "false",
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/multiAssetsMargin", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/multiAssetsMargin",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_multi_assets_mode(self) -> Dict:
        """Get current Multi-Assets mode."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("GET", "/fapi/v1/multiAssetsMargin", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/multiAssetsMargin",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, position_side: PositionSide = PositionSide.BOTH,
                         price: float = None, time_in_force: TimeInForce = None,
                         reduce_only: bool = None, stop_price: float = None,
                         close_position: bool = None, activation_price: float = None,
                         callback_rate: float = None, working_type: WorkingType = None,
                         new_client_order_id: str = None, price_protect: bool = None) -> Dict:
        """Place a new order."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": quantity,
            "positionSide": position_side.value,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        # Add optional parameters
        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force.value
        if reduce_only is not None:
            params["reduceOnly"] = "true" if reduce_only else "false"
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if close_position is not None:
            params["closePosition"] = "true" if close_position else "false"
        if activation_price is not None:
            params["activationPrice"] = activation_price
        if callback_rate is not None:
            params["callbackRate"] = callback_rate
        if working_type is not None:
            params["workingType"] = working_type.value
        if new_client_order_id is not None:
            params["newClientOrderId"] = new_client_order_id
        if price_protect is not None:
            params["priceProtect"] = "TRUE" if price_protect else "FALSE"

        signature = self._generate_signature("POST", "/fapi/v1/order", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/order",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def place_multiple_orders(self, orders: List[Dict]) -> List[Dict]:
        """Place multiple orders."""
        timestamp = int(time.time() * 1000)
        params = {
            "batchOrders": json.dumps(orders),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/batchOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/batchOrders",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def transfer_futures_spot(self, amount: float, asset: str, client_tran_id: str, kind_type: str) -> Dict:
        """Transfer between futures and spot."""
        timestamp = int(time.time() * 1000)
        params = {
            "amount": amount,
            "asset": asset,
            "clientTranId": client_tran_id,
            "kindType": kind_type,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/asset/wallet/transfer", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/asset/wallet/transfer",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def query_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """Query order status."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id

        signature = self._generate_signature("GET", "/fapi/v1/order", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/order",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """Cancel an order."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id

        signature = self._generate_signature("DELETE", "/fapi/v1/order", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.request("DELETE", f"{self.config.base_url}/fapi/v1/order",
                                      data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def cancel_all_orders(self, symbol: str) -> Dict:
        """Cancel all open orders for a symbol."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("DELETE", "/fapi/v1/allOpenOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.request("DELETE", f"{self.config.base_url}/fapi/v1/allOpenOrders",
                                      data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def cancel_multiple_orders(self, symbol: str, order_id_list: List[int] = None,
                                   orig_client_order_id_list: List[str] = None) -> List[Dict]:
        """Cancel multiple orders."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if order_id_list:
            params["orderIdList"] = json.dumps(order_id_list)
        if orig_client_order_id_list:
            params["origClientOrderIdList"] = json.dumps(orig_client_order_id_list)

        signature = self._generate_signature("DELETE", "/fapi/v1/batchOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.request("DELETE", f"{self.config.base_url}/fapi/v1/batchOrders",
                                      data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def auto_cancel_orders(self, symbol: str, countdown_time: int) -> Dict:
        """Auto-cancel all open orders after countdown."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "countdownTime": countdown_time,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/countdownCancelAll", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/countdownCancelAll",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def query_open_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """Query current open order."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id

        signature = self._generate_signature("GET", "/fapi/v1/openOrder", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/openOrder",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def query_all_open_orders(self, symbol: str = None) -> List[Dict]:
        """Query all open orders."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol

        signature = self._generate_signature("GET", "/fapi/v1/openOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/openOrders",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def query_all_orders(self, symbol: str, order_id: int = None, start_time: int = None,
                             end_time: int = None, limit: int = 500) -> List[Dict]:
        """Query all orders (active, canceled, filled)."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        signature = self._generate_signature("GET", "/fapi/v1/allOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/allOrders",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_account_balance_v2(self) -> List[AccountBalance]:
        """Get account balance V2."""
        result = await self._make_request('GET', '/fapi/v2/balance', signed=True)

        return [
            AccountBalance(
                account_alias=item["accountAlias"],
                asset=item["asset"],
                balance=float(item["balance"]),
                cross_wallet_balance=float(item["crossWalletBalance"]),
                cross_unpnl=float(item["crossUnPnl"]),
                available_balance=float(item["availableBalance"]),
                max_withdraw_amount=float(item["maxWithdrawAmount"]),
                margin_available=bool(item["marginAvailable"]),
                update_time=datetime.fromtimestamp(item["updateTime"] / 1000)
            ) for item in result
        ]

    @handle_api_error
    async def get_account_info_v4(self) -> Dict:
        """Get account information V4."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("GET", "/fapi/v4/account", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v4/account",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change initial leverage."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/leverage", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/leverage",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def change_margin_type(self, symbol: str, margin_type: MarginType) -> Dict:
        """Change margin type."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "marginType": margin_type.value,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/marginType", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/marginType",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def modify_position_margin(self, symbol: str, amount: float, type: int,
                                   position_side: PositionSide = PositionSide.BOTH) -> Dict:
        """Modify isolated position margin."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "amount": amount,
            "type": type,
            "positionSide": position_side.value,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("POST", "/fapi/v1/positionMargin", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.post(f"{self.config.base_url}/fapi/v1/positionMargin",
                                   data=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_position_margin_history(self, symbol: str, type: int = None, start_time: int = None,
                                        end_time: int = None, limit: int = 500) -> List[Dict]:
        """Get position margin change history."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "limit": min(limit, 500),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if type is not None:
            params["type"] = type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        signature = self._generate_signature("GET", "/fapi/v1/positionMargin/history", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/positionMargin/history",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_position_info_v2(self, symbol: str = None) -> List[AccountPosition]:
        """Get position information V2."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol

        signature = self._generate_signature("GET", "/fapi/v2/positionRisk", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v2/positionRisk",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                AccountPosition(
                    symbol=item["symbol"],
                    initial_margin=float(item["initialMargin"]),
                    maint_margin=float(item["maintMargin"]),
                    unrealized_profit=float(item["unrealizedProfit"]),
                    position_initial_margin=float(item["positionInitialMargin"]),
                    open_order_initial_margin=float(item["openOrderInitialMargin"]),
                    leverage=float(item["leverage"]),
                    isolated=bool(item["isolated"]),
                    entry_price=float(item["entryPrice"]),
                    max_notional_value=float(item["maxNotionalValue"]),
                    position_side=item["positionSide"],
                    position_amt=float(item["positionAmt"]),
                    update_time=datetime.fromtimestamp(item["updateTime"] / 1000)
                ) for item in data
            ]

    @handle_api_error
    async def get_account_trades(self, symbol: str, start_time: int = None, end_time: int = None,
                               from_id: int = None, limit: int = 500) -> List[UserTrade]:
        """Get account trade list."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "limit": min(limit, 1000),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if from_id:
            params["fromId"] = from_id

        signature = self._generate_signature("GET", "/fapi/v1/userTrades", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/userTrades",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                UserTrade(
                    buyer=bool(item["buyer"]),
                    commission=float(item["commission"]),
                    commission_asset=item["commissionAsset"],
                    id=int(item["id"]),
                    maker=bool(item["maker"]),
                    order_id=int(item["orderId"]),
                    price=float(item["price"]),
                    qty=float(item["qty"]),
                    quote_qty=float(item["quoteQty"]),
                    realized_pnl=float(item["realizedPnl"]),
                    side=item["side"],
                    position_side=item["positionSide"],
                    symbol=item["symbol"],
                    time=datetime.fromtimestamp(item["time"] / 1000)
                ) for item in data
            ]

    @handle_api_error
    async def get_income_history(self, symbol: str = None, income_type: str = None,
                               start_time: int = None, end_time: int = None, limit: int = 100) -> List[IncomeHistory]:
        """Get income history."""
        timestamp = int(time.time() * 1000)
        params = {
            "limit": min(limit, 1000),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol
        if income_type:
            params["incomeType"] = income_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        signature = self._generate_signature("GET", "/fapi/v1/income", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/income",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                IncomeHistory(
                    symbol=item["symbol"],
                    income_type=item["incomeType"],
                    income=float(item["income"]),
                    asset=item["asset"],
                    info=item["info"],
                    time=datetime.fromtimestamp(item["time"] / 1000),
                    tran_id=item["tranId"],
                    trade_id=item["tradeId"]
                ) for item in data
            ]

    @handle_api_error
    async def get_leverage_brackets(self, symbol: str = None) -> Dict:
        """Get leverage brackets."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol

        signature = self._generate_signature("GET", "/fapi/v1/leverageBracket", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/leverageBracket",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    @handle_api_error
    async def get_adl_quantile(self, symbol: str = None) -> List[ADLQuantile]:
        """Get ADL quantile information."""
        timestamp = int(time.time() * 1000)
        params = {
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol

        signature = self._generate_signature("GET", "/fapi/v1/adlQuantile", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/adlQuantile",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                ADLQuantile(
                    symbol=item["symbol"],
                    adl_quantile=item["adlQuantile"]
                ) for item in data
            ]

    @handle_api_error
    async def get_force_orders(self, symbol: str = None, auto_close_type: str = None,
                             start_time: int = None, end_time: int = None, limit: int = 50) -> List[ForceOrder]:
        """Get user's force orders (liquidations)."""
        timestamp = int(time.time() * 1000)
        params = {
            "limit": min(limit, 100),
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        if symbol:
            params["symbol"] = symbol
        if auto_close_type:
            params["autoCloseType"] = auto_close_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        signature = self._generate_signature("GET", "/fapi/v1/forceOrders", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/forceOrders",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            return [
                ForceOrder(
                    order_id=int(item["orderId"]),
                    symbol=item["symbol"],
                    status=item["status"],
                    client_order_id=item["clientOrderId"],
                    price=float(item["price"]),
                    avg_price=float(item["avgPrice"]),
                    orig_qty=float(item["origQty"]),
                    executed_qty=float(item["executedQty"]),
                    cum_quote=float(item["cumQuote"]),
                    time_in_force=item["timeInForce"],
                    type=item["type"],
                    reduce_only=bool(item["reduceOnly"]),
                    close_position=bool(item["closePosition"]),
                    side=item["side"],
                    position_side=item["positionSide"],
                    stop_price=float(item["stopPrice"]),
                    working_type=item["workingType"],
                    orig_type=item["origType"],
                    time=datetime.fromtimestamp(item["time"] / 1000),
                    update_time=datetime.fromtimestamp(item["updateTime"] / 1000)
                ) for item in data
            ]

    @handle_api_error
    async def get_user_commission_rate(self, symbol: str) -> Dict:
        """Get user commission rate."""
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "timestamp": timestamp,
            "recvWindow": self.config.recv_window
        }

        signature = self._generate_signature("GET", "/fapi/v1/commissionRate", params, str(timestamp))
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self.config.api_key}
        async with self.session.get(f"{self.config.base_url}/fapi/v1/commissionRate",
                                  params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


class AsterWebSocketClient:
    """WebSocket client for real-time data."""
    
    def __init__(self, config: AsterConfig):
        self.config = config
        self.websocket = None
        self.subscriptions = set()
        self.callbacks = {}
    
    async def connect(self):
        """Connect to WebSocket."""
        try:
            self.websocket = await websockets.connect(self.config.ws_url)
            logger.info("Connected to Aster WebSocket")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from Aster WebSocket")
    
    async def subscribe_ticker(self, symbol: str, callback):
        """Subscribe to ticker updates."""
        subscription = f"ticker:{symbol}"
        self.subscriptions.add(subscription)
        self.callbacks[subscription] = callback
        
        message = {
            'method': 'subscribe',
            'params': [subscription]
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to ticker for {symbol}")
    
    async def subscribe_orderbook(self, symbol: str, callback):
        """Subscribe to order book updates."""
        subscription = f"orderbook:{symbol}"
        self.subscriptions.add(subscription)
        self.callbacks[subscription] = callback
        
        message = {
            'method': 'subscribe',
            'params': [subscription]
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to order book for {symbol}")
    
    async def subscribe_trades(self, symbol: str, callback):
        """Subscribe to trade updates."""
        subscription = f"trades:{symbol}"
        self.subscriptions.add(subscription)
        self.callbacks[subscription] = callback
        
        message = {
            'method': 'subscribe',
            'params': [subscription]
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to trades for {symbol}")
    
    async def listen(self):
        """Listen for incoming messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                # Route message to appropriate callback
                if 'stream' in data:
                    stream = data['stream']
                    if stream in self.callbacks:
                        await self.callbacks[stream](data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")


class AsterClient:
    """Main Aster client combining REST and WebSocket."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.config = AsterConfig(api_key=api_key, secret_key=secret_key)
        self.rest_client = AsterRESTClient(self.config)
        self.ws_client = AsterWebSocketClient(self.config)
    
    async def __aenter__(self):
        await self.rest_client.__aenter__()
        await self.ws_client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.rest_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.ws_client.disconnect()

    # Legacy methods for backward compatibility
    async def connect(self):
        """Connect to Aster DEX (legacy method)."""
        await self.__aenter__()

    async def disconnect(self):
        """Disconnect from Aster DEX (legacy method)."""
        await self.__aexit__(None, None, None)
    
    # Delegate REST methods
    async def get_account_info(self) -> AccountInfo:
        return await self.rest_client.get_account_info()

    async def get_account_balance_v2(self) -> List[AccountBalance]:
        return await self.rest_client.get_account_balance_v2()

    async def test_connectivity(self) -> bool:
        """Test basic connectivity to Aster DEX."""
        try:
            # Try a simple ping/test endpoint
            result = await self.rest_client.ping()
            return result is not None
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False

    async def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics."""
        return await self.rest_client.get_24hr_ticker(symbol)

    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth."""
        return await self.rest_client.get_order_book(symbol, limit)

    async def get_server_time(self) -> int:
        """Get server time for timestamp synchronization."""
        return await self.rest_client.get_server_time()
    
    async def get_positions(self) -> List[Position]:
        return await self.rest_client.get_positions()
    
    async def get_orders(self, symbol: str = None, status: str = None) -> List[OrderResponse]:
        return await self.rest_client.get_orders(symbol, status)
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        return await self.rest_client.place_order(order_request)
    
    async def cancel_order(self, order_id: str) -> bool:
        return await self.rest_client.cancel_order(order_id)
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        return await self.rest_client.cancel_all_orders(symbol)
    
    async def get_market_data(self, symbol: str) -> Dict:
        return await self.rest_client.get_market_data(symbol)
    
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        return await self.rest_client.get_klines(symbol, interval, limit)
    
    # WebSocket methods
    async def subscribe_ticker(self, symbol: str, callback):
        await self.ws_client.subscribe_ticker(symbol, callback)
    
    async def subscribe_orderbook(self, symbol: str, callback):
        await self.ws_client.subscribe_orderbook(symbol, callback)
    
    async def subscribe_trades(self, symbol: str, callback):
        await self.ws_client.subscribe_trades(symbol, callback)
    
    async def listen(self):
        await self.ws_client.listen()
"""
Comprehensive unit tests for Aster DEX API client.
Tests all major API functions with proper mocking and error handling.
"""

import asyncio
import json
import pytest
import aiohttp
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

from mcp_trader.execution.aster_client import (
    AsterClient, AsterRESTClient, AsterConfig, OrderRequest, OrderResponse,
    AccountInfo, Position, KlineInterval, OrderSide, OrderType, PositionSide
)


class TestAsterClient:
    """Test suite for Aster DEX API client."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return AsterConfig(
            api_key="test_key",
            secret_key="test_secret",
            base_url="https://test.aster.network",
            timeout=10
        )

    @pytest.fixture
    def client(self, config):
        """Test client instance."""
        return AsterRESTClient(config)

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        session = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def mock_response(self):
        """Mock aiohttp response."""
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock()
        response.text = AsyncMock()
        return response

    class TestInitialization:
        """Test client initialization."""

        def test_config_validation(self):
            """Test configuration validation."""
            with pytest.raises(ValueError):
                AsterConfig(api_key="", secret_key="test")

            with pytest.raises(ValueError):
                AsterConfig(api_key="test", secret_key="")

            with pytest.raises(ValueError):
                AsterConfig(api_key="test", secret_key="test", base_url="")

        def test_client_creation(self, config):
            """Test client creation with valid config."""
            client = AsterClient(config)
            assert client.config == config
            assert client.session is None

    class TestAuthentication:
        """Test authentication methods."""

        def test_signature_generation(self, client):
            """Test HMAC signature generation."""
            method = "GET"
            endpoint = "/api/v1/test"
            params = {"param1": "value1", "param2": "value2"}
            timestamp = "1234567890"

            signature = client._generate_signature(method, endpoint, params, timestamp)
            assert isinstance(signature, str)
            assert len(signature) > 0

        def test_headers_generation(self, client):
            """Test header generation."""
            headers = client._get_headers()
            assert "Content-Type" in headers
            assert "User-Agent" in headers

            signed_headers = client._get_headers(signed=True)
            assert "X-MBX-APIKEY" in signed_headers

        def test_timestamp_sync(self, client):
            """Test timestamp synchronization."""
            timestamp = client._get_synced_timestamp()
            assert isinstance(timestamp, str)
            assert len(timestamp) > 0

    class TestPublicEndpoints:
        """Test public API endpoints."""

        @pytest.mark.asyncio
        async def test_ping(self, client, mock_session, mock_response):
            """Test ping endpoint."""
            mock_response.json.return_value = {"status": "ok"}
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.ping()
                assert result == {"status": "ok"}
                mock_session.get.assert_called_once()

        @pytest.mark.asyncio
        async def test_server_time(self, client, mock_session, mock_response):
            """Test server time endpoint."""
            mock_response.json.return_value = {"serverTime": 1234567890123}
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_server_time()
                assert result == 1234567890123

        @pytest.mark.asyncio
        async def test_24hr_ticker(self, client, mock_session, mock_response):
            """Test 24hr ticker endpoint."""
            ticker_data = {
                "symbol": "BTCUSDT",
                "priceChange": "100.00",
                "priceChangePercent": "1.00",
                "weightedAvgPrice": "50000.00",
                "prevClosePrice": "49500.00",
                "lastPrice": "50000.00",
                "lastQty": "0.00100000",
                "bidPrice": "49999.00",
                "bidQty": "0.50000000",
                "askPrice": "50001.00",
                "askQty": "0.50000000",
                "openPrice": "49500.00",
                "highPrice": "51000.00",
                "lowPrice": "49000.00",
                "volume": "100.00000000",
                "quoteVolume": "5000000.00",
                "openTime": 1640995200000,
                "closeTime": 1641081600000,
                "firstId": 1,
                "lastId": 100,
                "count": 100
            }
            mock_response.json.return_value = ticker_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_24hr_ticker("BTCUSDT")
                assert result == ticker_data

        @pytest.mark.asyncio
        async def test_order_book(self, client, mock_session, mock_response):
            """Test order book endpoint."""
            orderbook_data = {
                "lastUpdateId": 1027024,
                "bids": [
                    ["50000.00", "0.50000000"],
                    ["49999.00", "1.00000000"]
                ],
                "asks": [
                    ["50001.00", "0.50000000"],
                    ["50002.00", "1.00000000"]
                ]
            }
            mock_response.json.return_value = orderbook_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_order_book("BTCUSDT", 100)
                assert result == orderbook_data

        @pytest.mark.asyncio
        async def test_recent_trades(self, client, mock_session, mock_response):
            """Test recent trades endpoint."""
            trades_data = [
                {
                    "id": 12345,
                    "price": "50000.00",
                    "qty": "0.00100000",
                    "quoteQty": "50.00",
                    "time": 1640995200000,
                    "isBuyerMaker": True,
                    "isBestMatch": True
                }
            ]
            mock_response.json.return_value = trades_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_recent_trades("BTCUSDT", 500)
                assert result == trades_data

        @pytest.mark.asyncio
        async def test_klines(self, client, mock_session, mock_response):
            """Test klines endpoint."""
            klines_data = [
                [
                    1640995200000,  # Open time
                    "50000.00",     # Open
                    "51000.00",     # High
                    "49000.00",     # Low
                    "50000.00",     # Close
                    "100.00",       # Volume
                    1641081600000,  # Close time
                    "5000000.00",   # Quote asset volume
                    100,            # Number of trades
                    "50.00",        # Taker buy base asset volume
                    "2500000.00",   # Taker buy quote asset volume
                    "0"             # Unused field
                ]
            ]
            mock_response.json.return_value = klines_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_klines("BTCUSDT", KlineInterval.ONE_MINUTE, limit=100)
                assert isinstance(result, list)
                assert len(result) == 1

    class TestPrivateEndpoints:
        """Test private API endpoints requiring authentication."""

        @pytest.mark.asyncio
        async def test_account_info(self, client, mock_session, mock_response):
            """Test account info endpoint."""
            account_data = {
                "makerCommission": 10,
                "takerCommission": 10,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": True,
                "canWithdraw": True,
                "canDeposit": True,
                "updateTime": 1640995200000,
                "accountType": "SPOT",
                "balances": [
                    {
                        "asset": "BTC",
                        "free": "0.00100000",
                        "locked": "0.00000000"
                    },
                    {
                        "asset": "USDT",
                        "free": "1000.00",
                        "locked": "0.00"
                    }
                ],
                "permissions": ["SPOT"]
            }
            mock_response.json.return_value = account_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_account_info()
                assert isinstance(result, AccountInfo)
                assert result.can_trade == True
                assert len(result.balances) == 2

        @pytest.mark.asyncio
        async def test_place_order(self, client, mock_session, mock_response):
            """Test place order endpoint."""
            order_response_data = {
                "symbol": "BTCUSDT",
                "orderId": 12345,
                "orderListId": -1,
                "clientOrderId": "test_order_123",
                "transactTime": 1640995200000,
                "price": "50000.00",
                "origQty": "0.00100000",
                "executedQty": "0.00000000",
                "cummulativeQuoteQty": "0.00000000",
                "status": "NEW",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY",
                "workingTime": 1640995200000,
                "selfTradePreventionMode": "NONE",
                "fills": []
            }
            mock_response.json.return_value = order_response_data
            mock_session.post.return_value.__aenter__.return_value = mock_response

            order_request = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.00
            )

            with patch.object(client, 'session', mock_session):
                result = await client.place_order(order_request)
                assert isinstance(result, OrderResponse)
                assert result.order_id == 12345
                assert result.status == "NEW"

        @pytest.mark.asyncio
        async def test_cancel_order(self, client, mock_session, mock_response):
            """Test cancel order endpoint."""
            cancel_response = {
                "symbol": "BTCUSDT",
                "origClientOrderId": "test_order_123",
                "orderId": 12345,
                "orderListId": -1,
                "clientOrderId": "cancel_test_123",
                "price": "50000.00",
                "origQty": "0.00100000",
                "executedQty": "0.00000000",
                "cummulativeQuoteQty": "0.00000000",
                "status": "CANCELED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"
            }
            mock_response.json.return_value = cancel_response
            mock_session.delete.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.cancel_order("test_order_123")
                assert result == True

        @pytest.mark.asyncio
        async def test_get_positions(self, client, mock_session, mock_response):
            """Test get positions endpoint."""
            positions_data = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.00100000",
                    "entryPrice": "50000.00",
                    "markPrice": "51000.00",
                    "unRealizedProfit": "100.00",
                    "liquidationPrice": "25000.00",
                    "leverage": "10",
                    "maxNotionalValue": "1000000.00",
                    "marginType": "isolated",
                    "isolatedMargin": "50.00",
                    "isAutoAddMargin": "false",
                    "positionSide": "BOTH",
                    "notional": "51.00",
                    "isolatedWallet": "50.00",
                    "updateTime": 1640995200000
                }
            ]
            mock_response.json.return_value = positions_data
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                result = await client.get_positions()
                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], Position)
                assert result[0].symbol == "BTCUSDT"

    class TestErrorHandling:
        """Test error handling scenarios."""

        @pytest.mark.asyncio
        async def test_api_error_handling(self, client, mock_session, mock_response):
            """Test API error handling."""
            mock_response.status = 400
            mock_response.json.return_value = {
                "code": -2014,
                "msg": "API-key format invalid."
            }
            mock_session.get.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                with pytest.raises(Exception):
                    await client.get_account_info()

        @pytest.mark.asyncio
        async def test_network_timeout(self, client, mock_session):
            """Test network timeout handling."""
            mock_session.get.side_effect = asyncio.TimeoutError()
            mock_session.close = AsyncMock()

            with patch.object(client, 'session', mock_session):
                with pytest.raises(asyncio.TimeoutError):
                    await client.ping()

        @pytest.mark.asyncio
        async def test_connection_error(self, client, mock_session):
            """Test connection error handling."""
            mock_session.get.side_effect = aiohttp.ClientConnectionError()
            mock_session.close = AsyncMock()

            with patch.object(client, 'session', mock_session):
                with pytest.raises(aiohttp.ClientConnectionError):
                    await client.ping()

    class TestDataValidation:
        """Test data validation and parsing."""

        def test_order_request_validation(self):
            """Test order request validation."""
            # Valid order request
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.00
            )
            assert order.symbol == "BTCUSDT"

            # Invalid quantity
            with pytest.raises(ValueError):
                OrderRequest(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=-0.001,
                    price=50000.00
                )

        def test_position_parsing(self, client):
            """Test position data parsing."""
            position_data = {
                "symbol": "BTCUSDT",
                "positionAmt": "0.00100000",
                "entryPrice": "50000.00",
                "markPrice": "51000.00",
                "unRealizedProfit": "10.00",
                "liquidationPrice": "25000.00",
                "leverage": "10",
                "marginType": "isolated",
                "isolatedMargin": "50.00",
                "positionSide": "BOTH",
                "updateTime": 1640995200000
            }

            position = client._parse_position(position_data)
            assert position.symbol == "BTCUSDT"
            assert position.position_amount == 0.001
            assert position.entry_price == 50000.00
            assert position.unrealized_pnl == 10.00

    class TestIntegration:
        """Integration tests combining multiple API calls."""

        @pytest.mark.asyncio
        async def test_full_trading_workflow(self, client, mock_session, mock_response):
            """Test a complete trading workflow."""
            # Mock responses for different endpoints
            responses = {
                "ping": {"status": "ok"},
                "account": {
                    "canTrade": True,
                    "balances": [{"asset": "USDT", "free": "1000.00"}]
                },
                "ticker": {"lastPrice": "50000.00"},
                "order": {"orderId": 12345, "status": "NEW"},
                "positions": []
            }

            call_count = 0

            async def mock_json():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return responses["ping"]
                elif call_count == 2:
                    return responses["account"]
                elif call_count == 3:
                    return responses["ticker"]
                elif call_count == 4:
                    return responses["order"]
                elif call_count == 5:
                    return responses["positions"]
                return {}

            mock_response.json.side_effect = mock_json
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session.post.return_value.__aenter__.return_value = mock_response

            with patch.object(client, 'session', mock_session):
                # Test connectivity
                ping_result = await client.ping()
                assert ping_result["status"] == "ok"

                # Check account
                account = await client.get_account_info()
                assert account.can_trade == True

                # Get market data
                ticker = await client.get_24hr_ticker("BTCUSDT")
                assert "lastPrice" in ticker

                # Place order
                order_req = OrderRequest(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=0.001
                )
                order_resp = await client.place_order(order_req)
                assert order_resp.order_id == 12345

                # Check positions
                positions = await client.get_positions()
                assert isinstance(positions, list)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

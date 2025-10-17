# AsterAI API Documentation

Comprehensive API reference for the AsterAI trading system, including REST endpoints, WebSocket connections, and configuration options.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket API](#websocket-api)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)
- [Examples](#examples)

## Overview

The AsterAI API provides programmatic access to:
- Real-time trading data and positions
- Historical market data and analytics
- Trading system control and configuration
- Performance metrics and risk analysis
- System health and monitoring

### Base URL
```
http://localhost:8080/api/v1  # Local development
https://your-deployment-url/api/v1  # Production
```

### API Versions
- **v1**: Current stable version (default)
- **v1beta**: Beta features and experimental endpoints

## Authentication

### API Key Authentication
Include your API key in requests:

```bash
curl -H "X-API-Key: your_api_key_here" \
     http://localhost:8080/api/v1/status
```

### Service Account Authentication (GCP)
For production deployments, use service account authentication:

```bash
# Set up service account key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Authenticate requests automatically
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     https://your-deployment-url/api/v1/status
```

## REST API Endpoints

### System Status

#### Get System Health
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-17T10:30:00Z",
  "version": "1.0.0",
  "uptime": "2h 15m 30s",
  "services": {
    "trading_engine": "running",
    "data_pipeline": "running",
    "risk_manager": "running",
    "dashboard": "running"
  }
}
```

#### Get System Status
```http
GET /api/v1/status
```

**Response:**
```json
{
  "status": "operational",
  "mode": "live_trading",
  "capital": {
    "total": 150.75,
    "available": 125.50,
    "locked": 25.25
  },
  "positions": {
    "active": 3,
    "pending": 1,
    "total_pnl": 12.45
  },
  "performance": {
    "daily_pnl": 2.15,
    "win_rate": 0.68,
    "sharpe_ratio": 1.85,
    "max_drawdown": 0.045
  }
}
```

### Trading Operations

#### Get Portfolio
```http
GET /api/v1/portfolio
```

**Response:**
```json
{
  "total_value": 150.75,
  "positions": [
    {
      "symbol": "BTC/USDT",
      "type": "long",
      "quantity": 0.0015,
      "entry_price": 45000.0,
      "current_price": 45500.0,
      "pnl": 7.50,
      "pnl_percentage": 1.11
    }
  ],
  "cash_balance": 125.50,
  "day_pnl": 2.15
}
```

#### Get Positions
```http
GET /api/v1/positions
GET /api/v1/positions/{position_id}
```

**Response:**
```json
{
  "positions": [
    {
      "id": "pos_12345",
      "symbol": "ETH/USDT",
      "side": "long",
      "quantity": 2.5,
      "entry_price": 2500.0,
      "current_price": 2520.0,
      "unrealized_pnl": 50.0,
      "realized_pnl": 0.0,
      "entry_time": "2025-01-17T09:00:00Z",
      "stop_loss": 2400.0,
      "take_profit": 2700.0
    }
  ]
}
```

#### Get Orders
```http
GET /api/v1/orders
GET /api/v1/orders/{order_id}
```

**Response:**
```json
{
  "orders": [
    {
      "id": "ord_67890",
      "symbol": "BTC/USDT",
      "type": "limit",
      "side": "buy",
      "quantity": 0.001,
      "price": 44000.0,
      "status": "pending",
      "timestamp": "2025-01-17T10:25:00Z"
    }
  ]
}
```

### Market Data

#### Get Market Data
```http
GET /api/v1/market/{symbol}
```

**Parameters:**
- `symbol`: Trading pair (e.g., BTC/USDT, ETH/USDT)
- `interval`: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- `limit`: Number of data points (max 1000)

**Response:**
```json
{
  "symbol": "BTC/USDT",
  "interval": "1h",
  "data": [
    {
      "timestamp": "2025-01-17T10:00:00Z",
      "open": 45000.0,
      "high": 45200.0,
      "low": 44900.0,
      "close": 45100.0,
      "volume": 125.5
    }
  ]
}
```

#### Get Technical Indicators
```http
GET /api/v1/indicators/{symbol}
```

**Parameters:**
- `symbol`: Trading pair
- `indicators`: Comma-separated list (rsi,macd,bbands,sma,ema)
- `periods`: Time periods for indicators

**Response:**
```json
{
  "symbol": "BTC/USDT",
  "indicators": {
    "rsi": {
      "period": 14,
      "values": [65.5, 67.2, 63.1]
    },
    "macd": {
      "fast": 12,
      "slow": 26,
      "signal": 9,
      "values": [
        {
          "macd": 125.5,
          "signal": 120.3,
          "histogram": 5.2
        }
      ]
    }
  }
}
```

### Risk Management

#### Get Risk Metrics
```http
GET /api/v1/risk
```

**Response:**
```json
{
  "portfolio_risk": {
    "var_95": 0.025,
    "var_99": 0.035,
    "max_drawdown": 0.045,
    "current_drawdown": 0.015
  },
  "position_risk": [
    {
      "position_id": "pos_12345",
      "risk_percentage": 0.02,
      "stop_loss_distance": 0.03,
      "time_risk": 0.015
    }
  ],
  "system_limits": {
    "max_position_size": 0.05,
    "max_portfolio_risk": 0.08,
    "daily_loss_limit": 0.03
  }
}
```

#### Emergency Controls
```http
POST /api/v1/emergency/stop
POST /api/v1/emergency/pause
POST /api/v1/emergency/resume
```

**Request Body:**
```json
{
  "reason": "Market volatility too high",
  "duration_minutes": 60
}
```

### Configuration

#### Get Configuration
```http
GET /api/v1/config
```

#### Update Configuration
```http
PUT /api/v1/config
```

**Request Body:**
```json
{
  "risk_management": {
    "max_position_size": 0.03,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.06
  },
  "trading_parameters": {
    "min_trade_size": 10.0,
    "max_daily_trades": 20,
    "cooldown_minutes": 5
  }
}
```

### Performance Analytics

#### Get Performance Summary
```http
GET /api/v1/performance
```

**Response:**
```json
{
  "period": "30d",
  "total_return": 0.125,
  "annualized_return": 1.85,
  "sharpe_ratio": 1.75,
  "max_drawdown": 0.045,
  "win_rate": 0.68,
  "total_trades": 145,
  "avg_trade_duration": "4h 30m",
  "profit_factor": 2.15
}
```

#### Get Trade History
```http
GET /api/v1/trades
```

**Parameters:**
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `symbol`: Filter by trading pair
- `side`: Filter by side (buy/sell)
- `limit`: Number of trades (max 1000)

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_12345",
      "symbol": "BTC/USDT",
      "side": "buy",
      "quantity": 0.001,
      "entry_price": 45000.0,
      "exit_price": 45500.0,
      "pnl": 5.0,
      "pnl_percentage": 0.011,
      "entry_time": "2025-01-17T09:00:00Z",
      "exit_time": "2025-01-17T14:30:00Z",
      "duration": "5h 30m",
      "fees": 0.45
    }
  ],
  "pagination": {
    "total": 145,
    "page": 1,
    "per_page": 50,
    "pages": 3
  }
}
```

## WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/v1');

ws.onopen = function() {
  console.log('Connected to AsterAI WebSocket');
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Subscription Topics

#### Subscribe to Market Data
```json
{
  "type": "subscribe",
  "topic": "market_data",
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "interval": "1m"
}
```

#### Subscribe to Positions
```json
{
  "type": "subscribe",
  "topic": "positions"
}
```

#### Subscribe to Orders
```json
{
  "type": "subscribe",
  "topic": "orders"
}
```

#### Subscribe to Risk Metrics
```json
{
  "type": "subscribe",
  "topic": "risk"
}
```

### Message Types

#### Market Data Update
```json
{
  "type": "market_data",
  "symbol": "BTC/USDT",
  "data": {
    "timestamp": "2025-01-17T10:30:00Z",
    "price": 45100.0,
    "volume": 25.5,
    "bid": 45095.0,
    "ask": 45105.0
  }
}
```

#### Position Update
```json
{
  "type": "position_update",
  "position": {
    "id": "pos_12345",
    "symbol": "ETH/USDT",
    "current_price": 2520.0,
    "unrealized_pnl": 50.0
  }
}
```

#### Order Update
```json
{
  "type": "order_update",
  "order": {
    "id": "ord_67890",
    "status": "filled",
    "filled_quantity": 0.001,
    "average_price": 44000.0
  }
}
```

#### Risk Alert
```json
{
  "type": "risk_alert",
  "level": "warning",
  "message": "Portfolio risk approaching limit",
  "metrics": {
    "current_risk": 0.075,
    "max_risk": 0.08
  }
}
```

## Data Models

### Position
```typescript
interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  entry_time: string;
  stop_loss?: number;
  take_profit?: number;
}
```

### Order
```typescript
interface Order {
  id: string;
  symbol: string;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected';
  timestamp: string;
  filled_quantity?: number;
  average_price?: number;
}
```

### Risk Metrics
```typescript
interface RiskMetrics {
  var_95: number;        // 95% Value at Risk
  var_99: number;        // 99% Value at Risk
  max_drawdown: number;  // Maximum drawdown
  current_drawdown: number;
  leverage: number;      // Current leverage
  margin_used: number;   // Margin utilization
}
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found (resource doesn't exist)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error
- `503`: Service Unavailable

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Trading symbol 'INVALID/USDT' is not supported",
    "details": {
      "supported_symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    }
  },
  "timestamp": "2025-01-17T10:30:00Z"
}
```

## Rate Limits

### Request Limits
- **Per Minute**: 1000 requests
- **Per Hour**: 10,000 requests
- **Per Day**: 100,000 requests

### WebSocket Limits
- **Max Subscriptions**: 50 per connection
- **Message Rate**: 100 messages/second
- **Connection Timeout**: 5 minutes of inactivity

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1642412400
```

## Examples

### Python Client
```python
import requests
import json

class AsterAITradingAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {'X-API-Key': api_key}

    def get_portfolio(self):
        response = requests.get(
            f'{self.base_url}/api/v1/portfolio',
            headers=self.headers
        )
        return response.json()

    def get_positions(self):
        response = requests.get(
            f'{self.base_url}/api/v1/positions',
            headers=self.headers
        )
        return response.json()

# Usage
api = AsterAITradingAPI('http://localhost:8080', 'your_api_key')
portfolio = api.get_portfolio()
positions = api.get_positions()
```

### JavaScript Client
```javascript
class AsterAITradingAPI {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }

    async getPortfolio() {
        const response = await fetch(`${this.baseUrl}/api/v1/portfolio`, {
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            }
        });
        return response.json();
    }

    async getMarketData(symbol, interval = '1h', limit = 100) {
        const response = await fetch(
            `${this.baseUrl}/api/v1/market/${symbol}?interval=${interval}&limit=${limit}`,
            {
                headers: {
                    'X-API-Key': this.apiKey
                }
            }
        );
        return response.json();
    }
}

// Usage
const api = new AsterAITradingAPI('http://localhost:8080', 'your_api_key');
const portfolio = await api.getPortfolio();
const marketData = await api.getMarketData('BTC/USDT');
```

### cURL Examples
```bash
# Get system status
curl -H "X-API-Key: your_api_key" \
     http://localhost:8080/api/v1/status

# Get portfolio
curl -H "X-API-Key: your_api_key" \
     http://localhost:8080/api/v1/portfolio

# Get market data
curl -H "X-API-Key: your_api_key" \
     "http://localhost:8080/api/v1/market/BTC/USDT?interval=1h&limit=10"

# Emergency stop
curl -X POST -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"reason": "Market volatility"}' \
     http://localhost:8080/api/v1/emergency/stop
```

## Support

For API support and questions:
- **Documentation Issues**: [GitHub Issues](https://github.com/yourusername/asterai/issues)
- **API Questions**: [GitHub Discussions](https://github.com/yourusername/asterai/discussions)
- **Email Support**: api-support@asterai-trading.com

## Changelog

### v1.0.0 (Current)
- Initial API release
- Core trading operations
- Real-time market data
- Risk management endpoints
- WebSocket support

### Upcoming Features
- Advanced order types (OCO, trailing stops)
- Portfolio rebalancing endpoints
- Custom strategy deployment
- Multi-exchange support

---

*Happy trading with the AsterAI API! 🚀📈💰*

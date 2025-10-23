"""
Mock Asset Data Generator for Multi-Asset Testing
Generates realistic market data for perpetual contracts, spot markets, and stocks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MockMarketData:
    """Mock market data for testing."""
    symbol: str
    asset_class: str  # 'perpetual', 'spot', 'stock'
    price: float
    volume: float
    bid_price: float
    ask_price: float
    spread_bps: float
    volatility: float
    liquidity_score: float  # 0-1 scale
    timestamp: datetime


class MockAssetDataGenerator:
    """Generates realistic mock data for different asset classes."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.base_prices = {
            # Crypto perpetuals/spot
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 100.0,
            "ASTERUSDT": 1.0,
            "SUIUSDT": 0.8,
            "PENGUUSDT": 0.00001,

            # Stocks (simulated prices)
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "TSLA": 250.0,
            "NVDA": 800.0,
            "MSFT": 400.0
        }

        self.volatilities = {
            "high_vol": 0.05,    # 5% daily volatility
            "medium_vol": 0.02,  # 2% daily volatility
            "low_vol": 0.01      # 1% daily volatility
        }

    def generate_market_data(self, symbol: str, asset_class: str,
                           num_points: int = 100) -> List[MockMarketData]:
        """Generate time series of mock market data."""
        base_price = self.base_prices.get(symbol, 100.0)

        # Determine volatility based on asset characteristics
        if asset_class == "perpetual":
            vol_level = "high_vol"  # Crypto perps are volatile
        elif asset_class == "spot":
            vol_level = "medium_vol"  # Spot crypto moderate volatility
        else:  # stocks
            vol_level = "low_vol"  # Stocks generally less volatile

        volatility = self.volatilities[vol_level]

        # Generate price series with realistic volatility
        prices = self._generate_price_series(base_price, volatility, num_points)

        market_data = []
        base_time = datetime.now()

        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(minutes=i)

            # Generate realistic spread and liquidity
            spread_bps, liquidity = self._generate_spread_and_liquidity(
                asset_class, price, i/num_points)

            # Generate bid/ask
            half_spread = (spread_bps / 10000) * price / 2
            bid_price = price - half_spread
            ask_price = price + half_spread

            # Generate volume based on asset class
            volume = self._generate_volume(symbol, asset_class, price)

            market_data.append(MockMarketData(
                symbol=symbol,
                asset_class=asset_class,
                price=price,
                volume=volume,
                bid_price=bid_price,
                ask_price=ask_price,
                spread_bps=spread_bps,
                volatility=volatility,
                liquidity_score=liquidity,
                timestamp=timestamp
            ))

        return market_data

    def _generate_price_series(self, base_price: float, volatility: float,
                              num_points: int) -> np.ndarray:
        """Generate realistic price series with drift and volatility."""
        # Generate random returns
        returns = np.random.normal(0.0001, volatility/np.sqrt(252), num_points)  # Daily to intraday

        # Add some autocorrelation (momentum)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # 10% autocorrelation

        # Convert to price series
        prices = base_price * np.exp(np.cumsum(returns))
        return prices

    def _generate_spread_and_liquidity(self, asset_class: str, price: float,
                                     time_factor: float) -> tuple[float, float]:
        """Generate realistic bid-ask spread and liquidity score."""
        if asset_class == "perpetual":
            # Crypto perps: tight spreads, high liquidity
            base_spread = 0.5  # 0.5 bps base spread
            liquidity = 0.9   # 90% liquidity score

        elif asset_class == "spot":
            # Spot crypto: slightly wider spreads
            base_spread = 2.0  # 2 bps base spread
            liquidity = 0.8   # 80% liquidity score

        else:  # stocks
            # Stocks: wider spreads, variable liquidity
            base_spread = 5.0  # 5 bps base spread
            liquidity = 0.7   # 70% liquidity score

        # Add some randomness and time-based variation
        spread_variation = np.random.uniform(0.5, 1.5)
        time_variation = 1 + 0.2 * np.sin(2 * np.pi * time_factor)  # Market hours effect

        spread_bps = base_spread * spread_variation * time_variation
        liquidity_score = liquidity * (1 + 0.1 * np.random.normal(0, 1))

        # Clamp values
        spread_bps = np.clip(spread_bps, 0.1, 50.0)
        liquidity_score = np.clip(liquidity_score, 0.0, 1.0)

        return spread_bps, liquidity_score

    def _generate_volume(self, symbol: str, asset_class: str, price: float) -> float:
        """Generate realistic trading volume."""
        # Base volumes for different assets
        base_volumes = {
            "BTCUSDT": 1000000,   # $1M daily volume
            "ETHUSDT": 500000,    # $500K daily volume
            "SOLUSDT": 100000,    # $100K daily volume
            "AAPL": 50000,        # 50K shares daily
            "GOOGL": 25000,       # 25K shares daily
            "TSLA": 75000,        # 75K shares daily
        }

        base_volume = base_volumes.get(symbol, 10000)

        # Adjust for asset class
        if asset_class == "perpetual":
            # Perps often have higher volume due to leverage
            volume_multiplier = np.random.uniform(1.5, 3.0)
        elif asset_class == "spot":
            volume_multiplier = np.random.uniform(0.8, 1.5)
        else:  # stocks
            volume_multiplier = np.random.uniform(0.5, 1.2)

        # Add randomness
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)

        return volume

    def generate_perpetual_specific_data(self, symbol: str) -> Dict[str, Any]:
        """Generate perpetual contract specific data."""
        base_price = self.base_prices.get(symbol, 100.0)

        return {
            "funding_rate": np.random.uniform(-0.0005, 0.0005),  # ±0.05% per hour
            "open_interest": np.random.uniform(100000, 1000000),  # OI in USD
            "leverage_available": [1, 2, 5, 10, 25, 50, 100],
            "contract_size": 1.0,  # 1 unit per contract
            "settlement_asset": "USDT",
            "contract_type": "linear",  # or "inverse"
            "maintenance_margin": 0.005,  # 0.5%
            "liquidation_fee": 0.001,     # 0.1%
        }

    def generate_spot_specific_data(self, symbol: str) -> Dict[str, Any]:
        """Generate spot market specific data."""
        return {
            "maker_fee": 0.001,   # 0.1%
            "taker_fee": 0.001,   # 0.1%
            "min_trade_size": 0.000001,  # 1 satoshi for BTC
            "price_precision": 2,
            "quantity_precision": 6,
            "trading_enabled": True,
            "withdraw_enabled": True,
            "deposit_enabled": True,
        }

    def generate_stock_specific_data(self, symbol: str) -> Dict[str, Any]:
        """Generate stock-specific data (simulated)."""
        return {
            "commission_per_share": 0.01,  # $0.01 per share
            "market_hours": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "America/New_York"
            },
            "extended_hours": {
                "pre_market": "04:00-09:30",
                "after_hours": "16:00-20:00"
            },
            "lot_size": 100,  # Round lots
            "dividend_yield": np.random.uniform(0.01, 0.05),  # 1-5%
            "pe_ratio": np.random.uniform(10, 50),
            "market_cap": np.random.uniform(1e9, 1e12),  # $1B to $1T
        }

    def simulate_order_execution(self, symbol: str, asset_class: str,
                               side: str, quantity: float, order_type: str = "market") -> Dict[str, Any]:
        """Simulate order execution with realistic slippage."""
        # Get current market data
        market_data = self.generate_market_data(symbol, asset_class, 1)[0]

        if order_type == "market":
            # Market order: execute at current price with slippage
            slippage_factor = self._get_slippage_factor(asset_class, quantity, market_data.liquidity_score)

            if side == "buy":
                execution_price = market_data.ask_price * (1 + slippage_factor)
            else:  # sell
                execution_price = market_data.bid_price * (1 - slippage_factor)

        else:  # limit order
            # Limit order: simplified execution
            execution_price = market_data.price
            slippage_factor = 0

        # Calculate execution time based on liquidity
        execution_time_ms = self._get_execution_time(asset_class, market_data.liquidity_score)

        # Calculate fees
        fees = self._calculate_fees(asset_class, execution_price, quantity)

        return {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "execution_price": execution_price,
            "slippage": slippage_factor,
            "execution_time_ms": execution_time_ms,
            "fees": fees,
            "total_cost": execution_price * quantity + fees,
            "timestamp": datetime.now()
        }

    def _get_slippage_factor(self, asset_class: str, quantity: float, liquidity: float) -> float:
        """Calculate slippage factor based on order size and liquidity."""
        # Base slippage by asset class
        base_slippage = {
            "perpetual": 0.0005,  # 0.05%
            "spot": 0.001,        # 0.1%
            "stocks": 0.002       # 0.2%
        }

        # Adjust for quantity (market impact)
        quantity_factor = min(quantity / 1000, 5)  # Cap at 5x

        # Adjust for liquidity
        liquidity_factor = 2 * (1 - liquidity)  # Worse liquidity = more slippage

        total_slippage = base_slippage[asset_class] * (1 + quantity_factor) * (1 + liquidity_factor)

        # Add randomness
        total_slippage *= np.random.uniform(0.5, 1.5)

        return total_slippage

    def _get_execution_time(self, asset_class: str, liquidity: float) -> float:
        """Calculate order execution time in milliseconds."""
        base_times = {
            "perpetual": 50,   # 50ms average
            "spot": 100,       # 100ms average
            "stocks": 200      # 200ms average
        }

        # Adjust for liquidity
        liquidity_factor = 2 * (1 - liquidity)  # Worse liquidity = slower execution

        execution_time = base_times[asset_class] * (1 + liquidity_factor)

        # Add randomness
        execution_time *= np.random.uniform(0.5, 2.0)

        return execution_time

    def _calculate_fees(self, asset_class: str, price: float, quantity: float) -> float:
        """Calculate trading fees."""
        fee_rates = {
            "perpetual": 0.001,  # 0.1%
            "spot": 0.001,       # 0.1%
            "stocks": 0.01       # 1% (simplified)
        }

        fee_rate = fee_rates[asset_class]
        notional_value = price * quantity

        return notional_value * fee_rate

    def generate_arbitrage_opportunity(self, symbol1: str, symbol2: str,
                                     asset_class1: str, asset_class2: str) -> Optional[Dict[str, Any]]:
        """Generate potential arbitrage opportunity between two assets."""
        # Generate prices for both assets
        data1 = self.generate_market_data(symbol1, asset_class1, 1)[0]
        data2 = self.generate_market_data(symbol2, asset_class2, 1)[0]

        # Check for arbitrage (simplified)
        price_ratio = data1.price / data2.price
        expected_ratio = self.base_prices[symbol1] / self.base_prices[symbol2]

        deviation = abs(price_ratio - expected_ratio) / expected_ratio

        if deviation > 0.005:  # 0.5% deviation
            return {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "price1": data1.price,
                "price2": data2.price,
                "deviation_pct": deviation * 100,
                "direction": "buy1_sell2" if price_ratio > expected_ratio else "buy2_sell1",
                "potential_profit_pct": deviation * 100 * 0.8  # After fees
            }

        return None


# Utility functions for testing
def create_test_portfolio() -> Dict[str, Any]:
    """Create a test portfolio with positions in different asset classes."""
    return {
        "perpetual_positions": [
            {"symbol": "BTCUSDT", "quantity": 0.05, "entry_price": 50000, "leverage": 10},
            {"symbol": "ETHUSDT", "quantity": 1.0, "entry_price": 3000, "leverage": 5},
        ],
        "spot_positions": [
            {"symbol": "SOLUSDT", "quantity": 50, "entry_price": 100},
            {"symbol": "ASTERUSDT", "quantity": 1000, "entry_price": 1.0},
        ],
        "stock_positions": [
            {"symbol": "AAPL", "quantity": 10, "entry_price": 150},
            {"symbol": "TSLA", "quantity": 5, "entry_price": 250},
        ],
        "total_value": 25000.0,
        "cash": 15000.0
    }


def simulate_market_conditions(scenario: str) -> Dict[str, Any]:
    """Simulate different market conditions for testing."""
    scenarios = {
        "normal": {
            "volatility_multiplier": 1.0,
            "liquidity_multiplier": 1.0,
            "trend": "sideways"
        },
        "volatile": {
            "volatility_multiplier": 2.0,
            "liquidity_multiplier": 0.7,
            "trend": "upward"
        },
        "low_liquidity": {
            "volatility_multiplier": 1.5,
            "liquidity_multiplier": 0.3,
            "trend": "downward"
        },
        "high_frequency": {
            "volatility_multiplier": 0.5,
            "liquidity_multiplier": 1.5,
            "trend": "sideways"
        }
    }

    return scenarios.get(scenario, scenarios["normal"])

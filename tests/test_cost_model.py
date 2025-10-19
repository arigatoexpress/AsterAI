import asyncio
import math
from mcp_trader.backtesting.cost_model import OnChainCostModel, CostModelConfig


def test_cost_model_fallback_execution_price_buy():
    model = OnChainCostModel(client=None, config=CostModelConfig(max_slippage_pct=0.02))
    base = 100.0
    qty = 10.0
    # No client/orderbook → fallback adaptive slippage, capped by max_slippage_pct
    price = asyncio.run(model.estimate_execution_price('BTCUSDT', base, qty, 'buy'))
    assert price >= base
    assert price <= base * (1 + model.config.max_slippage_pct + 1e-9)


def test_cost_model_fee_fallback():
    model = OnChainCostModel(client=None, config=CostModelConfig())
    notional = 1000.0
    fee_taker = asyncio.run(model.estimate_fee('ETHUSDT', notional, is_maker=False))
    assert math.isclose(fee_taker, notional * model.config.default_taker_fee, rel_tol=1e-6)


import types
import asyncio
from datetime import datetime

from mcp_trader.trading.autonomous_trader import AutonomousTrader


class DummyFeed:
    async def get_ticker(self, symbol):
        return {'last': 100.0}


async def _noop(*args, **kwargs):
    return None


def test_global_risk_guard_daily_loss_trips(monkeypatch):
    trader = AutonomousTrader(config={})
    trader.data_feed = DummyFeed()
    # Fake portfolio state showing large loss vs start_of_day
    trader.portfolio_state.total_balance = 1000.0
    trader.portfolio_state.unrealized_pnl = -500.0
    trader.start_of_day_value = 2000.0
    trader._last_pnl_reset_date = datetime.utcnow().date()

    # Decisions list should be skipped due to daily loss limit
    decisions = []

    # Patch methods to avoid real IO
    monkeypatch.setattr(trader, '_update_portfolio_state', types.MethodType(lambda self: _noop(), trader))

    asyncio.run(trader._execute_decisions(decisions))
    assert trader.emergency_stop is True


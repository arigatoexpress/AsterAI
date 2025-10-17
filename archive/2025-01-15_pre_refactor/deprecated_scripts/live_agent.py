from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from mcp_trader.execution.trade_executor import TradeExecutor
from mcp_trader.strategies.rules import generate_positions_sma_crossover


@dataclass
class AgentConfig:
    symbol: str = "BTCUSDT"
    short_win: int = 20
    long_win: int = 50
    leverage: int = 5
    quantity: float = 0.001


class LiveAgent:
    def __init__(self, executor: TradeExecutor | None = None, cfg: AgentConfig | None = None):
        self.executor = executor or TradeExecutor()
        self.cfg = cfg or AgentConfig()
        self.current_position = 0.0

    def on_price_series(self, close_series: pd.Series) -> None:
        pos = generate_positions_sma_crossover(pd.DataFrame({"close": close_series}), self.cfg.short_win, self.cfg.long_win)
        target = float(pos.iloc[-1])
        if target > self.current_position:
            qty = self.cfg.quantity * (target - self.current_position)
            self.executor.market_buy(self.cfg.symbol, qty, leverage=self.cfg.leverage)
            self.current_position = target
        elif target < self.current_position:
            qty = self.cfg.quantity * (self.current_position - target)
            self.executor.market_sell(self.cfg.symbol, qty, leverage=self.cfg.leverage)
            self.current_position = target

    def run_polling_demo(self, close_series: pd.Series, interval_sec: int = 5) -> None:
        for _ in range(3):
            self.on_price_series(close_series)
            time.sleep(interval_sec)

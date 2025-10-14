from __future__ import annotations

import numpy as np
import pandas as pd

from mcp_trader.strategies.indicators import sma, rsi


def generate_positions_sma_crossover(data: pd.DataFrame, short_win: int, long_win: int) -> pd.Series:
    """Generate positions based on SMA crossover strategy."""
    short_ma = sma(data["close"], short_win)
    long_ma = sma(data["close"], long_win)
    position = np.where(short_ma > long_ma, 1.0, -1.0)
    return pd.Series(position, index=data.index)


def generate_positions_rsi(data: pd.DataFrame, period: int = 14, low: float = 30.0, high: float = 70.0) -> pd.Series:
    """Generate positions based on RSI strategy."""
    r = rsi(data["close"], period)
    position = np.where(r < low, 1.0, np.where(r > high, -1.0, 0.0))
    return pd.Series(position, index=data.index)

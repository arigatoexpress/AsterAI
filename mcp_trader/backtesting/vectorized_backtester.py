from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns_from_close(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)


def compute_strategy_returns(market_returns: pd.Series, positions: pd.Series, fee_bps: float = 0.0) -> pd.Series:
    pos_shifted = positions.shift(1).fillna(0.0)
    strat_ret = pos_shifted * market_returns
    if fee_bps > 0:
        trades = (pos_shifted != pos_shifted.shift(1)).fillna(False)
        costs = trades.astype(float) * (-abs(fee_bps) / 10000.0)
        strat_ret = strat_ret + costs
    return strat_ret.fillna(0.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365, risk_free_rate: float = 0.0) -> float:
    excess = returns - (risk_free_rate / periods_per_year)
    mu = excess.mean() * periods_per_year
    sigma = excess.std(ddof=0) * np.sqrt(periods_per_year)
    if sigma == 0:
        return 0.0
    return float(mu / sigma)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    equity = (1.0 + returns).cumprod()
    ann_ret = equity.iloc[-1] ** (periods_per_year / max(len(equity), 1)) - 1.0
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("inf") if ann_ret > 0 else 0.0
    return float(ann_ret / mdd)


def evaluate_positions(close: pd.Series, positions: pd.Series, fee_bps: float = 0.0, periods_per_year: int = 365) -> dict:
    market_ret = compute_returns_from_close(close)
    strat_returns = compute_strategy_returns(market_ret, positions, fee_bps=fee_bps)
    equity = (1.0 + strat_returns).cumprod()

    metrics = {
        "sharpe": sharpe_ratio(strat_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(strat_returns, periods_per_year=periods_per_year),
        "profit_factor": profit_factor(strat_returns),
        "total_return": float(equity.iloc[-1] - 1.0),
    }
    return {"returns": strat_returns, "equity": equity, "metrics": metrics}


"""
Risk metrics utilities for trading performance evaluation.
"""

import numpy as np
from typing import List, Union


def calculate_sortino_ratio(returns: Union[List[float], np.ndarray],
                           risk_free_rate: float = 0.0,
                           annualization_factor: float = 252) -> float:
    """
    Calculate the Sortino ratio, which measures risk-adjusted returns focusing only on downside volatility.

    Sortino Ratio = (Rp - Rf) / σ_d

    Where:
    - Rp = Portfolio return (annualized)
    - Rf = Risk-free rate (annualized)
    - σ_d = Downside deviation (annualized standard deviation of negative returns)

    Args:
        returns: Array of periodic returns (daily, hourly, etc.)
        risk_free_rate: Annualized risk-free rate (default: 0.0)
        annualization_factor: Factor to annualize returns (252 for daily, 365 for hourly, etc.)

    Returns:
        Sortino ratio as a float
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    # Calculate annualized portfolio return
    cumulative_return = np.prod(1 + returns_array) - 1
    if len(returns_array) > 0:
        annualized_return = (1 + cumulative_return) ** (annualization_factor / len(returns_array)) - 1
    else:
        annualized_return = 0.0

    # Calculate downside deviation (only negative returns)
    negative_returns = returns_array[returns_array < 0]
    if len(negative_returns) == 0:
        # No downside risk - return high positive value
        return float('inf') if annualized_return > risk_free_rate else 0.0

    # Annualize downside deviation
    downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(annualization_factor) if len(negative_returns) > 1 else 0

    if downside_deviation == 0:
        return float('inf') if annualized_return > risk_free_rate else 0.0

    # Calculate Sortino ratio
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation

    return float(sortino_ratio)


def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray],
                          risk_free_rate: float = 0.0,
                          annualization_factor: float = 252) -> float:
    """
    Calculate the Sharpe ratio, which measures risk-adjusted returns using total volatility.

    Sharpe Ratio = (Rp - Rf) / σ

    Where:
    - Rp = Portfolio return (annualized)
    - Rf = Risk-free rate (annualized)
    - σ = Total volatility (annualized standard deviation of all returns)

    Args:
        returns: Array of periodic returns (daily, hourly, etc.)
        risk_free_rate: Annualized risk-free rate (default: 0.0)
        annualization_factor: Factor to annualize returns (252 for daily, 365 for hourly, etc.)

    Returns:
        Sharpe ratio as a float
    """
    if len(returns) == 0:
        return 0.0

    returns_array = np.array(returns)

    # Calculate annualized portfolio return
    cumulative_return = np.prod(1 + returns_array) - 1
    if len(returns_array) > 0:
        annualized_return = (1 + cumulative_return) ** (annualization_factor / len(returns_array)) - 1
    else:
        annualized_return = 0.0

    # Calculate total volatility
    if len(returns_array) > 1:
        total_volatility = np.std(returns_array, ddof=1) * np.sqrt(annualization_factor)
    else:
        total_volatility = 0.0

    if total_volatility == 0:
        return float('inf') if annualized_return > risk_free_rate else 0.0

    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / total_volatility

    return float(sharpe_ratio)


def calculate_max_drawdown(portfolio_values: Union[List[float], np.ndarray]) -> float:
    """
    Calculate the maximum drawdown from a series of portfolio values.

    Args:
        portfolio_values: Array of portfolio values over time

    Returns:
        Maximum drawdown as a decimal (negative value)
    """
    if len(portfolio_values) == 0:
        return 0.0

    values_array = np.array(portfolio_values)
    peak = np.maximum.accumulate(values_array)
    drawdown = (values_array - peak) / peak
    max_drawdown = np.min(drawdown)

    return float(max_drawdown)


def calculate_calmar_ratio(returns: Union[List[float], np.ndarray],
                          portfolio_values: Union[List[float], np.ndarray],
                          risk_free_rate: float = 0.0,
                          annualization_factor: float = 252) -> float:
    """
    Calculate the Calmar ratio, which measures risk-adjusted returns relative to maximum drawdown.

    Calmar Ratio = (Rp - Rf) / |Max Drawdown|

    Args:
        returns: Array of periodic returns
        portfolio_values: Array of portfolio values over time
        risk_free_rate: Annualized risk-free rate
        annualization_factor: Factor to annualize returns

    Returns:
        Calmar ratio as a float
    """
    if len(returns) == 0 or len(portfolio_values) == 0:
        return 0.0

    # Calculate annualized return
    cumulative_return = np.prod(1 + np.array(returns)) - 1
    if len(returns) > 0:
        annualized_return = (1 + cumulative_return) ** (annualization_factor / len(returns)) - 1
    else:
        annualized_return = 0.0

    # Calculate max drawdown
    max_dd = calculate_max_drawdown(portfolio_values)

    if max_dd == 0:
        return float('inf') if annualized_return > risk_free_rate else 0.0

    # Calmar ratio
    calmar_ratio = (annualized_return - risk_free_rate) / abs(max_dd)

    return float(calmar_ratio)

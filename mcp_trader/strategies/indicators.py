import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(method="bfill")


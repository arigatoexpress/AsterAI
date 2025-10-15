import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_price_data(initial_price: float,
                                periods: int,
                                volatility: float = 0.02,
                                trend: float = 0.0001,
                                freq: str = 'H') -> pd.Series:
    """
    Generate synthetic price data using geometric Brownian motion.

    Args:
        initial_price: Starting price
        periods: Number of periods to generate
        volatility: Daily volatility
        trend: Drift term (trend)
        freq: Frequency ('H' for hourly, 'D' for daily)

    Returns:
        Pandas Series with price data
    """
    np.random.seed(42)  # For reproducible results

    # Adjust volatility for frequency
    if freq == 'H':
        volatility = volatility / np.sqrt(24)  # Convert daily to hourly volatility
        periods_per_day = 24
    elif freq == 'D':
        periods_per_day = 1
    else:
        periods_per_day = 1

    # Generate random returns
    dt = 1.0 / periods_per_day
    returns = np.random.normal(trend * dt, volatility * np.sqrt(dt), periods)

    # Calculate price path
    prices = initial_price * np.exp(np.cumsum(returns))

    return pd.Series(prices, name='price')


def generate_realistic_ohlcv_data(initial_price: float,
                                 periods: int,
                                 volatility: float = 0.02,
                                 trend: float = 0.0001,
                                 volume_mean: float = 1000,
                                 volume_std: float = 500) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with correlated high/low prices.

    Args:
        initial_price: Starting price
        periods: Number of periods
        volatility: Price volatility
        trend: Price trend
        volume_mean: Mean volume
        volume_std: Volume standard deviation

    Returns:
        DataFrame with OHLCV columns
    """
    # Generate base price series
    close_prices = generate_synthetic_price_data(initial_price, periods, volatility, trend)

    # Generate OHLC data
    high_multipliers = np.random.uniform(1.001, 1.02, periods)  # High is 0.1%-2% above close
    low_multipliers = np.random.uniform(0.98, 0.999, periods)   # Low is 0.1%-2% below close
    open_prices = close_prices.shift(1).fillna(initial_price)

    # Add some randomness to open prices (gap effects)
    open_multipliers = np.random.normal(1.0, 0.005, periods)
    open_prices = open_prices * open_multipliers

    # Ensure OHLC relationships
    highs = np.maximum(close_prices, open_prices) * high_multipliers
    lows = np.minimum(close_prices, open_prices) * low_multipliers

    # Generate volume data
    volumes = np.random.normal(volume_mean, volume_std, periods)
    volumes = np.maximum(volumes, 10)  # Minimum volume

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })

    # Round prices to 4 decimal places for crypto
    price_columns = ['open', 'high', 'low', 'close']
    df[price_columns] = df[price_columns].round(4)
    df['volume'] = df['volume'].round(2)

    return df


def generate_backtest_data(symbols: List[str],
                          periods: int = 1000,
                          start_date: datetime = None,
                          freq: str = 'H') -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic backtest data for multiple symbols.

    Args:
        symbols: List of trading symbols
        periods: Number of periods to generate
        start_date: Start date for data
        freq: Data frequency

    Returns:
        Dictionary of DataFrames keyed by symbol
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    # Create date range
    date_range = pd.date_range(start_date, periods=periods, freq=freq)

    data = {}

    # Base prices for different assets
    base_prices = {
        'BTCUSDT': 45000,
        'ETHUSDT': 2500,
        'SOLUSDT': 100,
        'ADAUSDT': 0.5,
        'DOTUSDT': 8,
        'LINKUSDT': 15,
        'AVAXUSDT': 35,
        'MATICUSDT': 1,
        'UNIUSDT': 8,
        'AAVEUSDT': 120,
        'SUSHIUSDT': 2,
        'COMPUSDT': 80,
        'MKRUSDT': 2000,
        'YFIUSDT': 8000,
        'BALUSDT': 5,
        'RENUSDT': 0.15,
        'KNCUSDT': 0.8,
        'ZRXUSDT': 0.4,
        'BATUSDT': 0.3,
        'OMGUSDT': 1.5
    }

    for symbol in symbols:
        initial_price = base_prices.get(symbol, 100)  # Default to 100 if not found

        # Adjust volatility and trend based on asset type
        if 'BTC' in symbol:
            volatility = 0.03  # BTC is more volatile
            trend = 0.0002
            volume_mean = 2000
        elif 'ETH' in symbol:
            volatility = 0.035
            trend = 0.00015
            volume_mean = 1500
        elif 'SOL' in symbol or 'ADA' in symbol:
            volatility = 0.04
            trend = 0.0001
            volume_mean = 800
        else:
            volatility = 0.025
            trend = 0.00005
            volume_mean = 500

        logger.info(f"Generating synthetic data for {symbol}: "
                   f"initial=${initial_price}, vol={volatility:.3f}, trend={trend:.6f}")

        # Generate OHLCV data
        df = generate_realistic_ohlcv_data(
            initial_price=initial_price,
            periods=periods,
            volatility=volatility,
            trend=trend,
            volume_mean=volume_mean
        )

        # Set datetime index
        df.index = date_range[:len(df)]

        data[symbol] = df

    logger.info(f"Generated synthetic data for {len(data)} symbols with {periods} periods each")
    return data


def add_realistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realistic technical indicators and features to the data.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Basic technical indicators
    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Exponential moving averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price momentum
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility
    df['realized_volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

    # Price ranges
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']

    # Trend indicators
    df['price_change_1d'] = df['close'].pct_change(1)
    df['price_change_7d'] = df['close'].pct_change(7)
    df['price_change_30d'] = df['close'].pct_change(30)

    # Remove NaN values created by rolling calculations
    df.dropna(inplace=True)

    return df


def generate_ai_training_data(symbols: List[str],
                            periods: int = 5000,
                            sequence_length: int = 60,
                            prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for AI models.

    Args:
        symbols: List of symbols
        periods: Total periods to generate
        sequence_length: Length of input sequences
        prediction_horizon: How many periods ahead to predict

    Returns:
        Tuple of (X, y) arrays for training
    """
    # Generate synthetic data
    raw_data = generate_backtest_data(symbols, periods)

    all_sequences = []
    all_targets = []

    for symbol, df in raw_data.items():
        # Add features
        df_featured = add_realistic_features(df)

        # Skip if not enough data
        if len(df_featured) < sequence_length + prediction_horizon:
            continue

        # Generate sequences
        for i in range(len(df_featured) - sequence_length - prediction_horizon + 1):
            # Input sequence
            sequence = df_featured.iloc[i:i+sequence_length]

            # Select relevant features for AI
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_middle', 'bb_lower',
                'returns', 'realized_volatility'
            ]

            # Normalize features
            sequence_features = sequence[feature_columns].values
            sequence_normalized = (sequence_features - sequence_features.mean(axis=0)) / (sequence_features.std(axis=0) + 1e-8)

            # Target: future price movement
            future_price = df_featured.iloc[i+sequence_length+prediction_horizon-1]['close']
            current_price = sequence.iloc[-1]['close']
            target = 1 if future_price > current_price else 0  # Binary classification: up/down

            all_sequences.append(sequence_normalized)
            all_targets.append(target)

    X = np.array(all_sequences)
    y = np.array(all_targets)

    logger.info(f"Generated AI training data: {X.shape[0]} samples, "
               f"sequence length: {sequence_length}, features: {X.shape[2]}")

    return X, y


def save_training_data(X: np.ndarray, y: np.ndarray, filename: str):
    """Save training data to disk."""
    np.savez(filename, X=X, y=y)
    logger.info(f"Saved training data to {filename}")


def load_training_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data from disk."""
    data = np.load(filename)
    logger.info(f"Loaded training data from {filename}")
    return data['X'], data['y']


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate sample data
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    data = generate_backtest_data(symbols, periods=500, freq='D')

    print("Generated data for symbols:", list(data.keys()))

    # Show sample data
    btc_data = data['BTCUSDT']
    print("\nBTCUSDT sample data:")
    print(btc_data.head())

    # Add features
    btc_featured = add_realistic_features(btc_data)
    print(f"\nBTCUSDT with features: {btc_featured.shape[1]} columns")
    print(btc_featured[['close', 'rsi', 'macd', 'bb_upper', 'bb_lower']].head())

    # Generate AI training data
    X, y = generate_ai_training_data(['BTCUSDT'], periods=1000, sequence_length=30)
    print(f"\nAI Training data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Save sample training data
    save_training_data(X, y, 'sample_training_data.npz')


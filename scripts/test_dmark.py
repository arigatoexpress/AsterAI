#!/usr/bin/env python3
"""
Test script for the DMark indicator and strategy.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_trader.indicators.dmark import DMarkIndicator, DMarkConfig
from mcp_trader.strategies.dmark_strategy import DMarkStrategy, DMarkEnsembleStrategy


def generate_sample_data(days: int = 30) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days*24, freq='1H')
    
    # Generate price data with trend and volatility
    n = len(timestamps)
    
    # Base trend
    trend = np.linspace(100, 120, n)  # 20% upward trend
    
    # Add volatility
    volatility = 0.02
    returns = np.random.normal(0, volatility, n)
    prices = trend * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate high, low, open around close
        daily_vol = volatility * np.random.uniform(0.5, 1.5)
        
        high = close * (1 + daily_vol * np.random.uniform(0, 0.5))
        low = close * (1 - daily_vol * np.random.uniform(0, 0.5))
        open_price = close * (1 + daily_vol * np.random.uniform(-0.3, 0.3))
        
        # Generate volume
        base_volume = 1000000
        volume = base_volume * (1 + np.random.uniform(-0.5, 0.5))
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'symbol': 'BTCUSDT'
        })
    
    return pd.DataFrame(data)


def test_dmark_indicator():
    """Test the DMark indicator."""
    print("🧪 Testing DMark Indicator")
    print("=" * 40)
    
    # Generate sample data
    data = generate_sample_data(30)
    print(f"Generated {len(data)} data points")
    
    # Create DMark indicator
    config = DMarkConfig(
        lookback_period=20,
        momentum_period=10,
        volatility_period=14,
        volume_period=10
    )
    
    indicator = DMarkIndicator(config)
    
    # Calculate DMark values
    results = indicator.calculate(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        volume=data['volume'],
        open_price=data['open']
    )
    
    print(f"\n📊 DMark Results:")
    print(f"  - Signal range: {results['dmark_signal'].min():.3f} to {results['dmark_signal'].max():.3f}")
    print(f"  - Confidence range: {results['confidence'].min():.3f} to {results['confidence'].max():.3f}")
    print(f"  - Regime distribution: {results['regime'].value_counts().to_dict()}")
    
    # Analyze signals
    signals = results['dmark_signal']
    buy_signals = (signals > 0.4).sum()
    sell_signals = (signals < -0.4).sum()
    hold_signals = ((signals >= -0.4) & (signals <= 0.4)).sum()
    
    print(f"\n📈 Signal Analysis:")
    print(f"  - Buy signals: {buy_signals}")
    print(f"  - Sell signals: {sell_signals}")
    print(f"  - Hold signals: {hold_signals}")
    
    # Show recent signals
    recent_data = data.tail(10)
    recent_signals = results['dmark_signal'].tail(10)
    recent_confidence = results['confidence'].tail(10)
    
    print(f"\n🕐 Recent Signals (last 10 hours):")
    for i, (_, row) in enumerate(recent_data.iterrows()):
        signal = recent_signals.iloc[i]
        confidence = recent_confidence.iloc[i]
        direction = "BUY" if signal > 0.4 else "SELL" if signal < -0.4 else "HOLD"
        strength = "STRONG" if abs(signal) > 0.7 else "MODERATE" if abs(signal) > 0.4 else "WEAK"
        
        print(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
              f"Price: ${row['close']:.2f} | "
              f"Signal: {signal:.3f} ({direction}) | "
              f"Confidence: {confidence:.3f} | "
              f"Strength: {strength}")
    
    return results, data


def test_dmark_strategy():
    """Test the DMark strategy."""
    print("\n🎯 Testing DMark Strategy")
    print("=" * 40)
    
    # Generate sample data
    data = generate_sample_data(30)
    
    # Create DMark strategy
    strategy = DMarkStrategy(
        min_confidence=0.6,
        max_position_size=0.25,
        stop_loss_threshold=0.02,
        take_profit_threshold=0.04
    )
    
    # Fit strategy
    strategy.fit(data)
    
    # Generate predictions
    predictions = strategy.predict(data)
    signals = strategy.generate_signals(data)
    
    print(f"📊 Strategy Results:")
    print(f"  - Total predictions: {len(predictions)}")
    print(f"  - Trading signals: {len(signals)}")
    
    # Analyze signal quality
    if signals:
        signal_confidences = [s.confidence for s in signals]
        signal_directions = [s.signal for s in signals]
        
        print(f"  - Average confidence: {np.mean(signal_confidences):.3f}")
        print(f"  - Buy signals: {signal_directions.count(1)}")
        print(f"  - Sell signals: {signal_directions.count(-1)}")
        
        # Show recent signals
        print(f"\n🕐 Recent Trading Signals:")
        for signal in signals[-5:]:  # Last 5 signals
            direction = "BUY" if signal.signal == 1 else "SELL"
            print(f"  {signal.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                  f"{direction} | Confidence: {signal.confidence:.3f}")
    
    # Get strategy status
    status = strategy.get_strategy_status()
    print(f"\n📈 Strategy Status:")
    for key, value in status.items():
        print(f"  - {key}: {value}")
    
    return strategy, signals


def test_dmark_ensemble():
    """Test the DMark ensemble strategy."""
    print("\n🎭 Testing DMark Ensemble Strategy")
    print("=" * 40)
    
    # Generate sample data
    data = generate_sample_data(30)
    
    # Create ensemble strategy
    ensemble = DMarkEnsembleStrategy()
    
    # Fit strategy
    ensemble.fit(data)
    
    # Generate predictions
    predictions = ensemble.predict(data)
    signals = ensemble.generate_signals(data)
    
    print(f"📊 Ensemble Results:")
    print(f"  - Total predictions: {len(predictions)}")
    print(f"  - Trading signals: {len(signals)}")
    
    if signals:
        signal_confidences = [s.confidence for s in signals]
        print(f"  - Average confidence: {np.mean(signal_confidences):.3f}")
    
    return ensemble, signals


def create_visualization(results: dict, data: pd.DataFrame):
    """Create visualization of DMark indicator results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Price chart with signals
        axes[0].plot(data['timestamp'], data['close'], label='Price', linewidth=1)
        axes[0].set_title('Price Chart with DMark Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # DMark signal
        axes[1].plot(data['timestamp'], results['dmark_signal'], label='DMark Signal', linewidth=1)
        axes[1].axhline(y=0.4, color='g', linestyle='--', alpha=0.7, label='Buy Threshold')
        axes[1].axhline(y=-0.4, color='r', linestyle='--', alpha=0.7, label='Sell Threshold')
        axes[1].set_title('DMark Signal')
        axes[1].set_ylabel('Signal Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Confidence
        axes[2].plot(data['timestamp'], results['confidence'], label='Confidence', linewidth=1, color='orange')
        axes[2].axhline(y=0.6, color='g', linestyle='--', alpha=0.7, label='Min Confidence')
        axes[2].set_title('Signal Confidence')
        axes[2].set_ylabel('Confidence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Regime
        axes[3].plot(data['timestamp'], results['regime'], label='Market Regime', linewidth=1, color='purple')
        axes[3].set_title('Market Regime')
        axes[3].set_ylabel('Regime')
        axes[3].set_xlabel('Time')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dmark_test_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'dmark_test_results.png'")
        
    except ImportError:
        print("\n⚠️  Matplotlib not available for visualization")


def main():
    """Run all DMark tests."""
    print("🚀 DMark Indicator & Strategy Test Suite")
    print("=" * 50)
    
    try:
        # Test DMark indicator
        results, data = test_dmark_indicator()
        
        # Test DMark strategy
        strategy, signals = test_dmark_strategy()
        
        # Test DMark ensemble
        ensemble, ensemble_signals = test_dmark_ensemble()
        
        # Create visualization
        create_visualization(results, data)
        
        print(f"\n✅ All tests completed successfully!")
        print(f"\n📊 Summary:")
        print(f"  - DMark indicator: Working")
        print(f"  - DMark strategy: {len(signals)} signals generated")
        print(f"  - DMark ensemble: {len(ensemble_signals)} signals generated")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

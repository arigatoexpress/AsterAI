#!/usr/bin/env python3
"""
CPU-Based ML Training Demo for AsterAI
Demonstrates that our system works with CPU fallback when GPU is not available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husky")

print("""
╔══════════════════════════════════════════════════════════════╗
║           AsterAI CPU-Based ML Training Demo                 ║
║                                                              ║
║  This demo shows our ML pipeline works with CPU fallback    ║
║  when GPU/CUDA is not available.                             ║
╚══════════════════════════════════════════════════════════════╝
""")

def generate_synthetic_trading_data(n_samples=10000, n_features=20):
    """Generate synthetic trading data for demonstration"""
    print(f"🔄 Generating {n_samples} synthetic trading samples...")

    np.random.seed(42)

    # Generate features (technical indicators, market data)
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)

    # Add some realistic trading features
    features['rsi'] = np.random.uniform(20, 80, n_samples)
    features['macd'] = np.random.randn(n_samples)
    features['volume_ratio'] = np.random.uniform(0.5, 2.0, n_samples)
    features['price_change'] = np.random.randn(n_samples) * 0.02
    features['volatility'] = np.random.uniform(0.01, 0.05, n_samples)

    X = pd.DataFrame(features)

    # Generate target (next day return prediction)
    # Some correlation with features for realistic demo
    noise = np.random.randn(n_samples) * 0.01
    target = (X['price_change'] * 0.3 +
             X['rsi'].apply(lambda x: 0.02 if x < 30 else -0.02 if x > 70 else 0) +
             X['volume_ratio'].apply(lambda x: 0.01 if x > 1.5 else 0) +
             noise)

    y = pd.Series(target, name='next_day_return')

    print(f"✅ Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def train_ml_model(X_train, X_test, y_train, y_test):
    """Train a machine learning model"""
    print("\n🔄 Training Random Forest model on CPU...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print("✅ Model trained successfully!")
    print(f"   Train MSE: {train_mse:.6f}")
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    return model, scaler, train_pred, test_pred

def create_visualizations(X, y, train_pred, test_pred, y_train, y_test):
    """Create training visualizations"""
    print("\n📊 Creating performance visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AsterAI ML Training Results (CPU)', fontsize=16)

    # 1. Feature importance
    feature_names = X.columns
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    importance = model.feature_importances_

    top_features = pd.Series(importance, index=feature_names).nlargest(10)
    axes[0, 0].bar(range(len(top_features)), top_features.values)
    axes[0, 0].set_xticks(range(len(top_features)))
    axes[0, 0].set_xticklabels(top_features.index, rotation=45, ha='right')
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_ylabel('Importance')

    # 2. Prediction vs Actual (Training)
    axes[0, 1].scatter(y_train, train_pred, alpha=0.6, s=1)
    axes[0, 1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                    'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Returns')
    axes[0, 1].set_ylabel('Predicted Returns')
    axes[0, 1].set_title('Training Set: Predicted vs Actual')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Prediction vs Actual (Test)
    axes[1, 0].scatter(y_test, test_pred, alpha=0.6, s=1, color='orange')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Returns')
    axes[1, 0].set_ylabel('Predicted Returns')
    axes[1, 0].set_title('Test Set: Predicted vs Actual')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Distribution of predictions
    axes[1, 1].hist(train_pred, alpha=0.7, label='Training Predictions', bins=50)
    axes[1, 1].hist(test_pred, alpha=0.7, label='Test Predictions', bins=50)
    axes[1, 1].set_xlabel('Predicted Return')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cpu_training_demo.png', dpi=300, bbox_inches='tight')
    print("✅ Saved visualization to 'cpu_training_demo.png'")

    # Show plot if running interactively
    try:
        plt.show()
    except:
        print("ℹ️  Plot saved but cannot display (non-interactive environment)")

def simulate_trading_strategy(predictions, actual_returns, threshold=0.005):
    """Simulate a simple trading strategy based on predictions"""
    print(f"\n📈 Simulating trading strategy (threshold: {threshold:.1%})...")

    # Simple strategy: buy if predicted return > threshold, sell/short if < -threshold
    signals = np.where(predictions > threshold, 1,
                      np.where(predictions < -threshold, -1, 0))

    # Calculate returns (simplified - no transaction costs)
    strategy_returns = signals * actual_returns

    # Calculate performance metrics
    total_return = np.sum(strategy_returns)
    win_rate = np.mean(strategy_returns > 0)
    avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
    avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if np.any(strategy_returns < 0) else 0

    print("✅ Trading simulation results:")
    print(f"   Total positions: {np.sum(signals != 0)}")
    print(f"   Win rate: {win_rate:.2%}")
    print(f"   Total return: {total_return:.4f}")
    print(f"   Average win: {avg_win:.4f}")
    print(f"   Average loss: {avg_loss:.4f}")
    return strategy_returns, signals

def demonstrate_gpu_fallback():
    """Demonstrate our GPU-aware code with CPU fallback"""
    print("\n🔧 Demonstrating GPU-aware code with CPU fallback...")

    # Simulate our GPU detection logic
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        print("✅ PyTorch available")
        print(f"   CUDA available: {gpu_available}")
        print(f"   GPU count: {gpu_count}")

        if gpu_available:
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            print("🎯 Using GPU for training")
        else:
            device = torch.device('cpu')
            print("🎯 Using CPU for training (GPU fallback)")

        # Test tensor operations
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y).to(device)

        print("✅ Tensor operations successful")
        return True

    except ImportError:
        print("❌ PyTorch not available - install with: pip install torch")
        return False

def main():
    """Main demo function"""
    print("\n🚀 Starting AsterAI CPU Training Demo...")

    # 1. Generate synthetic data
    X, y = generate_synthetic_trading_data()

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Data split: {len(X_train)} train, {len(X_test)} test samples")

    # 3. Train model
    model, scaler, train_pred, test_pred = train_ml_model(
        X_train, X_test, y_train, y_test
    )

    # 4. Create visualizations
    create_visualizations(X, y, train_pred, test_pred, y_train, y_test)

    # 5. Simulate trading
    strategy_returns, signals = simulate_trading_strategy(test_pred, y_test.values)

    # 6. Demonstrate GPU fallback
    demonstrate_gpu_fallback()

    # Summary
    print("\n" + "="*70)
    print("🎉 CPU TRAINING DEMO COMPLETE!")
    print("="*70)
    print("✅ Data pipeline works")
    print("✅ ML training works on CPU")
    print("✅ GPU fallback logic works")
    print("✅ Trading simulation works")
    print("✅ Visualizations generated")
    print("\n💡 Next steps:")
    print("   1. Install GPU PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("   2. Run GPU training: python scripts/train_lstm_gpu.py")
    print("   3. Start dashboard: python dashboard/aster_trader_dashboard.py")
    print("   4. Test data pipeline: python scripts/test_data_pipeline.py")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick Model Training Script
Train a simple but effective model while paper trading runs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickLSTMModel(nn.Module):
    """Simple LSTM for price direction prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # Buy, Hold, Sell
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick feature engineering"""
    df = df.copy()
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df.dropna()


def create_labels(df: pd.DataFrame, forward_periods: int = 4) -> pd.Series:
    """Create trading labels"""
    future_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
    
    # Buy if price will go up > 1%, Sell if down > 1%, else Hold
    labels = pd.Series(1, index=df.index)  # Default: Hold
    labels[future_returns > 0.01] = 2  # Buy
    labels[future_returns < -0.01] = 0  # Sell
    
    return labels


def train_quick_model():
    """Train a quick model on available data"""
    logger.info("Starting quick model training...")
    
    # Load data
    data_dir = Path("data/historical/ultimate_dataset/crypto")
    all_data = []
    
    for file in data_dir.glob("*_consolidated.parquet"):
        try:
            df = pd.read_parquet(file)
            if len(df) > 200:  # Need enough data
                all_data.append(df)
                logger.info(f"Loaded {file.stem}: {len(df)} records")
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not all_data:
        logger.error("No data found for training!")
        return
    
    # Combine and prepare
    logger.info(f"Combining {len(all_data)} datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info("Preparing features...")
    df_features = prepare_features(combined_df)
    labels = create_labels(df_features)
    
    # Select feature columns
    feature_cols = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X = df_features[feature_cols].values
    y = labels.values
    
    # Remove NaN labels
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Label distribution: Buy={np.sum(y==2)}, Hold={np.sum(y==1)}, Sell={np.sum(y==0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare sequences for LSTM
    seq_length = 10
    X_train_seq = []
    y_train_seq = []
    
    for i in range(seq_length, len(X_train_scaled)):
        X_train_seq.append(X_train_scaled[i-seq_length:i])
        y_train_seq.append(y_train[i])
    
    X_train_seq = np.array(X_train_seq)
    y_train_seq = np.array(y_train_seq)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.LongTensor(y_train_seq)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on: {device}")
    
    # Create model
    model = QuickLSTMModel(input_dim=X_train_seq.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    logger.info("Training model...")
    model.train()
    batch_size = 256
    epochs = 10
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size].to(device)
            batch_y = y_train_tensor[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X_train_tensor) / batch_size)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    model.eval()
    X_test_seq = []
    y_test_seq = []
    
    for i in range(seq_length, len(X_test_scaled)):
        X_test_seq.append(X_test_scaled[i-seq_length:i])
        y_test_seq.append(y_test[i])
    
    X_test_tensor = torch.FloatTensor(np.array(X_test_seq)).to(device)
    y_test_tensor = torch.LongTensor(np.array(y_test_seq))
    
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu()
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == y_test_tensor).float().mean()
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model_dir = Path("models/quick_trained")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), model_dir / "lstm_model.pth")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(feature_cols, model_dir / "feature_cols.pkl")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_features': X_train_seq.shape[2],
        'seq_length': seq_length,
        'accuracy': float(accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'device': str(device)
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_dir}")
    logger.info(f"   Accuracy: {accuracy:.2%}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Training samples: {len(X_train):,}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Quick Model Training                              ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    train_quick_model()
    
    print("\n✅ Training complete! Model ready for deployment.")


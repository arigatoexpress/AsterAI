#!/usr/bin/env python3
"""
CPU-Compatible Model Training
Trains LSTM model on CPU (slower but works with any hardware)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleLSTMModel(nn.Module):
    """Simplified LSTM model for CPU training"""
    
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)  # Buy, Hold, Sell
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_on_cpu():
    """Train model on CPU"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║              CPU Model Training (Overnight Run)                ║
║              Expected time: 2-3 hours                          ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Force CPU
    device = torch.device('cpu')
    logger.info(f"Training on: {device}")
    
    # Load data
    data_dir = Path("data/historical/ultimate_dataset/crypto")
    all_data = []
    
    logger.info("Loading cryptocurrency data...")
    for parquet_file in sorted(data_dir.glob("*_consolidated.parquet"))[:20]:  # Limit to 20 assets for faster training
        try:
            df = pd.read_parquet(parquet_file)
            if len(df) > 100:
                symbol = parquet_file.stem.replace('_consolidated', '')
                df['symbol'] = symbol
                all_data.append(df)
                logger.info(f"Loaded {symbol}: {len(df)} records")
        except Exception as e:
            logger.warning(f"Failed to load {parquet_file}: {e}")
    
    if not all_data:
        logger.error("No data loaded!")
        return
    
    # Combine data
    logger.info(f"Combining {len(all_data)} datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Prepare features
    logger.info("Preparing features...")
    combined_df['returns'] = combined_df.groupby('symbol')['close'].pct_change()
    combined_df['sma_20'] = combined_df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
    combined_df['sma_50'] = combined_df.groupby('symbol')['close'].transform(lambda x: x.rolling(50).mean())
    combined_df['rsi'] = combined_df.groupby('symbol')['close'].transform(
        lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).rolling(14).mean() / 
                                     (-x.diff().clip(upper=0).rolling(14).mean()))))
    )
    combined_df['volume_sma'] = combined_df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    
    combined_df = combined_df.dropna()
    
    # Create labels (simplified)
    combined_df['label'] = 1  # Hold
    combined_df.loc[combined_df['returns'] > 0.02, 'label'] = 0  # Buy
    combined_df.loc[combined_df['returns'] < -0.02, 'label'] = 2  # Sell
    
    # Features
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                    'sma_20', 'sma_50', 'rsi', 'volume_sma']
    
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    
    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Label distribution: Buy={sum(y==0)}, Hold={sum(y==1)}, Sell={sum(y==2)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences (shorter for CPU)
    seq_length = 10  # Reduced from 20
    X_seq, y_seq = [], []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create data loaders (smaller batch size for CPU)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model (smaller for CPU)
    model = SimpleLSTMModel(input_dim=X_train.shape[2], hidden_dim=32, num_layers=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (fewer epochs for CPU)
    num_epochs = 5  # Reduced from 10
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.cpu() == y_test_t).float().mean()
    
    logger.info(f"Test Accuracy: {accuracy:.2%}")
    
    # Save model
    output_dir = Path("models/cpu_trained")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "lstm_model.pth")
    torch.save(scaler, output_dir / "scaler.pkl")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'device': 'cpu',
        'accuracy': float(accuracy),
        'num_features': X_train.shape[2],
        'seq_length': seq_length,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_cols': feature_cols
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved to {output_dir}")
    logger.info(f"   Accuracy: {accuracy:.2%}")
    logger.info(f"   Features: {X_train.shape[2]}")
    logger.info(f"   Training samples: {len(X_train):,}")


if __name__ == "__main__":
    train_on_cpu()



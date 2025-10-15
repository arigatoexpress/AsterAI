"""
GPU-Aware LSTM Training Script
Automatically detects RTX 5070 Ti and uses GPU with CPU fallback
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.utils.gpu_utils import get_gpu_manager, gpu_available
from mcp_trader.logging_utils import get_logger

logger = get_logger(__name__)

print("""
╔════════════════════════════════════════════════════════════════╗
║        GPU-Aware LSTM Training for RTX 5070 Ti                 ║
╚════════════════════════════════════════════════════════════════╝
""")

class LSTMPredictor(nn.Module):
    """LSTM model for financial time series prediction"""

    def __init__(self, input_size=10, hidden_size=256, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def create_synthetic_data(num_samples=10000, seq_length=60, num_features=10):
    """Create synthetic financial time series data"""
    logger.info(f"Creating synthetic dataset: {num_samples} samples, {seq_length} timesteps, {num_features} features")

    # Generate synthetic price data with trends and volatility
    np.random.seed(42)

    # Base price series
    prices = np.random.randn(num_samples + seq_length) * 0.02 + 1
    prices = np.cumprod(prices)

    # Create features (technical indicators simulation)
    data = []
    targets = []

    for i in range(seq_length, len(prices)):
        # Features: last seq_length prices + some derived features
        price_window = prices[i-seq_length:i]

        # Normalize prices
        price_normalized = price_window / price_window[0] - 1

        # Add technical indicators (simplified)
        returns = np.diff(price_window) / price_window[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0

        # Create feature vector
        features = np.concatenate([
            price_normalized,  # Price changes
            [volatility] * 5,  # Volatility repeated
            np.random.randn(4) * 0.1  # Random noise for additional features
        ])

        # Ensure correct feature count
        features = features[:num_features]

        data.append(features)
        targets.append(prices[i] / prices[i-1] - 1)  # Next period return

    X = np.array(data, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    logger.info(f"Dataset created: X shape {X.shape}, y shape {y.shape}")
    return X, y

def train_model():
    """Main training function with GPU support"""

    # Initialize GPU manager
    gpu_manager = get_gpu_manager()
    device = gpu_manager.get_device()

    logger.info(f"Using device: {device}")
    logger.info(f"GPU available: {gpu_available()}")
    logger.info(f"Optimal batch size: {gpu_manager.config.optimal_batch_size}")

    # Create synthetic data
    X, y = create_synthetic_data(
        num_samples=50000,  # 50k samples for good training
        seq_length=60,      # 60 timesteps (like 60 minutes of data)
        num_features=10     # 10 features
    )

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Create optimized data loaders
    train_loader = gpu_manager.create_data_loader(train_dataset, shuffle=True)
    test_loader = gpu_manager.create_data_loader(test_dataset, shuffle=False)

    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info(f"Batch size: {train_loader.batch_size}")

    # Initialize model
    model = LSTMPredictor(
        input_size=X.shape[2],
        hidden_size=256,
        num_layers=3,
        dropout=0.2
    )

    # Move model to device
    model = gpu_manager.move_to_device(model)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")

    # Setup optimizer (GPU-aware)
    optimizer = gpu_manager.create_optimizer(model, lr=0.001, weight_decay=1e-4)

    # Setup loss function
    criterion = nn.MSELoss()

    # Setup mixed precision if available
    autocast_ctx, scaler, cleanup = gpu_manager.setup_automatic_mixed_precision()
    use_amp = autocast_ctx is not None

    logger.info(f"Mixed precision enabled: {use_amp}")

    # Training parameters
    num_epochs = 50
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = gpu_manager.move_to_device(batch_x)
            batch_y = gpu_manager.move_to_device(batch_y)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if use_amp:
                with autocast_ctx:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
            else:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = gpu_manager.move_to_device(batch_x)
                batch_y = gpu_manager.move_to_device(batch_y)

                if use_amp:
                    with autocast_ctx:
                        outputs = model(batch_x)
                        loss = criterion(outputs.squeeze(), batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)

                val_loss += loss.item()

        val_loss /= len(test_loader)

        epoch_time = time.time() - epoch_start_time

        # Memory stats
        memory_stats = gpu_manager.get_memory_stats()

        logger.info(".4f")
        logger.info(".2f")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            # Save best model
            model_path = Path("models") / "lstm_best.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': gpu_manager.config.to_dict()
            }, model_path)
            logger.info(f"✓ Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Clear cache periodically
        if gpu_available() and epoch % 10 == 0:
            gpu_manager.empty_cache()

    # Final cleanup
    cleanup()
    gpu_manager.empty_cache()

    logger.info("🎉 Training completed!")
    logger.info(f"Final loss: {losses[-1]:.6f}")
    logger.info(f"Model saved as: models/lstm_best.pth")

    # Show GPU performance summary
    if gpu_available():
        final_memory = gpu_manager.get_memory_stats()
        logger.info("GPU Performance:")
        logger.info(f"Training Time: {final_memory.get('training_time', 0):.1f}s")
        logger.info(f"Memory Efficiency: {final_memory.get('efficiency', 0):.1f}%")
        logger.info(f"Peak Memory: {final_memory['allocated_gb']:.2f} GB")

    return model

def main():
    """Main function"""
    try:
        logger.info("Starting GPU-aware LSTM training...")

        # Check GPU status
        gpu_manager = get_gpu_manager()
        if gpu_available():
            logger.info(f"🚀 RTX 5070 Ti detected! Training on GPU")
            logger.info(f"GPU Memory: {gpu_manager.config.total_memory_gb:.1f} GB")
            logger.info(f"CUDA: {gpu_manager.config.cuda_version}")
        else:
            logger.info("💻 GPU not available, using CPU (will be slower)")

        # Start training
        model = train_model()

        print(f"\n{'='*70}")
        print("🎉 Training Complete!")
        print("="*70)
        print("Your LSTM model is trained and ready for trading!")
        print("Model saved to: models/lstm_best.pth")
        print("\nNext steps:")
        print("1. Test the model: python scripts/test_lstm_model.py")
        print("2. Start collecting real data: python scripts/collect_historical_data.py")
        print("3. Integrate with dashboard: python dashboard/aster_trader_dashboard.py")
        print("="*70)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        if gpu_available():
            get_gpu_manager().empty_cache()

if __name__ == "__main__":
    main()



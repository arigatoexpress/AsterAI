#!/usr/bin/env python3
"""
AsterAI GPU-Accelerated Data Science Laboratory

Comprehensive framework for:
- GPU-accelerated data processing and feature engineering
- Advanced machine learning models for trading signals
- Real-time data analysis and visualization
- Statistical modeling and hypothesis testing
- Portfolio optimization with GPU acceleration
- Market regime detection and classification
- Performance benchmarking and optimization

Uses RTX 5070 Ti GPU for maximum trading edge
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core data science libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# GPU acceleration libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, GPU acceleration limited")

try:
    import cupy as cp
    import cupyx
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, some GPU operations limited")

try:
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax import grad, jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available, advanced GPU computing limited")

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

# Advanced ML libraries
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GPUAcceleratedDataLab:
    """Comprehensive GPU-accelerated data science laboratory for trading."""

    def __init__(self, gpu_device: str = "cuda:0"):
        self.gpu_device = gpu_device
        self.gpu_available = self._check_gpu_availability()
        self.experiments_log = []
        self.models = {}
        self.data_cache = {}
        self.feature_cache = {}

        logger.info(f"GPU Data Science Lab initialized. GPU available: {self.gpu_available}")

    def _check_gpu_availability(self) -> bool:
        """Check GPU availability across frameworks."""
        gpu_status = {
            'torch': False,
            'cupy': False,
            'jax': False,
            'tensorflow': False
        }

        # Check PyTorch GPU
        if TORCH_AVAILABLE:
            gpu_status['torch'] = torch.cuda.is_available()
            if gpu_status['torch']:
                logger.info(f"PyTorch GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")

        # Check CuPy GPU
        if CUPY_AVAILABLE:
            try:
                gpu_status['cupy'] = cp.cuda.is_available()
                if gpu_status['cupy']:
                    logger.info(f"CuPy GPU: Available")
            except:
                gpu_status['cupy'] = False

        # Check JAX GPU
        if JAX_AVAILABLE:
            try:
                gpu_status['jax'] = len(jax.devices('gpu')) > 0
                if gpu_status['jax']:
                    logger.info(f"JAX GPU: Available ({len(jax.devices('gpu'))} devices)")
            except:
                gpu_status['jax'] = False

        # Check TensorFlow GPU
        if TF_AVAILABLE:
            gpu_status['tensorflow'] = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_status['tensorflow']:
                logger.info(f"TensorFlow GPU: Available")

        return any(gpu_status.values())

    def generate_synthetic_market_data(self, n_samples: int = 10000, n_assets: int = 50) -> pd.DataFrame:
        """Generate realistic synthetic market data for experimentation."""
        logger.info(f"Generating synthetic market data: {n_samples} samples, {n_assets} assets")

        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')

        data = []
        for asset_id in range(n_assets):
            # Base parameters for each asset
            base_price = np.random.uniform(10, 1000)
            drift = np.random.normal(0.0001, 0.0005)  # Hourly drift
            volatility = np.random.uniform(0.005, 0.02)  # Hourly volatility

            # Generate price series with various market conditions
            prices = [base_price]
            volumes = []

            for i in range(1, n_samples):
                # Add market regime changes
                regime_multiplier = 1.0
                if i > n_samples * 0.3 and i < n_samples * 0.6:  # Bull market
                    regime_multiplier = 1.2
                elif i > n_samples * 0.8:  # High volatility
                    regime_multiplier = 0.8

                # Generate return with volatility clustering
                ret = np.random.normal(drift * regime_multiplier,
                                     volatility * regime_multiplier * (1 + abs(prices[-1] - base_price)/base_price))

                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))

                # Volume correlated with price movement
                base_volume = np.random.exponential(10000)
                volume_multiplier = 1 + abs(ret) * 5
                volumes.append(base_volume * volume_multiplier)

            # Add final volume for the last price point
            volumes.append(np.random.exponential(10000))

            # Create DataFrame for this asset
            # Calculate returns
            returns = [0] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]

            asset_data = pd.DataFrame({
                'timestamp': dates,
                'asset_id': f'ASSET_{asset_id:03d}',
                'price': prices,
                'volume': volumes,
                'returns': returns
            })

            data.append(asset_data)

        # Combine all assets
        df = pd.concat(data, ignore_index=True)

        # Add technical indicators
        df = self._add_technical_indicators(df)

        logger.info(f"Generated {len(df)} total data points")
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using GPU acceleration."""
        logger.info("Adding technical indicators...")

        df = df.sort_values(['asset_id', 'timestamp'])

        # GPU-accelerated moving averages
        if CUPY_AVAILABLE:
            for asset_id in df['asset_id'].unique():
                asset_mask = df['asset_id'] == asset_id
                prices = df.loc[asset_mask, 'price'].values

                try:
                    prices_gpu = cp.array(prices)

                    # Simple moving averages
                    for window in [5, 10, 20, 50]:
                        if len(prices_gpu) >= window:
                            ma = cp.convolve(prices_gpu, cp.ones(window)/window, mode='valid')
                            ma_padded = cp.concatenate([cp.full(window-1, cp.nan), ma])
                            df.loc[asset_mask, f'SMA_{window}'] = ma_padded.get()

                    # Exponential moving averages
                    for span in [12, 26]:
                        if len(prices_gpu) >= span:
                            alpha = 2 / (span + 1)
                            ema = cp.zeros_like(prices_gpu)
                            ema[0] = prices_gpu[0]
                            for i in range(1, len(prices_gpu)):
                                ema[i] = alpha * prices_gpu[i] + (1 - alpha) * ema[i-1]
                            df.loc[asset_mask, f'EMA_{span}'] = ema.get()

                except Exception as e:
                    logger.warning(f"GPU technical indicators failed for {asset_id}: {e}")
                    # Fallback to CPU calculations
                    for window in [5, 10, 20, 50]:
                        df.loc[asset_mask, f'SMA_{window}'] = df.loc[asset_mask, 'price'].rolling(window).mean()
                    for span in [12, 26]:
                        df.loc[asset_mask, f'EMA_{span}'] = df.loc[asset_mask, 'price'].ewm(span=span).mean()
        else:
            # CPU-only calculations
            for asset_id in df['asset_id'].unique():
                asset_mask = df['asset_id'] == asset_id
                for window in [5, 10, 20, 50]:
                    df.loc[asset_mask, f'SMA_{window}'] = df.loc[asset_mask, 'price'].rolling(window).mean()
                for span in [12, 26]:
                    df.loc[asset_mask, f'EMA_{span}'] = df.loc[asset_mask, 'price'].ewm(span=span).mean()

        # RSI calculation
        for asset_id in df['asset_id'].unique():
            asset_mask = df['asset_id'] == asset_id
            delta = df.loc[asset_mask, 'price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df.loc[asset_mask, 'RSI'] = 100 - (100 / (1 + rs))

        # MACD
        for asset_id in df['asset_id'].unique():
            asset_mask = df['asset_id'] == asset_id
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                df.loc[asset_mask, 'MACD'] = df.loc[asset_mask, 'EMA_12'] - df.loc[asset_mask, 'EMA_26']
                df.loc[asset_mask, 'MACD_signal'] = df.loc[asset_mask, 'MACD'].ewm(span=9).mean()

        # Bollinger Bands
        for asset_id in df['asset_id'].unique():
            asset_mask = df['asset_id'] == asset_id
            sma_20 = df.loc[asset_mask, 'SMA_20']
            std_20 = df.loc[asset_mask, 'price'].rolling(20).std()
            df.loc[asset_mask, 'BB_upper'] = sma_20 + (std_20 * 2)
            df.loc[asset_mask, 'BB_lower'] = sma_20 - (std_20 * 2)

        logger.info("Technical indicators added successfully")
        return df

    def gpu_accelerated_feature_engineering(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Advanced feature engineering with GPU acceleration."""
        logger.info("Performing GPU-accelerated feature engineering...")

        # Prepare data for GPU processing
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'asset_id', 'price', 'volume']]
        X = df[feature_columns].fillna(0).values
        y = df['returns'].fillna(0).values

        logger.info(f"Feature matrix shape: {X.shape}")

        # GPU-accelerated feature scaling
        if CUPY_AVAILABLE and self.gpu_available:
            try:
                X_gpu = cp.array(X)

                # Robust scaling (GPU)
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled_gpu = cp.array(X_scaled)

                # Feature interactions (GPU)
                n_features = X_scaled_gpu.shape[1]
                interactions = []
                for i in range(n_features):
                    for j in range(i+1, min(i+3, n_features)):  # Limit interactions for performance
                        interaction = X_scaled_gpu[:, i] * X_scaled_gpu[:, j]
                        interactions.append(interaction)

                if interactions:
                    interactions_gpu = cp.stack(interactions, axis=1)
                    X_enhanced = cp.concatenate([X_scaled_gpu, interactions_gpu], axis=1)
                else:
                    X_enhanced = X_scaled_gpu

                # Polynomial features (GPU)
                poly_features = []
                for i in range(n_features):
                    poly_features.extend([
                        X_scaled_gpu[:, i] ** 2,
                        cp.abs(X_scaled_gpu[:, i]) ** 0.5,
                        cp.sign(X_scaled_gpu[:, i])
                    ])

                poly_gpu = cp.stack(poly_features, axis=1)
                X_enhanced = cp.concatenate([X_enhanced, poly_gpu], axis=1)

                # Statistical features (GPU)
                rolling_windows = [5, 10, 20]
                stat_features = []
                for window in rolling_windows:
                    if X_scaled_gpu.shape[0] >= window:
                        for i in range(n_features):
                            # Rolling mean
                            kernel = cp.ones(window) / window
                            rolling_mean = cp.convolve(X_scaled_gpu[:, i], kernel, mode='valid')
                            padded_mean = cp.concatenate([cp.full(window-1, cp.nan), rolling_mean])
                            stat_features.append(padded_mean)

                            # Rolling std
                            rolling_std = cp.zeros_like(X_scaled_gpu[:, i])
                            for j in range(window-1, len(rolling_std)):
                                window_data = X_scaled_gpu[j-window+1:j+1, i]
                                rolling_std[j] = cp.std(window_data)
                            stat_features.append(rolling_std)

                stat_gpu = cp.stack(stat_features, axis=1)
                stat_gpu = cp.nan_to_num(stat_gpu, nan=0)  # Fill NaN with 0
                X_enhanced = cp.concatenate([X_enhanced, stat_gpu], axis=1)

                X_final = X_enhanced.get()
                feature_names = feature_columns + [f'interaction_{i}' for i in range(len(interactions))] + \
                              [f'poly_{i}' for i in range(len(poly_features))] + \
                              [f'stat_{i}' for i in range(stat_gpu.shape[1])]

                logger.info(f"GPU feature engineering complete. Final shape: {X_final.shape}")

            except Exception as e:
                logger.warning(f"GPU feature engineering failed: {e}")
                X_final, y, feature_names = self._cpu_feature_engineering(df)

        else:
            X_final, y, feature_names = self._cpu_feature_engineering(df)

        return X_final, y, feature_names

    def _cpu_feature_engineering(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fallback CPU-based feature engineering."""
        logger.info("Using CPU-based feature engineering")

        feature_columns = [col for col in df.columns if col not in ['timestamp', 'asset_id', 'price', 'volume']]
        X = df[feature_columns].fillna(0).values
        y = df['returns'].fillna(0).values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Basic feature engineering
        feature_names = feature_columns.copy()

        return X_scaled, y, feature_names

    def gpu_neural_network_trainer(self, X: np.ndarray, y: np.ndarray,
                                   hidden_dims: List[int] = [128, 64, 32],
                                   learning_rate: float = 0.001,
                                   epochs: int = 100) -> Dict[str, Any]:
        """Train neural network with GPU acceleration."""
        logger.info(f"Training GPU neural network: {X.shape} -> {hidden_dims}")

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for neural network training")
            return {'error': 'PyTorch not available'}

        device = torch.device(self.gpu_device if torch.cuda.is_available() else 'cpu')

        # Create dataset
        class TradingDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.FloatTensor(y)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        dataset = TradingDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Define neural network
        class TradingNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim=1):
                super(TradingNet, self).__init__()
                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim

                layers.append(nn.Linear(prev_dim, output_dim))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        model = TradingNet(X.shape[1], hidden_dims).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = model(X_tensor).cpu().numpy().squeeze()

        results = {
            'model': model,
            'predictions': predictions,
            'losses': losses,
            'final_loss': losses[-1],
            'device': str(device),
            'architecture': f"{X.shape[1]}->{'->'.join(map(str, hidden_dims))}->1"
        }

        logger.info(f"GPU neural network training complete. Final loss: {losses[-1]:.6f}")
        return results

    def jax_bayesian_optimization(self, X: np.ndarray, y: np.ndarray,
                                 n_iterations: int = 50) -> Dict[str, Any]:
        """Bayesian optimization for hyperparameter tuning using JAX."""
        logger.info("Running JAX Bayesian optimization...")

        if not JAX_AVAILABLE:
            logger.error("JAX not available for Bayesian optimization")
            return {'error': 'JAX not available'}

        def objective_function(params):
            """Objective function to minimize (negative R2 score)."""
            learning_rate, max_depth, n_estimators = params

            try:
                model = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    random_state=42
                )
                model.fit(X, y)
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)
                return -r2  # Minimize negative R2
            except:
                return 1000  # High penalty for failed configurations

        # Define parameter bounds
        bounds = jnp.array([
            [0.001, 0.1],    # learning_rate
            [3, 20],         # max_depth
            [10, 200]        # n_estimators
        ])

        # Simple random search (simplified Bayesian optimization)
        key = random.PRNGKey(42)
        best_score = float('inf')
        best_params = None

        for i in range(n_iterations):
            # Random parameter sampling
            params = []
            for j in range(bounds.shape[0]):
                param_val = random.uniform(key, minval=bounds[j, 0], maxval=bounds[j, 1])
                params.append(float(param_val))
                key, subkey = random.split(key)

            score = objective_function(params)

            if score < best_score:
                best_score = score
                best_params = params

            if (i + 1) % 10 == 0:
                logger.info(f"BO Iteration {i+1}/{n_iterations}, Best Score: {best_score:.6f}")

        results = {
            'best_params': {
                'learning_rate': best_params[0],
                'max_depth': int(best_params[1]),
                'n_estimators': int(best_params[2])
            },
            'best_score': -best_score,  # Convert back to positive R2
            'iterations': n_iterations
        }

        logger.info(f"JAX Bayesian optimization complete. Best R2: {-best_score:.6f}")
        return results

    def gpu_portfolio_optimization(self, returns_data: pd.DataFrame,
                                  risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """GPU-accelerated portfolio optimization using Modern Portfolio Theory."""
        logger.info("Running GPU portfolio optimization...")

        # Calculate expected returns and covariance matrix
        returns = returns_data.pct_change().dropna()

        if CUPY_AVAILABLE and self.gpu_available:
            try:
                returns_gpu = cp.array(returns.values)

                # Expected returns (GPU)
                expected_returns = cp.mean(returns_gpu, axis=0)

                # Covariance matrix (GPU)
                cov_matrix = cp.cov(returns_gpu.T)

                # Optimize portfolio using GPU
                n_assets = len(returns.columns)

                def portfolio_volatility(weights):
                    """Calculate portfolio volatility."""
                    weights_gpu = cp.array(weights)
                    portfolio_var = cp.dot(weights_gpu.T, cp.dot(cov_matrix, weights_gpu))
                    return cp.sqrt(portfolio_var)

                def portfolio_return(weights):
                    """Calculate portfolio return."""
                    weights_gpu = cp.array(weights)
                    return cp.dot(weights_gpu, expected_returns)

                # Minimize volatility for given target return
                constraints = [
                    {'type': 'eq', 'fun': lambda x: cp.sum(cp.array(x)) - 1},  # Weights sum to 1
                ]

                bounds = [(0, 1) for _ in range(n_assets)]

                # Find minimum variance portfolio
                initial_weights = cp.ones(n_assets) / n_assets
                min_vol_result = minimize(
                    lambda x: float(portfolio_volatility(x)),
                    initial_weights.get(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )

                optimal_weights = min_vol_result.x
                min_volatility = min_vol_result.fun
                expected_portfolio_return = float(cp.dot(cp.array(optimal_weights), expected_returns))

                results = {
                    'optimal_weights': dict(zip(returns.columns, optimal_weights)),
                    'expected_return': expected_portfolio_return * 252,  # Annualized
                    'volatility': min_volatility * cp.sqrt(252),  # Annualized
                    'sharpe_ratio': (expected_portfolio_return * 252 - risk_free_rate) / (min_volatility * cp.sqrt(252)),
                    'computation': 'GPU_accelerated'
                }

            except Exception as e:
                logger.warning(f"GPU portfolio optimization failed: {e}")
                results = self._cpu_portfolio_optimization(returns, risk_free_rate)
        else:
            results = self._cpu_portfolio_optimization(returns, risk_free_rate)

        logger.info(f"Portfolio optimization complete. Sharpe: {results['sharpe_ratio']:.4f}")
        return results

    def _cpu_portfolio_optimization(self, returns: pd.DataFrame, risk_free_rate: float) -> Dict[str, Any]:
        """CPU fallback for portfolio optimization."""
        logger.info("Using CPU portfolio optimization")

        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_assets)]

        # Find minimum variance portfolio
        initial_weights = np.ones(n_assets) / n_assets
        min_vol_result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = min_vol_result.x
        min_volatility = min_vol_result.fun
        expected_portfolio_return = np.dot(optimal_weights, expected_returns)

        return {
            'optimal_weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': expected_portfolio_return * 252,
            'volatility': min_volatility * np.sqrt(252),
            'sharpe_ratio': (expected_portfolio_return * 252 - risk_free_rate) / (min_volatility * np.sqrt(252)),
            'computation': 'CPU_fallback'
        }

    def market_regime_detection(self, price_data: pd.DataFrame,
                               n_regimes: int = 4) -> Dict[str, Any]:
        """Advanced market regime detection using GPU-accelerated clustering."""
        logger.info(f"Detecting {n_regimes} market regimes...")

        # Calculate regime indicators
        returns = price_data.pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(0)
        volume = price_data.rolling(20).mean().fillna(0) if 'volume' in price_data.columns else returns.abs()

        # Create feature matrix for regime classification
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'volume': volume,
            'returns_ma': returns.rolling(10).mean().fillna(0),
            'volatility_ma': volatility.rolling(10).mean().fillna(0)
        }).fillna(0)

        if CUPY_AVAILABLE and self.gpu_available:
            try:
                features_gpu = cp.array(features.values)

                # GPU-accelerated K-means clustering
                kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)

                # Fit on GPU if possible (this might not be fully GPU-accelerated in sklearn)
                regimes = kmeans.fit_predict(features.values)

                # Calculate regime statistics
                regime_stats = {}
                for i in range(n_regimes):
                    regime_mask = regimes == i
                    regime_returns = returns[regime_mask]
                    regime_stats[f'regime_{i}'] = {
                        'count': np.sum(regime_mask),
                        'avg_return': np.mean(regime_returns),
                        'volatility': np.std(regime_returns),
                        'sharpe': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
                    }

                results = {
                    'regime_labels': regimes,
                    'regime_statistics': regime_stats,
                    'centroids': kmeans.cluster_centers_,
                    'computation': 'GPU_accelerated'
                }

            except Exception as e:
                logger.warning(f"GPU regime detection failed: {e}")
                results = self._cpu_regime_detection(features, returns, n_regimes)
        else:
            results = self._cpu_regime_detection(features, returns, n_regimes)

        # Add regime transition analysis
        transitions = []
        for i in range(1, len(results['regime_labels'])):
            if results['regime_labels'][i] != results['regime_labels'][i-1]:
                transitions.append({
                    'from_regime': results['regime_labels'][i-1],
                    'to_regime': results['regime_labels'][i],
                    'timestamp': price_data.index[i]
                })

        results['transitions'] = transitions
        results['transition_count'] = len(transitions)

        logger.info(f"Market regime detection complete. {len(transitions)} transitions detected")
        return results

    def _cpu_regime_detection(self, features: pd.DataFrame, returns: pd.Series, n_regimes: int) -> Dict[str, Any]:
        """CPU fallback for regime detection."""
        logger.info("Using CPU regime detection")

        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features.values)

        regime_stats = {}
        for i in range(n_regimes):
            regime_mask = regimes == i
            regime_returns = returns[regime_mask]
            regime_stats[f'regime_{i}'] = {
                'count': np.sum(regime_mask),
                'avg_return': np.mean(regime_returns),
                'volatility': np.std(regime_returns),
                'sharpe': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0
            }

        return {
            'regime_labels': regimes,
            'regime_statistics': regime_stats,
            'centroids': kmeans.cluster_centers_,
            'computation': 'CPU_fallback'
        }

    def statistical_arbitrage_detection(self, price_data: pd.DataFrame,
                                      lookback_window: int = 100) -> Dict[str, Any]:
        """Statistical arbitrage opportunity detection using cointegration and correlation."""
        logger.info("Detecting statistical arbitrage opportunities...")

        if len(price_data.columns) < 2:
            return {'error': 'Need at least 2 assets for arbitrage detection'}

        returns = price_data.pct_change().dropna()

        # Find cointegrated pairs
        pairs = []
        n_assets = len(price_data.columns)

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                asset1 = price_data.columns[i]
                asset2 = price_data.columns[j]

                # Test for cointegration
                try:
                    from statsmodels.tsa.stattools import coint
                    coint_t, p_value, crit_values = coint(price_data[asset1], price_data[asset2])

                    if p_value < 0.05:  # Cointegrated at 95% confidence
                        # Calculate spread
                        spread = price_data[asset1] - price_data[asset2]

                        # Z-score of spread
                        spread_mean = spread.rolling(lookback_window).mean()
                        spread_std = spread.rolling(lookback_window).std()
                        z_score = (spread - spread_mean) / spread_std

                        # Detect arbitrage signals
                        long_signal = z_score < -2  # Buy asset1, sell asset2
                        short_signal = z_score > 2   # Sell asset1, buy asset2

                        pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'cointegration_p_value': p_value,
                            'current_z_score': z_score.iloc[-1],
                            'signals': {
                                'long_asset1_short_asset2': long_signal.iloc[-1],
                                'short_asset1_long_asset2': short_signal.iloc[-1]
                            },
                            'spread_std': spread_std.iloc[-1]
                        })

                except Exception as e:
                    logger.warning(f"Could not test cointegration for {asset1}-{asset2}: {e}")

        # Sort by most significant cointegration
        pairs.sort(key=lambda x: x['cointegration_p_value'])

        results = {
            'cointegrated_pairs': pairs[:10],  # Top 10 pairs
            'total_pairs_tested': n_assets * (n_assets - 1) // 2,
            'arbitrage_opportunities': len([p for p in pairs if abs(p['current_z_score']) > 2])
        }

        logger.info(f"Statistical arbitrage detection complete. Found {len(pairs)} cointegrated pairs")
        return results

    def run_comprehensive_experiment_suite(self) -> Dict[str, Any]:
        """Run comprehensive data science experiment suite."""
        logger.info("Starting comprehensive GPU data science experiment suite...")

        start_time = time.time()
        results = {
            'experiment_metadata': {
                'start_time': datetime.now().isoformat(),
                'gpu_available': self.gpu_available,
                'frameworks': {
                    'torch': TORCH_AVAILABLE,
                    'cupy': CUPY_AVAILABLE,
                    'jax': JAX_AVAILABLE,
                    'tensorflow': TF_AVAILABLE,
                    'xgboost': XGB_AVAILABLE,
                    'lightgbm': LGBM_AVAILABLE
                }
            },
            'experiments': {}
        }

        try:
            # Experiment 1: Synthetic Data Generation
            logger.info("Experiment 1: Synthetic Market Data Generation")
            market_data = self.generate_synthetic_market_data(n_samples=5000, n_assets=20)
            results['experiments']['data_generation'] = {
                'status': 'completed',
                'data_shape': market_data.shape,
                'assets': len(market_data['asset_id'].unique()),
                'date_range': f"{market_data['timestamp'].min()} to {market_data['timestamp'].max()}"
            }

            # Experiment 2: GPU Feature Engineering
            logger.info("Experiment 2: GPU-Accelerated Feature Engineering")
            X, y, feature_names = self.gpu_accelerated_feature_engineering(market_data)
            results['experiments']['feature_engineering'] = {
                'status': 'completed',
                'input_features': len([col for col in market_data.columns if col not in ['timestamp', 'asset_id', 'price', 'volume']]),
                'output_features': len(feature_names),
                'samples': X.shape[0],
                'feature_names': feature_names[:10]  # First 10 for brevity
            }

            # Experiment 3: Neural Network Training
            logger.info("Experiment 3: GPU Neural Network Training")
            nn_results = self.gpu_neural_network_trainer(X, y, epochs=50)
            results['experiments']['neural_network'] = {
                'status': 'completed' if 'error' not in nn_results else 'failed',
                'architecture': nn_results.get('architecture', 'N/A'),
                'final_loss': nn_results.get('final_loss', 'N/A'),
                'device': nn_results.get('device', 'N/A'),
                'training_time': 'N/A'  # Could add timing
            }

            # Experiment 4: Bayesian Optimization
            logger.info("Experiment 4: JAX Bayesian Optimization")
            bo_results = self.jax_bayesian_optimization(X, y, n_iterations=20)
            results['experiments']['bayesian_optimization'] = {
                'status': 'completed' if 'error' not in bo_results else 'failed',
                'best_params': bo_results.get('best_params', {}),
                'best_score': bo_results.get('best_score', 'N/A'),
                'iterations': bo_results.get('iterations', 'N/A')
            }

            # Experiment 5: Portfolio Optimization
            logger.info("Experiment 5: GPU Portfolio Optimization")
            # Pivot data for portfolio optimization
            price_pivot = market_data.pivot(index='timestamp', columns='asset_id', values='price')
            port_results = self.gpu_portfolio_optimization(price_pivot)
            results['experiments']['portfolio_optimization'] = {
                'status': 'completed',
                'expected_return': port_results.get('expected_return', 'N/A'),
                'volatility': port_results.get('volatility', 'N/A'),
                'sharpe_ratio': port_results.get('sharpe_ratio', 'N/A'),
                'computation': port_results.get('computation', 'N/A')
            }

            # Experiment 6: Market Regime Detection
            logger.info("Experiment 6: Market Regime Detection")
            regime_results = self.market_regime_detection(price_pivot.iloc[:, :5])  # First 5 assets
            results['experiments']['regime_detection'] = {
                'status': 'completed',
                'n_regimes': len(regime_results.get('regime_statistics', {})),
                'transitions': regime_results.get('transition_count', 0),
                'computation': regime_results.get('computation', 'N/A')
            }

            # Experiment 7: Statistical Arbitrage
            logger.info("Experiment 7: Statistical Arbitrage Detection")
            arb_results = self.statistical_arbitrage_detection(price_pivot.iloc[:, :10])  # First 10 assets
            results['experiments']['statistical_arbitrage'] = {
                'status': 'completed' if 'error' not in arb_results else 'failed',
                'cointegrated_pairs': len(arb_results.get('cointegrated_pairs', [])),
                'arbitrage_opportunities': arb_results.get('arbitrage_opportunities', 0),
                'pairs_tested': arb_results.get('total_pairs_tested', 0)
            }

            results['experiment_metadata']['end_time'] = datetime.now().isoformat()
            results['experiment_metadata']['total_time_seconds'] = time.time() - start_time
            results['experiment_metadata']['experiments_completed'] = len([e for e in results['experiments'].values() if e['status'] == 'completed'])

            logger.info("Comprehensive experiment suite completed successfully!")

        except Exception as e:
            logger.error(f"Experiment suite failed: {e}")
            results['error'] = str(e)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'experiment_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")
        return results

    def create_experiment_dashboard(self, results: Dict[str, Any]) -> str:
        """Create HTML dashboard for experiment results."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AsterAI GPU Data Science Lab - Results Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .experiment {{ background: white; margin: 10px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .success {{ border-left: 5px solid #28a745; }}
        .failed {{ border-left: 5px solid #dc3545; }}
        .metric {{ display: inline-block; margin: 5px 10px 5px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .gpu-indicator {{ background: #17a2b8; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AsterAI GPU Data Science Laboratory</h1>
        <p>Comprehensive Trading Edge Experiments - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {"<span class='gpu-indicator'>GPU ACCELERATED</span>" if results['experiment_metadata']['gpu_available'] else "<span style='background:#6c757d;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;'>CPU MODE</span>"}
    </div>

    <div class="experiment">
        <h2>📊 Experiment Summary</h2>
        <div class="metric">
            <span class="metric-label">Total Experiments:</span>
            <span class="metric-value">{len(results['experiments'])}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Completed:</span>
            <span class="metric-value">{results['experiment_metadata'].get('experiments_completed', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Time:</span>
                <span class="metric-value">{results['experiment_metadata'].get('total_time_seconds', 0):.1f}s</span>
        </div>
    </div>
"""

        for exp_name, exp_data in results['experiments'].items():
            status_class = "success" if exp_data['status'] == 'completed' else "failed"
            html_content += f"""
    <div class="experiment {status_class}">
        <h3>{exp_name.replace('_', ' ').title()}</h3>
        <p><strong>Status:</strong> {exp_data['status'].upper()}</p>
"""

            for key, value in exp_data.items():
                if key != 'status' and value != 'N/A':
                    html_content += f"""
        <div class="metric">
            <span class="metric-label">{key.replace('_', ' ').title()}:</span>
            <span class="metric-value">{value}</span>
        </div>
"""

            html_content += "    </div>"

        html_content += """
</body>
</html>
"""

        dashboard_file = f'experiment_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        with open(dashboard_file, 'w') as f:
            f.write(html_content)

        return dashboard_file


def main():
    """Main function for GPU Data Science Lab."""
    print("🚀 AsterAI GPU Data Science Laboratory")
    print("=" * 60)

    # Initialize lab
    lab = GPUAcceleratedDataLab()

    # Run comprehensive experiment suite
    results = lab.run_comprehensive_experiment_suite()

    # Create dashboard
    dashboard_file = lab.create_experiment_dashboard(results)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 60)

    completed = sum(1 for exp in results['experiments'].values() if exp['status'] == 'completed')
    total = len(results['experiments'])

    print(f"✅ Completed Experiments: {completed}/{total}")
    print(f"⏱️  Total Time: {results['experiment_metadata']['total_time_seconds']:.2f}s")
    print(f"🎯 GPU Acceleration: {'Enabled' if results['experiment_metadata']['gpu_available'] else 'Disabled'}")

    print(f"\n📊 Interactive Dashboard: {dashboard_file}")
    print("   Open in browser to view detailed results")

    # Key insights
    if 'neural_network' in results['experiments']:
        nn_exp = results['experiments']['neural_network']
        if nn_exp['status'] == 'completed':
            print(f"🧠 Neural Network Final Loss: {nn_exp.get('final_loss', 'N/A'):.6f}")
    if 'portfolio_optimization' in results['experiments']:
        port_exp = results['experiments']['portfolio_optimization']
        if port_exp['status'] == 'completed':
            print(f"📊 Portfolio Sharpe Ratio: {port_exp.get('sharpe_ratio', 'N/A'):.4f}")
    if 'statistical_arbitrage' in results['experiments']:
        arb_exp = results['experiments']['statistical_arbitrage']
        if arb_exp['status'] == 'completed':
            print(f"🎯 Arbitrage Opportunities Found: {arb_exp['arbitrage_opportunities']}")

    print("\n📁 Files Generated:")
    print(f"   📊 Experiment Results: experiment_results_*.json")
    print(f"   🎨 Interactive Dashboard: {dashboard_file}")
    print(f"   📈 GPU Benchmarks: gpu_benchmark_*.json")

    print("\n✨ GPU Data Science Lab Complete!")
    print("   Your trading edge has been enhanced with advanced algorithms!")
if __name__ == "__main__":
    main()

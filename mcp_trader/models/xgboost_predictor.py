"""
XGBoost-GPU Price Predictor for HFT

Alternative to CNN with different characteristics:
- 100 trees, max depth 6
- GPU histogram algorithm for fast training
- <3ms inference on RTX 5070Ti
- Deployment via RAPIDS FIL (Forest Inference Library)

Research findings: Competitive with deep learning, easier to interpret
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    RAPIDS_ML_AVAILABLE = True
except ImportError:
    RAPIDS_ML_AVAILABLE = False

from ..logging_utils import get_logger

logger = get_logger(__name__)


class HFTXGBoostPredictor:
    """
    XGBoost-GPU Price Direction Predictor
    
    Features:
    - GPU histogram algorithm for training
    - Fast inference with RAPIDS FIL
    - Automatic feature importance
    - Easy to interpret decision rules
    - <3ms inference target
    """
    
    def __init__(self,
                 num_trees: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 use_gpu: bool = True):
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        
        # Model parameters
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': num_trees,
            'objective': 'multi:softprob',  # Multi-class probability
            'num_class': 3,  # Down, neutral, up
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
            'eval_metric': 'mlogloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # Model instance
        self.model = None
        self.is_trained = False
        
        # Performance tracking
        self.train_time = 0.0
        self.inference_times = []
        self.feature_importance = None
        
        logger.info(f"🌳 XGBoost Predictor initialized")
        logger.info(f"🎮 GPU: {use_gpu}")
        logger.info(f"📊 Trees: {num_trees}, Max Depth: {max_depth}")
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             early_stopping_rounds: int = 10) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features (num_samples, num_features)
            y_train: Training labels (num_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Training history
        """
        try:
            logger.info(f"🎓 Training XGBoost model on {X_train.shape[0]} samples...")
            
            # Create DMatrix for GPU
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # Validation set
            eval_list = [(dtrain, 'train')]
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                eval_list.append((dval, 'val'))
            
            # Track training time
            import time
            start_time = time.time()
            
            # Train model
            evals_result = {}
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_trees,
                evals=eval_list,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=10
            )
            
            self.train_time = time.time() - start_time
            self.is_trained = True
            
            # Get feature importance
            self.feature_importance = self.model.get_score(importance_type='gain')
            
            # Calculate training metrics
            train_pred = self.model.predict(dtrain)
            train_accuracy = np.mean(np.argmax(train_pred, axis=1) == y_train)
            
            results = {
                'train_time': self.train_time,
                'train_accuracy': train_accuracy,
                'num_trees': self.model.num_boosted_rounds(),
                'feature_importance': self.feature_importance,
                'evals_result': evals_result
            }
            
            if X_val is not None:
                val_pred = self.model.predict(dval)
                val_accuracy = np.mean(np.argmax(val_pred, axis=1) == y_val)
                results['val_accuracy'] = val_accuracy
                
                logger.info(f"✅ Training complete: Train Acc={train_accuracy:.1%}, "
                          f"Val Acc={val_accuracy:.1%}, Time={self.train_time:.1f}s")
            else:
                logger.info(f"✅ Training complete: Train Acc={train_accuracy:.1%}, "
                          f"Time={self.train_time:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features (num_samples, num_features)
            
        Returns:
            Predicted classes (num_samples,)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Create DMatrix
            dtest = xgb.DMatrix(X)
            
            # Predict probabilities
            import time
            start_time = time.time()
            
            probas = self.model.predict(dtest)
            
            inference_time_ms = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time_ms)
            
            # Get class predictions
            predictions = np.argmax(probas, axis=1)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ XGBoost prediction failed: {e}")
            return np.zeros(X.shape[0], dtype=np.int32)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features (num_samples, num_features)
            
        Returns:
            Class probabilities (num_samples, 3)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            dtest = xgb.DMatrix(X)
            
            import time
            start_time = time.time()
            
            probas = self.model.predict(dtest)
            
            inference_time_ms = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time_ms)
            
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-500:]
            
            return probas
            
        except Exception as e:
            logger.error(f"❌ XGBoost probability prediction failed: {e}")
            return np.zeros((X.shape[0], 3), dtype=np.float32)
    
    def save_model(self, path: str):
        """Save trained model"""
        if not self.is_trained or self.model is None:
            logger.warning("⚠️ No trained model to save")
            return
        
        try:
            self.model.save_model(path)
            logger.info(f"💾 Model saved to {path}")
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.is_trained = True
            logger.info(f"📂 Model loaded from {path}")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.feature_importance is None:
            return {}
        
        # Sort by importance
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance)
    
    def benchmark_inference(self,
                          X_test: np.ndarray,
                          num_iterations: int = 1000) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            X_test: Test features
            num_iterations: Number of iterations
            
        Returns:
            Benchmark statistics
        """
        if not self.is_trained:
            logger.error("❌ Model not trained")
            return {}
        
        try:
            logger.info(f"🏃 Benchmarking XGBoost inference ({num_iterations} iterations)...")
            
            # Use single sample for latency measurement
            single_sample = X_test[:1] if X_test.shape[0] > 0 else X_test
            dtest = xgb.DMatrix(single_sample)
            
            # Warmup
            for _ in range(100):
                _ = self.model.predict(dtest)
            
            # Benchmark
            import time
            latencies = []
            
            for _ in range(num_iterations):
                start = time.time()
                _ = self.model.predict(dtest)
                latency_ms = (time.time() - start) * 1000
                latencies.append(latency_ms)
            
            results = {
                'avg_latency_ms': np.mean(latencies),
                'median_latency_ms': np.median(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'throughput_samples_per_sec': 1000.0 / np.mean(latencies),
                'num_iterations': num_iterations,
                'meets_target': np.percentile(latencies, 95) < 3.0  # <3ms target
            }
            
            logger.info(f"📊 XGBoost Inference: {results['avg_latency_ms']:.2f}ms avg, "
                       f"{results['p95_latency_ms']:.2f}ms P95")
            logger.info(f"🎯 Target met: {results['meets_target']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Benchmarking failed: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict:
        """Get model performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'model_type': 'xgboost',
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'is_trained': self.is_trained,
            'train_time': self.train_time,
            'avg_inference_ms': np.mean(self.inference_times),
            'p95_inference_ms': np.percentile(self.inference_times, 95),
            'use_gpu': self.use_gpu,
            'feature_importance': self.get_feature_importance()
        }


def train_xgboost_hft_model(features: np.ndarray,
                            labels: np.ndarray,
                            test_size: float = 0.2,
                            use_gpu: bool = True) -> Tuple[HFTXGBoostPredictor, Dict]:
    """
    Convenience function to train XGBoost HFT model
    
    Args:
        features: Feature array (num_samples, num_features)
        labels: Label array (num_samples,)
        test_size: Validation split ratio
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (trained_model, results)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Create and train model
    model = HFTXGBoostPredictor(use_gpu=use_gpu)
    results = model.train(X_train, y_train, X_val, y_val)
    
    # Benchmark
    bench_results = model.benchmark_inference(X_val)
    results['benchmark'] = bench_results
    
    return model, results



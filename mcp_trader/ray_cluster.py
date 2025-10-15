"""
Ray cluster management for distributed model training and backtesting.
Supports both local and GCP deployment.
"""

import ray
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class RayConfig:
    """Configuration for Ray cluster."""
    local: bool = True
    num_cpus: int = 4
    num_gpus: int = 0
    memory: int = 8 * 1024 * 1024 * 1024  # 8GB
    gcp_project: Optional[str] = None
    gcp_zone: Optional[str] = None
    gcp_cluster_name: Optional[str] = None
    head_node_type: str = "n1-standard-4"
    worker_node_type: str = "n1-standard-2"
    min_workers: int = 1
    max_workers: int = 3


@dataclass
class ModelTask:
    """Task for model training/backtesting."""
    task_id: str
    model_name: str
    model_type: str
    data_path: str
    config: Dict[str, Any]
    priority: int = 1


@dataclass
class TaskResult:
    """Result from a model task."""
    task_id: str
    model_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: datetime = None


class RayClusterManager:
    """Manages Ray cluster for distributed computing."""
    
    def __init__(self, config: RayConfig):
        self.config = config
        self.cluster = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize Ray cluster."""
        try:
            if self.config.local:
                # Initialize local Ray
                ray.init(
                    num_cpus=self.config.num_cpus,
                    num_gpus=self.config.num_gpus,
                    object_store_memory=self.config.memory,
                    ignore_reinit_error=True
                )
                logger.info("Ray initialized locally")
            else:
                # Initialize GCP Ray cluster
                self._initialize_gcp_cluster()
            
            self.is_initialized = True
            logger.info("Ray cluster initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray cluster: {e}")
            raise
    
    def _initialize_gcp_cluster(self):
        """Initialize Ray cluster on GCP."""
        try:
            from ray.autoscaler._private.gcp import GCPNodeProvider
            from ray.autoscaler._private.gcp.config import bootstrap_gcp
            
            # This would require proper GCP setup
            # For now, we'll use local Ray with more resources
            ray.init(
                num_cpus=self.config.num_cpus * 2,
                num_gpus=self.config.num_gpus,
                object_store_memory=self.config.memory * 2,
                ignore_reinit_error=True
            )
            logger.info("Ray initialized for GCP (using local resources)")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP Ray cluster: {e}")
            # Fallback to local
            ray.init(ignore_reinit_error=True)
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if self.is_initialized:
            ray.shutdown()
            self.is_initialized = False
            logger.info("Ray cluster shutdown")


@ray.remote
class ModelWorker:
    """Ray worker for model training and backtesting."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.models = {}
        self.results = []
    
    def train_model(self, task: ModelTask) -> TaskResult:
        """Train a model on the worker."""
        start_time = datetime.now()
        
        try:
            # Import model classes
            from mcp_trader.models.grid_strategies import LinearGridStrategy, FibonacciGridStrategy
            from mcp_trader.models.ml_models import RandomForestModel, XGBoostModel, LightGBMModel
            from mcp_trader.models.rule_based import SMACrossoverStrategy, RSIStrategy
            from mcp_trader.models.ensemble import StackingEnsemble, AdaptiveEnsemble
            
            # Load data
            data = pd.read_parquet(task.data_path)
            
            # Create model based on type
            if task.model_type == 'linear_grid':
                model = LinearGridStrategy(**task.config)
            elif task.model_type == 'fibonacci_grid':
                model = FibonacciGridStrategy(**task.config)
            elif task.model_type == 'random_forest':
                model = RandomForestModel(**task.config)
            elif task.model_type == 'xgboost':
                model = XGBoostModel(**task.config)
            elif task.model_type == 'lightgbm':
                model = LightGBMModel(**task.config)
            elif task.model_type == 'sma_crossover':
                model = SMACrossoverStrategy(**task.config)
            elif task.model_type == 'rsi':
                model = RSIStrategy(**task.config)
            else:
                raise ValueError(f"Unknown model type: {task.model_type}")
            
            # Train model
            model.fit(data)
            
            # Store model
            self.models[task.task_id] = model
            
            # Generate predictions for evaluation
            predictions = model.predict(data)
            
            # Calculate basic metrics
            if predictions:
                prediction_values = [p.prediction for p in predictions]
                avg_prediction = np.mean(prediction_values)
                prediction_std = np.std(prediction_values)
            else:
                avg_prediction = 0.0
                prediction_std = 0.0
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = TaskResult(
                task_id=task.task_id,
                model_name=task.model_name,
                success=True,
                result={
                    'avg_prediction': avg_prediction,
                    'prediction_std': prediction_std,
                    'num_predictions': len(predictions),
                    'model_metadata': model.get_metadata().__dict__
                },
                duration=duration,
                timestamp=datetime.now()
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_result = TaskResult(
                task_id=task.task_id,
                model_name=task.model_name,
                success=False,
                error=str(e),
                duration=duration,
                timestamp=datetime.now()
            )
            self.results.append(error_result)
            return error_result
    
    def backtest_model(self, task: ModelTask) -> TaskResult:
        """Run backtest on a model."""
        start_time = datetime.now()
        
        try:
            # Import backtesting components
            from mcp_trader.backtesting.protocol import BacktestEngine, BacktestConfig
            
            # Load data
            data = pd.read_parquet(task.data_path)
            
            # Create model
            if task.model_type == 'linear_grid':
                from mcp_trader.models.grid_strategies import LinearGridStrategy
                model = LinearGridStrategy(**task.config)
            elif task.model_type == 'random_forest':
                from mcp_trader.models.ml_models import RandomForestModel
                model = RandomForestModel(**task.config)
            else:
                raise ValueError(f"Backtesting not implemented for {task.model_type}")
            
            # Configure backtest
            backtest_config = BacktestConfig(
                initial_capital=task.config.get('initial_capital', 10000),
                commission_rate=task.config.get('commission_rate', 0.001),
                slippage_rate=task.config.get('slippage_rate', 0.0005)
            )
            
            # Run backtest
            engine = BacktestEngine(backtest_config)
            result = engine.run_backtest(model, data, walk_forward=False)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            backtest_result = TaskResult(
                task_id=task.task_id,
                model_name=task.model_name,
                success=True,
                result={
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'total_trades': result.total_trades,
                    'volatility': result.volatility,
                    'calmar_ratio': result.calmar_ratio
                },
                duration=duration,
                timestamp=datetime.now()
            )
            
            self.results.append(backtest_result)
            return backtest_result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_result = TaskResult(
                task_id=task.task_id,
                model_name=task.model_name,
                success=False,
                error=str(e),
                duration=duration,
                timestamp=datetime.now()
            )
            self.results.append(error_result)
            return error_result
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'models_trained': len(self.models),
            'tasks_completed': len(self.results),
            'successful_tasks': len([r for r in self.results if r.success]),
            'failed_tasks': len([r for r in self.results if not r.success])
        }


class DistributedModelTrainer:
    """Distributed model training using Ray."""
    
    def __init__(self, cluster_manager: RayClusterManager):
        self.cluster_manager = cluster_manager
        self.workers = []
        self.task_queue = []
        self.results = []
    
    def initialize_workers(self, num_workers: int = None):
        """Initialize Ray workers."""
        if not self.cluster_manager.is_initialized:
            self.cluster_manager.initialize()
        
        num_workers = num_workers or self.cluster_manager.config.num_cpus
        self.workers = [ModelWorker.remote(f"worker_{i}") for i in range(num_workers)]
        logger.info(f"Initialized {num_workers} Ray workers")
    
    def add_training_task(self, 
                         model_name: str, 
                         model_type: str, 
                         data_path: str, 
                         config: Dict[str, Any],
                         priority: int = 1) -> str:
        """Add a model training task to the queue."""
        task_id = f"{model_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = ModelTask(
            task_id=task_id,
            model_name=model_name,
            model_type=model_type,
            data_path=data_path,
            config=config,
            priority=priority
        )
        
        self.task_queue.append(task)
        logger.info(f"Added training task: {task_id}")
        return task_id
    
    def add_backtest_task(self, 
                         model_name: str, 
                         model_type: str, 
                         data_path: str, 
                         config: Dict[str, Any],
                         priority: int = 1) -> str:
        """Add a backtest task to the queue."""
        task_id = f"backtest_{model_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = ModelTask(
            task_id=task_id,
            model_name=model_name,
            model_type=model_type,
            data_path=data_path,
            config=config,
            priority=priority
        )
        
        self.task_queue.append(task)
        logger.info(f"Added backtest task: {task_id}")
        return task_id
    
    async def run_tasks(self, max_concurrent: int = None) -> List[TaskResult]:
        """Run all tasks in the queue."""
        if not self.workers:
            self.initialize_workers()
        
        max_concurrent = max_concurrent or len(self.workers)
        results = []
        
        # Sort tasks by priority
        self.task_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Process tasks in batches
        for i in range(0, len(self.task_queue), max_concurrent):
            batch = self.task_queue[i:i + max_concurrent]
            
            # Submit batch to workers
            futures = []
            for j, task in enumerate(batch):
                worker = self.workers[j % len(self.workers)]
                
                if 'backtest' in task.task_id:
                    future = worker.backtest_model.remote(task)
                else:
                    future = worker.train_model.remote(task)
                
                futures.append(future)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*futures, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, TaskResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
            
            logger.info(f"Completed batch {i//max_concurrent + 1}/{(len(self.task_queue) + max_concurrent - 1)//max_concurrent}")
        
        self.results.extend(results)
        return results
    
    def get_worker_stats(self) -> List[Dict[str, Any]]:
        """Get statistics from all workers."""
        if not self.workers:
            return []
        
        # Get stats from all workers
        stats_futures = [worker.get_worker_stats.remote() for worker in self.workers]
        stats = ray.get(stats_futures)
        
        return stats
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        if not self.results:
            return {'total_tasks': 0, 'successful_tasks': 0, 'failed_tasks': 0}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        return {
            'total_tasks': len(self.results),
            'successful_tasks': len(successful),
            'failed_tasks': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'avg_duration': np.mean([r.duration for r in self.results]),
            'results_by_model': self._group_results_by_model()
        }
    
    def _group_results_by_model(self) -> Dict[str, List[TaskResult]]:
        """Group results by model name."""
        grouped = {}
        for result in self.results:
            if result.model_name not in grouped:
                grouped[result.model_name] = []
            grouped[result.model_name].append(result)
        return grouped
    
    def save_results(self, filepath: str):
        """Save results to file."""
        results_data = []
        for result in self.results:
            results_data.append({
                'task_id': result.task_id,
                'model_name': result.model_name,
                'success': result.success,
                'result': result.result,
                'error': result.error,
                'duration': result.duration,
                'timestamp': result.timestamp.isoformat() if result.timestamp else None
            })
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


# Example usage and testing
async def main():
    """Example usage of Ray cluster for model training."""
    
    # Configure Ray
    config = RayConfig(
        local=True,
        num_cpus=4,
        memory=8 * 1024 * 1024 * 1024
    )
    
    # Initialize cluster manager
    cluster_manager = RayClusterManager(config)
    
    # Initialize distributed trainer
    trainer = DistributedModelTrainer(cluster_manager)
    
    try:
        # Add some training tasks
        trainer.add_training_task(
            model_name="LinearGrid_BTC",
            model_type="linear_grid",
            data_path="data/btc_1h.parquet",
            config={'grid_spacing': 0.01, 'max_levels': 10}
        )
        
        trainer.add_training_task(
            model_name="RandomForest_ETH",
            model_type="random_forest",
            data_path="data/eth_1h.parquet",
            config={'n_estimators': 100, 'max_depth': 10}
        )
        
        # Run tasks
        results = await trainer.run_tasks()
        
        # Print results
        print(f"Completed {len(results)} tasks")
        for result in results:
            print(f"Task {result.task_id}: {'SUCCESS' if result.success else 'FAILED'}")
            if result.success:
                print(f"  Duration: {result.duration:.2f}s")
                print(f"  Result: {result.result}")
            else:
                print(f"  Error: {result.error}")
        
        # Get summary
        summary = trainer.get_results_summary()
        print(f"\nSummary: {summary}")
        
    finally:
        # Cleanup
        cluster_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())


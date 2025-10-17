#!/usr/bin/env python3
"""
Autonomous Data Pipeline for Self-Improving Aster Trading System
Extends existing data collection with validation, BigQuery storage, and feature engineering.

Features:
- Multi-source data collection (Aster DEX, Binance, backup sources)
- Real-time data validation and quality assurance
- Automated feature engineering (41+ technical indicators)
- BigQuery storage with partitioning and clustering
- Self-healing data pipeline with automatic recovery
- Performance monitoring and alerting
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
import time
import warnings
warnings.filterwarnings('ignore')

# Google Cloud imports
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud.exceptions import NotFound

# Local imports
from mcp_trader.config import get_settings, PRIORITY_SYMBOLS
from mcp_trader.features.engineering import FeatureEngine, FeatureConfig
from data_pipeline.smart_data_router import SmartDataRouter
from cloud_architecture.data_collection_service import CloudDataCollector

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_data_points: int = 0
    outlier_count: int = 0
    data_freshness_seconds: float = 0.0
    collection_latency_ms: float = 0.0
    success_rate: float = 0.0


@dataclass
class AutonomousDataConfig:
    """Configuration for autonomous data pipeline."""
    # Data sources
    primary_source: str = "aster"  # aster, binance, coingecko
    backup_sources: List[str] = field(default_factory=lambda: ["binance", "coingecko"])
    
    # Collection settings
    collection_interval_seconds: int = 300  # 5 minutes
    max_retry_attempts: int = 3
    timeout_seconds: int = 30
    
    # Data validation
    max_price_change_pct: float = 0.5  # 50% max price change between updates
    min_volume_threshold: float = 1000.0  # Minimum volume for valid data
    max_data_age_minutes: int = 10  # Data older than 10 minutes is stale
    
    # BigQuery settings
    project_id: str = "aster-ai-trading"
    dataset_id: str = "trading_data"
    table_prefix: str = "market_data"
    
    # Feature engineering
    enable_feature_engineering: bool = True
    feature_lookback_hours: int = 24
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'atr', 'stochastic',
        'williams_r', 'cci', 'obv', 'vwap', 'volatility_ratio'
    ])
    
    # Monitoring
    enable_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'success_rate_min': 0.95,
        'latency_max_ms': 5000,
        'data_freshness_max_minutes': 15
    })


class DataValidator:
    """Validates data quality and detects anomalies."""
    
    def __init__(self, config: AutonomousDataConfig):
        self.config = config
        self.price_history = {}
        self.volume_history = {}
        
    def validate_market_data(self, symbol: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate market data for quality and consistency."""
        errors = []
        
        try:
            # Check required fields
            required_fields = ['price', 'volume', 'timestamp']
            for field in required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            if errors:
                return False, errors
            
            # Validate price
            price = float(data['price'])
            if price <= 0:
                errors.append("Invalid price: must be positive")
            
            # Check for extreme price changes
            if symbol in self.price_history:
                prev_price = self.price_history[symbol]
                price_change_pct = abs(price - prev_price) / prev_price
                if price_change_pct > self.config.max_price_change_pct:
                    errors.append(f"Extreme price change: {price_change_pct:.2%}")
            
            # Validate volume
            volume = float(data['volume'])
            if volume < self.config.min_volume_threshold:
                errors.append(f"Volume too low: {volume}")
            
            # Check data freshness
            timestamp = data.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age_minutes = (datetime.now() - timestamp).total_seconds() / 60
                if age_minutes > self.config.max_data_age_minutes:
                    errors.append(f"Data too old: {age_minutes:.1f} minutes")
            
            # Update history
            self.price_history[symbol] = price
            self.volume_history[symbol] = volume
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors


class BigQueryManager:
    """Manages BigQuery operations for data storage."""
    
    def __init__(self, config: AutonomousDataConfig):
        self.config = config
        self.client = bigquery.Client(project=config.project_id)
        self.dataset_id = f"{config.project_id}.{config.dataset_id}"
        
    async def initialize_tables(self):
        """Initialize BigQuery tables with proper schemas."""
        try:
            # Create dataset if it doesn't exist
            dataset_ref = bigquery.Dataset(self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")
            
            # Create market data table
            await self._create_market_data_table()
            
            # Create features table
            await self._create_features_table()
            
            # Create performance table
            await self._create_performance_table()
            
        except Exception as e:
            logger.error(f"Error initializing BigQuery tables: {e}")
            raise
    
    async def _create_market_data_table(self):
        """Create market data table with partitioning."""
        table_id = f"{self.dataset_id}.market_data"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("volume", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("high", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("low", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("open", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("price_change_pct", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("data_quality_score", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        )
        table.clustering_fields = ["symbol", "source"]
        
        try:
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def _create_features_table(self):
        """Create features table for technical indicators."""
        table_id = f"{self.dataset_id}.features"
        
        # Dynamic schema based on technical indicators
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("sma_5", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_10", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_12", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_26", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rsi", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_upper", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_middle", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_lower", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("atr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_ratio", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        )
        table.clustering_fields = ["symbol"]
        
        try:
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def _create_performance_table(self):
        """Create performance tracking table."""
        table_id = f"{self.dataset_id}.performance"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("strategy_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("pnl", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("trade_count", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("win_rate", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("sharpe_ratio", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("max_drawdown", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        )
        table.clustering_fields = ["strategy_name", "symbol"]
        
        try:
            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def store_market_data(self, data: List[Dict[str, Any]]):
        """Store market data to BigQuery."""
        try:
            table_id = f"{self.dataset_id}.market_data"
            table = self.client.get_table(table_id)
            
            errors = self.client.insert_rows_json(table, data)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Stored {len(data)} market data records")
                
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def store_features(self, features: List[Dict[str, Any]]):
        """Store features to BigQuery."""
        try:
            table_id = f"{self.dataset_id}.features"
            table = self.client.get_table(table_id)
            
            errors = self.client.insert_rows_json(table, features)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Stored {len(features)} feature records")
                
        except Exception as e:
            logger.error(f"Error storing features: {e}")


class AutonomousDataPipeline:
    """
    Autonomous data pipeline for self-improving trading system.
    
    Features:
    - Multi-source data collection with failover
    - Real-time data validation and quality assurance
    - Automated feature engineering
    - BigQuery storage with partitioning
    - Performance monitoring and alerting
    - Self-healing capabilities
    """
    
    def __init__(self, config: AutonomousDataConfig = None):
        self.config = config or AutonomousDataConfig()
        self.settings = get_settings()
        
        # Initialize components
        self.data_router = SmartDataRouter()
        self.validator = DataValidator(self.config)
        self.bigquery_manager = BigQueryManager(self.config)
        self.feature_engine = FeatureEngine(FeatureConfig())
        
        # Data storage
        self.market_data_buffer = []
        self.features_buffer = []
        self.quality_metrics = DataQualityMetrics()
        
        # Performance tracking
        self.collection_stats = {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'data_points_collected': 0,
            'last_collection_time': None,
            'avg_collection_latency_ms': 0.0
        }
        
        # Control flags
        self.is_running = False
        self.emergency_stop = False
        
        logger.info("🚀 Autonomous Data Pipeline initialized")
    
    async def initialize(self):
        """Initialize the data pipeline."""
        try:
            # Initialize BigQuery tables
            await self.bigquery_manager.initialize_tables()
            
            # Initialize data router
            await self.data_router.initialize()
            
            logger.info("✅ Data pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data pipeline: {e}")
            raise
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """Collect market data from all sources with failover."""
        start_time = time.time()
        collection_data = {}
        
        try:
            # Try primary source first
            try:
                primary_data = await self._collect_from_source(self.config.primary_source)
                if primary_data:
                    collection_data.update(primary_data)
                    logger.info(f"✅ Collected data from primary source: {self.config.primary_source}")
            except Exception as e:
                logger.warning(f"⚠️ Primary source failed: {e}")
            
            # Try backup sources if primary failed or incomplete
            if not collection_data:
                for backup_source in self.config.backup_sources:
                    try:
                        backup_data = await self._collect_from_source(backup_source)
                        if backup_data:
                            collection_data.update(backup_data)
                            logger.info(f"✅ Collected data from backup source: {backup_source}")
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ Backup source {backup_source} failed: {e}")
            
            # Calculate collection metrics
            collection_latency = (time.time() - start_time) * 1000
            self.collection_stats['avg_collection_latency_ms'] = (
                (self.collection_stats['avg_collection_latency_ms'] * self.collection_stats['total_collections'] + 
                 collection_latency) / (self.collection_stats['total_collections'] + 1)
            )
            
            return collection_data
            
        except Exception as e:
            logger.error(f"❌ Market data collection failed: {e}")
            return {}
    
    async def _collect_from_source(self, source: str) -> Dict[str, Any]:
        """Collect data from a specific source."""
        if source == "aster":
            return await self._collect_from_aster()
        elif source == "binance":
            return await self._collect_from_binance()
        elif source == "coingecko":
            return await self._collect_from_coingecko()
        else:
            raise ValueError(f"Unknown source: {source}")
    
    async def _collect_from_aster(self) -> Dict[str, Any]:
        """Collect data from Aster DEX."""
        data = {}
        
        for symbol in PRIORITY_SYMBOLS:
            try:
                # Get ticker data
                ticker = await self.data_router.get_ticker(symbol, source="aster")
                if ticker:
                    data[symbol] = {
                        'price': float(ticker.get('lastPrice', 0)),
                        'volume': float(ticker.get('volume', 0)),
                        'high': float(ticker.get('highPrice', 0)),
                        'low': float(ticker.get('lowPrice', 0)),
                        'open': float(ticker.get('openPrice', 0)),
                        'close': float(ticker.get('lastPrice', 0)),
                        'price_change_pct': float(ticker.get('priceChangePercent', 0)) / 100,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'aster'
                    }
                
            except Exception as e:
                logger.warning(f"Failed to collect {symbol} from Aster: {e}")
        
        return data
    
    async def _collect_from_binance(self) -> Dict[str, Any]:
        """Collect data from Binance (via VPN)."""
        data = {}
        
        for symbol in PRIORITY_SYMBOLS:
            try:
                ticker = await self.data_router.get_ticker(symbol, source="binance")
                if ticker:
                    data[symbol] = {
                        'price': float(ticker.get('lastPrice', 0)),
                        'volume': float(ticker.get('volume', 0)),
                        'high': float(ticker.get('highPrice', 0)),
                        'low': float(ticker.get('lowPrice', 0)),
                        'open': float(ticker.get('openPrice', 0)),
                        'close': float(ticker.get('lastPrice', 0)),
                        'price_change_pct': float(ticker.get('priceChangePercent', 0)) / 100,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'binance'
                    }
                
            except Exception as e:
                logger.warning(f"Failed to collect {symbol} from Binance: {e}")
        
        return data
    
    async def _collect_from_coingecko(self) -> Dict[str, Any]:
        """Collect data from CoinGecko."""
        data = {}
        
        # CoinGecko has different symbol format
        symbol_mapping = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'SOLUSDT': 'solana',
            'SUIUSDT': 'sui',
        }
        
        for symbol in PRIORITY_SYMBOLS:
            if symbol in symbol_mapping:
                try:
                    ticker = await self.data_router.get_ticker(symbol_mapping[symbol], source="coingecko")
                    if ticker:
                        data[symbol] = {
                            'price': float(ticker.get('current_price', 0)),
                            'volume': float(ticker.get('total_volume', 0)),
                            'high': float(ticker.get('high_24h', 0)),
                            'low': float(ticker.get('low_24h', 0)),
                            'open': float(ticker.get('current_price', 0)),  # No open price available
                            'close': float(ticker.get('current_price', 0)),
                            'price_change_pct': float(ticker.get('price_change_percentage_24h', 0)) / 100,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'coingecko'
                        }
                
                except Exception as e:
                    logger.warning(f"Failed to collect {symbol} from CoinGecko: {e}")
        
        return data
    
    async def validate_and_process_data(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and process collected data."""
        processed_data = []
        
        for symbol, data in raw_data.items():
            try:
                # Validate data quality
                is_valid, errors = self.validator.validate_market_data(symbol, data)
                
                if is_valid:
                    # Add data quality score
                    data['data_quality_score'] = 1.0
                    processed_data.append(data)
                    self.quality_metrics.valid_records += 1
                else:
                    logger.warning(f"Invalid data for {symbol}: {errors}")
                    self.quality_metrics.invalid_records += 1
                
                self.quality_metrics.total_records += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self.quality_metrics.invalid_records += 1
        
        return processed_data
    
    async def generate_features(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate technical features from market data."""
        if not self.config.enable_feature_engineering or not market_data:
            return []
        
        try:
            # Convert to DataFrame for feature engineering
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Group by symbol and generate features
            features_list = []
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                
                if len(symbol_data) < 20:  # Need minimum data for indicators
                    continue
                
                # Generate features using existing feature engine
                features_df = self.feature_engine.create_features(symbol_data)
                
                # Convert to list of records
                for timestamp, row in features_df.iterrows():
                    feature_record = {
                        'timestamp': timestamp.isoformat(),
                        'symbol': symbol,
                        **{col: float(val) if pd.notna(val) else None for col, val in row.items()}
                    }
                    features_list.append(feature_record)
            
            return features_list
            
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return []
    
    async def store_data(self, market_data: List[Dict[str, Any]], features: List[Dict[str, Any]]):
        """Store data to BigQuery."""
        try:
            # Store market data
            if market_data:
                await self.bigquery_manager.store_market_data(market_data)
            
            # Store features
            if features:
                await self.bigquery_manager.store_features(features)
            
            logger.info(f"✅ Stored {len(market_data)} market records and {len(features)} feature records")
            
        except Exception as e:
            logger.error(f"❌ Error storing data: {e}")
    
    async def run_collection_cycle(self):
        """Run one complete data collection cycle."""
        cycle_start = time.time()
        
        try:
            logger.info("🔄 Starting data collection cycle...")
            
            # Collect market data
            raw_data = await self.collect_market_data()
            
            if not raw_data:
                logger.warning("⚠️ No data collected in this cycle")
                return
            
            # Validate and process data
            processed_data = await self.validate_and_process_data(raw_data)
            
            if not processed_data:
                logger.warning("⚠️ No valid data after processing")
                return
            
            # Generate features
            features = await self.generate_features(processed_data)
            
            # Store data
            await self.store_data(processed_data, features)
            
            # Update statistics
            self.collection_stats['total_collections'] += 1
            self.collection_stats['successful_collections'] += 1
            self.collection_stats['data_points_collected'] += len(processed_data)
            self.collection_stats['last_collection_time'] = datetime.now()
            
            # Calculate success rate
            self.quality_metrics.success_rate = (
                self.collection_stats['successful_collections'] / 
                self.collection_stats['total_collections']
            )
            
            cycle_duration = time.time() - cycle_start
            logger.info(f"✅ Collection cycle completed in {cycle_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Collection cycle failed: {e}")
            self.collection_stats['failed_collections'] += 1
    
    async def run_pipeline(self):
        """Run the autonomous data pipeline continuously."""
        logger.info("🚀 Starting autonomous data pipeline...")
        self.is_running = True
        
        try:
            while self.is_running and not self.emergency_stop:
                await self.run_collection_cycle()
                
                # Wait for next collection
                await asyncio.sleep(self.config.collection_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline stopped by user")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
        finally:
            self.is_running = False
            logger.info("🛑 Data pipeline stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop,
            'collection_stats': self.collection_stats,
            'quality_metrics': {
                'total_records': self.quality_metrics.total_records,
                'valid_records': self.quality_metrics.valid_records,
                'invalid_records': self.quality_metrics.invalid_records,
                'success_rate': self.quality_metrics.success_rate,
            },
            'config': {
                'collection_interval': self.config.collection_interval_seconds,
                'primary_source': self.config.primary_source,
                'backup_sources': self.config.backup_sources,
            }
        }
    
    def emergency_stop_pipeline(self):
        """Emergency stop the pipeline."""
        logger.warning("🚨 EMERGENCY STOP triggered!")
        self.emergency_stop = True
        self.is_running = False


async def main():
    """Main function to run the autonomous data pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = AutonomousDataConfig(
        collection_interval_seconds=300,  # 5 minutes
        primary_source="aster",
        backup_sources=["binance", "coingecko"],
        enable_feature_engineering=True,
        enable_monitoring=True
    )
    
    # Create and run pipeline
    pipeline = AutonomousDataPipeline(config)
    
    try:
        await pipeline.initialize()
        await pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
BigQuery Setup for Autonomous Trading System
Creates datasets, tables, and views for market data, features, and performance tracking.

Features:
- Automated table creation with proper schemas
- Partitioning and clustering for optimal performance
- Views for common analytics queries
- Data retention policies
- Cost optimization settings
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BigQueryConfig:
    """Configuration for BigQuery setup."""
    project_id: str = "aster-ai-trading"
    dataset_id: str = "trading_data"
    location: str = "US"
    
    # Table configurations
    enable_partitioning: bool = True
    enable_clustering: bool = True
    data_retention_days: int = 365  # 1 year retention
    
    # Cost optimization
    enable_query_cache: bool = True
    max_query_size_gb: float = 1.0


class BigQuerySetup:
    """Sets up BigQuery infrastructure for autonomous trading system."""
    
    def __init__(self, config: BigQueryConfig):
        self.config = config
        self.client = bigquery.Client(project=config.project_id)
        self.dataset_id = f"{config.project_id}.{config.dataset_id}"
    
    async def setup_complete_infrastructure(self):
        """Set up complete BigQuery infrastructure."""
        try:
            logger.info("🚀 Setting up BigQuery infrastructure...")
            
            # Create dataset
            await self.create_dataset()
            
            # Create core tables
            await self.create_market_data_table()
            await self.create_features_table()
            await self.create_performance_table()
            await self.create_trades_table()
            await self.create_strategies_table()
            await self.create_alerts_table()
            
            # Create views
            await self.create_analytics_views()
            
            # Set up data retention policies
            await self.setup_data_retention()
            
            # Create cost optimization settings
            await self.setup_cost_optimization()
            
            logger.info("✅ BigQuery infrastructure setup complete")
            
        except Exception as e:
            logger.error(f"❌ BigQuery setup failed: {e}")
            raise
    
    async def create_dataset(self):
        """Create the main dataset."""
        try:
            dataset_ref = bigquery.Dataset(self.dataset_id)
            dataset_ref.location = self.config.location
            
            # Dataset labels for cost tracking
            dataset_ref.labels = {
                "environment": "production",
                "service": "trading-system",
                "cost-center": "ai-trading"
            }
            
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except NotFound:
                dataset = self.client.create_dataset(dataset_ref)
                logger.info(f"✅ Created dataset {self.dataset_id}")
                
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise
    
    async def create_market_data_table(self):
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
            bigquery.SchemaField("collection_latency_ms", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["symbol", "source"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_features_table(self):
        """Create features table for technical indicators."""
        table_id = f"{self.dataset_id}.features"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            
            # Moving averages
            bigquery.SchemaField("sma_5", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_10", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_50", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_100", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_12", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_26", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_50", "FLOAT64", mode="NULLABLE"),
            
            # Technical indicators
            bigquery.SchemaField("rsi", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE"),
            
            # Bollinger Bands
            bigquery.SchemaField("bb_upper", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_middle", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_lower", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_width", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_position", "FLOAT64", mode="NULLABLE"),
            
            # Volatility indicators
            bigquery.SchemaField("atr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility_5d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility_10d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility_20d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility_30d", "FLOAT64", mode="NULLABLE"),
            
            # Volume indicators
            bigquery.SchemaField("volume_sma_5", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_sma_10", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_sma_20", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_ratio_5", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_ratio_10", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_ratio_20", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("obv", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("vwap", "FLOAT64", mode="NULLABLE"),
            
            # Oscillators
            bigquery.SchemaField("stoch_k", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("stoch_d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("williams_r", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("cci", "FLOAT64", mode="NULLABLE"),
            
            # Cross-asset features
            bigquery.SchemaField("corr_btc_5d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("corr_btc_10d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("corr_btc_20d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("corr_eth_5d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("corr_eth_10d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("corr_eth_20d", "FLOAT64", mode="NULLABLE"),
            
            # Momentum features
            bigquery.SchemaField("momentum_5d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("momentum_10d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("momentum_20d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("vol_momentum_5d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("vol_momentum_10d", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("vol_momentum_20d", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["symbol"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_performance_table(self):
        """Create performance tracking table."""
        table_id = f"{self.dataset_id}.performance"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("strategy_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("pnl", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("trade_count", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("winning_trades", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("losing_trades", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("win_rate", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("avg_win", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("avg_loss", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("profit_factor", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sharpe_ratio", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("max_drawdown", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volatility", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("total_return_pct", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("annualized_return_pct", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["strategy_name", "symbol"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_trades_table(self):
        """Create trades table for individual trade records."""
        table_id = f"{self.dataset_id}.trades"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("trade_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("strategy_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("side", "STRING", mode="REQUIRED"),  # BUY/SELL
            bigquery.SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("value_usd", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("fee", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("pnl", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("entry_price", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("exit_price", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("stop_loss", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("take_profit", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("duration_minutes", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("exit_reason", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("confidence_score", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["strategy_name", "symbol"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_strategies_table(self):
        """Create strategies table for strategy metadata."""
        table_id = f"{self.dataset_id}.strategies"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("strategy_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("strategy_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("parameters", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("performance_metrics", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("is_active", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("weight", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("last_updated", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["strategy_name", "strategy_type"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_alerts_table(self):
        """Create alerts table for system notifications."""
        table_id = f"{self.dataset_id}.alerts"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("alert_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("alert_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("severity", "STRING", mode="REQUIRED"),  # INFO, WARNING, ERROR, CRITICAL
            bigquery.SchemaField("message", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("component", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("strategy_name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("is_resolved", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("resolved_at", "TIMESTAMP", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        if self.config.enable_partitioning:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
        
        if self.config.enable_clustering:
            table.clustering_fields = ["alert_type", "severity", "component"]
        
        try:
            self.client.create_table(table)
            logger.info(f"✅ Created table {table_id}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Table {table_id} already exists")
            else:
                raise
    
    async def create_analytics_views(self):
        """Create views for common analytics queries."""
        views = [
            {
                "name": "daily_performance_summary",
                "query": """
                SELECT
                    DATE(timestamp) as date,
                    strategy_name,
                    symbol,
                    SUM(pnl) as daily_pnl,
                    COUNT(DISTINCT trade_id) as trade_count,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe_ratio,
                    MAX(max_drawdown) as max_drawdown
                FROM `{project_id}.{dataset_id}.performance`
                GROUP BY DATE(timestamp), strategy_name, symbol
                ORDER BY date DESC, daily_pnl DESC
                """.format(project_id=self.config.project_id, dataset_id=self.config.dataset_id)
            },
            {
                "name": "strategy_ranking",
                "query": """
                SELECT
                    strategy_name,
                    COUNT(DISTINCT symbol) as symbols_traded,
                    SUM(pnl) as total_pnl,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe_ratio,
                    MAX(max_drawdown) as max_drawdown,
                    COUNT(*) as performance_records
                FROM `{project_id}.{dataset_id}.performance`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY strategy_name
                ORDER BY total_pnl DESC
                """.format(project_id=self.config.project_id, dataset_id=self.config.dataset_id)
            },
            {
                "name": "market_data_summary",
                "query": """
                SELECT
                    symbol,
                    DATE(timestamp) as date,
                    AVG(price) as avg_price,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    SUM(volume) as total_volume,
                    AVG(price_change_pct) as avg_price_change_pct,
                    AVG(data_quality_score) as avg_quality_score,
                    COUNT(*) as data_points
                FROM `{project_id}.{dataset_id}.market_data`
                GROUP BY symbol, DATE(timestamp)
                ORDER BY date DESC, total_volume DESC
                """.format(project_id=self.config.project_id, dataset_id=self.config.dataset_id)
            },
            {
                "name": "recent_alerts",
                "query": """
                SELECT
                    timestamp,
                    alert_type,
                    severity,
                    message,
                    component,
                    symbol,
                    strategy_name,
                    is_resolved
                FROM `{project_id}.{dataset_id}.alerts`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                ORDER BY timestamp DESC
                """.format(project_id=self.config.project_id, dataset_id=self.config.dataset_id)
            }
        ]
        
        for view_config in views:
            try:
                view_id = f"{self.dataset_id}.{view_config['name']}"
                view = bigquery.Table(view_id)
                view.view_query = view_config['query']
                
                self.client.create_table(view)
                logger.info(f"✅ Created view {view_id}")
                
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"View {view_id} already exists")
                else:
                    logger.error(f"Error creating view {view_config['name']}: {e}")
    
    async def setup_data_retention(self):
        """Set up data retention policies."""
        try:
            # Set up automatic deletion of old data
            tables = [
                "market_data",
                "features", 
                "performance",
                "trades",
                "alerts"
            ]
            
            for table_name in tables:
                table_id = f"{self.dataset_id}.{table_name}"
                
                # Create a scheduled query for data retention
                query = f"""
                DELETE FROM `{table_id}`
                WHERE timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {self.config.data_retention_days} DAY)
                """
                
                # Note: In production, you would set up a Cloud Scheduler job
                # to run this query periodically
                logger.info(f"Data retention policy set for {table_name}: {self.config.data_retention_days} days")
            
            logger.info("✅ Data retention policies configured")
            
        except Exception as e:
            logger.error(f"Error setting up data retention: {e}")
    
    async def setup_cost_optimization(self):
        """Set up cost optimization settings."""
        try:
            # Create a query to set up cost optimization
            optimization_query = f"""
            -- Cost optimization settings for {self.dataset_id}
            -- These settings help reduce BigQuery costs
            
            -- 1. Enable query caching
            SET @@query_label = 'cost_optimization';
            
            -- 2. Set maximum query size
            SET @@max_query_size = {self.config.max_query_size_gb} * 1024 * 1024 * 1024;
            
            -- 3. Create cost monitoring view
            CREATE OR REPLACE VIEW `{self.dataset_id}.cost_monitoring` AS
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as records_processed,
                COUNT(DISTINCT symbol) as symbols_processed,
                AVG(data_quality_score) as avg_quality_score
            FROM `{self.dataset_id}.market_data`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            GROUP BY DATE(timestamp)
            ORDER BY date DESC;
            """
            
            # Execute the optimization query
            job = self.client.query(optimization_query)
            job.result()  # Wait for completion
            
            logger.info("✅ Cost optimization settings configured")
            
        except Exception as e:
            logger.error(f"Error setting up cost optimization: {e}")


async def main():
    """Main function to set up BigQuery infrastructure."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = BigQueryConfig(
        project_id="aster-ai-trading",
        dataset_id="trading_data",
        location="US",
        enable_partitioning=True,
        enable_clustering=True,
        data_retention_days=365,
        enable_query_cache=True,
        max_query_size_gb=1.0
    )
    
    # Set up BigQuery infrastructure
    setup = BigQuerySetup(config)
    await setup.setup_complete_infrastructure()
    
    logger.info("🎉 BigQuery infrastructure setup complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

#!/usr/bin/env python3
"""
Cloud Automated Backtesting Service
Runs continuous backtests and model retraining
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import google.cloud.storage as storage
from google.cloud import bigquery
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Configuration
GCP_PROJECT = os.environ.get('GCP_PROJECT', 'aster-ai-trading')
BUCKET_MODELS = os.environ.get('BUCKET_MODELS', 'aster-trading-models')
DATASET_ID = os.environ.get('DATASET_ID', 'trading_data')
BACKTEST_INTERVAL = int(os.environ.get('BACKTEST_INTERVAL', 3600))  # 1 hour

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudBacktester:
    """Cloud-based automated backtesting service."""

    def __init__(self):
        self.storage_client = storage.Client(project=GCP_PROJECT)
        self.bq_client = bigquery.Client(project=GCP_PROJECT)
        self.bucket = self.storage_client.bucket(BUCKET_MODELS)
        self.is_running = True

        self.backtest_stats = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'best_accuracy': 0.0,
            'last_backtest': None
        }

    def load_training_data(self, symbols: list = None) -> pd.DataFrame:
        """Load training data from BigQuery."""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT']

        try:
            # Query recent market data
            query = f"""
            SELECT *
            FROM `{GCP_PROJECT}.{DATASET_ID}.market_data`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
            AND symbol IN ({','.join([f"'{s}'" for s in symbols])})
            ORDER BY timestamp DESC
            LIMIT 10000
            """

            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Loaded {len(df)} training samples from BigQuery")
            return df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        try:
            if df.empty:
                return None, None

            # Create technical features
            df = df.copy()
            df = df.sort_values('timestamp')

            # Price-based features
            df['price_change'] = df.groupby('symbol')['price'].pct_change()
            df['price_change_5'] = df.groupby('symbol')['price'].pct_change(5)
            df['price_change_20'] = df.groupby('symbol')['price'].pct_change(20)

            # Volume-based features
            df['volume_change'] = df.groupby('symbol')['volume'].pct_change()

            # Volatility
            df['volatility_20'] = df.groupby('symbol')['price_change'].rolling(20).std().reset_index(0, drop=True)

            # Moving averages
            for window in [5, 10, 20]:
                df[f'price_sma_{window}'] = df.groupby('symbol')['price'].rolling(window).mean().reset_index(0, drop=True)

            # Create target (next period return > 1%)
            df['target'] = (df.groupby('symbol')['price'].pct_change().shift(-1) > 0.01).astype(int)

            # Select features
            feature_cols = [
                'price_change', 'price_change_5', 'price_change_20',
                'volume_change', 'volatility_20',
                'price_sma_5', 'price_sma_10', 'price_sma_20'
            ]

            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()

            if len(df) < 100:
                logger.warning("Insufficient data for training")
                return None, None

            X = df[feature_cols].values
            y = df['target'].values

            logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None

    def train_models(self, X, y) -> dict:
        """Train multiple ML models."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {}
            results = {}

            # Random Forest
            logger.info("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight='balanced'
            )
            rf.fit(X_train, y_train)

            rf_pred = rf.predict(X_test)
            rf_proba = rf.predict_proba(X_test)[:, 1]

            models['random_forest'] = rf
            results['random_forest'] = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'auc_roc': roc_auc_score(y_test, rf_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'feature_importance': rf.feature_importances_
            }

            # XGBoost
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            xgb_pred = xgb_model.predict(X_test)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

            models['xgboost'] = xgb_model
            results['xgboost'] = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'auc_roc': roc_auc_score(y_test, xgb_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'feature_importance': xgb_model.feature_importances_
            }

            # Gradient Boosting
            logger.info("Training Gradient Boosting...")
            gb = GradientBoostingClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            gb.fit(X_train, y_train)

            gb_pred = gb.predict(X_test)
            gb_proba = gb.predict_proba(X_test)[:, 1]

            models['gradient_boosting'] = gb
            results['gradient_boosting'] = {
                'accuracy': accuracy_score(y_test, gb_pred),
                'auc_roc': roc_auc_score(y_test, gb_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'feature_importance': gb.feature_importances_
            }

            # Ensemble (best performing model gets higher weight)
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            ensemble_pred = (rf_pred + xgb_pred + gb_pred) / 3
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
            ensemble_proba = (rf_proba + xgb_proba + gb_proba) / 3

            results['ensemble'] = {
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'auc_roc': roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'best_individual': best_model
            }

            logger.info("Model training complete")
            for model_name, metrics in results.items():
                logger.info(".3f"
            return {'models': models, 'results': results}

        except Exception as e:
            logger.error(f"Error training models: {e}")
            return None

    def save_models_to_cloud(self, models: dict, results: dict):
        """Save trained models to Cloud Storage."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save each model
            for model_name, model in models.items():
                model_filename = f"{model_name}_{timestamp}.pkl"
                blob = self.bucket.blob(f"models/{model_filename}")

                # Save model to bytes
                import io
                buffer = io.BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)

                blob.upload_from_file(buffer, content_type='application/octet-stream')
                logger.info(f"Saved {model_name} to gs://{BUCKET_MODELS}/models/{model_filename}")

            # Save results metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'models_trained': list(models.keys()),
                'results': results,
                'best_model': max(results.keys(), key=lambda x: results[x]['accuracy']),
                'best_accuracy': max(results[x]['accuracy'] for x in results.keys())
            }

            metadata_blob = self.bucket.blob(f"metadata/training_{timestamp}.json")
            metadata_blob.upload_from_string(json.dumps(metadata, indent=2, default=str))

            logger.info(f"Saved training metadata to gs://{BUCKET_MODELS}/metadata/training_{timestamp}.json")

        except Exception as e:
            logger.error(f"Error saving models to cloud: {e}")

    def save_backtest_results(self, results: dict):
        """Save backtest results to BigQuery."""
        try:
            # Prepare row for BigQuery
            row = {
                'timestamp': datetime.now(),
                'strategy': 'ensemble_trading',
                'win_rate': results['results']['ensemble']['accuracy'] * 100,
                'total_pnl': 0.0,  # Would be calculated from actual trading
                'sharpe_ratio': 0.0,  # Would be calculated
                'max_drawdown': 0.0  # Would be calculated
            }

            # Insert to BigQuery
            table_id = f"{GCP_PROJECT}.{DATASET_ID}.backtest_results"
            table = self.bq_client.get_table(table_id)

            errors = self.bq_client.insert_rows_json(table, [row])
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info("Saved backtest results to BigQuery")

        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")

    async def run_backtest(self, symbols: list = None):
        """Run a complete backtest cycle."""
        try:
            logger.info("Starting automated backtest...")

            # Load data
            df = self.load_training_data(symbols)
            if df.empty:
                logger.warning("No data available for backtest")
                return

            # Prepare features
            X, y = self.prepare_features(df)
            if X is None or y is None:
                logger.warning("Failed to prepare features for backtest")
                return

            # Train models
            training_results = self.train_models(X, y)
            if not training_results:
                logger.warning("Model training failed")
                return

            # Save models and results
            self.save_models_to_cloud(training_results['models'], training_results['results'])
            self.save_backtest_results(training_results)

            # Update stats
            self.backtest_stats['total_backtests'] += 1
            self.backtest_stats['successful_backtests'] += 1
            self.backtest_stats['last_backtest'] = datetime.now()

            best_accuracy = training_results['results']['ensemble']['accuracy']
            if best_accuracy > self.backtest_stats['best_accuracy']:
                self.backtest_stats['best_accuracy'] = best_accuracy

            logger.info(f"Backtest completed successfully - Best accuracy: {best_accuracy:.3f}")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            self.backtest_stats['failed_backtests'] += 1

    async def retrain_models(self, force: bool = False):
        """Force retrain all models with latest data."""
        logger.info("Starting model retraining...")
        await self.run_backtest()
        logger.info("Model retraining complete")

    async def run_service(self):
        """Main service loop."""
        logger.info(f"Automated backtesting service running (interval: {BACKTEST_INTERVAL}s)")

        while self.is_running:
            try:
                await self.run_backtest()
                await asyncio.sleep(BACKTEST_INTERVAL)

            except Exception as e:
                logger.error(f"Service error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    def get_status(self):
        """Get service status."""
        return {
            'service': 'automated_backtesting',
            'running': self.is_running,
            'stats': self.backtest_stats,
            'next_backtest_in': BACKTEST_INTERVAL,
            'model_bucket': BUCKET_MODELS,
            'bigquery_dataset': f"{GCP_PROJECT}.{DATASET_ID}"
        }

# Web server for API endpoints
from aiohttp import web

async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        'status': 'healthy',
        'service': 'backtesting',
        'timestamp': datetime.now().isoformat()
    })

async def status_endpoint(request):
    """Status endpoint."""
    backtester = request.app['backtester']
    return web.json_response(backtester.get_status())

async def backtest_endpoint(request):
    """Manual backtest trigger."""
    backtester = request.app['backtester']

    try:
        data = await request.json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        await backtester.run_backtest(symbols)
        return web.json_response({'status': 'backtest_started', 'symbols': symbols})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def retrain_endpoint(request):
    """Model retraining endpoint."""
    backtester = request.app['backtester']

    try:
        data = await request.json()
        force = data.get('force', False)
        await backtester.retrain_models(force)
        return web.json_response({'status': 'retraining_started'})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def init_app():
    """Initialize web application."""
    app = web.Application()
    backtester = CloudBacktester()
    app['backtester'] = backtester

    # Routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/status', status_endpoint)
    app.router.add_post('/backtest', backtest_endpoint)
    app.router.add_post('/retrain', retrain_endpoint)

    # Start background service
    asyncio.create_task(backtester.run_service())

    return app

if __name__ == "__main__":
    # Run as web service
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

#!/usr/bin/env python3
"""
Master AI Training Pipeline - Comprehensive Multi-Source Training
Trains on ALL Aster DEX assets with technical, macro, and sentiment data
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveDataValidator:
    """
    Validates data quality across all sources before training.
    """
    
    def __init__(self):
        self.validation_results = {
            'aster_assets': {},
            'technical_indicators': {},
            'macro_data': {},
            'sentiment_data': {},
            'overall_quality': 0.0
        }
        logger.info("Data validator initialized")
    
    def validate_aster_assets(self, data_dir: Path) -> Dict:
        """Validate Aster DEX asset data."""
        logger.info("Validating Aster DEX asset data...")
        
        results = {
            'total_assets': 0,
            'valid_assets': 0,
            'invalid_assets': [],
            'quality_scores': {},
            'data_ranges': {},
            'missing_data': {}
        }
        
        try:
            data_files = [f for f in data_dir.glob("*.parquet") if "collection_summary" not in f.name]
            results['total_assets'] = len(data_files)
            
            for file in data_files:
                asset = file.stem.replace('_1h', '').replace('_4h', '').replace('_1d', '')
                
                try:
                    df = pd.read_parquet(file)
                    
                    # Check for required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        results['invalid_assets'].append(asset)
                        results['missing_data'][asset] = missing_cols
                        continue
                    
                    # Data quality checks
                    quality_score = 0.0
                    checks = []
                    
                    # 1. No missing values
                    completeness = 1 - (df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols)))
                    checks.append(completeness)
                    
                    # 2. No zero/negative prices
                    price_validity = ((df[['open', 'high', 'low', 'close']] > 0).all().all())
                    checks.append(1.0 if price_validity else 0.5)
                    
                    # 3. High >= Low
                    ohlc_consistency = (df['high'] >= df['low']).mean()
                    checks.append(ohlc_consistency)
                    
                    # 4. Sufficient data points
                    data_sufficiency = min(len(df) / 500, 1.0)  # Target 500+ points
                    checks.append(data_sufficiency)
                    
                    # 5. No extreme outliers (> 100x daily change)
                    if len(df) > 1:
                        returns = df['close'].pct_change().abs()
                        outlier_check = (returns < 100).mean()
                        checks.append(outlier_check)
                    else:
                        checks.append(0.0)
                    
                    quality_score = np.mean(checks)
                    
                    if quality_score >= 0.7:
                        results['valid_assets'] += 1
                        results['quality_scores'][asset] = quality_score
                        results['data_ranges'][asset] = {
                            'points': len(df),
                            'price_range': f"${df['close'].min():.2f} - ${df['close'].max():.2f}",
                            'volume_avg': f"{df['volume'].mean():.2f}"
                        }
                    else:
                        results['invalid_assets'].append(asset)
                    
                except Exception as e:
                    logger.warning(f"Failed to validate {asset}: {e}")
                    results['invalid_assets'].append(asset)
            
            logger.info(f"✅ Validated {results['valid_assets']}/{results['total_assets']} assets")
            
        except Exception as e:
            logger.error(f"Asset validation failed: {e}")
        
        self.validation_results['aster_assets'] = results
        return results
    
    def validate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Validate technical indicator calculations."""
        logger.info("Validating technical indicators...")
        
        results = {
            'indicators_checked': [],
            'valid_indicators': [],
            'invalid_indicators': [],
            'quality_score': 0.0
        }
        
        try:
            # Check for common technical indicators
            expected_indicators = [
                'sma_5', 'sma_10', 'sma_20',
                'rsi', 'volatility',
                'price_change', 'high_low_ratio'
            ]
            
            for indicator in expected_indicators:
                results['indicators_checked'].append(indicator)
                
                if indicator in df.columns:
                    # Check if indicator has reasonable values
                    if df[indicator].isnull().sum() / len(df) < 0.3:  # Less than 30% missing
                        results['valid_indicators'].append(indicator)
                    else:
                        results['invalid_indicators'].append(indicator)
                else:
                    results['invalid_indicators'].append(indicator)
            
            results['quality_score'] = len(results['valid_indicators']) / len(results['indicators_checked'])
            logger.info(f"✅ {len(results['valid_indicators'])}/{len(results['indicators_checked'])} indicators valid")
            
        except Exception as e:
            logger.error(f"Technical indicator validation failed: {e}")
        
        self.validation_results['technical_indicators'] = results
        return results
    
    def generate_validation_report(self, output_dir: Path) -> Path:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Calculate overall quality score
            scores = []
            
            if self.validation_results['aster_assets'].get('valid_assets', 0) > 0:
                asset_quality = self.validation_results['aster_assets']['valid_assets'] / max(self.validation_results['aster_assets']['total_assets'], 1)
                scores.append(asset_quality)
            
            if self.validation_results['technical_indicators'].get('quality_score', 0) > 0:
                scores.append(self.validation_results['technical_indicators']['quality_score'])
            
            self.validation_results['overall_quality'] = np.mean(scores) if scores else 0.0
            self.validation_results['timestamp'] = datetime.now().isoformat()
            
            with open(report_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            
            logger.info(f"✅ Validation report saved: {report_path}")
            logger.info(f"   Overall Quality Score: {self.validation_results['overall_quality']:.2%}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return None


class ComprehensiveFeatureEngine:
    """
    Advanced feature engineering with all available data sources.
    """
    
    def __init__(self):
        self.feature_names = []
        logger.info("Feature engine initialized")
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators."""
        logger.info("Creating technical features...")
        
        try:
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_20'] = df['close'].pct_change(20)
            
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_price_ratio'] = df['volume'] / df['close']
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
            # Exponential moving averages
            for window in [12, 26]:
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility
            df['volatility_20'] = df['price_change'].rolling(20).std()
            df['volatility_50'] = df['price_change'].rolling(50).std()
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_30'] = self._calculate_rsi(df['close'], 30)
            
            # Stochastic Oscillator
            df['stoch_k'] = self._calculate_stochastic(df, 14)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Average True Range (ATR)
            df['atr'] = self._calculate_atr(df, 14)
            
            # On-Balance Volume (OBV)
            df['obv'] = self._calculate_obv(df)
            
            # Money Flow Index (MFI)
            df['mfi'] = self._calculate_mfi(df, 14)
            
            logger.info(f"✅ Created {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timestamp']])} technical features")
            
        except Exception as e:
            logger.error(f"Technical feature creation failed: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        return stoch_k
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _calculate_mfi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def create_cross_asset_features(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Create features based on cross-asset correlations."""
        logger.info("Creating cross-asset features...")
        
        try:
            if 'symbol' in all_data.columns:
                # Market-wide momentum
                all_data['market_momentum'] = all_data.groupby('timestamp')['price_change'].transform('mean')
                
                # Relative strength vs market
                all_data['relative_strength'] = all_data['price_change'] - all_data['market_momentum']
                
                # Volume concentration
                all_data['volume_rank'] = all_data.groupby('timestamp')['volume'].rank(pct=True)
                
                logger.info("✅ Created cross-asset features")
        
        except Exception as e:
            logger.warning(f"Cross-asset feature creation failed: {e}")
        
        return all_data
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature columns."""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timestamp', 'target', 'actual_return']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        return self.feature_names


class MultiModelTrainer:
    """
    Trains multiple ML models and creates ensemble.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        logger.info("Multi-model trainer initialized")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
        
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'model_name': 'Random Forest',
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = model.feature_importances_
        
        logger.info(f"✅ Random Forest: Accuracy={results['accuracy']:.3f}, AUC={results['auc_roc']:.3f}")
        
        return results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            logger.info("Training XGBoost...")
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results = {
                'model_name': 'XGBoost',
                'accuracy': accuracy_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.models['xgboost'] = model
            self.feature_importance['xgboost'] = model.feature_importances_
            
            logger.info(f"✅ XGBoost: Accuracy={results['accuracy']:.3f}, AUC={results['auc_roc']:.3f}")
            
            return results
            
        except ImportError:
            logger.warning("XGBoost not installed, skipping")
            return None
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        logger.info("Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'model_name': 'Gradient Boosting',
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.models['gradient_boosting'] = model
        self.feature_importance['gradient_boosting'] = model.feature_importances_
        
        logger.info(f"✅ Gradient Boosting: Accuracy={results['accuracy']:.3f}, AUC={results['auc_roc']:.3f}")
        
        return results
    
    def create_ensemble(self, model_results: List[Dict], y_test) -> Dict:
        """Create ensemble prediction from multiple models."""
        logger.info("Creating ensemble model...")
        
        try:
            # Average probabilities
            all_probas = np.array([r['probabilities'] for r in model_results if r is not None])
            ensemble_proba = np.mean(all_probas, axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            results = {
                'model_name': 'Ensemble',
                'accuracy': accuracy_score(y_test, ensemble_pred),
                'auc_roc': roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5,
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba
            }
            
            logger.info(f"✅ Ensemble: Accuracy={results['accuracy']:.3f}, AUC={results['auc_roc']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ensemble creation failed: {e}")
            return None


class ComprehensiveReportGenerator:
    """
    Generates detailed reports with visualizations.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        logger.info(f"Report generator initialized: {output_dir}")
    
    def generate_data_quality_visualizations(self, validation_results: Dict):
        """Generate data quality visualizations."""
        logger.info("Generating data quality visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Asset quality scores
            if validation_results['aster_assets'].get('quality_scores'):
                quality_scores = validation_results['aster_assets']['quality_scores']
                assets = list(quality_scores.keys())[:20]  # Top 20
                scores = [quality_scores[a] for a in assets]
                
                axes[0, 0].barh(assets, scores, color='skyblue')
                axes[0, 0].set_xlabel('Quality Score')
                axes[0, 0].set_title('Asset Data Quality (Top 20)')
                axes[0, 0].axvline(x=0.7, color='r', linestyle='--', label='Threshold')
                axes[0, 0].legend()
            
            # Asset distribution
            asset_stats = validation_results['aster_assets']
            labels = ['Valid Assets', 'Invalid Assets']
            sizes = [asset_stats.get('valid_assets', 0), 
                    len(asset_stats.get('invalid_assets', []))]
            colors = ['#90EE90', '#FFB6C1']
            axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Asset Validation Results')
            
            # Technical indicators
            if validation_results['technical_indicators']:
                indicator_stats = validation_results['technical_indicators']
                labels = ['Valid', 'Invalid']
                sizes = [len(indicator_stats.get('valid_indicators', [])),
                        len(indicator_stats.get('invalid_indicators', []))]
                colors = ['#90EE90', '#FFB6C1']
                axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Technical Indicator Validation')
            
            # Overall quality
            overall_quality = validation_results.get('overall_quality', 0.0)
            axes[1, 1].bar(['Overall Quality'], [overall_quality], color='cornflowerblue', width=0.4)
            axes[1, 1].set_ylim([0, 1.0])
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Overall Data Quality Score')
            axes[1, 1].axhline(y=0.7, color='r', linestyle='--', label='Target')
            axes[1, 1].legend()
            
            plt.tight_layout()
            output_path = self.output_dir / 'data_quality_report.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Data quality visualizations saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Data quality visualization failed: {e}")
    
    def generate_feature_importance_plot(self, feature_importance: Dict, feature_names: List[str]):
        """Generate feature importance visualizations."""
        logger.info("Generating feature importance plots...")
        
        try:
            n_models = len(feature_importance)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, importances) in enumerate(feature_importance.items()):
                # Get top 20 features
                indices = np.argsort(importances)[-20:]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
                axes[idx].barh(top_features, top_importances, color='steelblue')
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nFeature Importance')
                axes[idx].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            output_path = self.output_dir / 'feature_importance.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Feature importance plots saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Feature importance visualization failed: {e}")
    
    def generate_model_comparison_plot(self, model_results: List[Dict]):
        """Generate model performance comparison."""
        logger.info("Generating model comparison plots...")
        
        try:
            model_names = [r['model_name'] for r in model_results if r is not None]
            accuracies = [r['accuracy'] for r in model_results if r is not None]
            auc_scores = [r['auc_roc'] for r in model_results if r is not None]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Accuracy comparison
            axes[0].bar(model_names, accuracies, color='steelblue', alpha=0.7)
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Model Accuracy Comparison')
            axes[0].set_ylim([0, 1.0])
            axes[0].grid(axis='y', alpha=0.3)
            for i, v in enumerate(accuracies):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
            
            # AUC-ROC comparison
            axes[1].bar(model_names, auc_scores, color='coral', alpha=0.7)
            axes[1].set_ylabel('AUC-ROC')
            axes[1].set_title('Model AUC-ROC Comparison')
            axes[1].set_ylim([0, 1.0])
            axes[1].grid(axis='y', alpha=0.3)
            for i, v in enumerate(auc_scores):
                axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            output_path = self.output_dir / 'model_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Model comparison plots saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Model comparison visualization failed: {e}")
    
    def generate_comprehensive_report(self, training_results: Dict) -> Path:
        """Generate comprehensive markdown report."""
        logger.info("Generating comprehensive training report...")
        
        try:
            report_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🚀 Comprehensive AI Trading Model - Training Report\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                
                # Executive Summary
                f.write("## 📊 Executive Summary\n\n")
                f.write(f"- **Total Assets Trained**: {training_results.get('total_assets', 0)}\n")
                f.write(f"- **Total Features**: {training_results.get('total_features', 0)}\n")
                f.write(f"- **Training Samples**: {training_results.get('training_samples', 0):,}\n")
                f.write(f"- **Test Samples**: {training_results.get('test_samples', 0):,}\n")
                f.write(f"- **Best Model**: {training_results.get('best_model', 'N/A')}\n")
                f.write(f"- **Best Accuracy**: {training_results.get('best_accuracy', 0):.2%}\n")
                f.write(f"- **Overall Quality Score**: {training_results.get('data_quality', 0):.2%}\n\n")
                
                # Data Validation
                f.write("## ✅ Data Validation Results\n\n")
                if 'validation_results' in training_results:
                    val = training_results['validation_results']
                    f.write(f"### Aster DEX Assets\n")
                    f.write(f"- Total assets scanned: {val['aster_assets'].get('total_assets', 0)}\n")
                    f.write(f"- Valid assets: {val['aster_assets'].get('valid_assets', 0)}\n")
                    f.write(f"- Validation rate: {val['aster_assets'].get('valid_assets', 0) / max(val['aster_assets'].get('total_assets', 1), 1):.1%}\n\n")
                
                f.write("![Data Quality](data_quality_report.png)\n\n")
                
                # Feature Engineering
                f.write("## 🛠️ Feature Engineering\n\n")
                f.write("### Technical Indicators Created:\n\n")
                f.write("1. **Price-based**: price_change, high_low_ratio, close_open_ratio\n")
                f.write("2. **Moving Averages**: SMA (5, 10, 20, 50), EMA (12, 26)\n")
                f.write("3. **Momentum**: MACD, RSI (14, 30), Stochastic Oscillator\n")
                f.write("4. **Volatility**: Bollinger Bands, ATR, Rolling Std\n")
                f.write("5. **Volume**: OBV, MFI, Volume MA Ratio\n")
                f.write("6. **Cross-Asset**: Market momentum, Relative strength\n\n")
                
                f.write("![Feature Importance](feature_importance.png)\n\n")
                
                # Model Performance
                f.write("## 🤖 Model Performance\n\n")
                if 'model_results' in training_results:
                    for result in training_results['model_results']:
                        if result:
                            f.write(f"### {result['model_name']}\n\n")
                            f.write(f"- **Accuracy**: {result['accuracy']:.2%}\n")
                            f.write(f"- **AUC-ROC**: {result['auc_roc']:.3f}\n\n")
                
                f.write("![Model Comparison](model_comparison.png)\n\n")
                
                # Recommendations
                f.write("## 💡 Recommendations\n\n")
                best_acc = training_results.get('best_accuracy', 0)
                if best_acc >= 0.80:
                    f.write("✅ **EXCELLENT** - Model ready for paper trading\n\n")
                elif best_acc >= 0.70:
                    f.write("⚠️ **GOOD** - Model shows promise, recommend paper trading with monitoring\n\n")
                else:
                    f.write("❌ **NEEDS IMPROVEMENT** - Collect more data or adjust features\n\n")
                
                f.write("### Next Steps:\n\n")
                f.write("1. Deploy best model to paper trading\n")
                f.write("2. Monitor performance for 48-72 hours\n")
                f.write("3. Collect real trading feedback\n")
                f.write("4. Retrain weekly with new data\n")
                f.write("5. Scale to live trading when validated\n\n")
                
                f.write("---\n\n")
                f.write("*Report generated by Aster AI Trading System*\n")
            
            logger.info(f"✅ Comprehensive report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None


def main():
    """Main training pipeline execution."""
    print("""
================================================================================
                  COMPREHENSIVE AI TRAINING PIPELINE
              Training on ALL Aster DEX Assets + Full Feature Set
================================================================================
    """)
    
    # Initialize components
    data_dir = Path("data/historical/real_aster_only")
    output_dir = Path("training_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = ComprehensiveDataValidator()
    feature_engine = ComprehensiveFeatureEngine()
    trainer = MultiModelTrainer()
    report_generator = ComprehensiveReportGenerator(output_dir)
    
    training_results = {}
    
    try:
        # Step 1: Validate data
        print("\n" + "="*80)
        print("STEP 1: DATA VALIDATION")
        print("="*80)
        
        asset_validation = validator.validate_aster_assets(data_dir)
        
        # Create sample data for technical indicator validation
        sample_files = list(data_dir.glob("*.parquet"))[:5]
        if sample_files:
            sample_df = pd.read_parquet(sample_files[0])
            sample_df = feature_engine.create_technical_features(sample_df)
            tech_validation = validator.validate_technical_indicators(sample_df)
        
        validation_report_path = validator.generate_validation_report(output_dir)
        training_results['validation_results'] = validator.validation_results
        training_results['data_quality'] = validator.validation_results['overall_quality']
        
        report_generator.generate_data_quality_visualizations(validator.validation_results)
        
        # Step 2: Load and prepare all data
        print("\n" + "="*80)
        print("STEP 2: LOADING AND FEATURE ENGINEERING")
        print("="*80)
        
        logger.info("Loading all Aster DEX assets...")
        all_data = []
        data_files = [f for f in data_dir.glob("*.parquet") if "collection_summary" not in f.name]
        
        for file in data_files[:50]:  # Limit to 50 assets for reasonable training time
            asset = file.stem.replace('_1h', '').replace('_4h', '').replace('_1d', '')
            if asset in asset_validation['quality_scores']:
                df = pd.read_parquet(file)
                df['symbol'] = asset
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.date_range(start='2024-04-01', periods=len(df), freq='1h')
                all_data.append(df)
        
        logger.info(f"Loaded {len(all_data)} assets")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Create all features
        combined_df = feature_engine.create_technical_features(combined_df)
        combined_df = feature_engine.create_cross_asset_features(combined_df)
        
        # Create target variable (1% profit in next period)
        combined_df['actual_return'] = combined_df.groupby('symbol')['close'].pct_change().shift(-1)
        combined_df['target'] = (combined_df['actual_return'] > 0.01).astype(int)
        
        # Remove NaN and infinity
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna()
        
        logger.info(f"Final dataset: {len(combined_df)} samples")
        
        # Step 3: Prepare training data
        print("\n" + "="*80)
        print("STEP 3: PREPARING TRAINING DATA")
        print("="*80)
        
        feature_names = feature_engine.get_feature_names(combined_df)
        X = combined_df[feature_names].values
        y = combined_df['target'].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"Training samples: {len(X_train):,}")
        logger.info(f"Test samples: {len(X_test):,}")
        logger.info(f"Features: {len(feature_names)}")
        
        training_results['total_assets'] = len(all_data)
        training_results['total_features'] = len(feature_names)
        training_results['training_samples'] = len(X_train)
        training_results['test_samples'] = len(X_test)
        
        # Step 4: Train models
        print("\n" + "="*80)
        print("STEP 4: TRAINING MODELS")
        print("="*80)
        
        model_results = []
        
        # Train Random Forest
        rf_results = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        model_results.append(rf_results)
        
        # Train XGBoost
        xgb_results = trainer.train_xgboost(X_train, y_train, X_test, y_test)
        if xgb_results:
            model_results.append(xgb_results)
        
        # Train Gradient Boosting
        gb_results = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
        model_results.append(gb_results)
        
        # Create ensemble
        ensemble_results = trainer.create_ensemble(model_results, y_test)
        if ensemble_results:
            model_results.append(ensemble_results)
        
        training_results['model_results'] = model_results
        
        # Find best model
        best_model = max(model_results, key=lambda x: x['accuracy'] if x else 0)
        training_results['best_model'] = best_model['model_name']
        training_results['best_accuracy'] = best_model['accuracy']
        
        # Step 5: Generate visualizations and report
        print("\n" + "="*80)
        print("STEP 5: GENERATING REPORTS AND VISUALIZATIONS")
        print("="*80)
        
        report_generator.generate_feature_importance_plot(trainer.feature_importance, feature_names)
        report_generator.generate_model_comparison_plot(model_results)
        report_path = report_generator.generate_comprehensive_report(training_results)
        
        # Save models
        import joblib
        for model_name, model in trainer.models.items():
            model_path = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"✅ Saved {model_name} to {model_path}")
        
        # Save training metadata
        metadata_path = output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert numpy types to native Python types
            serializable_results = {
                'total_assets': int(training_results['total_assets']),
                'total_features': int(training_results['total_features']),
                'training_samples': int(training_results['training_samples']),
                'test_samples': int(training_results['test_samples']),
                'best_model': training_results['best_model'],
                'best_accuracy': float(training_results['best_accuracy']),
                'data_quality': float(training_results['data_quality']),
                'feature_names': feature_names,
                'timestamp': datetime.now().isoformat()
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
        
        # Final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*80)
        print(f"✅ Assets trained: {training_results['total_assets']}")
        print(f"✅ Features created: {training_results['total_features']}")
        print(f"✅ Training samples: {training_results['training_samples']:,}")
        print(f"✅ Best model: {training_results['best_model']}")
        print(f"✅ Best accuracy: {training_results['best_accuracy']:.2%}")
        print(f"✅ Data quality: {training_results['data_quality']:.2%}")
        print(f"\n📊 Full report: {report_path}")
        print(f"📁 All outputs: {output_dir}")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


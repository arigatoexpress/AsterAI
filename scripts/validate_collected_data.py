#!/usr/bin/env python3
"""
Data Validation Script
Comprehensive validation of collected data quality, completeness, and integrity
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation and quality assessment"""

    def __init__(self, data_dir: str = "data/historical/ultimate_dataset"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        self.quality_scores = {}

    def validate_all_data(self) -> Dict:
        """Run comprehensive validation on all collected data"""
        print("""
╔════════════════════════════════════════════════════════════════╗
║               Data Quality Validation Report                   ║
╚════════════════════════════════════════════════════════════════╝
        """)

        # Validate each data category
        categories = ['crypto', 'traditional', 'alternative']

        for category in categories:
            logger.info(f"Validating {category} data...")
            self._validate_category(category)

        # Generate summary report
        self._generate_validation_report()

        return self.validation_results

    def _validate_category(self, category: str):
        """Validate a specific data category"""
        category_dir = self.data_dir / category

        if not category_dir.exists():
            logger.warning(f"Category directory {category} does not exist")
            self.validation_results[category] = {'status': 'missing_directory'}
            return

        # Get all data files
        data_files = list(category_dir.rglob('*.parquet'))
        metadata_files = list(category_dir.rglob('*metadata.json'))

        logger.info(f"Found {len(data_files)} data files in {category}")

        category_results = {
            'total_files': len(data_files),
            'assets_validated': 0,
            'quality_metrics': {},
            'issues': [],
            'recommendations': []
        }

        # Validate each asset
        for data_file in data_files:
            asset_name = data_file.stem.replace('_consolidated', '')
            self._validate_asset(asset_name, data_file, category_results)

        # Check for metadata consistency
        self._validate_metadata_consistency(metadata_files, category_results)

        # Calculate category quality score
        self._calculate_category_score(category, category_results)

        self.validation_results[category] = category_results

    def _validate_asset(self, asset_name: str, data_file: Path, category_results: Dict):
        """Validate individual asset data"""
        try:
            # Load data
            df = pd.read_parquet(data_file)
            category_results['assets_validated'] += 1

            # Basic validation
            asset_issues = []

            # 1. Check required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                asset_issues.append(f"Missing required columns: {missing_cols}")

            # 2. Check data types
            if 'close' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['close']):
                    asset_issues.append("Close price is not numeric")

            # 3. Check for null values
            null_pct = df.isnull().mean() * 100
            high_null_cols = null_pct[null_pct > 5].index.tolist()
            if high_null_cols:
                asset_issues.append(f"High null values in columns: {high_null_cols}")

            # 4. Check data range and outliers
            if 'close' in df.columns and df['close'].notna().any():
                close_prices = df['close'].dropna()

                # Price sanity checks
                if close_prices.min() <= 0:
                    asset_issues.append("Negative or zero prices found")

                # Outlier detection (using IQR method)
                Q1 = close_prices.quantile(0.25)
                Q3 = close_prices.quantile(0.75)
                IQR = Q3 - Q1
                outliers = close_prices[(close_prices < (Q1 - 3 * IQR)) | (close_prices > (Q3 + 3 * IQR))]
                if len(outliers) > len(close_prices) * 0.01:  # More than 1% outliers
                    asset_issues.append(f"Potential outliers detected: {len(outliers)} values")

            # 5. Check time series continuity
            if isinstance(df.index, pd.DatetimeIndex):
                # Check for gaps in time series
                time_diffs = df.index.to_series().diff()
                if hasattr(time_diffs, 'dt'):
                    # For datetime index
                    gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]  # Allow 2-hour gaps
                    if len(gaps) > 0:
                        asset_issues.append(f"Time series gaps detected: {len(gaps)} gaps")

                # Check data freshness
                if len(df) > 0:
                    days_old = (datetime.now() - df.index.max()).days
                    if days_old > 7:
                        asset_issues.append(f"Data is {days_old} days old")

            # 6. Volume validation (if available)
            if 'volume' in df.columns:
                volume_zeros = (df['volume'] == 0).sum()
                if volume_zeros > len(df) * 0.5:  # More than 50% zero volume
                    asset_issues.append("Excessive zero volume data")

            # 7. OHLC consistency
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                if invalid_ohlc > 0:
                    asset_issues.append(f"Invalid OHLC relationships: {invalid_ohlc} records")

            # Store validation results
            if asset_issues:
                category_results['issues'].extend([f"{asset_name}: {issue}" for issue in asset_issues])
            else:
                # Track successful validations
                if 'successful_assets' not in category_results:
                    category_results['successful_assets'] = []
                category_results['successful_assets'].append(asset_name)

            # Update quality metrics
            self._update_quality_metrics(df, category_results)

        except Exception as e:
            category_results['issues'].append(f"{asset_name}: Validation error - {str(e)}")
            logger.error(f"Error validating {asset_name}: {e}")

    def _validate_metadata_consistency(self, metadata_files: List[Path], category_results: Dict):
        """Validate metadata consistency"""
        if not metadata_files:
            return

        metadata_consistent = True
        expected_fields = ['symbol', 'records', 'date_range', 'source']

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Check required fields
                missing_fields = [field for field in expected_fields if field not in metadata]
                if missing_fields:
                    category_results['issues'].append(f"Metadata {metadata_file.name} missing fields: {missing_fields}")
                    metadata_consistent = False

            except Exception as e:
                category_results['issues'].append(f"Invalid metadata file {metadata_file.name}: {e}")
                metadata_consistent = False

        if metadata_consistent:
            category_results['metadata_consistent'] = True

    def _update_quality_metrics(self, df: pd.DataFrame, category_results: Dict):
        """Update quality metrics with data from this asset"""
        metrics = category_results['quality_metrics']

        # Record count
        record_count = len(df)
        if 'total_records' not in metrics:
            metrics['total_records'] = 0
        metrics['total_records'] += record_count

        # Data completeness
        if 'close' in df.columns:
            completeness = (1 - df['close'].isnull().mean()) * 100
            if 'avg_completeness' not in metrics:
                metrics['avg_completeness'] = []
            metrics['avg_completeness'].append(completeness)

        # Date range
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            date_range_days = (df.index.max() - df.index.min()).days
            if 'date_ranges' not in metrics:
                metrics['date_ranges'] = []
            metrics['date_ranges'].append(date_range_days)

    def _calculate_category_score(self, category: str, category_results: Dict):
        """Calculate overall quality score for category"""
        score = 100  # Start with perfect score

        # Deduct points for issues
        issues = len(category_results.get('issues', []))
        total_assets = category_results.get('assets_validated', 1)

        # Issue penalties
        if issues > 0:
            issue_penalty = min(50, (issues / total_assets) * 100)  # Max 50 point deduction
            score -= issue_penalty

        # Completeness bonus/penalty
        metrics = category_results.get('quality_metrics', {})
        if 'avg_completeness' in metrics and metrics['avg_completeness']:
            avg_completeness = np.mean(metrics['avg_completeness'])
            if avg_completeness < 95:
                completeness_penalty = (100 - avg_completeness) / 2  # Max 50 point deduction
                score -= completeness_penalty
            elif avg_completeness > 99:
                score += 5  # Bonus for near-perfect data

        # Freshness check
        if 'date_ranges' in metrics and metrics['date_ranges']:
            avg_date_range = np.mean(metrics['date_ranges'])
            if avg_date_range < 365:  # Less than 1 year of data
                score -= 20

        score = max(0, min(100, score))  # Clamp between 0-100
        category_results['quality_score'] = round(score, 1)

        # Quality grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"

        category_results['quality_grade'] = grade

    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {},
            'recommendations': [],
            'critical_issues': []
        }

        total_score = 0
        categories_validated = 0

        for category, results in self.validation_results.items():
            if results.get('status') == 'missing_directory':
                report['critical_issues'].append(f"Missing data directory: {category}")
                continue

            categories_validated += 1
            score = results.get('quality_score', 0)
            total_score += score

            report['summary'][category] = {
                'quality_score': score,
                'quality_grade': results.get('quality_grade', 'F'),
                'assets_validated': results.get('assets_validated', 0),
                'issues_found': len(results.get('issues', [])),
                'total_records': results.get('quality_metrics', {}).get('total_records', 0)
            }

        # Overall score
        if categories_validated > 0:
            overall_score = total_score / categories_validated
            report['overall_quality_score'] = round(overall_score, 1)

            if overall_score >= 85:
                report['overall_grade'] = "Excellent"
            elif overall_score >= 75:
                report['overall_grade'] = "Good"
            elif overall_score >= 65:
                report['overall_grade'] = "Acceptable"
            elif overall_score >= 50:
                report['overall_grade'] = "Poor"
            else:
                report['overall_grade'] = "Critical"

        # Generate recommendations
        self._generate_recommendations(report)

        # Save report
        self._save_validation_report(report)

        # Print summary
        self._print_validation_summary(report)

    def _generate_recommendations(self, report: Dict):
        """Generate recommendations based on validation results"""
        recommendations = []

        for category, summary in report['summary'].items():
            score = summary['quality_score']
            issues = summary.get('issues_found', 0)

            if score < 70:
                recommendations.append(f"🔴 HIGH PRIORITY: {category} data quality is poor ({score}%). Requires immediate attention.")

            if issues > 10:
                recommendations.append(f"🟡 REVIEW: {category} has {issues} data issues. Manual review recommended.")

            if summary.get('assets_validated', 0) < 50:
                recommendations.append(f"🟡 EXPAND: {category} has limited asset coverage. Consider adding more assets.")

        if not recommendations:
            recommendations.append("✅ All data categories show good quality. Ready for model training.")

        report['recommendations'] = recommendations

    def _save_validation_report(self, report: Dict):
        """Save validation report to file"""
        report_file = self.data_dir / "data_validation_report.json"

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    def _print_validation_summary(self, report: Dict):
        """Print formatted validation summary"""
        print("\n" + "="*80)
        print("DATA VALIDATION SUMMARY")
        print("="*80)

        print(f"\nOverall Quality: {report.get('overall_quality_score', 'N/A')}% - {report.get('overall_grade', 'Unknown')}")

        print("\nCategory Breakdown:")
        for category, summary in report['summary'].items():
            print(f"  {category.upper():12}: {summary['quality_score']:5.1f}% ({summary['quality_grade']}) - "
                  f"{summary['assets_validated']} assets, {summary.get('issues_found', 0)} issues")

        print("\nCritical Issues:")
        for issue in report.get('critical_issues', []):
            print(f"  ❌ {issue}")

        print("\nRecommendations:")
        for rec in report.get('recommendations', []):
            print(f"  {rec}")

        print(f"\nDetailed report saved to: {self.data_dir}/data_validation_report.json")


def main():
    """Main execution"""
    validator = DataValidator()

    try:
        results = validator.validate_all_data()

        # Check if data is ready for training
        overall_score = results.get('overall_quality_score', 0)
        if overall_score >= 70:
            print("\n🎉 Data quality is sufficient for model training!")
        else:
            print("\n⚠️  Data quality needs improvement before training.")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

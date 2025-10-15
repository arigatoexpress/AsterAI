#!/usr/bin/env python3
"""
Validate Training Readiness - Ensure Real Data Only
Comprehensive validation before training to prevent synthetic data usage.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingReadinessValidator:
    """
    Comprehensive validation to ensure training readiness with real data only.
    """

    def __init__(self):
        self.data_dirs = {
            'general': Path("data/historical/aster_dex"),
            'aster_native': Path("data/historical/aster_native"),
            'real_aster': Path("data/historical/real_aster_only")
        }

        self.validation_results = {
            'data_quality': {},
            'synthetic_check': {},
            'rate_limits': {},
            'asset_discovery': {},
            'training_readiness': {}
        }

    def run_full_validation(self) -> Dict:
        """Run complete validation suite."""
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING READINESS VALIDATION - REAL DATA ONLY")
        logger.info(f"{'='*80}\n")

        # 1. Check for synthetic data contamination
        self._check_synthetic_data_contamination()

        # 2. Validate data quality
        self._validate_data_quality()

        # 3. Check rate limit compliance
        self._check_rate_limit_compliance()

        # 4. Validate asset discovery
        self._validate_asset_discovery()

        # 5. Check training readiness
        self._check_training_readiness()

        # Generate final report
        return self._generate_validation_report()

    def _check_synthetic_data_contamination(self):
        """CRITICAL: Check for any synthetic data in datasets."""
        logger.info("🔍 Checking for synthetic data contamination...")

        synthetic_found = False

        for data_type, data_dir in self.data_dirs.items():
            summary_file = data_dir / "collection_summary.csv"

            if summary_file.exists():
                summary = pd.read_csv(summary_file)
                synthetic_data = summary[summary['source'] == 'synthetic']

                if not synthetic_data.empty:
                    logger.error(f"❌ SYNTHETIC DATA DETECTED in {data_type} dataset!")
                    synthetic_found = True
                    for _, row in synthetic_data.iterrows():
                        logger.error(f"   • {row['symbol']}: {row['source']}")

                    self.validation_results['synthetic_check'][data_type] = {
                        'status': 'FAILED',
                        'synthetic_symbols': synthetic_data['symbol'].tolist(),
                        'message': f"{len(synthetic_data)} synthetic assets found"
                    }
                else:
                    logger.info(f"  ✅ {data_type}: No synthetic data detected")
                    self.validation_results['synthetic_check'][data_type] = {
                        'status': 'PASSED',
                        'synthetic_symbols': [],
                        'message': "No synthetic data found"
                    }
            else:
                logger.warning(f"  ⚠️  {data_type}: No collection summary found")
                self.validation_results['synthetic_check'][data_type] = {
                    'status': 'UNKNOWN',
                    'message': "No collection summary available"
                }

        if synthetic_found:
            logger.error("🚨 CRITICAL: Synthetic data contamination detected!")
            logger.error("   Training cannot proceed with synthetic data.")
            logger.error("   Use scripts/collect_real_aster_data.py for real data only.")
        else:
            logger.info("✅ Synthetic data check: PASSED - All datasets clean")

    def _validate_data_quality(self):
        """Validate data quality metrics."""
        logger.info("\n🔍 Validating data quality...")

        for data_type, data_dir in self.data_dirs.items():
            quality_metrics = {
                'total_assets': 0,
                'avg_data_points': 0,
                'avg_quality_score': 0,
                'quality_threshold_met': 0,
                'assets_above_threshold': []
            }

            # Check data files
            parquet_files = list(data_dir.glob("*_1h.parquet"))
            quality_metrics['total_assets'] = len(parquet_files)

            if parquet_files:
                quality_scores = []
                data_points = []

                for file in parquet_files:
                    try:
                        df = pd.read_parquet(file)
                        data_points.append(len(df))

                        # Basic quality checks
                        quality_score = self._calculate_basic_quality_score(df)
                        quality_scores.append(quality_score)

                        symbol = file.stem.replace("_1h", "")
                        if quality_score >= 0.7:  # Quality threshold
                            quality_metrics['assets_above_threshold'] += 1
                            quality_metrics['assets_above_threshold'].append(symbol)

                    except Exception as e:
                        logger.warning(f"  Error reading {file}: {e}")

                if quality_scores:
                    quality_metrics['avg_quality_score'] = sum(quality_scores) / len(quality_scores)
                    quality_metrics['avg_data_points'] = sum(data_points) / len(data_points)

            # Determine status
            if quality_metrics['total_assets'] == 0:
                status = "NO_DATA"
                message = f"No data files found in {data_dir}"
            elif quality_metrics['avg_quality_score'] >= 0.7 and quality_metrics['avg_data_points'] >= 1000:
                status = "EXCELLENT"
                message = f"High quality data ({quality_metrics['avg_quality_score']:.2f} avg score)"
            elif quality_metrics['avg_quality_score'] >= 0.5:
                status = "GOOD"
                message = f"Acceptable quality data ({quality_metrics['avg_quality_score']:.2f} avg score)"
            else:
                status = "POOR"
                message = f"Poor quality data ({quality_metrics['avg_quality_score']:.2f} avg score)"

            quality_metrics['status'] = status
            quality_metrics['message'] = message

            self.validation_results['data_quality'][data_type] = quality_metrics

            logger.info(f"  {data_type}: {status} - {message}")

    def _calculate_basic_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate basic data quality score."""
        score = 1.0

        # Check data completeness
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            score *= 0.5

        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.05:
            score *= (1 - missing_pct * 2)

        # Check reasonable data ranges
        if (df['close'] <= 0).any():
            score *= 0.9

        # Check for extreme outliers
        returns = df['close'].pct_change().abs()
        extreme_changes = (returns > 0.5).sum()
        if extreme_changes > len(df) * 0.01:  # More than 1% extreme changes
            score *= 0.95

        return max(0.0, score)

    def _check_rate_limit_compliance(self):
        """Check rate limit test results."""
        logger.info("\n⏱️  Checking rate limit compliance...")

        rate_limit_file = Path("data/aster_rate_limit_test.json")

        if rate_limit_file.exists():
            with open(rate_limit_file) as f:
                rate_data = json.load(f)

            analysis = rate_data.get('analysis', {})

            if analysis.get('overall_status') == 'EXCELLENT':
                status = "EXCELLENT"
                message = "Optimal rate limits identified"
            elif analysis.get('overall_status') == 'GOOD':
                status = "GOOD"
                message = "Acceptable rate limits with minor warnings"
            else:
                status = "CAUTION"
                message = "Rate limit issues detected"

            warnings = analysis.get('warnings', [])
            recommended_settings = analysis.get('recommended_settings', {})

            self.validation_results['rate_limits'] = {
                'status': status,
                'message': message,
                'warnings': warnings,
                'recommended_delay': recommended_settings.get('request_delay_seconds'),
                'recommended_batch_size': recommended_settings.get('optimal_batch_size'),
                'expected_throughput': recommended_settings.get('expected_throughput_req_per_sec')
            }

            logger.info(f"  Rate limits: {status} - {message}")
            if warnings:
                for warning in warnings:
                    logger.warning(f"    ⚠️  {warning}")

        else:
            logger.warning("  ⚠️  No rate limit test results found")
            logger.info("    💡 Run scripts/test_aster_rate_limits.py first")
            self.validation_results['rate_limits'] = {
                'status': 'UNKNOWN',
                'message': 'Rate limit tests not run'
            }

    def _validate_asset_discovery(self):
        """Validate asset discovery results."""
        logger.info("\n📊 Validating asset discovery...")

        discovery_file = Path("data/aster_assets_discovery.json")

        if discovery_file.exists():
            with open(discovery_file) as f:
                discovery_data = json.load(f)

            assets = discovery_data.get('assets', {})
            trainable = sum(1 for a in assets.values() if a.get('is_trainable', False))

            if trainable >= 5:  # At least 5 trainable assets
                status = "EXCELLENT"
                message = f"{trainable} trainable assets discovered"
            elif trainable >= 3:
                status = "GOOD"
                message = f"{trainable} trainable assets discovered"
            elif trainable >= 1:
                status = "MINIMAL"
                message = f"Only {trainable} trainable asset found"
            else:
                status = "INSUFFICIENT"
                message = "No trainable assets found"

            self.validation_results['asset_discovery'] = {
                'status': status,
                'message': message,
                'total_assets': len(assets),
                'trainable_assets': trainable,
                'spot_assets': sum(1 for a in assets.values() if a.get('type') == 'spot'),
                'perpetual_assets': sum(1 for a in assets.values() if a.get('type') == 'perpetual')
            }

            logger.info(f"  Asset discovery: {status} - {message}")

        else:
            logger.warning("  ⚠️  No asset discovery results found")
            logger.info("    💡 Run scripts/discover_aster_assets.py first")
            self.validation_results['asset_discovery'] = {
                'status': 'UNKNOWN',
                'message': 'Asset discovery not run'
            }

    def _check_training_readiness(self):
        """Check overall training readiness."""
        logger.info("\n🎯 Assessing training readiness...")

        # Check if we have at least one good dataset
        data_ready = any(
            result.get('status') in ['EXCELLENT', 'GOOD']
            for result in self.validation_results['data_quality'].values()
        )

        # Check if synthetic data check passed
        synthetic_safe = all(
            result.get('status') != 'FAILED'
            for result in self.validation_results['synthetic_check'].values()
        )

        # Check if we have basic rate limit info
        rate_limits_known = self.validation_results['rate_limits'].get('status') != 'UNKNOWN'

        # Check if we know about assets
        assets_known = self.validation_results['asset_discovery'].get('status') != 'UNKNOWN'

        readiness_score = sum([
            data_ready,
            synthetic_safe,
            rate_limits_known,
            assets_known
        ]) / 4

        if readiness_score >= 0.75:
            status = "READY"
            message = "System ready for training with real data"
        elif readiness_score >= 0.5:
            status = "CONDITIONAL"
            message = "Training possible with some limitations"
        else:
            status = "NOT_READY"
            message = "Training not recommended - missing prerequisites"

        self.validation_results['training_readiness'] = {
            'status': status,
            'message': message,
            'readiness_score': readiness_score,
            'data_ready': data_ready,
            'synthetic_safe': synthetic_safe,
            'rate_limits_known': rate_limits_known,
            'assets_known': assets_known
        }

        logger.info(f"  Training readiness: {status} ({readiness_score:.1%}) - {message}")

    def _generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': self._calculate_overall_status(),
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations(),
            'detailed_results': self.validation_results
        }

        return report

    def _calculate_overall_status(self) -> str:
        """Calculate overall validation status."""
        readiness = self.validation_results['training_readiness']

        if readiness['status'] == 'READY':
            # Check for any critical failures
            synthetic_failed = any(
                result.get('status') == 'FAILED'
                for result in self.validation_results['synthetic_check'].values()
            )

            if synthetic_failed:
                return "CRITICAL_SYNTHETIC_DATA"
            else:
                return "READY_FOR_TRAINING"
        elif readiness['status'] == 'CONDITIONAL':
            return "CONDITIONAL_READINESS"
        else:
            return "NOT_READY_FOR_TRAINING"

    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            'synthetic_data_check': all(
                result.get('status') != 'FAILED'
                for result in self.validation_results['synthetic_check'].values()
            ),
            'data_quality_score': sum(
                result.get('avg_quality_score', 0)
                for result in self.validation_results['data_quality'].values()
            ) / max(1, len(self.validation_results['data_quality'])),
            'rate_limits_tested': self.validation_results['rate_limits'].get('status') != 'UNKNOWN',
            'assets_discovered': self.validation_results['asset_discovery'].get('trainable_assets', 0),
            'training_readiness_score': self.validation_results['training_readiness'].get('readiness_score', 0)
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Synthetic data check
        if any(result.get('status') == 'FAILED' for result in self.validation_results['synthetic_check'].values()):
            recommendations.append("CRITICAL: Remove all synthetic data from datasets")
            recommendations.append("Use scripts/collect_real_aster_data.py for real data only")

        # Data quality
        for data_type, result in self.validation_results['data_quality'].items():
            if result.get('status') == 'NO_DATA':
                recommendations.append(f"Collect {data_type} data first")
            elif result.get('status') in ['POOR']:
                recommendations.append(f"Improve {data_type} data quality or collect better data")

        # Rate limits
        if self.validation_results['rate_limits'].get('status') == 'UNKNOWN':
            recommendations.append("Run rate limit tests: scripts/test_aster_rate_limits.py")

        # Asset discovery
        if self.validation_results['asset_discovery'].get('status') == 'UNKNOWN':
            recommendations.append("Run asset discovery: scripts/discover_aster_assets.py")

        # Training readiness
        readiness = self.validation_results['training_readiness']
        if readiness['status'] != 'READY':
            recommendations.append("Address all validation issues before training")

        if not recommendations:
            recommendations.append("✅ All systems ready for training with real data only")

        return recommendations


def save_validation_report(report: Dict, output_file: str = "training_readiness_validation.json"):
    """Save validation report."""
    output_path = Path("data") / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"💾 Validation report saved to {output_path}")


def print_validation_summary(report: Dict):
    """Print human-readable validation summary."""
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    status = report['overall_status']
    if status == "READY_FOR_TRAINING":
        print("🎉 STATUS: READY FOR TRAINING")
        print("   ✅ All checks passed - safe to proceed with real data only")
    elif status == "CRITICAL_SYNTHETIC_DATA":
        print("🚨 STATUS: CRITICAL - SYNTHETIC DATA DETECTED")
        print("   ❌ Training cannot proceed with synthetic data")
    elif status == "CONDITIONAL_READINESS":
        print("⚠️  STATUS: CONDITIONAL READINESS")
        print("   ⚠️  Training possible but with limitations")
    else:
        print("❌ STATUS: NOT READY FOR TRAINING")
        print("   ❌ Prerequisites not met")

    print(f"\n📊 Summary:")
    summary = report['summary']
    print(f"   Synthetic data safe: {'✅' if summary['synthetic_data_check'] else '❌'}")
    print(f"   Average data quality: {summary['data_quality_score']:.2f}")
    print(f"   Rate limits tested: {'✅' if summary['rate_limits_tested'] else '❌'}")
    print(f"   Trainable assets: {summary['assets_discovered']}")
    print(f"   Readiness score: {summary['training_readiness_score']:.1%}")

    print(f"\n📋 Recommendations:")
    for rec in report['recommendations']:
        print(f"   • {rec}")

    print(f"\n💾 Full report saved to: data/training_readiness_validation.json")


def main():
    """Main execution."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║      Training Readiness Validation - Real Data Only            ║
║    Ensuring Safe Training Without Synthetic Data Contamination ║
╚════════════════════════════════════════════════════════════════╝
    """)

    validator = TrainingReadinessValidator()
    report = validator.run_full_validation()

    save_validation_report(report)
    print_validation_summary(report)

    # Exit with appropriate code
    if report['overall_status'] == "READY_FOR_TRAINING":
        print("\n✅ Validation passed - ready for training!")
        sys.exit(0)
    elif report['overall_status'] == "CRITICAL_SYNTHETIC_DATA":
        print("\n❌ Critical synthetic data detected - fix before training!")
        sys.exit(1)
    else:
        print("\n⚠️  Validation issues found - review recommendations!")
        sys.exit(1)


if __name__ == "__main__":
    main()





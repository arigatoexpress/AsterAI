#!/usr/bin/env python3
"""
Analyze specific data quality issues in detail
"""

import sys
from pathlib import Path
import pandas as pd
import json
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze_crypto_issues():
    """Analyze the 92 issues found in crypto data"""
    print("🔍 Analyzing Crypto Data Issues\n")

    data_dir = Path("data/historical/ultimate_dataset/crypto")
    issues_summary = {
        'missing_required_columns': [],
        'non_numeric_prices': [],
        'high_null_values': [],
        'outliers_detected': [],
        'time_gaps': [],
        'old_data': [],
        'zero_volume': [],
        'invalid_ohlc': [],
        'validation_errors': []
    }

    total_files = 0
    total_records = 0

    for parquet_file in data_dir.glob('*_consolidated.parquet'):
        total_files += 1
        asset_name = parquet_file.stem.replace('_consolidated', '')

        try:
            df = pd.read_parquet(parquet_file)
            total_records += len(df)

            # Check required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues_summary['missing_required_columns'].append(f"{asset_name}: {missing_cols}")

            # Check numeric types
            if 'close' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['close']):
                    issues_summary['non_numeric_prices'].append(asset_name)

            # Check null values
            null_pct = df.isnull().mean() * 100
            high_null_cols = null_pct[null_pct > 5].index.tolist()
            if high_null_cols:
                issues_summary['high_null_values'].append(f"{asset_name}: {high_null_cols}")

            # Check outliers
            if 'close' in df.columns and df['close'].notna().any():
                close_prices = df['close'].dropna()
                if len(close_prices) > 10:  # Need some data
                    Q1 = close_prices.quantile(0.25)
                    Q3 = close_prices.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = close_prices[(close_prices < (Q1 - 3 * IQR)) | (close_prices > (Q3 + 3 * IQR))]
                    if len(outliers) > len(close_prices) * 0.01:  # More than 1% outliers
                        issues_summary['outliers_detected'].append(f"{asset_name}: {len(outliers)} outliers")

            # Check time gaps
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
                time_diffs = df.index.to_series().diff()
                gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]  # Allow 2-hour gaps
                if len(gaps) > 0:
                    issues_summary['time_gaps'].append(f"{asset_name}: {len(gaps)} gaps")

            # Check data freshness
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                days_old = (pd.Timestamp.now() - df.index.max()).days
                if days_old > 7:
                    issues_summary['old_data'].append(f"{asset_name}: {days_old} days old")

            # Check volume
            if 'volume' in df.columns:
                volume_zeros = (df['volume'] == 0).sum()
                if volume_zeros > len(df) * 0.5:  # More than 50% zero volume
                    issues_summary['zero_volume'].append(f"{asset_name}: {volume_zeros}/{len(df)} zero volume")

            # Check OHLC consistency
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                if invalid_ohlc > 0:
                    issues_summary['invalid_ohlc'].append(f"{asset_name}: {invalid_ohlc} invalid records")

        except Exception as e:
            issues_summary['validation_errors'].append(f"{asset_name}: {str(e)}")

    # Print summary
    print(f"📊 Total Assets Analyzed: {total_files}")
    print(f"📈 Total Records: {total_records:,}")
    print(f"❌ Total Issues Found: {sum(len(v) for v in issues_summary.values())}")
    print()

    for issue_type, issues in issues_summary.items():
        if issues:
            issue_name = issue_type.replace('_', ' ').title()
            print(f"🔸 {issue_name}: {len(issues)} assets affected")
            # Show first few examples
            for issue in issues[:3]:
                print(f"   • {issue}")
            if len(issues) > 3:
                print(f"   ... and {len(issues) - 3} more")
            print()

    # Most common issues
    all_issues = []
    for issues in issues_summary.values():
        all_issues.extend([issue.split(':')[0] for issue in issues])

    if all_issues:
        most_common = Counter(all_issues).most_common(10)
        print("🎯 Most Problematic Assets:")
        for asset, count in most_common:
            print(f"   • {asset}: {count} issues")

def check_data_completeness():
    """Check data completeness across different sources"""
    print("\n📋 Data Completeness Analysis\n")

    data_dir = Path("data/historical/ultimate_dataset/crypto")

    source_stats = {
        'aster': {'assets': 0, 'total_records': 0, 'avg_records': 0},
        'cryptocompare': {'assets': 0, 'total_records': 0, 'avg_records': 0},
        'coingecko': {'assets': 0, 'total_records': 0, 'avg_records': 0},
        'yahoo': {'assets': 0, 'total_records': 0, 'avg_records': 0}
    }

    for json_file in data_dir.glob('*_metadata.json'):
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            if 'sources_used' in metadata:
                records = metadata.get('records', 0)
                for source in metadata['sources_used']:
                    if source in source_stats:
                        source_stats[source]['assets'] += 1
                        source_stats[source]['total_records'] += records
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Calculate averages
    for source, stats in source_stats.items():
        if stats['assets'] > 0:
            stats['avg_records'] = stats['total_records'] / stats['assets']

    # Print results
    print("Data Source Coverage:")
    for source, stats in source_stats.items():
        if stats['assets'] > 0:
            print(f"  • {source.capitalize()}: {stats['assets']} assets, "
                  f"{stats['total_records']:,} total records, "
                  f"{stats['avg_records']:.0f} avg per asset")

def main():
    """Main analysis"""
    print("="*80)
    print("CRYPTO DATA QUALITY ANALYSIS")
    print("="*80)

    analyze_crypto_issues()
    check_data_completeness()

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. 🛠️  Fix the 'truth value of DataFrame' error in collection script")
    print("2. 🔄 Re-run collection for assets with missing data")
    print("3. 🧹 Clean outliers and invalid OHLC data")
    print("4. 📅 Ensure time series continuity for all assets")
    print("5. 📊 Add volume data where missing")
    print("\nOverall: Crypto data shows good coverage but needs quality fixes")

if __name__ == "__main__":
    main()


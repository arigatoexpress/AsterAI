#!/usr/bin/env python3
"""
Validate Confluence Feature Generation
Test feature calculations and visualize confluence signals.
"""

print("Script starting...")
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.features.confluence_features import ConfluenceFeatureEngine, ConfluenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_data(data_dir: str = "data/historical/real_aster_only") -> dict:
    """Load collected historical data."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return {}
    
    # Load 1h data for all assets
    asset_data = {}
    for file in data_path.glob("*_1h.parquet"):
        symbol = file.stem.replace("_1h", "")
        try:
            df = pd.read_parquet(file)
            asset_data[symbol] = df
            logger.info(f"✅ Loaded {symbol}: {len(df)} records")
        except Exception as e:
            logger.error(f"❌ Error loading {symbol}: {e}")
    
    return asset_data


def validate_features(asset_data: dict):
    """Validate confluence features."""
    print("\n" + "="*60)
    print("CONFLUENCE FEATURE VALIDATION")
    print("="*60 + "\n")
    
    # Initialize engine
    config = ConfluenceConfig()
    engine = ConfluenceFeatureEngine(config)
    
    # Generate features
    logger.info("Generating confluence features...")
    enriched_data = engine.generate_all_features(asset_data)
    
    # Check results
    for symbol, df in enriched_data.items():
        confluence_cols = [col for col in df.columns if col.startswith('confluence_')]
        logger.info(f"\n{symbol}:")
        logger.info(f"  Confluence features: {len(confluence_cols)}")
        logger.info(f"  Sample features: {confluence_cols[:5]}")
        
        # Check for NaN values
        null_counts = df[confluence_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"  ⚠️  Null values found: {null_counts[null_counts > 0].to_dict()}")
        else:
            logger.info(f"  ✅ No null values")
    
    return enriched_data


def visualize_confluence(enriched_data: dict, symbol: str = "BTCUSDT"):
    """Visualize confluence signals for a symbol."""
    if symbol not in enriched_data:
        logger.error(f"Symbol {symbol} not found in data")
        return
    
    df = enriched_data[symbol].copy()
    
    # Use last 30 days of data for visualization
    df = df.tail(720)  # 30 days * 24 hours
    
    print(f"\nVisualizing confluence for {symbol}...")
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.suptitle(f'Confluence Analysis - {symbol}', fontsize=16, fontweight='bold')
    
    # 1. Price with correlation
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(df.index, df['close'], label='Price', color='blue', linewidth=2)
    if 'confluence_avg_correlation' in df.columns:
        ax1_twin.plot(df.index, df['confluence_avg_correlation'], 
                      label='Avg Correlation', color='orange', alpha=0.7)
        ax1_twin.set_ylabel('Correlation', color='orange')
        ax1_twin.legend(loc='upper right')
    ax1.set_ylabel('Price (USD)', color='blue')
    ax1.set_title('Price & Cross-Asset Correlation')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume confluence
    ax2 = axes[1]
    ax2.bar(df.index, df['volume'], label='Volume', alpha=0.5, color='gray')
    if 'confluence_volume_spike' in df.columns:
        # Highlight confluence periods
        confluence_periods = df[df['confluence_volume_spike'] > 0]
        ax2.scatter(confluence_periods.index, confluence_periods['volume'], 
                   color='red', s=50, label='Volume Confluence', zorder=5)
    ax2.set_ylabel('Volume')
    ax2.set_title('Volume with Confluence Signals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RSI confluence
    ax3 = axes[2]
    if 'rsi' in df.columns:
        ax3.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    if 'confluence_rsi_overbought_confluence' in df.columns:
        ax3_twin = ax3.twinx()
        ax3_twin.fill_between(df.index, 0, df['confluence_rsi_overbought_confluence'], 
                              alpha=0.3, color='red', label='Overbought Confluence')
        ax3_twin.fill_between(df.index, 0, df['confluence_rsi_oversold_confluence'], 
                              alpha=0.3, color='green', label='Oversold Confluence')
        ax3_twin.set_ylabel('Confluence Score')
        ax3_twin.set_ylim([0, 1])
        ax3_twin.legend(loc='upper right')
    ax3.set_ylabel('RSI')
    ax3.set_title('RSI with Confluence')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 4. MACD confluence
    ax4 = axes[3]
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        ax4.plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1.5)
        ax4.plot(df.index, df['macd_signal'], label='Signal', color='red', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    if 'confluence_macd_bullish_confluence' in df.columns:
        ax4_twin = ax4.twinx()
        ax4_twin.fill_between(df.index, 0, df['confluence_macd_bullish_confluence'], 
                              alpha=0.3, color='green', label='Bullish Confluence')
        ax4_twin.set_ylabel('Confluence Score')
        ax4_twin.set_ylim([0, 1])
        ax4_twin.legend(loc='upper right')
    ax4.set_ylabel('MACD')
    ax4.set_title('MACD with Confluence')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Momentum alignment
    ax5 = axes[4]
    if 'confluence_directional_alignment' in df.columns:
        ax5.plot(df.index, df['confluence_directional_alignment'], 
                label='Directional Alignment', color='purple', linewidth=2)
        ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax5.fill_between(df.index, 0.5, df['confluence_directional_alignment'], 
                        where=(df['confluence_directional_alignment'] > 0.5),
                        alpha=0.3, color='green', label='Bullish Alignment')
        ax5.fill_between(df.index, df['confluence_directional_alignment'], 0.5,
                        where=(df['confluence_directional_alignment'] < 0.5),
                        alpha=0.3, color='red', label='Bearish Alignment')
    ax5.set_ylabel('Alignment Score')
    ax5.set_xlabel('Date')
    ax5.set_title('Momentum Alignment Across Assets')
    ax5.set_ylim([0, 1])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("data/historical/real_aster_only/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"confluence_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Visualization saved to {output_file}")
    
    # plt.show()  # Disabled for headless execution


def generate_feature_summary(enriched_data: dict):
    """Generate summary statistics for confluence features."""
    print("\n" + "="*60)
    print("CONFLUENCE FEATURE SUMMARY")
    print("="*60 + "\n")
    
    for symbol, df in enriched_data.items():
        confluence_cols = [col for col in df.columns if col.startswith('confluence_')]
        
        if not confluence_cols:
            continue
        
        print(f"\n{symbol}:")
        print(f"  Features: {len(confluence_cols)}")
        
        # Summary statistics
        summary = df[confluence_cols].describe()
        print(f"\n  Summary Statistics:")
        print(f"  Mean: {summary.loc['mean'].mean():.4f}")
        print(f"  Std: {summary.loc['std'].mean():.4f}")
        print(f"  Max: {summary.loc['max'].max():.4f}")
        
        # Check for highly correlated features (might be redundant)
        corr_matrix = df[confluence_cols].corr().abs()
        high_corr = (corr_matrix > 0.9) & (corr_matrix < 1.0)
        if high_corr.any().any():
            print(f"  ⚠️  Highly correlated features found (>0.9)")


def main():
    """Main execution."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║           Confluence Feature Validation & Visualization        ║
╚════════════════════════════════════════════════════════════════╝
    """)
    print("Starting confluence feature validation...")
    
    # Load data
    logger.info("Loading historical data...")
    asset_data = load_historical_data()
    
    if not asset_data:
        logger.error("No data found. Run scripts/collect_6month_data.py first.")
        return
    
    logger.info(f"Loaded {len(asset_data)} assets")
    
    # Validate features
    enriched_data = validate_features(asset_data)
    
    # Generate summary
    generate_feature_summary(enriched_data)
    
    # Visualize for main assets
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        if symbol in enriched_data:
            try:
                visualize_confluence(enriched_data, symbol)
            except Exception as e:
                logger.error(f"Error visualizing {symbol}: {e}")
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                    ✅ Validation Complete!                     ║
╚════════════════════════════════════════════════════════════════╝

Next steps:
1. Review visualizations in data/historical/aster_dex/visualizations/
2. Proceed to model training (Phase 4)
    """)


if __name__ == "__main__":
    main()



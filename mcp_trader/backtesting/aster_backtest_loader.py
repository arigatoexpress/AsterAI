"""
Aster Backtesting Data Loader
Optimized data loading for backtesting trading strategies on Aster DEX assets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from mcp_trader.data.aster_asset_manager import AsterDataStructure, AsterAssetRegistry

logger = logging.getLogger(__name__)


class AsterBacktestLoader:
    """
    Optimized data loader for backtesting on Aster DEX perpetuals
    Provides fast access to historical data for trading strategy evaluation
    """
    
    def __init__(self, min_quality: float = 0.80):
        """
        Initialize backtest data loader
        
        Args:
            min_quality: Minimum data quality score (0-1) for assets
        """
        self.data_structure = AsterDataStructure()
        self.registry = self.data_structure.registry
        self.min_quality = min_quality
        
        # Load registry if not already loaded
        if not self.registry.assets:
            self.registry.load_registry()
        
        logger.info(f"Aster Backtest Loader initialized")
        logger.info(f"  Total assets: {len(self.registry.assets)}")
        logger.info(f"  Min quality threshold: {self.min_quality:.0%}")
    
    def get_trading_universe(
        self,
        asset_type: str = 'crypto_perp',
        min_quality: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> List[str]:
        """
        Get list of assets for trading universe
        
        Args:
            asset_type: 'crypto_perp' or 'stock_perp' or 'all'
            min_quality: Minimum data quality (uses default if None)
            top_n: Return only top N assets by quality
        
        Returns:
            List of asset symbols
        """
        min_q = min_quality if min_quality is not None else self.min_quality
        
        # Get assets by type
        if asset_type == 'crypto_perp':
            assets = self.registry.get_all_crypto_perps()
        elif asset_type == 'stock_perp':
            assets = self.registry.get_all_stock_perps()
        else:  # 'all'
            assets = list(self.registry.assets.values())
        
        # Filter by quality
        assets = [a for a in assets if a.data_quality_score >= min_q]
        
        # Sort by quality
        assets = sorted(assets, key=lambda x: x.data_quality_score, reverse=True)
        
        # Limit to top N
        if top_n:
            assets = assets[:top_n]
        
        symbols = [a.symbol for a in assets]
        logger.info(f"Trading universe: {len(symbols)} assets ({asset_type}, quality>={min_q:.0%})")
        
        return symbols
    
    def load_backtest_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '1h',
        resample: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for backtesting
        
        Args:
            symbols: List of symbols (None = high-quality assets)
            start_date: Start date 'YYYY-MM-DD' (None = earliest available)
            end_date: End date 'YYYY-MM-DD' (None = latest available)
            timeframe: Target timeframe ('1h', '4h', '1d')
            resample: Whether to resample to target timeframe
        
        Returns:
            Dict of {symbol: DataFrame with OHLCV data}
        """
        # Get symbols if not provided
        if symbols is None:
            symbols = self.get_trading_universe()
        
        logger.info(f"Loading backtest data for {len(symbols)} assets...")
        if start_date:
            logger.info(f"  Start date: {start_date}")
        if end_date:
            logger.info(f"  End date: {end_date}")
        logger.info(f"  Timeframe: {timeframe}")
        
        # Load data
        dataset = {}
        failed = []
        
        for symbol in symbols:
            try:
                df = self.data_structure.load_asset_data(symbol)
                
                if df is None or df.empty:
                    failed.append(symbol)
                    continue
                
                # Filter by date range
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                
                # Resample if requested
                if resample and timeframe != '1h':
                    df = self._resample_ohlcv(df, timeframe)
                
                if len(df) > 0:
                    dataset[symbol] = df
                else:
                    failed.append(symbol)
                    
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                failed.append(symbol)
        
        logger.info(f"✅ Loaded {len(dataset)}/{len(symbols)} assets")
        if failed:
            logger.warning(f"⚠️  Failed: {', '.join(failed[:10])}{' ...' if len(failed) > 10 else ''}")
        
        return dataset
    
    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe"""
        resample_map = {
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1D': '1D',
            '1w': '1W'
        }
        
        freq = resample_map.get(timeframe, '1H')
        
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def get_aligned_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fill_method: str = 'ffill'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data with aligned timestamps across all assets
        Useful for portfolio backtesting
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            fill_method: Method to fill missing data ('ffill', 'bfill', 'drop')
        
        Returns:
            Dict of DataFrames with aligned timestamps
        """
        # Load data
        dataset = self.load_backtest_data(symbols, start_date, end_date)
        
        if not dataset:
            logger.warning("No data loaded")
            return {}
        
        # Find common date range
        min_dates = [df.index.min() for df in dataset.values()]
        max_dates = [df.index.max() for df in dataset.values()]
        
        common_start = max(min_dates)
        common_end = min(max_dates)
        
        logger.info(f"Aligning data: {common_start} to {common_end}")
        
        # Create common index
        first_df = next(iter(dataset.values()))
        freq = pd.infer_freq(first_df.index)
        if freq is None:
            freq = '1H'  # Default to 1 hour
        
        common_index = pd.date_range(start=common_start, end=common_end, freq=freq)
        
        # Align all datasets
        aligned_dataset = {}
        for symbol, df in dataset.items():
            # Reindex to common timestamps
            df_aligned = df.reindex(common_index)
            
            # Handle missing data
            if fill_method == 'ffill':
                df_aligned = df_aligned.ffill()
            elif fill_method == 'bfill':
                df_aligned = df_aligned.bfill()
            elif fill_method == 'drop':
                df_aligned = df_aligned.dropna()
            
            aligned_dataset[symbol] = df_aligned
        
        logger.info(f"✅ Aligned {len(aligned_dataset)} assets to {len(common_index)} timestamps")
        
        return aligned_dataset
    
    def get_train_test_split(
        self,
        symbols: Optional[List[str]] = None,
        train_ratio: float = 0.7,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data into train and test sets for strategy optimization
        
        Args:
            symbols: List of symbols (None = high-quality assets)
            train_ratio: Ratio of data for training (0-1)
            start_date: Start date
            end_date: End date
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Load full dataset
        dataset = self.load_backtest_data(symbols, start_date, end_date)
        
        train_dataset = {}
        test_dataset = {}
        
        for symbol, df in dataset.items():
            split_idx = int(len(df) * train_ratio)
            
            train_dataset[symbol] = df.iloc[:split_idx].copy()
            test_dataset[symbol] = df.iloc[split_idx:].copy()
        
        logger.info(f"Train/Test split:")
        logger.info(f"  Train: {len(train_dataset)} assets, avg {np.mean([len(d) for d in train_dataset.values()]):.0f} candles")
        logger.info(f"  Test: {len(test_dataset)} assets, avg {np.mean([len(d) for d in test_dataset.values()]):.0f} candles")
        
        return train_dataset, test_dataset
    
    def get_asset_info(self, symbol: str) -> Optional[Dict]:
        """Get metadata for a specific asset"""
        metadata = self.registry.get_asset(symbol)
        if metadata:
            return metadata.to_dict()
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of available data"""
        summary = self.registry.get_summary()
        
        # Add date range info
        all_assets = list(self.registry.assets.values())
        if all_assets:
            start_dates = [pd.to_datetime(a.data_start_date) for a in all_assets if a.data_start_date]
            end_dates = [pd.to_datetime(a.data_end_date) for a in all_assets if a.data_end_date]
            
            if start_dates and end_dates:
                summary['date_range'] = {
                    'earliest': str(min(start_dates)),
                    'latest': str(max(end_dates)),
                    'days': (max(end_dates) - min(start_dates)).days
                }
        
        return summary


def create_backtest_loader(min_quality: float = 0.80) -> AsterBacktestLoader:
    """
    Convenience function to create backtest loader
    
    Args:
        min_quality: Minimum data quality score
    
    Returns:
        Initialized AsterBacktestLoader
    """
    loader = AsterBacktestLoader(min_quality=min_quality)
    return loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("ASTER BACKTEST LOADER - DEMO")
    print("="*70)
    
    # Create loader
    loader = create_backtest_loader(min_quality=0.85)
    
    # Get summary
    summary = loader.get_summary()
    print(f"\n📊 Data Summary:")
    print(f"   Total Assets: {summary['total_assets']}")
    print(f"   High Quality: {summary['high_quality_assets']}")
    print(f"   Avg Quality: {summary['avg_quality_score']:.2%}")
    if 'date_range' in summary:
        print(f"   Date Range: {summary['date_range']['earliest'][:10]} to {summary['date_range']['latest'][:10]}")
        print(f"   Days: {summary['date_range']['days']}")
    
    # Get trading universe
    universe = loader.get_trading_universe(top_n=20)
    print(f"\n🔝 Top 20 Assets:")
    print(f"   {', '.join(universe)}")
    
    # Load sample data
    print(f"\n📈 Loading sample data (top 5 assets)...")
    sample_data = loader.load_backtest_data(symbols=universe[:5])
    
    for symbol, df in sample_data.items():
        print(f"   {symbol:6} - {len(df):,} candles ({df.index[0]} to {df.index[-1]})")
    
    print("\n" + "="*70)
    print("✅ Ready for backtesting!")
    print("="*70)


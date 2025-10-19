"""
Standardized backtesting protocol with walk-forward validation.
Implements robust evaluation metrics and cost modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import asyncio
import inspect

from .cost_model import OnChainCostModel, CostModel


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    funding_rate: float = 0.0001  # 0.01% per 8 hours
    max_leverage: float = 3.0
    max_position_size: float = 0.25  # 25% of capital per position
    min_trade_size: float = 10.0  # $10 minimum trade
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_symbol: str = "BTCUSDT"


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    tail_ratio: float
    common_sense_ratio: float
    
    # Additional metrics
    exposure: float  # Average market exposure
    turnover: float  # Portfolio turnover
    fees_paid: float
    slippage_cost: float
    funding_cost: float
    
    # Time series data
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame
    
    # Metadata
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    duration_days: int


class WalkForwardValidator:
    """Walk-forward validation for robust backtesting."""
    
    def __init__(self, 
                 train_period: int = 30,  # 30 days training
                 test_period: int = 7,    # 7 days testing
                 step_size: int = 7,      # 7 days step
                 min_train_periods: int = 14):  # Minimum 14 days training
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.min_train_periods = min_train_periods
    
    def generate_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits for walk-forward validation."""
        splits = []
        
        # Convert to datetime if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('timestamp')
        
        start_date = data.index.min()
        end_date = data.index.max()
        
        current_date = start_date
        
        while current_date + timedelta(days=self.train_period + self.test_period) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_period)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period)
            
            # Check if we have enough data
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) >= self.min_train_periods and len(test_data) > 0:
                splits.append((train_data, test_data))
            
            current_date += timedelta(days=self.step_size)
        
        return splits


class BacktestEngine:
    """Main backtesting engine with comprehensive metrics."""
    
    def __init__(self, config: BacktestConfig = None, cost_model: Optional[CostModel] = None):
        self.config = config or BacktestConfig()
        self.validator = WalkForwardValidator()
        self.cost_model: CostModel = cost_model or OnChainCostModel(client=None)
    
    def run_backtest(self, 
                    model, 
                    data: pd.DataFrame, 
                    symbol: str = "BTCUSDT",
                    walk_forward: bool = True) -> Union[BacktestResult, List[BacktestResult]]:
        """Run backtest on a model."""
        if walk_forward:
            return self._run_walk_forward_backtest(model, data, symbol)
        else:
            return self._run_single_backtest(model, data, symbol)
    
    def _run_single_backtest(self, model, data: pd.DataFrame, symbol: str) -> BacktestResult:
        """Run single backtest."""
        # Generate signals
        signals = model.generate_signals(data)
        
        if not signals:
            return self._create_empty_result(data, symbol)
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'signal': s.signal,
                'confidence': s.confidence,
                'price': s.price
            }
            for s in signals
        ])
        
        # Merge with price data
        data_with_signals = data.merge(signals_df, on='timestamp', how='left')
        data_with_signals['signal'] = data_with_signals['signal'].fillna(0)
        data_with_signals['confidence'] = data_with_signals['confidence'].fillna(0)
        
        # Run simulation
        return self._simulate_trading(data_with_signals, symbol)
    
    def _run_walk_forward_backtest(self, model, data: pd.DataFrame, symbol: str) -> List[BacktestResult]:
        """Run walk-forward backtest."""
        splits = self.validator.generate_splits(data)
        results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            print(f"Running walk-forward split {i+1}/{len(splits)}")
            
            # Fit model on training data
            model.fit(train_data)
            
            # Test on out-of-sample data
            result = self._run_single_backtest(model, test_data, symbol)
            result.metadata = {'split': i+1, 'train_start': train_data.index.min(), 'train_end': train_data.index.max()}
            results.append(result)
        
        return results
    
    def _simulate_trading(self, data: pd.DataFrame, symbol: str) -> BacktestResult:
        """Simulate trading with realistic costs."""
        # Initialize portfolio
        capital = self.config.initial_capital
        position = 0.0  # Current position size
        position_value = 0.0  # Current position value
        equity_curve = []
        trade_log = []
        
        total_fees = 0.0
        total_slippage = 0.0
        total_funding = 0.0
        
        for idx, row in data.iterrows():
            current_price = row['close']
            signal = row['signal']
            confidence = row['confidence']
            
            # Calculate position size based on Kelly Criterion
            if signal != 0:
                # Simple position sizing based on confidence
                target_position = min(
                    confidence * self.config.max_position_size * capital / current_price,
                    self.config.max_position_size * capital / current_price
                )
            else:
                target_position = 0.0
            
            # Calculate trade size
            trade_size = target_position - position
            
            if abs(trade_size) > self.config.min_trade_size / current_price:
                # Execute trade
                is_buy = trade_size > 0
                side = 'buy' if is_buy else 'sell'
                # Estimate execution price via cost model (orderbook walk if available)
                exec_price = self._estimate_execution_price(symbol, current_price, abs(trade_size), side, row)
                trade_value = abs(trade_size) * exec_price
                # Estimate fees via on-chain commission (fallback to defaults)
                commission = self._estimate_fee(symbol, trade_value, is_maker=False, row=row)
                # Slippage cost as difference between exec and base
                slippage_cost = abs(exec_price - current_price) * abs(trade_size)
                
                # Update capital
                capital -= commission
                total_fees += commission
                total_slippage += slippage_cost
                
                # Update position
                position = target_position
                position_value = position * exec_price
                
                # Log trade
                trade_log.append({
                    'timestamp': row['timestamp'],
                    'signal': signal,
                    'price': exec_price,
                    'size': trade_size,
                    'value': trade_value,
                    'commission': commission,
                    'slippage': slippage_cost,
                    'confidence': confidence
                })
            
            # Calculate funding costs (for perpetuals)
            if position != 0:
                funding_cost = abs(position_value) * self.config.funding_rate
                capital -= funding_cost
                total_funding += funding_cost
            
            # Calculate current equity
            current_equity = capital + position_value
            equity_curve.append(current_equity)
        
        # Convert to pandas Series
        equity_series = pd.Series(equity_curve, index=data.index)
        
        # Calculate metrics
        return self._calculate_metrics(
            equity_series, 
            trade_log, 
            data, 
            symbol,
            total_fees,
            total_slippage,
            total_funding
        )

    def _estimate_execution_price(self, symbol: str, base_price: float, quantity: float,
                                  side: str, row: pd.Series) -> float:
        fn = getattr(self.cost_model, 'estimate_execution_price')
        kwargs = {
            'symbol': symbol,
            'base_price': base_price,
            'quantity': quantity,
            'side': side,
            'timestamp': row.get('timestamp', None),
            'orderbook_snapshot': row.get('orderbook', None)
        }
        if inspect.iscoroutinefunction(fn):
            try:
                return asyncio.run(fn(**kwargs))
            except RuntimeError:
                # Already in an event loop; fallback to base price
                return base_price
        else:
            return fn(**kwargs)  # type: ignore

    def _estimate_fee(self, symbol: str, notional: float, is_maker: bool, row: pd.Series) -> float:
        fn = getattr(self.cost_model, 'estimate_fee')
        kwargs = {
            'symbol': symbol,
            'notional': notional,
            'is_maker': is_maker,
            'timestamp': row.get('timestamp', None)
        }
        if inspect.iscoroutinefunction(fn):
            try:
                return asyncio.run(fn(**kwargs))
            except RuntimeError:
                # Already in an event loop; use default commission rate fallback
                return notional * self.config.commission_rate
        else:
            return fn(**kwargs)  # type: ignore
    
    def _calculate_metrics(self, 
                          equity_curve: pd.Series, 
                          trade_log: List[Dict], 
                          data: pd.DataFrame,
                          symbol: str,
                          total_fees: float,
                          total_slippage: float,
                          total_funding: float) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        # Basic returns
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized metrics
        days = (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1
        volatility = returns.std() * np.sqrt(365)
        
        # Risk metrics
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Max drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_duration = self._max_consecutive_drawdown_duration(drawdown_periods)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            winning_trades = trade_df[trade_df['value'] > 0]
            losing_trades = trade_df[trade_df['value'] < 0]
            
            total_trades = len(trade_df)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            win_rate = winning_count / total_trades if total_trades > 0 else 0
            
            avg_win = winning_trades['value'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['value'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = (winning_trades['value'].sum() / abs(losing_trades['value'].sum())) if len(losing_trades) > 0 and losing_trades['value'].sum() != 0 else float('inf')
            
            # Average trade duration (simplified)
            avg_trade_duration = 1.0  # Placeholder
        else:
            total_trades = 0
            winning_count = 0
            losing_count = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Tail ratio
        tail_ratio = abs(returns[returns > 0].mean() / returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 and returns[returns < 0].mean() != 0 else 0
        
        # Common sense ratio (Sharpe * Profit Factor)
        common_sense_ratio = sharpe_ratio * profit_factor if profit_factor != float('inf') else sharpe_ratio
        
        # Exposure and turnover
        exposure = 1.0  # Placeholder - would need position tracking
        turnover = total_trades / days if days > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            tail_ratio=tail_ratio,
            common_sense_ratio=common_sense_ratio,
            exposure=exposure,
            turnover=turnover,
            fees_paid=total_fees,
            slippage_cost=total_slippage,
            funding_cost=total_funding,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            trade_log=pd.DataFrame(trade_log) if trade_log else pd.DataFrame(),
            config=self.config,
            start_date=data['timestamp'].iloc[0],
            end_date=data['timestamp'].iloc[-1],
            duration_days=days
        )
    
    def _max_consecutive_drawdown_duration(self, drawdown_periods: pd.Series) -> int:
        """Calculate maximum consecutive drawdown duration."""
        max_duration = 0
        current_duration = 0
        
        for period in drawdown_periods:
            if period == 1:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _create_empty_result(self, data: pd.DataFrame, symbol: str) -> BacktestResult:
        """Create empty result when no signals generated."""
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            avg_trade_duration=0.0,
            var_95=0.0,
            cvar_95=0.0,
            tail_ratio=0.0,
            common_sense_ratio=0.0,
            exposure=0.0,
            turnover=0.0,
            fees_paid=0.0,
            slippage_cost=0.0,
            funding_cost=0.0,
            equity_curve=pd.Series([self.config.initial_capital]),
            drawdown_curve=pd.Series([0.0]),
            trade_log=pd.DataFrame(),
            config=self.config,
            start_date=data['timestamp'].iloc[0] if len(data) > 0 else datetime.now(),
            end_date=data['timestamp'].iloc[-1] if len(data) > 0 else datetime.now(),
            duration_days=0
        )


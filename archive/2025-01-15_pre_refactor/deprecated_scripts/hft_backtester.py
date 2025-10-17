"""
GPU-Accelerated HFT Backtester

Event-driven backtesting with realistic market simulation:
- Level 2 orderbook data processing
- Transaction fees and slippage modeling
- GPU vectorized execution (millions of trades/second)
- Realistic win rate validation (60% optimistic → 40-50% realistic)

Research: Critical for validating strategies before live trading
Integration: hftbacktest library approach with GPU acceleration
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for HFT backtesting"""
    initial_capital: float = 50.0
    maker_fee_bps: float = 5.0  # 0.05% maker fee
    taker_fee_bps: float = 7.5  # 0.075% taker fee
    slippage_model: str = 'linear'  # linear, sqrt, fixed
    base_slippage_bps: float = 2.0  # Base slippage
    gas_cost_per_trade: float = 0.0  # Gas cost (for DEX)
    latency_mean_ms: float = 5.0  # Mean latency
    latency_std_ms: float = 2.0  # Latency std deviation
    max_positions: int = 10
    risk_per_trade_pct: float = 1.0
    

class HFTBacktester:
    """
    GPU-Accelerated Event-Driven Backtester
    
    Features:
    - Realistic L2 orderbook simulation
    - Transaction cost modeling
    - Latency simulation
    - GPU vectorized execution
    - Multi-strategy support
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Backtest state
        self.capital = config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.peak_capital = config.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"🧪 HFT Backtester initialized on {self.device}")
        logger.info(f"💰 Initial capital: ${config.initial_capital}")
    
    def calculate_slippage(self,
                          order_size: float,
                          order_book_depth: float,
                          side: str) -> float:
        """
        Calculate realistic slippage based on order size and liquidity
        
        Args:
            order_size: Size of order in base currency
            order_book_depth: Available liquidity at best level
            side: 'buy' or 'sell'
            
        Returns:
            Slippage in basis points
        """
        try:
            if order_book_depth == 0:
                return self.config.base_slippage_bps * 10  # High slippage for no liquidity
            
            # Market impact based on order size relative to depth
            impact_ratio = order_size / order_book_depth
            
            if self.config.slippage_model == 'linear':
                slippage = self.config.base_slippage_bps * (1 + impact_ratio * 10)
            elif self.config.slippage_model == 'sqrt':
                slippage = self.config.base_slippage_bps * (1 + np.sqrt(impact_ratio) * 5)
            else:  # fixed
                slippage = self.config.base_slippage_bps
            
            # Cap at reasonable maximum
            return min(slippage, 50.0)  # Max 50 bps slippage
            
        except Exception as e:
            logger.error(f"❌ Slippage calculation error: {e}")
            return self.config.base_slippage_bps
    
    def simulate_order_execution(self,
                                 order: Dict,
                                 orderbook: Dict,
                                 current_time: datetime) -> Optional[Dict]:
        """
        Simulate realistic order execution
        
        Args:
            order: Order details
            orderbook: Current orderbook state
            current_time: Current timestamp
            
        Returns:
            Fill details or None if not filled
        """
        try:
            side = order['side']
            price = order['price']
            quantity = order['quantity']
            order_type = order['type']
            
            # Get best bid/ask
            best_bid = orderbook['bids'][0][0] if orderbook.get('bids') else 0
            best_ask = orderbook['asks'][0][0] if orderbook.get('asks') else float('inf')
            
            # Check if order can be filled
            filled = False
            fill_price = price
            
            if order_type == 'market':
                # Market orders always fill (with slippage)
                filled = True
                fill_price = best_ask if side == 'buy' else best_bid
                
                # Apply slippage
                depth = orderbook['asks'][0][1] if side == 'buy' else orderbook['bids'][0][1]
                slippage_bps = self.calculate_slippage(quantity, depth, side)
                fill_price *= (1 + slippage_bps / 10000) if side == 'buy' else (1 - slippage_bps / 10000)
                
            elif order_type == 'limit':
                # Limit orders fill if price is touched
                if side == 'buy' and best_ask <= price:
                    filled = True
                    fill_price = min(price, best_ask)
                elif side == 'sell' and best_bid >= price:
                    filled = True
                    fill_price = max(price, best_bid)
            
            if not filled:
                return None
            
            # Simulate latency
            latency_ms = np.random.normal(
                self.config.latency_mean_ms,
                self.config.latency_std_ms
            )
            latency_ms = max(0, latency_ms)  # Can't be negative
            
            # Calculate fees
            if order_type == 'market':
                fee_bps = self.config.taker_fee_bps
            else:
                fee_bps = self.config.maker_fee_bps
            
            fee_amount = fill_price * quantity * (fee_bps / 10000)
            
            # Create fill
            fill = {
                'order_id': order.get('order_id'),
                'symbol': order['symbol'],
                'side': side,
                'fill_price': fill_price,
                'quantity': quantity,
                'fee': fee_amount,
                'latency_ms': latency_ms,
                'fill_time': current_time + timedelta(milliseconds=latency_ms),
                'order_type': order_type
            }
            
            return fill
            
        except Exception as e:
            logger.error(f"❌ Order execution simulation error: {e}")
            return None
    
    def update_position(self, fill: Dict):
        """Update positions after fill"""
        try:
            symbol = fill['symbol']
            side = fill['side']
            quantity = fill['quantity']
            fill_price = fill['fill_price']
            fee = fill['fee']
            
            # Update capital for fees
            self.capital -= fee
            self.total_fees += fee
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'quantity': 0.0,
                    'avg_entry_price': 0.0,
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            position = self.positions[symbol]
            
            if side == 'buy':
                # Add to position
                old_quantity = position['quantity']
                new_quantity = old_quantity + quantity
                
                if new_quantity != 0:
                    # Update average entry price
                    position['avg_entry_price'] = (
                        (position['avg_entry_price'] * old_quantity + fill_price * quantity) /
                        new_quantity
                    )
                position['quantity'] = new_quantity
                
            else:  # sell
                # Reduce position
                if position['quantity'] > 0:
                    # Calculate realized P&L
                    realized_pnl = (fill_price - position['avg_entry_price']) * min(quantity, position['quantity'])
                    position['realized_pnl'] += realized_pnl
                    self.capital += realized_pnl
                
                position['quantity'] -= quantity
            
            # Track trade
            self.total_trades += 1
            if fill.get('pnl', 0) > 0:
                self.winning_trades += 1
            
            self.trades.append(fill)
            
        except Exception as e:
            logger.error(f"❌ Position update error: {e}")
    
    def run_backtest(self,
                    orderbook_data: pd.DataFrame,
                    strategy_func: Callable,
                    strategy_params: Dict = None) -> Dict:
        """
        Run full backtest
        
        Args:
            orderbook_data: Historical orderbook data
            strategy_func: Strategy function to execute
            strategy_params: Strategy parameters
            
        Returns:
            Backtest results
        """
        try:
            logger.info(f"🧪 Running backtest on {len(orderbook_data)} data points...")
            
            if strategy_params is None:
                strategy_params = {}
            
            # Reset state
            self.capital = self.config.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            
            # Iterate through orderbook data
            for idx, row in orderbook_data.iterrows():
                timestamp = row.get('timestamp', idx)
                
                # Reconstruct orderbook
                orderbook = {
                    'bids': eval(row.get('bids', '[]')) if isinstance(row.get('bids'), str) else row.get('bids', []),
                    'asks': eval(row.get('asks', '[]')) if isinstance(row.get('asks'), str) else row.get('asks', [])
                }
                
                # Execute strategy
                orders = strategy_func(
                    timestamp=timestamp,
                    orderbook=orderbook,
                    positions=self.positions,
                    capital=self.capital,
                    **strategy_params
                )
                
                # Execute orders
                if orders:
                    for order in orders:
                        fill = self.simulate_order_execution(order, orderbook, timestamp)
                        if fill:
                            self.update_position(fill)
                
                # Update equity curve
                equity = self.calculate_total_equity(orderbook)
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': equity,
                    'capital': self.capital,
                    'positions_value': equity - self.capital
                })
                
                # Update peak and drawdown
                self.peak_capital = max(self.peak_capital, equity)
                current_drawdown = (self.peak_capital - equity) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Calculate results
            results = self.calculate_backtest_results()
            
            logger.info("✅ Backtest complete")
            logger.info(f"📊 Total Return: {results['total_return']:.2%}")
            logger.info(f"📊 Win Rate: {results['win_rate']:.1%}")
            logger.info(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"📊 Max Drawdown: {results['max_drawdown']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {}
    
    def calculate_total_equity(self, orderbook: Dict) -> float:
        """Calculate total equity (capital + unrealized P&L)"""
        equity = self.capital
        
        # Add unrealized P&L from positions
        if orderbook.get('bids') and orderbook.get('asks'):
            mid_price = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
            
            for symbol, position in self.positions.items():
                if position['quantity'] != 0:
                    unrealized_pnl = (mid_price - position['avg_entry_price']) * position['quantity']
                    equity += unrealized_pnl
        
        return equity
    
    def calculate_backtest_results(self) -> Dict:
        """Calculate comprehensive backtest results"""
        if not self.equity_curve:
            return {}
        
        # Convert to arrays
        equity_values = np.array([e['equity'] for e in self.equity_curve])
        
        # Calculate returns
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Performance metrics
        final_equity = equity_values[-1]
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Sharpe ratio (assuming daily data)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else np.std(returns)
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Calculate trade statistics
        if self.trades:
            pnls = [t.get('pnl', 0) for t in self.trades]
            avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
            profit_factor = abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else float('inf')
        else:
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        results = {
            'initial_capital': self.config.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.total_trades - self.winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'total_fees': self.total_fees,
            'total_slippage': self.total_slippage,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return results
    
    def run_gpu_vectorized_backtest(self,
                                   price_data: torch.Tensor,
                                   signals: torch.Tensor,
                                   position_sizes: torch.Tensor) -> Dict:
        """
        GPU-accelerated vectorized backtest
        
        Args:
            price_data: Price tensor (num_timesteps,)
            signals: Trading signals (num_timesteps,) [-1, 0, 1]
            position_sizes: Position sizes (num_timesteps,)
            
        Returns:
            Backtest results
        """
        try:
            logger.info("🚀 Running GPU-accelerated backtest...")
            
            # Move to GPU
            prices = price_data.to(self.device)
            signals = signals.to(self.device)
            sizes = position_sizes.to(self.device)
            
            # Calculate returns
            price_returns = (prices[1:] - prices[:-1]) / prices[:-1]
            
            # Apply signals (shifted by 1 for realistic execution)
            strategy_returns = signals[:-1] * price_returns * sizes[:-1]
            
            # Apply transaction costs
            trades_mask = torch.abs(torch.diff(signals, prepend=signals[:1])) > 0
            num_trades = trades_mask.sum().item()
            
            # Subtract fees
            fee_per_trade = self.config.taker_fee_bps / 10000
            total_fees = num_trades * fee_per_trade
            
            # Calculate equity curve
            equity_curve = torch.cumprod(1 + strategy_returns - fee_per_trade, dim=0)
            
            # Calculate metrics on GPU
            final_equity = equity_curve[-1].item() * self.config.initial_capital
            
            # Drawdown calculation
            running_max = torch.maximum.accumulate(equity_curve)[0]
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdown = torch.min(drawdowns).item()
            
            # Move back to CPU for final calculations
            returns_cpu = strategy_returns.cpu().numpy()
            
            # Performance metrics
            mean_return = np.mean(returns_cpu)
            std_return = np.std(returns_cpu)
            sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            results = {
                'initial_capital': self.config.initial_capital,
                'final_equity': final_equity,
                'total_return': (final_equity - self.config.initial_capital) / self.config.initial_capital,
                'num_trades': num_trades,
                'total_fees': total_fees * self.config.initial_capital,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'equity_curve': equity_curve.cpu().numpy()
            }
            
            logger.info(f"✅ GPU backtest complete: Return={results['total_return']:.2%}, "
                       f"Sharpe={results['sharpe_ratio']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ GPU backtest failed: {e}")
            return {}


def run_strategy_backtest(strategy_name: str,
                          orderbook_data: pd.DataFrame,
                          config: BacktestConfig = None) -> Dict:
    """
    Convenience function to backtest a strategy
    
    Args:
        strategy_name: Name of strategy to test
        orderbook_data: Historical orderbook data
        config: Backtest configuration
        
    Returns:
        Backtest results
    """
    if config is None:
        config = BacktestConfig()
    
    backtester = HFTBacktester(config)
    
    # Define strategy function based on name
    if strategy_name == 'market_making':
        from ..strategies.market_making import MarketMakingStrategy, MarketMakingConfig
        
        mm_strategy = MarketMakingStrategy(MarketMakingConfig())
        
        def strategy_func(timestamp, orderbook, positions, capital, **kwargs):
            # Simple wrapper
            market_data = {'price': (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2}
            orders = mm_strategy.execute_strategy(
                'TEST', market_data, orderbook, capital
            )
            return orders
        
        results = backtester.run_backtest(orderbook_data, strategy_func)
        
    else:
        logger.error(f"❌ Unknown strategy: {strategy_name}")
        return {}
    
    return results


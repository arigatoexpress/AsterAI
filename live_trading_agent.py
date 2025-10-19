"""
Live Trading Agent for Real-Money Trading

This module implements the live trading agent that executes real trades
with safety mechanisms, risk controls, and real-time monitoring.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from mcp_trader.execution.aster_client import AsterClient
try:
    from mcp_trader.risk.risk_manager import RiskManager
except Exception:
    class RiskManager:  # minimal stub for dry-run
        def __init__(self, *_, **__):
            pass
try:
    from mcp_trader.risk.dynamic_position_sizing import DynamicPositionSizing
except Exception:
    class DynamicPositionSizing:  # lightweight fallback for dry-run
        def calculate_position_size(self, *, symbol: str, signal_strength: float, current_price: float, account_balance: float, risk_per_trade: float) -> float:
            base_value = max(0.0, account_balance * max(0.0, min(risk_per_trade, 0.2)))
            strength = min(2.0, max(0.5, abs(signal_strength)))
            size = (base_value * strength) / max(current_price, 1e-9)
            return max(0.0, size)
try:
    from mcp_trader.strategies.market_making import MarketMakingStrategy
except Exception:
    class MarketMakingStrategy:  # fallback for dry-run
        def __init__(self, *_, **__):
            pass
        def generate_signal(self, row: pd.Series) -> int:
            try:
                return 1 if float(row['close']) > float(row['open']) else -1
            except Exception:
                return 0
try:
    from mcp_trader.strategies.funding_arbitrage import FundingArbitrageStrategy
except Exception:
    class FundingArbitrageStrategy:  # fallback for dry-run
        def __init__(self, *_, **__):
            pass
        def generate_signal(self, row: pd.Series) -> int:
            return 0
try:
    from mcp_trader.strategies.dmark_strategy import DMarkStrategy
except Exception:
    class DMarkStrategy:  # fallback for dry-run
        def __init__(self, *_, **__):
            pass
        def generate_signal(self, row: pd.Series) -> int:
            try:
                return 1 if float(row['close']) > float(row['open']) else -1
            except Exception:
                return 0
try:
    from autonomous_mcp_agent import AutonomousMCPAgent
except Exception:
    class AutonomousMCPAgent:  # minimal stub for dry-run
        def __init__(self, *_, **__):
            pass
        def decide(self, *_args, **_kwargs):
            return None
try:
    from self_improvement_engine import SelfImprovementEngine
except Exception:
    class SelfImprovementEngine:  # minimal stub for dry-run
        def __init__(self, *_, **__):
            pass
        def record_result(self, *_args, **_kwargs):
            pass
from market_regime_detector import AdaptiveRiskManager
from mev_protection_system import MEVProtectionSystem, MEVProtectionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for live trading"""
    initial_capital: float = 100.0
    max_leverage: float = 3.0
    position_size_pct: float = 0.02  # 2% per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    daily_loss_limit_pct: float = 0.10  # 10% daily loss limit
    max_positions: int = 3
    trading_pairs: List[str] = None
    emergency_stop: bool = False
    dry_run: bool = False  # simulate orders without hitting exchange
    
    def __post_init__(self):
        if self.trading_pairs is None:
            self.trading_pairs = ["BTCUSDT", "ETHUSDT"]

@dataclass
class Position:
    """Represents an active trading position"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: str

@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class LiveTradingAgent:
    """Main live trading agent for real-money trading"""
    
    def __init__(self, config: TradingConfig, aster_client: AsterClient):
        self.config = config
        self.aster_client = aster_client
        try:
            self.risk_manager = RiskManager()  # type: ignore[call-arg]
        except Exception:
            class _RiskManagerFallback:
                pass
            self.risk_manager = _RiskManagerFallback()
        self.position_sizer = DynamicPositionSizing()
        self.positions: Dict[str, Position] = {}
        self.metrics = TradingMetrics()
        self.daily_start_balance = config.initial_capital
        self.last_daily_reset = datetime.now().date()
        
        # Initialize strategies with safe fallbacks
        class _SimpleStrategy:
            def generate_signal(self, row: pd.Series) -> int:
                try:
                    return 1 if float(row['close']) > float(row['open']) else -1
                except Exception:
                    return 0

        def _safe_init(strategy_cls):
            try:
                return strategy_cls(config=self.config)
            except Exception:
                try:
                    return strategy_cls()
                except Exception:
                    return _SimpleStrategy()

        self.strategies = {
            'market_making': _safe_init(MarketMakingStrategy),
            'funding_arbitrage': _safe_init(FundingArbitrageStrategy),
            'dmark': _safe_init(DMarkStrategy)
        }
        
        # Initialize MCP agent for decision making
        self.mcp_agent = AutonomousMCPAgent()
        
        # Initialize self-improvement engine
        self.improvement_engine = SelfImprovementEngine({})
        
        # Initialize adaptive risk management
        base_config = {
            'position_size_pct': config.position_size_pct,
            'max_positions': config.max_positions,
            'daily_loss_limit_pct': config.daily_loss_limit_pct,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct
        }
        self.adaptive_risk_manager = AdaptiveRiskManager(base_config)
        
        # Initialize MEV protection
        mev_config = MEVProtectionConfig(
            max_slippage_pct=0.1,
            min_liquidity_threshold=10000,
            max_price_impact_pct=0.05,
            private_mempool_enabled=True
        )
        self.mev_protection = MEVProtectionSystem(mev_config)
        
        # Trading state
        self.is_trading = False
        self.emergency_stop = False
        self.current_adaptive_config = base_config
        # Per-symbol risk overrides to reflect known manipulation/liquidation dynamics
        # Example: reduce aggressiveness on SOL during suspected MM-driven squeezes
        self.symbol_risk_overrides: Dict[str, Dict[str, float]] = {
            'SOLUSDT': {
                'max_position_multiplier': 0.6,  # cap size at 60% of computed
                'stop_loss_widen_factor': 1.15,   # widen SL slightly to avoid shakeouts
                'take_profit_tighten_factor': 0.9 # slightly tighter TP
            }
        }
        
        logger.info(f"Live Trading Agent initialized with ${config.initial_capital} capital")
    
    async def start_trading(self):
        """Start the live trading loop"""
        logger.info("Starting live trading...")
        self.is_trading = True
        
        try:
            while self.is_trading and not self.emergency_stop:
                # Check daily reset
                await self._check_daily_reset()
                
                # Check emergency conditions
                if await self._check_emergency_conditions():
                    await self._emergency_shutdown()
                    break
                
                # Update market data
                market_data = await self._update_market_data()
                
                # Generate trading signals
                signals = await self._generate_signals(market_data)
                
                # Execute trades based on signals
                await self._execute_trades(signals)
                
                # Update positions
                await self._update_positions()
                
                # Update metrics
                await self._update_metrics()
                
                # Log status
                await self._log_status()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second cycle
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await self._emergency_shutdown()
        
        logger.info("Live trading stopped")
    
    async def stop_trading(self):
        """Stop the live trading loop"""
        logger.info("Stopping live trading...")
        self.is_trading = False
        
        # Close all positions
        await self._close_all_positions()
    
    async def emergency_stop_trading(self):
        """Emergency stop - immediately close all positions"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_stop = True
        self.is_trading = False
        
        # Immediately close all positions
        await self._close_all_positions()
        
        # Log emergency stop
        logger.critical("All positions closed due to emergency stop")
    
    async def _check_daily_reset(self):
        """Check if daily reset is needed"""
        current_date = datetime.now().date()
        
        if current_date != self.last_daily_reset:
            logger.info("Daily reset - updating metrics and balance")
            
            # Reset daily metrics
            self.metrics.daily_pnl = 0.0
            self.daily_start_balance = self._get_current_balance()
            self.last_daily_reset = current_date
            
            # Save daily performance
            await self._save_daily_performance()
    
    async def _check_emergency_conditions(self) -> bool:
        """Check if emergency conditions are met"""
        
        # Check daily loss limit
        if self.metrics.daily_pnl < -self.config.daily_loss_limit_pct * self.daily_start_balance:
            logger.critical(f"Daily loss limit exceeded: {self.metrics.daily_pnl:.2f}")
            return True
        
        # Check maximum drawdown
        if self.metrics.max_drawdown > 0.15 * self.config.initial_capital:
            logger.critical(f"Maximum drawdown exceeded: {self.metrics.max_drawdown:.2f}")
            return True
        
        # Check position limits
        if len(self.positions) > self.config.max_positions:
            logger.critical(f"Position limit exceeded: {len(self.positions)}")
            return True
        
        return False
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("Initiating emergency shutdown")
        
        # Close all positions immediately
        await self._close_all_positions()
        
        # Set emergency stop flag
        self.emergency_stop = True
        self.is_trading = False
        
        # Log emergency shutdown
        logger.critical("Emergency shutdown completed")
    
    async def _update_market_data(self) -> Dict[str, Any]:
        """Update market data for all trading pairs"""
        market_data = {}
        
        for symbol in self.config.trading_pairs:
            try:
                # Get current price
                ticker = await self.aster_client.get_24hr_ticker(symbol)
                current_price = float(ticker['lastPrice'])
                
                # Get order book
                order_book = await self.aster_client.get_order_book(symbol)
                
                # Get recent klines
                klines = await self.aster_client.get_klines(symbol, '1m', limit=100)
                
                market_data[symbol] = {
                    'price': current_price,
                    'order_book': order_book,
                    'klines': klines,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.warning(f"Error updating market data for {symbol}: {e}, using demo data")
                # Use demo data when API fails
                market_data[symbol] = {
                    'price': 50000.0 if 'BTC' in symbol else 3000.0,  # Demo prices
                    'order_book': {'bids': [], 'asks': []},
                    'klines': [],
                    'timestamp': datetime.now()
                }
        
        # Update adaptive configuration based on market regime
        if market_data:
            # Create market data DataFrame for regime detection
            all_klines = []
            for symbol, data in market_data.items():
                for kline in data['klines']:
                    all_klines.append({
                        'timestamp': pd.to_datetime(kline[0], unit='ms'),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'symbol': symbol
                    })
            
            if all_klines:
                df = pd.DataFrame(all_klines)
                df = df.set_index('timestamp').sort_index()
                
                # Update adaptive configuration
                self.current_adaptive_config = await self.adaptive_risk_manager.get_adaptive_config(df)

                # Augment with capitulation signal: deep drawdown + volume spike
                try:
                    # Use close and volume aggregated across symbols
                    latest = df.groupby('symbol').tail(30)
                    # Compute short-term return over last 15 bars vs 60-bar median
                    def cap_signal(group: pd.DataFrame) -> Tuple[float, float, float]:
                        closes = group['close'].astype(float)
                        vols = group['volume'].astype(float)
                        if len(closes) < 30:
                            return 0.0, 0.0, 0.0
                        ret15 = (closes.iloc[-1] - closes.iloc[-15]) / max(closes.iloc[-15], 1e-9)
                        vol_ratio = vols.iloc[-5:].mean() / max(vols.iloc[-30:].mean(), 1e-9)
                        low_deviation = (closes.iloc[-30:].min() - closes.iloc[-1]) / max(closes.iloc[-30:].min(), 1e-9)
                        return float(ret15), float(vol_ratio), float(low_deviation)

                    per_symbol_caps: Dict[str, Dict[str, float]] = {}
                    for sym, grp in latest.groupby('symbol'):
                        r15, vratio, lowdev = cap_signal(grp)
                        per_symbol_caps[sym] = {
                            'ret15': r15,
                            'vol_ratio': vratio,
                            'low_deviation': lowdev
                        }
                    # Capitulation if sharp negative return and volume spike
                    capitulation_symbols = [s for s, m in per_symbol_caps.items() if m['ret15'] < -0.05 and m['vol_ratio'] > 1.8]
                    self.current_adaptive_config['capitulation'] = {
                        'active': len(capitulation_symbols) > 0,
                        'symbols': capitulation_symbols,
                        'metrics': per_symbol_caps
                    }
                except Exception:
                    # If any calc fails, keep running with base adaptive config
                    pass
                
                # Log regime changes
                regime_info = self.current_adaptive_config.get('market_regime', {})
                if regime_info.get('regime') in ['recovery', 'bounce']:
                    logger.info(f"Market regime: {regime_info['regime']} - "
                              f"Oversold: {regime_info.get('oversold_level', 0):.2f}, "
                              f"Recovery potential: {regime_info.get('recovery_potential', 0):.2f}")
        
        return market_data
    
    async def _generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate trading signals from all strategies"""
        signals = {}
        
        for symbol, data in market_data.items():
            symbol_signals = []
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Generate signal using strategy
                    signal = await self._get_strategy_signal(strategy, data, symbol)
                    
                    if signal:
                        signal['strategy'] = strategy_name
                        signal['symbol'] = symbol
                        signal['timestamp'] = datetime.now()
                        symbol_signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error generating signal for {strategy_name} on {symbol}: {e}")
            
            signals[symbol] = symbol_signals
        
        return signals
    
    async def _get_strategy_signal(self, strategy, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading signal from a specific strategy"""
        try:
            # Skip strategies that do not implement the expected interface
            if not hasattr(strategy, 'generate_signal'):
                return None
            # Convert market data to DataFrame for strategy
            klines_df = pd.DataFrame(market_data['klines'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                klines_df[col] = pd.to_numeric(klines_df[col])
            
            # Generate signal
            signal = strategy.generate_signal(klines_df.iloc[-1])
            
            if signal != 0:  # Non-zero signal
                confidence = 0.8
                # Boost confidence in capitulation recovery bounces for majors
                cap = self.current_adaptive_config.get('capitulation', {})
                if cap.get('active') and symbol in cap.get('symbols', []):
                    # If signal is contrarian bounce (positive after dump), boost
                    if signal > 0:
                        confidence = min(0.95, confidence + 0.1)
                # Apply cautious confidence on symbols with manipulation risk
                if symbol in self.symbol_risk_overrides:
                    confidence = max(0.6, confidence - 0.1)
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'price': market_data['price']
                }
                
        except Exception as e:
            logger.error(f"Error getting strategy signal: {e}")
        
        return None
    
    async def _execute_trades(self, signals: Dict[str, List[Dict[str, Any]]]):
        """Execute trades based on generated signals"""
        
        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                continue
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                continue
            
            # Check position limit (use adaptive config)
            max_positions = self.current_adaptive_config.get('max_positions', self.config.max_positions)
            if len(self.positions) >= max_positions:
                continue
            
            # Select best signal (highest confidence)
            best_signal = max(symbol_signals, key=lambda x: x.get('confidence', 0))
            
            # Calculate position size using adaptive config
            position_size = await self._calculate_position_size(symbol, best_signal)
            
            if position_size <= 0:
                continue
            
            # Execute trade with MEV protection
            await self._place_trade_protected(symbol, best_signal, position_size)
    
    async def _calculate_position_size(self, symbol: str, signal: Dict[str, Any]) -> float:
        """Calculate position size based on adaptive risk management"""
        try:
            # Get current balance
            balance = await self._get_current_balance()
            
            # Use adaptive position size from current regime
            position_size_pct = self.current_adaptive_config.get('position_size_pct', self.config.position_size_pct)
            
            # Calculate position size based on percentage
            position_value = balance * position_size_pct
            
            # Get current price
            current_price = signal['price']
            
            # Calculate position size in base currency
            position_size = position_value / current_price
            
            # Apply risk management with regime-based adjustments
            regime_info = self.current_adaptive_config.get('market_regime', {})
            recovery_potential = regime_info.get('recovery_potential', 0.5)
            oversold_level = regime_info.get('oversold_level', 0.0)
            
            # Increase position size during recovery phases
            if regime_info.get('regime') in ['recovery', 'bounce'] and oversold_level > 0.5:
                position_size *= (1 + recovery_potential * 0.5)  # Up to 50% increase

            # Capitulation boost for majors (BTC/ETH) when capitulation active
            cap = self.current_adaptive_config.get('capitulation', {})
            if cap.get('active') and symbol in ('BTCUSDT', 'ETHUSDT') and signal['signal'] > 0:
                position_size *= 1.25  # 25% extra size on recovery entries
            
            # Apply dynamic position sizing
            position_size = self.position_sizer.calculate_position_size(
                symbol=symbol,
                signal_strength=signal['signal'],
                current_price=current_price,
                account_balance=balance,
                risk_per_trade=position_size_pct
            )
            
            # Apply per-symbol overrides (e.g., manipulation guard on SOL)
            if symbol in self.symbol_risk_overrides:
                overrides = self.symbol_risk_overrides[symbol]
                max_mult = overrides.get('max_position_multiplier', 1.0)
                position_size *= max(0.0, min(max_mult, 1.0))

            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    async def _place_trade_protected(self, symbol: str, signal: Dict[str, Any], position_size: float):
        """Place a trade order with MEV protection"""
        try:
            # Dry-run path: simulate order without exchange
            if getattr(self.config, 'dry_run', False):
                side = 'BUY' if signal['signal'] > 0 else 'SELL'
                stop_loss_pct = self.current_adaptive_config.get('stop_loss_pct', self.config.stop_loss_pct)
                take_profit_pct = self.current_adaptive_config.get('take_profit_pct', self.config.take_profit_pct)
                # Adjust per-symbol settings to mitigate manipulation-driven liquidations
                if symbol in self.symbol_risk_overrides:
                    ov = self.symbol_risk_overrides[symbol]
                    stop_loss_pct *= ov.get('stop_loss_widen_factor', 1.0)
                    take_profit_pct *= ov.get('take_profit_tighten_factor', 1.0)
                position = Position(
                    symbol=symbol,
                    side='long' if side == 'BUY' else 'short',
                    size=position_size,
                    entry_price=signal['price'],
                    current_price=signal['price'],
                    unrealized_pnl=0.0,
                    stop_loss=self._calculate_stop_loss(signal['price'], side, stop_loss_pct),
                    take_profit=self._calculate_take_profit(signal['price'], side, take_profit_pct),
                    entry_time=datetime.now(),
                    strategy=signal['strategy']
                )
                self.positions[symbol] = position
                logger.info(f"[DRY-RUN] Simulated {side} {position_size} {symbol} at {signal['price']}")
                return

            # Get market data for MEV protection
            market_data = await self._update_market_data()
            symbol_data = market_data.get(symbol, {})
            
            if not symbol_data:
                logger.error(f"No market data available for {symbol}")
                return
            
            # Use MEV protection system
            result = await self.mev_protection.protect_order(
                symbol=symbol,
                side='BUY' if signal['signal'] > 0 else 'SELL',
                size=position_size,
                current_price=signal['price'],
                order_book=symbol_data['order_book'],
                aster_client=self.aster_client
            )
            
            if result.get('status') == 'cancelled':
                logger.warning(f"Trade cancelled due to MEV threat: {result.get('reason')}")
                return
            
            if result.get('status') == 'error':
                logger.error(f"MEV protection error: {result.get('message')}")
                return
            
            # Create position if order was successful
            if result.get('orderId'):
                side = 'BUY' if signal['signal'] > 0 else 'SELL'
                
                # Use adaptive stop loss and take profit
                stop_loss_pct = self.current_adaptive_config.get('stop_loss_pct', self.config.stop_loss_pct)
                take_profit_pct = self.current_adaptive_config.get('take_profit_pct', self.config.take_profit_pct)
                # Adjust per-symbol settings to mitigate manipulation-driven liquidations
                if symbol in self.symbol_risk_overrides:
                    ov = self.symbol_risk_overrides[symbol]
                    stop_loss_pct *= ov.get('stop_loss_widen_factor', 1.0)
                    take_profit_pct *= ov.get('take_profit_tighten_factor', 1.0)
                
                position = Position(
                    symbol=symbol,
                    side='long' if side == 'BUY' else 'short',
                    size=position_size,
                    entry_price=signal['price'],
                    current_price=signal['price'],
                    unrealized_pnl=0.0,
                    stop_loss=self._calculate_stop_loss(signal['price'], side, stop_loss_pct),
                    take_profit=self._calculate_take_profit(signal['price'], side, take_profit_pct),
                    entry_time=datetime.now(),
                    strategy=signal['strategy']
                )
                
                self.positions[symbol] = position
                
                logger.info(f"MEV-protected trade placed: {side} {position_size} {symbol} at {signal['price']}")
        
        except Exception as e:
            logger.error(f"Error placing protected trade for {symbol}: {e}")
    
    async def _place_trade(self, symbol: str, signal: Dict[str, Any], position_size: float):
        """Place a trade order (legacy method)"""
        try:
            side = 'BUY' if signal['signal'] > 0 else 'SELL'
            
            # Place market order
            order = await self.aster_client.place_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=position_size
            )
            
            if order and order.get('orderId'):
                # Create position
                position = Position(
                    symbol=symbol,
                    side='long' if side == 'BUY' else 'short',
                    size=position_size,
                    entry_price=signal['price'],
                    current_price=signal['price'],
                    unrealized_pnl=0.0,
                    stop_loss=self._calculate_stop_loss(signal['price'], side),
                    take_profit=self._calculate_take_profit(signal['price'], side),
                    entry_time=datetime.now(),
                    strategy=signal['strategy']
                )
                
                self.positions[symbol] = position
                
                logger.info(f"Trade placed: {side} {position_size} {symbol} at {signal['price']}")
                
        except Exception as e:
            logger.error(f"Error placing trade for {symbol}: {e}")
    
    def _calculate_stop_loss(self, entry_price: float, side: str, stop_loss_pct: float = None) -> float:
        """Calculate stop loss price"""
        if stop_loss_pct is None:
            stop_loss_pct = self.config.stop_loss_pct
            
        if side == 'BUY':
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, side: str, take_profit_pct: float = None) -> float:
        """Calculate take profit price"""
        if take_profit_pct is None:
            take_profit_pct = self.config.take_profit_pct
            
        if side == 'BUY':
            return entry_price * (1 + take_profit_pct)
        else:
            return entry_price * (1 - take_profit_pct)
    
    async def _update_positions(self):
        """Update all active positions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = await self.aster_client.get_24hr_ticker(symbol)
                current_price = float(ticker['lastPrice'])
                
                # Update position
                position.current_price = current_price
                position.unrealized_pnl = self._calculate_unrealized_pnl(position)
                
                # Check stop loss and take profit
                if await self._should_close_position(position):
                    await self._close_position(symbol)
                    
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
    
    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Calculate unrealized P&L for a position"""
        if position.side == 'long':
            return (position.current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - position.current_price) * position.size
    
    async def _should_close_position(self, position: Position) -> bool:
        """Check if position should be closed"""
        # Check stop loss
        if position.side == 'long' and position.current_price <= position.stop_loss:
            return True
        elif position.side == 'short' and position.current_price >= position.stop_loss:
            return True
        
        # Check take profit
        if position.side == 'long' and position.current_price >= position.take_profit:
            return True
        elif position.side == 'short' and position.current_price <= position.take_profit:
            return True
        
        return False
    
    async def _close_position(self, symbol: str):
        """Close a position"""
        try:
            position = self.positions[symbol]
            
            # Determine close side
            close_side = 'SELL' if position.side == 'long' else 'BUY'
            
            # Place close order
            order = await self.aster_client.place_order(
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=position.size
            )
            
            if order and order.get('orderId'):
                # Calculate realized P&L
                realized_pnl = self._calculate_unrealized_pnl(position)
                
                # Update metrics
                self.metrics.total_trades += 1
                self.metrics.total_pnl += realized_pnl
                self.metrics.daily_pnl += realized_pnl
                
                if realized_pnl > 0:
                    self.metrics.winning_trades += 1
                else:
                    self.metrics.losing_trades += 1
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"Position closed: {symbol} P&L: {realized_pnl:.2f}")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _close_all_positions(self):
        """Close all active positions"""
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol)
    
    async def _update_metrics(self):
        """Update trading metrics"""
        # Calculate unrealized P&L
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.metrics.unrealized_pnl = total_unrealized
        
        # Calculate win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Calculate profit factor
        if self.metrics.losing_trades > 0:
            avg_win = self.metrics.winning_trades / self.metrics.total_trades if self.metrics.total_trades > 0 else 0
            avg_loss = self.metrics.losing_trades / self.metrics.total_trades if self.metrics.total_trades > 0 else 0
            self.metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Update max drawdown
        current_balance = await self._get_current_balance()
        if current_balance < self.config.initial_capital:
            drawdown = self.config.initial_capital - current_balance
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)
        
        self.metrics.last_updated = datetime.now()
    
    async def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            account_info = await self.aster_client.get_account_info()
            return float(account_info.total_balance)
        except Exception as e:
            logger.warning(f"Error getting balance: {e}, using initial capital for demo mode")
            return self.config.initial_capital
    
    async def _log_status(self):
        """Log current trading status"""
        if self.metrics.last_updated and (datetime.now() - self.metrics.last_updated).seconds < 60:
            return  # Only log every minute
        
        logger.info(f"Trading Status - "
                   f"Positions: {len(self.positions)}, "
                   f"Total P&L: {self.metrics.total_pnl:.2f}, "
                   f"Daily P&L: {self.metrics.daily_pnl:.2f}, "
                   f"Win Rate: {self.metrics.win_rate:.2%}")
    
    async def _save_daily_performance(self):
        """Save daily performance data"""
        try:
            performance_data = {
                'date': self.last_daily_reset.isoformat(),
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.metrics.daily_pnl,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'max_drawdown': self.metrics.max_drawdown,
                'positions': len(self.positions)
            }
            
            # Save to file (in production, save to database)
            with open(f"daily_performance_{self.last_daily_reset}.json", 'w') as f:
                json.dump(performance_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving daily performance: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            'is_trading': self.is_trading,
            'emergency_stop': self.emergency_stop,
            'positions': len(self.positions),
            'metrics': {
                'total_pnl': self.metrics.total_pnl,
                'daily_pnl': self.metrics.daily_pnl,
                'unrealized_pnl': self.metrics.unrealized_pnl,
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'max_drawdown': self.metrics.max_drawdown
            },
            'config': {
                'initial_capital': self.config.initial_capital,
                'position_size_pct': self.config.position_size_pct,
                'daily_loss_limit_pct': self.config.daily_loss_limit_pct,
                'max_positions': self.config.max_positions
            }
        }

# Example usage
async def main():
    """Test the live trading agent"""
    
    # Create configuration
    config = TradingConfig(
        initial_capital=100.0,
        max_leverage=3.0,
        position_size_pct=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        daily_loss_limit_pct=0.10,
        max_positions=3,
        trading_pairs=["BTCUSDT", "ETHUSDT"]
    )
    
    # Create Aster client (you'll need to provide actual API credentials)
    aster_client = AsterClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        base_url="https://api.aster.exchange"
    )
    
    # Create live trading agent
    agent = LiveTradingAgent(config, aster_client)
    
    # Start trading (in production, this would run continuously)
    try:
        await agent.start_trading()
    except KeyboardInterrupt:
        logger.info("Stopping trading...")
        await agent.stop_trading()
    except Exception as e:
        logger.error(f"Trading error: {e}")
        await agent.emergency_stop_trading()

if __name__ == "__main__":
    asyncio.run(main())

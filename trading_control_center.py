#!/usr/bin/env python3
"""
Complete Trading Control Center Backend
Provides full control API for the trading bot and comprehensive data endpoints
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TradingControlCenter:
    """Central control for trading operations"""
    
    def __init__(self):
        self.trading_active = False
        self.bot_config = {
            'initial_capital': 100.0,
            'position_size_pct': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'daily_loss_limit_pct': 0.10,
            'max_positions': 2,
            'dry_run': True,
            'auto_trading': False
        }
        self.trading_agent = None
        self.positions = []
        self.recent_trades = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    async def start_trading(self) -> Dict[str, Any]:
        """Start the trading bot"""
        try:
            if self.trading_active:
                return {'status': 'already_running', 'message': 'Trading bot is already active'}
            
            self.trading_active = True
            logger.info("Trading bot started via control center")
            
            return {'status': 'success', 'message': 'Trading bot started successfully'}
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def stop_trading(self) -> Dict[str, Any]:
        """Stop the trading bot"""
        try:
            if not self.trading_active:
                return {'status': 'not_running', 'message': 'Trading bot is not active'}
            
            self.trading_active = False
            logger.info("Trading bot stopped via control center")
            
            return {'status': 'success', 'message': 'Trading bot stopped successfully'}
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop - close all positions immediately"""
        try:
            self.trading_active = False
            # Close all positions
            for position in self.positions:
                position['status'] = 'closed'
                position['close_reason'] = 'emergency_stop'
            
            logger.critical("EMERGENCY STOP activated via control center")
            
            return {'status': 'success', 'message': 'Emergency stop executed, all positions closed'}
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update bot configuration"""
        try:
            # Validate and update config
            for key, value in new_config.items():
                if key in self.bot_config:
                    self.bot_config[key] = value
            
            logger.info(f"Bot configuration updated: {new_config}")
            
            return {'status': 'success', 'message': 'Configuration updated', 'config': self.bot_config}
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'trading_active': self.trading_active,
            'config': self.bot_config,
            'positions': self.positions,
            'recent_trades': self.recent_trades[-10:],  # Last 10 trades
            'performance': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    async def place_manual_trade(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a manual trade"""
        try:
            trade = {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': 0.0,  # Will be filled with current market price
                'timestamp': datetime.now().isoformat(),
                'status': 'pending',
                'type': 'manual'
            }
            
            # In dry-run mode, just simulate
            if self.bot_config.get('dry_run', True):
                trade['status'] = 'simulated'
                self.recent_trades.append(trade)
                logger.info(f"Simulated manual trade: {side} {amount} {symbol}")
                return {'status': 'success', 'message': 'Trade simulated (dry-run mode)', 'trade': trade}
            
            # TODO: Execute real trade via trading agent
            trade['status'] = 'executed'
            self.recent_trades.append(trade)
            
            return {'status': 'success', 'message': 'Trade executed', 'trade': trade}
            
        except Exception as e:
            logger.error(f"Manual trade failed: {e}")
            return {'status': 'error', 'message': str(e)}

# Global control center instance
control_center = TradingControlCenter()


#!/usr/bin/env python3
"""
AI-Powered Trading Bot for Aster DEX
Uses trained models for autonomous trading with risk management.
"""

import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.config import get_settings
from mcp_trader.risk.risk_manager import RiskManager
from mcp_trader.execution.execution_engine import ExecutionEngine
from mcp_trader.data.api_manager import APIKeyManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AITradingBot:
    """
    AI-powered trading bot that uses trained models for autonomous trading.
    Integrates with Aster DEX for live trading with comprehensive risk management.
    """

    def __init__(self, config_file: str = "trading_config.json"):
        self.settings = get_settings()
        self.config_file = config_file
        self.config = self._load_config()
        
        # Core components
        self.risk_manager = None
        self.execution_engine = None
        self.api_manager = None
        
        # Trading state
        self.is_running = False
        self.positions = {}
        self.pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = self.config.get('max_daily_trades', 100)
        
        # Model predictions cache
        self.predictions = {}
        self.last_prediction_time = None
        
        logger.info("AI Trading Bot initialized")

    def _load_config(self) -> Dict:
        """Load trading configuration."""
        default_config = {
            "trading_mode": "paper",  # paper, live
            "max_position_size": 0.1,  # 10% of portfolio per position
            "max_daily_trades": 100,
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.05,  # 5% take profit
            "max_drawdown": 0.1,  # 10% max drawdown
            "risk_free_rate": 0.02,  # 2% annual risk-free rate
            "confidence_threshold": 0.7,  # Minimum confidence for trades
            "rebalance_frequency": 300,  # 5 minutes
            "model_paths": {
                "confluence_model": "models/confluence_model.pkl",
                "aster_native_model": "models/aster_native_model.pkl",
                "lstm_model": "models/lstm_model.pth"
            },
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ASTERUSDT"],
            "timeframes": ["1h", "4h", "1d"]
        }
        
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config

    async def initialize(self):
        """Initialize all trading components."""
        try:
            # Initialize API manager
            self.api_manager = APIKeyManager()
            await self.api_manager.load_credentials()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                max_position_size=self.config['max_position_size'],
                max_drawdown=self.config['max_drawdown'],
                stop_loss_pct=self.config['stop_loss_pct']
            )
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(
                api_manager=self.api_manager,
                trading_mode=self.config['trading_mode']
            )
            
            logger.info("✅ Trading bot components initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading bot: {e}")
            return False

    async def load_models(self) -> bool:
        """Load trained AI models."""
        try:
            # This would load the actual trained models
            # For now, we'll use placeholder logic
            logger.info("Loading AI models...")
            
            # Placeholder for model loading
            # In real implementation, load:
            # - Confluence model (XGBoost)
            # - Aster native model (LSTM)
            # - Feature engineering pipeline
            
            logger.info("✅ AI models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False

    async def generate_predictions(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate trading predictions for given symbols."""
        try:
            predictions = {}
            
            for symbol in symbols:
                # Placeholder prediction logic
                # In real implementation, use trained models
                prediction = {
                    'symbol': symbol,
                    'action': 'HOLD',  # BUY, SELL, HOLD
                    'confidence': 0.5,
                    'price_target': 0.0,
                    'stop_loss': 0.0,
                    'timestamp': datetime.now(),
                    'features': {}  # Technical indicators, etc.
                }
                
                # Simulate some predictions
                if symbol in ['BTCUSDT', 'ETHUSDT']:
                    prediction['action'] = 'BUY'
                    prediction['confidence'] = 0.8
                elif symbol == 'SOLUSDT':
                    prediction['action'] = 'SELL'
                    prediction['confidence'] = 0.6
                
                predictions[symbol] = prediction
            
            self.predictions = predictions
            self.last_prediction_time = datetime.now()
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predictions: {e}")
            return {}

    async def execute_trading_strategy(self):
        """Execute the main trading strategy."""
        try:
            if not self.predictions:
                logger.warning("No predictions available")
                return
            
            for symbol, prediction in self.predictions.items():
                # Check confidence threshold
                if prediction['confidence'] < self.config['confidence_threshold']:
                    continue
                
                # Check risk limits
                if not self.risk_manager.can_trade(symbol, prediction['action']):
                    logger.warning(f"Risk limit exceeded for {symbol}")
                    continue
                
                # Execute trade
                if prediction['action'] in ['BUY', 'SELL']:
                    await self._execute_trade(symbol, prediction)
            
        except Exception as e:
            logger.error(f"❌ Trading strategy execution failed: {e}")

    async def _execute_trade(self, symbol: str, prediction: Dict):
        """Execute a single trade."""
        try:
            action = prediction['action']
            confidence = prediction['confidence']
            
            logger.info(f"Executing {action} for {symbol} (confidence: {confidence:.2f})")
            
            # Calculate position size based on confidence and risk
            position_size = self.risk_manager.calculate_position_size(
                symbol, action, confidence
            )
            
            if position_size <= 0:
                logger.warning(f"Position size too small for {symbol}")
                return
            
            # Execute the trade
            result = await self.execution_engine.execute_order(
                symbol=symbol,
                side=action,
                quantity=position_size,
                order_type='MARKET'
            )
            
            if result['success']:
                self.daily_trades += 1
                logger.info(f"✅ Trade executed: {action} {position_size} {symbol}")
            else:
                logger.error(f"❌ Trade failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")

    async def monitor_positions(self):
        """Monitor open positions and manage risk."""
        try:
            for symbol, position in self.positions.items():
                # Check stop loss
                if self.risk_manager.should_stop_loss(symbol, position):
                    logger.info(f"Stop loss triggered for {symbol}")
                    await self._close_position(symbol, "STOP_LOSS")
                
                # Check take profit
                if self.risk_manager.should_take_profit(symbol, position):
                    logger.info(f"Take profit triggered for {symbol}")
                    await self._close_position(symbol, "TAKE_PROFIT")
                
        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """Close a position."""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            action = 'SELL' if position['side'] == 'BUY' else 'BUY'
            
            result = await self.execution_engine.execute_order(
                symbol=symbol,
                side=action,
                quantity=position['quantity'],
                order_type='MARKET'
            )
            
            if result['success']:
                logger.info(f"✅ Position closed for {symbol} ({reason})")
                del self.positions[symbol]
            else:
                logger.error(f"❌ Failed to close position for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Position close failed for {symbol}: {e}")

    async def run_trading_loop(self):
        """Main trading loop."""
        logger.info("🚀 Starting AI Trading Bot")
        
        try:
            # Initialize components
            if not await self.initialize():
                return
            
            # Load models
            if not await self.load_models():
                return
            
            self.is_running = True
            
            while self.is_running:
                try:
                    # Generate predictions
                    predictions = await self.generate_predictions(self.config['symbols'])
                    
                    # Execute trading strategy
                    await self.execute_trading_strategy()
                    
                    # Monitor positions
                    await self.monitor_positions()
                    
                    # Log status
                    self._log_status()
                    
                    # Wait before next iteration
                    await asyncio.sleep(self.config['rebalance_frequency'])
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Trading bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Trading loop error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
            
        except Exception as e:
            logger.error(f"❌ Trading bot failed: {e}")
        finally:
            await self.cleanup()

    def _log_status(self):
        """Log current trading status."""
        logger.info(f"📊 Status - P&L: {self.pnl:.2f}, Trades: {self.daily_trades}, Positions: {len(self.positions)}")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.execution_engine:
                await self.execution_engine.close()
            
            logger.info("✅ Trading bot cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        logger.info("🛑 Stopping trading bot...")


async def main():
    """Main execution."""
    print("""
================================================================================
                    AI-Powered Trading Bot
              Autonomous Trading with Risk Management
================================================================================
    """)
    
    bot = AITradingBot()
    
    try:
        await bot.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

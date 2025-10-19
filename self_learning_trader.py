#!/usr/bin/env python3
"""
Self-Learning Aggressive Perpetual Trading Bot
Advanced AI-powered trading system with continuous learning and adaptation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import aiohttp
import websockets
import hashlib
import hmac
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuration for self-learning trading bot"""
    # Capital and Risk
    initial_capital: float = 100.0
    max_position_size: float = 0.3  # 30% of capital per trade
    max_leverage: float = 10.0  # Aggressive leverage
    stop_loss_pct: float = 0.015  # 1.5% stop loss
    take_profit_pct: float = 0.03  # 3% take profit
    
    # Learning Parameters
    learning_rate: float = 0.001
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100  # Update model every 100 trades
    
    # Trading Parameters
    trading_pairs: List[str] = None
    min_confidence: float = 0.7  # Minimum confidence for trades
    max_trades_per_hour: int = 20
    cooldown_seconds: int = 30
    
    # Market Regime Detection
    volatility_threshold: float = 0.02
    trend_strength_threshold: float = 0.6
    
    def __post_init__(self):
        if self.trading_pairs is None:
            self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT']

@dataclass
class Trade:
    """Represents a trading position"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    timestamp: datetime
    stop_loss: float
    take_profit: float
    leverage: float
    confidence: float
    strategy: str

class SelfLearningTrader:
    """Advanced self-learning trading bot with aggressive perpetual strategies"""
    
    def __init__(self, config: TradingConfig, api_key: str, secret_key: str):
        self.config = config
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Trading state
        self.balance = config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        # Learning components
        self.models = {
            'price_prediction': MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000),
            'volatility_prediction': RandomForestRegressor(n_estimators=100),
            'trend_prediction': GradientBoostingRegressor(n_estimators=100),
            'risk_assessment': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500)
        }
        
        self.scalers = {
            'features': StandardScaler(),
            'targets': StandardScaler()
        }
        
        self.memory_buffer = []
        self.last_model_update = 0
        
        # Market data
        self.market_data = {}
        self.technical_indicators = {}
        
        # Strategy weights (learned over time)
        self.strategy_weights = {
            'momentum': 0.3,
            'mean_reversion': 0.2,
            'breakout': 0.25,
            'arbitrage': 0.15,
            'liquidation_hunt': 0.1
        }
        
        # Session management
        self.session = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize the trading bot"""
        self.session = aiohttp.ClientSession()
        await self.load_models()
        await self.update_market_data()
        logger.info("Self-learning trader initialized successfully")
    
    async def load_models(self):
        """Load pre-trained models or initialize new ones"""
        try:
            # Try to load existing models
            for model_name, model in self.models.items():
                model_path = f"models/{model_name}_model.pkl"
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
            
            # Load scalers
            scaler_path = "models/feature_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scalers['features'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
                
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Starting with fresh models.")
    
    async def save_models(self):
        """Save trained models"""
        os.makedirs("models", exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f"models/{model_name}_model.pkl"
            joblib.dump(model, model_path)
        
        joblib.dump(self.scalers['features'], "models/feature_scaler.pkl")
        logger.info("Models saved successfully")
    
    async def update_market_data(self):
        """Fetch and process market data"""
        try:
            for symbol in self.config.trading_pairs:
                # Fetch price data
                price_data = await self.fetch_price_data(symbol)
                if price_data:
                    self.market_data[symbol] = price_data
                    
                    # Calculate technical indicators
                    self.technical_indicators[symbol] = self.calculate_technical_indicators(price_data)
                    
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def fetch_price_data(self, symbol: str) -> Optional[Dict]:
        """Fetch price data for a symbol"""
        try:
            # Mock data for now - in production, connect to real exchange
            base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
            volatility = random.uniform(0.01, 0.05)
            
            return {
                'symbol': symbol,
                'price': base_price * (1 + random.uniform(-volatility, volatility)),
                'volume': random.uniform(1000000, 10000000),
                'timestamp': datetime.now(),
                'high_24h': base_price * 1.02,
                'low_24h': base_price * 0.98,
                'change_24h': random.uniform(-0.05, 0.05)
            }
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, price_data: Dict) -> Dict:
        """Calculate technical indicators"""
        price = price_data['price']
        volume = price_data['volume']
        
        # Mock indicators - in production, use real calculations
        return {
            'rsi': random.uniform(30, 70),
            'macd': random.uniform(-100, 100),
            'bollinger_upper': price * 1.02,
            'bollinger_lower': price * 0.98,
            'volume_sma': volume * 0.8,
            'price_sma_20': price * 0.99,
            'price_sma_50': price * 1.01,
            'volatility': random.uniform(0.01, 0.05),
            'momentum': random.uniform(-0.1, 0.1)
        }
    
    def extract_features(self, symbol: str) -> np.ndarray:
        """Extract features for ML models"""
        if symbol not in self.market_data or symbol not in self.technical_indicators:
            return np.zeros(20)  # Default feature vector
        
        price_data = self.market_data[symbol]
        indicators = self.technical_indicators[symbol]
        
        features = [
            price_data['price'],
            price_data['volume'],
            price_data['change_24h'],
            indicators['rsi'],
            indicators['macd'],
            indicators['volatility'],
            indicators['momentum'],
            (price_data['price'] - indicators['bollinger_lower']) / (indicators['bollinger_upper'] - indicators['bollinger_lower']),
            indicators['price_sma_20'] / price_data['price'] - 1,
            indicators['price_sma_50'] / price_data['price'] - 1,
            # Add more features as needed
        ]
        
        # Pad or truncate to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])
    
    def predict_market_direction(self, symbol: str) -> Tuple[float, float]:
        """Predict market direction and confidence"""
        features = self.extract_features(symbol)
        features_scaled = self.scalers['features'].transform([features])
        
        # Get predictions from all models
        price_pred = self.models['price_prediction'].predict(features_scaled)[0]
        volatility_pred = self.models['volatility_prediction'].predict(features_scaled)[0]
        trend_pred = self.models['trend_prediction'].predict(features_scaled)[0]
        risk_pred = self.models['risk_assessment'].predict(features_scaled)[0]
        
        # Combine predictions
        current_price = self.market_data[symbol]['price']
        price_change = (price_pred - current_price) / current_price
        
        # Calculate confidence based on model agreement
        confidence = min(0.95, max(0.1, abs(price_change) * 10 + 0.5))
        
        return price_change, confidence
    
    def select_strategy(self, symbol: str, price_change: float, confidence: float) -> str:
        """Select the best strategy based on market conditions"""
        indicators = self.technical_indicators[symbol]
        
        # Momentum strategy
        if abs(price_change) > 0.02 and confidence > 0.7:
            return 'momentum'
        
        # Mean reversion strategy
        if indicators['rsi'] < 30 or indicators['rsi'] > 70:
            return 'mean_reversion'
        
        # Breakout strategy
        if abs(indicators['macd']) > 50:
            return 'breakout'
        
        # Liquidation hunt strategy (aggressive)
        if indicators['volatility'] > 0.03:
            return 'liquidation_hunt'
        
        # Default to arbitrage
        return 'arbitrage'
    
    def calculate_position_size(self, symbol: str, confidence: float, strategy: str) -> float:
        """Calculate position size based on confidence and strategy"""
        base_size = self.balance * self.config.max_position_size
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on strategy
        strategy_multiplier = self.strategy_weights.get(strategy, 0.2)
        
        # Adjust based on volatility
        volatility = self.technical_indicators[symbol]['volatility']
        volatility_multiplier = min(2.0, max(0.5, 1.0 / volatility))
        
        position_size = base_size * confidence_multiplier * strategy_multiplier * volatility_multiplier
        
        return min(position_size, self.balance * 0.5)  # Cap at 50% of balance
    
    def calculate_leverage(self, symbol: str, strategy: str) -> float:
        """Calculate leverage based on strategy and market conditions"""
        base_leverage = self.config.max_leverage
        
        # Adjust based on strategy
        if strategy == 'liquidation_hunt':
            return min(15.0, base_leverage * 1.5)  # Higher leverage for liquidation hunting
        elif strategy == 'momentum':
            return min(12.0, base_leverage * 1.2)
        elif strategy == 'arbitrage':
            return min(8.0, base_leverage * 0.8)  # Lower leverage for arbitrage
        
        return base_leverage
    
    async def execute_trade(self, symbol: str, side: str, quantity: float, 
                          price: float, strategy: str, confidence: float) -> Optional[Trade]:
        """Execute a trade"""
        try:
            trade_id = f"{symbol}_{side}_{int(time.time())}_{random.randint(1000, 9999)}"
            leverage = self.calculate_leverage(symbol, strategy)
            
            # Calculate stop loss and take profit
            if side == 'BUY':
                stop_loss = price * (1 - self.config.stop_loss_pct)
                take_profit = price * (1 + self.config.take_profit_pct)
            else:
                stop_loss = price * (1 + self.config.stop_loss_pct)
                take_profit = price * (1 - self.config.take_profit_pct)
            
            trade = Trade(
                id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                timestamp=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                confidence=confidence,
                strategy=strategy
            )
            
            # In production, execute real trade here
            logger.info(f"Executing {side} trade: {quantity} {symbol} at {price} (Leverage: {leverage}x, Strategy: {strategy})")
            
            # Update balance (mock)
            self.balance -= quantity * price / leverage  # Margin requirement
            
            # Store trade
            self.positions[trade_id] = trade
            self.trade_history.append(trade)
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    async def update_positions(self):
        """Update existing positions and check for exits"""
        positions_to_close = []
        
        for trade_id, trade in self.positions.items():
            current_price = self.market_data[trade.symbol]['price']
            
            # Check stop loss
            if trade.side == 'BUY' and current_price <= trade.stop_loss:
                positions_to_close.append((trade_id, 'STOP_LOSS'))
            elif trade.side == 'SELL' and current_price >= trade.stop_loss:
                positions_to_close.append((trade_id, 'STOP_LOSS'))
            
            # Check take profit
            elif trade.side == 'BUY' and current_price >= trade.take_profit:
                positions_to_close.append((trade_id, 'TAKE_PROFIT'))
            elif trade.side == 'SELL' and current_price <= trade.take_profit:
                positions_to_close.append((trade_id, 'TAKE_PROFIT'))
        
        # Close positions
        for trade_id, reason in positions_to_close:
            await self.close_position(trade_id, reason)
    
    async def close_position(self, trade_id: str, reason: str):
        """Close a position"""
        if trade_id not in self.positions:
            return
        
        trade = self.positions[trade_id]
        current_price = self.market_data[trade.symbol]['price']
        
        # Calculate PnL
        if trade.side == 'BUY':
            pnl = (current_price - trade.entry_price) * trade.quantity * trade.leverage
        else:
            pnl = (trade.entry_price - current_price) * trade.quantity * trade.leverage
        
        # Update balance
        self.balance += pnl
        
        # Update metrics
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        # Calculate win rate
        total_trades = self.performance_metrics['total_trades']
        winning_trades = self.performance_metrics['winning_trades']
        self.performance_metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        logger.info(f"Closed {trade.symbol} position: {reason}, PnL: ${pnl:.2f}")
        
        # Remove from active positions
        del self.positions[trade_id]
        
        # Add to learning buffer
        self.memory_buffer.append({
            'features': self.extract_features(trade.symbol),
            'action': 1 if trade.side == 'BUY' else 0,
            'reward': pnl / trade.quantity / trade.entry_price,  # Normalized reward
            'timestamp': datetime.now()
        })
    
    async def learn_from_experience(self):
        """Update models based on recent experience"""
        if len(self.memory_buffer) < self.config.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory_buffer, min(self.config.batch_size, len(self.memory_buffer)))
        
        features = np.array([item['features'] for item in batch])
        rewards = np.array([item['reward'] for item in batch])
        
        # Scale features
        features_scaled = self.scalers['features'].fit_transform(features)
        
        # Update models (simplified - in production, use more sophisticated RL)
        try:
            # Update price prediction model
            if len(features_scaled) > 10:
                self.models['price_prediction'].partial_fit(features_scaled, rewards)
                
            # Update strategy weights based on performance
            strategy_performance = {}
            for item in batch:
                strategy = item.get('strategy', 'unknown')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(item['reward'])
            
            # Update strategy weights
            for strategy, performance in strategy_performance.items():
                avg_performance = np.mean(performance)
                if avg_performance > 0:
                    self.strategy_weights[strategy] = min(0.5, self.strategy_weights.get(strategy, 0.2) + 0.01)
                else:
                    self.strategy_weights[strategy] = max(0.05, self.strategy_weights.get(strategy, 0.2) - 0.01)
            
            # Normalize strategy weights
            total_weight = sum(self.strategy_weights.values())
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
            
            logger.info(f"Updated strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Error in learning: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting self-learning trading loop")
        
        while self.is_running:
            try:
                # Update market data
                await self.update_market_data()
                
                # Update existing positions
                await self.update_positions()
                
                # Check if we can make new trades
                if len(self.positions) < 5:  # Max 5 concurrent positions
                    for symbol in self.config.trading_pairs:
                        if symbol in self.market_data:
                            # Predict market direction
                            price_change, confidence = self.predict_market_direction(symbol)
                            
                            # Check if we should trade
                            if confidence > self.config.min_confidence:
                                # Select strategy
                                strategy = self.select_strategy(symbol, price_change, confidence)
                                
                                # Determine trade direction
                                side = 'BUY' if price_change > 0 else 'SELL'
                                
                                # Calculate position size
                                position_size = self.calculate_position_size(symbol, confidence, strategy)
                                
                                # Execute trade
                                if position_size > 0:
                                    await self.execute_trade(
                                        symbol, side, position_size, 
                                        self.market_data[symbol]['price'], 
                                        strategy, confidence
                                    )
                                    
                                    # Cooldown between trades
                                    await asyncio.sleep(self.config.cooldown_seconds)
                
                # Learn from experience
                if len(self.memory_buffer) >= self.config.batch_size:
                    await self.learn_from_experience()
                
                # Save models periodically
                if len(self.trade_history) - self.last_model_update > self.config.update_frequency:
                    await self.save_models()
                    self.last_model_update = len(self.trade_history)
                
                # Log performance
                self.log_performance()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5-second intervals
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
    
    def log_performance(self):
        """Log current performance metrics"""
        metrics = self.performance_metrics
        logger.info(f"Performance - Trades: {metrics['total_trades']}, "
                   f"Win Rate: {metrics['win_rate']:.2%}, "
                   f"PnL: ${metrics['total_pnl']:.2f}, "
                   f"Balance: ${self.balance:.2f}")
    
    async def start_trading(self):
        """Start the trading bot"""
        self.is_running = True
        await self.initialize()
        await self.trading_loop()
    
    async def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        await self.save_models()
        if self.session:
            await self.session.close()
        logger.info("Trading bot stopped")

# Example usage
async def main():
    config = TradingConfig(
        initial_capital=100.0,
        max_position_size=0.3,
        max_leverage=10.0,
        trading_pairs=['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    )
    
    # In production, use real API keys
    api_key = os.getenv("ASTER_API_KEY", "demo_key")
    secret_key = os.getenv("ASTER_API_SECRET", "demo_secret")
    
    trader = SelfLearningTrader(config, api_key, secret_key)
    
    try:
        await trader.start_trading()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await trader.stop_trading()

if __name__ == "__main__":
    asyncio.run(main())

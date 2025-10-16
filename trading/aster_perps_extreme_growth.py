#!/usr/bin/env python3
"""
ASTER PERPETUALS EXTREME GROWTH STRATEGY
Target: $150 → $1,000,000 (6,667x) in Bull Market Downturn

Strategy: Asymmetric volatility plays with intelligent leverage
- $50 → Ultra-high leverage scalping (10-50x)
- $100 → Momentum breakout trading (3-10x leverage)
"""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsterPerpsExtremeStrategy:
    """
    Extreme growth strategy for Aster Perpetuals during volatile markets.
    
    RISK WARNING: This is an ULTRA-AGGRESSIVE strategy designed for:
    - Small capital ($150)
    - High risk tolerance
    - Crypto bull market downturn (high volatility)
    - Goal: 1000x+ returns
    
    This strategy accepts high risk of total loss for asymmetric upside potential.
    """
    
    def __init__(self, total_capital: float = 150):
        self.total_capital = total_capital
        
        # Portfolio allocation
        self.scalping_capital = 50   # Ultra-aggressive scalping (10-50x leverage)
        self.momentum_capital = 100  # Breakout momentum (3-10x leverage)
        
        # Risk parameters (AGGRESSIVE)
        self.max_leverage_scalping = 50  # Max 50x for scalping
        self.max_leverage_momentum = 20  # Max 20x for momentum
        self.max_loss_per_trade_pct = 10  # 10% of allocated capital
        self.daily_loss_limit_pct = 30   # 30% daily loss limit
        
        # Strategy parameters
        self.scalping_target_profit = 0.02  # 2% per scalp
        self.momentum_target_profit = 0.10  # 10% per momentum trade
        self.stop_loss_tight = 0.005  # 0.5% tight stop for scalping
        self.stop_loss_wide = 0.03    # 3% wider stop for momentum
        
        # Current positions
        self.positions = {}
        self.daily_pnl = 0
        self.total_pnl = 0
        
        # Performance tracking
        self.trades = []
        self.equity_curve = [total_capital]
        
        # Aster Perps specific
        self.aster_perps_assets = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'SUIUSDT',
            'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT',
            'LINKUSDT', 'UNIUSDT', 'AAVEUSDT', 'ATOMUSDT'
        ]
        
        # AI model (from training)
        self.model = None
        self.load_ai_model()
        
        logger.info(f"Extreme Growth Strategy initialized: ${total_capital}")
        logger.info(f"  Scalping Pool: ${self.scalping_capital} (max {self.max_leverage_scalping}x)")
        logger.info(f"  Momentum Pool: ${self.momentum_capital} (max {self.max_leverage_momentum}x)")
        logger.info(f"  Target: $1,000,000 (6,667x return)")
    
    def load_ai_model(self):
        """Load the trained ensemble model."""
        try:
            import joblib
            model_dir = Path("training_results/20251015_184036")
            
            # Load all 3 models
            self.rf_model = joblib.load(model_dir / "random_forest_model.pkl")
            self.xgb_model = joblib.load(model_dir / "xgboost_model.pkl")
            self.gb_model = joblib.load(model_dir / "gradient_boosting_model.pkl")
            
            logger.info("✅ AI models loaded (82.44% accuracy ensemble)")
            
        except Exception as e:
            logger.warning(f"Could not load AI models: {e}")
            logger.info("Continuing with pure technical analysis")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 41 features for AI prediction."""
        try:
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_20'] = df['close'].pct_change(20)
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_price_ratio'] = df['volume'] / df['close']
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Additional volume features (to match training data)
            df['quote_volume'] = df['volume'] * df['close']  # Volume in quote currency
            df['taker_buy_volume'] = df['volume'] * 0.6  # Approximate taker buy volume
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['close'].rolling(window).mean()
                df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
            # EMA
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility
            df['volatility_20'] = df['price_change'].rolling(20).std()
            df['volatility_50'] = df['price_change'].rolling(50).std()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            delta_30 = df['close'].diff()
            gain_30 = (delta_30.where(delta_30 > 0, 0)).rolling(30).mean()
            loss_30 = (-delta_30.where(delta_30 < 0, 0)).rolling(30).mean()
            rs_30 = gain_30 / loss_30
            df['rsi_30'] = 100 - (100 / (1 + rs_30))
            
            # Stochastic
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            
            # OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # MFI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
            df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
            
            # Placeholder cross-asset features (will be calculated from multiple assets)
            df['market_momentum'] = df['price_change']  # Simplified
            df['relative_strength'] = 0  # Will be calculated properly
            df['volume_rank'] = 0.5  # Will be calculated properly

            # Return only the calculated features, not the original OHLCV data
            feature_columns = [
                'quote_volume', 'taker_buy_volume', 'price_change', 'price_change_5', 'price_change_20',
                'high_low_ratio', 'close_open_ratio', 'volume_change', 'volume_price_ratio', 'volume_ma_ratio',
                'sma_5', 'price_sma_5_ratio', 'sma_10', 'price_sma_10_ratio', 'sma_20', 'price_sma_20_ratio',
                'sma_50', 'price_sma_50_ratio', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'volatility_20', 'volatility_50', 'rsi', 'rsi_30', 'stoch_k', 'stoch_d',
                'atr', 'obv', 'mfi', 'market_momentum', 'relative_strength', 'volume_rank'
            ]

            return df[feature_columns]
            
        except Exception as e:
            logger.error(f"Feature calculation error: {e}")
            return df
    
    def get_ai_prediction(self, features: pd.DataFrame) -> Tuple[float, float]:
        """Get ensemble AI prediction and confidence."""
        try:
            if not all([self.rf_model, self.xgb_model, self.gb_model]):
                return 0.5, 0.5  # Neutral if models not loaded
            
            # Get last row features
            feature_cols = [
                'price_change', 'price_change_5', 'price_change_20',
                'high_low_ratio', 'close_open_ratio', 'volume_change',
                'volume_price_ratio', 'volume_ma_ratio',
                'sma_5', 'price_sma_5_ratio', 'sma_10', 'price_sma_10_ratio',
                'sma_20', 'price_sma_20_ratio', 'sma_50', 'price_sma_50_ratio',
                'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'volatility_20', 'volatility_50', 'rsi', 'rsi_30',
                'stoch_k', 'stoch_d', 'atr', 'obv', 'mfi',
                'market_momentum', 'relative_strength', 'volume_rank'
            ]
            
            # Handle missing columns
            for col in feature_cols:
                if col not in features.columns:
                    features[col] = 0
            
            X = features[feature_cols].iloc[-1:].values
            
            # Replace inf/nan
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            # Ensemble prediction
            rf_prob = self.rf_model.predict_proba(X)[0][1]
            xgb_prob = self.xgb_model.predict_proba(X)[0][1]
            gb_prob = self.gb_model.predict_proba(X)[0][1]
            
            ensemble_prob = (rf_prob + xgb_prob + gb_prob) / 3
            confidence = max(abs(ensemble_prob - 0.5) * 2, 0.5)  # Scale confidence
            
            return ensemble_prob, confidence
            
        except Exception as e:
            logger.warning(f"AI prediction error: {e}")
            return 0.5, 0.5
    
    def identify_volatility_opportunity(self, df: pd.DataFrame) -> Dict:
        """
        Identify HIGH VOLATILITY opportunities perfect for leverage trading.
        
        Returns signal dict with:
        - type: 'scalping' or 'momentum' or None
        - direction: 'long' or 'short'
        - leverage: suggested leverage
        - confidence: 0-1
        - entry_price: suggested entry
        - stop_loss: suggested stop
        - take_profit: suggested target
        """
        
        if len(df) < 50:
            return {'type': None}
        
        # Calculate features
        df = self.calculate_features(df)
        
        # Get AI prediction
        ai_prob, ai_confidence = self.get_ai_prediction(df)
        
        # Current market state
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = current['close']
        rsi = current['rsi']
        bb_position = current['bb_position']
        volatility = current['volatility_20']
        volume_ratio = current['volume_ma_ratio']
        macd_hist = current['macd_histogram']
        
        # SCALPING SIGNALS (Ultra-tight, high-frequency)
        scalping_signal = None
        
        # 1. RSI Extreme Reversal (HIGH LEVERAGE 30-50x)
        if rsi < 25 and volume_ratio > 1.5:  # Oversold with volume
            scalping_signal = {
                'type': 'scalping',
                'direction': 'long',
                'leverage': 40,
                'confidence': min(ai_confidence * 1.2, 1.0),
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss_tight),
                'take_profit': price * (1 + self.scalping_target_profit),
                'reason': 'RSI oversold bounce'
            }
        elif rsi > 75 and volume_ratio > 1.5:  # Overbought with volume
            scalping_signal = {
                'type': 'scalping',
                'direction': 'short',
                'leverage': 40,
                'confidence': min(ai_confidence * 1.2, 1.0),
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss_tight),
                'take_profit': price * (1 - self.scalping_target_profit),
                'reason': 'RSI overbought rejection'
            }
        
        # 2. Bollinger Band Squeeze Breakout (MEDIUM LEVERAGE 20-30x)
        elif bb_position > 0.95 and volatility > 0.02 and macd_hist > 0:
            scalping_signal = {
                'type': 'scalping',
                'direction': 'long',
                'leverage': 25,
                'confidence': ai_confidence,
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss_tight * 1.5),
                'take_profit': price * (1 + self.scalping_target_profit * 2),
                'reason': 'BB breakout long'
            }
        elif bb_position < 0.05 and volatility > 0.02 and macd_hist < 0:
            scalping_signal = {
                'type': 'scalping',
                'direction': 'short',
                'leverage': 25,
                'confidence': ai_confidence,
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss_tight * 1.5),
                'take_profit': price * (1 - self.scalping_target_profit * 2),
                'reason': 'BB breakout short'
            }
        
        # MOMENTUM SIGNALS (Trend-following, wider stops)
        momentum_signal = None
        
        # 3. Strong Trend Continuation (MODERATE LEVERAGE 10-20x)
        if ai_prob > 0.7 and macd_hist > prev['macd_histogram'] and rsi > 50 and rsi < 70:
            momentum_signal = {
                'type': 'momentum',
                'direction': 'long',
                'leverage': 15,
                'confidence': ai_confidence,
                'entry_price': price,
                'stop_loss': price * (1 - self.stop_loss_wide),
                'take_profit': price * (1 + self.momentum_target_profit),
                'reason': 'AI bullish + MACD momentum'
            }
        elif ai_prob < 0.3 and macd_hist < prev['macd_histogram'] and rsi < 50 and rsi > 30:
            momentum_signal = {
                'type': 'momentum',
                'direction': 'short',
                'leverage': 15,
                'confidence': ai_confidence,
                'entry_price': price,
                'stop_loss': price * (1 + self.stop_loss_wide),
                'take_profit': price * (1 - self.momentum_target_profit),
                'reason': 'AI bearish + MACD weakness'
            }
        
        # 4. Volatility Breakout (HIGH LEVERAGE 15-25x)
        if volatility > df['volatility_20'].quantile(0.8) and volume_ratio > 2.0:
            if current['close'] > current['open'] and macd_hist > 0:
                momentum_signal = {
                    'type': 'momentum',
                    'direction': 'long',
                    'leverage': 20,
                    'confidence': min(ai_confidence * 1.5, 1.0),
                    'entry_price': price,
                    'stop_loss': price * (1 - self.stop_loss_wide * 0.8),
                    'take_profit': price * (1 + self.momentum_target_profit * 1.5),
                    'reason': 'High volatility breakout long'
                }
            elif current['close'] < current['open'] and macd_hist < 0:
                momentum_signal = {
                    'type': 'momentum',
                    'direction': 'short',
                    'leverage': 20,
                    'confidence': min(ai_confidence * 1.5, 1.0),
                    'entry_price': price,
                    'stop_loss': price * (1 + self.stop_loss_wide * 0.8),
                    'take_profit': price * (1 - self.momentum_target_profit * 1.5),
                    'reason': 'High volatility breakout short'
                }
        
        # Return best signal based on confidence
        if scalping_signal and momentum_signal:
            if scalping_signal['confidence'] > momentum_signal['confidence']:
                return scalping_signal
            else:
                return momentum_signal
        elif scalping_signal:
            return scalping_signal
        elif momentum_signal:
            return momentum_signal
        else:
            return {'type': None}
    
    def calculate_position_size(self, signal: Dict, capital_pool: float) -> Dict:
        """Calculate position size with leverage."""
        
        if signal['type'] is None:
            return None
        
        leverage = signal['leverage']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        # Max loss per trade
        max_loss = capital_pool * (self.max_loss_per_trade_pct / 100)
        
        # Calculate position size
        risk_per_unit = abs(entry_price - stop_loss)
        position_size = (max_loss / risk_per_unit) if risk_per_unit > 0 else 0
        
        # Apply leverage
        notional_value = position_size * entry_price
        margin_required = notional_value / leverage
        
        # Don't exceed capital pool
        if margin_required > capital_pool:
            margin_required = capital_pool
            notional_value = margin_required * leverage
            position_size = notional_value / entry_price
        
        return {
            'symbol': signal.get('symbol', 'UNKNOWN'),
            'type': signal['type'],
            'direction': signal['direction'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': signal['take_profit'],
            'position_size': position_size,
            'notional_value': notional_value,
            'margin_required': margin_required,
            'leverage': leverage,
            'confidence': signal['confidence'],
            'reason': signal['reason']
        }
    
    def generate_trading_plan(self) -> Dict:
        """Generate complete trading plan for current market conditions."""
        
        print("\n" + "="*80)
        print("🎯 ASTER PERPS EXTREME GROWTH STRATEGY")
        print("="*80)
        print(f"\n💰 Portfolio: ${self.total_capital:.2f}")
        print(f"   Scalping Pool: ${self.scalping_capital:.2f} (10-50x leverage)")
        print(f"   Momentum Pool: ${self.momentum_capital:.2f} (3-20x leverage)")
        print(f"\n🎯 Target: $1,000,000 (6,667x return)")
        print(f"\n📊 Current Market: Bull Market Downturn (HIGH VOLATILITY)")
        print("\n" + "="*80)
        
        # Strategy explanation
        print("\n📋 STRATEGY BREAKDOWN:\n")
        print("1️⃣ SCALPING POOL ($50 → $50K):")
        print("   • Ultra-tight stops (0.5%)")
        print("   • High leverage (30-50x)")
        print("   • Quick 2% profits")
        print("   • 10-20 trades per day")
        print("   • Focus: RSI extremes, BB squeezes")
        print()
        print("2️⃣ MOMENTUM POOL ($100 → $950K):")
        print("   • Wider stops (3%)")
        print("   • Moderate leverage (10-20x)")
        print("   • Larger 10% profits")
        print("   • 3-5 trades per day")
        print("   • Focus: Trend breakouts, AI signals")
        print()
        
        # Risk warnings
        print("⚠️  RISK WARNINGS:")
        print("   • This is ULTRA-HIGH RISK strategy")
        print("   • High probability of total loss")
        print("   • Requires constant monitoring")
        print("   • Liquidation risk with high leverage")
        print("   • Only use money you can afford to lose")
        print()
        
        # Sample signals (would be live in production)
        print("="*80)
        print("📡 SAMPLE OPPORTUNITY SCAN")
        print("="*80)
        
        # This would scan actual Aster perps in production
        print("\nScanning Aster Perpetuals for opportunities...")
        print("(In production, this would connect to live Aster DEX data)")
        print()
        
        # Milestones to $1M
        print("="*80)
        print("🚀 GROWTH MILESTONES TO $1M")
        print("="*80)
        
        milestones = [
            (150, "Starting Capital"),
            (500, "3.3x - First milestone"),
            (1500, "10x - Prove strategy"),
            (5000, "33x - Serious gains"),
            (15000, "100x - Life-changing"),
            (50000, "333x - Quit your job"),
            (150000, "1000x - Generational"),
            (500000, "3333x - Almost there"),
            (1000000, "6667x - GOAL ACHIEVED! 🎉")
        ]
        
        print()
        for value, description in milestones:
            status = "✅" if self.total_capital >= value else "⏳"
            print(f"{status} ${value:,} - {description}")
        
        print("\n" + "="*80)
        print("💡 NEXT STEPS:")
        print("="*80)
        print("\n1. Paper trade for 24-48 hours to validate")
        print("2. Start with MINIMUM position sizes")
        print("3. Scale up ONLY after proven wins")
        print("4. Set strict daily loss limits (30%)")
        print("5. Take profits regularly (don't get greedy)")
        print()
        
        return {
            'status': 'ready',
            'capital': self.total_capital,
            'target': 1000000,
            'multiplier_needed': 6667,
            'strategy': 'asymmetric_volatility',
            'risk_level': 'ULTRA_HIGH'
        }


def main():
    """Main execution."""
    print("""
================================================================================
                   ASTER PERPS EXTREME GROWTH STRATEGY
                        $150 → $1,000,000 (6,667x)
================================================================================
    """)
    
    strategy = AsterPerpsExtremeStrategy(total_capital=150)
    plan = strategy.generate_trading_plan()
    
    print("\n" + "="*80)
    print("📖 DOCUMENTATION:")
    print("="*80)
    print("\nFor detailed strategy explanation, see:")
    print("  • ASTER_PERPS_EXTREME_GUIDE.md")
    print("  • training_results/20251015_184036/ (AI models)")
    print()
    print("To deploy:")
    print("  python trading/aster_perps_extreme_growth.py --mode paper")
    print()
    
    return plan


if __name__ == "__main__":
    plan = main()


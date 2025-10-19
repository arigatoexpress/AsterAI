"""
Market Regime Detection and Adaptive Risk Management

This module detects market regimes (bull, bear, sideways, recovery) and adjusts
trading parameters accordingly, especially during market downturns and recovery phases.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import ta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime: str  # 'bull', 'bear', 'sideways', 'recovery', 'crash', 'bounce'
    confidence: float  # 0-1 confidence score
    duration_days: int  # How long this regime has lasted
    volatility: float  # Current volatility level
    trend_strength: float  # Trend strength (-1 to 1)
    oversold_level: float  # How oversold (0-1, higher = more oversold)
    recovery_potential: float  # Recovery potential (0-1)
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    recommended_action: str  # 'conservative', 'moderate', 'aggressive', 'maximum'

@dataclass
class RegimeConfig:
    """Configuration for regime detection"""
    lookback_days: int = 30
    volatility_threshold: float = 0.02  # 2% daily volatility
    trend_threshold: float = 0.1  # 10% trend strength
    oversold_threshold: float = 0.7  # RSI < 30
    recovery_threshold: float = 0.6  # Recovery potential > 60%
    regime_change_threshold: float = 0.3  # 30% confidence change needed

class MarketRegimeDetector:
    """Advanced market regime detection using multiple indicators"""
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.regime_history = []
        self.current_regime = None
        self.regime_transitions = []
        
    async def detect_current_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from historical data"""
        
        try:
            # Calculate technical indicators
            indicators = self._calculate_indicators(market_data)
            
            # Detect regime based on multiple factors
            regime = await self._classify_regime(indicators, market_data)
            
            # Update regime history
            self.regime_history.append(regime)
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            # Check for regime transitions
            if self.current_regime and self.current_regime.regime != regime.regime:
                self.regime_transitions.append({
                    'from': self.current_regime.regime,
                    'to': regime.regime,
                    'timestamp': datetime.now(),
                    'confidence': regime.confidence
                })
            
            self.current_regime = regime
            
            logger.info(f"Market regime detected: {regime.regime} (confidence: {regime.confidence:.2f})")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return self._get_default_regime()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for regime detection"""
        
        indicators = {}
        
        try:
            # Price-based indicators
            indicators['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi().iloc[-1]
            indicators['macd'] = ta.trend.MACD(data['close']).macd().iloc[-1]
            indicators['macd_signal'] = ta.trend.MACD(data['close']).macd_signal().iloc[-1]
            indicators['bb_position'] = self._calculate_bb_position(data['close'])
            indicators['stoch'] = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close']).stoch().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma_ratio'] = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
            indicators['obv_trend'] = self._calculate_obv_trend(data)
            
            # Volatility indicators
            indicators['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range().iloc[-1]
            indicators['volatility'] = data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Trend indicators
            indicators['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['close'].rolling(50).mean().iloc[-1]
            indicators['ema_12'] = data['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = data['close'].ewm(span=26).mean().iloc[-1]
            
            # Price action indicators
            indicators['price_vs_sma20'] = (data['close'].iloc[-1] - indicators['sma_20']) / indicators['sma_20']
            indicators['price_vs_sma50'] = (data['close'].iloc[-1] - indicators['sma_50']) / indicators['sma_50']
            indicators['sma_slope'] = self._calculate_sma_slope(data['close'], 20)
            
            # Market structure indicators
            indicators['higher_highs'] = self._count_higher_highs(data['high'], 10)
            indicators['lower_lows'] = self._count_lower_lows(data['low'], 10)
            indicators['support_resistance'] = self._calculate_support_resistance_strength(data)
            
            # Sentiment indicators (simulated)
            indicators['fear_greed'] = self._calculate_fear_greed_index(indicators)
            indicators['market_cap_rank'] = self._get_market_cap_rank(data)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Return default values
            indicators = {key: 0.0 for key in [
                'rsi', 'macd', 'macd_signal', 'bb_position', 'stoch',
                'volume_sma_ratio', 'obv_trend', 'atr', 'volatility',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'price_vs_sma20', 'price_vs_sma50', 'sma_slope',
                'higher_highs', 'lower_lows', 'support_resistance',
                'fear_greed', 'market_cap_rank'
            ]}
        
        return indicators
    
    def _calculate_bb_position(self, prices: pd.Series) -> float:
        """Calculate Bollinger Bands position (0-1)"""
        try:
            bb = ta.volatility.BollingerBands(prices)
            upper = bb.bollinger_hband().iloc[-1]
            lower = bb.bollinger_lband().iloc[-1]
            current = prices.iloc[-1]
            
            if upper != lower:
                return (current - lower) / (upper - lower)
            return 0.5
        except:
            return 0.5
    
    def _calculate_obv_trend(self, data: pd.DataFrame) -> float:
        """Calculate OBV trend strength"""
        try:
            obv = ta.volume.OnBalanceVolume(data['close'], data['volume']).on_balance_volume()
            return obv.pct_change().rolling(10).mean().iloc[-1]
        except:
            return 0.0
    
    def _calculate_sma_slope(self, prices: pd.Series, period: int) -> float:
        """Calculate SMA slope (trend direction)"""
        try:
            sma = prices.rolling(period).mean()
            slope = sma.diff().rolling(5).mean().iloc[-1]
            return slope / prices.iloc[-1]  # Normalize by current price
        except:
            return 0.0
    
    def _count_higher_highs(self, highs: pd.Series, period: int) -> int:
        """Count higher highs in recent period"""
        try:
            recent_highs = highs.tail(period)
            count = 0
            for i in range(1, len(recent_highs)):
                if recent_highs.iloc[i] > recent_highs.iloc[i-1]:
                    count += 1
            return count
        except:
            return 0
    
    def _count_lower_lows(self, lows: pd.Series, period: int) -> int:
        """Count lower lows in recent period"""
        try:
            recent_lows = lows.tail(period)
            count = 0
            for i in range(1, len(recent_lows)):
                if recent_lows.iloc[i] < recent_lows.iloc[i-1]:
                    count += 1
            return count
        except:
            return 0
    
    def _calculate_support_resistance_strength(self, data: pd.DataFrame) -> float:
        """Calculate support/resistance strength"""
        try:
            # Find recent highs and lows
            highs = data['high'].rolling(20).max()
            lows = data['low'].rolling(20).min()
            
            current_price = data['close'].iloc[-1]
            recent_high = highs.iloc[-1]
            recent_low = lows.iloc[-1]
            
            # Calculate position between support and resistance
            if recent_high != recent_low:
                return (current_price - recent_low) / (recent_high - recent_low)
            return 0.5
        except:
            return 0.5
    
    def _calculate_fear_greed_index(self, indicators: Dict[str, float]) -> float:
        """Calculate fear/greed index (0-1, 0=fear, 1=greed)"""
        try:
            # Combine multiple indicators
            rsi_score = (indicators['rsi'] - 30) / 40  # RSI 30-70 range
            bb_score = indicators['bb_position']
            volume_score = min(indicators['volume_sma_ratio'] / 2, 1.0)
            
            # Weighted average
            fear_greed = (rsi_score * 0.4 + bb_score * 0.3 + volume_score * 0.3)
            return max(0, min(1, fear_greed))
        except:
            return 0.5
    
    def _get_market_cap_rank(self, data: pd.DataFrame) -> float:
        """Get market cap rank (simulated)"""
        # In production, this would fetch from CoinGecko API
        return np.random.uniform(0.1, 0.9)
    
    async def _classify_regime(self, indicators: Dict[str, float], data: pd.DataFrame) -> MarketRegime:
        """Classify market regime based on indicators"""
        
        try:
            # Calculate regime scores
            regime_scores = {
                'bull': 0.0,
                'bear': 0.0,
                'sideways': 0.0,
                'recovery': 0.0,
                'crash': 0.0,
                'bounce': 0.0
            }
            
            # Bull market indicators
            if indicators['rsi'] > 50 and indicators['price_vs_sma20'] > 0:
                regime_scores['bull'] += 0.3
            if indicators['macd'] > indicators['macd_signal']:
                regime_scores['bull'] += 0.2
            if indicators['higher_highs'] > indicators['lower_lows']:
                regime_scores['bull'] += 0.2
            if indicators['sma_slope'] > 0:
                regime_scores['bull'] += 0.3
            
            # Bear market indicators
            if indicators['rsi'] < 50 and indicators['price_vs_sma20'] < 0:
                regime_scores['bear'] += 0.3
            if indicators['macd'] < indicators['macd_signal']:
                regime_scores['bear'] += 0.2
            if indicators['lower_lows'] > indicators['higher_highs']:
                regime_scores['bear'] += 0.2
            if indicators['sma_slope'] < 0:
                regime_scores['bear'] += 0.3
            
            # Recovery indicators (key for current market)
            if indicators['rsi'] < 30 and indicators['bb_position'] < 0.2:
                regime_scores['recovery'] += 0.4  # Oversold conditions
            if indicators['price_vs_sma50'] < -0.2 and indicators['sma_slope'] > -0.01:
                regime_scores['recovery'] += 0.3  # Below long-term average but stabilizing
            if indicators['fear_greed'] < 0.3:
                regime_scores['recovery'] += 0.3  # Fear levels high
            
            # Crash indicators
            if indicators['volatility'] > 0.05 and indicators['rsi'] < 20:
                regime_scores['crash'] += 0.4
            if indicators['price_vs_sma20'] < -0.1 and indicators['volume_sma_ratio'] > 2:
                regime_scores['crash'] += 0.3
            if indicators['fear_greed'] < 0.2:
                regime_scores['crash'] += 0.3
            
            # Bounce indicators (recovery from oversold)
            if indicators['rsi'] > 30 and indicators['rsi'] < 50:
                regime_scores['bounce'] += 0.3
            if indicators['bb_position'] > 0.2 and indicators['bb_position'] < 0.5:
                regime_scores['bounce'] += 0.3
            if indicators['volume_sma_ratio'] > 1.5:
                regime_scores['bounce'] += 0.2
            if indicators['stoch'] > 20 and indicators['stoch'] < 50:
                regime_scores['bounce'] += 0.2
            
            # Sideways indicators
            if abs(indicators['price_vs_sma20']) < 0.02 and indicators['volatility'] < 0.02:
                regime_scores['sideways'] += 0.4
            if indicators['rsi'] > 40 and indicators['rsi'] < 60:
                regime_scores['sideways'] += 0.3
            if abs(indicators['sma_slope']) < 0.005:
                regime_scores['sideways'] += 0.3
            
            # Find best regime
            best_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[best_regime]
            
            # Calculate additional metrics
            volatility = indicators['volatility']
            trend_strength = abs(indicators['sma_slope']) * 100
            oversold_level = max(0, (30 - indicators['rsi']) / 30) if indicators['rsi'] < 30 else 0
            recovery_potential = self._calculate_recovery_potential(indicators)
            
            # Determine risk level and recommended action
            risk_level, recommended_action = self._determine_risk_strategy(
                best_regime, confidence, oversold_level, recovery_potential
            )
            
            return MarketRegime(
                regime=best_regime,
                confidence=confidence,
                duration_days=self._calculate_regime_duration(best_regime),
                volatility=volatility,
                trend_strength=trend_strength,
                oversold_level=oversold_level,
                recovery_potential=recovery_potential,
                risk_level=risk_level,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return self._get_default_regime()
    
    def _calculate_recovery_potential(self, indicators: Dict[str, float]) -> float:
        """Calculate recovery potential based on oversold conditions"""
        try:
            # RSI oversold
            rsi_recovery = max(0, (30 - indicators['rsi']) / 30) if indicators['rsi'] < 30 else 0
            
            # Bollinger Bands oversold
            bb_recovery = max(0, (0.2 - indicators['bb_position']) / 0.2) if indicators['bb_position'] < 0.2 else 0
            
            # Price vs moving averages
            price_recovery = max(0, abs(indicators['price_vs_sma50']) / 0.3) if indicators['price_vs_sma50'] < -0.1 else 0
            
            # Fear levels
            fear_recovery = max(0, (0.3 - indicators['fear_greed']) / 0.3) if indicators['fear_greed'] < 0.3 else 0
            
            # Combine factors
            recovery_potential = (rsi_recovery * 0.3 + bb_recovery * 0.3 + 
                                price_recovery * 0.2 + fear_recovery * 0.2)
            
            return min(1.0, recovery_potential)
            
        except Exception as e:
            logger.error(f"Error calculating recovery potential: {e}")
            return 0.5
    
    def _determine_risk_strategy(self, regime: str, confidence: float, 
                               oversold_level: float, recovery_potential: float) -> Tuple[str, str]:
        """Determine risk level and recommended action based on regime"""
        
        # During recovery phases, be more aggressive
        if regime in ['recovery', 'bounce'] and oversold_level > 0.5:
            if recovery_potential > 0.7:
                return 'extreme', 'maximum'
            elif recovery_potential > 0.5:
                return 'high', 'aggressive'
            else:
                return 'medium', 'moderate'
        
        # During bear markets with high oversold levels
        elif regime == 'bear' and oversold_level > 0.6:
            if recovery_potential > 0.6:
                return 'high', 'aggressive'
            else:
                return 'medium', 'moderate'
        
        # During bull markets
        elif regime == 'bull':
            if confidence > 0.7:
                return 'high', 'aggressive'
            else:
                return 'medium', 'moderate'
        
        # During sideways markets
        elif regime == 'sideways':
            return 'low', 'conservative'
        
        # During crashes
        elif regime == 'crash':
            if oversold_level > 0.8 and recovery_potential > 0.8:
                return 'extreme', 'maximum'  # Maximum opportunity
            else:
                return 'low', 'conservative'
        
        # Default
        return 'medium', 'moderate'
    
    def _calculate_regime_duration(self, regime: str) -> int:
        """Calculate how long current regime has lasted"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime == regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _get_default_regime(self) -> MarketRegime:
        """Get default regime when detection fails"""
        return MarketRegime(
            regime='sideways',
            confidence=0.5,
            duration_days=1,
            volatility=0.02,
            trend_strength=0.0,
            oversold_level=0.0,
            recovery_potential=0.5,
            risk_level='medium',
            recommended_action='moderate'
        )

class AdaptiveRiskManager:
    """Adaptive risk management based on market regime"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.regime_detector = MarketRegimeDetector(RegimeConfig())
        self.current_regime = None
        
    async def get_adaptive_config(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get adaptive configuration based on current market regime"""
        
        # Detect current regime
        regime = await self.regime_detector.detect_current_regime(market_data)
        self.current_regime = regime
        
        # Create adaptive config
        adaptive_config = self.base_config.copy()
        
        # Adjust position sizing based on regime
        if regime.recommended_action == 'maximum':
            adaptive_config['position_size_pct'] = min(0.05, self.base_config['position_size_pct'] * 2.5)  # 5% max
            adaptive_config['max_positions'] = min(5, self.base_config['max_positions'] + 2)
            adaptive_config['daily_loss_limit_pct'] = min(0.15, self.base_config['daily_loss_limit_pct'] * 1.5)
        elif regime.recommended_action == 'aggressive':
            adaptive_config['position_size_pct'] = min(0.04, self.base_config['position_size_pct'] * 2.0)  # 4% max
            adaptive_config['max_positions'] = min(4, self.base_config['max_positions'] + 1)
            adaptive_config['daily_loss_limit_pct'] = min(0.12, self.base_config['daily_loss_limit_pct'] * 1.2)
        elif regime.recommended_action == 'moderate':
            adaptive_config['position_size_pct'] = self.base_config['position_size_pct'] * 1.2  # 20% increase
        else:  # conservative
            adaptive_config['position_size_pct'] = self.base_config['position_size_pct'] * 0.8  # 20% decrease
        
        # Adjust stop loss and take profit based on volatility
        if regime.volatility > 0.03:  # High volatility
            adaptive_config['stop_loss_pct'] = min(0.03, self.base_config['stop_loss_pct'] * 1.5)
            adaptive_config['take_profit_pct'] = min(0.08, self.base_config['take_profit_pct'] * 1.5)
        elif regime.volatility < 0.01:  # Low volatility
            adaptive_config['stop_loss_pct'] = max(0.01, self.base_config['stop_loss_pct'] * 0.7)
            adaptive_config['take_profit_pct'] = max(0.02, self.base_config['take_profit_pct'] * 0.7)
        
        # Adjust strategy weights based on regime
        if regime.regime in ['recovery', 'bounce']:
            # Favor momentum and trend-following strategies
            adaptive_config['strategy_weights'] = {
                'market_making': 0.2,
                'funding_arbitrage': 0.3,
                'dmark': 0.3,
                'degen_trading': 0.2  # Higher weight for momentum
            }
        elif regime.regime == 'bull':
            # Favor all strategies equally
            adaptive_config['strategy_weights'] = {
                'market_making': 0.25,
                'funding_arbitrage': 0.25,
                'dmark': 0.25,
                'degen_trading': 0.25
            }
        elif regime.regime == 'bear':
            # Favor defensive strategies
            adaptive_config['strategy_weights'] = {
                'market_making': 0.4,
                'funding_arbitrage': 0.4,
                'dmark': 0.1,
                'degen_trading': 0.1
            }
        else:  # sideways, crash
            # Conservative approach
            adaptive_config['strategy_weights'] = {
                'market_making': 0.5,
                'funding_arbitrage': 0.5,
                'dmark': 0.0,
                'degen_trading': 0.0
            }
        
        # Add regime-specific parameters
        adaptive_config['market_regime'] = {
            'regime': regime.regime,
            'confidence': regime.confidence,
            'oversold_level': regime.oversold_level,
            'recovery_potential': regime.recovery_potential,
            'volatility': regime.volatility,
            'risk_level': regime.risk_level,
            'recommended_action': regime.recommended_action
        }
        
        logger.info(f"Adaptive config updated for {regime.regime} regime: "
                   f"position_size={adaptive_config['position_size_pct']:.3f}, "
                   f"max_positions={adaptive_config['max_positions']}")
        
        return adaptive_config

# Example usage and testing
async def main():
    """Test the market regime detection system"""
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    market_data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, len(dates)),
        'high': np.random.uniform(40000, 50000, len(dates)),
        'low': np.random.uniform(40000, 50000, len(dates)),
        'close': np.random.uniform(40000, 50000, len(dates)),
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Simulate a market downturn and recovery
    market_data['close'] = market_data['close'] * (1 - np.linspace(0, 0.3, len(dates)))  # 30% decline
    market_data['close'] = market_data['close'] * (1 + np.linspace(0, 0.1, len(dates)))  # 10% recovery
    
    # Create adaptive risk manager
    base_config = {
        'position_size_pct': 0.02,
        'max_positions': 3,
        'daily_loss_limit_pct': 0.10,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }
    
    risk_manager = AdaptiveRiskManager(base_config)
    
    # Get adaptive configuration
    adaptive_config = await risk_manager.get_adaptive_config(market_data)
    
    print("Market Regime Detection Results:")
    print(f"Regime: {adaptive_config['market_regime']['regime']}")
    print(f"Confidence: {adaptive_config['market_regime']['confidence']:.2f}")
    print(f"Oversold Level: {adaptive_config['market_regime']['oversold_level']:.2f}")
    print(f"Recovery Potential: {adaptive_config['market_regime']['recovery_potential']:.2f}")
    print(f"Risk Level: {adaptive_config['market_regime']['risk_level']}")
    print(f"Recommended Action: {adaptive_config['market_regime']['recommended_action']}")
    print(f"\nAdaptive Configuration:")
    print(f"Position Size: {adaptive_config['position_size_pct']:.3f}")
    print(f"Max Positions: {adaptive_config['max_positions']}")
    print(f"Daily Loss Limit: {adaptive_config['daily_loss_limit_pct']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())

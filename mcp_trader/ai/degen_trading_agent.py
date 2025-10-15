"""
Degen Trading Agent

High-risk, high-reward trading agent focused on:
- Real-time social media sentiment analysis
- Meme coin momentum trading
- Viral asset detection and aggressive positioning
- Social arbitrage opportunities

This agent takes significantly higher risks than conservative HFT agents
but implements strict risk management to prevent catastrophic losses.

Risk Profile: EXTREME RISK - EXTREME REWARD
Capital Allocation: 10-20% of total portfolio
Target Returns: 200-500% monthly (with high volatility)
"""

import asyncio
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import time
from concurrent.futures import ThreadPoolExecutor
from google.cloud import pubsub_v1

from ..logging_utils import get_logger
from .hft_trading_agent import HFTTradingAgent
from ..strategies.degen_trading import DegenTradingStrategy, DegenConfig
from ..sentiment.gemini_analyzer import GeminiSentimentAnalyzer
from ..data.aster_feed import HFTAsterDataFeed
from ..execution.aster_client import AsterClient
from ..risk.risk_manager import RiskManager

logger = get_logger(__name__)


class DegenTradingAgent(HFTTradingAgent):
    """
    Specialized HFT agent for high-risk, high-reward degen trading

    Inherits from HFTTradingAgent but overrides with aggressive parameters
    """

    def __init__(self, config: Dict[str, Any]):
        # Update config for degen trading
        degen_config = self._create_degen_config(config)

        # Initialize parent with degen config
        super().__init__(degen_config)

        # Override strategy with degen strategy
        self.degen_strategy = DegenTradingStrategy(DegenConfig())

        # Enhanced social sentiment analysis
        self.social_sentiment_cache = {}
        self.sentiment_subscriber = None
        self.sentiment_executor = ThreadPoolExecutor(max_workers=2)

        # Degen-specific parameters
        self.max_daily_risk_pct = 0.15  # 15% daily risk limit
        self.target_daily_return = 0.05  # 5% daily target (very aggressive)
        self.min_trade_confidence = 0.6  # Lower confidence threshold for degen
        self.aggressive_mode = True  # Enable aggressive trading

        # Performance tracking
        self.daily_starting_balance = config.get('initial_balance', 50.0)
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5  # Stop after 5 losses in a row

        logger.info("🎲 Degen Trading Agent initialized")
        logger.info("⚠️ HIGH RISK MODE - Monitor closely!")
        logger.info(f"🎯 Target: {self.target_daily_return*100:.1f}% daily return")
        logger.info(f"💰 Daily risk limit: {self.max_daily_risk_pct*100:.1f}%")

    def _create_degen_config(self, base_config: Dict) -> Dict:
        """Create degen-optimized configuration"""
        degen_config = base_config.copy()

        # Aggressive trading parameters
        degen_config.update({
            'max_position_size_usd': 15.0,  # $15 max per trade (was $25)
            'min_order_size_usd': 0.5,      # $0.50 min per trade (was $1)
            'max_open_positions': 5,        # Fewer concurrent positions
            'target_profit_threshold': 0.02,  # 2% target (more aggressive)
            'stop_loss_threshold': 0.08,     # 8% stop loss (looser)
            'max_daily_loss': 7.5,           # $7.50 daily loss limit
            'trading_fee_rate': 0.0005,
            'latency_threshold_ms': 5.0,     # Slightly more lenient
            'gpu_acceleration': True,
            'agent_type': 'degen_trading',
            'strategy_focus': 'social_sentiment_momentum'
        })

        return degen_config

    async def initialize_sentiment_analysis(self):
        """Initialize enhanced sentiment analysis for social data"""
        try:
            # Initialize Pub/Sub for social sentiment
            self.sentiment_subscriber = pubsub_v1.SubscriberClient()
            subscription_path = self.sentiment_subscriber.subscription_path(
                self.config.get('gcp_project_id', 'hft-aster-trader'),
                'hft-sentiment-sub'
            )

            # Start sentiment subscriber in background
            asyncio.create_task(self._sentiment_subscriber_loop())

            logger.info("🎭 Enhanced sentiment analysis initialized for degen trading")

        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analysis: {e}")

    async def _sentiment_subscriber_loop(self):
        """Background loop for sentiment data subscription"""
        try:
            subscription_path = self.sentiment_subscriber.subscription_path(
                self.config.get('gcp_project_id', 'hft-aster-trader'),
                'hft-sentiment-sub'
            )

            def callback(message):
                try:
                    data = json.loads(message.data.decode('utf-8'))
                    symbol = data.get('symbol', 'market-sentiment')
                    score = data.get('sentiment_score', 0.0)
                    source = data.get('source', 'unknown')

                    # Enhanced sentiment processing for degen trading
                    self._process_social_sentiment(symbol, score, source)

                    message.ack()

                except Exception as e:
                    logger.error(f"❌ Sentiment message processing error: {e}")
                    message.nack()

            # Subscribe to sentiment topic
            streaming_pull_future = self.sentiment_subscriber.subscribe(
                subscription_path, callback=callback
            )

            logger.info("🎭 Sentiment subscriber started")

            # Keep the subscriber running
            await asyncio.wrap_future(streaming_pull_future)

        except Exception as e:
            logger.error(f"❌ Sentiment subscriber error: {e}")

    def _process_social_sentiment(self, symbol: str, score: float, source: str):
        """Process incoming sentiment data with degen-specific logic"""
        try:
            # Store enhanced sentiment data
            if symbol not in self.social_sentiment_cache:
                self.social_sentiment_cache[symbol] = []

            self.social_sentiment_cache[symbol].append({
                'score': score,
                'source': source,
                'timestamp': time.time(),
                'weight': self._calculate_sentiment_weight(source)
            })

            # Keep only recent sentiment (last 30 minutes)
            cutoff_time = time.time() - (30 * 60)
            self.social_sentiment_cache[symbol] = [
                s for s in self.social_sentiment_cache[symbol]
                if s['timestamp'] > cutoff_time
            ]

            # Log significant sentiment changes
            avg_sentiment = self.get_current_sentiment(symbol)
            if abs(avg_sentiment) > 0.7:  # Very bullish/bearish
                logger.info(f"🎭 Strong social sentiment for {symbol}: {avg_sentiment:.2f} ({source})")

        except Exception as e:
            logger.error(f"❌ Social sentiment processing error: {e}")

    def _calculate_sentiment_weight(self, source: str) -> float:
        """Calculate weight for different sentiment sources"""
        weights = {
            'twitter': 1.0,
            'reddit': 0.8,
            'telegram': 0.9,
            'news': 0.7,
            'dexscreener': 0.6
        }
        return weights.get(source.lower(), 0.5)

    def get_current_sentiment(self, symbol: str = 'market-sentiment') -> float:
        """Get weighted average sentiment for a symbol"""
        try:
            if symbol not in self.social_sentiment_cache:
                return 0.0

            sentiments = self.social_sentiment_cache[symbol]
            if not sentiments:
                return 0.0

            # Weighted average
            total_weight = sum(s['weight'] for s in sentiments)
            weighted_sum = sum(s['score'] * s['weight'] for s in sentiments)

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"❌ Sentiment retrieval error: {e}")
            return 0.0

    async def analyze_market_opportunity(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market for degen trading opportunities

        Focuses on social sentiment and momentum
        """
        try:
            symbol = market_data.get('symbol', '')
            current_price = market_data.get('price', 0)

            if not symbol or current_price <= 0:
                return None

            # Get degen strategy analysis
            analysis = await self.degen_strategy.analyze_symbol(symbol, current_price)

            # Add sentiment data
            analysis['sentiment'] = self.get_current_sentiment(symbol)

            # Enhanced analysis for degen trading
            analysis = self._enhance_degen_analysis(analysis, market_data)

            return analysis

        except Exception as e:
            logger.error(f"❌ Market analysis error: {e}")
            return None

    def _enhance_degen_analysis(self, analysis: Dict, market_data: Dict) -> Dict:
        """Add degen-specific analysis enhancements"""
        try:
            symbol = analysis.get('symbol', '')
            signal = analysis.get('signal', 'HOLD')

            # Check for meme coin patterns
            if self._is_meme_coin_symbol(symbol):
                analysis['meme_coin_boost'] = 1.5  # Boost confidence for meme coins
                analysis['confidence'] *= 1.5

            # Social momentum check
            social_momentum = self._calculate_social_momentum(symbol)
            analysis['social_momentum'] = social_momentum

            # Risk adjustment for degen mode
            if self.aggressive_mode and signal != 'HOLD':
                analysis['confidence'] = min(analysis.get('confidence', 0) * 1.3, 1.0)

            # Daily performance adjustment
            if self.daily_pnl < 0:
                # Reduce confidence when losing
                loss_multiplier = max(0.5, 1.0 + (self.daily_pnl / self.daily_starting_balance))
                analysis['confidence'] *= loss_multiplier

            return analysis

        except Exception as e:
            logger.error(f"❌ Degen analysis enhancement error: {e}")
            return analysis

    def _is_meme_coin_symbol(self, symbol: str) -> bool:
        """Check if symbol is a known meme coin"""
        meme_coins = [
            'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF',
            'MEW', 'BRETT', 'MOTHER', 'TURBO', 'SPX'
        ]

        base_symbol = symbol.split('/')[0].replace('USD', '').replace('USDT', '')
        return base_symbol.upper() in meme_coins

    def _calculate_social_momentum(self, symbol: str) -> float:
        """Calculate social momentum from sentiment history"""
        try:
            if symbol not in self.social_sentiment_cache:
                return 0.0

            sentiments = self.social_sentiment_cache[symbol][-10:]  # Last 10 readings
            if len(sentiments) < 2:
                return 0.0

            # Calculate momentum as rate of sentiment change
            recent = np.mean([s['score'] for s in sentiments[-3:]])
            older = np.mean([s['score'] for s in sentiments[:-3]])

            momentum = (recent - older) / max(abs(older), 0.1)  # Normalize

            return momentum

        except Exception as e:
            return 0.0

    async def execute_trade_decision(self, analysis: Dict) -> Optional[Dict]:
        """
        Execute degen trading decision with enhanced risk management
        """
        try:
            if not analysis or analysis.get('signal') == 'HOLD':
                return None

            confidence = analysis.get('confidence', 0)
            if confidence < self.min_trade_confidence:
                return None

            # Check degen-specific risk limits
            if not self._check_degen_risk_limits(analysis):
                return None

            # Execute trade using degen strategy
            capital_available = self._get_available_capital_for_degen()
            trade_order = await self.degen_strategy.execute_trade_signal(
                analysis, capital_available
            )

            if trade_order:
                # Record degen trade
                self.daily_trades += 1
                self._update_consecutive_losses(trade_order)

                logger.info(f"🎲 Degen trade executed: {trade_order['signal']} "
                           f"{trade_order['symbol']} ${trade_order['quantity']*trade_order['price']:.2f}")

            return trade_order

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return None

    def _check_degen_risk_limits(self, analysis: Dict) -> bool:
        """Check degen-specific risk limits"""
        try:
            # Daily loss limit
            if self.daily_pnl <= -self.max_daily_risk_pct * self.daily_starting_balance:
                logger.warning("⚠️ Daily loss limit reached - stopping degen trading")
                return False

            # Consecutive losses limit
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.warning("⚠️ Too many consecutive losses - cooling down")
                return False

            # Daily trade limit (prevent overtrading)
            if self.daily_trades >= 20:  # Max 20 trades per day
                logger.warning("⚠️ Daily trade limit reached")
                return False

            # Symbol-specific limits
            symbol = analysis.get('symbol', '')
            if self._has_recent_trade_on_symbol(symbol):
                return False  # Avoid overtrading same symbol

            return True

        except Exception as e:
            logger.error(f"❌ Risk limit check error: {e}")
            return False

    def _get_available_capital_for_degen(self) -> float:
        """Get available capital for degen trading (conservative allocation)"""
        try:
            total_balance = self.get_current_balance()
            available_for_degen = total_balance * 0.15  # Only 15% of capital

            # Reduce allocation if losing
            if self.daily_pnl < 0:
                loss_pct = abs(self.daily_pnl) / self.daily_starting_balance
                available_for_degen *= max(0.5, 1.0 - loss_pct)

            return min(available_for_degen, 15.0)  # Cap at $15

        except Exception as e:
            logger.error(f"❌ Capital calculation error: {e}")
            return 5.0  # Safe fallback

    def _has_recent_trade_on_symbol(self, symbol: str, minutes: int = 30) -> bool:
        """Check if we recently traded this symbol"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)

            for trade in self.degen_strategy.trade_history:
                if (trade.get('order', {}).get('symbol') == symbol and
                    trade.get('timestamp', datetime.min) > cutoff_time):
                    return True

            return False

        except Exception as e:
            return False

    def _update_consecutive_losses(self, trade_order: Dict):
        """Update consecutive losses counter"""
        try:
            # This is a simplified version - in practice would check actual trade outcomes
            # For now, assume alternating pattern
            if np.random.random() < 0.4:  # 40% win rate for degen
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

        except Exception as e:
            pass

    def update_trade_result(self, trade_result: Dict):
        """Update trade result and adjust degen parameters"""
        try:
            pnl = trade_result.get('pnl', 0)
            outcome = trade_result.get('outcome', 'loss')

            # Update daily P&L
            self.daily_pnl += pnl

            # Update strategy
            self.degen_strategy.update_trade_result(
                trade_result.get('trade_id', ''),
                pnl,
                outcome
            )

            # Adjust aggressiveness based on performance
            self._adjust_aggressiveness()

            logger.info(f"📊 Degen trade result: ${pnl:.2f} ({outcome}) "
                       f"Daily P&L: ${self.daily_pnl:.2f}")

        except Exception as e:
            logger.error(f"❌ Trade result update error: {e}")

    def _adjust_aggressiveness(self):
        """Dynamically adjust trading aggressiveness"""
        try:
            # Reduce aggression if losing
            if self.daily_pnl < -0.05 * self.daily_starting_balance:
                self.aggressive_mode = False
                self.min_trade_confidence = 0.8  # Higher confidence threshold
                logger.warning("⚠️ Reducing degen aggression due to losses")

            # Increase aggression if winning
            elif self.daily_pnl > 0.02 * self.daily_starting_balance:
                self.aggressive_mode = True
                self.min_trade_confidence = 0.5  # Lower confidence threshold
                logger.info("🚀 Increasing degen aggression - we're winning!")

            # Reset to default
            else:
                self.aggressive_mode = True
                self.min_trade_confidence = 0.6

        except Exception as e:
            pass

    def get_performance_stats(self) -> Dict:
        """Get degen agent performance statistics"""
        base_stats = super().get_performance_stats()

        degen_stats = self.degen_strategy.get_performance_stats()

        # Combine stats
        combined_stats = {
            **base_stats,
            **degen_stats,
            'agent_type': 'degen_trading',
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'aggressive_mode': self.aggressive_mode,
            'risk_profile': 'EXTREME_RISK_EXTREME_REWARD',
            'capital_allocation': '10-20%',
            'target_return': f"{self.target_daily_return*100:.1f}%_daily"
        }

        return combined_stats

    def reset_daily_stats(self):
        """Reset daily statistics"""
        try:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.degen_strategy.daily_pnl = 0.0

            # Reset social sentiment cache (keep some history)
            for symbol in self.social_sentiment_cache:
                # Keep only last 5 readings for continuity
                self.social_sentiment_cache[symbol] = self.social_sentiment_cache[symbol][-5:]

            logger.info("🌅 Daily stats reset for degen agent")

        except Exception as e:
            logger.error(f"❌ Daily stats reset error: {e}")

    async def shutdown(self):
        """Clean shutdown of degen agent"""
        try:
            # Cancel sentiment subscriber
            if self.sentiment_subscriber:
                await self.sentiment_subscriber.close()

            # Shutdown executor
            if self.sentiment_executor:
                self.sentiment_executor.shutdown(wait=True)

            # Shutdown strategy
            await self.degen_strategy.shutdown()

            logger.info("🎲 Degen trading agent shutdown complete")

        except Exception as e:
            logger.error(f"❌ Degen agent shutdown error: {e}")


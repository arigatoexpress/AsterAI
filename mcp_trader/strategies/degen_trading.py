"""
Degen Trading Strategy for HFT

High-risk, high-reward trading focused on:
- Real-time social media sentiment (Twitter, Reddit, Telegram)
- Meme coin momentum trading
- Viral asset detection and early entry
- Aggressive position sizing with strict risk controls
- Social arbitrage opportunities

Risk Profile: HIGH RISK - HIGH REWARD
Capital Allocation: 10-20% of total capital
Target: 200-500% monthly returns (with high volatility)
Stop Loss: 5% per trade (aggressive but controlled)
"""

import asyncio
import logging
import numpy as np
import torch
import aiohttp
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import re
import json

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DegenConfig:
    """Configuration for degen trading strategy"""
    social_data_sources: List[str] = None  # ['twitter', 'reddit', 'telegram']
    min_viral_score: float = 0.8  # Minimum virality score to trade
    max_position_size_pct: float = 0.15  # Max 15% of capital per trade
    momentum_threshold: float = 0.05  # 5% price change for momentum
    sentiment_threshold: float = 0.7  # 0.7+ sentiment for bullish trades
    holding_period_minutes: int = 30  # Hold for 30 minutes max
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    cooldown_minutes: int = 60  # 1 hour cooldown between trades
    max_trades_per_hour: int = 3  # Limit trading frequency
    min_volume_threshold: float = 10000  # Minimum $10k volume

    def __post_init__(self):
        if self.social_data_sources is None:
            self.social_data_sources = ['twitter', 'reddit', 'telegram']


class SocialDataCollector:
    """
    Collects real-time social media data for degen trading
    """

    def __init__(self, config: DegenConfig):
        self.config = config
        self.social_cache = {}  # source -> data
        self.sentiment_cache = {}  # symbol -> sentiment_score
        self.viral_scores = {}  # symbol -> virality_score

        # API endpoints (placeholder - would use real APIs)
        self.api_endpoints = {
            'twitter': 'https://api.twitter.com/2/tweets/search/recent',
            'reddit': 'https://www.reddit.com/r/cryptocurrency/hot.json',
            'telegram': 'https://api.telegram.org/bot{token}/getUpdates',
            'dexscreener': 'https://api.dexscreener.com/latest/dex/tokens'
        }

    async def collect_social_data(self, symbol: str) -> Dict:
        """
        Collect social media data for a symbol

        Args:
            symbol: Trading symbol (e.g., 'DOGE/USD')

        Returns:
            Social data dictionary
        """
        try:
            # This is a placeholder implementation
            # In production, would integrate with real social APIs

            # Simulate collecting data from multiple sources
            social_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'twitter_mentions': await self._get_twitter_mentions(symbol),
                'reddit_posts': await self._get_reddit_posts(symbol),
                'telegram_messages': await self._get_telegram_messages(symbol),
                'dex_pairs': await self._get_dex_pairs(symbol),
                'virality_score': 0.0,
                'sentiment_score': 0.0,
                'momentum_indicators': {}
            }

            # Calculate virality score
            social_data['virality_score'] = self._calculate_virality_score(social_data)

            # Calculate sentiment score (would integrate with Gemini)
            social_data['sentiment_score'] = self._calculate_social_sentiment(social_data)

            # Store in cache
            self.social_cache[symbol] = social_data

            return social_data

        except Exception as e:
            logger.error(f"❌ Social data collection failed for {symbol}: {e}")
            return {}

    async def _get_twitter_mentions(self, symbol: str) -> List[Dict]:
        """Get Twitter mentions (placeholder)"""
        # In production: Use Twitter API v2
        # For now, return simulated data
        base_symbol = symbol.split('/')[0].replace('USD', '').replace('USDT', '')
        return [
            {
                'text': f"${base_symbol} to the moon! 🚀",
                'likes': np.random.randint(10, 1000),
                'retweets': np.random.randint(5, 500),
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 60))
            } for _ in range(np.random.randint(1, 20))
        ]

    async def _get_reddit_posts(self, symbol: str) -> List[Dict]:
        """Get Reddit posts (placeholder)"""
        # In production: Use Reddit API
        base_symbol = symbol.split('/')[0].replace('USD', '').replace('USDT', '')
        return [
            {
                'title': f"{base_symbol} Discussion Thread",
                'score': np.random.randint(10, 1000),
                'comments': np.random.randint(5, 200),
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 120))
            } for _ in range(np.random.randint(0, 10))
        ]

    async def _get_telegram_messages(self, symbol: str) -> List[Dict]:
        """Get Telegram messages (placeholder)"""
        # In production: Use Telegram Bot API
        return [
            {
                'text': f"Buying ${symbol} now!",
                'timestamp': datetime.now() - timedelta(minutes=np.random.randint(0, 30))
            } for _ in range(np.random.randint(0, 50))
        ]

    async def _get_dex_pairs(self, symbol: str) -> List[Dict]:
        """Get DEX pairs data (placeholder)"""
        # In production: Use DexScreener API
        base_symbol = symbol.split('/')[0]
        return [{
            'pair': f"{base_symbol}/USDT",
            'price': np.random.uniform(0.000001, 1.0),
            'volume_24h': np.random.uniform(1000, 1000000),
            'price_change_24h': np.random.uniform(-50, 500),
            'liquidity': np.random.uniform(1000, 100000)
        }]

    def _calculate_virality_score(self, social_data: Dict) -> float:
        """
        Calculate virality score from social metrics

        Score components:
        - Twitter: mentions * (likes + retweets)
        - Reddit: posts * (score + comments)
        - Telegram: message frequency
        - DEX: volume and price change
        """
        try:
            score = 0.0

            # Twitter score
            twitter_mentions = social_data.get('twitter_mentions', [])
            if twitter_mentions:
                twitter_score = sum(
                    (mention.get('likes', 0) + mention.get('retweets', 0))
                    for mention in twitter_mentions
                ) / len(twitter_mentions)
                score += min(twitter_score / 1000, 1.0) * 0.3

            # Reddit score
            reddit_posts = social_data.get('reddit_posts', [])
            if reddit_posts:
                reddit_score = sum(
                    (post.get('score', 0) + post.get('comments', 0))
                    for post in reddit_posts
                ) / len(reddit_posts)
                score += min(reddit_score / 500, 1.0) * 0.3

            # Telegram score (message frequency)
            telegram_messages = social_data.get('telegram_messages', [])
            telegram_score = len(telegram_messages) / 10.0  # Normalize
            score += min(telegram_score, 1.0) * 0.2

            # DEX score (volume and momentum)
            dex_pairs = social_data.get('dex_pairs', [])
            if dex_pairs:
                pair = dex_pairs[0]
                volume_score = min(pair.get('volume_24h', 0) / 100000, 1.0)
                momentum_score = min(abs(pair.get('price_change_24h', 0)) / 50, 1.0)
                dex_score = (volume_score + momentum_score) / 2
                score += dex_score * 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"❌ Virality score calculation error: {e}")
            return 0.0

    def _calculate_social_sentiment(self, social_data: Dict) -> float:
        """
        Calculate sentiment score from social data

        Uses simple keyword analysis + emoji sentiment
        In production: Would integrate with Gemini for advanced analysis
        """
        try:
            all_text = []

            # Collect all text
            for mentions in social_data.get('twitter_mentions', []):
                all_text.append(mentions.get('text', ''))

            for posts in social_data.get('reddit_posts', []):
                all_text.append(posts.get('title', ''))

            for messages in social_data.get('telegram_messages', []):
                all_text.append(messages.get('text', ''))

            if not all_text:
                return 0.5  # Neutral

            # Simple sentiment analysis
            bullish_keywords = ['moon', 'pump', 'bull', 'buy', 'long', 'up', '🚀', '📈', '💎']
            bearish_keywords = ['dump', 'bear', 'sell', 'short', 'down', 'crash', '📉', '💩']

            bullish_count = 0
            bearish_count = 0
            total_words = 0

            for text in all_text:
                text_lower = text.lower()
                total_words += len(text.split())

                for keyword in bullish_keywords:
                    bullish_count += text_lower.count(keyword.lower())

                for keyword in bearish_keywords:
                    bearish_count += text_lower.count(keyword.lower())

            # Calculate sentiment ratio
            if bullish_count + bearish_count == 0:
                return 0.5  # Neutral

            sentiment_ratio = bullish_count / (bullish_count + bearish_count)

            return sentiment_ratio

        except Exception as e:
            logger.error(f"❌ Sentiment calculation error: {e}")
            return 0.5


class DegenTradingStrategy:
    """
    High-risk, high-reward degen trading strategy

    Focuses on:
    - Social media momentum
    - Viral coin detection
    - Aggressive position sizing
    - Quick profit taking
    """

    def __init__(self, config: DegenConfig):
        self.config = config
        self.social_collector = SocialDataCollector(config)

        # Trading state
        self.active_positions = {}
        self.last_trade_time = datetime.min
        self.hourly_trade_count = 0
        self.last_hour_reset = datetime.now()

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.trade_history = deque(maxlen=100)

        # Risk management
        self.daily_loss_limit = 0.0  # Set by agent
        self.daily_pnl = 0.0

        logger.info("🎲 Degen Trading Strategy initialized")
        logger.info(f"🎯 Target: High-risk, high-reward (200-500% monthly)")
        logger.info(f"⚠️  Risk: {self.config.stop_loss_pct*100:.1f}% stop loss per trade")

    async def analyze_symbol(self, symbol: str, current_price: float) -> Dict:
        """
        Analyze a symbol for degen trading opportunities

        Args:
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            Analysis results
        """
        try:
            # Collect social data
            social_data = await self.social_collector.collect_social_data(symbol)

            # Calculate momentum
            momentum = self._calculate_momentum(symbol, current_price)

            # Get sentiment from Gemini (integrated at agent level)
            sentiment_score = social_data.get('sentiment_score', 0.5)

            # Determine trade signal
            signal = self._generate_trade_signal(
                social_data, momentum, sentiment_score, current_price
            )

            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'social_data': social_data,
                'momentum': momentum,
                'sentiment': sentiment_score,
                'signal': signal,
                'confidence': self._calculate_signal_confidence(signal, social_data),
                'timestamp': datetime.now()
            }

            return analysis

        except Exception as e:
            logger.error(f"❌ Symbol analysis failed for {symbol}: {e}")
            return {}

    def _calculate_momentum(self, symbol: str, current_price: float) -> Dict:
        """
        Calculate momentum indicators

        Args:
            symbol: Trading symbol
            current_price: Current price

        Returns:
            Momentum metrics
        """
        try:
            # Simple momentum calculation
            # In production: Would use historical data

            # Simulate recent price changes
            recent_changes = np.random.normal(0, 0.02, 10)  # Last 10 price changes
            momentum_5min = np.mean(recent_changes[-1:])  # 5-minute momentum
            momentum_15min = np.mean(recent_changes[-3:])  # 15-minute momentum
            momentum_1hour = np.mean(recent_changes)  # 1-hour momentum

            return {
                '5min': momentum_5min,
                '15min': momentum_15min,
                '1hour': momentum_1hour,
                'strength': abs(momentum_15min),
                'direction': 1 if momentum_15min > 0 else -1
            }

        except Exception as e:
            logger.error(f"❌ Momentum calculation error: {e}")
            return {'5min': 0, '15min': 0, '1hour': 0, 'strength': 0, 'direction': 0}

    def _generate_trade_signal(self, social_data: Dict, momentum: Dict,
                              sentiment: float, current_price: float) -> str:
        """
        Generate trading signal based on social + technical data

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        try:
            virality = social_data.get('virality_score', 0)
            momentum_strength = momentum.get('strength', 0)
            momentum_direction = momentum.get('direction', 0)

            # High virality + positive sentiment + momentum = BUY
            if (virality > self.config.min_viral_score and
                sentiment > self.config.sentiment_threshold and
                momentum_direction > 0 and
                momentum_strength > self.config.momentum_threshold):

                # Check for meme coin patterns
                if self._is_meme_coin_signal(social_data):
                    return 'BUY_AGGRESSIVE'

                return 'BUY'

            # High virality + negative sentiment + negative momentum = SELL
            elif (virality > self.config.min_viral_score and
                  sentiment < (1 - self.config.sentiment_threshold) and
                  momentum_direction < 0 and
                  momentum_strength > self.config.momentum_threshold):

                return 'SELL'

            # Social pump detection
            elif self._detect_social_pump(social_data):
                return 'BUY_PUMP'

            return 'HOLD'

        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            return 'HOLD'

    def _is_meme_coin_signal(self, social_data: Dict) -> bool:
        """
        Detect if this is a meme coin signal

        Looks for:
        - High social mention volume
        - Rocket ship emojis 🚀
        - Community excitement keywords
        """
        try:
            mentions = social_data.get('twitter_mentions', [])
            if not mentions:
                return False

            # Count rocket ship emojis and hype keywords
            rocket_count = 0
            hype_keywords = ['moon', 'diamond', 'gem', 'pump', 'ape', 'diamond hands']

            for mention in mentions:
                text = mention.get('text', '').lower()
                rocket_count += text.count('🚀')
                for keyword in hype_keywords:
                    if keyword in text:
                        rocket_count += 1

            # High rocket/hype score indicates meme coin
            return rocket_count >= 3

        except Exception as e:
            return False

    def _detect_social_pump(self, social_data: Dict) -> bool:
        """
        Detect coordinated social media pump

        Looks for:
        - Sudden spike in mentions
        - Coordinated messaging
        - New account patterns
        """
        try:
            # Simple pump detection (placeholder)
            # In production: Would analyze mention velocity, account age distribution, etc.
            mentions = social_data.get('twitter_mentions', [])
            recent_mentions = [m for m in mentions
                             if (datetime.now() - m.get('timestamp', datetime.min)).seconds < 300]  # Last 5 min

            # Sudden spike in mentions
            if len(recent_mentions) > len(mentions) * 0.7:  # 70% of mentions in last 5 min
                return True

            return False

        except Exception as e:
            return False

    def _calculate_signal_confidence(self, signal: str, social_data: Dict) -> float:
        """
        Calculate confidence score for the trading signal

        Args:
            signal: Trading signal
            social_data: Social data

        Returns:
            Confidence score (0-1)
        """
        try:
            if signal == 'HOLD':
                return 0.0

            virality = social_data.get('virality_score', 0)
            sentiment = social_data.get('sentiment_score', 0.5)

            # Base confidence on virality and sentiment strength
            confidence = (virality + abs(sentiment - 0.5) * 2) / 2

            # Boost for aggressive signals
            if 'AGGRESSIVE' in signal or 'PUMP' in signal:
                confidence *= 1.2

            return min(confidence, 1.0)

        except Exception as e:
            return 0.0

    async def execute_trade_signal(self, analysis: Dict, capital_available: float) -> Optional[Dict]:
        """
        Execute a trade based on analysis

        Args:
            analysis: Symbol analysis results
            capital_available: Available capital

        Returns:
            Trade order or None
        """
        try:
            signal = analysis.get('signal', 'HOLD')
            confidence = analysis.get('confidence', 0)
            symbol = analysis.get('symbol', '')
            current_price = analysis.get('current_price', 0)

            if signal == 'HOLD' or confidence < 0.6:
                return None

            # Check trading limits
            if not self._can_execute_trade():
                return None

            # Calculate position size (aggressive but with limits)
            position_size = self._calculate_position_size(
                capital_available, confidence, signal
            )

            if position_size < 1.0:  # Minimum $1
                return None

            # Determine order details
            if 'BUY' in signal:
                side = 'buy'
                quantity = position_size / current_price
            elif signal == 'SELL':
                side = 'sell'
                quantity = position_size / current_price
            else:
                return None

            # Create order with tight time limits
            order = {
                'symbol': symbol,
                'side': side,
                'type': 'market',  # Execute immediately for degen trading
                'quantity': quantity,
                'price': current_price,
                'timestamp': datetime.now(),
                'strategy': 'degen',
                'signal': signal,
                'confidence': confidence,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
                'max_holding_minutes': self.config.holding_period_minutes
            }

            # Record trade
            self._record_trade(order)

            logger.info(f"🎲 Degen trade executed: {signal} {symbol} "
                       f"${position_size:.2f} (Confidence: {confidence:.1%})")

            return order

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return None

    def _can_execute_trade(self) -> bool:
        """
        Check if we can execute a trade based on limits

        Returns:
            True if can trade
        """
        try:
            now = datetime.now()

            # Reset hourly counter if needed
            if (now - self.last_hour_reset).seconds >= 3600:
                self.hourly_trade_count = 0
                self.last_hour_reset = now

            # Check hourly limit
            if self.hourly_trade_count >= self.config.max_trades_per_hour:
                return False

            # Check cooldown period
            if (now - self.last_trade_time).seconds < self.config.cooldown_minutes * 60:
                return False

            # Check daily loss limit
            if self.daily_pnl < self.daily_loss_limit:
                logger.warning("⚠️ Daily loss limit reached - pausing degen trading")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Trade limit check error: {e}")
            return False

    def _calculate_position_size(self, capital: float, confidence: float, signal: str) -> float:
        """
        Calculate position size for degen trading

        Args:
            capital: Available capital
            confidence: Signal confidence
            signal: Trading signal

        Returns:
            Position size in USD
        """
        try:
            # Base size as percentage of capital
            base_pct = self.config.max_position_size_pct

            # Scale by confidence
            confidence_multiplier = confidence

            # Boost for aggressive signals
            if 'AGGRESSIVE' in signal or 'PUMP' in signal:
                base_pct *= 2.0  # Double the position size

            position_pct = base_pct * confidence_multiplier
            position_size = capital * position_pct

            # Apply limits
            max_size = min(self.config.max_position_size_pct * capital * 2, capital * 0.5)
            position_size = min(position_size, max_size)

            return position_size

        except Exception as e:
            logger.error(f"❌ Position size calculation error: {e}")
            return 0.0

    def _record_trade(self, order: Dict):
        """Record trade execution"""
        try:
            self.total_trades += 1
            self.hourly_trade_count += 1
            self.last_trade_time = datetime.now()

            # Add to history
            self.trade_history.append({
                'order': order,
                'timestamp': datetime.now(),
                'status': 'executed'
            })

        except Exception as e:
            logger.error(f"❌ Trade recording error: {e}")

    def update_trade_result(self, trade_id: str, pnl: float, outcome: str):
        """
        Update trade result

        Args:
            trade_id: Trade identifier
            pnl: Profit/loss
            outcome: 'win' or 'loss'
        """
        try:
            self.total_pnl += pnl
            self.daily_pnl += pnl

            if outcome == 'win':
                self.winning_trades += 1

            # Update trade history
            for trade in self.trade_history:
                if trade.get('order', {}).get('timestamp') == trade_id:
                    trade['pnl'] = pnl
                    trade['outcome'] = outcome
                    break

            logger.info(f"📊 Degen trade result: ${pnl:.2f} ({outcome})")

        except Exception as e:
            logger.error(f"❌ Trade result update error: {e}")

    def get_performance_stats(self) -> Dict:
        """Get strategy performance statistics"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        return {
            'strategy': 'degen_trading',
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.active_positions),
            'hourly_trade_count': self.hourly_trade_count,
            'target_win_rate': 0.40,  # Lower target due to high risk
            'target_daily_return': 0.05,  # 5% daily target (very aggressive)
            'risk_profile': 'HIGH_RISK_HIGH_REWARD'
        }


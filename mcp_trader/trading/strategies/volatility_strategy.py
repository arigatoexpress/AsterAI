"""
Volatility Trading Strategy for Aster DEX
Capitalizes on price swings in volatile assets through momentum and mean reversion.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ...execution.aster_client import AccountPosition, TickerPrice
from ..types import TradingDecision, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for volatility trading."""
    min_volatility_threshold: float = 3.0  # Minimum daily volatility %
    max_volatility_threshold: float = 15.0  # Maximum daily volatility %
    momentum_window: int = 6  # Hours to analyze momentum
    mean_reversion_window: int = 24  # Hours for mean reversion
    position_size_scaler: float = 0.5  # Scale position size with volatility
    profit_taking_threshold: float = 2.0  # Take profit at X% gain
    stop_loss_threshold: float = 3.0  # Stop loss at X% loss
    min_holding_time: int = 30  # Minimum holding time in minutes
    max_holding_time: int = 480  # Maximum holding time in minutes (8 hours)


class VolatilityStrategy:
    """
    Volatility-based trading strategy for high-volatility assets.
    Uses momentum detection and mean reversion signals.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = VolatilityConfig(**config)
        self.position_tracking: Dict[str, Dict] = {}  # Track position entry details

        logger.info("VolatilityStrategy initialized for high-volatility trading")

    async def make_decisions(self, symbol: str, ticker: TickerPrice,
                           existing_position: Optional[AccountPosition],
                           market_regime: MarketRegime) -> List[TradingDecision]:
        """
        Generate trading decisions based on volatility analysis.
        """
        decisions = []

        try:
            # Only trade in high volatility conditions
            if market_regime != MarketRegime.HIGH_VOLATILITY:
                return decisions

            # Calculate current volatility
            volatility = self._calculate_volatility(ticker)
            if volatility < self.config.min_volatility_threshold:
                return decisions

            # Check existing position
            if existing_position and existing_position.position_amt != 0:
                decisions.extend(self._manage_existing_position(symbol, existing_position, ticker))
            else:
                decisions.extend(self._find_entry_opportunities(symbol, ticker, volatility))

        except Exception as e:
            logger.error(f"Error in volatility strategy for {symbol}: {e}")

        return decisions

    def _calculate_volatility(self, ticker: TickerPrice) -> float:
        """Calculate current volatility from ticker data."""
        # Use 24h price change as proxy for volatility
        # In production, this would use more sophisticated volatility calculation
        return abs(ticker.price_change_percent)

    def _manage_existing_position(self, symbol: str, position: AccountPosition,
                                ticker: TickerPrice) -> List[TradingDecision]:
        """Manage existing volatility positions."""
        decisions = []

        entry_details = self.position_tracking.get(symbol, {})
        if not entry_details:
            return decisions

        entry_price = entry_details['entry_price']
        entry_time = entry_details['entry_time']
        holding_time = (datetime.now() - entry_time).total_seconds() / 60  # minutes

        current_price = ticker.last_price
        pnl_percent = (current_price - entry_price) / entry_price * 100

        # Check profit taking
        if pnl_percent >= self.config.profit_taking_threshold:
            decisions.append(TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="SELL",
                quantity=abs(position.position_amt),
                price=None,
                reason=f"Profit taking: {pnl_percent:.1f}% gain after {holding_time:.0f} minutes",
                confidence=0.9,
                risk_score=0.1,
                expected_profit=position.unrealized_profit
            ))
            return decisions

        # Check stop loss
        if pnl_percent <= -self.config.stop_loss_threshold:
            decisions.append(TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="SELL",
                quantity=abs(position.position_amt),
                price=None,
                reason=f"Stop loss: {pnl_percent:.1f}% loss after {holding_time:.0f} minutes",
                confidence=0.95,
                risk_score=0.2,
                expected_profit=position.unrealized_profit
            ))
            return decisions

        # Check holding time limits
        if holding_time > self.config.max_holding_time:
            decisions.append(TradingDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action="SELL",
                quantity=abs(position.position_amt),
                price=None,
                reason=f"Time-based exit: held for {holding_time:.0f} minutes",
                confidence=0.8,
                risk_score=0.3,
                expected_profit=position.unrealized_profit
            ))

        return decisions

    def _find_entry_opportunities(self, symbol: str, ticker: TickerPrice,
                               volatility: float) -> List[TradingDecision]:
        """Find entry opportunities based on volatility signals."""
        decisions = []

        # Analyze recent price action for momentum
        momentum_signal = self._analyze_momentum(ticker)
        mean_reversion_signal = self._analyze_mean_reversion(ticker)

        # Combine signals
        combined_signal = momentum_signal * 0.6 + mean_reversion_signal * 0.4

        if abs(combined_signal) > 0.7:  # Strong signal threshold
            # Determine position size based on volatility
            position_size = self._calculate_position_size(volatility, ticker.last_price)

            if combined_signal > 0:  # Bullish signal
                decisions.append(TradingDecision(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="BUY",
                    quantity=position_size,
                    price=None,
                    reason=f"Volatility long entry: momentum={momentum_signal:.2f}, "
                           f"mean_rev={mean_reversion_signal:.2f}, vol={volatility:.1f}%",
                    confidence=min(abs(combined_signal), 0.9),
                    risk_score=self._calculate_entry_risk(volatility),
                    expected_profit=None
                ))

                # Track position entry
                self.position_tracking[symbol] = {
                    'entry_price': ticker.last_price,
                    'entry_time': datetime.now(),
                    'position_type': 'long',
                    'volatility': volatility
                }

            elif combined_signal < -0.7:  # Bearish signal
                # For now, focus on long positions only to reduce complexity
                # Bearish signals could be used for short positions in future
                pass

        return decisions

    def _analyze_momentum(self, ticker: TickerPrice) -> float:
        """Analyze momentum from ticker data."""
        # Simple momentum calculation using price changes
        # In production, this would use more sophisticated momentum indicators

        # Use 24h change as momentum proxy
        momentum = ticker.price_change_percent / 10.0  # Normalize to -1 to 1 range

        # Add some weighting for recent activity
        if ticker.price_change_percent > 5:  # Strong upward momentum
            momentum += 0.3
        elif ticker.price_change_percent < -5:  # Strong downward momentum
            momentum -= 0.3

        return max(min(momentum, 1.0), -1.0)  # Clamp to [-1, 1]

    def _analyze_mean_reversion(self, ticker: TickerPrice) -> float:
        """Analyze mean reversion signals."""
        # Simple mean reversion based on recent vs longer-term price
        # This is a placeholder - real implementation would use moving averages

        # For now, use a simple heuristic
        price_change_24h = ticker.price_change_percent

        # Mean reversion signal: if price has moved significantly in one direction,
        # expect pullback
        if price_change_24h > 8:  # Overbought
            return -0.8  # Expect downward correction
        elif price_change_24h < -8:  # Oversold
            return 0.8   # Expect upward correction
        else:
            return 0.0   # No strong mean reversion signal

    def _calculate_position_size(self, volatility: float, price: float) -> float:
        """Calculate position size based on volatility."""
        # Base position size
        base_size_usd = 100.0  # $100 base position

        # Scale with volatility (higher volatility = smaller position)
        volatility_scaler = self.config.position_size_scaler
        if volatility > 10:
            volatility_scaler *= 0.5  # Reduce size in extreme volatility

        position_size_usd = base_size_usd * volatility_scaler
        quantity = position_size_usd / price

        return quantity

    def _calculate_entry_risk(self, volatility: float) -> float:
        """Calculate risk score for entry."""
        # Higher volatility = higher risk
        base_risk = 0.5
        volatility_risk = min(volatility / 10.0, 0.5)  # Max 50% additional risk

        return min(base_risk + volatility_risk, 1.0)

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status."""
        return {
            'active_positions': len(self.position_tracking),
            'positions': self.position_tracking,
            'config': {
                'min_volatility_threshold': self.config.min_volatility_threshold,
                'max_volatility_threshold': self.config.max_volatility_threshold,
                'profit_taking_threshold': self.config.profit_taking_threshold,
                'stop_loss_threshold': self.config.stop_loss_threshold
            }
        }


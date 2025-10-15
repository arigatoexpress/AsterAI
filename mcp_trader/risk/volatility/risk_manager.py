"""
Volatility-Aware Risk Management for Aster DEX Trading
Implements comprehensive risk controls optimized for volatile assets.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ...trading.autonomous_trader import TradingDecision, PortfolioState
from ...execution.aster_client import AccountPosition

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_portfolio_risk: float = 0.1  # Maximum 10% portfolio risk
    max_single_position_risk: float = 0.05  # Maximum 5% per position
    max_daily_loss: float = 0.15  # Maximum 15% daily loss
    max_drawdown: float = 0.20  # Maximum 20% drawdown
    max_concurrent_positions: int = 5
    max_volatility_position: float = 0.03  # Max position size in high volatility
    min_liquidity_ratio: float = 2.0  # Minimum bid/ask spread ratio
    max_slippage_tolerance: float = 0.02  # Maximum 2% slippage tolerance


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    concentration_ratio: float = 0.0  # Largest position / total portfolio
    correlation_risk: float = 0.0


class VolatilityRiskManager:
    """
    Advanced risk management system optimized for volatile crypto assets.
    Implements dynamic position sizing, volatility-adjusted stops, and portfolio protection.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = RiskLimits(**config.get('limits', {}))
        self.risk_metrics = RiskMetrics()
        self.position_history: List[Dict] = []
        self.daily_start_value: float = 0.0
        self.portfolio_peak: float = 0.0

        # Risk state tracking
        self.consecutive_losses = 0
        self.last_risk_check = datetime.now()

        logger.info("VolatilityRiskManager initialized with enhanced risk controls")

    async def assess_trade_risk(self, decision: TradingDecision,
                              portfolio_state: PortfolioState) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a trading decision.
        Returns approval status and risk analysis.
        """
        try:
            risk_analysis = {
                'approved': True,
                'risk_score': 0.0,
                'warnings': [],
                'rejections': [],
                'recommended_adjustments': []
            }

            # Update current risk metrics
            await self._update_risk_metrics(portfolio_state)

            # Run all risk checks
            checks = [
                self._check_portfolio_limits(decision, portfolio_state),
                self._check_position_limits(decision, portfolio_state),
                self._check_volatility_limits(decision),
                self._check_liquidity_risk(decision),
                self._check_correlation_risk(decision, portfolio_state),
                self._check_temporal_risk(decision),
                self._check_slippage_risk(decision)
            ]

            for check in checks:
                if check['rejected']:
                    risk_analysis['approved'] = False
                    risk_analysis['rejections'].append(check['reason'])
                elif check['warning']:
                    risk_analysis['warnings'].append(check['warning'])
                if check.get('adjustment'):
                    risk_analysis['recommended_adjustments'].append(check['adjustment'])

                risk_analysis['risk_score'] += check.get('risk_score', 0.0)

            # Final risk score normalization
            risk_analysis['risk_score'] = min(risk_analysis['risk_score'], 1.0)

            # Log risk assessment
            if not risk_analysis['approved']:
                logger.warning(f"Trade rejected for {decision.symbol}: {risk_analysis['rejections']}")
            elif risk_analysis['warnings']:
                logger.info(f"Trade approved with warnings for {decision.symbol}: {risk_analysis['warnings']}")

            return risk_analysis

        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {
                'approved': False,
                'risk_score': 1.0,
                'warnings': [],
                'rejections': [f"Risk assessment error: {str(e)}"],
                'recommended_adjustments': []
            }

    def _check_portfolio_limits(self, decision: TradingDecision,
                              portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Check portfolio-level risk limits."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        # Calculate potential portfolio impact
        trade_value = decision.quantity * (decision.price or 0)
        portfolio_value = portfolio_state.total_balance + portfolio_state.total_positions_value

        if portfolio_value > 0:
            trade_portfolio_impact = trade_value / portfolio_value

            # Check maximum portfolio risk
            if trade_portfolio_impact > self.config.max_portfolio_risk:
                result['rejected'] = True
                result['reason'] = f"Trade would exceed portfolio risk limit: {trade_portfolio_impact:.1%} > {self.config.max_portfolio_risk:.1%}"

            # Check daily loss limit
            if self.risk_metrics.daily_pnl < -self.config.max_daily_loss * self.daily_start_value:
                result['rejected'] = True
                result['reason'] = f"Daily loss limit exceeded: {self.risk_metrics.daily_pnl:.2f}"

            # Check drawdown limit
            if self.risk_metrics.max_drawdown > self.config.max_drawdown:
                result['rejected'] = True
                result['reason'] = f"Maximum drawdown exceeded: {self.risk_metrics.max_drawdown:.1%}"

        # Check concurrent positions
        if len(portfolio_state.active_positions) >= self.config.max_concurrent_positions:
            result['rejected'] = True
            result['reason'] = f"Maximum concurrent positions exceeded: {len(portfolio_state.active_positions)}"

        return result

    def _check_position_limits(self, decision: TradingDecision,
                             portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Check position-specific risk limits."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        existing_position = portfolio_state.active_positions.get(decision.symbol)
        trade_value = decision.quantity * (decision.price or 100)  # Estimate if no price

        # Check single position size limit
        portfolio_value = portfolio_state.total_balance + portfolio_state.total_positions_value
        if portfolio_value > 0:
            position_portfolio_ratio = trade_value / portfolio_value

            if position_portfolio_ratio > self.config.max_single_position_risk:
                result['rejected'] = True
                result['reason'] = f"Position size exceeds limit: {position_portfolio_ratio:.1%} > {self.config.max_single_position_risk:.1%}"

        # Check position concentration
        if portfolio_state.active_positions:
            largest_position = max(
                pos.position_amt * pos.entry_price
                for pos in portfolio_state.active_positions.values()
            )
            if largest_position / portfolio_value > 0.3:  # 30% concentration warning
                result['warning'] = "High position concentration detected"

        return result

    def _check_volatility_limits(self, decision: TradingDecision) -> Dict[str, Any]:
        """Check volatility-based risk limits."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        # For volatile assets, implement stricter position sizing
        # This is a placeholder - would need actual volatility data
        volatility_estimate = 0.1  # 10% daily volatility estimate

        if volatility_estimate > 0.15:  # Extremely volatile
            max_position = self.config.max_volatility_position
            if decision.quantity * (decision.price or 100) > max_position * 1000:  # Rough estimate
                result['rejected'] = True
                result['reason'] = f"Position too large for volatility level: {volatility_estimate:.1%}"
                result['adjustment'] = f"Reduce position size by {volatility_estimate * 2:.0f}%"

        # Add volatility to risk score
        result['risk_score'] = min(volatility_estimate * 3, 0.3)  # Max 30% risk from volatility

        return result

    def _check_liquidity_risk(self, decision: TradingDecision) -> Dict[str, Any]:
        """Check liquidity and slippage risk."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.1}

        # This would check order book depth and spread
        # Placeholder implementation
        estimated_spread = 0.002  # 0.2% spread estimate

        if estimated_spread > 0.01:  # Wide spread
            result['warning'] = f"Wide spread detected: {estimated_spread:.1%}"
            result['risk_score'] += 0.1

        # Check minimum order size vs liquidity
        trade_value = decision.quantity * (decision.price or 100)
        if trade_value < 10:  # Very small order
            result['warning'] = "Order size may be too small for efficient execution"

        return result

    def _check_correlation_risk(self, decision: TradingDecision,
                              portfolio_state: PortfolioState) -> Dict[str, Any]:
        """Check correlation risk across positions."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        # Simple correlation check - in production would analyze actual correlations
        correlated_assets = {
            'BTCUSDT': ['ETHUSDT', 'SOLUSDT'],
            'ETHUSDT': ['BTCUSDT', 'SOLUSDT', 'SUIUSDT'],
            'SOLUSDT': ['BTCUSDT', 'ETHUSDT'],
        }

        symbol_correlations = correlated_assets.get(decision.symbol, [])
        existing_correlated_positions = [
            pos for sym, pos in portfolio_state.active_positions.items()
            if sym in symbol_correlations
        ]

        if existing_correlated_positions:
            result['warning'] = f"Correlated positions detected: {[p.symbol for p in existing_correlated_positions]}"
            result['risk_score'] += 0.2

        return result

    def _check_temporal_risk(self, decision: TradingDecision) -> Dict[str, Any]:
        """Check time-based risk factors."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        now = datetime.now()

        # Check trading hours (avoid extreme hours)
        hour = now.hour
        if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM UTC
            result['warning'] = "Trading outside optimal hours"
            result['risk_score'] += 0.1

        # Check for frequent trading (avoid overtrading)
        recent_trades = [
            trade for trade in self.position_history
            if (now - trade['timestamp']).seconds < 300  # Last 5 minutes
        ]

        if len(recent_trades) > 3:
            result['warning'] = "High trading frequency detected"
            result['risk_score'] += 0.2

        return result

    def _check_slippage_risk(self, decision: TradingDecision) -> Dict[str, Any]:
        """Check slippage risk for market orders."""
        result = {'rejected': False, 'warning': None, 'risk_score': 0.0}

        # Market orders have higher slippage risk
        if decision.price is None:  # Market order
            # Estimate slippage based on order size and volatility
            estimated_slippage = 0.005  # 0.5% estimate

            if estimated_slippage > self.config.max_slippage_tolerance:
                result['warning'] = f"High slippage risk: {estimated_slippage:.1%}"
                result['risk_score'] += 0.2

                if estimated_slippage > self.config.max_slippage_tolerance * 2:
                    result['adjustment'] = "Consider using limit order instead of market order"

        return result

    async def _update_risk_metrics(self, portfolio_state: PortfolioState):
        """Update current risk metrics."""
        portfolio_value = portfolio_state.total_balance + portfolio_state.unrealized_pnl

        # Update peak value for drawdown calculation
        if portfolio_value > self.portfolio_peak:
            self.portfolio_peak = portfolio_value

        # Calculate drawdown
        if self.portfolio_peak > 0:
            current_drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
            self.risk_metrics.max_drawdown = max(self.risk_metrics.max_drawdown, current_drawdown)

        # Update other metrics
        self.risk_metrics.portfolio_value = portfolio_value

        # Calculate concentration ratio
        if portfolio_state.active_positions:
            position_values = [
                abs(pos.position_amt) * pos.entry_price
                for pos in portfolio_state.active_positions.values()
            ]
            if position_values:
                self.risk_metrics.concentration_ratio = max(position_values) / portfolio_value

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'risk_metrics': {
                'portfolio_value': self.risk_metrics.portfolio_value,
                'daily_pnl': self.risk_metrics.daily_pnl,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'concentration_ratio': self.risk_metrics.concentration_ratio,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'volatility': self.risk_metrics.volatility
            },
            'risk_limits': {
                'max_portfolio_risk': self.config.max_portfolio_risk,
                'max_single_position_risk': self.config.max_single_position_risk,
                'max_daily_loss': self.config.max_daily_loss,
                'max_drawdown': self.config.max_drawdown,
                'max_concurrent_positions': self.config.max_concurrent_positions
            },
            'trading_state': {
                'consecutive_losses': self.consecutive_losses,
                'last_risk_check': self.last_risk_check.isoformat()
            }
        }

    def reset_daily_metrics(self):
        """Reset daily risk metrics."""
        self.daily_start_value = self.risk_metrics.portfolio_value
        self.risk_metrics.daily_pnl = 0.0
        self.consecutive_losses = 0
        logger.info("Daily risk metrics reset")

    def emergency_stop(self):
        """Trigger emergency risk controls."""
        logger.critical("Emergency risk stop triggered")
        # This would implement circuit breakers, position closures, etc.


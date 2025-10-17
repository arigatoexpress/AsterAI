import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskParameters:
    """Configuration for risk management parameters."""
    max_position_size: float = 0.1  # Max position as % of capital
    max_daily_loss: float = 0.05   # 5% max daily loss
    max_drawdown: float = 0.25     # 25% max drawdown for kill switch
    safety_factor: float = 0.25    # Quarter Kelly for position sizing
    max_order_size: float = 0.01   # Max single order as % of capital
    consecutive_losses_threshold: int = 3  # Trigger reduce after this many losses

class RiskManager:
    """
    Risk management layer for the HFT trading system.
    Implements dynamic position sizing and emergency kill switch.
    """
    def __init__(self, initial_capital: float = 100.0, params: RiskParameters = None):
        self.params = params or RiskParameters()
        self.capital = initial_capital
        self.current_position = 0.0
        self.daily_pnl = 0.0
        self.max_capital = initial_capital  # For drawdown calculation
        self.consecutive_losses = 0
        self.is_killed = False
        self.trade_history = []  # For win/loss tracking
        
    def calculate_position_size(self, win_rate: float, avg_win: float, avg_loss: float, 
                               current_volatility: float) -> float:
        """
        Calculate dynamic position size using Kelly Criterion with adjustments.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            current_volatility: Current market volatility
            
        Returns:
            Suggested position size as fraction of capital
        """
        if avg_win == 0 or avg_loss == 0:
            return 0.0
            
        # Kelly Criterion: f = (bp - q) / b, where b = avg_win / avg_loss, p = win_rate, q = 1-p
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply safety factor (quarter Kelly)
        safe_fraction = kelly_fraction * self.params.safety_factor
        
        # Adjust for volatility (reduce size in high vol)
        vol_adjustment = 1 / (1 + current_volatility)
        adjusted_size = safe_fraction * vol_adjustment
        
        # Apply position limits
        max_size = min(adjusted_size, self.params.max_position_size)
        
        return max_size
    
    def check_kill_switch(self, current_pnl: float, drawdown: float) -> bool:
        """
        Check emergency conditions and trigger kill switch if needed.
        
        Args:
            current_pnl: Current session PnL
            drawdown: Current drawdown percentage
            
        Returns:
            True if kill switch activated
        """
        if self.is_killed:
            return True
            
        # Daily loss limit
        if current_pnl < -self.params.max_daily_loss * self.capital:
            self._activate_kill_switch("Daily loss limit exceeded")
            return True
            
        # Max drawdown
        if drawdown > self.params.max_drawdown:
            self._activate_kill_switch("Max drawdown exceeded")
            return True
            
        # Consecutive losses
        if self.consecutive_losses >= self.params.consecutive_losses_threshold:
            self._reduce_risk("Too many consecutive losses")
            
        return False
    
    def _activate_kill_switch(self, reason: str):
        """Activate emergency kill switch and close all positions."""
        self.is_killed = True
        self.current_position = 0.0  # Close all positions
        print(f"🚨 EMERGENCY KILL SWITCH ACTIVATED: {reason}")
        print("All positions closed. System paused for review.")
        
    def _reduce_risk(self, reason: str):
        """Reduce risk parameters temporarily."""
        self.params.max_position_size *= 0.5  # Halve position sizes
        self.consecutive_losses = 0  # Reset counter
        print(f"⚠️ Risk reduced: {reason}")
        
    def validate_order(self, proposed_size: float, proposed_leverage: float) -> Tuple[bool, Optional[str]]:
        """
        Validate proposed order against risk parameters.
        
        Args:
            proposed_size: Proposed position size
            proposed_leverage: Proposed leverage
            
        Returns:
            (is_valid, error_message)
        """
        if self.is_killed:
            return False, "System killed - no new orders"
            
        order_value = proposed_size * self.capital * proposed_leverage
        if order_value > self.params.max_order_size * self.capital:
            return False, "Order size exceeds limit"
            
        if abs(proposed_size) > self.params.max_position_size:
            return False, "Position size exceeds limit"
            
        return True, None
    
    def update_capital(self, pnl: float):
        """Update capital after a trade."""
        self.capital += pnl
        self.daily_pnl += pnl
        self.max_capital = max(self.max_capital, self.capital)
        
        # Update drawdown
        drawdown = (self.max_capital - self.capital) / self.max_capital if self.max_capital > 0 else 0
        
        # Check kill switch
        self.check_kill_switch(self.daily_pnl, drawdown)
        
        # Update trade history for win/loss
        if pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        drawdown = (self.max_capital - self.capital) / self.max_capital if self.max_capital > 0 else 0
        return {
            'capital': self.capital,
            'current_position': self.current_position,
            'daily_pnl': self.daily_pnl,
            'drawdown': drawdown,
            'consecutive_losses': self.consecutive_losses,
            'is_killed': int(self.is_killed)
        }

# Example usage
if __name__ == "__main__":
    risk_mgr = RiskManager(initial_capital=100.0)
    
    # Simulate position sizing
    size = risk_mgr.calculate_position_size(win_rate=0.6, avg_win=2.0, avg_loss=1.0, current_volatility=0.02)
    print(f"Suggested position size: {size:.4f}")
    
    # Simulate order validation
    valid, msg = risk_mgr.validate_order(proposed_size=0.05, proposed_leverage=5)
    print(f"Order valid: {valid}, {msg}")
    
    # Simulate capital update and kill switch
    risk_mgr.update_capital(-30.0)  # Simulate big loss
    print("Risk metrics:", risk_mgr.get_risk_metrics())

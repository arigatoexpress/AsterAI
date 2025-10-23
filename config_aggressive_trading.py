"""
Aggressive Trading Configuration
10% max position size, 5 max open positions
Optimized for high-frequency trading with elevated risk
"""

from typing import Dict, Any

# HIGHLY AGGRESSIVE Risk Parameters for Maximum Profits
AGGRESSIVE_CONFIG = {
    # Position Sizing - AGGRESSIVE
    "max_position_size": 0.10,  # 10% of portfolio per position
    "max_open_positions": 5,    # Maximum 5 open positions
    "min_position_size": 0.05,  # 5% minimum position (larger minimums)

    # HIGH RISK Management - Calculated but Aggressive
    "max_portfolio_risk": 0.60,  # 60% max portfolio risk (higher)
    "max_single_position_risk": 0.15,  # 15% max single position risk (higher)
    "stop_loss_pct": 0.005,     # 0.5% stop loss (TIGHT for perps)
    "take_profit_pct": 0.025,   # 2.5% take profit (aggressive targets)

    # Trading Frequency
    "max_trades_per_hour": 60,  # Up to 60 trades/hour
    "max_trades_per_day": 500,  # Up to 500 trades/day
    "trading_cooldown_seconds": 5,  # 5 second cooldown between trades

    # HIGHLY AGGRESSIVE Strategy Weights - Maximum Profit Focus
    "strategy_weights": {
        "aggressive_perps": 0.7,   # 70% - Our custom aggressive perps strategy
        "degen_trading": 0.2,      # 20% - High risk momentum
        "latency_arbitrage": 0.1   # 10% - Speed-based (reduced for perps focus)
    },

    # MID/SMALL CAP Focus for MAXIMUM PROFIT POTENTIAL
    "target_symbols": [
        "SOLUSDT", "SUIUSDT", "ASTERUSDT", "PENGUUSDT",  # Mid caps
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT"   # Small caps with explosive potential
    ],

    # AGGRESSIVE Entry Conditions (Lower thresholds for more opportunities)
    "min_volume_threshold": 50000,   # $50K minimum volume (lower for small caps)
    "min_price_change_threshold": 0.003,  # 0.3% minimum price movement (more sensitive)
    "max_spread_bps": 10,  # 10 basis points maximum spread (wider for small caps)

    # FAST Exit Conditions (Quick profits, tight stops)
    "profit_lock_in_pct": 0.015,  # Lock in 1.5% profits (faster exits)
    "trailing_stop_pct": 0.008,   # 0.8% trailing stop (tighter)

    # AMBITIOUS Performance Targets
    "daily_profit_target": 0.10,  # 10% daily profit target (aggressive)
    "max_daily_loss": 0.05,       # 5% max daily loss (higher risk tolerance)
    "sharpe_ratio_target": 3.0,   # Target 3.0 Sharpe ratio (very high)

    # Monitoring & Alerts
    "alert_pnl_threshold": 50.0,  # Alert on $50 P&L changes
    "alert_drawdown_threshold": 0.02,  # Alert on 2% drawdown
}

def get_aggressive_config() -> Dict[str, Any]:
    """Get aggressive trading configuration."""
    return AGGRESSIVE_CONFIG.copy()

def validate_aggressive_config(config: Dict[str, Any]) -> bool:
    """Validate aggressive configuration parameters."""
    required_keys = [
        "max_position_size", "max_open_positions", "max_portfolio_risk",
        "stop_loss_pct", "take_profit_pct"
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate ranges
    if not 0 < config["max_position_size"] <= 1.0:
        raise ValueError("max_position_size must be between 0 and 1.0")

    if not 1 <= config["max_open_positions"] <= 10:
        raise ValueError("max_open_positions must be between 1 and 10")

    if not 0 < config["max_portfolio_risk"] <= 1.0:
        raise ValueError("max_portfolio_risk must be between 0 and 1.0")

    return True

# Environment variable overrides
def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    import os

    overrides = {
        "max_position_size": float(os.getenv("MAX_POSITION_SIZE", config["max_position_size"])),
        "max_open_positions": int(os.getenv("MAX_OPEN_POSITIONS", config["max_open_positions"])),
        "max_portfolio_risk": float(os.getenv("MAX_PORTFOLIO_RISK", config["max_portfolio_risk"])),
        "trading_mode": os.getenv("TRADING_MODE", "aggressive")
    }

    config.update(overrides)
    return config

if __name__ == "__main__":
    # Test configuration
    config = get_aggressive_config()
    config = apply_environment_overrides(config)

    print("🚀 Aggressive Trading Configuration:")
    print(f"   Max Position Size: {config['max_position_size']:.0%}")
    print(f"   Max Open Positions: {config['max_open_positions']}")
    print(f"   Max Portfolio Risk: {config['max_portfolio_risk']:.0%}")
    print(f"   Stop Loss: {config['stop_loss_pct']:.1%}")
    print(f"   Take Profit: {config['take_profit_pct']:.1%}")
    print(f"   Daily Profit Target: {config['daily_profit_target']:.0%}")
    print(f"   Max Daily Loss: {config['max_daily_loss']:.0%}")

    try:
        validate_aggressive_config(config)
        print("✅ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")

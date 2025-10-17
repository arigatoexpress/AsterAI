# Risk Management & Capital Scaling Guide

This guide explains how Rari Trade manages risk and grows your capital safely, with practical examples and calculations.

## Understanding Risk in Trading

### Types of Risk
1. **Market Risk**: Prices moving against you
2. **Liquidity Risk**: Can't buy/sell when needed
3. **Operational Risk**: System or execution failures
4. **Capital Risk**: Losing more than you can afford

### Risk Metrics Explained

#### Sharpe Ratio
- **Formula**: (Average Return - Risk-Free Rate) / Standard Deviation
- **What it means**: How much return you get per unit of risk
- **Target**: > 1.5 (good), > 2.0 (excellent)
- **Example**: Sharpe of 2.0 means you earn $2 for every $1 of risk

#### Maximum Drawdown
- **Definition**: Largest peak-to-trough loss
- **Importance**: Measures worst-case scenario
- **Target**: < 10% (conservative), < 20% (moderate)
- **Example**: $10,000 account, 10% drawdown = $1,000 loss

#### Value at Risk (VaR)
- **Definition**: Maximum expected loss over a period
- **Common calculation**: 95% VaR = "95% chance loss won't exceed this amount"
- **Example**: 95% VaR of $500 means only 5% chance of losing more than $500

## Position Sizing Methods

### Kelly Criterion (Mathematical Approach)
```
Formula: Position Size = (Win Rate × Win Size) - Loss Rate) / Win Size
Where:
- Win Rate = Probability of winning (e.g., 0.55)
- Win Size = Average win amount
- Loss Rate = 1 - Win Rate
```

**Example Calculation:**
```
Win Rate: 55% (0.55)
Average Win: $20
Average Loss: $15
Win/Loss Ratio: 20/15 = 1.33

Kelly % = (0.55 × 1.33 - 0.45) / 1.33 = 0.1165 or 11.65%
```

### Fixed Percentage Method (Simple Approach)
- Risk 1-2% of total capital per trade
- Example: $10,000 account = $100-$200 per trade

### Volatility-Adjusted Sizing
- Increase size in low volatility
- Decrease size in high volatility
- Uses ATR (Average True Range) for adjustment

## Capital Scaling Framework

### Four Capital Tiers

#### Tier 1: $0 - $500 (Starting Phase)
```
Risk per Trade: 0.5% ($2.50 max loss)
Position Size: 10% of capital
Daily Loss Limit: 2%
Emergency Stop: 5% drawdown
Strategy: Conservative learning
```

#### Tier 2: $500 - $2,000 (Growth Phase)
```
Risk per Trade: 1.0% ($10 max loss at $2K)
Position Size: 15% of capital
Daily Loss Limit: 2.5%
Emergency Stop: 7% drawdown
Strategy: Balanced growth
```

#### Tier 3: $2,000 - $5,000 (Scaling Phase)
```
Risk per Trade: 1.5% ($75 max loss at $5K)
Position Size: 20% of capital
Daily Loss Limit: 3%
Emergency Stop: 8% drawdown
Strategy: Aggressive growth
```

#### Tier 4: $5,000+ (Professional Phase)
```
Risk per Trade: 2.0% ($200+ max loss at $10K)
Position Size: 25% of capital
Daily Loss Limit: 3%
Emergency Stop: 10% drawdown
Strategy: Full optimization
```

## Practical Risk Management

### Daily Loss Limits
```
Account Size: $1,000
Daily Limit: 3% = $30 maximum loss per day
If loss reaches $30: System stops trading for the day
```

### Emergency Stops
```
Type 1: Daily Loss (resets daily)
Type 2: Portfolio Drawdown (cumulative)
Type 3: Single Trade Loss (per position)
Type 4: System Error (technical issues)
```

### Position Limits
```
Maximum Open Positions: 5 (diversification)
Maximum Single Position: 20% of capital
Correlation Limits: No 2 positions >70% correlated
```

## Real-World Examples

### Example 1: Starting with $100
```
Day 1-7: Paper trading validation
Week 1: $100 → $103 (3% gain, meets criteria)
Week 2: $103 → $108 (4.8% gain)
Week 3: $108 → $115 (6.5% gain)
Week 4: $115 → $125 (8.7% gain)

Total: $100 → $125 (25% in 4 weeks)
Next Tier: Can move to Tier 1 live trading
```

### Example 2: Risk Event
```
Account: $2,000 (Tier 2)
Position Size: 1% risk = $20 max loss
Trade goes wrong: -$18 loss (90% of limit)
System: Reduces next position size by 20%
Next trade risk: $16 instead of $20
```

### Example 3: Scaling Success
```
Month 1: $500 → $600 (20% gain)
Month 2: $600 → $750 (25% gain)
Month 3: $750 → $950 (26.7% gain)
Month 4: $950 → $1,200 (26.3% gain)

Total: $500 → $1,200 (140% in 4 months)
Tier Change: Move from Tier 1 to Tier 2
New limits: 1% risk instead of 0.5%
```

## Monitoring Risk

### Daily Checklist
- [ ] Check portfolio value vs previous day
- [ ] Review open positions and P&L
- [ ] Verify risk limits not exceeded
- [ ] Check system health indicators
- [ ] Review recent trade performance

### Weekly Review
- [ ] Calculate weekly Sharpe ratio
- [ ] Check maximum drawdown
- [ ] Review win rate by strategy
- [ ] Assess position sizing effectiveness
- [ ] Plan for next week's trades

### Monthly Assessment
- [ ] Full performance review
- [ ] Risk parameter adjustment
- [ ] Capital scaling decision
- [ ] Strategy optimization
- [ ] Goal progress evaluation

## Common Risk Scenarios

### Scenario 1: Sudden Market Crash
```
Response: Emergency stop triggered
Action: Close all positions immediately
Result: Preserve remaining capital
Recovery: Wait for market stabilization
```

### Scenario 2: Winning Streak
```
Response: Scale up position sizes gradually
Action: Increase risk by 25% per tier
Result: Compound gains safely
Recovery: Monitor for overconfidence
```

### Scenario 3: Losing Period
```
Response: Reduce position sizes
Action: Drop to lower tier risk levels
Result: Give system time to recover
Recovery: Analyze what went wrong
```

## Risk Management Tools

### Stop Loss Types
1. **Fixed Percentage**: 2% below entry
2. **ATR-Based**: 1.5 × Average True Range
3. **Support Level**: Major technical support
4. **Time-Based**: Exit after X hours/days

### Position Scaling
```
Initial Size: 1% of capital
Add to Winners: Scale up by 50% every 10% profit
Example:
- Entry: $100 position
- +10%: Add $50, total $150
- +10% more: Add $75, total $225
```

### Diversification Rules
```
Maximum per asset: 20% of capital
Maximum per sector: 40% of capital
Maximum correlation: 0.7 between positions
Minimum assets: 3-5 positions
```

## Emergency Procedures

### System Alert Levels
- **Green**: All systems normal
- **Yellow**: Warning, monitor closely
- **Red**: Critical, immediate action required
- **Black**: Emergency shutdown

### Manual Override
```
When to use: Only in extreme situations
How to use: Dashboard emergency stop button
After use: Full system review required
```

## Learning from Risk Events

### Post-Mortem Process
1. **Document the event**: What happened, when, why
2. **Analyze causes**: Technical, market, or system factors
3. **Update procedures**: Modify rules to prevent recurrence
4. **Test changes**: Paper trade new rules
5. **Implement gradually**: Don't change everything at once

### Continuous Improvement
- Track all risk events in log
- Monthly risk review meeting
- Update risk parameters based on data
- Train on past mistakes

---

## Key Takeaways

1. **Risk management is more important than returns**
2. **Scale capital gradually, not aggressively**
3. **Have multiple layers of protection**
4. **Monitor continuously, act decisively**
5. **Learn from every loss and win**

*Risk management turns trading from gambling into a probability game you can win over time.*

# Telegram Bot - Quick Command Reference

## Essential Commands (Memorize These)

```
/status              - Quick system check
/emergency_close_all - CLOSE ALL POSITIONS NOW
```

---

## Monitoring Commands

### Quick Checks
```
/status          - Bot status, balance, positions, trades
/positions       - Open positions with entry prices and P&L
/portfolio       - Balance, equity, margin used
```

### Detailed Analysis
```
/performance     - Win rate, P&L, Sharpe ratio, drawdown
/metrics         - Detailed trading metrics
/system_status   - Cloud + local infrastructure status
```

### Diagnostics
```
/logs            - Last 20 log entries
/logs 50         - Last 50 log entries
/architecture    - System architecture and services
```

---

## Control Commands

### Trading Control
```
/start_trading   - Start automated trading
/stop_trading    - Stop trading (close positions gracefully)
/restart         - Restart the trading bot
```

### Emergency Controls
```
/emergency_stop       - IMMEDIATE SHUTDOWN
/emergency_close_all  - CLOSE ALL POSITIONS NOW
```

### Configuration
```
/leverage        - Check current leverage setting
/leverage 1.5    - Set leverage to 1.5x (max 2x)
/rebalance       - Rebalance portfolio
```

---

## Command Details

### /status
**What it shows**:
- Bot running or stopped
- Current balance
- Number of active positions
- Total trades executed
- Win rate percentage
- Total P&L

**When to use**: Every 2-4 hours for quick check

---

### /positions
**What it shows**:
- List of all open positions
- Entry price for each
- Current quantity
- Side (LONG/SHORT)
- Unrealized P&L

**When to use**: Before going to bed, after waking up

---

### /performance
**What it shows**:
- Total trades executed
- Winning vs losing trades
- Win rate percentage
- Total P&L (realized + unrealized)
- Current balance
- Active positions count

**When to use**: Daily review, end of day

---

### /system_status
**What it shows**:
- Cloud service status (healthy/unhealthy)
- Local PC status (online/offline)
- Current balance
- Active positions
- Win rate

**When to use**: If something seems wrong, troubleshooting

---

### /emergency_close_all
**What it does**:
- Immediately stops all trading
- Closes all open positions at market price
- Logs emergency action
- Sends confirmation

**When to use**: 
- Extreme market volatility
- System behaving unexpectedly
- You need to stop everything NOW

**IMPORTANT**: This is instant and irreversible for current positions

---

### /leverage [X]
**What it does**:
- Without number: Shows current leverage
- With number: Sets leverage (max 2x for safety)

**Examples**:
```
/leverage         # Shows: "Current leverage: 2.0x"
/leverage 1.5     # Sets to 1.5x
/leverage 3       # Sets to 2.0x (capped)
```

**When to use**: 
- To reduce risk: `/leverage 1`
- To check current: `/leverage`
- Normal operation: Keep at 2x

---

### /logs [N]
**What it shows**:
- Recent log entries
- Error messages
- Trading actions
- System events

**Examples**:
```
/logs       # Last 20 entries
/logs 50    # Last 50 entries
```

**When to use**: Investigating issues, understanding recent activity

---

## Monitoring Schedule

### Every 2-4 Hours
```
/status
```
Just to check everything is running

### Morning & Evening
```
/performance
/positions
```
Review P&L and open trades

### Daily
```
/system_status
/logs 20
```
Full health check

### As Needed
```
/leverage              # Check settings
/rebalance             # Rebalance portfolio
/emergency_close_all   # Emergency use only
```

---

## Expected Responses

### Normal Operation
```
/status

[ACTIVE] Bot Status: Running

[DATA] Details:
{
  "status": "running",
  "balance": 103.50,
  "active_positions": 1,
  "total_trades": 15
}
```

### With Open Positions
```
/positions

[TARGET] OPEN POSITIONS

[CHART] BTCUSDT
[MONEY] Entry: $67,234.50
[NUM] Quantity: 0.0015
[DATA] Side: LONG
```

### Good Performance
```
/performance

[DATA] TRADING METRICS

[TARGET] Total Trades: 25
[MONEY] Total P&L: $5.75
[CHART] Win Rate: 64.0%
[NUM] Active Positions: 1
```

### Circuit Breaker Triggered
```
[WARNING] CIRCUIT BREAKER TRIPPED

Daily loss limit reached: -5.1% (limit: 5.0%)
Trading halted until tomorrow
```

---

## Error Responses

### Bot Not Initialized
```
[ERROR] Trading bot not initialized

Please wait for system startup or contact support
```
**Action**: Wait 1-2 minutes, try again

### Service Unavailable
```
[ERROR] Error getting status: Connection timeout
```
**Action**: Check `/system_status`, service may be restarting

### Invalid Command
```
[ERROR] Unknown command. Use /help for available commands.
```
**Action**: Check spelling, use `/help` to see valid commands

---

## Quick Decision Matrix

### If Balance is Growing
- Continue monitoring with `/status`
- Check `/performance` daily
- No action needed
- Let system trade automatically

### If Balance is Flat
- Check `/positions` for activity
- Review `/logs 20` for issues
- Consider `/rebalance`
- System is being conservative (normal)

### If Balance is Declining
- Check how much: `/performance`
- If < -3%: Monitor closely
- If approaching -5%: Circuit breaker will halt
- If > -5%: Already halted (check for alert)

**Manual action**: `/emergency_close_all` if you want to exit all

### If Circuit Breaker Trips
- **Automatic**: Trading halts, alert sent
- **Your action**: Review `/logs 50` to understand why
- **Decision**: Let it resume tomorrow, or keep it halted
- **To resume**: Bot auto-resumes next day, or send `/start_trading`

---

## Safety Reminders

1. **Daily Loss Limit**: System automatically halts at -5%
2. **Max Leverage**: Hard-capped at 2x (cannot exceed)
3. **Position Size**: 1.5% of capital per trade
4. **Stop Loss**: 1.5% per position (automatic)
5. **Emergency Control**: `/emergency_close_all` always available

---

## Tips for Successful Remote Trading

### Before Leaving
1. Test all commands once
2. Verify you receive messages
3. Have `/emergency_close_all` command ready
4. Set Telegram notifications to alert you

### During Travel
1. Check `/status` every few hours
2. Don't over-manage - let system work
3. Use `/emergency_close_all` only if truly needed
4. Trust the circuit breaker (it will protect you)

### When to Intervene
- Circuit breaker trips multiple times
- Win rate drops below 40%
- System not responding to commands
- Unusual position sizes or behavior

### When to Let It Run
- Win rate above 50%
- Profit is growing steadily
- Circuit breaker hasn't tripped
- System responding normally to commands

---

## Command Response Times

- `/status`: <2 seconds
- `/positions`: <3 seconds
- `/performance`: <3 seconds
- `/system_status`: <5 seconds
- `/logs`: <5 seconds
- `/emergency_close_all`: <10 seconds

**If slower**: System may be under load, wait and retry

---

## Troubleshooting via Telegram

### Bot Not Responding
1. Wait 1-2 minutes
2. Send `/help` (simplest command)
3. If still no response, check webhook configuration

### Commands Return Errors
1. Send `/system_status` - Check if services are up
2. Send `/logs 20` - Look for error messages
3. If cloud down, local may still work (hybrid mode)

### Can't Close Positions
1. Try `/emergency_close_all` multiple times
2. If fails, positions may have already closed
3. Check `/positions` to verify
4. Contact support if positions stuck

---

## Support & Help

### Self-Help
1. `/help` - Full command list
2. `/logs` - Recent activity
3. `/system_status` - Infrastructure health

### Documentation
- `QUICK_START_REMOTE_TRADING.md` - Full guide
- `DEPLOYMENT_CHECKLIST.md` - Setup steps
- `ACTION_PLAN_BEFORE_TRAVEL.md` - Pre-departure tasks

### Emergency
- `/emergency_close_all` - Close everything
- `/emergency_stop` - Halt trading
- Circuit breaker - Automatic protection at 5% loss

---

**MEMORIZE**:
```
/status
/emergency_close_all
```

**CHECK REGULARLY**:
```
/status (every few hours)
/performance (daily)
```

**USE IF NEEDED**:
```
/emergency_close_all (market crash)
/stop_trading (pause system)
/start_trading (resume system)
```

---

*Quick Reference Guide*
*AsterAI v2.0.0*
*Production Remote Trading System*


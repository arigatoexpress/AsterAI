# AsterAI Quick Start - Remote Trading Setup

## Complete Setup in 30 Minutes

This guide will get you from zero to fully operational remote-controlled trading system.

---

## Prerequisites

- Windows PC with Python 3.10+
- Google Cloud account with billing enabled
- Aster DEX API credentials
- Telegram account

---

## Step 1: Configure API Keys (5 minutes)

```powershell
# Run the manual API key setup
python scripts/manual_update_keys.py
```

When prompted, paste your:
- Aster API Key
- Aster Secret Key

Keys are saved securely to `local/.api_keys.json`

---

## Step 2: Setup Telegram Bot (10 minutes)

### Create Bot

1. Open Telegram, search for `@BotFather`
2. Send `/newbot`
3. Follow prompts to create your bot
4. Copy the bot token

### Configure Bot

```powershell
python setup_telegram_bot.py
```

Enter:
- Bot token (from BotFather)
- Your chat ID (get from https://api.telegram.org/botYOUR_TOKEN/getUpdates after sending /start to bot)
- Webhook URL (your Cloud Run URL or leave blank for polling)

---

## Step 3: Deploy Complete System (15 minutes)

```powershell
# One command to deploy everything
python deploy_complete_system.py
```

This will:
1. Deploy cloud trading service to GCP
2. Start local background service
3. Connect Telegram bot
4. Run system tests

---

## Step 4: Start Trading via Telegram

Send these commands to your bot:

```
/status - Check system status
/start_trading - Begin automated trading
/positions - View open positions
```

---

## Remote Control Commands

### Monitoring
- `/status` - Bot status and metrics
- `/portfolio` - Current balance and equity
- `/positions` - Open positions
- `/performance` - Trading performance
- `/system_status` - Cloud + local status
- `/logs [N]` - Recent log entries

### Control
- `/start_trading` - Start automated trading
- `/stop_trading` - Stop trading
- `/restart` - Restart bot
- `/emergency_stop` - Emergency shutdown
- `/emergency_close_all` - Close all positions immediately

### Configuration
- `/leverage` - Check current leverage
- `/leverage 2` - Set leverage to 2x (max 2x)
- `/rebalance` - Rebalance portfolio

---

## Safety Features

### Automatic Protection
- **Daily Loss Limit**: Auto-halts at 5% daily loss
- **Loss Streak Pause**: Pauses after 3 consecutive losses
- **Position Limits**: Max 2 concurrent positions
- **Conservative Sizing**: 1.5% of capital per trade
- **Tight Stop Losses**: 1.5% per position
- **Leverage Cap**: 2x maximum

### Manual Controls
- Emergency stop via `/emergency_stop`
- Emergency close all positions via `/emergency_close_all`
- Remote shutdown capability

---

## Monitoring While Away

### Recommended Schedule
- **Every 2-4 hours**: Send `/status`
- **Daily**: Check `/performance`
- **As needed**: Use `/positions` to monitor

### Alert Notifications
You'll receive automatic alerts for:
- Trade executions
- Circuit breaker trips
- System errors
- Daily summaries

---

## System Architecture

```
Your Trading System
├── Cloud (Google Cloud Run)
│   ├── Trading execution
│   ├── Risk management
│   ├── Always available
│   └── Auto-scaling
│
├── Local PC (Background Service)
│   ├── ML model training (GPU)
│   ├── Advanced analytics
│   ├── Backup execution
│   └── Development mode
│
└── Telegram Bot
    ├── Remote monitoring
    ├── Full control
    ├── Alerts & notifications
    └── Emergency controls
```

---

## Troubleshooting

### Cloud Service Not Responding
```powershell
# Check cloud status
curl https://aster-self-learning-trader-880429861698.us-central1.run.app/health

# Redeploy if needed
python deploy_complete_system.py
```

### Local Service Not Starting
```powershell
# Check if running
curl http://localhost:8081/api/control/status

# Restart
local\run_local_trader.bat
```

### Telegram Bot Not Responding
```powershell
# Verify configuration
type local\.telegram_config

# Reconfigure
python setup_telegram_bot.py
```

---

## Configuration Files

- `local/.api_keys.json` - Aster DEX API credentials
- `local/.telegram_config` - Telegram bot configuration
- `cloud/cloudbuild.yaml` - Cloud deployment configuration
- `requirements.txt` - Python dependencies

---

## Expected Performance

### Conservative Mode (Default)
- **Target Return**: 2-5% during travel period
- **Win Rate**: 55-65%
- **Max Drawdown**: <10%
- **Daily Loss Limit**: 5% (auto-halt)

### Risk Controls
- Max capital at risk: $100-500
- Max leverage: 2x
- Position size: 1.5% per trade
- Stop loss: 1.5%
- Take profit: 3%

---

## Important Notes

1. **Keep PC On**: If using hybrid mode, keep your PC running for local service
2. **Monitor Telegram**: Check bot messages regularly
3. **Respect Limits**: System will auto-halt at 5% daily loss
4. **Emergency Access**: Always have emergency_close_all command ready
5. **Test First**: Use `/status` to verify everything works before traveling

---

## Support

If you encounter issues:
1. Check `/system_status` for diagnostics
2. Review `/logs 50` for error messages
3. Use `/emergency_stop` if something seems wrong
4. System auto-halts on 5% daily loss for safety

---

**System Status**: Production Ready
**Mode**: Conservative Live Trading
**Remote Control**: Fully Operational via Telegram

---

*Last Updated: January 2025*
*Version: 2.0.0*


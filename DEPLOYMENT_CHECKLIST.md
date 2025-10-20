# AsterAI Production Deployment Checklist

## Pre-Deployment (Before You Leave Tomorrow)

### Phase 1: Clean Remaining Emojis (10 minutes)
```powershell
# Remove any remaining emojis from all files
python scripts/remove_all_emojis.py

# Verify no encoding errors
$env:PYTHONIOENCODING="utf-8"
python -c "import mcp_trader; print('[OK] No encoding errors')"
```

**Expected Result**: All emojis replaced with ASCII equivalents

---

### Phase 2: Configure Telegram Bot (15 minutes)

#### Step 1: Create Telegram Bot
1. Open Telegram
2. Search for `@BotFather`
3. Send `/newbot`
4. Follow prompts, choose a name like "AsterAI Trader Bot"
5. **Copy the bot token** (looks like: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)

#### Step 2: Get Your Chat ID
1. Send `/start` to your new bot
2. Visit: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
3. Look for `"chat":{"id":YOUR_CHAT_ID}`
4. **Copy your chat ID** (number like: 123456789)

#### Step 3: Configure Bot
```powershell
python setup_telegram_bot.py
```

Enter:
- Bot token (from step 1)
- Chat ID (from step 2)
- Webhook URL: `https://aster-self-learning-trader-880429861698.us-central1.run.app`

**Expected Result**: Configuration saved, webhook set

---

### Phase 3: Add Telegram Secrets to GCP (5 minutes)

```powershell
# Get your bot token and chat ID from local/.telegram_config
type local\.telegram_config

# Add to GCP Secret Manager
echo -n "YOUR_BOT_TOKEN" | gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=-
echo -n "YOUR_CHAT_ID" | gcloud secrets create TELEGRAM_CHAT_ID --data-file=-

# Or update if they exist
echo -n "YOUR_BOT_TOKEN" | gcloud secrets versions add TELEGRAM_BOT_TOKEN --data-file=-
echo -n "YOUR_CHAT_ID" | gcloud secrets versions add TELEGRAM_CHAT_ID --data-file=-
```

**Expected Result**: Secrets created/updated in GCP

---

### Phase 4: Deploy Cloud Service (20 minutes)

```powershell
# Deploy using cloud build
gcloud builds submit --config=cloud/cloudbuild.yaml --timeout=1200s
```

**This will**:
1. Build Docker image
2. Push to Google Container Registry
3. Deploy to Cloud Run
4. Configure auto-scaling (1-3 instances)
5. Set environment variables and secrets

**Expected Result**: Service URL displayed, deployment successful

#### Verify Cloud Deployment
```powershell
# Check health
curl https://aster-self-learning-trader-880429861698.us-central1.run.app/health

# Check status (should NOT be "not_initialized")
curl https://aster-self-learning-trader-880429861698.us-central1.run.app/status

# Check system summary
curl https://aster-self-learning-trader-880429861698.us-central1.run.app/system/summary
```

**Expected**: All endpoints respond with 200 OK, status shows "stopped" or "running" (not "not_initialized")

---

### Phase 5: Start Local Service (5 minutes)

```powershell
# Start local background service
local\run_local_trader.bat
```

**Expected Result**: 
- Local dashboard starts on port 8081
- Background window opens (minimized)
- Message: "[OK] Local trader started successfully"

#### Verify Local Service
```powershell
# Check status
curl http://localhost:8081/api/control/status

# View in browser
start http://localhost:8081
```

**Expected**: Dashboard loads, shows real-time price data

---

### Phase 6: Test Telegram Bot (10 minutes)

Send these commands to your Telegram bot to verify everything works:

```
/status
```
**Expected**: Bot responds with system status

```
/system_status
```
**Expected**: Shows both cloud and local status

```
/portfolio
```
**Expected**: Shows balance ($100), equity, margin

```
/help
```
**Expected**: Lists all available commands

```
/leverage
```
**Expected**: Shows current leverage (2x)

---

### Phase 7: Final System Check (5 minutes)

```powershell
# Run comprehensive test
python test_complete_integration.py
```

**Expected**: All tests pass

#### Manual Verification Checklist
- [ ] Cloud service responds to /health
- [ ] Cloud service status is NOT "not_initialized"
- [ ] Local dashboard loads in browser
- [ ] Telegram bot responds to /status
- [ ] /system_status shows both cloud and local
- [ ] No emoji-related errors in logs
- [ ] Circuit breaker is initialized

---

## Deployment Day (Day of Travel)

### Start Trading (Option A: Conservative Start)

**Don't start trading yet** - Monitor system for a few hours first:

```
Send to bot: /system_status
```

Check every hour to ensure stability.

### Start Trading (Option B: Immediate Start)

```
Send to bot: /start_trading
```

**Expected**: Bot confirms trading started

**Monitor for first 2 hours**:
- Send `/status` every 15 minutes
- Watch for any errors
- Check `/positions` for activity

---

## While Away - Monitoring Protocol

### Every 2-4 Hours
```
/status          # Quick check
/positions       # Monitor positions
```

### Daily (Morning/Evening)
```
/performance     # Full P&L review
/system_status   # Infrastructure health
```

### If Issues Arise
```
/logs 20         # Check recent activity
/emergency_stop  # Pause trading
/emergency_close_all  # Close everything
```

---

## Emergency Procedures

### Circuit Breaker Triggered (5% Daily Loss)
**What happens**:
- Trading auto-halts immediately
- Telegram alert sent
- No new positions opened
- Existing positions remain (unless you close them)

**Your action**:
```
/positions               # Check what's open
/emergency_close_all     # Close if you want
# System will resume tomorrow automatically
```

### System Offline
**If cloud service down**:
- Local service continues (if PC on)
- System will auto-recover
- No action needed

**If local service down**:
- Cloud service continues
- No GPU training until PC back online
- Trading execution unaffected

**If both down**:
- Send `/system_status` to check
- Wait 5-10 minutes for auto-recovery
- If still down, services will resume when systems restart

### Extreme Market Volatility
```
/emergency_close_all     # Close all positions
/stop_trading            # Halt trading
# Wait for stability, then /start_trading
```

---

## Return Home - Post-Trip Analysis

### Day 1: Review Performance
```powershell
# Check final results
Send to bot: /performance

# Review detailed logs
Send to bot: /logs 100

# Analyze trades
python -c "from live_trading_agent import LiveTradingAgent; # analyze"
```

### Day 2: Optimize Based on Results
1. Review which strategies performed best
2. Adjust weights if needed
3. Consider scaling up capital if profitable
4. Update risk parameters based on experience

---

## Expected Outcomes

### Conservative Scenario (Most Likely)
- **7 Days**: +3-5% ($103-105)
- **14 Days**: +6-10% ($106-110)
- **Circuit Breaker Trips**: 0-1 times
- **Win Rate**: 55-65%
- **Max Drawdown**: 3-7%

### Optimistic Scenario
- **7 Days**: +7-10% ($107-110)
- **14 Days**: +15-20% ($115-120)
- **Win Rate**: 65-75%
- **Max Drawdown**: <5%

### Worst Case (Protected)
- **Any Day**: -5% ($95) then AUTO-HALT
- System pauses, you get alert
- Resume next day or manual restart
- Capital preserved

---

## Quick Reference Commands

### Essential Telegram Commands
```
/status              # Quick health check
/positions           # What's open now
/performance         # How much money made/lost
/system_status       # Full infrastructure status
/emergency_close_all # CLOSE EVERYTHING NOW
/stop_trading        # Pause bot
/start_trading       # Resume bot
/help                # Show all commands
```

### Emergency Stop Procedures
1. Send `/emergency_stop` - Stops trading
2. Send `/emergency_close_all` - Closes all positions
3. Send `/stop_trading` - Pauses bot
4. System will auto-halt at 5% daily loss anyway

---

## Files to Review After Deployment

### Logs
- `logs/dashboard.log` - Local service logs
- GCP Cloud Logging - Cloud service logs

### Performance Data
- `live_trading_results/*.json` - Trading history
- Telegram message history - Real-time updates

### System Status
- `/status` endpoint - Current state
- `/metrics` endpoint - Performance metrics
- `/system/summary` - Full architecture status

---

## Troubleshooting Quick Fixes

### "Bot not responding in Telegram"
```powershell
# Check webhook
curl https://api.telegram.org/botYOUR_TOKEN/getWebhookInfo

# Reset webhook
python setup_telegram_bot.py
```

### "Cloud service shows not_initialized"
```powershell
# Redeploy with auto-init fix
gcloud builds submit --config=cloud/cloudbuild.yaml --timeout=1200s
```

### "Local service won't start"
```powershell
# Check if port 8081 is in use
netstat -ano | findstr :8081

# Kill existing process if needed
Get-Process -Id PID_NUMBER | Stop-Process -Force

# Restart
local\run_local_trader.bat
```

### "Encoding errors with emojis"
```powershell
# Run emoji removal tool
python scripts/remove_all_emojis.py

# Set encoding
$env:PYTHONIOENCODING="utf-8"
```

---

## Success Criteria

### System is Ready When:
- [ ] Cloud /health returns 200 OK
- [ ] Cloud /status shows "stopped" (not "not_initialized")
- [ ] Local dashboard loads at http://localhost:8081
- [ ] Telegram /status command works
- [ ] /system_status shows both cloud and local
- [ ] No emoji errors in any logs
- [ ] All tests in test_complete_integration.py pass

### Trading is Safe When:
- [ ] Conservative mode enabled (check /leverage)
- [ ] Circuit breaker initialized (check logs)
- [ ] Daily loss limit is 5%
- [ ] Max leverage is 2x
- [ ] Position size is 1.5%
- [ ] You can execute /emergency_close_all

---

## Final Checklist Before Leaving

### Technical
- [ ] All emojis removed from codebase
- [ ] Cloud service deployed and responding
- [ ] Local service running (if using hybrid mode)
- [ ] Telegram bot configured and tested
- [ ] GCP secrets configured (API keys, Telegram)
- [ ] Circuit breaker tested

### Safety
- [ ] Conservative mode confirmed (max 2x leverage)
- [ ] Daily loss limit set to 5%
- [ ] Emergency commands tested
- [ ] Telegram alerts working
- [ ] PC will stay on (if using local service)

### Communication
- [ ] You can send/receive Telegram messages
- [ ] Bot responds to all commands
- [ ] Alerts are received
- [ ] You have the emergency commands ready

---

**READY FOR DEPLOYMENT**: All systems go!

**Contact Info**: Telegram bot for real-time support

**Emergency Command**: `/emergency_close_all` (memorize this!)

---

*Last Updated: January 20, 2025*
*Deployment Version: 2.0.0 - Production*
*Status: Ready for Remote Trading*


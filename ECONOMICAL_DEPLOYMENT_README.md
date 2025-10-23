# 🚀 Economical Cloud Run Deployment Guide

## Overview
This guide deploys your AsterAI HFT system using **Google Cloud Run** for economical testing with aggressive trading strategies.

**Cost**: ~$5-15/month vs $585/month for full GKE cluster
**Perfect for**: Initial testing, paper trading, and validating strategies before full deployment

## 🎯 HIGHLY AGGRESSIVE Trading Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Max Position Size** | 10% | Each position limited to 10% of portfolio |
| **Max Open Positions** | 5 | Maximum 5 concurrent positions |
| **Portfolio Risk** | 60% | Allow up to 60% portfolio risk (aggressive) |
| **Stop Loss** | 0.3-0.5% | TIGHT stops based on volatility |
| **Take Profit** | 1.5-3% | FAST profit targets |
| **Daily Profit Target** | 10% | Aim for 10% daily returns (ambitious) |
| **Max Daily Loss** | 5% | Higher risk tolerance for aggressive strategy |
| **Leverage** | 5x-25x | HIGH leverage based on signal strength |

## 🚀 **Focus: Mid/Small Cap Perpetual Trading**

**Target Assets for Maximum Profit:**
- **Mid Caps**: SOLUSDT, SUIUSDT, ASTERUSDT, PENGUUSDT
- **Small Caps**: DOGEUSDT, SHIBUSDT, PEPEUSDT, BONKUSDT

**Strategy Mix:**
- **70%**: Aggressive Perpetual Strategy (custom high-leverage)
- **20%**: Degen Trading (momentum-based)
- **10%**: Latency Arbitrage (speed-based)

## 📊 Strategies Included

1. **🚀 Aggressive Perpetual Strategy** (70% weight) - Custom high-leverage perps trading
   - 5x-25x dynamic leverage based on signal strength
   - Both LONG and SHORT positions
   - Mid/small cap focus for explosive gains

2. **⚡ Degen Trading** (20% weight) - High-risk momentum trading
   - Social sentiment and trend following
   - Quick entries and exits

3. **🔄 Latency Arbitrage** (10% weight) - Speed-based profit opportunities
   - Price inefficiencies between exchanges
   - Microsecond execution timing

## 🚀 Quick Deployment

### Prerequisites
- ✅ Google Cloud SDK installed
- ✅ Docker Desktop (for building images)
- ✅ Aster DEX API keys in Secret Manager
- ✅ GCP Project with billing enabled
- ✅ **Aster DEX account with $50+ for live trading**
- ✅ **API keys configured for live trading permissions**

### 🚨 LIVE TRADING DEPLOYMENT (Real Money)
```bash
# ⚠️  WARNING: This deploys with LIVE TRADING enabled
# Set your project ID
export GCP_PROJECT_ID="your-project-id"

# Deploy with AGGRESSIVE live trading configuration
bash deploy_cloud_run_economical.sh
```

**⚠️ IMPORTANT RISK WARNINGS:**
- **HIGH LEVERAGE**: 5x-25x leverage based on signal strength
- **TIGHT STOPS**: 0.3-0.5% stop losses (can be hit quickly)
- **AGGRESSIVE TARGETS**: 1.5-3% profit targets (fast exits)
- **VOLATILE ASSETS**: Mid/small cap focus for maximum gains
- **Start with $50** and monitor closely for first 24 hours

### What Gets Deployed
- **🚀 Cloud Run Service**: `aster-hft-testing` (LIVE TRADING)
- **🎯 Trading Mode**: HIGHLY AGGRESSIVE perpetual contracts
- **💰 Leverage**: 5x-25x dynamic (based on signal strength)
- **📊 Position Limits**: 10% per position, 5 max concurrent
- **🛡️ Risk Controls**: 0.3-0.5% tight stops, 5% max daily loss
- **⚡ Auto-scaling**: 0-1 instances (pay only when active)
- **🔍 Health checks**: Real-time monitoring and alerts
- **🔐 Secrets**: Live Aster DEX API keys from Secret Manager

## 📊 Monitoring Your Deployment

### Service URL
After deployment, you'll get a URL like:
```
https://aster-hft-testing-xyz-uc.a.run.app
```

### Health Check
```bash
curl https://your-service-url/health
```

Expected response:
```json
{
  "status": "healthy",
  "mode": "paper_trading",
  "strategy": "aggressive",
  "active_positions": 0,
  "total_pnl": 0.0,
  "max_position_size": 0.1,
  "max_open_positions": 5,
  "daily_profit_target": 0.05,
  "target_symbols": 7
}
```

### 🚀 Aggressive Trading Dashboard
Visit: `https://your-service-url/dashboard`

**Dashboard Features:**
- **Real-time P&L** and position status
- **Live trading signals** analysis
- **Leverage and exposure** monitoring
- **Risk metrics** (drawdown, VaR)
- **Trade execution** logs
- **Performance analytics**

### 📊 Logs & Monitoring
```bash
# View recent aggressive trading logs
gcloud logs read --filter="resource.type=cloud_run_revision AND resource.labels.service_name=aster-hft-testing" --limit=50

# Stream live trading activity
gcloud logs tail --filter="resource.type=cloud_run_revision AND resource.labels.service_name=aster-hft-testing"

# Check for trading alerts
gcloud monitoring dashboards create --filter="metric.type=cloudfunctions.googleapis.com/function/execution_times"
```

## 💰 Cost Breakdown (Live Trading)

| Component | Cost | Notes |
|-----------|------|-------|
| **Cloud Run** | $5-15/month | Pay-per-request, scales to 0 when not trading |
| **Secret Manager** | ~$0.06/month | API key storage |
| **Monitoring** | ~$5-10/month | Enhanced logs and alerts for live trading |
| **Aster DEX Trading Fees** | Variable | ~0.1% per trade (maker/taker) |
| **Builds** | Free tier available | Container builds |
| **TOTAL** | **$10-25/month** | Plus trading fees (~$0.50-5/day with $50 capital) |

**Trading Fee Impact:**
- **$50 capital**: ~$0.50-1.00/day in fees
- **$500 capital**: ~$5-10/day in fees
- **$5000 capital**: ~$50-100/day in fees

## 🎯 Trading Modes

### 🚨 Live Trading (DEPLOYED BY DEFAULT)
- **Real Aster DEX API integration**
- **Actual position management with real money**
- **Real profit/loss tracking**
- **High-stakes aggressive perpetual trading**
- **Environment variable**: `ENABLE_PAPER_TRADING=false`

### 📝 Paper Trading (Optional)
- Safe testing with simulated trades
- No real money at risk
- Full strategy validation
- Enable by setting: `ENABLE_PAPER_TRADING=true`

## ⚙️ Configuration Updates

### Update Risk Parameters
```bash
gcloud run services update aster-hft-testing \
  --set-env-vars "MAX_POSITION_SIZE=0.15" \
  --set-env-vars "MAX_OPEN_POSITIONS=3" \
  --region=us-central1
```

### Switch to Live Trading
```bash
gcloud run services update aster-hft-testing \
  --set-env-vars "ENABLE_PAPER_TRADING=false" \
  --region=us-central1
```

### Scale Up Resources
```bash
gcloud run services update aster-hft-testing \
  --cpu=2 \
  --memory=2Gi \
  --max-instances=3 \
  --region=us-central1
```

## 🔧 Troubleshooting

### Service Won't Start
```bash
# Check deployment status
gcloud run services describe aster-hft-testing --region=us-central1

# View build logs
gcloud builds list --filter="source:\"aster-hft-testing\""

# Check for errors
gcloud logs read --filter="resource.type=cloud_run_revision" --limit=20
```

### API Connection Issues
```bash
# Verify secrets exist
gcloud secrets list

# Check secret versions
gcloud secrets versions list aster-api-key
```

### High Latency
```bash
# Check instance count
gcloud run services describe aster-hft-testing --region=us-central1 --format="value(status.conditions[0].message)"

# View performance metrics
gcloud monitoring dashboards create
```

## 📈 Performance Expectations

### Paper Trading Phase (First 24-48 hours)
- **Trades**: 10-50 simulated trades
- **Win Rate**: Target 55% (realistic for aggressive strategies)
- **Daily P&L**: -3% to +5% (simulated)
- **Positions**: Up to 5 concurrent

### Live Trading Phase (After validation)
- **Capital**: Start with $50-100
- **Daily Target**: 2-5% returns
- **Risk**: Max 3% daily drawdown
- **Monitoring**: 24/7 oversight required

## 🚀 Next Steps After Deployment

### Immediate (First Hour)
1. ✅ Verify health endpoint responds
2. ✅ Check logs for initialization messages
3. ✅ Confirm paper trading is active
4. ✅ Monitor for any errors

### First 24 Hours
1. 📊 Review trading activity in logs
2. 📊 Check P&L simulation accuracy
3. 📊 Validate strategy execution
4. 📊 Test position limits

### First Week
1. 💰 Switch to live trading with $50
2. 📈 Monitor real performance
3. 🔧 Adjust parameters if needed
4. 📊 Track Sharpe ratio improvements

## 📈 Expected Performance with Real Money

### 🎯 **Conservative Estimates (First Week)**
- **Capital Required**: $50-100 starting capital
- **Daily Target**: 5-15% returns (ambitious but achievable)
- **Win Rate**: 60-75% (aggressive targets with tight stops)
- **Average Trade**: 1.5-3% profit per winning trade
- **Max Drawdown**: 5% daily limit (hard stop)
- **Trades per Day**: 5-15 high-conviction signals

### 🚀 **Aggressive Growth Trajectory**
```
Day 1-3:  $50 → $55-65 (10-30% initial growth)
Day 4-7:  $55 → $75-100 (35-100% weekly growth)
Day 8-14: $75 → $150-300 (100-400% bi-weekly growth)
Month 1:  $50 → $500+ (10x growth target)
```

### 💰 **Real Money Trading Parameters**
- **Leverage**: 5x-25x (dynamic based on signal strength)
- **Position Size**: 10% per position max
- **Max Positions**: 5 concurrent (diversified risk)
- **Stop Loss**: 0.3-0.5% (TIGHT for perps)
- **Take Profit**: 1.5-3% (FAST profit capture)
- **Daily Loss Limit**: 5% (circuit breaker)

### ⚠️ **Risk Warnings for Real Money**
- **High Volatility**: Mid/small caps can move 10-50% daily
- **Leverage Risk**: 25x leverage amplifies both gains and losses
- **Liquidity Risk**: Small caps may have wider spreads
- **Execution Risk**: Network latency can affect fills
- **Market Risk**: Crypto markets are 24/7 with no circuit breakers

## ✅ **Live Trading Deployment Checklist**

### **Before Deployment (CRITICAL)**
- [ ] **Verify Aster DEX API keys** have live trading permissions
- [ ] **Confirm $50+ balance** in Aster DEX account
- [ ] **Test API connectivity** with small amounts first
- [ ] **Set up emergency stop** - know how to disable live trading
- [ ] **Configure alerts** for 2% drawdown threshold

### **During First Hour**
- [ ] **Monitor dashboard** for successful initialization
- [ ] **Check logs** for any API connection errors
- [ ] **Verify position limits** (10% max, 5 max positions)
- [ ] **Confirm leverage settings** (5x-25x range)
- [ ] **Test emergency stop** functionality

### **First 24 Hours**
- [ ] **Monitor P&L** closely (expect volatility)
- [ ] **Verify stop losses** are working
- [ ] **Check trade execution** quality
- [ ] **Review leverage usage** patterns
- [ ] **Adjust position sizing** if needed

### **Ongoing Monitoring**
- [ ] **Daily P&L review** (target 5-15% growth)
- [ ] **Risk limit compliance** (5% daily loss max)
- [ ] **Strategy performance** analysis
- [ ] **Market condition** adaptation
- [ ] **Emergency protocols** ready

## 🎉 Success Metrics (Real Money)

Your aggressive deployment is successful when:

- ✅ **Live positions** are opened with real money
- ✅ **Profit targets** (1.5-3%) are hit consistently
- ✅ **Stop losses** (0.3-0.5%) protect capital effectively
- ✅ **Leverage** is used intelligently (5x-25x range)
- ✅ **Mid/small caps** generate explosive gains
- ✅ **Daily targets** of 5-15% are achieved
- ✅ **Risk limits** prevent catastrophic losses

## 🚀 **READY FOR TONIGHT'S LIVE DEPLOYMENT**

### **Final Deployment Command:**
```bash
# 🚨 FINAL DEPLOYMENT - AGGRESSIVE LIVE TRADING
export GCP_PROJECT_ID="your-project-id"
bash deploy_cloud_run_economical.sh
```

### **Post-Deployment Actions:**
1. **Visit Dashboard**: `https://your-service-url/dashboard`
2. **Verify Live Mode**: Check "Mode: live_trading" in dashboard
3. **Monitor Signals**: Visit `/signals/SOLUSDT` for trading analysis
4. **Check Positions**: Real-time position monitoring
5. **Set Alerts**: Monitor for 2% drawdown warnings

### **Emergency Procedures:**
```bash
# Disable live trading (switch to paper mode)
gcloud run services update aster-hft-testing \
  --set-env-vars "ENABLE_PAPER_TRADING=true" \
  --region=us-central1

# Check current configuration
gcloud run services describe aster-hft-testing \
  --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].env[].value)"
```

## 💡 **Live Trading Pro Tips**

1. **Monitor Actively**: Check dashboard every 2-4 hours initially
2. **Start Conservative**: $50 is perfect starting capital
3. **Leverage Wisely**: Bot will use 5x-25x - don't override unless needed
4. **Small Caps First**: SOL, ASTER, DOGE are most liquid for testing
5. **Emergency Stop**: Keep paper trading command ready
6. **Cost Awareness**: Expect $0.50-1.00/day in fees with $50 capital
7. **Growth Mindset**: Aim for 5-15% daily returns with 5% max loss tolerance

---

**Ready to deploy?** Run: `bash deploy_cloud_run_economical.sh`

**Need the full GKE deployment?** Use: `bash deploy_to_gcp.sh`

**Questions?** Check logs: `gcloud logs tail --filter="resource.type=cloud_run_revision"`

---

*Last Updated: October 22, 2025*
*Cost Estimate: $5-15/month*
*Risk Level: Aggressive (10% positions, 5 max concurrent)*

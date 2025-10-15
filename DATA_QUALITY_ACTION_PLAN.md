# 🎯 Data Quality Action Plan - Crypto Data Streams

## 📊 **Current Status Analysis**

### ❌ **Critical Issues Identified:**

1. **Single Source Dependency** - CRITICAL
   - **Problem:** 97% of assets (97/100) rely solely on CryptoCompare
   - **Risk:** Single point of failure, no redundancy
   - **Impact:** Data outages will halt trading
   
2. **Missing Assets** - HIGH PRIORITY
   - **KLAY:** 0 data points (no sources available)
   - **IOTA:** 0 data points (no sources available)
   - **FXS:** 0 data points (no sources available)
   - **Failure Rate:** 3% complete failures

3. **No Quality Validation** - HIGH PRIORITY
   - **Problem:** No quality scores in summary
   - **Risk:** Undetected bad data feeding into models
   - **Impact:** Poor model accuracy, bad trades

4. **Unused Data Sources** - MEDIUM PRIORITY
   - **Aster DEX:** 0 assets using (configured but not working)
   - **Binance:** 0 assets using (fallback not triggered)
   - **CoinGecko:** 0 assets using (fallback not triggered)
   - **Yahoo Finance:** 0 assets using

5. **Limited Historical Depth** - MEDIUM PRIORITY
   - **Current:** 2001 data points (83 days @ hourly)
   - **Needed:** 6-24 months for robust training
   - **Gap:** ~180-700 days missing

---

## 🎯 **Action Plan - Priority Order**

### **Phase 1: Immediate Fixes** (1-2 hours)

#### Step 1.1: Fix Missing Assets
```bash
python scripts/improve_data_quality.py
```

**What it does:**
- Recollects KLAY, IOTA, FXS with multi-source approach
- Tries Binance, CoinGecko, Yahoo Finance as alternatives
- Validates data quality for all assets

#### Step 1.2: Add Source Diversity
```bash
python scripts/collect_multi_source_crypto.py --diversify
```

**What it does:**
- Adds Binance as backup for top 20 assets
- Adds CoinGecko for remaining assets
- Validates both sources match

#### Step 1.3: Validate All Data
```bash
python scripts/validate_collected_data.py
```

**What it does:**
- Calculates quality scores (0-1) for each asset
- Checks OHLCV integrity
- Detects outliers and anomalies
- Generates quality report

---

### **Phase 2: Source Redundancy** (2-3 hours)

#### Step 2.1: Enable Aster DEX Integration
```python
# Fix Aster DEX collection
python scripts/collect_aster_data_sync.py --symbols BTC,ETH,SOL,ADA,DOT
```

**Expected Result:**
- 5-10 assets collected from Aster DEX
- Primary source for Aster-native trading
- Live order book data

#### Step 2.2: Add Binance Fallback
```python
# Add Binance for all major assets
python scripts/add_binance_backup.py
```

**Creates:**
- Binance backup for top 50 assets
- Real-time order book access
- Higher update frequency (1m candles)

#### Step 2.3: Configure CoinGecko Free Tier
```python
# Maximize free tier usage
python scripts/optimize_coingecko.py
```

**Implements:**
- Smart rate limiting (30 calls/min)
- Asset prioritization
- Cache optimization

---

### **Phase 3: Data Quality Monitoring** (1-2 hours)

#### Step 3.1: Implement Quality Metrics
```python
# Add quality tracking
python scripts/monitor_data_quality.py --continuous
```

**Monitors:**
- Data completeness (missing bars)
- OHLCV integrity (high >= low, etc.)
- Outlier detection (extreme price moves)
- Source health (API status)
- Update frequency (staleness)

#### Step 3.2: Set Up Alerts
```python
# Configure quality alerts
python scripts/setup_quality_alerts.py
```

**Alerts on:**
- Data gaps > 1 hour
- Quality score < 0.7
- Source failures
- Extreme outliers

#### Step 3.3: Build Quality Dashboard
```python
# Real-time quality monitoring
streamlit run dashboard/data_quality_dashboard.py
```

**Shows:**
- Real-time quality scores
- Source availability
- Data freshness
- Alert history

---

### **Phase 4: Historical Depth** (3-6 hours)

#### Step 4.1: Extend Historical Data
```python
# Collect 6 months of history
python scripts/extend_historical_data.py --months 6
```

**Downloads:**
- 6 months of hourly data (4320 points)
- For all 100 assets
- From multiple sources

#### Step 4.2: Validate Historical Quality
```python
# Check historical data integrity
python scripts/validate_historical.py
```

**Verifies:**
- No gaps in historical data
- Consistent OHLCV across sources
- Reasonable price continuity

---

## 📈 **Expected Improvements**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Source Diversity** | 1.0 avg | 2.5 avg | +150% |
| **Missing Assets** | 3 (3%) | 0 (0%) | -100% |
| **Quality Score** | Unknown | 85%+ | N/A |
| **Data Points** | 2,001 | 4,320+ | +116% |
| **Source Coverage** | 1/5 (20%) | 4/5 (80%) | +300% |
| **Update Frequency** | 1h | 1m-1h | +6000% |
| **Redundancy** | 0% | 90%+ | ∞ |

---

## 🚀 **Quick Start Guide**

### **Option 1: Automated Fix (Recommended)**

Run the master improvement script:

```bash
python scripts/improve_data_quality.py --full
```

This will:
1. Fix missing assets (KLAY, IOTA, FXS)
2. Add source diversity (Binance + CoinGecko)
3. Validate all data quality
4. Extend historical depth
5. Generate comprehensive report

**Time:** 2-4 hours (mostly automated)

### **Option 2: Manual Step-by-Step**

Follow the phases above in order:

```bash
# Phase 1
python scripts/improve_data_quality.py
python scripts/validate_collected_data.py

# Phase 2
python scripts/collect_aster_data_sync.py
python scripts/add_binance_backup.py

# Phase 3
python scripts/monitor_data_quality.py
streamlit run dashboard/data_quality_dashboard.py

# Phase 4
python scripts/extend_historical_data.py
```

**Time:** 6-10 hours (hands-on)

---

## 📊 **Quality Metrics Defined**

### **Overall Quality Score** (0-1)

Calculated as weighted average:

```python
quality_score = (
    completeness * 0.30 +      # No missing bars
    integrity * 0.25 +          # Valid OHLCV relationships
    consistency * 0.20 +        # Matches across sources
    freshness * 0.15 +          # Recent updates
    outlier_filter * 0.10       # No extreme anomalies
)
```

### **Quality Grades**

- **A (0.9-1.0):** Excellent - Production ready
- **B (0.8-0.9):** Good - Minor issues, usable
- **C (0.7-0.8):** Fair - Some concerns, monitor
- **D (0.6-0.7):** Poor - Needs improvement
- **F (<0.6):** Failing - Do not use

### **Target Quality Levels**

- **Training Data:** Grade A (0.9+) required
- **Backtesting:** Grade B (0.8+) acceptable
- **Live Trading:** Grade A (0.9+) required
- **Research:** Grade C (0.7+) acceptable

---

## 🔧 **Technical Implementation**

### **Multi-Source Data Consolidation**

```python
def consolidate_sources(sources: Dict) -> pd.DataFrame:
    """
    Consolidate data from multiple sources with priority:
    1. Aster DEX (if available, most recent)
    2. Binance (high volume, reliable)
    3. CryptoCompare (broad coverage)
    4. CoinGecko (fallback)
    5. Yahoo Finance (last resort)
    """
    priority = ['aster', 'binance', 'cryptocompare', 'coingecko', 'yahoo']
    
    consolidated = None
    for source in priority:
        if source in sources and sources[source] is not None:
            if consolidated is None:
                consolidated = sources[source].copy()
            else:
                # Fill gaps with lower priority sources
                consolidated = consolidated.combine_first(sources[source])
    
    return consolidated
```

### **Quality Validation**

```python
def validate_ohlcv(df: pd.DataFrame) -> Dict:
    """
    Validate OHLCV data integrity:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - Volume >= 0
    - Prices > 0
    - No extreme jumps (>50% in 1 bar)
    """
    issues = {}
    
    # Check OHLC relationships
    issues['invalid_high'] = (df['high'] < df['low']).sum()
    issues['invalid_close'] = ((df['close'] > df['high']) | 
                               (df['close'] < df['low'])).sum()
    issues['negative_price'] = (df[['open','high','low','close']] <= 0).any(axis=1).sum()
    issues['negative_volume'] = (df['volume'] < 0).sum()
    
    # Check for extreme outliers
    returns = df['close'].pct_change().abs()
    issues['extreme_moves'] = (returns > 0.5).sum()
    
    return issues
```

---

## 📝 **Next Steps**

1. **Run immediate fixes:**
   ```bash
   python scripts/improve_data_quality.py
   ```

2. **Review quality report:**
   ```bash
   cat data/historical/ultimate_dataset/crypto/data_quality_report.json
   ```

3. **Fix identified issues:**
   - Address any assets with quality < 0.7
   - Recollect failed assets
   - Add missing sources

4. **Monitor ongoing quality:**
   ```bash
   streamlit run dashboard/data_quality_dashboard.py
   ```

5. **Integrate with training:**
   - Update training scripts to check quality scores
   - Filter out low-quality data
   - Prefer high-quality sources

---

## 🎯 **Success Criteria**

- ✅ **All 100 assets:** Quality score >= 0.8
- ✅ **Source diversity:** 90%+ assets have 2+ sources
- ✅ **No missing data:** 0 assets with 0 data points
- ✅ **Historical depth:** 6+ months for all assets
- ✅ **Update frequency:** <5min staleness for top 20
- ✅ **Validation:** Automated quality checks running
- ✅ **Monitoring:** Real-time dashboard operational

---

**Ready to improve data quality?** Start with:

```bash
python scripts/improve_data_quality.py
```

This will automatically analyze and fix the major issues! 🚀



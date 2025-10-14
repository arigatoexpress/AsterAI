# MCP Trader - Data Pipeline Architecture

## Overview

A production-grade, cost-optimized data pipeline for crypto market data on Aster/Jewster trading pairs.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  Binance Futures API  │  OKX Swap API  │  Aster DEX Events      │
└───────────┬─────────────────────┬─────────────────┬─────────────┘
            │                     │                 │
            ▼                     ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│ │ Cloud Function   │  │ Cloud Function   │  │  WebSocket     │ │
│ │ (Batch Hourly)   │  │ (Backfill Jobs)  │  │  (Streaming)   │ │
│ └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘ │
└──────────┼────────────────────┼────────────────────┼───────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                      BigQuery Dataset: market_data               │
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Table: ohlcv                                                  ││
│ │ - Partitioned by: DATE(timestamp)                            ││
│ │ - Clustered by: symbol, venue                                ││
│ │ - Retention: 2 years                                         ││
│ └──────────────────────────────────────────────────────────────┘│
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Table: funding_rates                                         ││
│ │ - Partitioned by: DATE(timestamp)                            ││
│ │ - Clustered by: symbol, venue                                ││
│ └──────────────────────────────────────────────────────────────┘│
│ ┌──────────────────────────────────────────────────────────────┐│
│ │ Materialized View: daily_ohlcv (aggregated)                  ││
│ │ Materialized View: hourly_stats (metrics)                    ││
│ └──────────────────────────────────────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORMATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│ ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│ │ Feature Eng.     │  │ Strategy Metrics │  │  Risk Calcs    │ │
│ │ (dbt models)     │  │ (SQL views)      │  │  (Python UDFs) │ │
│ └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘ │
└──────────┼────────────────────┼────────────────────┼───────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONSUMPTION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  Backtesting Engine  │  Live Trading    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Schema Design

### Table 1: `ohlcv` (Raw Market Data)

```sql
CREATE TABLE market_data.ohlcv (
  timestamp TIMESTAMP NOT NULL,
  symbol STRING NOT NULL,
  venue STRING NOT NULL,
  interval STRING NOT NULL,
  open FLOAT64 NOT NULL,
  high FLOAT64 NOT NULL,
  low FLOAT64 NOT NULL,
  close FLOAT64 NOT NULL,
  volume FLOAT64 NOT NULL,
  quote_volume FLOAT64,
  num_trades INT64,
  taker_buy_volume FLOAT64,
  taker_buy_quote_volume FLOAT64,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY symbol, venue
OPTIONS(
  partition_expiration_days=730,  -- 2 years retention
  description="Raw OHLCV data from Binance/OKX for Aster/Jewster pairs"
);
```

**Why this design**:
- `timestamp`: Primary temporal dimension
- `symbol`: Aster/Jewster standardized symbol (e.g., BTCUSDT)
- `venue`: Data source (binance, okx) for multi-sourcing
- `interval`: 1h, 4h, 1d for different timeframes
- `ingestion_time`: Audit trail for data quality

### Table 2: `funding_rates`

```sql
CREATE TABLE market_data.funding_rates (
  timestamp TIMESTAMP NOT NULL,
  symbol STRING NOT NULL,
  venue STRING NOT NULL,
  funding_rate FLOAT64 NOT NULL,
  predicted_rate FLOAT64,
  ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(timestamp)
CLUSTER BY symbol, venue;
```

### Table 3: `trade_executions` (Future: Live Trading)

```sql
CREATE TABLE market_data.trade_executions (
  execution_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  symbol STRING NOT NULL,
  venue STRING NOT NULL,
  side STRING NOT NULL,  -- BUY/SELL
  price FLOAT64 NOT NULL,
  quantity FLOAT64 NOT NULL,
  fee FLOAT64,
  strategy_id STRING,
  pnl FLOAT64,
  metadata JSON
)
PARTITION BY DATE(timestamp)
CLUSTER BY symbol, strategy_id;
```

---

## Materialized Views (Performance Optimization)

### View 1: Daily OHLCV Aggregation

```sql
CREATE MATERIALIZED VIEW market_data.daily_ohlcv AS
SELECT 
  DATE(timestamp) as date,
  symbol,
  venue,
  FIRST_VALUE(open ORDER BY timestamp ASC) as open,
  MAX(high) as high,
  MIN(low) as low,
  LAST_VALUE(close ORDER BY timestamp ASC) as close,
  SUM(volume) as volume,
  SUM(quote_volume) as quote_volume,
  COUNT(*) as candle_count
FROM market_data.ohlcv
WHERE interval = '1h'
GROUP BY date, symbol, venue;
```

**Benefit**: Daily backtests query this view instead of raw hourly data.
**Cost savings**: 24x reduction in data scanned.

### View 2: Real-time Statistics

```sql
CREATE MATERIALIZED VIEW market_data.latest_stats AS
SELECT 
  symbol,
  venue,
  MAX(timestamp) as latest_timestamp,
  LAST_VALUE(close ORDER BY timestamp ASC) as latest_price,
  AVG(volume) OVER (
    PARTITION BY symbol, venue 
    ORDER BY timestamp 
    ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
  ) as avg_volume_24h
FROM market_data.ohlcv
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
GROUP BY symbol, venue, timestamp, close, volume;
```

**Benefit**: Dashboard can query latest price without scanning full table.

---

## Data Pipeline Flows

### Flow 1: Hourly Batch Ingestion

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Cloud Scheduler triggers Cloud Function at :00           │
├──────────────────────────────────────────────────────────────┤
│ 2. Function queries BigQuery for latest timestamp per symbol │
├──────────────────────────────────────────────────────────────┤
│ 3. Calculates delta window (last_ts → now)                  │
├──────────────────────────────────────────────────────────────┤
│ 4. Fetches data from Binance/OKX (parallel requests)        │
├──────────────────────────────────────────────────────────────┤
│ 5. Validates data quality (schema, nulls, duplicates)       │
├──────────────────────────────────────────────────────────────┤
│ 6. Inserts to BigQuery (idempotent WRITE_APPEND)            │
├──────────────────────────────────────────────────────────────┤
│ 7. Logs metrics to Cloud Logging                            │
└──────────────────────────────────────────────────────────────┘
```

### Flow 2: Backfill Historical Data

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Manual trigger: python scripts/backfill.py START END     │
├──────────────────────────────────────────────────────────────┤
│ 2. Chunks date range into daily windows                     │
├──────────────────────────────────────────────────────────────┤
│ 3. For each window:                                          │
│    - Check if data exists in BigQuery                       │
│    - If missing, fetch from API                             │
│    - Insert with backfill metadata flag                     │
├──────────────────────────────────────────────────────────────┤
│ 4. Resume on failure (checkpointing)                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Quality Framework

### 1. Schema Validation

```python
from pydantic import BaseModel, validator

class OHLCVRow(BaseModel):
    timestamp: datetime
    symbol: str
    venue: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @validator('high')
    def high_gte_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        return v
    
    @validator('volume')
    def volume_positive(cls, v):
        if v < 0:
            raise ValueError('volume must be >= 0')
        return v
```

### 2. Freshness Checks

```sql
-- Alert if no data ingested in last 2 hours
SELECT symbol, MAX(ingestion_time) as last_ingestion
FROM market_data.ohlcv
GROUP BY symbol
HAVING TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(ingestion_time), HOUR) > 2
```

### 3. Completeness Checks

```sql
-- Detect missing hourly candles
WITH expected AS (
  SELECT symbol, GENERATE_TIMESTAMP_ARRAY(
    '2024-01-01', 
    CURRENT_TIMESTAMP(), 
    INTERVAL 1 HOUR
  ) as expected_timestamps
  FROM (SELECT DISTINCT symbol FROM market_data.ohlcv)
),
actual AS (
  SELECT symbol, timestamp
  FROM market_data.ohlcv
  WHERE interval = '1h'
)
SELECT e.symbol, et as missing_timestamp
FROM expected e, UNNEST(expected_timestamps) et
LEFT JOIN actual a ON e.symbol = a.symbol AND et = a.timestamp
WHERE a.timestamp IS NULL
```

---

## Cost Optimization

### Storage Costs (Projected)

| Data Type | Rows/Day | Size/Row | Daily Storage | Monthly Cost |
|-----------|----------|----------|---------------|--------------|
| OHLCV (5 symbols × 24h) | 120 | 200 bytes | 24 KB | $0.00 |
| OHLCV (30 days) | 3,600 | 200 bytes | 720 KB | $0.00 |
| OHLCV (1 year) | 43,800 | 200 bytes | 8.76 MB | $0.00 |
| OHLCV (2 years) | 87,600 | 200 bytes | 17.52 MB | $0.00 |

**Total storage cost**: < $1/month

### Query Costs (Projected)

| Query Type | Frequency | Data Scanned | Monthly Cost |
|------------|-----------|--------------|--------------|
| Dashboard load | 100/day | 1 MB/query | $0.00 |
| Backtest (30 days) | 10/day | 100 MB/query | $0.05 |
| Feature extraction | 1/hour | 10 MB/query | $0.04 |

**Total query cost**: ~$0.10/month

**Total BigQuery cost**: **~$1/month**

---

## Monitoring & Alerting

### Key Metrics

1. **Data Freshness**: Time since last ingestion per symbol
2. **Completeness**: % of expected hourly candles present
3. **API Latency**: P50/P95/P99 response times
4. **Error Rate**: Failed ingestions / total attempts
5. **Cost**: Daily spend on BigQuery

### Alert Thresholds

```python
ALERTS = {
    'data_freshness': {'threshold': 120, 'unit': 'minutes'},
    'completeness': {'threshold': 95, 'unit': 'percent'},
    'error_rate': {'threshold': 5, 'unit': 'percent'},
    'daily_cost': {'threshold': 1.0, 'unit': 'dollars'}
}
```

---

## Future Enhancements

### Phase 1 (Current): Batch Hourly
- ✅ Cloud Functions + Cloud Scheduler
- ✅ BigQuery partitioned tables
- ✅ Priority symbols (BTC, ETH, UNI, LINK, AAVE)

### Phase 2: Real-time Streaming
- [ ] WebSocket connectors for sub-second data
- [ ] Pub/Sub message queue
- [ ] Dataflow streaming pipeline
- [ ] Low-latency serving layer (Redis/Bigtable)

### Phase 3: Advanced Analytics
- [ ] dbt models for feature engineering
- [ ] Airflow DAGs for complex workflows
- [ ] ML feature store integration
- [ ] Real-time anomaly detection

### Phase 4: Multi-Region
- [ ] Regional BigQuery datasets (EU, ASIA)
- [ ] CDN for dashboard static assets
- [ ] Geo-routed API endpoints

---

## Best Practices Checklist

- [x] **Idempotency**: Duplicate runs don't corrupt data
- [x] **Partitioning**: Tables partitioned by date
- [x] **Clustering**: High-cardinality columns clustered
- [x] **Schema Evolution**: Nullable new columns, versioned tables
- [x] **Data Quality**: Validation at ingestion
- [ ] **Testing**: Unit tests for transformations
- [ ] **Documentation**: dbt docs for lineage
- [ ] **Monitoring**: Alerts on freshness/completeness
- [ ] **Cost Controls**: Query budgets, partition expiration
- [ ] **Disaster Recovery**: Automated backups, failover

---

## References

- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [Time-Series Data in BigQuery](https://cloud.google.com/blog/topics/developers-practitioners/bigquery-explained-storage-ingestion)
- [Lambda Architecture](http://lambda-architecture.net/)
- [Data Engineering Cookbook](https://github.com/andkret/Cookbook)

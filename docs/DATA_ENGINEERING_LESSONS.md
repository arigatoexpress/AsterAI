# Data Engineering Lessons - MCP Trader

A comprehensive guide to the data engineering concepts and best practices implemented in this project.

---

## 📖 Table of Contents

1. [Foundation Concepts](#foundation-concepts)
2. [Architecture Patterns](#architecture-patterns)
3. [Data Modeling](#data-modeling)
4. [Quality & Monitoring](#quality--monitoring)
5. [Cost Optimization](#cost-optimization)
6. [Hands-On Exercises](#hands-on-exercises)

---

## Foundation Concepts

### What is Data Engineering?

Data engineering is the practice of designing and building systems for:
- **Collecting** data from various sources
- **Storing** data efficiently
- **Processing** data at scale
- **Serving** data to consumers (analysts, ML models, dashboards)

**Your trading system** is a real-world data engineering project!

### The Data Lifecycle

```
Source → Ingest → Store → Transform → Serve → Archive
```

**Example flow**:
1. **Source**: Binance API has price data
2. **Ingest**: Cloud Function fetches it hourly
3. **Store**: BigQuery stores it in partitioned tables
4. **Transform**: Materialized views create daily aggregates
5. **Serve**: Dashboard queries optimized views
6. **Archive**: Old data moves to cold storage after 2 years

---

## Architecture Patterns

### Pattern 1: Lambda Architecture

**Purpose**: Handle both batch (historical) and streaming (real-time) data.

```
         Batch Layer           Streaming Layer
              │                      │
              ▼                      ▼
         Historical Data        Real-time Data
              │                      │
              └──────────┬───────────┘
                         ▼
                   Serving Layer
```

**Your implementation**:
- **Batch**: Cloud Function runs hourly (catches up historical gaps)
- **Streaming**: Future WebSocket connection (real-time ticks)
- **Serving**: BigQuery (unified view of both)

**When to use**: When you need both historical analysis and real-time monitoring.

### Pattern 2: ETL vs. ELT

**ETL (Extract, Transform, Load)**:
```
API → Transform in Cloud Function → Load to BigQuery
```
- Transform data before storing
- Good when: Source data is messy, storage is expensive

**ELT (Extract, Load, Transform)**:
```
API → Load raw to BigQuery → Transform with SQL views
```
- Store raw data, transform later
- Good when: Storage is cheap, flexibility is needed

**Your choice**: ELT (raw OHLCV → BigQuery → materialized views for aggregation)

**Why**: BigQuery storage is cheap (~$0.02/GB/month), and you can create many different transformations without re-ingesting data.

---

## Data Modeling

### Time-Series Schemas

**Bad schema** (wide table):
```sql
CREATE TABLE prices (
  symbol STRING,
  price_2024_01_01_00 FLOAT,
  price_2024_01_01_01 FLOAT,
  price_2024_01_01_02 FLOAT,
  ...  -- 8,760 columns for 1 year!
)
```
Problems: Can't add new timestamps, queries are nightmares.

**Good schema** (normalized time-series):
```sql
CREATE TABLE prices (
  timestamp TIMESTAMP,
  symbol STRING,
  price FLOAT
)
PARTITION BY DATE(timestamp)
CLUSTER BY symbol
```
Benefits: Unlimited timestamps, easy queries, partition pruning.

### Partitioning Deep Dive

**What**: Physically divide table into smaller chunks.

**Why**: Query only relevant chunks, not entire table.

**Example**:
```sql
-- Without partitioning: Scans 365 days = $5
SELECT * FROM prices WHERE symbol = 'BTC'

-- With partitioning: Scans 1 day = $0.01
SELECT * FROM prices 
WHERE symbol = 'BTC' 
  AND timestamp >= '2024-01-01'
  AND timestamp < '2024-01-02'
```

**Your table**:
```sql
PARTITION BY DATE(timestamp)  -- One partition per day
```

**Cost savings**: 365x reduction when querying single day.

### Clustering Deep Dive

**What**: Sort data within each partition by specified columns.

**Why**: Faster scans when filtering on clustered columns.

**Example**:
```sql
-- Partition: 2024-01-01 (unsorted)
BTCUSDT, binance, 100
ETHUSDT, okx, 200
BTCUSDT, okx, 150
ETHUSDT, binance, 250

-- Partition: 2024-01-01 (clustered by symbol, venue)
BTCUSDT, binance, 100
BTCUSDT, okx, 150
ETHUSDT, binance, 250
ETHUSDT, okx, 200
```

Now `WHERE symbol = 'BTCUSDT'` only scans first 2 rows.

**Your table**:
```sql
CLUSTER BY symbol, venue
```

**Cost savings**: Additional 5-10x reduction on top of partitioning.

---

## Quality & Monitoring

### The Data Quality Pyramid

```
         ┌─────────────────┐
         │   Observability │  (Monitor everything)
         └─────────────────┘
        ┌───────────────────┐
        │   Data Quality    │  (Validate at ingestion)
        └───────────────────┘
       ┌─────────────────────┐
       │  Schema Validation  │  (Pydantic models)
       └─────────────────────┘
      ┌───────────────────────┐
      │   Type Safety         │  (Static typing)
      └───────────────────────┘
```

### Validation Levels

**Level 1: Type safety** (Python type hints)
```python
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

**Level 2: Schema validation** (Pydantic)
```python
class OHLCVRow(BaseModel):
    timestamp: datetime
    open: float = Field(gt=0)  # Must be > 0
    high: float
```

**Level 3: Business logic** (Custom validators)
```python
@validator('high')
def high_gte_low(cls, v, values):
    if v < values['low']:
        raise ValueError('high must be >= low')
    return v
```

**Level 4: Statistical checks** (Outlier detection)
```python
if abs(funding_rate) > 0.1:
    warnings.warn('Extreme funding rate')
```

### Monitoring Metrics

**The Four Golden Signals**:

1. **Latency**: How long does ingestion take?
   ```python
   start = time.time()
   ingest_data()
   duration = time.time() - start
   log_metric('ingestion_latency_seconds', duration)
   ```

2. **Traffic**: How much data are we processing?
   ```python
   log_metric('rows_ingested', len(df))
   log_metric('bytes_ingested', df.memory_usage().sum())
   ```

3. **Errors**: What's failing?
   ```python
   try:
       ingest_data()
   except Exception as e:
       log_metric('ingestion_errors', 1)
       alert(f'Ingestion failed: {e}')
   ```

4. **Saturation**: Are we hitting limits?
   ```python
   api_calls_remaining = response.headers['X-RateLimit-Remaining']
   if api_calls_remaining < 10:
       alert('Approaching API rate limit')
   ```

---

## Cost Optimization

### BigQuery Cost Model

**Storage**: $0.02/GB/month (first 10GB free)
**Queries**: $5/TB scanned (first 1TB/month free)

**Your projected costs**:
- Storage: 5 symbols × 24 hours/day × 365 days × 200 bytes = **8.76 MB/year** → **$0.00**
- Queries: 100 queries/day × 10 MB/query = **30 GB/month** → **$0.00** (under free tier)

**Total**: **~$0-1/month**

### Optimization Techniques

**1. Partition Pruning**
```sql
-- Scans entire table (expensive)
SELECT AVG(close) FROM ohlcv WHERE symbol = 'BTCUSDT'

-- Scans 1 week only (cheap)
SELECT AVG(close) FROM ohlcv 
WHERE symbol = 'BTCUSDT' 
  AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 7 DAY
```

**2. Column Projection**
```sql
-- Scans 10 columns (expensive)
SELECT * FROM ohlcv

-- Scans 2 columns (cheap)
SELECT timestamp, close FROM ohlcv
```

**3. Materialized Views**
```sql
-- Daily backtest: Query daily_ohlcv view
-- Scans 30 rows instead of 720 hourly rows
-- 24x cost reduction
```

**4. Clustering**
```sql
-- Automatically skips irrelevant data blocks
-- Additional 5-10x speedup on filtered queries
```

**Combined savings**: **100-500x cost reduction** vs. unoptimized queries!

---

## Hands-On Exercises

### Exercise 1: Query Optimization

**Task**: Optimize this expensive query.

**Bad query** (scans entire table):
```sql
SELECT symbol, AVG(close) 
FROM market_data.ohlcv
GROUP BY symbol
```

**Good query** (scans last 30 days only):
```sql
SELECT symbol, AVG(close)
FROM market_data.ohlcv
WHERE timestamp >= CURRENT_TIMESTAMP() - INTERVAL 30 DAY
GROUP BY symbol
```

**Best query** (uses materialized view):
```sql
SELECT symbol, AVG(close)
FROM market_data.daily_ohlcv
WHERE date >= CURRENT_DATE() - INTERVAL 30 DAY
GROUP BY symbol
```

**Your turn**: Write a query to get the highest price for BTCUSDT in the last 7 days, using partitioning and column projection.

<details>
<summary>Solution</summary>

```sql
SELECT MAX(high) as highest_price
FROM market_data.ohlcv
WHERE symbol = 'BTCUSDT'
  AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 7 DAY
```
</details>

### Exercise 2: Data Quality Check

**Task**: Write a validator to check if volume is suspiciously low.

```python
from pydantic import BaseModel, validator

class OHLCVRow(BaseModel):
    timestamp: datetime
    symbol: str
    volume: float
    
    @validator('volume')
    def volume_not_suspiciously_low(cls, v, values):
        # Your code here
        # Hint: Check if volume < some threshold
        pass
```

<details>
<summary>Solution</summary>

```python
@validator('volume')
def volume_not_suspiciously_low(cls, v, values):
    MIN_VOLUME = 1000  # Minimum expected volume
    if v < MIN_VOLUME:
        import warnings
        warnings.warn(
            f"Suspiciously low volume for {values.get('symbol')}: {v}"
        )
    return v
```
</details>

### Exercise 3: Monitoring Alert

**Task**: Create an alert if no data ingested in last 2 hours.

```python
def check_data_freshness(client, symbol):
    # Your code here
    # 1. Query latest timestamp from BigQuery
    # 2. Calculate hours since that timestamp
    # 3. Alert if > 2 hours
    pass
```

<details>
<summary>Solution</summary>

```python
def check_data_freshness(client, symbol):
    query = f"""
    SELECT MAX(timestamp) as latest
    FROM market_data.ohlcv
    WHERE symbol = '{symbol}'
    """
    result = client.query(query).to_dataframe()
    latest = result.iloc[0]['latest']
    hours_old = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
    
    if hours_old > 2:
        send_alert(f"No data for {symbol} in {hours_old:.1f} hours!")
```
</details>

---

## Key Takeaways

✅ **Start with raw data** (ELT pattern)
✅ **Partition by time** (day-level for market data)
✅ **Cluster by high-cardinality columns** (symbol, venue)
✅ **Validate at ingestion** (Pydantic schemas)
✅ **Monitor everything** (freshness, completeness, quality)
✅ **Optimize for cost** (partition pruning, column selection, materialized views)
✅ **Make it idempotent** (duplicate runs = same result)
✅ **Document lineage** (where data came from, how it transformed)

---

## Further Learning

**Books**:
- *Designing Data-Intensive Applications* by Martin Kleppmann
- *The Data Warehouse Toolkit* by Ralph Kimball
- *Streaming Systems* by Tyler Akidau

**Courses**:
- [BigQuery Best Practices](https://cloud.google.com/bigquery/docs/best-practices)
- [Data Engineering on GCP](https://www.cloudskillsboost.google/paths/16)

**Your next steps**:
1. Deploy the pipeline to GCP
2. Monitor data quality for 7 days
3. Add WebSocket streaming layer
4. Implement automated alerts
5. Build feature engineering pipelines

---

**Congratulations!** You now understand production-grade data engineering. 🎉

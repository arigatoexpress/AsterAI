-- Materialized views for performance optimization
-- Run this after deploying BigQuery tables

-- View 1: Daily OHLCV Aggregation
-- Reduces query costs for daily backtests by 24x
CREATE OR REPLACE MATERIALIZED VIEW `PROJECT_ID.market_data.daily_ohlcv` AS
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
  COUNT(*) as candle_count,
  MIN(timestamp) as first_timestamp,
  MAX(timestamp) as last_timestamp
FROM `PROJECT_ID.market_data.ohlcv`
WHERE interval = '1h'
GROUP BY date, symbol, venue;

-- View 2: Latest Market Stats
-- Fast lookups for current prices and 24h averages
CREATE OR REPLACE MATERIALIZED VIEW `PROJECT_ID.market_data.latest_stats` AS
WITH recent_data AS (
  SELECT 
    symbol,
    venue,
    timestamp,
    close,
    volume,
    ROW_NUMBER() OVER (PARTITION BY symbol, venue ORDER BY timestamp DESC) as rn
  FROM `PROJECT_ID.market_data.ohlcv`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
)
SELECT 
  symbol,
  venue,
  MAX(CASE WHEN rn = 1 THEN timestamp END) as latest_timestamp,
  MAX(CASE WHEN rn = 1 THEN close END) as latest_price,
  AVG(volume) as avg_volume_24h,
  COUNT(*) as candle_count_24h
FROM recent_data
GROUP BY symbol, venue;

-- View 3: Funding Rate Summary
-- Aggregated funding rates for analysis
CREATE OR REPLACE MATERIALIZED VIEW `PROJECT_ID.market_data.funding_summary` AS
SELECT 
  DATE(timestamp) as date,
  symbol,
  venue,
  AVG(funding_rate) as avg_funding_rate,
  MIN(funding_rate) as min_funding_rate,
  MAX(funding_rate) as max_funding_rate,
  STDDEV(funding_rate) as stddev_funding_rate,
  COUNT(*) as rate_count
FROM `PROJECT_ID.market_data.funding_rates`
GROUP BY date, symbol, venue;

-- View 4: Data Quality Metrics
-- Monitor data completeness and freshness
CREATE OR REPLACE MATERIALIZED VIEW `PROJECT_ID.market_data.quality_metrics` AS
WITH hourly_expected AS (
  SELECT 
    symbol,
    venue,
    GENERATE_TIMESTAMP_ARRAY(
      TIMESTAMP_TRUNC(MIN(timestamp), HOUR),
      TIMESTAMP_TRUNC(MAX(timestamp), HOUR),
      INTERVAL 1 HOUR
    ) as expected_hours
  FROM `PROJECT_ID.market_data.ohlcv`
  WHERE interval = '1h'
  GROUP BY symbol, venue
),
hourly_actual AS (
  SELECT 
    symbol,
    venue,
    TIMESTAMP_TRUNC(timestamp, HOUR) as hour
  FROM `PROJECT_ID.market_data.ohlcv`
  WHERE interval = '1h'
  GROUP BY symbol, venue, hour
)
SELECT 
  e.symbol,
  e.venue,
  ARRAY_LENGTH(e.expected_hours) as expected_candles,
  COUNT(DISTINCT a.hour) as actual_candles,
  ROUND(COUNT(DISTINCT a.hour) / ARRAY_LENGTH(e.expected_hours) * 100, 2) as completeness_pct,
  MAX(a.hour) as latest_data_timestamp,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(a.hour), HOUR) as hours_since_update
FROM hourly_expected e
LEFT JOIN UNNEST(e.expected_hours) as expected_hour
LEFT JOIN hourly_actual a 
  ON e.symbol = a.symbol 
  AND e.venue = a.venue 
  AND expected_hour = a.hour
GROUP BY e.symbol, e.venue, e.expected_hours;

-- Refresh schedule (automatic)
-- Materialized views refresh automatically when base tables change
-- You can also manually refresh: CALL BQ.REFRESH_MATERIALIZED_VIEW('PROJECT_ID.market_data.daily_ohlcv');

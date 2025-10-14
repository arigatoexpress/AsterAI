# MCP Trader - GCP Deployment Guide

Complete deployment guide for running the MCP AI Trading Protocol on Google Cloud Platform with automated data ingestion.

## 🚀 Quick Deploy

```bash
# 1. Set your GCP project
export GCP_PROJECT="your-project-id"

# 2. Deploy everything
./scripts/deploy_gcp.sh $GCP_PROJECT
```

## 📋 Prerequisites

1. **GCP Project** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Python 3.11+** with virtual environment

```bash
# Install gcloud CLI (macOS)
brew install google-cloud-sdk

# Authenticate
gcloud auth login
gcloud auth application-default login
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cloud         │    │   Cloud          │    │   BigQuery      │
│   Scheduler     │───▶│   Functions      │───▶│   (Data Lake)   │
│   (Hourly)      │    │   (Ingestion)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Binance/OKX    │
                       │   (Data Sources) │
                       └──────────────────┘
```

## 📊 Data Sources

- **Binance Futures**: BTCUSDT, ETHUSDT (1h candles + funding)
- **OKX Swap**: BTC-USDT-SWAP, ETH-USDT-SWAP (1H candles + funding)
- **Storage**: BigQuery partitioned tables (daily partitions)
- **Cost**: ~$5-10/month for 1TB of data

## 🔧 Manual Setup (Step by Step)

### 1. Enable APIs
```bash
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Setup BigQuery
```bash
python scripts/setup_gcp.py your-project-id
```

### 3. Deploy Cloud Function
```bash
gcloud functions deploy ingest-market-data \
    --source cloud_functions/ \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 512MB \
    --timeout 540s \
    --set-env-vars GCP_PROJECT=your-project-id,BIGQUERY_DATASET=market_data
```

### 4. Setup Scheduler
```bash
python scripts/setup_scheduler.py your-project-id
```

## 💰 Cost Breakdown

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| BigQuery Storage | 1TB | ~$20 |
| BigQuery Queries | 1M rows | ~$5 |
| Cloud Functions | 744 invocations | ~$0.50 |
| Cloud Scheduler | 744 jobs | ~$0.10 |
| **Total** | | **~$25-30** |

## 📈 Monitoring

- **BigQuery Console**: View data and query performance
- **Cloud Functions**: Monitor execution logs and errors
- **Cloud Scheduler**: Check job status and history
- **Dashboard**: Real-time strategy backtesting

## 🔍 Usage

### Load Data in Dashboard
1. Open http://localhost:8501
2. Select "BigQuery (GCP)" data source
3. Enter your GCP Project ID
4. Choose symbol, venue, and date range
5. Click "Load from BigQuery"

### Query Data Directly
```python
from mcp_trader.data.bigquery_client import BigQueryClient

client = BigQueryClient(project_id="your-project")
df = client.query_ohlcv("BTCUSDT", "binance", "2024-01-01", "2024-12-31")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   gcloud auth application-default login
   ```

2. **Function Deployment Fails**
   ```bash
   gcloud functions logs read ingest-market-data
   ```

3. **No Data in BigQuery**
   ```bash
   python scripts/setup_scheduler.py your-project-id --test
   ```

4. **Scheduler Not Running**
   ```bash
   gcloud scheduler jobs list
   gcloud scheduler jobs run market-data-ingestion
   ```

### Logs
```bash
# Function logs
gcloud functions logs read ingest-market-data --limit 50

# Scheduler logs
gcloud logging read "resource.type=cloud_scheduler_job" --limit 10
```

## 🔄 Data Pipeline

1. **Every hour**: Cloud Scheduler triggers Cloud Function
2. **Function**: Fetches latest data from Binance/OKX APIs
3. **BigQuery**: Appends new data to partitioned tables
4. **Dashboard**: Queries BigQuery for real-time backtesting

## 📝 Environment Variables

```bash
export GCP_PROJECT="your-project-id"
export BIGQUERY_DATASET="market_data"
```

## 🎯 Next Steps

1. **Add more symbols**: Edit `cloud_functions/main.py`
2. **Add more venues**: Implement new data adapters
3. **Real-time trading**: Connect to Aster DEX API
4. **ML features**: Add feature engineering pipeline
5. **Alerts**: Setup monitoring and notifications

## 📚 Resources

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Cloud Functions Documentation](https://cloud.google.com/functions/docs)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
- [GCP Pricing Calculator](https://cloud.google.com/products/calculator)

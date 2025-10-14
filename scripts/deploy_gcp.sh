#!/bin/bash
# Deploy complete GCP market data pipeline

set -e

PROJECT_ID=$1
if [ -z "$PROJECT_ID" ]; then
    echo "Usage: $0 <PROJECT_ID>"
    echo "Example: $0 my-gcp-project-123"
    exit 1
fi

echo "🚀 Deploying MCP Trader to GCP Project: $PROJECT_ID"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📋 Enabling required APIs..."
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Setup BigQuery tables and load initial data
echo "🗄️ Setting up BigQuery..."
python scripts/setup_gcp.py $PROJECT_ID

# Deploy Cloud Function
echo "☁️ Deploying Cloud Function..."
gcloud functions deploy ingest-market-data \
    --source cloud_functions/ \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 512MB \
    --timeout 540s \
    --set-env-vars GCP_PROJECT=$PROJECT_ID,BIGQUERY_DATASET=market_data

# Setup Cloud Scheduler
echo "⏰ Setting up Cloud Scheduler..."
python scripts/setup_scheduler.py $PROJECT_ID

echo "✅ Deployment complete!"
echo ""
echo "🔗 Next steps:"
echo "1. View data in BigQuery: https://console.cloud.google.com/bigquery?project=$PROJECT_ID"
echo "2. Monitor functions: https://console.cloud.google.com/functions?project=$PROJECT_ID"
echo "3. Check scheduler: https://console.cloud.google.com/cloudscheduler?project=$PROJECT_ID"
echo "4. Run dashboard: streamlit run dashboard/app.py"
echo ""
echo "📊 Your market data pipeline is now running automatically every hour!"

#!/bin/bash

# Deploy Autonomous Trading System to Google Cloud Platform
# This script automates the complete deployment process

set -e  # Exit on any error

# Configuration
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_ACCOUNT="aster-trading@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME="aster-trading-data-${PROJECT_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if gcloud is installed
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI is not installed. Please install it first."
    fi
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install it first."
    fi
    
    log "Prerequisites check passed"
}

# Set up project
setup_project() {
    log "Setting up GCP project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    log "Enabling required APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        bigquery.googleapis.com \
        storage.googleapis.com \
        cloudscheduler.googleapis.com \
        monitoring.googleapis.com \
        secretmanager.googleapis.com \
        iam.googleapis.com
    
    log "APIs enabled successfully"
}

# Create service account
create_service_account() {
    log "Creating service account..."
    
    # Create service account
    gcloud iam service-accounts create aster-trading \
        --display-name="Aster Trading Service Account" \
        --description="Service account for autonomous trading system" \
        --project=$PROJECT_ID
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/bigquery.dataEditor"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/bigquery.jobUser"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/storage.objectAdmin"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/monitoring.metricWriter"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor"
    
    log "Service account created and configured"
}

# Create secrets
create_secrets() {
    log "Creating secrets..."
    
    # Create API key secret
    echo -n "your-aster-api-key" | gcloud secrets create aster-api-key \
        --data-file=- \
        --project=$PROJECT_ID
    
    # Create API secret
    echo -n "your-aster-api-secret" | gcloud secrets create aster-api-secret \
        --data-file=- \
        --project=$PROJECT_ID
    
    # Create database credentials
    echo -n "your-database-url" | gcloud secrets create database-url \
        --data-file=- \
        --project=$PROJECT_ID
    
    log "Secrets created successfully"
}

# Create storage bucket
create_storage() {
    log "Creating Cloud Storage bucket..."
    
    # Create bucket
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME
    
    # Set bucket permissions
    gsutil iam ch serviceAccount:${SERVICE_ACCOUNT}:objectAdmin gs://$BUCKET_NAME
    
    log "Storage bucket created: gs://$BUCKET_NAME"
}

# Setup BigQuery
setup_bigquery() {
    log "Setting up BigQuery..."
    
    # Create dataset
    bq mk --dataset --location=US $PROJECT_ID:aster_trading
    
    # Create market data table
    bq mk --table \
        --schema="timestamp:TIMESTAMP,symbol:STRING,open:FLOAT,high:FLOAT,low:FLOAT,close:FLOAT,volume:FLOAT,quote_volume:FLOAT,trades:INTEGER" \
        --time_partitioning_field=timestamp \
        --clustering_fields=symbol \
        $PROJECT_ID:aster_trading.market_data
    
    # Create features table
    bq mk --table \
        --schema="timestamp:TIMESTAMP,symbol:STRING,feature_name:STRING,feature_value:FLOAT" \
        --time_partitioning_field=timestamp \
        --clustering_fields=symbol,feature_name \
        $PROJECT_ID:aster_trading.features
    
    # Create trades table
    bq mk --table \
        --schema="timestamp:TIMESTAMP,symbol:STRING,side:STRING,size:FLOAT,price:FLOAT,pnl:FLOAT,strategy:STRING" \
        --time_partitioning_field=timestamp \
        --clustering_fields=symbol,strategy \
        $PROJECT_ID:aster_trading.trades
    
    # Create performance table
    bq mk --table \
        --schema="timestamp:TIMESTAMP,strategy:STRING,total_pnl:FLOAT,daily_pnl:FLOAT,win_rate:FLOAT,sharpe_ratio:FLOAT,max_drawdown:FLOAT" \
        --time_partitioning_field=timestamp \
        --clustering_fields=strategy \
        $PROJECT_ID:aster_trading.performance
    
    log "BigQuery setup completed"
}

# Build and deploy services
deploy_services() {
    log "Building and deploying services..."
    
    # Submit build
    gcloud builds submit --config cloudbuild_autonomous.yaml .
    
    log "Services deployed successfully"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create notification channel
    gcloud alpha monitoring channels create \
        --display-name="Aster Trading Alerts" \
        --type=email \
        --channel-labels=email_address=your-email@example.com \
        --project=$PROJECT_ID
    
    # Create alerting policies
    cat > monitoring-policy.yaml << EOF
displayName: "Daily Loss Limit Alert"
conditions:
  - displayName: "Daily P&L below threshold"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="aster-trading-agent"'
      comparison: COMPARISON_LT
      thresholdValue: -10.0
      duration: 300s
notificationChannels:
  - projects/$PROJECT_ID/notificationChannels/$(gcloud alpha monitoring channels list --filter="displayName='Aster Trading Alerts'" --format="value(name)")
EOF
    
    gcloud alpha monitoring policies create --policy-from-file=monitoring-policy.yaml --project=$PROJECT_ID
    
    log "Monitoring setup completed"
}

# Setup Cloud Scheduler
setup_scheduler() {
    log "Setting up Cloud Scheduler jobs..."
    
    # Get service URLs
    TRADING_AGENT_URL=$(gcloud run services describe aster-trading-agent --region=$REGION --format='value(status.url)')
    DATA_COLLECTOR_URL=$(gcloud run services describe aster-data-collector --region=$REGION --format='value(status.url)')
    
    # Create data collection job (every 5 minutes)
    gcloud scheduler jobs create http aster-data-collection \
        --schedule="*/5 * * * *" \
        --uri="${DATA_COLLECTOR_URL}/collect" \
        --http-method=POST \
        --time-zone="UTC" \
        --max-retry-attempts=3 \
        --max-retry-duration=300s \
        --project=$PROJECT_ID
    
    # Create model retraining job (weekly)
    gcloud scheduler jobs create http aster-model-retraining \
        --schedule="0 2 * * 0" \
        --uri="${TRADING_AGENT_URL}/retrain" \
        --http-method=POST \
        --time-zone="UTC" \
        --max-retry-attempts=3 \
        --max-retry-duration=1800s \
        --project=$PROJECT_ID
    
    log "Cloud Scheduler jobs created"
}

# Test deployment
test_deployment() {
    log "Testing deployment..."
    
    # Get service URLs
    TRADING_AGENT_URL=$(gcloud run services describe aster-trading-agent --region=$REGION --format='value(status.url)')
    DASHBOARD_URL=$(gcloud run services describe aster-dashboard --region=$REGION --format='value(status.url)')
    
    # Test trading agent health
    if curl -f -s "${TRADING_AGENT_URL}/health" > /dev/null; then
        log "Trading agent is healthy"
    else
        warning "Trading agent health check failed"
    fi
    
    # Test dashboard
    if curl -f -s "${DASHBOARD_URL}/" > /dev/null; then
        log "Dashboard is accessible"
    else
        warning "Dashboard health check failed"
    fi
    
    log "Deployment test completed"
}

# Display deployment summary
display_summary() {
    log "Deployment Summary"
    echo "=================="
    echo "Project ID: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Service Account: $SERVICE_ACCOUNT"
    echo "Storage Bucket: gs://$BUCKET_NAME"
    echo ""
    echo "Services:"
    echo "- Trading Agent: $(gcloud run services describe aster-trading-agent --region=$REGION --format='value(status.url)')"
    echo "- Data Collector: $(gcloud run services describe aster-data-collector --region=$REGION --format='value(status.url)')"
    echo "- Dashboard: $(gcloud run services describe aster-dashboard --region=$REGION --format='value(status.url)')"
    echo ""
    echo "BigQuery Dataset: $PROJECT_ID:aster_trading"
    echo "Monitoring: https://console.cloud.google.com/monitoring"
    echo ""
    echo "Next Steps:"
    echo "1. Update API keys in Secret Manager"
    echo "2. Configure notification channels"
    echo "3. Start trading with $100 capital"
    echo "4. Monitor performance via dashboard"
}

# Main deployment function
main() {
    log "Starting Autonomous Trading System deployment..."
    
    check_prerequisites
    setup_project
    create_service_account
    create_secrets
    create_storage
    setup_bigquery
    deploy_services
    setup_monitoring
    setup_scheduler
    test_deployment
    display_summary
    
    log "Deployment completed successfully! 🚀"
}

# Run main function
main "$@"

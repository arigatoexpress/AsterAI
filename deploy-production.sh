#!/bin/bash
# Production Deployment Script for Aster Trading Dashboard
# This script deploys the application to Google Cloud Run with production settings

set -e

echo "🚀 Deploying Aster Trading Dashboard to Production..."
echo "=================================================="

# Configuration
PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="aster-trading-dashboard"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Validate required variables
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please set PROJECT_ID environment variable"
    echo "   export PROJECT_ID=your-actual-project-id"
    exit 1
fi

echo "📋 Configuration:"
echo "   Project ID: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service: ${SERVICE_NAME}"
echo "   Image: ${IMAGE_NAME}"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    echo "❌ Not authenticated with GCP. Please run 'gcloud auth login'"
    exit 1
fi

# Set project
echo "🔧 Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com

# Build and push Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run with production settings
echo "🚀 Deploying to Cloud Run with production configuration..."

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 1 \
    --timeout 300 \
    --concurrency 80 \
    --execution-environment gen2 \
    --cpu-throttling \
    --set-env-vars ENVIRONMENT=production \
    --set-env-vars GCP_PROJECT=${PROJECT_ID} \
    --set-env-vars BIGQUERY_DATASET=market_data \
    --set-env-vars ENABLE_CACHING=true \
    --set-env-vars CACHE_TTL=300 \
    --set-env-vars LOG_LEVEL=INFO \
    --set-env-vars LOG_FORMAT=json \
    --set-env-vars HEALTH_CHECK_ENABLED=true \
    --set-env-vars METRICS_ENABLED=true \
    --set-env-vars DEFAULT_STRATEGY=SMA_CROSSOVER \
    --set-env-vars MAX_POSITION_SIZE=0.25 \
    --set-env-vars DEFAULT_FEE_BPS=5 \
    --set-env-vars DEFAULT_DATA_SOURCE=SYNTHETIC \
    --set-env-vars ENABLE_BIGQUERY=true \
    --set-env-vars ENABLE_FILE_UPLOAD=true

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "✅ Production Deployment Complete!"
echo "================================="
echo "🌐 Dashboard URL: ${SERVICE_URL}"
echo ""
echo "📊 Health Check Endpoints:"
echo "   Health: ${SERVICE_URL}?endpoint=health"
echo "   Status: ${SERVICE_URL}?endpoint=status"
echo "   Metrics: ${SERVICE_URL}?endpoint=metrics"
echo ""
echo "🔧 Production Features Enabled:"
echo "   ✅ Production environment mode"
echo "   ✅ Enhanced logging (JSON format)"
echo "   ✅ Performance monitoring"
echo "   ✅ Health check endpoints"
echo "   ✅ Caching enabled (5min TTL)"
echo "   ✅ Resource limits configured"
echo "   ✅ Auto-scaling (1-10 instances)"
echo ""
echo "📖 Next Steps:"
echo "   1. Test the dashboard: ${SERVICE_URL}"
echo "   2. Check health status: ${SERVICE_URL}?endpoint=health"
echo "   3. Monitor logs: gcloud logging read 'resource.type=cloud_run_revision'"
echo "   4. Set up monitoring alerts in Cloud Console"
echo ""
echo "⚠️  Security Recommendations:"
echo "   • Configure authentication for production use"
echo "   • Set up proper CORS origins"
echo "   • Enable Cloud Armor for DDoS protection"
echo "   • Set up monitoring and alerting"

#!/bin/bash
# Deploy Aster AI Trading Dashboard to Google Cloud Platform

set -e

echo "🚀 Deploying Aster AI Trading Dashboard to GCP..."
echo "==============================================="

# Configuration
PROJECT_ID="${PROJECT_ID:-quant-ai-trader-credits}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="aster-trading-dashboard"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

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

# Build and push Docker image
echo "🏗️ Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 3 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars ENVIRONMENT=CLOUD \
    --set-env-vars GCP_PROJECT=${PROJECT_ID}

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo ""
echo "✅ Deployment Complete!"
echo "======================"
echo "🌐 Dashboard URL: ${SERVICE_URL}"
echo ""
echo "📖 Access your dashboard at:"
echo "   ${SERVICE_URL}"
echo ""
echo "🔧 To update the dashboard:"
echo "   ./deploy_to_gcp.sh"
echo ""
echo "📊 Dashboard Features:"
echo "   • Cloud Deployment Status"
echo "   • Local Development Progress"
echo "   • Trading Performance Metrics"
echo "   • AI Model Analytics"
echo "   • Extreme Growth Strategy"
echo ""
echo "⚠️  Security Note:"
echo "   Dashboard is publicly accessible. Consider adding authentication for production use."

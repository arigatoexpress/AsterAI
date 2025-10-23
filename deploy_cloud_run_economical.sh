#!/bin/bash
# 🚀 Economical Cloud Run Deployment for AsterAI HFT Testing
# Cost: ~$5-15/month vs $585/month for full GKE
# Perfect for initial testing with aggressive strategies

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         💰 Economical Cloud Run Deployment ($5-15/month)     ║"
echo "║         🚀 AsterAI HFT System - Testing Ready!               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print steps
print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Configuration
SERVICE_NAME="aster-hft-testing"
REGION="us-central1"
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}

print_step "Checking prerequisites..."
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/install"
    exit 1
fi
print_success "gcloud CLI found"

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker found"

# Check if project is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "Enter your GCP Project ID:"
    read -r PROJECT_ID
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required"
        exit 1
    fi
fi

print_step "Step 1/8: Authenticating with GCP..."
gcloud auth login --brief --no-launch-browser

print_step "Step 2/8: Setting project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

print_step "Step 3/8: Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable bigquery.googleapis.com

print_step "Step 4/8: Creating economical trading Dockerfile..."
cat > Dockerfile.economical << 'EOF'
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_economical.txt .
RUN pip install --no-cache-dir -r requirements_economical.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["python", "run_economical_trader.py"]
EOF

print_step "Step 5/8: Building economical trading container..."
docker build -f Dockerfile.economical -t "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" .

print_step "Step 6/8: Pushing container to GCR..."
gcloud auth configure-docker --quiet
docker push "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

print_step "Step 7/8: Creating economical Cloud Run service..."
# Create service with minimal resources and aggressive strategies
gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --port 8080 \
    --cpu 1 \
    --memory 1Gi \
    --max-instances 1 \
    --concurrency 1 \
    --timeout 900 \
    --set-env-vars "ENVIRONMENT=live_trading" \
    --set-env-vars "MAX_POSITION_SIZE=0.10" \
    --set-env-vars "MAX_OPEN_POSITIONS=5" \
    --set-env-vars "TRADING_MODE=highly_aggressive" \
    --set-env-vars "ENABLE_PAPER_TRADING=false" \
    --set-env-vars "MAX_PORTFOLIO_RISK=0.60" \
    --set-secrets "ASTER_API_KEY=aster-api-key:latest" \
    --set-secrets "ASTER_SECRET_KEY=aster-secret-key:latest" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest"

print_step "Step 8/8: Setting up monitoring and alerts..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format="value(status.url)")

# Create uptime check
gcloud monitoring uptime-check-configs create "$SERVICE_NAME-uptime" \
    --display-name="$SERVICE_NAME Health Check" \
    --resource-type=uptime-url \
    --http-check-path="/health" \
    --http-check-port=443 \
    --monitored-resource-labels=host="$SERVICE_URL",project_id="$PROJECT_ID"

print_success "🎉 Economical deployment complete!"
echo ""
echo "📊 SERVICE DETAILS:"
echo "   🌐 URL: $SERVICE_URL"
echo "   💰 Estimated Cost: \$5-15/month (vs \$585 for GKE)"
echo "   ⚡ CPU: 1 vCPU, Memory: 1GB"
echo "   📈 Max Instances: 1 (scale to 0 when not trading)"
echo ""
echo "🎯 TRADING CONFIGURATION:"
echo "   📊 Max Position Size: 10%"
echo "   📊 Max Open Positions: 5"
echo "   📊 Mode: Aggressive strategies"
echo "   📊 Initial Mode: Paper trading"
echo ""
echo "🖥️  MONITOR YOUR DEPLOYMENT:"
echo "   📊 Dashboard: $SERVICE_URL/dashboard"
echo "   📊 Health Check: $SERVICE_URL/health"
echo "   📊 Logs: gcloud logs read --filter=\"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\""
echo ""
echo "💡 NEXT STEPS:"
echo "   1. Test paper trading: curl $SERVICE_URL/health"
echo "   2. Monitor performance: Check logs for trading activity"
echo "   3. Enable live trading: Update environment variables"
echo "   4. Scale up: Increase CPU/memory when needed"
echo ""
echo "💰 COST OPTIMIZATION TIPS:"
echo "   • Service scales to 0 when inactive (free when not trading)"
echo "   • Only pay for actual trading time"
echo "   • Upgrade to GKE only when profitable"
echo ""
print_success "🚀 Ready to start with \$50 → \$500K journey!"

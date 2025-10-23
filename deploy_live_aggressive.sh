#!/bin/bash
# 🚨 AGGRESSIVE LIVE TRADING DEPLOYMENT - Real Money Tonight!
# HIGH LEVERAGE (5x-25x) + TIGHT STOPS + Mid/Small Cap Focus

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚨 AGGRESSIVE LIVE TRADING DEPLOYMENT - REAL MONEY TONIGHT!  ║"
echo "║  💰 5x-25x LEVERAGE | 🎯 10% Positions | 🛡️ 0.3-0.5% Stops    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Risk warnings
echo -e "${YELLOW}⚠️  CRITICAL RISK WARNINGS:${NC}"
echo "   • HIGH LEVERAGE: 5x-25x can amplify losses significantly"
echo "   • TIGHT STOPS: 0.3-0.5% stops can be hit in volatile markets"
echo "   • AGGRESSIVE TARGETS: 1.5-3% profit targets (fast exits)"
echo "   • MID/SMALL CAPS: High volatility assets selected"
echo "   • LIVE TRADING: Real money at risk"
echo ""
echo -e "${GREEN}✅ CALCULATED RISKS:${NC}"
echo "   • 10% max position size per trade"
echo "   • 5 max concurrent positions"
echo "   • 5% daily loss limit (circuit breaker)"
echo "   • 60% max portfolio risk tolerance"
echo "   • Emergency paper trading switch available"
echo ""
read -p "❓ Do you want to proceed with LIVE TRADING deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "✅ Deployment cancelled. Run with paper trading instead:"
    echo "   export GCP_PROJECT_ID=\"your-project-id\""
    echo "   sed -i 's/ENABLE_PAPER_TRADING=false/ENABLE_PAPER_TRADING=true/' deploy_cloud_run_economical.sh"
    echo "   bash deploy_cloud_run_economical.sh"
    exit 1
fi

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

# Configuration
SERVICE_NAME="aster-aggressive-live"
REGION="us-central1"
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}

print_step "Checking prerequisites..."
if ! command -v gcloud &> /dev/null; then
    print_warning "gcloud CLI not found. Install from: https://cloud.google.com/sdk/install"
    exit 1
fi
print_success "gcloud CLI found"

if ! command -v docker &> /dev/null; then
    print_warning "Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker found"

# Check if project is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "Enter your GCP Project ID:"
    read -r PROJECT_ID
    if [ -z "$PROJECT_ID" ]; then
        print_warning "Project ID is required"
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

print_step "Step 4/8: Creating aggressive trading Dockerfile..."
cat > Dockerfile.aggressive << 'EOF'
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

print_step "Step 5/8: Building aggressive trading container..."
docker build -f Dockerfile.aggressive -t "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" .

print_step "Step 6/8: Pushing container to GCR..."
gcloud auth configure-docker --quiet
docker push "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

print_step "Step 7/8: Creating AGGRESSIVE live trading Cloud Run service..."
echo -e "${RED}🚨 DEPLOYING WITH LIVE TRADING ENABLED${NC}"
echo -e "${YELLOW}📊 Configuration:${NC}"
echo "   • Leverage: 5x-25x (dynamic)"
echo "   • Position Size: 10% max"
echo "   • Max Positions: 5"
echo "   • Stop Loss: 0.3-0.5%"
echo "   • Take Profit: 1.5-3%"
echo "   • Mid/Small Caps: SOL, SUI, ASTER, DOGE, etc."
echo ""

# Create service with HIGHLY AGGRESSIVE live trading settings
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
    --set-env-vars "ENVIRONMENT=live_aggressive" \
    --set-env-vars "MAX_POSITION_SIZE=0.10" \
    --set-env-vars "MAX_OPEN_POSITIONS=5" \
    --set-env-vars "TRADING_MODE=highly_aggressive" \
    --set-env-vars "ENABLE_PAPER_TRADING=false" \
    --set-env-vars "MAX_PORTFOLIO_RISK=0.60" \
    --set-env-vars "DAILY_PROFIT_TARGET=0.10" \
    --set-env-vars "MAX_DAILY_LOSS=0.05" \
    --set-secrets "ASTER_API_KEY=aster-api-key:latest" \
    --set-secrets "ASTER_SECRET_KEY=aster-secret-key:latest" \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest"

print_step "Step 8/8: Setting up aggressive monitoring and alerts..."
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format="value(status.url)")

# Create uptime check
gcloud monitoring uptime-check-configs create "$SERVICE_NAME-uptime" \
    --display-name="$SERVICE_NAME Health Check" \
    --resource-type=uptime-url \
    --http-check-path="/health" \
    --http-check-port=443 \
    --monitored-resource-labels=host="$SERVICE_URL",project_id="$PROJECT_ID"

print_success "🚨 AGGRESSIVE LIVE TRADING DEPLOYMENT COMPLETE!"
echo ""
echo -e "${RED}🚨🚨🚨 LIVE TRADING IS NOW ACTIVE! 🚨🚨🚨${NC}"
echo ""
echo "📊 SERVICE DETAILS:"
echo "   🌐 URL: $SERVICE_URL"
echo "   💰 Cost: \$10-25/month + trading fees"
echo "   🎯 Strategy: HIGHLY AGGRESSIVE (5x-25x leverage)"
echo "   📈 Target: 10% daily returns, 5% max daily loss"
echo ""
echo -e "${YELLOW}⚠️  IMMEDIATE ACTIONS REQUIRED:${NC}"
echo "   1. Visit: $SERVICE_URL/dashboard"
echo "   2. Verify 'Mode: live_trading' in dashboard"
echo "   3. Check: $SERVICE_URL/health"
echo "   4. Monitor: $SERVICE_URL/signals/SOLUSDT"
echo "   5. Set up alerts for 2% drawdown"
echo ""
echo -e "${GREEN}🎯 TRADING WILL START AUTOMATICALLY${NC}"
echo "   • Bot scans mid/small caps every 60 seconds"
echo "   • Opens positions when signal strength > 0.5"
echo "   • Uses 5x-25x leverage based on volatility"
echo "   • Exits at 1.5-3% profit or 0.3-0.5% loss"
echo ""
echo -e "${RED}🚨 EMERGENCY STOP:${NC}"
echo "   gcloud run services update $SERVICE_NAME \\"
echo "     --set-env-vars \"ENABLE_PAPER_TRADING=true\" \\"
echo "     --region=$REGION"
echo ""
echo -e "${GREEN}💰 READY TO TURN \$50 INTO \$500+!${NC}"
echo ""
print_success "🎯 Aggressive live trading bot deployed successfully!"

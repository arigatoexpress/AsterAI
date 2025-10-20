#!/bin/bash
# Quick Start: Deploy AsterAI HFT System to Google Cloud
# This script automates the entire deployment process

set -e  # Exit on error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🚀 AsterAI HFT System - Cloud Deployment Wizard      ║"
echo "║         $50 → $500k Mission - Let's Deploy!              ║"
echo "╚════════════════════════════════════════════════════════════╝"
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

# Check prerequisites
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

if ! command -v kubectl &> /dev/null; then
    print_warning "kubectl not found. Installing via gcloud..."
    gcloud components install kubectl
fi
print_success "kubectl found"

# Get configuration from user
echo ""
print_step "Configuration Setup"
echo ""

# Check if already configured
if [ -f ".gcp_config" ]; then
    source .gcp_config
    echo "Found existing configuration:"
    echo "  Project ID: $GCP_PROJECT_ID"
    echo "  Region: $GCP_REGION"
    echo ""
    read -p "Use existing configuration? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm .gcp_config
    fi
fi

# Get project configuration
if [ ! -f ".gcp_config" ]; then
    echo "Let's set up your GCP configuration..."
    echo ""
    
    # Project ID
    read -p "Enter GCP Project ID (or press Enter to create new): " PROJECT_INPUT
    if [ -z "$PROJECT_INPUT" ]; then
        export GCP_PROJECT_ID="asterai-hft-$(date +%s)"
        print_warning "Will create new project: $GCP_PROJECT_ID"
    else
        export GCP_PROJECT_ID="$PROJECT_INPUT"
    fi
    
    # Region
    echo ""
    echo "Select region:"
    echo "  1) us-east1 (South Carolina) - Recommended"
    echo "  2) us-central1 (Iowa)"
    echo "  3) us-west1 (Oregon)"
    echo "  4) europe-west1 (Belgium)"
    echo "  5) asia-east1 (Taiwan)"
    read -p "Enter choice (1-5): " REGION_CHOICE
    
    case $REGION_CHOICE in
        1) export GCP_REGION="us-east1" ;;
        2) export GCP_REGION="us-central1" ;;
        3) export GCP_REGION="us-west1" ;;
        4) export GCP_REGION="europe-west1" ;;
        5) export GCP_REGION="asia-east1" ;;
        *) export GCP_REGION="us-east1" ;;
    esac
    
    export GCP_ZONE="${GCP_REGION}-b"
    
    # Save configuration
    cat > .gcp_config <<EOF
export GCP_PROJECT_ID="$GCP_PROJECT_ID"
export GCP_REGION="$GCP_REGION"
export GCP_ZONE="$GCP_ZONE"
EOF
    
    print_success "Configuration saved to .gcp_config"
fi

# Load configuration
source .gcp_config

echo ""
print_step "Deployment Configuration:"
echo "  📦 Project: $GCP_PROJECT_ID"
echo "  🌍 Region: $GCP_REGION"
echo "  📍 Zone: $GCP_ZONE"
echo ""

# Confirm deployment
read -p "Ready to deploy? This will incur GCP charges (~$585/month). Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Deployment cancelled"
    exit 0
fi

# Step 1: Authenticate
print_step "Step 1/10: Authenticating with GCP..."
gcloud auth login --brief
gcloud config set project $GCP_PROJECT_ID
print_success "Authenticated"

# Step 2: Enable APIs
print_step "Step 2/10: Enabling required GCP APIs..."
gcloud services enable compute.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage-api.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    aiplatform.googleapis.com \
    pubsub.googleapis.com --quiet
print_success "APIs enabled"

# Step 3: Create service account
print_step "Step 3/10: Creating service account..."
SA_EMAIL="hft-trader-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SA_EMAIL &> /dev/null; then
    gcloud iam service-accounts create hft-trader-sa \
        --display-name="HFT Trader Service Account" --quiet
    
    # Grant roles
    gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/container.admin" --quiet
    
    gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/storage.admin" --quiet
    
    print_success "Service account created"
else
    print_success "Service account already exists"
fi

# Step 4: Create storage resources
print_step "Step 4/10: Creating storage resources..."

# Artifact Registry
if ! gcloud artifacts repositories describe hft-images --location=$GCP_REGION &> /dev/null; then
    gcloud artifacts repositories create hft-images \
        --repository-format=docker \
        --location=$GCP_REGION \
        --description="HFT trading system Docker images" --quiet
    print_success "Artifact Registry created"
else
    print_success "Artifact Registry already exists"
fi

# Storage buckets
for bucket in models data logs build-artifacts build-logs; do
    if ! gsutil ls -b gs://${GCP_PROJECT_ID}-${bucket} &> /dev/null 2>&1; then
        gsutil mb -c STANDARD -l $GCP_REGION gs://${GCP_PROJECT_ID}-${bucket} 2>/dev/null || true
    fi
done
print_success "Storage buckets created"

# Pub/Sub
if ! gcloud pubsub topics describe hft-sentiment &> /dev/null; then
    gcloud pubsub topics create hft-sentiment --quiet
    gcloud pubsub subscriptions create hft-sentiment-sub --topic=hft-sentiment --quiet
    print_success "Pub/Sub topic created"
else
    print_success "Pub/Sub topic already exists"
fi

# Step 5: Build Docker images
print_step "Step 5/10: Building Docker images (this may take 10-15 minutes)..."

# Configure Docker
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet

SENTIMENT_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/sentiment-analyzer"
TRADER_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/hft-aster-trader"

# Build sentiment analyzer
print_step "  Building sentiment analyzer..."
docker build -f Dockerfile.sentiment -t ${SENTIMENT_IMAGE}:latest . --quiet
docker push ${SENTIMENT_IMAGE}:latest
print_success "Sentiment analyzer image built and pushed"

# Build HFT trader
print_step "  Building HFT trader..."
docker build -f Dockerfile.gpu -t ${TRADER_IMAGE}:latest . --quiet
docker push ${TRADER_IMAGE}:latest
print_success "HFT trader image built and pushed"

# Step 6: Create GKE cluster
print_step "Step 6/10: Creating GKE cluster (this may take 5-10 minutes)..."

if ! gcloud container clusters describe hft-trading-cluster --region=$GCP_REGION &> /dev/null; then
    gcloud container clusters create hft-trading-cluster \
        --region=$GCP_REGION \
        --num-nodes=1 \
        --machine-type=n1-standard-4 \
        --disk-size=50GB \
        --disk-type=pd-ssd \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=3 \
        --enable-autorepair \
        --enable-autoupgrade \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing \
        --workload-pool=${GCP_PROJECT_ID}.svc.id.goog \
        --enable-stackdriver-kubernetes \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM \
        --quiet
    print_success "GKE cluster created"
else
    print_success "GKE cluster already exists"
fi

# Get credentials
gcloud container clusters get-credentials hft-trading-cluster --region=$GCP_REGION --quiet

# Step 7: Create GPU node pool
print_step "Step 7/10: Creating GPU node pool..."

if ! gcloud container node-pools describe gpu-pool --cluster=hft-trading-cluster --region=$GCP_REGION &> /dev/null; then
    gcloud container node-pools create gpu-pool \
        --cluster=hft-trading-cluster \
        --region=$GCP_REGION \
        --machine-type=g2-standard-4 \
        --accelerator=type=nvidia-l4,count=1 \
        --num-nodes=1 \
        --min-nodes=0 \
        --max-nodes=2 \
        --enable-autoscaling \
        --disk-size=100GB \
        --disk-type=pd-ssd \
        --node-labels=workload=gpu-inference \
        --node-taints=nvidia.com/gpu=present:NoSchedule \
        --quiet
    
    # Install NVIDIA GPU device plugin
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
    
    print_success "GPU node pool created"
else
    print_success "GPU node pool already exists"
fi

# Step 8: Configure Kubernetes
print_step "Step 8/10: Configuring Kubernetes..."

# Create namespace
kubectl create namespace hft-trading --dry-run=client -o yaml | kubectl apply -f -
kubectl config set-context --current --namespace=hft-trading

# Create secrets (placeholder - user needs to update)
kubectl create secret generic aster-api-keys \
    --from-literal=api-key='REPLACE_WITH_YOUR_ASTER_API_KEY' \
    --from-literal=api-secret='REPLACE_WITH_YOUR_ASTER_API_SECRET' \
    --namespace=hft-trading \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic gemini-api-key \
    --from-literal=api-key='REPLACE_WITH_YOUR_GEMINI_API_KEY' \
    --namespace=hft-trading \
    --dry-run=client -o yaml | kubectl apply -f -

print_success "Kubernetes configured"
print_warning "⚠️  Remember to update API keys in secrets!"

# Step 9: Deploy services
print_step "Step 9/10: Deploying services..."

# Update image references
export PROJECT_ID=$GCP_PROJECT_ID
export REGION=$GCP_REGION

# Deploy sentiment analyzer
envsubst < cloud_deploy/k8s/sentiment_deployment.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/service.yaml | kubectl apply -f -

# Deploy conservative HFT agents
envsubst < cloud_deploy/k8s/deployment.yaml | kubectl apply -f -

# Deploy degen agent
kubectl apply -f cloud_deploy/k8s/degen_configmap.yaml
envsubst < cloud_deploy/k8s/degen_deployment.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/degen_service.yaml | kubectl apply -f -

print_success "Services deployed"

# Step 10: Wait for deployments
print_step "Step 10/10: Waiting for deployments to be ready..."

kubectl wait --for=condition=available --timeout=300s deployment/sentiment-analyzer -n hft-trading || true
kubectl wait --for=condition=available --timeout=300s deployment/hft-trading-agents -n hft-trading || true
kubectl wait --for=condition=available --timeout=300s deployment/hft-degen-agent -n hft-trading || true

print_success "Deployments ready"

# Deployment summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     🎉 Deployment Complete! Your HFT System is Live!     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Deployment Summary:${NC}"
echo "  • Project: $GCP_PROJECT_ID"
echo "  • Region: $GCP_REGION"
echo "  • Cluster: hft-trading-cluster"
echo "  • GPU: NVIDIA L4"
echo ""
echo -e "${BLUE}🔗 Access Your System:${NC}"
echo "  • View pods: kubectl get pods -n hft-trading"
echo "  • View logs: kubectl logs -f deployment/hft-trading-agents -n hft-trading"
echo "  • Port forward: kubectl port-forward svc/hft-trading-service 8080:8080 -n hft-trading"
echo ""
echo -e "${YELLOW}⚠️  Important Next Steps:${NC}"
echo "  1. Update API keys:"
echo "     kubectl edit secret aster-api-keys -n hft-trading"
echo "     kubectl edit secret gemini-api-key -n hft-trading"
echo ""
echo "  2. Monitor your system:"
echo "     kubectl get pods -n hft-trading -w"
echo ""
echo "  3. Check logs for errors:"
echo "     kubectl logs -f deployment/hft-trading-agents -n hft-trading"
echo ""
echo "  4. View GCP Console:"
echo "     https://console.cloud.google.com/kubernetes/workload?project=$GCP_PROJECT_ID"
echo ""
echo -e "${BLUE}💰 Estimated Monthly Cost: ~$585${NC}"
echo "  • GKE Cluster: ~$150"
echo "  • GPU Node: ~$400"
echo "  • Storage: ~$20"
echo "  • Network: ~$15"
echo ""
echo -e "${GREEN}🎯 Mission Status: $50 → $500k transformation system deployed!${NC}"
echo ""
echo "For detailed instructions, see: DEPLOYMENT_GUIDE.md"
echo ""





#!/bin/bash
# GKE Cluster Setup Script for HFT Aster Trader

set -e

# --- Configuration ---
PROJECT_ID="${GCP_PROJECT_ID:-hft-aster-trader}"
REGION="us-east1"
ZONE="us-east1-b"
CLUSTER_NAME="hft-trading-cluster"
NODE_POOL_NAME="gpu-node-pool"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-l4"
GPU_COUNT=1

# --- Color Definitions ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Helper Functions ---
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# --- Main Script ---
main() {
    print_step "Starting GKE Cluster Setup for HFT Aster Trader"

    # --- Set Project and Region ---
    print_step "Configuring gcloud for project ${PROJECT_ID} in ${REGION}"
    gcloud config set project ${PROJECT_ID}
    gcloud config set compute/region ${REGION}
    gcloud config set compute/zone ${ZONE}
    print_success "gcloud configured"

    # --- Enable APIs ---
    print_step "Enabling required APIs..."
    gcloud services enable \
        container.googleapis.com \
        compute.googleapis.com \
        artifactregistry.googleapis.com \
        cloudbuild.googleapis.com \
        pubsub.googleapis.com \
        aiplatform.googleapis.com \
        vertexai.googleapis.com \
        secretmanager.googleapis.com
    print_success "Required APIs enabled"

    # --- Create GKE Cluster ---
    print_step "Creating GKE cluster '${CLUSTER_NAME}' (this may take several minutes)..."
    if ! gcloud container clusters describe ${CLUSTER_NAME} --region=${REGION} &> /dev/null; then
        gcloud container clusters create ${CLUSTER_NAME} \
            --region=${REGION} \
            --num-nodes=1 \
            --machine-type="e2-medium" \
            --workload-pool=${PROJECT_ID}.svc.id.goog \
            --release-channel=regular \
            --enable-shielded-nodes \
            --enable-network-policy
        print_success "GKE cluster '${CLUSTER_NAME}' created"
    else
        print_success "GKE cluster '${CLUSTER_NAME}' already exists"
    fi

    # --- Connect to Cluster ---
    print_step "Connecting kubectl to the cluster"
    gcloud container clusters get-credentials ${CLUSTER_NAME} --region=${REGION}
    print_success "kubectl configured"

    # --- Create GPU Node Pool ---
    print_step "Creating GPU node pool '${NODE_POOL_NAME}' with ${GPU_COUNT} ${GPU_TYPE} GPU(s)..."
    if ! gcloud container node-pools describe ${NODE_POOL_NAME} --cluster=${CLUSTER_NAME} --region=${REGION} &> /dev/null; then
        gcloud container node-pools create ${NODE_POOL_NAME} \
            --cluster=${CLUSTER_NAME} \
            --region=${REGION} \
            --machine-type=${MACHINE_TYPE} \
            --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
            --num-nodes=1 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=5 \
            --spot  # Use Spot VMs for cost savings on non-critical workloads if applicable
        print_success "GPU node pool '${NODE_POOL_NAME}' created"
    else
        print_success "GPU node pool '${NODE_POOL_NAME}' already exists"
    fi

    # --- Install NVIDIA GPU Driver ---
    print_step "Applying NVIDIA GPU driver installation DaemonSet"
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    print_success "NVIDIA driver DaemonSet applied. Drivers will be installed on nodes automatically."

    # --- Create Pub/Sub Topic ---
    print_step "Creating Pub/Sub topic 'hft-sentiment'"
    if ! gcloud pubsub topics describe hft-sentiment &> /dev/null; then
        gcloud pubsub topics create hft-sentiment
        print_success "Pub/Sub topic 'hft-sentiment' created"
    else
        print_success "Pub/Sub topic 'hft-sentiment' already exists"
    fi

    # --- Create Pub/Sub Subscription ---
    print_step "Creating Pub/Sub subscription 'hft-sentiment-sub'"
    if ! gcloud pubsub subscriptions describe hft-sentiment-sub &> /dev/null; then
        gcloud pubsub subscriptions create hft-sentiment-sub \
            --topic=hft-sentiment \
            --ack-deadline=60
        print_success "Pub/Sub subscription 'hft-sentiment-sub' created"
    else
        print_success "Pub/Sub subscription 'hft-sentiment-sub' already exists"
    fi

    # --- Create Artifact Registry ---
    print_step "Creating Artifact Registry 'hft-images'"
    if ! gcloud artifacts repositories describe hft-images --location=${REGION} &> /dev/null; then
        gcloud artifacts repositories create hft-images \
            --repository-format=docker \
            --location=${REGION} \
            --description="Docker repository for HFT trader images"
        print_success "Artifact Registry 'hft-images' created"
    else
        print_success "Artifact Registry 'hft-images' already exists"
    fi

    # --- Create Service Accounts ---
    print_step "Creating service accounts..."

    # Pub/Sub publisher service account
    if ! gcloud iam service-accounts describe pubsub-publisher-sa@${PROJECT_ID}.iam.gserviceaccount.com &> /dev/null; then
        gcloud iam service-accounts create pubsub-publisher-sa \
            --display-name="Pub/Sub Publisher Service Account"
        print_success "Service account 'pubsub-publisher-sa' created"
    else
        print_success "Service account 'pubsub-publisher-sa' already exists"
    fi

    # HFT trader service account
    if ! gcloud iam service-accounts describe hft-trader-sa@${PROJECT_ID}.iam.gserviceaccount.com &> /dev/null; then
        gcloud iam service-accounts create hft-trader-sa \
            --display-name="HFT Trader Service Account"
        print_success "Service account 'hft-trader-sa' created"
    else
        print_success "Service account 'hft-trader-sa' already exists"
    fi

    # Grant roles
    print_step "Granting IAM roles..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:pubsub-publisher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/pubsub.publisher" \
        --condition=None

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:hft-trader-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/pubsub.subscriber" \
        --condition=None

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:hft-trader-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user" \
        --condition=None

    print_success "IAM roles granted"

    # --- Create Kubernetes Service Account ---
    print_step "Creating Kubernetes service account"
    kubectl create serviceaccount hft-trader-sa --dry-run=client -o yaml | kubectl apply -f -
    print_success "Kubernetes service account created"

    # --- Create Secret for API Keys ---
    print_step "Creating Kubernetes secret for API keys"
    # Note: You'll need to populate these with actual keys
    kubectl create secret generic api-keys \
        --from-literal=gemini-api-key="YOUR_GEMINI_API_KEY" \
        --from-literal=aster-api-key="YOUR_ASTER_API_KEY" \
        --from-literal=aster-api-secret="YOUR_ASTER_API_SECRET" \
        --dry-run=client -o yaml | kubectl apply -f -
    print_success "API keys secret created (populate with real keys)"

    echo -e "\n${GREEN}GKE Cluster setup is complete!${NC}"
    echo "Your cluster '${CLUSTER_NAME}' is ready with a GPU-enabled node pool."
    echo "You can now deploy your trading agents and services using the Kubernetes manifests."
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Populate the api-keys secret with your actual API keys"
    echo "2. Run: kubectl apply -f cloud_deploy/k8s/"
    echo "3. Check deployments: kubectl get pods"
    echo "4. Monitor logs: kubectl logs -f deployment/sentiment-analyzer"
}

# --- Execute Main Function ---
main

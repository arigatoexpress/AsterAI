#!/bin/bash
# HFT Aster Trader - GCP Deployment Script
# Deploys HFT system to Google Cloud Platform with L4 GPU

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-hft-aster-trader}"
REGION="${GCP_REGION:-us-east1}"
ZONE="${GCP_ZONE:-us-east1-b}"
INSTANCE_NAME="hft-trader-gpu"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-l4"
GPU_COUNT=1

# Functions
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_requirements() {
    print_step "Checking requirements..."
    
    # Check gcloud
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/install"
        exit 1
    fi
    print_success "gcloud CLI found"
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker found"
    
    # Check kubectl (optional but recommended)
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not found (optional for GKE)"
    else
        print_success "kubectl found"
    fi
}

setup_gcp_project() {
    print_step "Setting up GCP project..."
    
    # Set project
    gcloud config set project ${PROJECT_ID}
    print_success "Project set to: ${PROJECT_ID}"
    
    # Enable required APIs
    print_step "Enabling required GCP APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage-api.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable monitoring.googleapis.com
    gcloud services enable logging.googleapis.com
    print_success "APIs enabled"
}

create_service_account() {
    print_step "Creating service account..."
    
    SA_NAME="hft-trader-sa"
    SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe ${SA_EMAIL} &> /dev/null; then
        gcloud iam service-accounts create ${SA_NAME} \
            --display-name="HFT Trader Service Account"
        print_success "Service account created: ${SA_EMAIL}"
    else
        print_success "Service account already exists: ${SA_EMAIL}"
    fi
    
    # Grant roles
    print_step "Granting IAM roles..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/compute.instanceAdmin.v1" \
        --condition=None
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/storage.objectAdmin" \
        --condition=None
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/logging.logWriter" \
        --condition=None
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/monitoring.metricWriter" \
        --condition=None
    
    print_success "IAM roles granted"
}

create_storage_buckets() {
    print_step "Creating storage buckets..."
    
    # Model bucket
    MODEL_BUCKET="hft-models-${PROJECT_ID}"
    if ! gsutil ls -b gs://${MODEL_BUCKET} &> /dev/null; then
        gsutil mb -c STANDARD -l ${REGION} gs://${MODEL_BUCKET}
        print_success "Model bucket created: gs://${MODEL_BUCKET}"
    else
        print_success "Model bucket exists: gs://${MODEL_BUCKET}"
    fi
    
    # Data bucket
    DATA_BUCKET="hft-data-${PROJECT_ID}"
    if ! gsutil ls -b gs://${DATA_BUCKET} &> /dev/null; then
        gsutil mb -c STANDARD -l ${REGION} gs://${DATA_BUCKET}
        print_success "Data bucket created: gs://${DATA_BUCKET}"
    else
        print_success "Data bucket exists: gs://${DATA_BUCKET}"
    fi
    
    # Logs bucket
    LOGS_BUCKET="hft-logs-${PROJECT_ID}"
    if ! gsutil ls -b gs://${LOGS_BUCKET} &> /dev/null; then
        gsutil mb -c NEARLINE -l ${REGION} gs://${LOGS_BUCKET}
        print_success "Logs bucket created: gs://${LOGS_BUCKET}"
    else
        print_success "Logs bucket exists: gs://${LOGS_BUCKET}"
    fi
}

build_docker_image() {
    print_step "Building Docker image..."
    
    IMAGE_NAME="gcr.io/${PROJECT_ID}/hft-aster-trader"
    IMAGE_TAG=$(git rev-parse --short HEAD || echo "latest")
    FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Build image
    docker build -f ../Dockerfile.gpu -t ${FULL_IMAGE} ..
    print_success "Image built: ${FULL_IMAGE}"
    
    # Configure docker for GCR
    gcloud auth configure-docker --quiet
    
    # Push to GCR
    docker push ${FULL_IMAGE}
    print_success "Image pushed to GCR"
    
    # Tag as latest
    docker tag ${FULL_IMAGE} ${IMAGE_NAME}:latest
    docker push ${IMAGE_NAME}:latest
    print_success "Latest tag pushed"
    
    echo "${FULL_IMAGE}" > .image_name.txt
}

create_firewall_rules() {
    print_step "Creating firewall rules..."
    
    # Allow HFT API
    if ! gcloud compute firewall-rules describe allow-hft-api &> /dev/null; then
        gcloud compute firewall-rules create allow-hft-api \
            --direction=INGRESS \
            --priority=1000 \
            --network=default \
            --action=ALLOW \
            --rules=tcp:8080,tcp:9090 \
            --source-ranges=0.0.0.0/0
        print_success "Firewall rule created: allow-hft-api"
    else
        print_success "Firewall rule exists: allow-hft-api"
    fi
}

create_gpu_instance() {
    print_step "Creating GPU instance..."
    
    # Check if instance exists
    if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} &> /dev/null; then
        print_warning "Instance ${INSTANCE_NAME} already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
            print_success "Old instance deleted"
        else
            print_step "Skipping instance creation"
            return
        fi
    fi
    
    SA_EMAIL="hft-trader-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    IMAGE_NAME=$(cat .image_name.txt || echo "gcr.io/${PROJECT_ID}/hft-aster-trader:latest")
    
    # Create instance with GPU
    gcloud compute instances create ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --machine-type=${MACHINE_TYPE} \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --maintenance-policy=TERMINATE \
        --image-family=ubuntu-2004-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --network-tier=PREMIUM \
        --service-account=${SA_EMAIL} \
        --scopes=cloud-platform \
        --metadata=startup-script='#!/bin/bash
# Install NVIDIA drivers
apt-get update
apt-get install -y ubuntu-drivers-common
ubuntu-drivers autoinstall
reboot
'
    
    print_success "Instance created: ${INSTANCE_NAME}"
    print_warning "Instance is rebooting to complete driver installation (wait 2-3 minutes)"
}

deploy_containers() {
    print_step "Deploying containers to instance..."
    
    SA_EMAIL="hft-trader-sa@${PROJECT_ID}.iam.gserviceaccount.com"
    IMAGE_NAME=$(cat .image_name.txt || echo "gcr.io/${PROJECT_ID}/hft-aster-trader:latest")
    
    print_warning "Waiting for instance to be ready..."
    sleep 120  # Wait for reboot
    
    # Wait for SSH
    for i in {1..10}; do
        if gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="echo 'SSH ready'" &> /dev/null; then
            print_success "Instance is ready"
            break
        fi
        print_step "Waiting for SSH... (attempt $i/10)"
        sleep 30
    done
    
    # Install Docker on instance
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker
'
    
    # Install NVIDIA Container Toolkit
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
'
    
    print_success "Docker and NVIDIA toolkit installed"
    
    # Pull and run container
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="
sudo docker pull ${IMAGE_NAME}
sudo docker run -d \
    --name hft-trader \
    --gpus all \
    --restart unless-stopped \
    -p 8080:8080 \
    -p 9090:9090 \
    -e ASTER_API_KEY=\${ASTER_API_KEY} \
    -e ASTER_API_SECRET=\${ASTER_API_SECRET} \
    -e INITIAL_CAPITAL=50 \
    -e GPU_ACCELERATION=true \
    ${IMAGE_NAME}
"
    
    print_success "Container deployed"
}

setup_monitoring() {
    print_step "Setting up monitoring..."
    
    # Copy Prometheus config
    gcloud compute scp prometheus.yml ${INSTANCE_NAME}:/tmp/ --zone=${ZONE}
    
    # Start Prometheus
    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command='
sudo docker run -d \
    --name prometheus \
    --restart unless-stopped \
    -p 9091:9090 \
    -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus:latest
'
    
    print_success "Prometheus started"
}

print_summary() {
    print_step "Deployment Summary"
    
    EXTERNAL_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     HFT Aster Trader Deployment Complete!     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Instance:${NC} ${INSTANCE_NAME}"
    echo -e "${BLUE}Zone:${NC} ${ZONE}"
    echo -e "${BLUE}External IP:${NC} ${EXTERNAL_IP}"
    echo ""
    echo -e "${BLUE}Endpoints:${NC}"
    echo -e "  API:        http://${EXTERNAL_IP}:8080"
    echo -e "  Metrics:    http://${EXTERNAL_IP}:9090"
    echo -e "  Prometheus: http://${EXTERNAL_IP}:9091"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Set API keys: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
    echo "  2. Check logs: sudo docker logs -f hft-trader"
    echo "  3. Monitor: http://${EXTERNAL_IP}:9091"
    echo ""
    echo -e "${YELLOW}⚠  Remember to:${NC}"
    echo "  - Set your Aster API keys as environment variables"
    echo "  - Test with small capital first"
    echo "  - Monitor latency and performance"
    echo "  - Enable alerting for critical metrics"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║   HFT Aster Trader - GCP Deployment Script    ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    
    check_requirements
    setup_gcp_project
    create_service_account
    create_storage_buckets
    build_docker_image
    create_firewall_rules
    create_gpu_instance
    deploy_containers
    setup_monitoring
    print_summary
}

# Run main
main "$@"


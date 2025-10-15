# 🚀 AsterAI Cloud Deployment Guide

**Mission**: Deploy HFT trading system to Google Cloud Platform with GPU acceleration

**Estimated Time**: 2-3 hours for complete deployment

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Google Cloud Platform Account** with billing enabled
- [ ] **gcloud CLI** installed and configured
- [ ] **Docker Desktop** installed and running
- [ ] **kubectl** installed (comes with gcloud)
- [ ] **Git** repository set up (for Cloud Build triggers)
- [ ] **Aster DEX API Keys** (for live trading)
- [ ] **RTX 5070 Ti GPU** (for local development/testing)
- [ ] **Python 3.13** environment set up

---

## 🎯 Deployment Overview

```
Phase 1: Pre-Deployment (30 min)
├── Fix local code issues
├── Verify dependencies
└── Test core components

Phase 2: GCP Setup (30 min)
├── Create GCP project
├── Enable required APIs
├── Set up authentication
└── Create service accounts

Phase 3: Build & Push Images (30 min)
├── Build sentiment analyzer image
├── Build HFT trader image
└── Push to Artifact Registry

Phase 4: Infrastructure (45 min)
├── Create GKE cluster
├── Set up GPU node pools
├── Configure networking
└── Set up storage buckets

Phase 5: Deploy Services (30 min)
├── Deploy sentiment analyzer
├── Deploy conservative HFT agents
├── Deploy degen trading agent
└── Configure load balancers

Phase 6: Configuration & Testing (30 min)
├── Set API keys and secrets
├── Run integration tests
├── Set up monitoring
└── Validate live trading
```

---

## 📝 Step-by-Step Instructions

### Phase 1: Pre-Deployment Preparation

#### Step 1.1: Fix Local Code Issues

Based on your test results, we need to fix import and strategy issues:

```bash
# Check current test status
python local_development_test.py

# Expected issues to fix:
# - Import errors in mcp_trader modules
# - Strategy execution issues
```

**Action Items**:
1. Fix import errors in `mcp_trader/` modules
2. Resolve strategy test failures
3. Ensure all core components pass tests

#### Step 1.2: Verify GPU Setup

```bash
# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run GPU verification
python scripts/test_gpu_utils.py
```

#### Step 1.3: Install Cloud Dependencies

```bash
# Install GCP-specific dependencies
pip install google-cloud-aiplatform google-cloud-storage google-cloud-bigquery
pip install kubernetes docker

# Verify installations
gcloud version
docker --version
kubectl version --client
```

---

### Phase 2: GCP Project Setup

#### Step 2.1: Create GCP Project

```bash
# Set your project ID (must be globally unique)
export GCP_PROJECT_ID="asterai-hft-$(date +%s)"
export GCP_REGION="us-east1"
export GCP_ZONE="us-east1-b"

# Create new project
gcloud projects create $GCP_PROJECT_ID --name="AsterAI HFT Trading"

# Set as active project
gcloud config set project $GCP_PROJECT_ID

# Link billing account (replace with your billing account ID)
# Find your billing account: gcloud billing accounts list
export BILLING_ACCOUNT_ID="YOUR_BILLING_ACCOUNT_ID"
gcloud billing projects link $GCP_PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID
```

#### Step 2.2: Enable Required APIs

```bash
# Enable all required GCP APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable pubsub.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled
```

#### Step 2.3: Set Up Authentication

```bash
# Authenticate with GCP
gcloud auth login

# Set up application default credentials
gcloud auth application-default login

# Create service account for HFT system
gcloud iam service-accounts create hft-trader-sa \
    --display-name="HFT Trader Service Account" \
    --description="Service account for AsterAI HFT trading system"

# Grant necessary roles
export SA_EMAIL="hft-trader-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/container.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/monitoring.metricWriter"
```

#### Step 2.4: Create Storage Resources

```bash
# Create Artifact Registry for Docker images
gcloud artifacts repositories create hft-images \
    --repository-format=docker \
    --location=$GCP_REGION \
    --description="HFT trading system Docker images"

# Create Cloud Storage buckets
gsutil mb -c STANDARD -l $GCP_REGION gs://${GCP_PROJECT_ID}-models
gsutil mb -c STANDARD -l $GCP_REGION gs://${GCP_PROJECT_ID}-data
gsutil mb -c NEARLINE -l $GCP_REGION gs://${GCP_PROJECT_ID}-logs
gsutil mb -c STANDARD -l $GCP_REGION gs://${GCP_PROJECT_ID}-build-artifacts
gsutil mb -c STANDARD -l $GCP_REGION gs://${GCP_PROJECT_ID}-build-logs

# Create Pub/Sub topic for sentiment analysis
gcloud pubsub topics create hft-sentiment
gcloud pubsub subscriptions create hft-sentiment-sub --topic=hft-sentiment
```

---

### Phase 3: Build and Push Docker Images

#### Step 3.1: Configure Docker for Artifact Registry

```bash
# Configure Docker authentication
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Set image names
export SENTIMENT_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/sentiment-analyzer"
export TRADER_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/hft-aster-trader"
```

#### Step 3.2: Build Sentiment Analyzer Image

```bash
# Build sentiment analyzer
docker build -f Dockerfile.sentiment -t ${SENTIMENT_IMAGE}:latest .

# Test image locally (optional)
docker run --rm ${SENTIMENT_IMAGE}:latest python -c "import sys; print(f'Python {sys.version}')"

# Push to Artifact Registry
docker push ${SENTIMENT_IMAGE}:latest
```

#### Step 3.3: Build HFT Trader Image

```bash
# Build HFT trader with GPU support
docker build -f Dockerfile.gpu -t ${TRADER_IMAGE}:latest .

# Tag with version
export IMAGE_TAG=$(git rev-parse --short HEAD || echo "v1.0")
docker tag ${TRADER_IMAGE}:latest ${TRADER_IMAGE}:${IMAGE_TAG}

# Push both tags
docker push ${TRADER_IMAGE}:latest
docker push ${TRADER_IMAGE}:${IMAGE_TAG}

# Verify images in registry
gcloud artifacts docker images list ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images
```

---

### Phase 4: Deploy GKE Infrastructure

#### Step 4.1: Create GKE Cluster

```bash
# Create GKE cluster with GPU support
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
    --maintenance-window-start="2024-01-01T00:00:00Z" \
    --maintenance-window-duration=4h \
    --maintenance-window-recurrence="FREQ=WEEKLY;BYDAY=SU" \
    --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
    --workload-pool=${GCP_PROJECT_ID}.svc.id.goog \
    --enable-stackdriver-kubernetes \
    --logging=SYSTEM,WORKLOAD \
    --monitoring=SYSTEM

# Get cluster credentials
gcloud container clusters get-credentials hft-trading-cluster --region=$GCP_REGION

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

#### Step 4.2: Create GPU Node Pool

```bash
# Create GPU node pool with NVIDIA L4 GPUs
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
    --node-taints=nvidia.com/gpu=present:NoSchedule

# Install NVIDIA GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Verify GPU nodes
kubectl get nodes -l workload=gpu-inference
kubectl describe nodes -l workload=gpu-inference | grep -A 5 "Allocatable"
```

#### Step 4.3: Configure Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace hft-trading

# Set as default namespace
kubectl config set-context --current --namespace=hft-trading

# Create secret for Aster API keys (replace with your actual keys)
kubectl create secret generic aster-api-keys \
    --from-literal=api-key='YOUR_ASTER_API_KEY' \
    --from-literal=api-secret='YOUR_ASTER_API_SECRET' \
    --namespace=hft-trading

# Create secret for Gemini API key (for sentiment analysis)
kubectl create secret generic gemini-api-key \
    --from-literal=api-key='YOUR_GEMINI_API_KEY' \
    --namespace=hft-trading

# Verify secrets
kubectl get secrets -n hft-trading
```

---

### Phase 5: Deploy Services

#### Step 5.1: Update Kubernetes Manifests

```bash
# Update image references in deployment files
export PROJECT_ID=$GCP_PROJECT_ID
export REGION=$GCP_REGION

# Process deployment files with environment variables
envsubst < cloud_deploy/k8s/sentiment_deployment.yaml > /tmp/sentiment_deployment.yaml
envsubst < cloud_deploy/k8s/deployment.yaml > /tmp/deployment.yaml
envsubst < cloud_deploy/k8s/degen_deployment.yaml > /tmp/degen_deployment.yaml
envsubst < cloud_deploy/k8s/service.yaml > /tmp/service.yaml
envsubst < cloud_deploy/k8s/degen_service.yaml > /tmp/degen_service.yaml
```

#### Step 5.2: Deploy Sentiment Analyzer

```bash
# Deploy sentiment analyzer
kubectl apply -f /tmp/sentiment_deployment.yaml -n hft-trading

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s \
    deployment/sentiment-analyzer -n hft-trading

# Check status
kubectl get pods -l app=sentiment-analyzer -n hft-trading
kubectl logs -l app=sentiment-analyzer -n hft-trading --tail=50
```

#### Step 5.3: Deploy Conservative HFT Agents

```bash
# Deploy conservative trading agents
kubectl apply -f /tmp/deployment.yaml -n hft-trading
kubectl apply -f /tmp/service.yaml -n hft-trading

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s \
    deployment/hft-trading-agents -n hft-trading

# Check status
kubectl get pods -l app=hft-trading-agents -n hft-trading
kubectl logs -l app=hft-trading-agents -n hft-trading --tail=50
```

#### Step 5.4: Deploy Degen Trading Agent

```bash
# Deploy degen agent (HIGH RISK - monitor closely!)
kubectl apply -f cloud_deploy/k8s/degen_configmap.yaml -n hft-trading
kubectl apply -f /tmp/degen_deployment.yaml -n hft-trading
kubectl apply -f /tmp/degen_service.yaml -n hft-trading

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s \
    deployment/hft-degen-agent -n hft-trading

# Check status
kubectl get pods -l app=hft-degen-agent -n hft-trading
kubectl logs -l app=hft-degen-agent -n hft-trading --tail=50
```

#### Step 5.5: Verify All Services

```bash
# Check all deployments
kubectl get deployments -n hft-trading
kubectl get pods -n hft-trading
kubectl get services -n hft-trading

# Get service endpoints
kubectl get svc -n hft-trading -o wide
```

---

### Phase 6: Configuration and Testing

#### Step 6.1: Set Up Port Forwarding (for testing)

```bash
# Forward sentiment service
kubectl port-forward svc/sentiment-service 8081:8081 -n hft-trading &

# Forward trading service
kubectl port-forward svc/hft-trading-service 8080:8080 -n hft-trading &

# Forward degen service
kubectl port-forward svc/hft-degen-service 8082:8080 -n hft-trading &

# Forward metrics
kubectl port-forward svc/hft-trading-service 9090:9090 -n hft-trading &
```

#### Step 6.2: Run Health Checks

```bash
# Test sentiment analyzer
curl http://localhost:8081/health

# Test conservative trading service
curl http://localhost:8080/health

# Test degen trading service
curl http://localhost:8082/health

# Check metrics
curl http://localhost:9090/metrics
```

#### Step 6.3: Set Up Monitoring

```bash
# Create Cloud Monitoring dashboard
gcloud monitoring dashboards create --config-from-file=- <<EOF
{
  "displayName": "HFT Trading System",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Pod CPU Usage",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"k8s_pod\" AND resource.labels.namespace_name=\"hft-trading\""
                }
              }
            }]
          }
        }
      }
    ]
  }
}
EOF

# Set up log-based alerts
gcloud logging metrics create trading-errors \
    --description="Count of trading errors" \
    --log-filter='resource.type="k8s_container"
    resource.labels.namespace_name="hft-trading"
    severity>=ERROR'
```

#### Step 6.4: Run Integration Tests

```bash
# Run integration tests from local machine
python integration_tests.py --cloud --project-id=$GCP_PROJECT_ID

# Or run tests in cluster
kubectl run integration-test \
    --image=${TRADER_IMAGE}:latest \
    --restart=Never \
    --namespace=hft-trading \
    --command -- python integration_tests.py

# Check test results
kubectl logs integration-test -n hft-trading
```

---

## 🎯 Post-Deployment Checklist

After deployment, verify:

- [ ] All pods are running: `kubectl get pods -n hft-trading`
- [ ] Services are accessible: `kubectl get svc -n hft-trading`
- [ ] Health checks pass for all services
- [ ] Logs show no critical errors
- [ ] Metrics are being collected
- [ ] API keys are configured correctly
- [ ] GPU resources are allocated to pods
- [ ] Monitoring dashboard shows data
- [ ] Alerts are configured

---

## 📊 Monitoring and Management

### View Logs

```bash
# Sentiment analyzer logs
kubectl logs -f deployment/sentiment-analyzer -n hft-trading

# Conservative agent logs
kubectl logs -f deployment/hft-trading-agents -n hft-trading

# Degen agent logs (monitor closely!)
kubectl logs -f deployment/hft-degen-agent -n hft-trading

# All logs
kubectl logs -f -l app=hft-trading-agents -n hft-trading --all-containers
```

### Scale Services

```bash
# Scale conservative agents
kubectl scale deployment/hft-trading-agents --replicas=3 -n hft-trading

# Scale degen agent (be careful!)
kubectl scale deployment/hft-degen-agent --replicas=1 -n hft-trading

# Auto-scaling
kubectl autoscale deployment/hft-trading-agents \
    --min=2 --max=5 --cpu-percent=80 -n hft-trading
```

### Update Deployment

```bash
# Build new image
docker build -f Dockerfile.gpu -t ${TRADER_IMAGE}:v2 .
docker push ${TRADER_IMAGE}:v2

# Update deployment
kubectl set image deployment/hft-trading-agents \
    hft-trader=${TRADER_IMAGE}:v2 -n hft-trading

# Check rollout status
kubectl rollout status deployment/hft-trading-agents -n hft-trading

# Rollback if needed
kubectl rollout undo deployment/hft-trading-agents -n hft-trading
```

---

## 🚨 Troubleshooting

### Common Issues

**Issue: Pods stuck in Pending state**
```bash
kubectl describe pod <pod-name> -n hft-trading
# Check for resource constraints or image pull errors
```

**Issue: GPU not available**
```bash
kubectl describe nodes -l workload=gpu-inference
# Verify GPU driver installation and node labels
```

**Issue: API connection errors**
```bash
kubectl logs <pod-name> -n hft-trading
# Check API keys in secrets
kubectl get secret aster-api-keys -n hft-trading -o yaml
```

**Issue: High latency**
```bash
# Check pod location and network
kubectl get pods -n hft-trading -o wide
# Ensure pods are in same zone as data sources
```

---

## 💰 Cost Optimization

### Estimated Monthly Costs

- **GKE Cluster**: ~$150/month (n1-standard-4, 1 node)
- **GPU Node Pool**: ~$400/month (g2-standard-4 with L4 GPU, when running)
- **Storage**: ~$20/month (100GB total)
- **Network**: ~$10/month (egress)
- **Monitoring**: ~$5/month

**Total**: ~$585/month (with GPU running 24/7)

### Cost Reduction Tips

1. **Scale down GPU nodes when not needed**:
   ```bash
   gcloud container node-pools update gpu-pool \
       --cluster=hft-trading-cluster \
       --region=$GCP_REGION \
       --enable-autoscaling \
       --min-nodes=0 \
       --max-nodes=1
   ```

2. **Use preemptible VMs for non-critical workloads**:
   ```bash
   --preemptible flag when creating node pools
   ```

3. **Set up budget alerts**:
   ```bash
   gcloud billing budgets create \
       --billing-account=$BILLING_ACCOUNT_ID \
       --display-name="HFT Trading Budget" \
       --budget-amount=1000USD \
       --threshold-rule=percent=90
   ```

---

## 🔐 Security Best Practices

1. **Rotate API keys regularly**
2. **Use Workload Identity for GCP services**
3. **Enable Binary Authorization for container images**
4. **Set up VPC firewall rules**
5. **Enable audit logging**
6. **Use Secret Manager instead of Kubernetes secrets**

---

## 📚 Additional Resources

- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Artifact Registry Guide](https://cloud.google.com/artifact-registry/docs)
- [GPU on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [Cloud Monitoring](https://cloud.google.com/monitoring/docs)

---

## ✅ Success Criteria

Your deployment is successful when:

1. ✅ All pods show "Running" status
2. ✅ Health checks return 200 OK
3. ✅ Logs show successful API connections
4. ✅ GPU utilization visible in monitoring
5. ✅ Trading signals being generated
6. ✅ No critical errors in logs
7. ✅ Latency < 10ms for order execution
8. ✅ P&L tracking shows activity

---

**Next Steps**: Once deployed, monitor for 24 hours with paper trading before enabling live trading with real capital.

**Support**: Check logs and monitoring dashboards regularly. Set up alerts for critical errors.

---

*Last Updated: 2025-10-15*
*Version: 1.0*





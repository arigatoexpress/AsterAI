# 🚀 Quick Deploy Guide - AsterAI HFT System

**Get your HFT trading system running on Google Cloud in 30 minutes!**

---

## 📋 Prerequisites (5 minutes)

### 1. Install Required Tools

**Windows:**
```powershell
# Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/install

# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify installations
gcloud version
docker --version
```

**Linux/Mac:**
```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install Docker
# Follow: https://docs.docker.com/engine/install/

# Verify
gcloud version
docker --version
```

### 2. Get API Keys

You'll need:
- **Aster DEX API Key** - Get from [Aster DEX Dashboard](https://app.asterdex.com)
- **Google Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **GCP Billing Account** - Set up at [GCP Console](https://console.cloud.google.com/billing)

---

## 🎯 One-Command Deployment

### Option 1: Automated Script (Recommended)

```bash
# Make script executable (Linux/Mac)
chmod +x deploy_to_gcp.sh

# Run deployment wizard
./deploy_to_gcp.sh
```

**Windows PowerShell:**
```powershell
# Run with Git Bash or WSL
bash deploy_to_gcp.sh
```

The script will:
1. ✅ Check prerequisites
2. ✅ Create GCP project
3. ✅ Enable required APIs
4. ✅ Build Docker images
5. ✅ Create GKE cluster with GPU
6. ✅ Deploy all services
7. ✅ Configure monitoring

**Total time: ~20-30 minutes**

---

## 📝 Manual Deployment (Step-by-Step)

If you prefer manual control, follow these steps:

### Step 1: Set Up GCP Project (5 minutes)

```bash
# Set variables
export GCP_PROJECT_ID="asterai-hft-$(date +%s)"
export GCP_REGION="us-east1"
export GCP_ZONE="us-east1-b"

# Create project
gcloud projects create $GCP_PROJECT_ID

# Set as active
gcloud config set project $GCP_PROJECT_ID

# Link billing (replace with your billing account ID)
gcloud billing projects link $GCP_PROJECT_ID \
    --billing-account=YOUR_BILLING_ACCOUNT_ID

# Enable APIs
gcloud services enable \
    compute.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    storage-api.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    aiplatform.googleapis.com \
    pubsub.googleapis.com
```

### Step 2: Create Infrastructure (5 minutes)

```bash
# Create Artifact Registry
gcloud artifacts repositories create hft-images \
    --repository-format=docker \
    --location=$GCP_REGION

# Create storage buckets
gsutil mb -l $GCP_REGION gs://${GCP_PROJECT_ID}-models
gsutil mb -l $GCP_REGION gs://${GCP_PROJECT_ID}-data
gsutil mb -l $GCP_REGION gs://${GCP_PROJECT_ID}-logs

# Create Pub/Sub topic
gcloud pubsub topics create hft-sentiment
```

### Step 3: Build & Push Images (10 minutes)

```bash
# Configure Docker
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Build images
docker build -f Dockerfile.sentiment \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/sentiment-analyzer:latest .

docker build -f Dockerfile.gpu \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/hft-aster-trader:latest .

# Push images
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/sentiment-analyzer:latest
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/hft-images/hft-aster-trader:latest
```

### Step 4: Create GKE Cluster (10 minutes)

```bash
# Create cluster
gcloud container clusters create hft-trading-cluster \
    --region=$GCP_REGION \
    --num-nodes=1 \
    --machine-type=n1-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=3

# Create GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster=hft-trading-cluster \
    --region=$GCP_REGION \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=2

# Get credentials
gcloud container clusters get-credentials hft-trading-cluster --region=$GCP_REGION

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### Step 5: Deploy Services (5 minutes)

```bash
# Create namespace
kubectl create namespace hft-trading
kubectl config set-context --current --namespace=hft-trading

# Create secrets
kubectl create secret generic aster-api-keys \
    --from-literal=api-key='YOUR_ASTER_API_KEY' \
    --from-literal=api-secret='YOUR_ASTER_API_SECRET'

kubectl create secret generic gemini-api-key \
    --from-literal=api-key='YOUR_GEMINI_API_KEY'

# Deploy services
export PROJECT_ID=$GCP_PROJECT_ID
export REGION=$GCP_REGION

envsubst < cloud_deploy/k8s/sentiment_deployment.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/deployment.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/degen_deployment.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/service.yaml | kubectl apply -f -
envsubst < cloud_deploy/k8s/degen_service.yaml | kubectl apply -f -
kubectl apply -f cloud_deploy/k8s/degen_configmap.yaml

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s \
    deployment/sentiment-analyzer \
    deployment/hft-trading-agents \
    deployment/hft-degen-agent
```

---

## ✅ Verify Deployment

### Check Pod Status

```bash
# View all pods
kubectl get pods -n hft-trading

# Expected output:
# NAME                                  READY   STATUS    RESTARTS   AGE
# sentiment-analyzer-xxx                1/1     Running   0          2m
# hft-trading-agents-xxx                1/1     Running   0          2m
# hft-degen-agent-xxx                   1/1     Running   0          2m
```

### Check Logs

```bash
# Sentiment analyzer
kubectl logs -f deployment/sentiment-analyzer -n hft-trading

# Conservative agents
kubectl logs -f deployment/hft-trading-agents -n hft-trading

# Degen agent (monitor closely!)
kubectl logs -f deployment/hft-degen-agent -n hft-trading
```

### Test Endpoints

```bash
# Port forward services
kubectl port-forward svc/sentiment-service 8081:8081 -n hft-trading &
kubectl port-forward svc/hft-trading-service 8080:8080 -n hft-trading &

# Test health
curl http://localhost:8081/health
curl http://localhost:8080/health
```

---

## 🎛️ Manage Your System

### View Dashboard

```bash
# Access GCP Console
echo "https://console.cloud.google.com/kubernetes/workload?project=$GCP_PROJECT_ID"

# Or use kubectl
kubectl get all -n hft-trading
```

### Scale Services

```bash
# Scale conservative agents
kubectl scale deployment/hft-trading-agents --replicas=3 -n hft-trading

# Scale degen agent (be careful!)
kubectl scale deployment/hft-degen-agent --replicas=1 -n hft-trading
```

### Update API Keys

```bash
# Edit secrets
kubectl edit secret aster-api-keys -n hft-trading
kubectl edit secret gemini-api-key -n hft-trading

# Restart pods to pick up changes
kubectl rollout restart deployment/hft-trading-agents -n hft-trading
```

### Monitor Performance

```bash
# View metrics
kubectl top pods -n hft-trading
kubectl top nodes

# View GPU usage
kubectl describe nodes -l workload=gpu-inference | grep -A 5 "Allocated"
```

---

## 💰 Cost Management

### Estimated Monthly Costs

| Resource | Cost/Month |
|----------|-----------|
| GKE Cluster (n1-standard-4) | ~$150 |
| GPU Node (g2-standard-4 + L4) | ~$400 |
| Storage (100GB) | ~$20 |
| Network Egress | ~$15 |
| **Total** | **~$585** |

### Reduce Costs

**1. Scale down when not trading:**
```bash
# Stop GPU node pool
gcloud container node-pools update gpu-pool \
    --cluster=hft-trading-cluster \
    --region=$GCP_REGION \
    --enable-autoscaling \
    --min-nodes=0
```

**2. Use preemptible VMs (70% cheaper):**
```bash
# Add --preemptible flag when creating node pools
```

**3. Set budget alerts:**
```bash
gcloud billing budgets create \
    --billing-account=$BILLING_ACCOUNT_ID \
    --display-name="HFT Trading Budget" \
    --budget-amount=1000USD \
    --threshold-rule=percent=90
```

---

## 🚨 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n hft-trading

# Common issues:
# - Image pull errors: Check Artifact Registry permissions
# - Resource limits: Check node capacity
# - API key errors: Verify secrets
```

### GPU Not Available

```bash
# Check GPU nodes
kubectl get nodes -l workload=gpu-inference

# Check NVIDIA driver
kubectl get pods -n kube-system | grep nvidia

# Reinstall if needed
kubectl delete -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml
```

### High Latency

```bash
# Check pod location
kubectl get pods -n hft-trading -o wide

# Ensure pods are in same region as Aster DEX
# Consider using regional clusters
```

---

## 🔐 Security Checklist

- [ ] API keys stored in Kubernetes secrets
- [ ] Service account with minimal permissions
- [ ] Network policies configured
- [ ] Audit logging enabled
- [ ] Budget alerts set up
- [ ] Regular security updates scheduled

---

## 📚 Next Steps

1. **Monitor for 24 hours** with paper trading
2. **Verify all strategies** are executing correctly
3. **Check latency** is < 10ms for order execution
4. **Review logs** for any errors or warnings
5. **Enable live trading** with small capital ($50)
6. **Set up alerts** for critical metrics
7. **Schedule regular backups** of models and data

---

## 🆘 Get Help

- **Full Documentation**: See `DEPLOYMENT_GUIDE.md`
- **GCP Console**: https://console.cloud.google.com
- **Kubernetes Dashboard**: `kubectl proxy` then visit http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
- **Logs**: `kubectl logs -f deployment/<name> -n hft-trading`

---

## 🎉 Success!

Your HFT trading system is now live on Google Cloud with:
- ✅ GPU-accelerated inference
- ✅ Real-time sentiment analysis
- ✅ Multi-agent trading (conservative + degen)
- ✅ Auto-scaling and monitoring
- ✅ Enterprise-grade security

**Mission Status**: Ready to transform $50 → $500k!

---

*For detailed step-by-step instructions, see `DEPLOYMENT_GUIDE.md`*





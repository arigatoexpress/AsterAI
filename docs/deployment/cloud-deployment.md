# Cloud Deployment Guide

Deploy AsterAI to Google Cloud Platform for production-grade reliability and scalability with GPU acceleration.

## Prerequisites

### Google Cloud Account
- Active GCP account with billing enabled
- Project created in GCP Console
- gcloud CLI installed locally

### Verify GCP Setup
```bash
# Check gcloud installation
gcloud --version

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Verify project
gcloud config get-value project
```

## Architecture Overview

### GCP Services Used
- **Cloud Run**: Containerized application deployment
- **Cloud Build**: Automated container building
- **Container Registry**: Docker image storage
- **Secret Manager**: Secure API key storage
- **Cloud Storage**: Data and model storage
- **Cloud Logging**: Centralized logging

### Deployment Flow
```
Local Code → Cloud Build → Artifact Registry → GKE → Production Services
```

### Supported Deployment Methods

1. **Automated Deployment** (Recommended): Use `deploy_to_gcp.sh` for full automation
2. **Manual Deployment**: Step-by-step GCP Console deployment
3. **CI/CD Pipeline**: Automated deployments via Cloud Build triggers

## Step 1: Project Configuration

### Create GCP Project
```bash
# Create new project (or use existing)
gcloud projects create rari-trade-prod --name="Rari Trade Production"

# Set as default
gcloud config set project rari-trade-prod
```

### Enable Required APIs
```bash
# Enable Cloud Run
gcloud services enable run.googleapis.com

# Enable Container Registry
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build
gcloud services enable cloudbuild.googleapis.com

# Enable Secret Manager
gcloud services enable secretmanager.googleapis.com
```

## Step 2: Container Setup

### Dockerfile Verification
Check that `Dockerfile` exists in project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start application
CMD ["python", "scripts/deploy_live_trading.py"]
```

### Build Container Locally (Optional)
```bash
# Test build locally
docker build -t rari-trade:test .

# Test run
docker run -p 8080:8080 rari-trade:test
```

## Step 3: Secret Management

### Create Secrets in GCP
```bash
# API Keys
echo -n "your_api_key" | gcloud secrets create rari-trade-api-key --data-file=-
echo -n "your_api_secret" | gcloud secrets create rari-trade-api-secret --data-file=-

# Database credentials (if used)
echo -n "db_password" | gcloud secrets create rari-trade-db-password --data-file=-
```

### Grant Access to Secrets
```bash
# Get service account email
SERVICE_ACCOUNT=$(gcloud run services describe rari-trade-api --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || echo "YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com")

# Grant secret access
gcloud secrets add-iam-policy-binding rari-trade-api-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding rari-trade-api-secret \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"
```

## Method 1: Automated Deployment (Recommended)

### Quick Start Deployment
The easiest way to deploy AsterAI to GCP is using the automated deployment script:

```bash
# Make the script executable (first time only)
chmod +x deploy_to_gcp.sh

# Run the automated deployment
./deploy_to_gcp.sh
```

**What the script does:**
1. ✅ Checks prerequisites (gcloud, docker, kubectl)
2. ✅ Creates/configures GCP project
3. ✅ Enables required APIs
4. ✅ Creates service account with proper permissions
5. ✅ Sets up Artifact Registry for Docker images
6. ✅ Creates Cloud Storage buckets for data/models
7. ✅ Builds and pushes Docker images
8. ✅ Creates GKE cluster with GPU node pool
9. ✅ Deploys all services (trading agents, sentiment analyzer)
10. ✅ Configures monitoring and logging

**Expected deployment time:** 15-25 minutes
**Estimated monthly cost:** ~$585 (GKE + GPU + Storage)

### Manual Deployment Steps

If you prefer manual deployment or need more control:

## Step 1: Project Configuration

**Expected output:**
```
🚀 Deploying AsterAI to Google Cloud Platform
📦 Building container...
Creating temporary tarball...
Uploading tarball...
Building container image...
✅ Build successful
🚀 Creating GKE cluster...
✅ GKE cluster created
🚀 Deploying services...
✅ Services deployed successfully!
🌐 Access your deployment:
kubectl get pods -n asterai-trading
```

## Step 5: Environment Configuration

### Environment Variables
```bash
# Set production environment variables
gcloud run services update rari-trade-api \
  --set-env-vars "LOG_LEVEL=INFO" \
  --set-env-vars "MAX_POSITIONS=5" \
  --set-env-vars "RISK_PER_TRADE=0.02" \
  --set-env-vars "EMERGENCY_STOP=0.10"
```

### Resource Allocation
```bash
# Update resource limits
gcloud run services update rari-trade-api \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 20 \
  --concurrency 100
```

## Step 6: Monitoring Setup

### Enable Cloud Logging
```bash
# Logging is automatically enabled for Cloud Run
# View logs in GCP Console or via CLI
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rari-trade-api" --limit 10
```

### Set Up Alerts
```bash
# Create uptime check
gcloud monitoring uptime-check-configs create rari-trade-uptime \
  --display-name="Rari Trade API Uptime" \
  --http-check-path="/" \
  --http-check-port=8080 \
  --checked-interval=60s \
  --timeout=10s \
  --resource-type=cloud-run

# Create alert policy
gcloud monitoring alert-policies create rari-trade-alerts \
  --display-name="Rari Trade System Alerts" \
  --condition="uptime_check{rari-trade-uptime}.uptime < 0.95" \
  --notification-channels="email@example.com"
```

## Step 7: Data Storage Setup

### Create Cloud Storage Bucket
```bash
# Create bucket for data and models
gsutil mb -p rari-trade-prod -c standard gs://rari-trade-data

# Set lifecycle policy (optional)
gsutil lifecycle set lifecycle.json gs://rari-trade-data
```

### Configure Access
```bash
# Grant service account access to bucket
gcloud storage buckets add-iam-policy-binding gs://rari-trade-data \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectAdmin"
```

## Step 8: Backup and Recovery

### Automated Backups
```bash
# Create backup script
cat > scripts/backup_models.sh << 'EOF'
#!/bin/bash

# Backup models to Cloud Storage
DATE=$(date +%Y%m%d_%H%M%S)
gsutil -m cp -r models/ gs://rari-trade-data/backups/models_$DATE/

# Clean old backups (keep last 7 days)
gsutil -m rm gs://rari-trade-data/backups/models_$(date -d '7 days ago' +%Y%m%d_*) 2>/dev/null || true
EOF

chmod +x scripts/backup_models.sh
```

### Schedule Backups
```bash
# Create Cloud Scheduler job
gcloud scheduler jobs create http backup-models \
  --schedule="0 2 * * *" \
  --uri="https://rari-trade-api-REGION-PROJECT.cloudfunctions.net/backup-function" \
  --http-method=POST
```

## Step 9: Testing Production Deployment

### Health Check
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe rari-trade-api --region us-central1 --format "value(status.url)")

# Test health endpoint
curl -f $SERVICE_URL/health
```

### Load Testing
```bash
# Basic load test
for i in {1..10}; do
  curl -s $SERVICE_URL/status &
done
wait
```

### Performance Monitoring
```bash
# Check Cloud Run metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"

# View logs
gcloud logging read "resource.type=cloud_run_revision" --filter="resource.labels.service_name=rari-trade-api" --limit 50
```

## Step 10: Scaling and Optimization

### Auto-scaling Configuration
```bash
# Configure scaling based on CPU utilization
gcloud run services update rari-trade-api \
  --min-instances 1 \
  --max-instances 50 \
  --concurrency 100 \
  --cpu-throttling
```

### Cost Optimization
```bash
# Set resource limits appropriately
gcloud run services update rari-trade-api \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300

# Monitor costs
gcloud billing accounts list
```

## Troubleshooting Cloud Deployment

### Common Issues

#### Build Failures
```bash
# Check build logs
gcloud builds list --limit 5
gcloud builds log $(gcloud builds list --limit 1 --format "value(ID)")
```

#### Deployment Failures
```bash
# Check deployment status
gcloud run services describe rari-trade-api --region us-central1

# View detailed logs
gcloud logging read "resource.type=cloud_run_revision" --filter="severity>=ERROR" --limit 20
```

#### Performance Issues
```bash
# Check resource usage
gcloud monitoring metrics query \
  'fetch cloud_run_revision::run.googleapis.com/request_count' \
  --period 1h

# Scale up resources if needed
gcloud run services update rari-trade-api --memory 4Gi --cpu 2
```

#### Access Issues
```bash
# Check IAM permissions
gcloud run services get-iam-policy rari-trade-api --region us-central1

# Verify service account permissions
gcloud iam service-accounts get-iam-policy $SERVICE_ACCOUNT
```

## Cost Management

### Estimated Monthly Costs
- **Cloud Run**: $10-50 (based on usage)
- **Cloud Build**: $5-20 (build minutes)
- **Cloud Storage**: $1-10 (data storage)
- **Secret Manager**: <$1
- **Monitoring**: $5-15
- **Total**: $22-95/month

### Cost Optimization Tips
1. Use spot instances for non-critical workloads
2. Set appropriate resource limits
3. Clean up old container images
4. Monitor and optimize data storage
5. Use committed use discounts for steady usage

## Security Best Practices

### Network Security
```bash
# Enable VPC if needed
gcloud run services update rari-trade-api \
  --vpc-connector my-vpc-connector
```

### Access Control
```bash
# Restrict to specific IPs if needed
gcloud run services update rari-trade-api \
  --no-allow-unauthenticated
```

### Data Encryption
- Secrets are automatically encrypted
- Data in transit uses HTTPS
- Cloud Storage supports customer-managed encryption keys

## Post-Deployment Configuration

### Update API Keys
After deployment, you need to update the API keys in Kubernetes secrets:

```bash
# Update AsterAI API keys
kubectl edit secret aster-api-keys -n asterai-trading

# Update Gemini API key (for sentiment analysis)
kubectl edit secret gemini-api-key -n asterai-trading
```

### Verify Deployment
```bash
# Check pod status
kubectl get pods -n asterai-trading -w

# View logs
kubectl logs -f deployment/hft-trading-agents -n asterai-trading

# Port forward for local access
kubectl port-forward svc/hft-trading-service 8080:8080 -n asterai-trading
```

### Monitor Your System
```bash
# Check resource usage
kubectl top pods -n asterai-trading

# View all services
kubectl get all -n asterai-trading

# Check persistent volumes
kubectl get pvc -n asterai-trading
```

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check build logs in Cloud Build
gcloud builds list --limit 5
gcloud builds log $(gcloud builds list --limit 1 --format "value(ID)")
```

#### Pod Failures
```bash
# Describe pod for error details
kubectl describe pod POD_NAME -n asterai-trading

# Check events
kubectl get events -n asterai-trading --sort-by=.metadata.creationTimestamp
```

#### GPU Issues
```bash
# Verify GPU allocation
kubectl describe node | grep -A 10 "Capacity"

# Check GPU driver installation
kubectl logs ds/nvidia-driver-installer -n kube-system
```

#### Network Issues
```bash
# Check service endpoints
kubectl get endpoints -n asterai-trading

# Test service connectivity
kubectl run test-pod --image=curlimages/curl --rm -i --tty -- sh
```

## Cost Management

### Monthly Cost Breakdown
- **GKE Cluster**: $150 (3 nodes n1-standard-4)
- **GPU Node**: $400 (1 node with L4 GPU)
- **Storage**: $20 (Cloud Storage buckets)
- **Network**: $15 (inter-zone traffic)
- **Total**: ~$585/month

### Cost Optimization Tips
1. **Auto-scaling**: Configure appropriate min/max nodes
2. **Spot instances**: Use preemptible VMs for non-critical workloads
3. **Storage lifecycle**: Set retention policies for old data
4. **Monitoring**: Use free tier for basic monitoring

## Advanced Features

### CI/CD Pipeline
Set up automated deployments with Cloud Build triggers:

```bash
# Create build trigger
gcloud builds triggers create github \
  --name=asterai-deploy-trigger \
  --repository=owner/repo \
  --branch-pattern=main \
  --build-config=cloudbuild.yaml
```

### Multi-Region Deployment
For high availability across regions:

```bash
# Deploy to multiple regions
./deploy_to_gcp.sh --region us-west1
./deploy_to_gcp.sh --region europe-west1
```

### Database Integration
Add PostgreSQL for persistent data:

```bash
# Create Cloud SQL instance
gcloud sql instances create asterai-db \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=us-central1
```

## Security Best Practices

### Network Security
```bash
# Enable VPC-native cluster
gcloud container clusters update hft-trading-cluster \
  --enable-ip-alias

# Configure network policies
kubectl apply -f cloud_deployment/k8s/network-policy.yaml
```

### Access Control
```bash
# Create custom IAM roles
gcloud iam roles create AsterAI.Deployer \
  --project=$PROJECT_ID \
  --permissions=container.clusters.get,container.deployments.get
```

### Data Encryption
- All data in Cloud Storage is encrypted by default
- Use customer-managed encryption keys for sensitive data
- Enable VPC Service Controls for data exfiltration protection

## Support and Monitoring

### Monitoring Dashboard
Access the web dashboard at:
```
http://localhost:8080 (after port-forward)
```

### Logging
```bash
# View application logs
kubectl logs -f deployment/hft-trading-agents -n asterai-trading

# View system logs in Cloud Logging
gcloud logging read "resource.type=k8s_pod" \
  --filter="resource.labels.namespace_name=asterai-trading" \
  --limit 50
```

### Alerting
Set up alerts for:
- Pod failures or restarts
- High resource utilization
- Trading system errors
- GPU availability issues

## Next Steps

### Production Readiness Checklist
- [ ] Update all API keys in secrets
- [ ] Configure proper resource limits
- [ ] Set up monitoring and alerting
- [ ] Test end-to-end functionality
- [ ] Configure backup strategies
- [ ] Set up CI/CD pipeline
- [ ] Document operational procedures

### Performance Optimization
1. **Model Optimization**: Quantize models for faster inference
2. **Caching**: Implement Redis for frequently accessed data
3. **CDN**: Use Cloud CDN for global static content
4. **Auto-scaling**: Fine-tune HPA configurations

### Scaling Considerations
- Monitor resource usage patterns
- Scale horizontally by adding more pods
- Scale vertically by upgrading node types
- Consider serverless options for burst workloads

---

*Your AsterAI trading system is now deployed with enterprise-grade reliability, GPU acceleration, and comprehensive monitoring. Ready for ultra-performance trading operations!* 🚀📈💰

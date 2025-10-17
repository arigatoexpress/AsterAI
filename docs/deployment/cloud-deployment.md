# Cloud Deployment Guide

Deploy Rari Trade to Google Cloud Platform for production-grade reliability and scalability.

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
Local Code → Cloud Build → Container Registry → Cloud Run → Production
```

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

## Step 4: Build and Deploy

### Automated Deployment Script
Create `scripts/deploy_cloud.sh`:

```bash
#!/bin/bash

# Configuration
PROJECT_ID="rari-trade-prod"
SERVICE_NAME="rari-trade-api"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Rari Trade to Google Cloud Run"

# Build container
echo "📦 Building container..."
gcloud builds submit --tag $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 1 \
  --concurrency 80 \
  --timeout 900 \
  --set-env-vars "ENVIRONMENT=production" \
  --set-secrets "API_KEY=rari-trade-api-key:latest" \
  --set-secrets "API_SECRET=rari-trade-api-secret:latest"

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Service URL:"
    gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"
else
    echo "❌ Deployment failed"
    exit 1
fi
```

### Make Script Executable
```bash
chmod +x scripts/deploy_cloud.sh
```

### Run Deployment
```bash
./scripts/deploy_cloud.sh
```

**Expected output:**
```
🚀 Deploying Rari Trade to Google Cloud Run
📦 Building container...
Creating temporary tarball...
Uploading tarball...
Building container image...
✅ Build successful
🚀 Deploying to Cloud Run...
Deploying container to Cloud Run service [rari-trade-api]...
✅ Deployment successful!
🌐 Service URL:
https://rari-trade-api-abcdef1234-uc.a.run.app
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

## Next Steps

### Advanced Features
1. **Multi-region deployment** for high availability
2. **CI/CD pipeline** with Cloud Build triggers
3. **Database integration** with Cloud SQL
4. **Load balancing** with external HTTP load balancer

### Monitoring Enhancements
1. **Custom metrics** with Cloud Monitoring
2. **Log-based alerts** for specific events
3. **Performance dashboards** in Cloud Monitoring

---

*Your Rari Trade system is now running in the cloud with enterprise-grade reliability and scalability.*

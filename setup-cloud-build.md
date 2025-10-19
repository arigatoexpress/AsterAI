# 🚀 Cloud Build Setup Guide

## Step-by-Step Cloud Build Trigger Setup

### 1. **Access Google Cloud Console**
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Select your project
- Navigate to **Cloud Build → Triggers**

### 2. **Create New Trigger**
Click **"Create Trigger"** and configure:

#### **Basic Settings:**
```
Name: aster-trading-dashboard-deploy
Description: Deploy Aster Trading Dashboard on push to main
Region: us-central1 (or your preferred region)
```

#### **Event Configuration:**
```
Event: Push to a branch
Source: [Connect your GitHub repository]
Repository: [Your GitHub repo]
Branch: ^main$
```

#### **Configuration:**
```
Type: Cloud Build configuration file (yaml or json)
Location: Repository
Cloud Build configuration file location: cloudbuild.yaml
```

#### **Advanced Settings:**
Click **"Advanced"** and add substitution variables:
```
_REGION: us-central1
```

### 3. **Enable Required APIs**
In Google Cloud Console, go to **APIs & Services → Library** and enable:

- ✅ **Cloud Build API** (`cloudbuild.googleapis.com`)
- ✅ **Cloud Run API** (`run.googleapis.com`)
- ✅ **Container Registry API** (`containerregistry.googleapis.com`)
- ✅ **Cloud Resource Manager API** (`cloudresourcemanager.googleapis.com`)

### 4. **Set Up Service Account Permissions**
Go to **IAM & Admin → IAM** and ensure your Cloud Build service account has:

- **Cloud Run Admin** (`roles/run.admin`)
- **Storage Admin** (`roles/storage.admin`)
- **Service Account User** (`roles/iam.serviceAccountUser`)

### 5. **Test the Deployment**

#### **Option A: Automatic (Recommended)**
1. Push changes to your `main` branch
2. Cloud Build will automatically trigger
3. Monitor the build in Cloud Build console

#### **Option B: Manual Deployment**
```bash
# Set your project ID
export PROJECT_ID=your-actual-project-id

# Run the production deployment script
./deploy-production.sh
```

### 6. **Monitor Your Deployment**

#### **Check Build Status:**
- Go to **Cloud Build → History**
- Click on your build to see logs

#### **Check Cloud Run Service:**
- Go to **Cloud Run**
- Find your `aster-trading-dashboard` service
- Click on it to see details and URL

#### **Test Health Endpoints:**
Once deployed, test these endpoints:
- **Health Check**: `https://your-service-url?endpoint=health`
- **Status**: `https://your-service-url?endpoint=status`
- **Metrics**: `https://your-service-url?endpoint=metrics`

### 7. **Production Environment Variables**

Your Cloud Run service will be configured with these production settings:

```yaml
Environment Variables:
  ENVIRONMENT: production
  PORT: 8080
  DEBUG: false
  GCP_PROJECT: your-project-id
  BIGQUERY_DATASET: market_data
  ENABLE_CACHING: true
  CACHE_TTL: 300
  LOG_LEVEL: INFO
  LOG_FORMAT: json
  HEALTH_CHECK_ENABLED: true
  METRICS_ENABLED: true
  DEFAULT_STRATEGY: SMA_CROSSOVER
  MAX_POSITION_SIZE: 0.25
  DEFAULT_FEE_BPS: 5
  DEFAULT_DATA_SOURCE: SYNTHETIC
  ENABLE_BIGQUERY: true
  ENABLE_FILE_UPLOAD: true
```

### 8. **Troubleshooting**

#### **Common Issues:**

**Build Fails:**
- Check Cloud Build logs for specific errors
- Ensure all required APIs are enabled
- Verify service account permissions

**Service Won't Start:**
- Check Cloud Run logs
- Verify environment variables are set correctly
- Ensure Docker image builds successfully

**Health Check Fails:**
- Verify the service is listening on port 8080
- Check that health endpoints are working
- Review application logs for errors

#### **Useful Commands:**
```bash
# Check build status
gcloud builds list --limit=5

# View service logs
gcloud logging read 'resource.type=cloud_run_revision' --limit=50

# Get service URL
gcloud run services describe aster-trading-dashboard --region=us-central1 --format="value(status.url)"

# Test health endpoint
curl "https://your-service-url?endpoint=health"
```

### 9. **Security Recommendations**

For production deployment, consider:

- **Authentication**: Set up Cloud Run authentication
- **CORS**: Configure proper CORS origins
- **DDoS Protection**: Enable Cloud Armor
- **Monitoring**: Set up alerts and monitoring
- **Secrets**: Use Secret Manager for sensitive data
- **VPC**: Consider VPC connector for private resources

### 10. **Next Steps After Deployment**

1. **Test the dashboard** at your deployed URL
2. **Set up monitoring** and alerting
3. **Configure custom domain** (optional)
4. **Set up CI/CD** for other environments
5. **Implement backup strategies** for data

---

## 🎯 Quick Start Commands

```bash
# 1. Set your project ID
export PROJECT_ID=your-actual-project-id

# 2. Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# 3. Deploy manually (if needed)
./deploy-production.sh

# 4. Check deployment
gcloud run services list --region=us-central1
```

Your Aster Trading Dashboard will be automatically deployed whenever you push to the main branch! 🚀

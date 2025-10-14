# 🚀 AI Trading System Setup Instructions

## Step 1: Configure Your Credentials

**IMPORTANT**: Before proceeding, you need to add your Aster DEX API credentials.

### Edit the `.env` file:

```bash
nano .env
```

Add your credentials:
```
ASTER_API_KEY=your_actual_api_key_here
ASTER_API_SECRET=your_actual_api_secret_here
GCP_PROJECT_ID=your_gcp_project_id_here
```

**Security Note**: The `.env` file is already in `.gitignore` and will NOT be committed to version control.

## Step 2: Verify Credentials

Once you've added your credentials, run:

```bash
python scripts/setup_credentials.py
```

This will:
- Validate your Aster DEX API credentials
- Test API connectivity
- Check account permissions
- Verify balance and trading capabilities

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This may take 10-15 minutes as it installs:
- PyTorch (deep learning)
- Stable-Baselines3 (reinforcement learning)
- Google Cloud libraries
- Trading and analytics libraries

## Step 4: Run Unit Tests

```bash
pytest tests/ -v
```

This validates all core components are working correctly.

## Step 5: Run Backtest Simulation

```bash
python run_complete_system.py backtest
```

This will simulate 2 years of trading and show if the $10k → $1M target is achievable.

## Step 6: Setup GCP (If Deploying to Cloud)

```bash
# Enable required APIs
gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com run.googleapis.com secretmanager.googleapis.com

# Setup BigQuery
python scripts/setup_gcp.py YOUR_PROJECT_ID

# Deploy secrets
echo -n "YOUR_API_KEY" | gcloud secrets create aster-api-key --data-file=-
echo -n "YOUR_API_SECRET" | gcloud secrets create aster-api-secret --data-file=-
```

## Step 7: Deploy to Cloud Run (Production)

```bash
# Build and deploy
docker build -t ai-trading-system .
docker tag ai-trading-system gcr.io/YOUR_PROJECT_ID/ai-trading-system
docker push gcr.io/YOUR_PROJECT_ID/ai-trading-system
gcloud run deploy ai-trading-system --image gcr.io/YOUR_PROJECT_ID/ai-trading-system --platform managed
```

## Next Steps

You're now ready! Choose your mode:

- **Paper Trading**: `python run_complete_system.py paper`
- **Live Trading**: `python run_complete_system.py live` (CAUTION!)

Monitor the dashboard at: `streamlit run dashboard/app.py`


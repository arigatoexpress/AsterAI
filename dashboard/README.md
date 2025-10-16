# Aster AI Trading Dashboard

A unified web dashboard for monitoring and managing the complete AI trading system.

## Features

### 🏠 Overview
- System health metrics (CPU, memory, bot status)
- Active projects progress tracking
- Recent activity timeline

### ☁️ Cloud Deployment
- GCP credentials and API status
- Kubernetes manifests validation
- Docker images availability
- GKE cluster monitoring
- Vertex AI model deployment
- Cloud Run services status

### 💻 Local Development
- AI training status and accuracy
- Model performance metrics
- Training result visualizations
- Development tools status

### 📈 Trading Performance
- Real-time bot status and P&L
- Equity curve visualization
- Trade history and statistics
- Risk management dashboard

### 🤖 AI Models
- Model comparison (Ensemble, XGBoost, etc.)
- Feature importance analysis
- Prediction confidence distributions
- Training metrics overview

### ⚡ Extreme Growth
- $150→$1M strategy progress
- Growth milestone tracking
- Risk management parameters
- Projected growth paths

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run unified_trading_dashboard.py

# Access at: http://localhost:8501
```

### GCP Deployment
```bash
# Make script executable
chmod +x deploy_to_gcp.sh

# Deploy to Cloud Run
./deploy_to_gcp.sh

# Get public URL from output
```

### Docker Deployment
```bash
# Build image
docker build -t aster-dashboard .

# Run container
docker run -p 8501:8080 aster-dashboard

# Access at: http://localhost:8501
```

## Prerequisites

### Local Development
- Python 3.11+
- Streamlit, Plotly, Pandas
- Access to training data and models

### GCP Deployment
- Google Cloud account
- gcloud CLI installed
- Project with billing enabled
- Required APIs enabled:
  - Cloud Run
  - Container Registry
  - Cloud Build

## Configuration

### Environment Variables
```bash
# For cloud deployment
ENVIRONMENT=CLOUD
GCP_PROJECT=your-project-id

# For local development
ENVIRONMENT=LOCAL
```

## Architecture

```
dashboard/
├── unified_trading_dashboard.py  # Main Streamlit app
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
├── deploy_to_gcp.sh            # GCP deployment script
└── README.md                   # This file
```

## Data Sources

- **System Metrics**: psutil for CPU/memory monitoring
- **Training Results**: JSON/metadata from training runs
- **Trading Performance**: Bot logs and performance data
- **Cloud Status**: GCP API calls for resource status
- **AI Models**: Pickled models and feature data

## Security

- Environment-based access control
- No sensitive data exposed
- Ready for OAuth authentication
- Secure API key handling

## Monitoring

- Real-time system health checks
- Trading bot status monitoring
- Error logging and alerts
- Performance metrics tracking

## Contributing

1. Follow the existing code structure
2. Add new pages using the sidebar pattern
3. Include proper error handling
4. Update documentation

## Support

- Check logs in `../logs/` directory
- View training results in `../training_results/`
- Monitor bot status in real-time
- Access all visualizations and metrics

---

Built with Aster AI Trading System - October 2025

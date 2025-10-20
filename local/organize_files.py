#!/usr/bin/env python3
"""
Organize AsterAI files into cloud vs local categories
"""

import os
import shutil
from pathlib import Path

def organize_files():
    """Organize files into cloud vs local categories"""

    print("📁 Organizing AsterAI Files")
    print("="*50)

    # Define file categories
    cloud_files = [
        # Cloud Build & Deployment
        'cloudbuild.yaml',
        'cloudbuild_autonomous.yaml',
        'cloudbuild_dashboard.yaml',
        'cloudbuild_self_learning.yaml',
        'cloud-run-env-vars.yaml',

        # Docker files
        'Dockerfile',
        'Dockerfile.autonomous',
        'Dockerfile.dashboard',
        'Dockerfile.gpu',
        'Dockerfile.paper_trading',
        'Dockerfile.self_learning',
        'Dockerfile.sentiment',

        # Shell deployment scripts
        'deploy_all_systems.sh',
        'deploy_autonomous_system.sh',
        'deploy_dashboard_gpu.sh',
        'deploy_paper_trading.sh',
        'deploy_production.py',
        'deploy_profit_maximizer_*.sh',
        'deploy_to_gcp.sh',
        'deploy-production.sh',

        # Kubernetes
        'k8s_paper_trading.yaml',

        # Cloud-specific directories
        'cloud_architecture',
        'cloud_deployment',
        'cloud_functions',
        'services',
        'vertex',
    ]

    local_files = [
        # Development scripts
        'scripts',
        'tests',

        # Windows batch files
        'launch_dashboard.bat',
        'launch_master_dashboard.bat',
        'quick_deploy.bat',
        'start_services.bat',
        'stop_services.bat',
        'train_model.ps1',

        # Local environment and config
        'env.example',
        'requirements.txt',
        'requirements-gpu.txt',
        'requirements_trading.txt',
        'pyproject.toml',
        'secrets-config.yaml',

        # Local development files
        'local_test_results.txt',
        'local_training',
        'TEST_DATA_ARCHIVE',

        # API and credential templates
        'API_CREDENTIALS_TEMPLATE.txt',
        '.api_keys.json',

        # Development tools and experiments
        'simple_gpu_experiments.py',
        'rtx_trading_edge_demo.py',
        'gpu_data_science_lab.py',
        'gpu_benchmark_suite.py',
        'gpu_comprehensive_test.py',
        'gpu_strategy_comparison.py',
        'organize_files.py',  # This script itself
    ]

    shared_files = [
        # Core application code (works in both environments)
        'mcp_trader',
        'config',
        'data',
        'models',
        'dashboard',
        'dashboard-next',
        'modern_frontend',

        # Main application files
        'autonomous_data_pipeline.py',
        'autonomous_mcp_agent.py',
        'live_trading_agent.py',
        'paper_trading_system.py',
        'self_learning_trader.py',
        'trading_server.py',
        'enhanced_trading_server.py',
        'self_learning_server.py',
        'telegram_bot.py',
        'market_regime_detector.py',

        # Analysis and reporting
        'comprehensive_trading_analysis.py',
        'trading_data_analysis.py',
        'trading_analysis_reports',
        'COMPREHENSIVE_REPORT.md',
        'SUMMARY_OF_WORK_COMPLETED.md',
        'PROJECT_COMPLETION_PLAN.md',
        'PROJECT_PROGRESS_OUTLINE.md',

        # Core configuration and requirements
        'config-gpu.yaml',
        'dashboard_requirements.txt',
        'run_dashboard.py',

        # Documentation
        'README.md',
        'README_DASHBOARD.md',
        'LICENSE',

        # Other core files
        'profit_maximization_strategy.py',
        'mev_protection_system.py',
        'self_improvement_engine.py',
        'monitoring',
        'logs',
        'backtest_results',
        'results',
        'training_results',
        'gpu_benchmarks_20251019_144855',
        'benchmarks',
        'visual_reports',
        'trading_analysis_reports',
    ]

    # Create organization directories
    cloud_dir = Path('cloud')
    local_dir = Path('local')
    shared_dir = Path('.')  # Root directory for shared files

    cloud_dir.mkdir(exist_ok=True)
    local_dir.mkdir(exist_ok=True)

    print("🔄 Moving files to organized structure...")

    # Move cloud files
    print("\n☁️  Moving cloud-specific files...")
    for file_pattern in cloud_files:
        if '*' in file_pattern:
            # Handle wildcards
            for file_path in Path('.').glob(file_pattern):
                if file_path.is_file():
                    dest = cloud_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                    print(f"   Moved: {file_path.name} → cloud/")
        else:
            file_path = Path(file_pattern)
            if file_path.exists():
                dest = cloud_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                print(f"   Moved: {file_path.name} → cloud/")

    # Move local files
    print("\n💻 Moving local-specific files...")
    for file_pattern in local_files:
        if '*' in file_pattern:
            # Handle wildcards
            for file_path in Path('.').glob(file_pattern):
                if file_path.is_file():
                    dest = local_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                    print(f"   Moved: {file_path.name} → local/")
        else:
            file_path = Path(file_pattern)
            if file_path.exists() and file_path.is_file():
                dest = local_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                print(f"   Moved: {file_path.name} → local/")

    # Verify shared files are still in place
    print("\n✅ Verifying shared files...")
    for file_pattern in shared_files:
        file_path = Path(file_pattern)
        if file_path.exists():
            print(f"   Kept: {file_pattern} (shared)")
        else:
            print(f"   ⚠️  Missing shared file: {file_pattern}")

    # Create organization documentation
    create_organization_docs()

    print("\n" + "="*50)
    print("✅ File Organization Complete!")
    print("="*50)
    print("\n📂 New Structure:")
    print("☁️  cloud/     - Cloud deployment files (Docker, K8s, scripts)")
    print("💻 local/     - Local development files (scripts, tests, configs)")
    print("📁 ./         - Shared application code (mcp_trader, config, etc.)")
    print("\n📖 See ORGANIZATION_README.md for details")

def create_organization_docs():
    """Create documentation for the file organization"""

    docs = """# AsterAI File Organization Guide

## Overview
This document explains how AsterAI files are organized for cloud vs local development.

## Directory Structure

```
AsterAI/
├── cloud/              # Cloud-specific files
│   ├── cloudbuild_*.yaml    # Cloud Build configurations
│   ├── Dockerfile*          # Docker container definitions
│   ├── deploy_*.sh          # Shell deployment scripts
│   ├── k8s_*.yaml          # Kubernetes manifests
│   ├── services/           # Microservices architecture
│   └── vertex/             # Google Vertex AI components
│
├── local/              # Local development files
│   ├── scripts/            # Development and testing scripts
│   ├── tests/              # Unit and integration tests
│   ├── *.bat               # Windows batch files
│   ├── requirements*.txt   # Python dependencies
│   └── .api_keys.json      # Local API credentials
│
└── [shared files]      # Core application code
    ├── mcp_trader/         # Main trading engine
    ├── config/             # Configuration files
    ├── models/             # ML models
    ├── dashboard/          # Web dashboards
    └── data/               # Data processing
```

## File Categories

### Cloud-Specific Files (`cloud/`)
These files are primarily used for cloud deployment and production environments:

- **Docker files**: Container definitions for different services
- **Cloud Build configs**: CI/CD pipeline configurations
- **Deployment scripts**: Automated deployment tools
- **Kubernetes manifests**: Container orchestration
- **Cloud service configs**: GCP, AWS, Azure specific configurations

### Local-Specific Files (`local/`)
These files are primarily used for local development and testing:

- **Development scripts**: Testing, debugging, and utility scripts
- **Test files**: Unit tests, integration tests
- **Local configs**: Environment variables, API keys
- **Batch files**: Windows-specific automation scripts
- **Development tools**: Profiling, benchmarking tools

### Shared Files (root directory)
These files work in both cloud and local environments:

- **Core application**: `mcp_trader/`, trading algorithms, ML models
- **Configuration**: Settings that adapt to environment
- **Data processing**: Code that works with different data sources
- **Dashboards**: Web interfaces that run locally or in containers

## Usage Guidelines

### For Local Development
1. Use files in the `local/` directory for development tools
2. Keep API keys in `local/.api_keys.json`
3. Use `local/scripts/` for development utilities
4. Run tests from `local/tests/`

### For Cloud Deployment
1. Use files in the `cloud/` directory for deployment
2. Copy configurations from `local/` to `cloud/` as needed
3. Modify Docker files in `cloud/` for production requirements
4. Use cloud-specific deployment scripts

### Environment-Specific Configuration
- **Local**: Use `.env` files and local configurations
- **Cloud**: Use environment variables and cloud-specific configs
- **Shared**: Use configuration files that adapt to environment

## Adding New Files

When adding new files, consider:

1. **Is it cloud-specific?** → Put in `cloud/`
2. **Is it local development only?** → Put in `local/`
3. **Does it work in both environments?** → Keep in root

## Migration Notes

- Files were moved during organization - check git history if needed
- Some files may need path updates after reorganization
- Update any hardcoded paths in code to use relative paths
- Test both local and cloud deployments after reorganization

## Troubleshooting

- **Missing file?** Check if it was moved to `cloud/` or `local/`
- **Import errors?** Update import paths for moved modules
- **Configuration issues?** Ensure correct config files are in place
"""

    with open('ORGANIZATION_README.md', 'w') as f:
        f.write(docs)

    print("📖 Created ORGANIZATION_README.md")

if __name__ == "__main__":
    organize_files()

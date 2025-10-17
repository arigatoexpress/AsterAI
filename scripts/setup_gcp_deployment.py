"""
GCP Deployment Setup Script

Complete GCP deployment setup for the trading system:
- Docker containerization
- Kubernetes manifests
- Cloud Build CI/CD pipeline
- Cloud Run service deployment
- Secret Manager integration
- Cloud Storage setup
- Monitoring and logging setup

Features:
- Automated deployment pipeline
- Environment-specific configurations
- Security best practices
- Scalable architecture
- Cost optimization
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_trader.config import GCPConfig


class GCPDeploymentSetup:
    """
    Complete GCP deployment setup and management
    """

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.config = GCPConfig(
            project_id=project_id,
            region=region
        )

        # Setup directories
        self.root_dir = Path(__file__).parent.parent
        self.gcp_dir = self.root_dir / "gcp_deployment"
        self.gcp_dir.mkdir(exist_ok=True)

    def setup_complete_deployment(self):
        """Setup complete GCP deployment infrastructure"""

        print("🚀 Setting up complete GCP deployment for Aster AI Trading System")
        print("=" * 80)

        try:
            # 1. Enable required APIs
            self.enable_gcp_apis()

            # 2. Create service account
            self.create_service_account()

            # 3. Setup Secret Manager
            self.setup_secret_manager()

            # 4. Create Cloud Storage buckets
            self.create_storage_buckets()

            # 5. Build Docker images
            self.build_docker_images()

            # 6. Setup Cloud Run services
            self.setup_cloud_run()

            # 7. Setup Cloud Build pipeline
            self.setup_cloud_build()

            # 8. Setup monitoring
            self.setup_monitoring()

            # 9. Deploy application
            self.deploy_application()

            print("\n✅ GCP deployment setup completed successfully!")
            print("\n📋 Next steps:")
            print("1. Update your domain DNS settings")
            print("2. Configure monitoring alerts")
            print("3. Set up backup procedures")
            print("4. Test the deployment")

        except Exception as e:
            print(f"\n❌ Deployment setup failed: {str(e)}")
            raise

    def enable_gcp_apis(self):
        """Enable required GCP APIs"""

        print("\n🔧 Enabling GCP APIs...")

        apis = [
            "run.googleapis.com",
            "secretmanager.googleapis.com",
            "storage-api.googleapis.com",
            "cloudbuild.googleapis.com",
            "monitoring.googleapis.com",
            "logging.googleapis.com",
            "containerregistry.googleapis.com",
            "cloudresourcemanager.googleapis.com"
        ]

        for api in apis:
            print(f"  Enabling {api}...")
            self.run_gcloud_command([
                "services", "enable", api,
                f"--project={self.project_id}"
            ])

    def create_service_account(self):
        """Create GCP service account for the application"""

        print("\n🔐 Creating service account...")

        sa_name = "aster-trading-sa"
        sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"

        # Create service account
        self.run_gcloud_command([
            "iam", "service-accounts", "create", sa_name,
            f"--description=Aster AI Trading System Service Account",
            f"--display-name=Aster Trading SA",
            f"--project={self.project_id}"
        ])

        # Grant necessary permissions
        roles = [
            "roles/secretmanager.secretAccessor",
            "roles/storage.objectAdmin",
            "roles/monitoring.metricWriter",
            "roles/logging.logWriter",
            "roles/run.invoker"
        ]

        for role in roles:
            print(f"  Granting {role}...")
            self.run_gcloud_command([
                "projects", "add-iam-policy-binding", self.project_id,
                f"--member=serviceAccount:{sa_email}",
                f"--role={role}"
            ])

        # Create and download key
        key_file = self.gcp_dir / "service-account-key.json"
        self.run_gcloud_command([
            "iam", "service-accounts", "keys", "create", str(key_file),
            f"--iam-account={sa_email}"
        ])

        print(f"  Service account key saved to: {key_file}")

    def setup_secret_manager(self):
        """Setup Secret Manager for sensitive configuration"""

        print("\n🔒 Setting up Secret Manager...")

        secrets = {
            "api-keys": {
                "description": "API keys for exchanges and data providers",
                "data": {
                    "binance_api_key": "",
                    "binance_secret_key": "",
                    "aster_api_key": "",
                    "aster_secret_key": ""
                }
            },
            "database-config": {
                "description": "Database connection configuration",
                "data": {
                    "db_host": "",
                    "db_port": "5432",
                    "db_name": "aster_trading",
                    "db_user": "",
                    "db_password": ""
                }
            },
            "trading-config": {
                "description": "Trading system configuration",
                "data": {
                    "max_position_size": "0.1",
                    "risk_limit": "0.05",
                    "trading_enabled": "false"
                }
            }
        }

        for secret_name, secret_config in secrets.items():
            # Create secret
            self.run_gcloud_command([
                "secrets", "create", secret_name,
                f"--data-file=/dev/stdin",
                f"--project={self.project_id}"
            ], input=json.dumps(secret_config["data"], indent=2))

            print(f"  Created secret: {secret_name}")

    def create_storage_buckets(self):
        """Create Cloud Storage buckets for data and models"""

        print("\n📦 Creating Cloud Storage buckets...")

        buckets = [
            f"{self.project_id}-aster-data",
            f"{self.project_id}-aster-models",
            f"{self.project_id}-aster-logs",
            f"{self.project_id}-aster-backups"
        ]

        for bucket in buckets:
            print(f"  Creating bucket: {bucket}")
            self.run_gcloud_command([
                "storage", "buckets", "create",
                f"gs://{bucket}",
                f"--project={self.project_id}",
                f"--location={self.region}"
            ])

    def build_docker_images(self):
        """Build and push Docker images"""

        print("\n🐳 Building Docker images...")

        # Build trading system image
        image_name = f"gcr.io/{self.project_id}/aster-trading-system"
        dockerfile_path = self.root_dir / "Dockerfile"

        if dockerfile_path.exists():
            print(f"  Building image: {image_name}")
            self.run_command([
                "docker", "build", "-t", image_name, "-f", str(dockerfile_path), "."
            ])

            print(f"  Pushing image: {image_name}")
            self.run_command(["docker", "push", image_name])
        else:
            print("  Dockerfile not found, skipping Docker build")

    def setup_cloud_run(self):
        """Setup Cloud Run services"""

        print("\n☁️ Setting up Cloud Run services...")

        services = {
            "aster-trading-api": {
                "image": f"gcr.io/{self.project_id}/aster-trading-system",
                "port": 8080,
                "memory": "2Gi",
                "cpu": "1000m",
                "max_instances": 10,
                "concurrency": 80,
                "env_vars": {
                    "ENVIRONMENT": "production",
                    "PROJECT_ID": self.project_id,
                    "REGION": self.region
                }
            }
        }

        for service_name, config in services.items():
            print(f"  Deploying service: {service_name}")

            cmd = [
                "run", "deploy", service_name,
                f"--image={config['image']}",
                f"--platform=managed",
                f"--region={self.region}",
                f"--project={self.project_id}",
                f"--port={config['port']}",
                f"--memory={config['memory']}",
                f"--cpu={config['cpu']}",
                f"--max-instances={config['max_instances']}",
                f"--concurrency={config['concurrency']}",
                "--allow-unauthenticated"
            ]

            # Add environment variables
            for key, value in config['env_vars'].items():
                cmd.extend(["--set-env-vars", f"{key}={value}"])

            self.run_gcloud_command(cmd)

    def setup_cloud_build(self):
        """Setup Cloud Build CI/CD pipeline"""

        print("\n🔄 Setting up Cloud Build pipeline...")

        # Create cloudbuild.yaml
        cloudbuild_config = {
            "steps": [
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": [
                        "build",
                        "-t",
                        f"gcr.io/{self.project_id}/aster-trading-system:$COMMIT_SHA",
                        "."
                    ]
                },
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": [
                        "push",
                        f"gcr.io/{self.project_id}/aster-trading-system:$COMMIT_SHA"
                    ]
                },
                {
                    "name": "gcr.io/cloud-builders/gcloud",
                    "args": [
                        "run",
                        "deploy",
                        "aster-trading-api",
                        "--image",
                        f"gcr.io/{self.project_id}/aster-trading-system:$COMMIT_SHA",
                        "--region",
                        self.region,
                        "--platform",
                        "managed",
                        "--allow-unauthenticated"
                    ]
                }
            ],
            "images": [
                f"gcr.io/{self.project_id}/aster-trading-system:$COMMIT_SHA"
            ]
        }

        cloudbuild_file = self.root_dir / "cloudbuild.yaml"
        with open(cloudbuild_file, 'w') as f:
            yaml.dump(cloudbuild_config, f, default_flow_style=False)

        print(f"  Created cloudbuild.yaml: {cloudbuild_file}")

        # Create build trigger (manual setup required)
        print("  Note: Build triggers need to be configured manually in GCP Console")

    def setup_monitoring(self):
        """Setup monitoring and alerting"""

        print("\n📊 Setting up monitoring...")

        # Create monitoring dashboard config
        dashboard_config = {
            "displayName": "Aster Trading System Dashboard",
            "mosaicLayout": {
                "columns": 12,
                "tiles": [
                    {
                        "height": 4,
                        "width": 6,
                        "title": "Portfolio Value",
                        "xyChart": {
                            "dataSets": [{
                                "timeSeriesFilter": {
                                    "filter": "metric.type=\"custom.googleapis.com/portfolio_value\""
                                }
                            }]
                        }
                    },
                    {
                        "height": 4,
                        "width": 6,
                        "title": "Trade Performance",
                        "xyChart": {
                            "dataSets": [{
                                "timeSeriesFilter": {
                                    "filter": "metric.type=\"custom.googleapis.com/trade_pnl\""
                                }
                            }]
                        }
                    }
                ]
            }
        }

        dashboard_file = self.gcp_dir / "monitoring-dashboard.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)

        print(f"  Created monitoring dashboard config: {dashboard_file}")

    def deploy_application(self):
        """Deploy the complete application"""

        print("\n🚀 Deploying Aster AI Trading System...")

        # Create deployment manifest
        deployment_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "aster-config"
            },
            "data": {
                "config.json": json.dumps({
                    "environment": "production",
                    "project_id": self.project_id,
                    "region": self.region,
                    "trading_enabled": False,
                    "max_risk": 0.05
                }, indent=2)
            }
        }

        k8s_file = self.gcp_dir / "deployment.yaml"
        with open(k8s_file, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False)

        print(f"  Created deployment manifest: {k8s_file}")

        # Final deployment steps
        print("\n📋 Deployment Summary:")
        print(f"   Project ID: {self.project_id}")
        print(f"   Region: {self.region}")
        print(f"   API URL: https://{self.region}-run.googleapis.com"        print(f"   Service Account: aster-trading-sa@{self.project_id}.iam.gserviceaccount.com"
    def run_gcloud_command(self, args: List[str], input: Optional[str] = None):
        """Run gcloud command with error handling"""

        cmd = ["gcloud"] + args

        try:
            result = subprocess.run(
                cmd,
                input=input,
                text=True,
                capture_output=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ gcloud command failed: {' '.join(cmd)}")
            print(f"   Error: {e.stderr}")
            raise

    def run_command(self, args: List[str], input: Optional[str] = None):
        """Run general command with error handling"""

        try:
            result = subprocess.run(
                args,
                input=input,
                text=True,
                capture_output=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(args)}")
            print(f"   Error: {e.stderr}")
            raise

    def create_deployment_scripts(self):
        """Create deployment helper scripts"""

        scripts = {
            "deploy.sh": """#!/bin/bash
set -e

echo "🚀 Deploying Aster AI Trading System to GCP"

# Build and push Docker image
docker build -t gcr.io/$PROJECT_ID/aster-trading-system .
docker push gcr.io/$PROJECT_ID/aster-trading-system

# Deploy to Cloud Run
gcloud run deploy aster-trading-api \\
  --image gcr.io/$PROJECT_ID/aster-trading-system \\
  --platform managed \\
  --region $REGION \\
  --allow-unauthenticated \\
  --set-env-vars ENVIRONMENT=production

echo "✅ Deployment completed!"
""",
            "setup-gcp.sh": """#!/bin/bash
set -e

echo "🔧 Setting up GCP infrastructure for Aster AI Trading System"

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable storage-api.googleapis.com

# Create service account
gcloud iam service-accounts create aster-trading-sa \\
  --description="Aster AI Trading System Service Account" \\
  --display-name="Aster Trading SA"

echo "✅ GCP setup completed!"
"""
        }

        for script_name, content in scripts.items():
            script_path = self.gcp_dir / script_name
            with open(script_path, 'w') as f:
                f.write(content)

            # Make executable
            os.chmod(script_path, 0o755)

        print(f"  Created deployment scripts in: {self.gcp_dir}")


def main():
    """Main deployment setup function"""

    parser = argparse.ArgumentParser(description="Setup GCP deployment for Aster AI Trading System")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--setup-only", action="store_true", help="Only setup infrastructure, don't deploy")

    args = parser.parse_args()

    # Setup deployment
    setup = GCPDeploymentSetup(args.project_id, args.region)

    if args.setup_only:
        print("🔧 Setting up GCP infrastructure only...")
        setup.enable_gcp_apis()
        setup.create_service_account()
        setup.setup_secret_manager()
        setup.create_storage_buckets()
        setup.create_deployment_scripts()
    else:
        setup.setup_complete_deployment()

    print("\n🎉 GCP deployment setup completed!")


if __name__ == "__main__":
    main()

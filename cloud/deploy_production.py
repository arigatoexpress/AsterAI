#!/usr/bin/env python3
"""
AsterAI HFT Trading System - Production Deployment Script

Comprehensive deployment automation for cloud production environment.
Deploys the complete HFT trading system with security, monitoring, and scalability.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Production deployment orchestrator for AsterAI"""

    def __init__(self, config_file: str = "deploy_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.project_root = Path(__file__).parent

    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)

        # Default configuration
        return {
            "project": {
                "name": "asterai-hft-trader",
                "version": "2.0.0",
                "environment": "production"
            },
            "cloud": {
                "provider": "gcp",
                "project_id": "asterai-hft-trader-2024",
                "region": "us-central1",
                "zone": "us-central1-a"
            },
            "infrastructure": {
                "cluster_name": "asterai-hft-cluster",
                "node_pool": {
                    "machine_type": "n1-standard-4",
                    "min_nodes": 3,
                    "max_nodes": 10,
                    "gpu_type": "nvidia-tesla-t4",
                    "gpu_count": 1
                }
            },
            "services": {
                "hft_trader": {
                    "replicas": 1,
                    "cpu_limit": "2",
                    "memory_limit": "4Gi",
                    "gpu_limit": 1
                },
                "degen_trader": {
                    "replicas": 1,
                    "cpu_limit": "2",
                    "memory_limit": "4Gi",
                    "gpu_limit": 1
                },
                "sentiment_analyzer": {
                    "replicas": 1,
                    "cpu_limit": "500m",
                    "memory_limit": "1Gi"
                },
                "monitoring": {
                    "replicas": 1,
                    "cpu_limit": "500m",
                    "memory_limit": "1Gi"
                }
            },
            "security": {
                "enable_https": True,
                "ssl_cert": "letsencrypt",
                "firewall_rules": True,
                "iam_roles": True
            },
            "monitoring": {
                "prometheus": True,
                "grafana": True,
                "alertmanager": True,
                "log_aggregation": True
            },
            "scaling": {
                "hpa_enabled": True,
                "cpu_target": 70,
                "memory_target": 80,
                "max_replicas": 5
            }
        }

    def save_config(self):
        """Save deployment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {self.config_file}")

    def run_deployment(self):
        """Execute complete production deployment"""
        logger.info("🚀 Starting AsterAI Production Deployment")
        logger.info("=" * 60)

        try:
            # Pre-deployment checks
            self.pre_deployment_checks()

            # Infrastructure setup
            self.setup_infrastructure()

            # Build and push images
            self.build_and_push_images()

            # Deploy services
            self.deploy_services()

            # Configure monitoring
            self.setup_monitoring()

            # Security hardening
            self.configure_security()

            # Post-deployment validation
            self.post_deployment_validation()

            # Final status
            self.deployment_complete()

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            self.rollback_deployment()
            sys.exit(1)

    def pre_deployment_checks(self):
        """Perform pre-deployment validation checks"""
        logger.info("🔍 Running pre-deployment checks...")

        # Check required tools
        required_tools = ['docker', 'kubectl', 'gcloud']
        for tool in required_tools:
            if not self.check_tool_available(tool):
                raise Exception(f"Required tool not found: {tool}")

        # Check GCP authentication
        if not self.check_gcp_auth():
            raise Exception("GCP authentication required")

        # Check required environment variables
        required_env_vars = [
            'ASTER_API_KEY', 'ASTER_API_SECRET',
            'GEMINI_API_KEY', 'GCP_PROJECT_ID'
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise Exception(f"Missing required environment variables: {missing_vars}")

        # Validate configuration
        self.validate_config()

        logger.info("✅ Pre-deployment checks passed")

    def check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available in PATH"""
        try:
            subprocess.run([tool, '--version'],
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def check_gcp_auth(self) -> bool:
        """Check GCP authentication"""
        try:
            result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'],
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip()) > 0
        except subprocess.CalledProcessError:
            return False

    def validate_config(self):
        """Validate deployment configuration"""
        config = self.config

        # Validate project settings
        if not config['project']['name']:
            raise Exception("Project name is required")

        # Validate cloud settings
        if not config['cloud']['project_id']:
            raise Exception("GCP project ID is required")

        # Validate infrastructure
        infra = config['infrastructure']
        if infra['node_pool']['min_nodes'] > infra['node_pool']['max_nodes']:
            raise Exception("Min nodes cannot exceed max nodes")

        logger.info("✅ Configuration validation passed")

    def setup_infrastructure(self):
        """Set up cloud infrastructure"""
        logger.info("🏗️  Setting up cloud infrastructure...")

        config = self.config
        cloud_config = config['cloud']
        infra_config = config['infrastructure']

        # Set GCP project
        self.run_command([
            'gcloud', 'config', 'set', 'project', cloud_config['project_id']
        ])

        # Set region and zone
        self.run_command([
            'gcloud', 'config', 'set', 'compute/region', cloud_config['region']
        ])
        self.run_command([
            'gcloud', 'config', 'set', 'compute/zone', cloud_config['zone']
        ])

        # Create GKE cluster
        cluster_name = infra_config['cluster_name']
        node_pool = infra_config['node_pool']

        logger.info(f"Creating GKE cluster: {cluster_name}")

        cluster_cmd = [
            'gcloud', 'container', 'clusters', 'create', cluster_name,
            '--region', cloud_config['region'],
            '--num-nodes', str(node_pool['min_nodes']),
            '--machine-type', node_pool['machine_type'],
            '--enable-autoscaling',
            '--min-nodes', str(node_pool['min_nodes']),
            '--max-nodes', str(node_pool['max_nodes'])
        ]

        # Add GPU support if specified
        if node_pool.get('gpu_type'):
            cluster_cmd.extend([
                '--accelerator', f'type={node_pool["gpu_type"]},count={node_pool["gpu_count"]}'
            ])

        self.run_command(cluster_cmd)

        # Get cluster credentials
        self.run_command([
            'gcloud', 'container', 'clusters', 'get-credentials', cluster_name,
            '--region', cloud_config['region']
        ])

        # Create GPU node pool if needed
        if node_pool.get('gpu_type'):
            self.run_command([
                'gcloud', 'container', 'node-pools', 'create', 'gpu-pool',
                '--cluster', cluster_name,
                '--region', cloud_config['region'],
                '--machine-type', node_pool['machine_type'],
                '--accelerator', f'type={node_pool["gpu_type"]},count={node_pool["gpu_count"]}',
                '--num-nodes', '0',
                '--enable-autoscaling',
                '--min-nodes', '0',
                '--max-nodes', '3'
            ])

        logger.info("✅ Infrastructure setup complete")

    def build_and_push_images(self):
        """Build and push Docker images"""
        logger.info("🐳 Building and pushing Docker images...")

        config = self.config
        project_id = config['cloud']['project_id']
        region = config['cloud']['region']

        # Set Google Container Registry
        registry = f"{region}-docker.pkg.dev/{project_id}/asterai"

        images = {
            'hft-trader': 'Dockerfile.gpu',
            'degen-trader': 'Dockerfile.gpu',
            'sentiment-analyzer': 'Dockerfile.sentiment',
            'monitoring': 'Dockerfile'
        }

        for service, dockerfile in images.items():
            logger.info(f"Building {service} image...")

            # Build image
            self.run_command([
                'docker', 'build', '-f', dockerfile,
                '-t', f"{registry}/{service}:v{config['project']['version']}",
                '--build-arg', f"ENVIRONMENT={config['project']['environment']}",
                '.'
            ])

            # Push image
            self.run_command([
                'docker', 'push', f"{registry}/{service}:v{config['project']['version']}"
            ])

        logger.info("✅ Image build and push complete")

    def deploy_services(self):
        """Deploy Kubernetes services"""
        logger.info("🚢 Deploying Kubernetes services...")

        config = self.config
        services = config['services']

        # Apply Kubernetes manifests
        k8s_files = [
            'cloud_deploy/k8s/namespace.yaml',
            'cloud_deploy/k8s/configmaps.yaml',
            'cloud_deploy/k8s/secrets.yaml',
            'cloud_deploy/k8s/services.yaml',
            'cloud_deploy/k8s/deployments.yaml',
            'cloud_deploy/k8s/ingress.yaml'
        ]

        for k8s_file in k8s_files:
            if os.path.exists(k8s_file):
                logger.info(f"Applying {k8s_file}")
                self.run_command(['kubectl', 'apply', '-f', k8s_file])

        # Wait for deployments to be ready
        for service_name, service_config in services.items():
            logger.info(f"Waiting for {service_name} deployment...")
            self.run_command([
                'kubectl', 'wait', '--for=condition=available',
                '--timeout=300s', f'deployment/{service_name}'
            ])

        logger.info("✅ Service deployment complete")

    def setup_monitoring(self):
        """Set up monitoring and observability"""
        logger.info("📊 Setting up monitoring stack...")

        monitoring_config = self.config['monitoring']

        if monitoring_config['prometheus']:
            self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/monitoring/prometheus.yaml'])

        if monitoring_config['grafana']:
            self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/monitoring/grafana.yaml'])

        if monitoring_config['alertmanager']:
            self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/monitoring/alertmanager.yaml'])

        logger.info("✅ Monitoring setup complete")

    def configure_security(self):
        """Configure security settings"""
        logger.info("🔐 Configuring security...")

        security_config = self.config['security']

        # Apply network policies
        if os.path.exists('cloud_deploy/security/network-policies.yaml'):
            self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/security/network-policies.yaml'])

        # Configure SSL/TLS
        if security_config['enable_https'] and security_config['ssl_cert'] == 'letsencrypt':
            self.setup_letsencrypt_ssl()

        # Apply RBAC
        if security_config['iam_roles']:
            self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/security/rbac.yaml'])

        logger.info("✅ Security configuration complete")

    def setup_letsencrypt_ssl(self):
        """Set up Let's Encrypt SSL certificates"""
        logger.info("🔒 Setting up SSL certificates...")

        # Install cert-manager
        self.run_command([
            'kubectl', 'apply',
            '-f', 'https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml'
        ])

        # Wait for cert-manager
        time.sleep(30)

        # Apply cluster issuer
        self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/security/cluster-issuer.yaml'])

        # Apply certificate
        self.run_command(['kubectl', 'apply', '-f', 'cloud_deploy/security/certificate.yaml'])

    def post_deployment_validation(self):
        """Validate deployment health"""
        logger.info("🔍 Running post-deployment validation...")

        # Check pod status
        result = self.run_command(['kubectl', 'get', 'pods'], capture_output=True)
        logger.info("Pod Status:")
        logger.info(result.stdout.decode())

        # Check service endpoints
        services = ['hft-trader-service', 'degen-trader-service', 'sentiment-analyzer-service']
        for service in services:
            result = self.run_command(['kubectl', 'get', 'svc', service], capture_output=True)
            logger.info(f"Service {service}: {result.stdout.decode().strip()}")

        # Test basic connectivity
        self.test_service_connectivity()

        logger.info("✅ Post-deployment validation complete")

    def test_service_connectivity(self):
        """Test service connectivity"""
        logger.info("Testing service connectivity...")

        # Test HFT trader health endpoint
        try:
            result = self.run_command([
                'kubectl', 'exec', 'deployment/hft-trader', '--',
                'curl', '-f', 'http://localhost:8080/health'
            ], timeout=30)
            logger.info("✅ HFT Trader health check passed")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  HFT Trader health check timeout")

        # Test sentiment analyzer
        try:
            result = self.run_command([
                'kubectl', 'exec', 'deployment/sentiment-analyzer', '--',
                'curl', '-f', 'http://localhost:8080/health'
            ], timeout=30)
            logger.info("✅ Sentiment Analyzer health check passed")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Sentiment Analyzer health check timeout")

    def deployment_complete(self):
        """Final deployment status and instructions"""
        logger.info("🎉 DEPLOYMENT COMPLETE!")
        logger.info("=" * 60)

        config = self.config

        # Get service URLs
        try:
            result = self.run_command([
                'kubectl', 'get', 'ingress', 'asterai-ingress'
            ], capture_output=True)
            ingress_info = result.stdout.decode()
            logger.info("Ingress Information:")
            logger.info(ingress_info)
        except:
            pass

        logger.info("\n📋 NEXT STEPS:")
        logger.info("1. Monitor services: kubectl get pods -w")
        logger.info("2. Check logs: kubectl logs deployment/hft-trader")
        logger.info("3. Access dashboard: https://your-domain.com")
        logger.info("4. Configure trading parameters via dashboard")
        logger.info("5. Start with small position sizes for testing")

        logger.info("\n🚨 IMPORTANT REMINDERS:")
        logger.info("- Monitor risk limits continuously")
        logger.info("- Start with conservative position sizes")
        logger.info("- Have emergency stop procedures ready")
        logger.info("- Regular backup and security updates required")

        logger.info(f"\n🔒 Security Score: Check security_audit_results.json")
        logger.info(f"📊 Monitoring: Access Grafana dashboard")

        # Save deployment info
        deployment_info = {
            'timestamp': time.time(),
            'version': config['project']['version'],
            'environment': config['project']['environment'],
            'cluster': config['infrastructure']['cluster_name'],
            'services': list(config['services'].keys()),
            'status': 'deployed'
        }

        with open('deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)

        logger.info("✅ Deployment information saved to deployment_info.json")

    def rollback_deployment(self):
        """Rollback deployment on failure"""
        logger.error("🔄 Rolling back deployment...")

        try:
            # Delete all resources
            self.run_command(['kubectl', 'delete', 'all', '--all'])

            # Delete cluster
            config = self.config
            self.run_command([
                'gcloud', 'container', 'clusters', 'delete',
                config['infrastructure']['cluster_name'],
                '--region', config['cloud']['region'],
                '--quiet'
            ])

            logger.info("✅ Rollback complete")

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")

    def run_command(self, cmd: List[str], capture_output: bool = False, timeout: int = 300):
        """Run a shell command with logging"""
        cmd_str = ' '.join(cmd)
        logger.info(f"Running: {cmd_str}")

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout
                )
                return result
            else:
                subprocess.run(cmd, check=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {cmd_str}")
            logger.error(f"Error: {e}")
            if e.stdout:
                logger.error(f"Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"Stderr: {e.stderr}")
            raise


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='AsterAI Production Deployment')
    parser.add_argument('--config', default='deploy_config.json',
                       help='Deployment configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without actual deployment')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback previous deployment')

    args = parser.parse_args()

    deployer = ProductionDeployer(args.config)

    if args.rollback:
        logger.info("Rolling back deployment...")
        deployer.rollback_deployment()
        return

    if args.dry_run:
        logger.info("DRY RUN - Validating configuration only")
        deployer.pre_deployment_checks()
        logger.info("✅ Dry run validation passed")
        return

    # Full deployment
    deployer.run_deployment()


if __name__ == "__main__":
    main()

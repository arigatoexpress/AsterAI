"""
Google Cloud Platform Configuration
Handles GCP services integration including Secret Manager, Cloud Storage, etc.
"""

import os
import json
from typing import Dict, Optional, Any
from google.cloud import secretmanager_v1
from google.api_core.exceptions import NotFound
import logging

logger = logging.getLogger(__name__)

class GCPConfig:
    """Google Cloud Platform configuration and services"""

    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.secret_client = None

        # Initialize secret manager client
        try:
            self.secret_client = secretmanager_v1.SecretManagerServiceClient()
            logger.info("GCP Secret Manager client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GCP Secret Manager: {e}")

    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """Get a secret from Google Secret Manager"""
        if not self.secret_client:
            logger.error("Secret Manager client not available")
            return None

        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"

            # Access the secret
            response = self.secret_client.access_secret_version(request={"name": name})

            # Return the decoded payload
            return response.payload.data.decode('UTF-8')

        except NotFound:
            logger.warning(f"Secret {secret_name} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name}: {e}")
            return None

    def set_secret(self, secret_name: str, value: str) -> bool:
        """Set a secret in Google Secret Manager"""
        if not self.secret_client:
            logger.error("Secret Manager client not available")
            return False

        try:
            # Create or update the secret
            parent = f"projects/{self.project_id}"

            # Check if secret exists
            secret_path = f"{parent}/secrets/{secret_name}"
            try:
                self.secret_client.get_secret(request={"name": secret_path})
                # Secret exists, add new version
                self.secret_client.add_secret_version(
                    request={
                        "parent": secret_path,
                        "payload": {"data": value.encode('UTF-8')}
                    }
                )
            except NotFound:
                # Secret doesn't exist, create it
                secret = self.secret_client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_name,
                        "secret": {"replication": {"automatic": {}}}
                    }
                )
                # Add the first version
                self.secret_client.add_secret_version(
                    request={
                        "parent": secret.name,
                        "payload": {"data": value.encode('UTF-8')}
                    }
                )

            logger.info(f"Successfully set secret {secret_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False

    def get_aster_credentials(self) -> Dict[str, str]:
        """Get Aster DEX API credentials from Secret Manager"""
        api_key = self.get_secret("ASTER_API_KEY")
        secret_key = self.get_secret("ASTER_SECRET_KEY")

        return {
            "api_key": api_key,
            "secret_key": secret_key
        }

    def validate_gcp_setup(self) -> Dict[str, bool]:
        """Validate GCP setup and permissions"""
        validation = {}

        # Check if secrets exist
        aster_creds = self.get_aster_credentials()
        validation["aster_api_key"] = aster_creds["api_key"] is not None
        validation["aster_secret_key"] = aster_creds["secret_key"] is not None

        # Check other common secrets
        validation["grok_api_key"] = self.get_secret("GROK_API_KEY") is not None
        validation["telegram_bot_token"] = self.get_secret("TELEGRAM_BOT_TOKEN") is not None

        return validation

    def get_cloud_storage_config(self) -> Dict[str, str]:
        """Get Cloud Storage configuration"""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "data_bucket": f"{self.project_id}-aster-data",
            "models_bucket": f"{self.project_id}-aster-models",
            "logs_bucket": f"{self.project_id}-aster-logs"
        }

    @classmethod
    def from_env(cls) -> Optional['GCPConfig']:
        """Create GCP config from environment variables"""
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT_ID')
        region = os.getenv('GCP_REGION', 'us-central1')

        if not project_id:
            logger.warning("No GCP project ID found in environment")
            return None

        return cls(project_id, region)

# Global GCP config instance
_gcp_config = None

def get_gcp_config() -> Optional[GCPConfig]:
    """Get global GCP config instance"""
    global _gcp_config
    if _gcp_config is None:
        _gcp_config = GCPConfig.from_env()
    return _gcp_config

def get_aster_credentials_gcp() -> Dict[str, str]:
    """Convenience function to get Aster credentials from GCP"""
    config = get_gcp_config()
    if config:
        return config.get_aster_credentials()
    return {"api_key": None, "secret_key": None}

# Backwards compatibility with existing secret manager
def update_secret_manager_with_gcp():
    """Update the local secret manager with GCP secrets"""
    from mcp_trader.security.secrets import get_secret_manager

    gcp_config = get_gcp_config()
    if not gcp_config:
        logger.warning("GCP config not available, skipping secret sync")
        return

    secret_manager = get_secret_manager()

    # Sync Aster credentials
    aster_creds = gcp_config.get_aster_credentials()
    if aster_creds["api_key"]:
        secret_manager.set_secret("ASTER_API_KEY", aster_creds["api_key"])
    if aster_creds["secret_key"]:
        secret_manager.set_secret("ASTER_SECRET_KEY", aster_creds["secret_key"])

    # Sync other common secrets
    grok_key = gcp_config.get_secret("GROK_API_KEY")
    if grok_key:
        secret_manager.set_secret("GROK_API_KEY", grok_key)

    telegram_token = gcp_config.get_secret("TELEGRAM_BOT_TOKEN")
    if telegram_token:
        secret_manager.set_secret("TELEGRAM_BOT_TOKEN", telegram_token)

    logger.info("Successfully synced GCP secrets to local secret manager")

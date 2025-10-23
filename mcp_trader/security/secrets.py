"""
Secure secret management for API keys and sensitive data.
Uses environment variables and optional encryption.
"""

import os
import base64
import json
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class SecretManager:
    """Secure secret management with optional encryption."""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password
        self._fernet = None
        self._secrets_cache = {}
        
        if master_password:
            self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption using master password."""
        try:
            # Derive key from master password
            password = self.master_password.encode()
            salt = b'aster_trader_salt_2024'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._fernet = Fernet(key)
            logger.info("Encryption setup completed")
        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self._fernet = None
    
    def _encrypt_value(self, value: str) -> str:
        """Encrypt a string value."""
        if not self._fernet:
            return value
        try:
            return self._fernet.encrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return value
    
    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a string value."""
        if not self._fernet:
            return encrypted_value
        try:
            return self._fernet.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_value
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        # Check cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]
        
        # Try environment variable first
        env_value = os.getenv(key)
        if env_value:
            # Check if it's encrypted
            if env_value.startswith('encrypted:'):
                decrypted_value = self._decrypt_value(env_value[10:])  # Remove 'encrypted:' prefix
                self._secrets_cache[key] = decrypted_value
                return decrypted_value
            else:
                self._secrets_cache[key] = env_value
                return env_value
        
        # Try secrets file
        secrets_file = os.getenv('SECRETS_FILE', '.secrets.json')
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                
                if key in secrets:
                    value = secrets[key]
                    if isinstance(value, str) and value.startswith('encrypted:'):
                        decrypted_value = self._decrypt_value(value[10:])
                        self._secrets_cache[key] = decrypted_value
                        return decrypted_value
                    else:
                        self._secrets_cache[key] = value
                        return value
            except Exception as e:
                logger.error(f"Failed to read secrets file: {e}")
        
        return default
    
    def set_secret(self, key: str, value: str, encrypt: bool = True) -> bool:
        """Set a secret value."""
        try:
            if encrypt and self._fernet:
                encrypted_value = self._encrypt_value(value)
                env_value = f"encrypted:{encrypted_value}"
            else:
                env_value = value
            
            # Update cache
            self._secrets_cache[key] = value
            
            # Set environment variable
            os.environ[key] = env_value
            
            logger.info(f"Secret {key} set successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set secret {key}: {e}")
            return False
    
    def save_secrets_to_file(self, filepath: str = '.secrets.json') -> bool:
        """Save all secrets to a file."""
        try:
            secrets = {}
            for key, value in self._secrets_cache.items():
                if self._fernet:
                    encrypted_value = self._encrypt_value(value)
                    secrets[key] = f"encrypted:{encrypted_value}"
                else:
                    secrets[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            logger.info(f"Secrets saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            return False
    
    def load_secrets_from_file(self, filepath: str = '.secrets.json') -> bool:
        """Load secrets from a file."""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Secrets file {filepath} not found")
                return False
            
            with open(filepath, 'r') as f:
                secrets = json.load(f)
            
            for key, value in secrets.items():
                if isinstance(value, str) and value.startswith('encrypted:'):
                    decrypted_value = self._decrypt_value(value[10:])
                    self._secrets_cache[key] = decrypted_value
                    os.environ[key] = f"encrypted:{value[10:]}"
                else:
                    self._secrets_cache[key] = value
                    os.environ[key] = value
            
            logger.info(f"Loaded {len(secrets)} secrets from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return False
    
    def get_aster_credentials(self) -> Dict[str, str]:
        """Get Aster API credentials."""
        # Try GCP Secret Manager first if available
        try:
            from mcp_trader.config.gcp import get_aster_credentials_gcp
            gcp_creds = get_aster_credentials_gcp()
            if gcp_creds['api_key'] and gcp_creds['secret_key']:
                logger.info("Using Aster credentials from GCP Secret Manager")
                return gcp_creds
        except Exception as e:
            logger.warning(f"Failed to get credentials from GCP: {e}")

        # Fallback to local secrets
        return {
            'api_key': self.get_secret('ASTER_API_KEY'),
            'secret_key': self.get_secret('ASTER_SECRET_KEY')
        }
    
    def get_gcp_credentials(self) -> Dict[str, str]:
        """Get GCP credentials."""
        return {
            'project_id': self.get_secret('GCP_PROJECT_ID'),
            'credentials_path': self.get_secret('GOOGLE_APPLICATION_CREDENTIALS')
        }
    
    def get_news_api_credentials(self) -> Dict[str, str]:
        """Get news API credentials."""
        return {
            'cryptopanic_api_key': self.get_secret('CRYPTOPANIC_API_KEY'),
            'newsapi_api_key': self.get_secret('NEWSAPI_API_KEY')
        }
    
    def get_ai_api_credentials(self) -> Dict[str, str]:
        """Get AI API credentials."""
        return {
            'openai_api_key': self.get_secret('OPENAI_API_KEY'),
            'groq_api_key': self.get_secret('GROQ_API_KEY'),
            'anthropic_api_key': self.get_secret('ANTHROPIC_API_KEY')
        }
    
    def validate_required_secrets(self, required_keys: list) -> Dict[str, bool]:
        """Validate that required secrets are available."""
        validation = {}
        for key in required_keys:
            value = self.get_secret(key)
            validation[key] = value is not None and value != ""
        
        missing = [key for key, valid in validation.items() if not valid]
        if missing:
            logger.warning(f"Missing required secrets: {missing}")
        
        return validation


# Global secret manager instance
_secret_manager = None


def get_secret_manager(master_password: Optional[str] = None) -> SecretManager:
    """Get global secret manager instance."""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager(master_password)
    return _secret_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return get_secret_manager().get_secret(key, default)


def set_secret(key: str, value: str, encrypt: bool = True) -> bool:
    """Convenience function to set a secret."""
    return get_secret_manager().set_secret(key, value, encrypt)


# Example usage and setup
def setup_aster_credentials(api_key: str, secret_key: str, master_password: Optional[str] = None):
    """Setup Aster API credentials securely."""
    sm = get_secret_manager(master_password)
    
    sm.set_secret('ASTER_API_KEY', api_key, encrypt=True)
    sm.set_secret('ASTER_SECRET_KEY', secret_key, encrypt=True)
    
    logger.info("Aster credentials configured securely")


def setup_news_credentials(cryptopanic_key: Optional[str] = None, 
                          newsapi_key: Optional[str] = None,
                          master_password: Optional[str] = None):
    """Setup news API credentials securely."""
    sm = get_secret_manager(master_password)
    
    if cryptopanic_key:
        sm.set_secret('CRYPTOPANIC_API_KEY', cryptopanic_key, encrypt=True)
    if newsapi_key:
        sm.set_secret('NEWSAPI_API_KEY', newsapi_key, encrypt=True)
    
    logger.info("News API credentials configured securely")


def setup_ai_credentials(openai_key: Optional[str] = None,
                        groq_key: Optional[str] = None,
                        anthropic_key: Optional[str] = None,
                        master_password: Optional[str] = None):
    """Setup AI API credentials securely."""
    sm = get_secret_manager(master_password)
    
    if openai_key:
        sm.set_secret('OPENAI_API_KEY', openai_key, encrypt=True)
    if groq_key:
        sm.set_secret('GROQ_API_KEY', groq_key, encrypt=True)
    if anthropic_key:
        sm.set_secret('ANTHROPIC_API_KEY', anthropic_key, encrypt=True)
    
    logger.info("AI API credentials configured securely")


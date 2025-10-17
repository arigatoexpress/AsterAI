"""
Data Sharing Module
Manages centralized data repository with access controls.
Ensures data consistency and availability across components.
"""
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DataSharer:
    """Centralized data sharing with role-based access."""

    def __init__(self, repo_path: str = "data/centralized_repo"):
        self.repo_path = repo_path
        self.access_roles = {
            'collector': ['read', 'write'],
            'validator': ['read', 'write'],
            'analyzer': ['read'],
            'utilizer': ['read'],
            'admin': ['read', 'write', 'delete']
        }
        os.makedirs(repo_path, exist_ok=True)

    def store_data(self, data_key: str, data: bytes, role: str = 'collector') -> bool:
        """Store data in centralized repo."""
        if 'write' not in self.access_roles.get(role, []):
            logger.error(f"Role {role} lacks write access")
            return False

        file_path = os.path.join(self.repo_path, f"{data_key}.parquet")
        with open(file_path, 'wb') as f:
            f.write(data)
        logger.info(f"Data stored: {data_key}")
        return True

    def retrieve_data(self, data_key: str, role: str = 'analyzer') -> bytes or None:
        """Retrieve data from repo."""
        if 'read' not in self.access_roles.get(role, []):
            logger.error(f"Role {role} lacks read access")
            return None

        file_path = os.path.join(self.repo_path, f"{data_key}.parquet")
        if not os.path.exists(file_path):
            logger.error(f"Data not found: {data_key}")
            return None

        with open(file_path, 'rb') as f:
            data = f.read()
        logger.info(f"Data retrieved: {data_key}")
        return data

    def list_available_data(self, role: str = 'analyzer') -> List[str]:
        """List available data keys."""
        if 'read' not in self.access_roles.get(role, []):
            return []

        files = os.listdir(self.repo_path)
        return [f.replace('.parquet', '') for f in files if f.endswith('.parquet')]

# Example usage
if __name__ == "__main__":
    sharer = DataSharer()
    data = b"sample data"
    sharer.store_data("sample_key", data, role="collector")
    retrieved = sharer.retrieve_data("sample_key", role="analyzer")
    print(f"Retrieved: {retrieved == data}")

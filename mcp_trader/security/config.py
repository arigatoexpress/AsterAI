"""
Security Configuration for AsterAI Trading System

Centralized security settings and policies
"""

import os
from typing import Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class SecurityConfig:
    """Security configuration settings"""

    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_limit: int = 200

    # Request validation
    max_request_size_bytes: int = 1024 * 1024  # 1MB
    max_url_length: int = 2048
    max_query_params: int = 50
    max_header_size: int = 4096

    # Authentication
    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = os.getenv("JWT_SECRET", "change-this-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # CORS settings
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:8080", "http://127.0.0.1:8080"])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True

    # Content Security Policy
    csp_directives: Dict[str, str] = field(default_factory=lambda: {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src": "'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src": "'self' https://fonts.gstatic.com",
        "img-src": "'self' data: https:",
        "connect-src": "'self' ws: wss:"
    })

    # Input validation limits
    max_string_length: int = 1000
    max_list_length: int = 100
    max_dict_depth: int = 5

    # File upload limits
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {'.txt', '.csv', '.json', '.log'})

    # Database security
    sql_injection_patterns: List[str] = field(default_factory=lambda: [
        r';\s*--',  # SQL comment
        r';\s*/\*',  # SQL block comment start
        r'union\s+select',  # Union-based injection
        r'exec\s*\(',  # Command execution
        r'xp_cmdshell',  # System command execution
        r'information_schema',  # Database enumeration
        r'load_file\s*\(',  # File reading
        r'into\s+outfile',  # File writing
    ])

    # XSS prevention
    xss_patterns: List[str] = field(default_factory=lambda: [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'vbscript:',
        r'on\w+\s*=',
        r'style\s*=.*expression',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
    ])

    # IP blocking
    blocked_ip_ranges: List[str] = field(default_factory=lambda: [
        "0.0.0.0/8",      # Current network
        "127.0.0.0/8",    # Loopback
        "169.254.0.0/16", # Link local
        "224.0.0.0/4",    # Multicast
    ])

    # Monitoring
    enable_security_logging: bool = True
    log_security_events: bool = True
    alert_on_suspicious_activity: bool = True

    # Encryption
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 30

    # Session security
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    require_https: bool = False  # Set to True in production

    # Backup security
    encrypt_backups: bool = True
    backup_retention_days: int = 30

    def get_csp_header(self) -> str:
        """Generate Content Security Policy header"""
        directives = []
        for directive, value in self.csp_directives.items():
            directives.append(f"{directive} {value}")
        return "; ".join(directives)

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is in blocked ranges"""
        try:
            from ipaddress import ip_address, ip_network
            client_ip = ip_address(ip_address)

            for blocked_range in self.blocked_ip_ranges:
                if client_ip in ip_network(blocked_range):
                    return True

            return False
        except Exception:
            # If IP parsing fails, allow the request (fail-safe)
            return False


# Global security configuration
security_config = SecurityConfig()

# Override with environment variables
if os.getenv("RATE_LIMIT_REQUESTS"):
    security_config.rate_limit_requests_per_minute = int(os.getenv("RATE_LIMIT_REQUESTS"))

if os.getenv("MAX_REQUEST_SIZE"):
    security_config.max_request_size_bytes = int(os.getenv("MAX_REQUEST_SIZE"))

if os.getenv("REQUIRE_HTTPS", "").lower() == "true":
    security_config.require_https = True

# Production security settings
if os.getenv("ENVIRONMENT") == "production":
    security_config.require_https = True
    security_config.allowed_origins = ["https://your-domain.com"]
    security_config.enable_security_logging = True
    security_config.alert_on_suspicious_activity = True
    security_config.encrypt_backups = True

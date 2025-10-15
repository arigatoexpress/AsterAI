"""
Security Middleware for AsterAI Trading System

Implements security headers, rate limiting, input validation,
and request sanitization for all endpoints.
"""

import time
import hashlib
import hmac
import logging
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
import ipaddress
import re

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .input_validation import InputValidator

logger = logging.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for FastAPI applications"""

    def __init__(self, app, rate_limit_requests: int = 100, rate_limit_window: int = 60):
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: set = set()

    async def dispatch(self, request: Request, call_next):
        """Process each request through security checks"""

        # Get client IP
        client_ip = self.get_client_ip(request)

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "IP address blocked"}
            )

        # Rate limiting
        if not self.check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"}
            )

        # Validate request
        validation_error = self.validate_request(request)
        if validation_error:
            logger.warning(f"Request validation failed from {client_ip}: {validation_error}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": validation_error}
            )

        # Add security headers to response
        response = await call_next(request)
        response = self.add_security_headers(response)

        # Log security events
        if request.method in ['POST', 'PUT', 'DELETE']:
            logger.info(f"Security: {request.method} {request.url.path} from {client_ip}")

        return response

    def get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check X-Forwarded-For header (for proxies)
        x_forwarded_for = request.headers.get('X-Forwarded-For')
        if x_forwarded_for:
            # Take the first IP (original client)
            return x_forwarded_for.split(',')[0].strip()

        # Check X-Real-IP header
        x_real_ip = request.headers.get('X-Real-IP')
        if x_real_ip:
            return x_real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()

        # Initialize if not exists
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # Clean old requests
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < self.rate_limit_window
        ]

        # Check limit
        if len(self.request_counts[client_ip]) >= self.rate_limit_requests:
            return False

        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True

    def validate_request(self, request: Request) -> Optional[str]:
        """Validate incoming request"""
        try:
            # Validate path
            if not self.is_valid_path(request.url.path):
                return "Invalid request path"

            # Validate query parameters
            for key, value in request.query_params.items():
                if not self.is_valid_param(key, value):
                    return f"Invalid query parameter: {key}"

            # Validate headers
            suspicious_headers = self.check_suspicious_headers(request.headers)
            if suspicious_headers:
                return f"Suspicious headers detected: {suspicious_headers}"

            # Validate content type for POST/PUT requests
            if request.method in ['POST', 'PUT']:
                content_type = request.headers.get('content-type', '')
                if not content_type.startswith('application/json'):
                    return "Invalid content type. Only application/json accepted"

            return None

        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return "Request validation failed"

    def is_valid_path(self, path: str) -> bool:
        """Validate request path"""
        # Allow only safe characters
        if not re.match(r'^[a-zA-Z0-9/_.-]+$', path):
            return False

        # Prevent path traversal
        if '..' in path or path.startswith('/'):
            return False

        return True

    def is_valid_param(self, key: str, value: str) -> bool:
        """Validate query parameter"""
        # Check key
        if not re.match(r'^[a-zA-Z0-9_.-]+$', key):
            return False

        # Check value length
        if len(value) > 1000:
            return False

        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script', r'javascript:', r'data:', r'vbscript:',
            r'on\w+\s*=', r'style\s*=.*expression',
            r'<iframe', r'<object', r'<embed'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False

        return True

    def check_suspicious_headers(self, headers) -> List[str]:
        """Check for suspicious headers"""
        suspicious = []

        # Check User-Agent
        user_agent = headers.get('user-agent', '')
        if not user_agent or len(user_agent) < 10:
            suspicious.append('user-agent')

        # Check for unusual headers
        unusual_headers = [
            'x-forwarded-for', 'x-real-ip', 'x-client-ip',
            'x-forwarded', 'x-remote-addr', 'x-host'
        ]

        for header in unusual_headers:
            if header in headers and len(headers.getlist(header)) > 1:
                suspicious.append(header)

        return suspicious

    def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # Strict transport security (if HTTPS)
        # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' ws: wss:"
        )
        response.headers['Content-Security-Policy'] = csp

        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Feature Policy
        response.headers['Permissions-Policy'] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )

        return response


class APISecurity:
    """API security utilities"""

    @staticmethod
    def validate_api_request(api_key: str, signature: str, timestamp: str,
                           payload: str, secret: str) -> bool:
        """
        Validate API request with HMAC signature

        Args:
            api_key: API key
            signature: HMAC signature
            timestamp: Request timestamp
            payload: Request payload
            secret: API secret

        Returns:
            True if valid
        """
        try:
            # Check timestamp (prevent replay attacks)
            request_time = datetime.fromtimestamp(int(timestamp))
            now = datetime.now()

            # Allow 5 minute window
            if abs((now - request_time).total_seconds()) > 300:
                return False

            # Create signature
            message = f"{api_key}{timestamp}{payload}"
            expected_signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            # Use constant time comparison
            return hmac.compare_digest(signature, expected_signature)

        except Exception:
            return False

    @staticmethod
    def sanitize_api_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize API response data"""
        return InputValidator.sanitize_api_response(response_data)

    @staticmethod
    def validate_request_size(request: Request, max_size: int = 1024*1024) -> bool:
        """Validate request size"""
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                size = int(content_length)
                return size <= max_size
            except ValueError:
                return False
        return True


def require_authentication(roles: List[str] = None):
    """Decorator for requiring authentication"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )

            # Check authentication (implement your auth logic here)
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )

            # Validate API key format
            if not InputValidator.validate_api_key(api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key format"
                )

            # Check roles if specified
            if roles:
                user_role = request.headers.get('X-User-Role', 'user')
                if user_role not in roles:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting"""
    def decorator(func: Callable):
        # Simple in-memory rate limiting (use Redis in production)
        request_counts: Dict[str, List[float]] = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get client IP
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request:
                client_ip = request.client.host if request.client else "unknown"

                current_time = time.time()
                window_start = current_time - 60  # 1 minute window

                # Clean old requests
                if client_ip in request_counts:
                    request_counts[client_ip] = [
                        t for t in request_counts[client_ip] if t > window_start
                    ]
                else:
                    request_counts[client_ip] = []

                # Check limit
                if len(request_counts[client_ip]) >= requests_per_minute:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded"
                    )

                # Add current request
                request_counts[client_ip].append(current_time)

            return await func(*args, **kwargs)
        return wrapper
    return decorator


class SecurityLogger:
    """Security event logging"""

    def __init__(self, log_file: str = "security.log"):
        self.logger = logging.getLogger('security')
        self.logger.setLevel(logging.INFO)

        # File handler
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          ip_address: str = None, user_id: str = None):
        """Log security events"""
        message = f"SECURITY_EVENT: {event_type}"
        if ip_address:
            message += f" | IP: {ip_address}"
        if user_id:
            message += f" | User: {user_id}"
        if details:
            message += f" | Details: {details}"

        self.logger.warning(message)

    def log_auth_failure(self, ip_address: str, reason: str):
        """Log authentication failures"""
        self.log_security_event(
            "AUTH_FAILURE",
            {"reason": reason},
            ip_address=ip_address
        )

    def log_rate_limit_hit(self, ip_address: str, endpoint: str):
        """Log rate limit violations"""
        self.log_security_event(
            "RATE_LIMIT",
            {"endpoint": endpoint},
            ip_address=ip_address
        )

    def log_suspicious_request(self, ip_address: str, details: Dict[str, Any]):
        """Log suspicious requests"""
        self.log_security_event(
            "SUSPICIOUS_REQUEST",
            details,
            ip_address=ip_address
        )

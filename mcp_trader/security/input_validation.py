"""
Input Validation and Sanitization for AsterAI Trading System

Comprehensive input validation to prevent injection attacks, ensure data integrity,
and provide robust error handling for all user inputs and API responses.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime
import ipaddress

logger = logging.getLogger(__name__)


class InputValidator:
    """Centralized input validation and sanitization"""

    # Regex patterns for validation
    SYMBOL_PATTERN = re.compile(r'^[A-Z]{2,10}$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9]{32,128}$')
    URL_PATTERN = re.compile(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/.*)?$')
    IP_PATTERN = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')

    # Trading limits
    MAX_ORDER_SIZE = Decimal('1000000')  # $1M max order
    MIN_ORDER_SIZE = Decimal('0.01')     # $0.01 min order
    MAX_LEVERAGE = Decimal('100')        # 100x max leverage
    MIN_LEVERAGE = Decimal('1')          # 1x min leverage

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input by removing dangerous characters

        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")

        # Remove null bytes and other dangerous characters
        sanitized = input_str.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"String truncated to {max_length} characters")

        return sanitized.strip()

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate trading symbol format

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Returns:
            True if valid
        """
        if not isinstance(symbol, str):
            return False

        # Must be uppercase letters only, 2-10 characters
        return bool(InputValidator.SYMBOL_PATTERN.match(symbol))

    @staticmethod
    def validate_price(price: Union[str, float, Decimal]) -> Decimal:
        """
        Validate and convert price to Decimal

        Args:
            price: Price value

        Returns:
            Decimal price

        Raises:
            ValueError: If price is invalid
        """
        try:
            if isinstance(price, str):
                # Remove commas and spaces
                price = price.replace(',', '').replace(' ', '')

            decimal_price = Decimal(str(price))

            # Check reasonable bounds
            if decimal_price <= 0:
                raise ValueError("Price must be positive")
            if decimal_price > InputValidator.MAX_ORDER_SIZE:
                raise ValueError(f"Price too large: {decimal_price}")

            return decimal_price

        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid price: {price}") from e

    @staticmethod
    def validate_quantity(quantity: Union[str, float, Decimal]) -> Decimal:
        """
        Validate and convert quantity to Decimal

        Args:
            quantity: Quantity value

        Returns:
            Decimal quantity
        """
        try:
            decimal_qty = Decimal(str(quantity))

            if decimal_qty <= 0:
                raise ValueError("Quantity must be positive")
            if decimal_qty > InputValidator.MAX_ORDER_SIZE:
                raise ValueError(f"Quantity too large: {decimal_qty}")

            return decimal_qty

        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid quantity: {quantity}") from e

    @staticmethod
    def validate_leverage(leverage: Union[str, float, Decimal]) -> Decimal:
        """
        Validate leverage value

        Args:
            leverage: Leverage ratio

        Returns:
            Decimal leverage
        """
        try:
            decimal_lev = Decimal(str(leverage))

            if decimal_lev < InputValidator.MIN_LEVERAGE:
                raise ValueError(f"Leverage too low: {decimal_lev}")
            if decimal_lev > InputValidator.MAX_LEVERAGE:
                raise ValueError(f"Leverage too high: {decimal_lev}")

            return decimal_lev

        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid leverage: {leverage}") from e

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate API key format

        Args:
            api_key: API key string

        Returns:
            True if valid format
        """
        if not isinstance(api_key, str):
            return False

        return bool(InputValidator.API_KEY_PATTERN.match(api_key))

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format

        Args:
            email: Email address

        Returns:
            True if valid format
        """
        if not isinstance(email, str):
            return False

        return bool(InputValidator.EMAIL_PATTERN.match(email))

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format

        Args:
            url: URL string

        Returns:
            True if valid format
        """
        if not isinstance(url, str):
            return False

        return bool(InputValidator.URL_PATTERN.match(url))

    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """
        Validate IP address format

        Args:
            ip: IP address string

        Returns:
            True if valid IP address
        """
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_timestamp(timestamp: Union[str, float, int]) -> datetime:
        """
        Validate and convert timestamp

        Args:
            timestamp: Timestamp value

        Returns:
            datetime object
        """
        try:
            if isinstance(timestamp, str):
                # Try parsing as ISO format first
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    # Try as unix timestamp
                    return datetime.fromtimestamp(float(timestamp))
            else:
                return datetime.fromtimestamp(float(timestamp))

        except (ValueError, TypeError, OSError) as e:
            raise ValueError(f"Invalid timestamp: {timestamp}") from e

    @staticmethod
    def validate_percentage(percentage: Union[str, float, Decimal]) -> Decimal:
        """
        Validate percentage value (0-100)

        Args:
            percentage: Percentage value

        Returns:
            Decimal percentage
        """
        try:
            decimal_pct = Decimal(str(percentage))

            if decimal_pct < 0:
                raise ValueError(f"Percentage cannot be negative: {decimal_pct}")
            if decimal_pct > 100:
                raise ValueError(f"Percentage cannot exceed 100%: {decimal_pct}")

            return decimal_pct

        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Invalid percentage: {percentage}") from e

    @staticmethod
    def sanitize_api_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize API response data

        Args:
            response: Raw API response

        Returns:
            Sanitized response
        """
        if not isinstance(response, dict):
            return {}

        sanitized = {}

        for key, value in response.items():
            # Sanitize key
            if isinstance(key, str):
                clean_key = InputValidator.sanitize_string(key, max_length=100)
            else:
                clean_key = str(key)

            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = InputValidator.sanitize_string(value, max_length=1000)
            elif isinstance(value, (int, float)):
                # Basic numeric validation
                if abs(value) > 1e15:  # Prevent extreme values
                    clean_value = 0
                else:
                    clean_value = value
            elif isinstance(value, list):
                # Sanitize list elements (shallow)
                clean_value = [
                    InputValidator.sanitize_string(str(item), max_length=100) if isinstance(item, str) else item
                    for item in value[:100]  # Limit list size
                ]
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts (shallow to prevent stack overflow)
                clean_value = InputValidator.sanitize_api_response(value)
            else:
                clean_value = value

            sanitized[clean_key] = clean_value

        return sanitized

    @staticmethod
    def validate_order_params(symbol: str, side: str, quantity: Union[str, float, Decimal],
                            price: Optional[Union[str, float, Decimal]] = None) -> Dict[str, Any]:
        """
        Validate complete order parameters

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price (optional for market orders)

        Returns:
            Validated order parameters

        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []

        # Validate symbol
        if not InputValidator.validate_symbol(symbol):
            errors.append(f"Invalid symbol: {symbol}")

        # Validate side
        if side not in ['buy', 'sell']:
            errors.append(f"Invalid side: {side}. Must be 'buy' or 'sell'")

        # Validate quantity
        try:
            valid_quantity = InputValidator.validate_quantity(quantity)
        except ValueError as e:
            errors.append(str(e))

        # Validate price if provided
        if price is not None:
            try:
                valid_price = InputValidator.validate_price(price)
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise ValueError("Order validation failed: " + "; ".join(errors))

        result = {
            'symbol': symbol,
            'side': side,
            'quantity': valid_quantity
        }

        if price is not None:
            result['price'] = valid_price

        return result

    @classmethod
    def create_secure_config(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a secure configuration by validating all values

        Args:
            config_dict: Raw configuration dictionary

        Returns:
            Validated and sanitized configuration
        """
        secure_config = {}

        for key, value in config_dict.items():
            try:
                if key in ['api_key', 'api_secret', 'secret_key']:
                    # Validate sensitive keys
                    if not cls.validate_api_key(str(value)):
                        logger.warning(f"Invalid API key format for {key}")
                        continue
                elif key in ['email', 'notification_email']:
                    # Validate emails
                    if not cls.validate_email(str(value)):
                        logger.warning(f"Invalid email format for {key}")
                        continue
                elif key in ['base_url', 'ws_url', 'callback_url']:
                    # Validate URLs
                    if not cls.validate_url(str(value)):
                        logger.warning(f"Invalid URL format for {key}")
                        continue
                elif 'price' in key.lower():
                    # Validate price values
                    value = cls.validate_price(value)
                elif 'quantity' in key.lower() or 'size' in key.lower():
                    # Validate quantity values
                    value = cls.validate_quantity(value)
                elif 'leverage' in key.lower():
                    # Validate leverage
                    value = cls.validate_leverage(value)
                elif 'percentage' in key.lower() or 'pct' in key.lower():
                    # Validate percentages
                    value = cls.validate_percentage(value)
                elif isinstance(value, str):
                    # Sanitize string values
                    value = cls.sanitize_string(value)

                secure_config[key] = value

            except Exception as e:
                logger.warning(f"Failed to validate config key {key}: {e}")
                continue

        return secure_config

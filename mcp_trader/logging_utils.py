import logging
import logging.handlers
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, Any]
    thread_id: int
    process_id: int


class StructuredLogger:
    """Enhanced structured logger with monitoring capabilities."""

    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
        )

        # Create handlers
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(self.console_formatter)

        # File handlers for different log levels
        self.error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}_errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        self.error_handler.setLevel(logging.ERROR)
        self.error_handler.setFormatter(self.file_formatter)

        self.info_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}_info.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        self.info_handler.setLevel(logging.INFO)
        self.info_handler.setFormatter(self.file_formatter)

        # Add handlers
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.error_handler)
        self.logger.addHandler(self.info_handler)

        # Performance tracking
        self.performance_metrics = {}
        self.start_times = {}

    def _create_log_entry(self, level: str, message: str, extra: Optional[Dict] = None) -> LogEntry:
        """Create a structured log entry."""
        import inspect

        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None

        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            logger_name=self.name,
            message=message,
            module=caller_frame.f_globals.get('__name__', 'unknown') if caller_frame else 'unknown',
            function=caller_frame.f_code.co_name if caller_frame else 'unknown',
            line=caller_frame.f_lineno if caller_frame else 0,
            extra=extra or {},
            thread_id=threading.get_ident(),
            process_id=os.getpid()
        )

    def _log_structured(self, level: str, message: str, extra: Optional[Dict] = None):
        """Log with structured data."""
        log_entry = self._create_log_entry(level, message, extra)

        # Log to appropriate handler
        if level in ['ERROR', 'CRITICAL']:
            self.logger.error(f"{message} | Extra: {json.dumps(log_entry.extra)}")
        elif level == 'WARNING':
            self.logger.warning(f"{message} | Extra: {json.dumps(log_entry.extra)}")
        elif level == 'INFO':
            self.logger.info(f"{message} | Extra: {json.dumps(log_entry.extra)}")
        else:
            self.logger.debug(f"{message} | Extra: {json.dumps(log_entry.extra)}")

        # Export to JSON if configured
        if os.getenv('EXPORT_JSON_LOGS', 'false').lower() == 'true':
            self._export_json_log(log_entry)

    def _export_json_log(self, entry: LogEntry):
        """Export log entry to JSON file."""
        json_file = self.log_dir / f"{self.name}_structured.jsonl"

        try:
            with open(json_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
        except Exception as e:
            # Fallback to console if file write fails
            print(f"Failed to write structured log: {e}")

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_structured('DEBUG', message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_structured('INFO', message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_structured('WARNING', message, kwargs if kwargs else None)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_structured('ERROR', message, kwargs if kwargs else None)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_structured('CRITICAL', message, kwargs if kwargs else None)

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration in seconds."""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.performance_metrics[operation] = duration

            # Log performance metric
            self.info(f"Operation '{operation}' completed in {duration:.3f}s",
                     operation=operation, duration=duration)

            del self.start_times[operation]
            return duration

        return 0.0

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def log_trading_event(self, event_type: str, symbol: str, **kwargs):
        """Log a trading-related event."""
        self.info(f"Trading event: {event_type} for {symbol}",
                 event_type=event_type, symbol=symbol, **kwargs)

    def log_system_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a system metric."""
        self.info(f"System metric: {metric_name} = {value}{unit}",
                 metric_name=metric_name, value=value, unit=unit)

    def log_ai_learning(self, learning_type: str, **kwargs):
        """Log AI learning events."""
        self.info(f"AI Learning: {learning_type}",
                 learning_type=learning_type, **kwargs)


# Global logger instance
_logger_instance = None
_logger_lock = threading.Lock()


def get_logger(name: str = "mcp_trader") -> StructuredLogger:
    """Get or create a structured logger instance."""
    global _logger_instance

    with _logger_lock:
        if _logger_instance is None or _logger_instance.name != name:
            _logger_instance = StructuredLogger(name)
        return _logger_instance


def setup_logger(name: str | None = None, level: str = "INFO") -> StructuredLogger:
    """Setup structured logging configuration."""
    return get_logger(name if name else "mcp_trader")


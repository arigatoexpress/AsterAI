import logging
from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Minimal metrics aggregator that emits structured logs.

    Replace/extend with Prometheus/OpenTelemetry as needed.
    """

    counters: Dict[str, float] = field(default_factory=dict)

    def inc(self, name: str, value: float = 1.0, labels: Dict[str, Any] | None = None) -> None:
        key = self._format_key(name, labels)
        self.counters[key] = self.counters.get(key, 0.0) + value
        logger.info({
            'metric': name,
            'type': 'counter',
            'value': self.counters[key],
            'labels': labels or {},
            'timestamp': datetime.utcnow().isoformat()
        })

    def observe(self, name: str, value: float, labels: Dict[str, Any] | None = None) -> None:
        logger.info({
            'metric': name,
            'type': 'gauge',
            'value': value,
            'labels': labels or {},
            'timestamp': datetime.utcnow().isoformat()
        })

    @staticmethod
    def _format_key(name: str, labels: Dict[str, Any] | None) -> str:
        if not labels:
            return name
        parts = [f"{k}={labels[k]}" for k in sorted(labels.keys())]
        return f"{name}|" + ",".join(parts)



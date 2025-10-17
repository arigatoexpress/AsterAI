# Self-Healing Trading System Guide

## Overview

The Self-Healing Trading System provides **ultimate robustness and uptime** through automatic detection and repair of data issues, API endpoint failures, and system faults. This system ensures your trading operations continue uninterrupted even under adverse conditions.

## Key Features

### 🔄 Self-Healing Data
- **Gap Detection**: Automatically identifies missing data periods
- **Corruption Repair**: Detects and fixes data quality issues
- **Fallback Sources**: Switches to alternative data providers when primary sources fail
- **Quality Monitoring**: Continuous assessment of data integrity

### 🌐 Self-Healing Endpoints
- **Circuit Breaker**: Prevents cascade failures by isolating faulty endpoints
- **Load Balancing**: Distributes requests across multiple providers
- **Rate Limit Handling**: Automatic throttling and backoff strategies
- **Failover**: Seamless switching between backup endpoints

### 📊 Monitoring & Dashboard
- **Real-time Health Reports**: Live system status monitoring
- **Alert System**: Automatic notifications for critical issues
- **Web Dashboard**: Visual interface for system management
- **Performance Metrics**: Comprehensive system analytics

## Quick Start

### 1. Start the Self-Healing System

```bash
# Start with default configuration
python scripts/start_self_healing_system.py

# Or specify custom config
python scripts/start_self_healing_system.py --config config/my_config.json
```

### 2. Access the Dashboard

Once started, the system provides a web dashboard at:
- **Health Check**: http://localhost:8080/health
- **System Status**: http://localhost:8080/status
- **Data Health**: http://localhost:8080/data/health
- **Endpoint Health**: http://localhost:8080/endpoints/health

## Configuration

The system uses a comprehensive configuration file (`config/self_healing_config.json`):

```json
{
  "data_healing": {
    "enabled": true,
    "gap_detection_window_hours": 24,
    "max_gap_fill_attempts": 3,
    "corruption_threshold": 0.05,
    "auto_repair_enabled": true
  },
  "endpoint_healing": {
    "enabled": true,
    "circuit_breaker_threshold": 5,
    "load_balancing_enabled": true
  },
  "monitoring": {
    "enabled": true,
    "dashboard_port": 8080,
    "alerts_enabled": true
  }
}
```

## API Endpoints

### Health Monitoring

```bash
# System health check
GET /health

# Comprehensive system status
GET /status

# Data health report
GET /data/health

# Endpoint health report
GET /endpoints/health

# Active alerts
GET /alerts

# System metrics
GET /metrics
```

### Manual Controls

```bash
# Trigger data repair for specific symbol
POST /data/repair
{
  "symbol": "BTC",
  "repair_type": "auto"
}

# Test endpoint functionality
POST /endpoints/test
{
  "endpoint_type": "market_data",
  "test_data": {}
}

# Restart self-healing systems
POST /system/restart
```

## Self-Healing Capabilities

### Data Healing

1. **Gap Detection**: Identifies missing data periods automatically
2. **Gap Filling**: Uses multiple strategies:
   - Forward/backward fill for small gaps
   - Linear interpolation for medium gaps
   - External API calls for large gaps
3. **Corruption Detection**: Identifies invalid prices, duplicates, outliers
4. **Quality Scoring**: Assigns health scores to data assets

### Endpoint Healing

1. **Health Monitoring**: Continuous endpoint availability checks
2. **Circuit Breaker**: Isolates failing endpoints to prevent cascade failures
3. **Load Balancing**: Distributes load across healthy endpoints
4. **Failover**: Automatic switching to backup endpoints
5. **Rate Limiting**: Respects API limits with intelligent throttling

### Recovery Strategies

1. **Automatic Repair**: Most issues resolved without human intervention
2. **Graceful Degradation**: System continues operating with reduced capacity
3. **Emergency Protocols**: Critical failure triggers human alerts
4. **Self-Diagnosis**: System can identify and report its own issues

## Monitoring & Alerts

### Health Metrics

- **Data Quality Score**: Overall data integrity (0-1)
- **Endpoint Availability**: Percentage of healthy endpoints
- **Response Times**: API response performance
- **Error Rates**: Failure frequency tracking

### Alert Types

- **Data Quality Alerts**: When data integrity falls below threshold
- **Endpoint Failure Alerts**: When critical APIs become unavailable
- **System Performance Alerts**: When response times degrade
- **Capacity Alerts**: When system resources are constrained

### Dashboard Features

- Real-time health visualization
- Historical performance charts
- Alert management interface
- Manual repair controls
- Configuration management

## Advanced Configuration

### Custom Endpoints

Add custom endpoints to the configuration:

```json
{
  "endpoints": {
    "custom_api": [
      {
        "name": "my_custom_endpoint",
        "url": "https://api.myservice.com",
        "endpoint_type": "market_data",
        "priority": 5,
        "rate_limit_requests": 100,
        "rate_limit_window_seconds": 60,
        "health_check_endpoint": "/health"
      }
    ]
  }
}
```

### Custom Healing Rules

Implement custom healing logic by extending the base classes:

```python
from mcp_trader.data.self_healing_data_manager import DataGapFiller

class CustomGapFiller(DataGapFiller):
    async def fill_gaps(self, symbol: str, df, gap_start, gap_end):
        # Custom gap filling logic
        return await self._custom_fill_method(symbol, df, gap_start, gap_end)
```

## Troubleshooting

### Common Issues

1. **Dashboard Not Starting**
   - Check if port 8080 is available
   - Verify configuration file exists
   - Check logs for initialization errors

2. **Data Repair Failing**
   - Verify fallback data sources are configured
   - Check API keys for external services
   - Review data quality thresholds

3. **High Memory Usage**
   - Reduce monitoring intervals
   - Enable data cleanup policies
   - Configure resource limits

### Logs and Debugging

```bash
# View system logs
tail -f logs/self_healing_system.log

# Check data health reports
cat data/data_health_report.json

# Monitor endpoint status
curl http://localhost:8080/endpoints/health
```

## Performance Considerations

### Optimization Strategies

1. **Monitoring Frequency**: Adjust check intervals based on needs
2. **Data Retention**: Configure automatic cleanup of old data
3. **Resource Limits**: Set memory and CPU usage limits
4. **Caching**: Enable intelligent caching for frequently accessed data

### Scaling Considerations

1. **Horizontal Scaling**: Multiple instances behind load balancer
2. **Data Partitioning**: Distribute data across multiple storage systems
3. **Endpoint Pooling**: Maintain larger pools of backup endpoints
4. **Monitoring Centralization**: Aggregate metrics from multiple instances

## Security Considerations

### Data Protection

- Encrypted storage of sensitive configuration
- Secure API key management
- Audit trails for all repair operations
- Access control for dashboard endpoints

### Network Security

- SSL/TLS verification for all external connections
- IP whitelisting for critical endpoints
- Request signing for authenticated APIs
- Rate limiting to prevent abuse

## Integration Guide

### Using with Existing Systems

```python
from mcp_trader.data.self_healing_endpoint_manager import SelfHealingEndpointManager

# Create endpoint manager
manager = SelfHealingEndpointManager()

# Make resilient API call
result = await manager.make_request(
    EndpointType.MARKET_DATA,
    method='GET',
    url_path='/api/v3/ticker/price'
)
```

### Custom Monitoring

```python
from mcp_trader.monitoring.self_healing_dashboard import create_dashboard

# Create custom dashboard
app = create_dashboard(data_manager, endpoint_manager)

# Add custom endpoints
@app.get("/custom/metrics")
async def custom_metrics():
    return {"custom_metric": 42}
```

## Conclusion

The Self-Healing Trading System provides enterprise-grade reliability for algorithmic trading operations. Through intelligent monitoring, automatic repair, and graceful degradation, it ensures maximum uptime and data integrity even in challenging market conditions.

For support or feature requests, please refer to the project documentation or create an issue in the repository.

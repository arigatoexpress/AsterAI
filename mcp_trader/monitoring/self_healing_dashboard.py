"""
Self-Healing System Dashboard

FastAPI endpoints for monitoring and controlling the self-healing system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemStatusResponse(BaseModel):
    """System status response model"""
    system_running: bool
    timestamp: float
    components: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    checks: Dict[str, Any]


class RepairRequest(BaseModel):
    """Repair request model"""
    symbol: str
    repair_type: str = "auto"  # auto, gap_fill, corruption_repair


class EndpointTestRequest(BaseModel):
    """Endpoint test request model"""
    endpoint_type: str
    test_data: Optional[Dict[str, Any]] = None


class SelfHealingDashboard:
    """
    Dashboard for monitoring and controlling self-healing systems
    """

    def __init__(self, data_manager=None, endpoint_manager=None):
        self.data_manager = data_manager
        self.endpoint_manager = endpoint_manager
        self.app = FastAPI(title="Self-Healing Trading System Dashboard")

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Basic health check endpoint"""
            return HealthCheckResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                checks={
                    "system": "operational",
                    "data_manager": "active" if self.data_manager else "inactive",
                    "endpoint_manager": "active" if self.endpoint_manager else "inactive"
                }
            )

        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status"""
            if hasattr(self, '_get_system_status'):
                return await self._get_system_status()
            else:
                # Mock response if managers not available
                return SystemStatusResponse(
                    system_running=True,
                    timestamp=asyncio.get_event_loop().time(),
                    components={
                        "data_manager": {"status": "mock"},
                        "endpoint_manager": {"status": "mock"}
                    }
                )

        @self.app.get("/data/health")
        async def get_data_health():
            """Get data health report"""
            if self.data_manager:
                return self.data_manager.get_health_report()
            else:
                raise HTTPException(status_code=503, detail="Data manager not available")

        @self.app.get("/endpoints/health")
        async def get_endpoints_health():
            """Get endpoints health report"""
            if self.endpoint_manager:
                return self.endpoint_manager.get_health_report()
            else:
                raise HTTPException(status_code=503, detail="Endpoint manager not available")

        @self.app.post("/data/repair")
        async def trigger_data_repair(request: RepairRequest, background_tasks: BackgroundTasks):
            """Trigger manual data repair"""
            if not self.data_manager:
                raise HTTPException(status_code=503, detail="Data manager not available")

            # Add repair task to background
            background_tasks.add_task(self._perform_data_repair, request.symbol, request.repair_type)

            return {"message": f"Repair initiated for {request.symbol}", "status": "running"}

        @self.app.post("/endpoints/test")
        async def test_endpoint(request: EndpointTestRequest):
            """Test endpoint functionality"""
            if not self.endpoint_manager:
                raise HTTPException(status_code=503, detail="Endpoint manager not available")

            try:
                from mcp_trader.data.self_healing_endpoint_manager import EndpointType

                endpoint_type = EndpointType(request.endpoint_type)
                result = await self.endpoint_manager.make_request(
                    endpoint_type,
                    method='GET',
                    url_path='/health'  # Test health endpoint
                )

                return {
                    "endpoint_type": request.endpoint_type,
                    "success": result is not None,
                    "response": result
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Endpoint test failed: {str(e)}")

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics for monitoring"""
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime": "N/A",  # Would need to track actual uptime
                "version": "1.0.0"
            }

            if self.data_manager:
                data_health = self.data_manager.get_health_report()
                metrics["data_health_score"] = data_health.get("overall_health", 0)

            if self.endpoint_manager:
                endpoint_health = self.endpoint_manager.get_health_report()
                metrics["endpoint_health_score"] = endpoint_health.get("overall_health", 0)

            return metrics

        @self.app.post("/system/restart")
        async def restart_system():
            """Restart self-healing systems (admin only)"""
            # This would require authentication in production
            logger.warning("System restart requested via API")

            try:
                # Restart data manager
                if self.data_manager:
                    await self.data_manager.stop_monitoring()
                    await self.data_manager.start_monitoring()

                # Restart endpoint manager
                if self.endpoint_manager:
                    await self.endpoint_manager.stop_monitoring()
                    await self.endpoint_manager.start_monitoring()

                return {"message": "System restart completed", "status": "success"}

            except Exception as e:
                logger.error(f"System restart failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}")

        @self.app.get("/alerts")
        async def get_alerts():
            """Get active alerts and warnings"""
            alerts = []

            if self.data_manager:
                data_health = self.data_manager.get_health_report()
                for symbol, health in data_health.get("asset_health", {}).items():
                    if health.get("quality_score", 1.0) < 0.8:
                        alerts.append({
                            "type": "data_quality",
                            "severity": "warning",
                            "symbol": symbol,
                            "message": f"Low data quality for {symbol}: {health['quality_score']:.2f}",
                            "timestamp": datetime.now().isoformat()
                        })

            if self.endpoint_manager:
                endpoint_health = self.endpoint_manager.get_health_report()
                for endpoint_name, health in endpoint_health.get("endpoints", {}).items():
                    if health.get("status") != "healthy":
                        alerts.append({
                            "type": "endpoint_health",
                            "severity": "error" if health.get("consecutive_failures", 0) > 3 else "warning",
                            "endpoint": endpoint_name,
                            "message": f"Endpoint {endpoint_name} is {health['status']}",
                            "timestamp": datetime.now().isoformat()
                        })

            return {"alerts": alerts, "total": len(alerts)}

    async def _perform_data_repair(self, symbol: str, repair_type: str):
        """Perform data repair operation"""
        try:
            logger.info(f"Starting manual repair for {symbol}, type: {repair_type}")

            if repair_type == "auto":
                success = await self.data_manager.force_repair(symbol)
            else:
                # Specific repair types would be implemented here
                success = False
                logger.warning(f"Specific repair type {repair_type} not implemented yet")

            logger.info(f"Manual repair for {symbol} completed: {'success' if success else 'failed'}")

        except Exception as e:
            logger.error(f"Manual repair failed for {symbol}: {str(e)}")

    async def _get_system_status(self) -> SystemStatusResponse:
        """Get comprehensive system status"""
        components = {}

        if self.data_manager:
            try:
                components["data_manager"] = {
                    "active": self.data_manager.monitoring_active,
                    "health_report": self.data_manager.get_health_report()
                }
            except Exception as e:
                components["data_manager"] = {"error": str(e)}

        if self.endpoint_manager:
            try:
                components["endpoint_manager"] = {
                    "active": self.endpoint_manager.monitoring_active,
                    "health_report": self.endpoint_manager.get_health_report()
                }
            except Exception as e:
                components["endpoint_manager"] = {"error": str(e)}

        return SystemStatusResponse(
            system_running=True,
            timestamp=asyncio.get_event_loop().time(),
            components=components
        )


# Global dashboard instance
dashboard = None


def create_dashboard(data_manager=None, endpoint_manager=None) -> FastAPI:
    """Create and return the dashboard FastAPI app"""
    global dashboard
    dashboard = SelfHealingDashboard(data_manager, endpoint_manager)
    return dashboard.app


def get_dashboard() -> SelfHealingDashboard:
    """Get the global dashboard instance"""
    return dashboard


# Example usage and testing
async def test_dashboard():
    """Test the dashboard functionality"""
    app = create_dashboard()

    # Test health endpoint (this would normally be done via HTTP requests)
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Test health check
    response = client.get("/health")
    assert response.status_code == 200

    health_data = response.json()
    assert health_data["status"] == "healthy"

    print("✅ Dashboard health check passed")

    # Test status endpoint
    response = client.get("/status")
    assert response.status_code == 200

    print("✅ Dashboard status check passed")

    print("✅ All dashboard tests passed")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_dashboard())

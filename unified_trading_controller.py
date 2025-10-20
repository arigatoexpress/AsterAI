"""
Unified Trading Controller
Manages hybrid cloud/local trading system for maximum profit
Routes ML training to local GPU, executes trades on cloud with local fallback
"""

import asyncio
import aiohttp
import os
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTradingController:
    """Manages hybrid cloud/local trading system for maximum reliability and performance"""
    
    def __init__(self):
        self.cloud_url = os.getenv("CLOUD_AGENT_URL", "https://aster-self-learning-trader-880429861698.us-central1.run.app")
        self.local_url = os.getenv("LOCAL_AGENT_URL", "http://localhost:8081")
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_combined_status(self) -> Dict[str, Any]:
        """Get status from both cloud and local systems"""
        status = {
            "cloud": {"status": "unknown", "available": False},
            "local": {"status": "unknown", "available": False},
            "timestamp": datetime.now().isoformat(),
            "primary_system": "cloud",
            "backup_system": "local"
        }
        
        # Check cloud
        try:
            async with self.session.get(f"{self.cloud_url}/status", timeout=5) as response:
                if response.status == 200:
                    status["cloud"] = await response.json()
                    status["cloud"]["available"] = True
                    logger.info("[OK] Cloud service available")
        except Exception as e:
            logger.warning(f"[WARNING] Cloud unavailable: {e}")
        
        # Check local
        try:
            async with self.session.get(f"{self.local_url}/api/control/status", timeout=2) as response:
                if response.status == 200:
                    status["local"] = await response.json()
                    status["local"]["available"] = True
                    logger.info("[OK] Local service available")
        except Exception as e:
            logger.warning(f"[WARNING] Local unavailable: {e}")
        
        # Determine primary system
        if status["cloud"]["available"] and status["local"]["available"]:
            status["mode"] = "hybrid"
            status["reliability"] = "maximum"
        elif status["cloud"]["available"]:
            status["mode"] = "cloud_only"
            status["reliability"] = "high"
        elif status["local"]["available"]:
            status["mode"] = "local_only"
            status["reliability"] = "medium"
        else:
            status["mode"] = "offline"
            status["reliability"] = "none"
        
        return status
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: str = "manual"
    ) -> Dict[str, Any]:
        """Execute trade on cloud (primary) or local (fallback)"""
        
        # Try cloud first (lower latency, always available)
        try:
            async with self.session.post(
                f"{self.cloud_url}/manual-trade",
                json={
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "strategy": strategy
                },
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"[OK] Trade executed on cloud: {symbol} {side} {quantity}")
                    return {
                        "success": True,
                        "execution": "cloud",
                        **result
                    }
                else:
                    logger.warning(f"[WARNING] Cloud trade failed: {response.status}")
        except Exception as e:
            logger.warning(f"[WARNING] Cloud trade failed: {e}")
        
        # Fallback to local
        logger.info("[INFO] Falling back to local execution...")
        try:
            async with self.session.post(
                f"{self.local_url}/api/control/execute-trade",
                json={"symbol": symbol, "side": side, "quantity": quantity},
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"[OK] Trade executed on local: {symbol} {side} {quantity}")
                    return {
                        "success": True,
                        "execution": "local",
                        **result
                    }
                else:
                    logger.error(f"[ERROR] Local trade failed: {response.status}")
        except Exception as e:
            logger.error(f"[ERROR] Local trade failed: {e}")
        
        # Both failed
        raise Exception("Both cloud and local execution failed")
    
    async def train_model(self, model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML model on local GPU (if available) or cloud CPU"""
        
        # Try local first (has GPU)
        if await self._is_local_available():
            logger.info(f"[INFO] Training {model_type} on local GPU...")
            try:
                async with self.session.post(
                    f"{self.local_url}/api/ml/train",
                    json={"model_type": model_type, "data": data},
                    timeout=300  # 5 min timeout for training
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"[OK] Model trained on local GPU: {model_type}")
                        return {
                            "success": True,
                            "execution": "local_gpu",
                            **result
                        }
            except Exception as e:
                logger.warning(f"[WARNING] Local training failed: {e}")
        
        # Fallback to cloud CPU training
        logger.info(f"[INFO] Training {model_type} on cloud CPU...")
        try:
            async with self.session.post(
                f"{self.cloud_url}/ml/train",
                json={"model_type": model_type, "data": data},
                timeout=600  # 10 min timeout for CPU training
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"[OK] Model trained on cloud CPU: {model_type}")
                    return {
                        "success": True,
                        "execution": "cloud_cpu",
                        **result
                    }
        except Exception as e:
            logger.error(f"[ERROR] Cloud training failed: {e}")
            raise Exception("Both local GPU and cloud CPU training failed")
    
    async def get_optimal_system(self) -> str:
        """Determine which system should handle which task"""
        status = await self.get_combined_status()
        
        if status["mode"] == "hybrid":
            return "Use cloud for trading, local for ML training"
        elif status["mode"] == "cloud_only":
            return "Cloud only - all operations on cloud"
        elif status["mode"] == "local_only":
            return "Local only - all operations on local PC"
        else:
            return "System offline - no services available"
    
    async def _is_local_available(self) -> bool:
        """Check if local system is available"""
        try:
            async with self.session.get(f"{self.local_url}/api/control/status", timeout=2) as response:
                return response.status == 200
        except:
            return False
    
    async def _is_cloud_available(self) -> bool:
        """Check if cloud system is available"""
        try:
            async with self.session.get(f"{self.cloud_url}/health", timeout=5) as response:
                return response.status == 200
        except:
            return False

# Global instance
_controller: Optional[UnifiedTradingController] = None

async def get_controller() -> UnifiedTradingController:
    """Get or create global controller instance"""
    global _controller
    if _controller is None:
        _controller = UnifiedTradingController()
        await _controller.__aenter__()
    return _controller

async def execute_trade_unified(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    """Execute trade using unified controller"""
    controller = await get_controller()
    return await controller.execute_trade(symbol, side, quantity)

async def train_model_unified(model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Train model using unified controller"""
    controller = await get_controller()
    return await controller.train_model(model_type, data)

async def get_system_status() -> Dict[str, Any]:
    """Get combined system status"""
    controller = await get_controller()
    return await controller.get_combined_status()

if __name__ == "__main__":
    # Test unified controller
    async def test():
        async with UnifiedTradingController() as controller:
            status = await controller.get_combined_status()
            print(json.dumps(status, indent=2))
            
            recommendation = await controller.get_optimal_system()
            print(f"\n[RECOMMENDATION] {recommendation}")
    
    asyncio.run(test())


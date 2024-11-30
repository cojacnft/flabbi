from fastapi import APIRouter, HTTPException
from typing import Dict
from ..services.resource_manager import ResourceManager, ResourceLimits

router = APIRouter()
resource_manager = ResourceManager()

@router.post("/mode/background")
async def set_background_mode():
    """Set system to background mode (low resource usage)."""
    try:
        await resource_manager.optimize_for_background()
        resource_manager.set_priority("low")
        return {"status": "success", "mode": "background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mode/boost")
async def set_boost_mode():
    """Set system to boost mode (high performance)."""
    try:
        await resource_manager.optimize_for_active()
        resource_manager.set_priority("high")
        return {"status": "success", "mode": "boost"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics() -> Dict:
    """Get current performance metrics."""
    try:
        return resource_manager.get_resource_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/limits")
async def update_limits(limits: ResourceLimits):
    """Update resource limits."""
    try:
        resource_manager.limits = limits
        await resource_manager._adjust_resources(
            resource_manager.metrics["cpu_usage"][-1] if resource_manager.metrics["cpu_usage"] else 0,
            resource_manager.metrics["memory_usage"][-1] if resource_manager.metrics["memory_usage"] else 0
        )
        return {"status": "success", "limits": limits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
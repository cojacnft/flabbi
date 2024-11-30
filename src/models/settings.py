from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class PerformanceMode(str, Enum):
    BACKGROUND = "background"
    BALANCED = "balanced"
    BOOST = "boost"

class MEVProtectionSettings(BaseModel):
    enabled: bool = True
    flashbots_enabled: bool = True
    eden_network_enabled: bool = False
    bloxroute_enabled: bool = False
    min_bribe_percentage: float = Field(0.1, ge=0, le=1)  # 0-100%
    frontrunning_protection: bool = True
    max_attempts: int = Field(3, ge=1, le=10)
    private_tx_only: bool = False

class GasOptimizationSettings(BaseModel):
    enabled: bool = True
    max_gas_price_gwei: int = Field(300, ge=0)
    priority_gas_percentage: float = Field(0.1, ge=0, le=1)
    min_profit_ratio: float = Field(0.1, ge=0)  # Minimum profit after gas
    gas_price_sources: List[str] = ["node", "etherscan", "blocknative"]
    auto_adjust: bool = True

class NetworkSettings(BaseModel):
    rpc_nodes: List[str] = []
    websocket_enabled: bool = True
    auto_failover: bool = True
    max_latency_ms: int = Field(1000, ge=0)
    mempool_monitoring: bool = True

class PathFindingSettings(BaseModel):
    max_hops: int = Field(4, ge=2, le=6)
    parallel_search: bool = True
    ml_prediction_enabled: bool = False
    min_success_rate: float = Field(0.8, ge=0, le=1)
    max_paths_per_token: int = Field(10, ge=1)
    min_profit_threshold_usd: float = Field(100.0, ge=0)

class RiskManagementSettings(BaseModel):
    max_slippage_percent: float = Field(1.0, ge=0, le=100)
    max_position_size_usd: float = Field(10000.0, ge=0)
    max_daily_loss_usd: float = Field(1000.0, ge=0)
    emergency_stop_loss_percent: float = Field(5.0, ge=0, le=100)
    revert_protection: bool = True
    auto_position_sizing: bool = True

class PerformanceSettings(BaseModel):
    mode: PerformanceMode = PerformanceMode.BACKGROUND
    max_cpu_percent: float = Field(70.0, ge=0, le=100)
    max_memory_percent: float = Field(80.0, ge=0, le=100)
    thread_pool_size: Optional[int] = None
    batch_size: int = Field(100, ge=1)

class SystemSettings(BaseModel):
    performance: PerformanceSettings
    mev_protection: MEVProtectionSettings
    gas_optimization: GasOptimizationSettings
    network: NetworkSettings
    path_finding: PathFindingSettings
    risk_management: RiskManagementSettings
    
    class Config:
        use_enum_values = True

# Default settings
DEFAULT_SETTINGS = SystemSettings(
    performance=PerformanceSettings(
        mode=PerformanceMode.BACKGROUND,
        max_cpu_percent=40.0,
        max_memory_percent=60.0,
        batch_size=50
    ),
    mev_protection=MEVProtectionSettings(),
    gas_optimization=GasOptimizationSettings(),
    network=NetworkSettings(),
    path_finding=PathFindingSettings(),
    risk_management=RiskManagementSettings()
)
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class GelatoCondition(BaseModel):
    min_profit_usd: float = Field(100.0, ge=0)
    max_gas_price_gwei: int = Field(300, ge=0)
    min_liquidity_usd: float = Field(100000.0, ge=0)
    max_slippage: float = Field(0.01, ge=0, le=1)

class GelatoConfig(BaseModel):
    enabled: bool = True
    use_private_mempool: bool = True
    auto_retry: bool = True
    max_retries: int = Field(3, ge=1)
    retry_delay_seconds: int = Field(1, ge=0)
    
    # Execution conditions
    conditions: GelatoCondition = GelatoCondition()
    
    # Gas settings
    priority_fee_percentage: float = Field(0.1, ge=0, le=1)
    max_base_fee_gwei: int = Field(100, ge=0)
    
    # Task settings
    auto_fund_tasks: bool = True
    min_task_balance_eth: float = Field(0.1, ge=0)
    max_task_age_hours: int = Field(24, ge=1)
    
    # Network settings
    preferred_executors: List[str] = []
    blacklisted_executors: List[str] = []
    
    # Monitoring
    monitor_interval_seconds: int = Field(5, ge=1)
    alert_on_failure: bool = True
    
    class Config:
        arbitrary_types_allowed = True
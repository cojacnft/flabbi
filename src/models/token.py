from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class TokenMetadata(BaseModel):
    name: str
    symbol: str
    decimals: int
    total_supply: str
    holders: Optional[int] = None
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class LiquidityPool(BaseModel):
    dex_name: str
    pool_address: str
    token0_address: str
    token1_address: str
    reserves_token0: str
    reserves_token1: str
    total_liquidity_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    fee_tier: Optional[float] = None  # e.g., 0.3% for Uniswap V2
    last_updated: datetime = Field(default_factory=datetime.utcnow)

class Token(BaseModel):
    address: str
    metadata: TokenMetadata
    liquidity_pools: List[LiquidityPool] = []
    is_active: bool = True
    is_verified: bool = False
    custom_tags: List[str] = []
    min_liquidity_threshold_usd: float = 10000.0  # Minimum liquidity required for arbitrage
    max_slippage: float = 0.005  # 0.5% default max slippage
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
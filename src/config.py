from typing import Dict, List
from pydantic import BaseModel
from web3 import Web3

class NetworkConfig(BaseModel):
    chain_id: int
    rpc_url: str
    backup_rpc_urls: List[str] = []

class FlashLoanConfig(BaseModel):
    providers: List[str]
    max_loan_amount: str
    priority_order: List[str]

class ProtocolConfig(BaseModel):
    enabled: List[str]
    blacklist: List[str]
    max_slippage: float

class SystemConfig(BaseModel):
    network: NetworkConfig
    flash_loan: FlashLoanConfig
    protocols: ProtocolConfig

# Default configuration
DEFAULT_CONFIG = SystemConfig(
    network=NetworkConfig(
        chain_id=1,  # Ethereum mainnet
        rpc_url="http://localhost:8545",  # Local node
        backup_rpc_urls=[],
    ),
    flash_loan=FlashLoanConfig(
        providers=["AAVE_V2", "BALANCER", "DODO"],
        max_loan_amount="1000000000000000000",  # 1 ETH
        priority_order=["AAVE_V2", "BALANCER", "DODO"],
    ),
    protocols=ProtocolConfig(
        enabled=["UNISWAP_V2", "SUSHISWAP", "CURVE"],
        blacklist=[],
        max_slippage=0.005,  # 0.5%
    ),
)
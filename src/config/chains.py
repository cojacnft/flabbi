from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class ChainType(str, Enum):
    MAINNET = "mainnet"
    TESTNET = "testnet"
    LOCAL = "local"

class FlashLoanProvider(BaseModel):
    name: str
    contract_address: str
    supported_tokens: List[str]
    fee_percentage: float
    min_amount: str = "0"  # In wei
    max_amount: Optional[str] = None  # In wei
    enabled: bool = True

class DEXProtocol(BaseModel):
    name: str
    factory_address: str
    router_address: str
    fee_percentage: float
    supported_tokens: List[str]
    enabled: bool = True
    min_liquidity_usd: float = 10000.0
    max_slippage: float = 0.005

class ChainConfig(BaseModel):
    chain_id: int
    name: str
    type: ChainType
    currency_symbol: str
    rpc_urls: List[str]
    ws_urls: Optional[List[str]] = None
    block_time: int  # Average block time in seconds
    flash_loan_providers: Dict[str, FlashLoanProvider]
    dex_protocols: Dict[str, DEXProtocol]
    token_standards: List[str] = ["ERC20"]
    explorer_url: Optional[str] = None
    explorer_api_key: Optional[str] = None
    
    # Gas settings
    gas_token_address: Optional[str] = None  # For chains that use different gas tokens
    max_gas_price: float  # in gwei
    priority_fee: float  # in gwei
    gas_multiplier: float = 1.1  # Safety multiplier for gas estimation
    
    # Chain-specific settings
    confirmations_required: int = 1
    max_pending_transactions: int = 5
    reorg_protection_blocks: int = 0

# Chain Configurations
ETHEREUM_MAINNET = ChainConfig(
    chain_id=1,
    name="Ethereum",
    type=ChainType.MAINNET,
    currency_symbol="ETH",
    rpc_urls=[
        "https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        "https://mainnet.infura.io/v3/${INFURA_KEY}"
    ],
    ws_urls=[
        "wss://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        "wss://mainnet.infura.io/ws/v3/${INFURA_KEY}"
    ],
    block_time=12,
    flash_loan_providers={
        "aave_v2": FlashLoanProvider(
            name="Aave V2",
            contract_address="0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            supported_tokens=[
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
            ],
            fee_percentage=0.0009  # 0.09%
        ),
        "balancer": FlashLoanProvider(
            name="Balancer",
            contract_address="0xBA12222222228d8Ba445958a75a0704d566BF2C8",
            supported_tokens=[
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
            ],
            fee_percentage=0.0
        )
    },
    dex_protocols={
        "uniswap_v2": DEXProtocol(
            name="Uniswap V2",
            factory_address="0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
            router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            fee_percentage=0.003,
            supported_tokens=[
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
            ]
        ),
        "sushiswap": DEXProtocol(
            name="SushiSwap",
            factory_address="0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
            router_address="0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            fee_percentage=0.003,
            supported_tokens=[
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
            ]
        )
    },
    explorer_url="https://etherscan.io",
    explorer_api_key="${ETHERSCAN_API_KEY}",
    max_gas_price=150.0,
    priority_fee=1.5,
    confirmations_required=3,
    reorg_protection_blocks=6
)

POLYGON_MAINNET = ChainConfig(
    chain_id=137,
    name="Polygon",
    type=ChainType.MAINNET,
    currency_symbol="MATIC",
    rpc_urls=[
        "https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        "https://polygon-rpc.com"
    ],
    ws_urls=[
        "wss://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}"
    ],
    block_time=2,
    flash_loan_providers={
        "aave_v2": FlashLoanProvider(
            name="Aave V2",
            contract_address="0x8dFf5E27EA6b7AC08EbFdf9eB090F32ee9a30fcf",
            supported_tokens=[
                "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
                "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC
                "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",  # USDT
                "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"   # DAI
            ],
            fee_percentage=0.0009
        )
    },
    dex_protocols={
        "quickswap": DEXProtocol(
            name="QuickSwap",
            factory_address="0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
            router_address="0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
            fee_percentage=0.003,
            supported_tokens=[
                "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
                "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC
                "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",  # USDT
                "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063"   # DAI
            ]
        )
    },
    explorer_url="https://polygonscan.com",
    explorer_api_key="${POLYGONSCAN_API_KEY}",
    max_gas_price=300.0,
    priority_fee=30.0,
    confirmations_required=1,
    max_pending_transactions=20
)

ARBITRUM_ONE = ChainConfig(
    chain_id=42161,
    name="Arbitrum One",
    type=ChainType.MAINNET,
    currency_symbol="ETH",
    rpc_urls=[
        "https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        "https://arbitrum-one.public.blastapi.io"
    ],
    ws_urls=[
        "wss://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}"
    ],
    block_time=1,
    flash_loan_providers={
        "aave_v3": FlashLoanProvider(
            name="Aave V3",
            contract_address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            supported_tokens=[
                "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH
                "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",  # USDC
                "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",  # USDT
                "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"   # DAI
            ],
            fee_percentage=0.0005
        )
    },
    dex_protocols={
        "sushiswap_v2": DEXProtocol(
            name="SushiSwap V2",
            factory_address="0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
            router_address="0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
            fee_percentage=0.003,
            supported_tokens=[
                "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH
                "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",  # USDC
                "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",  # USDT
                "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"   # DAI
            ]
        )
    },
    explorer_url="https://arbiscan.io",
    explorer_api_key="${ARBISCAN_API_KEY}",
    max_gas_price=1.0,  # Lower gas prices on Arbitrum
    priority_fee=0.1,
    confirmations_required=1
)

OPTIMISM = ChainConfig(
    chain_id=10,
    name="Optimism",
    type=ChainType.MAINNET,
    currency_symbol="ETH",
    rpc_urls=[
        "https://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}",
        "https://mainnet.optimism.io"
    ],
    ws_urls=[
        "wss://opt-mainnet.g.alchemy.com/v2/${ALCHEMY_KEY}"
    ],
    block_time=2,
    flash_loan_providers={
        "aave_v3": FlashLoanProvider(
            name="Aave V3",
            contract_address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            supported_tokens=[
                "0x4200000000000000000000000000000000000006",  # WETH
                "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",  # USDC
                "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",  # USDT
                "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"   # DAI
            ],
            fee_percentage=0.0005
        )
    },
    dex_protocols={
        "velodrome": DEXProtocol(
            name="Velodrome",
            factory_address="0x25CbdDb98b35ab1FF77413456B31EC81A6B6B746",
            router_address="0xa132DAB612dB5cB9fC9Ac426A0Cc215A3423F9c9",
            fee_percentage=0.002,
            supported_tokens=[
                "0x4200000000000000000000000000000000000006",  # WETH
                "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",  # USDC
                "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",  # USDT
                "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"   # DAI
            ]
        )
    },
    explorer_url="https://optimistic.etherscan.io",
    explorer_api_key="${OPTIMISM_API_KEY}",
    max_gas_price=0.1,
    priority_fee=0.01,
    confirmations_required=1
)

# Additional Chain Configurations
BSC_MAINNET = ChainConfig(
    chain_id=56,
    name="BNB Smart Chain",
    type=ChainType.MAINNET,
    currency_symbol="BNB",
    rpc_urls=[
        "https://bsc-dataseed.binance.org",
        "https://bsc-dataseed1.defibit.io",
        "https://bsc-dataseed1.ninicoin.io"
    ],
    ws_urls=[
        "wss://bsc-ws-node.nariox.org:443"
    ],
    block_time=3,
    flash_loan_providers={
        "pancakeswap": FlashLoanProvider(
            name="PancakeSwap",
            contract_address="0x10ED43C718714eb63d5aA57B78B54704E256024E",
            supported_tokens=[
                "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
                "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",  # BUSD
                "0x55d398326f99059fF775485246999027B3197955",  # USDT
                "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d"   # USDC
            ],
            fee_percentage=0.003
        ),
        "venus": FlashLoanProvider(
            name="Venus",
            contract_address="0xfD36E2c2a6789Db23113685031d7F16329158384",
            supported_tokens=[
                "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
                "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",  # BUSD
                "0x55d398326f99059fF775485246999027B3197955"   # USDT
            ],
            fee_percentage=0.0009
        )
    },
    dex_protocols={
        "pancakeswap": DEXProtocol(
            name="PancakeSwap",
            factory_address="0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
            router_address="0x10ED43C718714eb63d5aA57B78B54704E256024E",
            fee_percentage=0.0025,
            supported_tokens=[
                "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
                "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",  # BUSD
                "0x55d398326f99059fF775485246999027B3197955",  # USDT
                "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d"   # USDC
            ]
        ),
        "biswap": DEXProtocol(
            name="Biswap",
            factory_address="0x858E3312ed3A876947EA49d572A7C42DE08af7EE",
            router_address="0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8",
            fee_percentage=0.001,  # 0.1% fee
            supported_tokens=[
                "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
                "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",  # BUSD
                "0x55d398326f99059fF775485246999027B3197955"   # USDT
            ]
        )
    },
    explorer_url="https://bscscan.com",
    explorer_api_key="${BSCSCAN_API_KEY}",
    max_gas_price=5.0,  # BNB gas prices are lower
    priority_fee=0.1,
    confirmations_required=1,
    max_pending_transactions=50
)

AVALANCHE_MAINNET = ChainConfig(
    chain_id=43114,
    name="Avalanche",
    type=ChainType.MAINNET,
    currency_symbol="AVAX",
    rpc_urls=[
        "https://api.avax.network/ext/bc/C/rpc",
        "https://avalanche-c-chain.publicnode.com"
    ],
    ws_urls=[
        "wss://api.avax.network/ext/bc/C/ws"
    ],
    block_time=2,
    flash_loan_providers={
        "aave_v3": FlashLoanProvider(
            name="Aave V3",
            contract_address="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            supported_tokens=[
                "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
                "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # USDC
                "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",  # USDT
                "0xd586E7F844cEa2F87f50152665BCbc2C279D8d70"   # DAI
            ],
            fee_percentage=0.0005
        )
    },
    dex_protocols={
        "traderjoe": DEXProtocol(
            name="Trader Joe",
            factory_address="0x9Ad6C38BE94206cA50bb0d90783181662f0Cfa10",
            router_address="0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
            fee_percentage=0.003,
            supported_tokens=[
                "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
                "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # USDC
                "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",  # USDT
                "0xd586E7F844cEa2F87f50152665BCbc2C279D8d70"   # DAI
            ]
        ),
        "pangolin": DEXProtocol(
            name="Pangolin",
            factory_address="0xefa94DE7a4656D787667C749f7E1223D71E9FD88",
            router_address="0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106",
            fee_percentage=0.003,
            supported_tokens=[
                "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",  # WAVAX
                "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",  # USDC
                "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",  # USDT
                "0xd586E7F844cEa2F87f50152665BCbc2C279D8d70"   # DAI
            ]
        )
    },
    explorer_url="https://snowtrace.io",
    explorer_api_key="${SNOWTRACE_API_KEY}",
    max_gas_price=100.0,
    priority_fee=2.0,
    confirmations_required=1,
    max_pending_transactions=25
)

BASE_MAINNET = ChainConfig(
    chain_id=8453,
    name="Base",
    type=ChainType.MAINNET,
    currency_symbol="ETH",
    rpc_urls=[
        "https://mainnet.base.org",
        "https://base.gateway.tenderly.co"
    ],
    ws_urls=[
        "wss://base.gateway.tenderly.co"
    ],
    block_time=2,
    flash_loan_providers={
        "balancer": FlashLoanProvider(
            name="Balancer",
            contract_address="0xBA12222222228d8Ba445958a75a0704d566BF2C8",
            supported_tokens=[
                "0x4200000000000000000000000000000000000006",  # WETH
                "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb"   # DAI
            ],
            fee_percentage=0.0
        )
    },
    dex_protocols={
        "baseswap": DEXProtocol(
            name="BaseSwap",
            factory_address="0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB",
            router_address="0x327Df1E6de05895d2ab08513aaDD9313Fe505d86",
            fee_percentage=0.003,
            supported_tokens=[
                "0x4200000000000000000000000000000000000006",  # WETH
                "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC
                "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb"   # DAI
            ]
        )
    },
    explorer_url="https://basescan.org",
    explorer_api_key="${BASESCAN_API_KEY}",
    max_gas_price=0.1,
    priority_fee=0.001,
    confirmations_required=1
)

ZKSYNC_ERA = ChainConfig(
    chain_id=324,
    name="zkSync Era",
    type=ChainType.MAINNET,
    currency_symbol="ETH",
    rpc_urls=[
        "https://mainnet.era.zksync.io",
        "https://zksync.drpc.org"
    ],
    block_time=2,
    flash_loan_providers={
        "syncswap": FlashLoanProvider(
            name="SyncSwap",
            contract_address="0x2da10A1e27bF85cEdD8FFb1AbBe97e53391C0295",
            supported_tokens=[
                "0x5AEa5775959fBC2557Cc8789bC1bf90A239D9a91",  # WETH
                "0x3355df6D4c9C3035724Fd0e3914dE96A5a83aaf4",  # USDC
                "0x493257fD37EDB34451f62EDf8D2a0C418852bA4C"   # USDT
            ],
            fee_percentage=0.003
        )
    },
    dex_protocols={
        "syncswap": DEXProtocol(
            name="SyncSwap",
            factory_address="0xf2DAd89f2788a8CD54625C60b55cD3d2D0ACa7Cb",
            router_address="0x2da10A1e27bF85cEdD8FFb1AbBe97e53391C0295",
            fee_percentage=0.003,
            supported_tokens=[
                "0x5AEa5775959fBC2557Cc8789bC1bf90A239D9a91",  # WETH
                "0x3355df6D4c9C3035724Fd0e3914dE96A5a83aaf4",  # USDC
                "0x493257fD37EDB34451f62EDf8D2a0C418852bA4C"   # USDT
            ]
        )
    },
    explorer_url="https://explorer.zksync.io",
    explorer_api_key="${ZKSYNC_API_KEY}",
    max_gas_price=0.1,
    priority_fee=0.001,
    confirmations_required=1
)

# Map of supported chains
SUPPORTED_CHAINS = {
    1: ETHEREUM_MAINNET,
    137: POLYGON_MAINNET,
    42161: ARBITRUM_ONE,
    10: OPTIMISM,
    56: BSC_MAINNET,
    43114: AVALANCHE_MAINNET,
    8453: BASE_MAINNET,
    324: ZKSYNC_ERA
}

def get_chain_config(chain_id: int) -> Optional[ChainConfig]:
    """Get configuration for a specific chain."""
    return SUPPORTED_CHAINS.get(chain_id)

def is_chain_supported(chain_id: int) -> bool:
    """Check if a chain is supported."""
    return chain_id in SUPPORTED_CHAINS

def get_supported_chains() -> List[ChainConfig]:
    """Get list of all supported chains."""
    return list(SUPPORTED_CHAINS.values())
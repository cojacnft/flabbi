import pytest
import asyncio
from web3 import Web3
from eth_account import Account
import json
import os
from typing import Dict

from src.models.settings import SystemSettings, DEFAULT_SETTINGS
from src.services.network_manager import NetworkManager
from src.services.market_data import MarketDataAggregator
from src.services.simulator import TransactionSimulator
from src.services.profit_analyzer import ProfitAnalyzer
from src.services.gelato_service import GelatoService

# Test constants
TEST_PRIVATE_KEY = "0x" + "1" * 64
TEST_ADDRESS = Account.from_key(TEST_PRIVATE_KEY).address

# Test tokens (Mainnet addresses)
TEST_TOKENS = {
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
}

# Test DEX addresses
TEST_DEXES = {
    "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
    "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def web3():
    """Initialize Web3 instance with fork of mainnet."""
    return Web3(Web3.HTTPProvider("http://localhost:8545"))

@pytest.fixture(scope="session")
def settings():
    """Get test settings."""
    return DEFAULT_SETTINGS

@pytest.fixture(scope="session")
def account():
    """Get test account."""
    return Account.from_key(TEST_PRIVATE_KEY)

@pytest.fixture
async def network_manager(web3):
    """Initialize network manager."""
    manager = NetworkManager()
    await manager.initialize()
    yield manager
    await manager.cleanup()

@pytest.fixture
async def market_data(web3):
    """Initialize market data aggregator."""
    aggregator = MarketDataAggregator(web3)
    yield aggregator

@pytest.fixture
async def simulator(web3):
    """Initialize transaction simulator."""
    simulator = TransactionSimulator(web3)
    yield simulator

@pytest.fixture
async def profit_analyzer(web3, market_data, simulator):
    """Initialize profit analyzer."""
    analyzer = ProfitAnalyzer(web3, market_data, simulator)
    yield analyzer

@pytest.fixture
async def gelato_service(web3, account):
    """Initialize Gelato service."""
    service = GelatoService(web3, account.key.hex())
    yield service

@pytest.fixture
def mock_pool_data():
    """Get mock pool data for testing."""
    return {
        "uniswap_v2": {
            "WETH-USDC": {
                "address": "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
                "token0": TEST_TOKENS["WETH"],
                "token1": TEST_TOKENS["USDC"],
                "reserve0": "1000000000000000000000",  # 1000 WETH
                "reserve1": "2000000000000"  # 2M USDC
            }
        }
    }

@pytest.fixture
def mock_price_data():
    """Get mock price data for testing."""
    return {
        TEST_TOKENS["WETH"]: 2000.0,
        TEST_TOKENS["USDC"]: 1.0,
        TEST_TOKENS["USDT"]: 1.0,
        TEST_TOKENS["DAI"]: 1.0
    }

@pytest.fixture
def mock_opportunity():
    """Get mock arbitrage opportunity for testing."""
    return {
        "path_data": {
            "path_id": "test-path-1",
            "tokens": [
                TEST_TOKENS["WETH"],
                TEST_TOKENS["USDC"],
                TEST_TOKENS["WETH"]
            ],
            "dexes": [
                "uniswap_v2",
                "sushiswap"
            ]
        },
        "amount_in": "1000000000000000000",  # 1 WETH
        "amount_in_usd": 2000.0,
        "expected_output": "1050000000000000000",  # 1.05 WETH
        "flash_loan_data": {
            "target": TEST_DEXES["uniswap_v2"],
            "from": TEST_ADDRESS,
            "data": "0x" + "00" * 100
        }
    }

@pytest.fixture
def load_contract_abi():
    """Load contract ABI from file."""
    def _load_abi(contract_name: str) -> Dict:
        path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "abi",
            f"{contract_name}.json"
        )
        with open(path, "r") as f:
            return json.load(f)
    return _load_abi

# Helper functions for tests
async def simulate_network_latency():
    """Simulate network latency."""
    await asyncio.sleep(0.1)

async def simulate_block_confirmation():
    """Simulate block confirmation time."""
    await asyncio.sleep(0.2)

def create_test_transaction(web3, account, to, value=0, data="0x"):
    """Create a test transaction."""
    return {
        "from": account.address,
        "to": to,
        "value": value,
        "gas": 2000000,
        "gasPrice": web3.eth.gas_price,
        "nonce": web3.eth.get_transaction_count(account.address),
        "data": data
    }

def sign_test_transaction(web3, account, tx):
    """Sign a test transaction."""
    signed = account.sign_transaction(tx)
    return signed.rawTransaction
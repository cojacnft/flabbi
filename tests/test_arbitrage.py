import pytest
import asyncio
from web3 import Web3
from eth_account import Account
import os
import json
from decimal import Decimal

from src.services.flash_loan_executor import FlashLoanExecutor
from src.services.execution_strategy import ExecutionStrategyOptimizer
from src.services.market_analyzer import MarketAnalyzer
from src.services.parameter_tuner import ParameterTuner

# Constants
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

@pytest.fixture(scope="module")
def web3():
    """Initialize Web3 instance."""
    # Use Alchemy or Infura
    provider_url = os.getenv("WEB3_PROVIDER_URI", "http://localhost:8545")
    return Web3(Web3.HTTPProvider(provider_url))

@pytest.fixture(scope="module")
def account():
    """Create test account."""
    return Account.create()

@pytest.fixture(scope="module")
def contract(web3):
    """Load arbitrage contract."""
    with open("contracts/FlashLoanArbitrage.json") as f:
        contract_data = json.load(f)
    
    return web3.eth.contract(
        address=contract_data["address"],
        abi=contract_data["abi"]
    )

@pytest.fixture(scope="module")
def flash_loan_executor(web3, contract):
    """Initialize flash loan executor."""
    return FlashLoanExecutor(
        web3=web3,
        market_data_aggregator=MarketAnalyzer(web3),
        strategy_optimizer=ExecutionStrategyOptimizer(
            web3=web3,
            flash_loan_executor=None,  # Will be set after creation
            market_data_aggregator=MarketAnalyzer(web3),
            settings={"min_profit": 100.0}
        ),
        settings={
            "contract_address": contract.address,
            "min_profit": 100.0,
            "max_gas_price": 200,
            "execution_timeout": 30
        }
    )

@pytest.mark.asyncio
async def test_find_opportunity(web3, flash_loan_executor):
    """Test opportunity finding."""
    # Define test path
    path = [
        {
            "token_in": WETH,
            "token_out": USDC,
            "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"  # Uniswap V3
        },
        {
            "token_in": USDC,
            "token_out": DAI,
            "pool": "0x5777d92f208679db4b9778590fa3cab3ac9e2168"  # Uniswap V3
        },
        {
            "token_in": DAI,
            "token_out": WETH,
            "pool": "0x60594a405d53811d3bc4766596efd80fd545a270"  # Uniswap V3
        }
    ]

    # Find opportunity
    opportunity = await flash_loan_executor.find_opportunity(
        path=path,
        amount_in=Web3.to_wei(1, "ether")
    )

    assert opportunity is not None
    assert opportunity["expected_profit"] > 0
    assert len(opportunity["path"]) == len(path)

@pytest.mark.asyncio
async def test_validate_opportunity(web3, flash_loan_executor):
    """Test opportunity validation."""
    # Create test opportunity
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether")
    }

    # Validate
    is_valid = await flash_loan_executor.validate_opportunity(
        opportunity=opportunity,
        context={"network_load": 0.5}
    )

    assert is_valid is True

@pytest.mark.asyncio
async def test_execute_opportunity(web3, flash_loan_executor, contract):
    """Test opportunity execution."""
    # Create test opportunity
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether"),
        "flash_loan_config": {
            "provider": "aave_v2",
            "token_address": WETH,
            "amount": Web3.to_wei(1, "ether"),
            "fee": Decimal("0.0009")
        }
    }

    # Execute
    result = await flash_loan_executor.execute_opportunity(
        opportunity=opportunity,
        context={"network_load": 0.5}
    )

    assert result["success"] is True
    assert result["profit"] > 0
    assert result["gas_used"] > 0

@pytest.mark.asyncio
async def test_strategy_optimization(web3, flash_loan_executor):
    """Test strategy optimization."""
    optimizer = ExecutionStrategyOptimizer(
        web3=web3,
        flash_loan_executor=flash_loan_executor,
        market_data_aggregator=MarketAnalyzer(web3),
        settings={"min_profit": 100.0}
    )

    # Create test opportunity
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether")
    }

    # Get optimal strategy
    plan = await optimizer.optimize_execution(
        opportunity=opportunity,
        context={"network_load": 0.5}
    )

    assert plan is not None
    assert plan.strategy is not None
    assert plan.transactions is not None
    assert len(plan.transactions) > 0

@pytest.mark.asyncio
async def test_parameter_tuning(web3, flash_loan_executor):
    """Test parameter tuning."""
    tuner = ParameterTuner(
        strategy_optimizer=ExecutionStrategyOptimizer(
            web3=web3,
            flash_loan_executor=flash_loan_executor,
            market_data_aggregator=MarketAnalyzer(web3),
            settings={"min_profit": 100.0}
        ),
        risk_manager=None,  # Not needed for this test
        settings={"update_interval": 60}
    )

    # Get optimal parameters
    params = await tuner.get_optimal_parameters(
        context={
            "gas_price": Web3.to_wei(50, "gwei"),
            "network_load": 0.5,
            "volatility": 0.02
        }
    )

    assert params is not None
    assert "profit_threshold" in params
    assert "position_size" in params
    assert "slippage_tolerance" in params

@pytest.mark.asyncio
async def test_mev_protection(web3, flash_loan_executor):
    """Test MEV protection."""
    optimizer = ExecutionStrategyOptimizer(
        web3=web3,
        flash_loan_executor=flash_loan_executor,
        market_data_aggregator=MarketAnalyzer(web3),
        settings={"min_profit": 100.0}
    )

    # Create test opportunity
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether")
    }

    # Get execution plan with MEV protection
    plan = await optimizer.optimize_execution(
        opportunity=opportunity,
        context={
            "network_load": 0.5,
            "mev_activity": {"sandwich_count": 5}
        }
    )

    assert plan is not None
    assert plan.strategy.mev_protection is True
    assert plan.bundle_data is not None

@pytest.mark.asyncio
async def test_gas_optimization(web3, flash_loan_executor):
    """Test gas optimization."""
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether")
    }

    # Get gas estimate
    gas_estimate = await flash_loan_executor._estimate_gas(
        opportunity=opportunity
    )

    assert gas_estimate > 0
    assert gas_estimate < 1000000  # Reasonable gas limit

@pytest.mark.asyncio
async def test_profit_calculation(web3, flash_loan_executor):
    """Test profit calculation."""
    opportunity = {
        "token_in": WETH,
        "amount_in": Web3.to_wei(1, "ether"),
        "path": [
            {
                "token_in": WETH,
                "token_out": USDC,
                "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
            }
        ],
        "expected_profit": Web3.to_wei(0.1, "ether")
    }

    # Calculate profit
    profit = await flash_loan_executor._calculate_profit(
        opportunity=opportunity,
        gas_cost=Web3.to_wei(0.01, "ether")
    )

    assert profit > 0
    assert profit < opportunity["expected_profit"]  # Account for gas costs
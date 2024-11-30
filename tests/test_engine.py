import pytest
import asyncio
from web3 import Web3
from eth_typing import Address

from src.config import SystemConfig, DEFAULT_CONFIG
from src.core.engine import ArbitrageEngine, ArbitrageOpportunity

@pytest.fixture
def config():
    return DEFAULT_CONFIG

@pytest.fixture
def web3():
    return Web3(Web3.HTTPProvider('http://localhost:8545'))

@pytest.fixture
def engine(config, web3):
    return ArbitrageEngine(config, web3)

@pytest.mark.asyncio
async def test_find_opportunities(engine):
    opportunities = await engine.find_opportunities()
    assert isinstance(opportunities, list)

@pytest.mark.asyncio
async def test_validate_opportunity(engine):
    opportunity = ArbitrageOpportunity(
        token_in="0x1234...",
        token_out="0x5678...",
        amount_in=1000000,
        expected_profit=100000,
        route=["UNISWAP_V2", "SUSHISWAP"],
        gas_estimate=200000
    )
    is_valid = await engine.validate_opportunity(opportunity)
    assert isinstance(is_valid, bool)

@pytest.mark.asyncio
async def test_execute_arbitrage(engine):
    opportunity = ArbitrageOpportunity(
        token_in="0x1234...",
        token_out="0x5678...",
        amount_in=1000000,
        expected_profit=100000,
        route=["UNISWAP_V2", "SUSHISWAP"],
        gas_estimate=200000
    )
    success = await engine.execute_arbitrage(opportunity)
    assert isinstance(success, bool)
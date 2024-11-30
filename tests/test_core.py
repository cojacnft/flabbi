import pytest
import asyncio
from web3 import Web3
from decimal import Decimal
from datetime import datetime, timedelta

from src.core.arbitrage_engine import ArbitrageEngine
from src.models.settings import SystemSettings

pytestmark = pytest.mark.asyncio

async def test_market_data_price_fetching(market_data, mock_price_data):
    """Test price fetching from multiple sources."""
    # Test price fetching for WETH
    price = await market_data.get_token_price(
        TEST_TOKENS["WETH"],
        force_refresh=True
    )
    
    assert price is not None
    assert isinstance(price, float)
    assert price > 0
    
    # Test price caching
    cached_price = await market_data.get_token_price(
        TEST_TOKENS["WETH"]
    )
    assert cached_price == price
    
    # Test multiple tokens
    prices = await asyncio.gather(*[
        market_data.get_token_price(addr)
        for addr in TEST_TOKENS.values()
    ])
    
    assert all(p is not None for p in prices)
    assert all(isinstance(p, float) for p in prices)
    assert all(p > 0 for p in prices)

async def test_market_depth_calculation(market_data, mock_pool_data):
    """Test market depth calculation."""
    # Test depth calculation for WETH-USDC pair
    depth_score, price_impact = await market_data.get_market_depth(
        TEST_TOKENS["WETH"],
        1000.0  # $1000 trade
    )
    
    assert 0 <= depth_score <= 1
    assert 0 <= price_impact <= 1
    
    # Test larger trade size
    depth_score_large, price_impact_large = await market_data.get_market_depth(
        TEST_TOKENS["WETH"],
        100000.0  # $100k trade
    )
    
    # Larger trade should have higher price impact
    assert price_impact_large > price_impact
    
    # Larger trade should have lower depth score
    assert depth_score_large < depth_score

async def test_simulation(simulator, mock_opportunity):
    """Test transaction simulation."""
    # Simulate arbitrage transaction
    result = await simulator.simulate_arbitrage(
        mock_opportunity["flash_loan_data"],
        mock_opportunity["path_data"]
    )
    
    assert result is not None
    assert "success" in result
    assert "gas_used" in result
    assert "profit" in result
    
    # Test batch simulation
    opportunities = [mock_opportunity] * 3
    results = await simulator.simulate_batch(opportunities)
    
    assert len(results) <= len(opportunities)
    assert all("success" in r for r in results)

async def test_profit_analysis(
    profit_analyzer,
    mock_opportunity,
    mock_price_data
):
    """Test profit analysis."""
    # Analyze opportunity
    analysis = await profit_analyzer.analyze_opportunity(mock_opportunity)
    
    assert "profitable" in analysis
    assert "expected_profit" in analysis
    assert "success_probability" in analysis
    assert "risk_factors" in analysis
    
    # Test with simulation result
    simulation_result = {
        "success": True,
        "gas_used": 200000,
        "profit": 50.0
    }
    
    analysis_with_sim = await profit_analyzer.analyze_opportunity(
        mock_opportunity,
        simulation_result
    )
    
    assert "simulated_profit" in analysis_with_sim
    assert "profit_accuracy" in analysis_with_sim

async def test_gelato_integration(gelato_service, mock_opportunity):
    """Test Gelato service integration."""
    # Create execution task
    task_id = await gelato_service.create_arbitrage_task(
        mock_opportunity["flash_loan_data"],
        {
            "minProfit": 50.0,
            "maxGasPrice": 100,
            "minLiquidity": 1000000,
            "maxSlippage": 0.01
        }
    )
    
    assert task_id is not None
    
    # Monitor task
    status = await gelato_service.monitor_task(task_id)
    assert "status" in status

async def test_network_management(network_manager):
    """Test network management."""
    # Test RPC request
    result = await network_manager.make_request(
        "eth_blockNumber",
        []
    )
    assert result is not None
    
    # Test gas price fetching
    gas_price = await network_manager.get_best_gas_price()
    assert gas_price is not None
    assert gas_price > 0
    
    # Test provider status
    status = network_manager.get_provider_status()
    assert "alchemy" in status
    assert "gelato" in status

async def test_full_arbitrage_flow(
    web3,
    settings,
    account,
    mock_opportunity,
    market_data,
    simulator,
    profit_analyzer,
    gelato_service
):
    """Test complete arbitrage flow."""
    # Initialize engine
    engine = ArbitrageEngine(settings, web3, account.key.hex())
    
    # Start engine
    await engine.start()
    
    try:
        # Wait for initial setup
        await asyncio.sleep(2)
        
        # Check metrics
        metrics = engine.get_metrics()
        assert "opportunities_found" in metrics
        assert "successful_executions" in metrics
        assert "total_profit_usd" in metrics
        
        # Test opportunity processing
        opportunities = await engine._find_opportunities_batch([
            mock_opportunity["path_data"]["tokens"][0]
        ])
        assert isinstance(opportunities, list)
        
        if opportunities:
            opp = opportunities[0]
            
            # Validate opportunity
            is_valid = await engine._validate_opportunity(opp)
            assert isinstance(is_valid, bool)
            
            if is_valid:
                # Create execution task
                await engine._create_execution_task(opp)
                
                # Check active tasks
                assert len(engine.active_tasks) > 0
                
                # Wait for execution
                await asyncio.sleep(5)
                
                # Check metrics again
                new_metrics = engine.get_metrics()
                assert new_metrics["total_profit_usd"] >= metrics["total_profit_usd"]
    
    finally:
        # Cleanup
        await engine.cleanup()

@pytest.mark.parametrize("token_address", list(TEST_TOKENS.values()))
async def test_token_price_monitoring(market_data, token_address):
    """Test price monitoring for different tokens."""
    # Get initial price
    price1 = await market_data.get_token_price(token_address)
    assert price1 is not None
    
    # Wait and get updated price
    await asyncio.sleep(1)
    price2 = await market_data.get_token_price(
        token_address,
        force_refresh=True
    )
    assert price2 is not None
    
    # Prices should be similar (within 1%)
    assert abs(price1 - price2) / price1 < 0.01

@pytest.mark.parametrize("amount_usd", [1000, 10000, 100000])
async def test_market_depth_scaling(market_data, mock_pool_data, amount_usd):
    """Test market depth calculation with different amounts."""
    depth_score, price_impact = await market_data.get_market_depth(
        TEST_TOKENS["WETH"],
        amount_usd
    )
    
    assert 0 <= depth_score <= 1
    assert 0 <= price_impact <= 1
    
    # Larger amounts should have higher price impact
    if amount_usd > 1000:
        small_impact = (await market_data.get_market_depth(
            TEST_TOKENS["WETH"],
            1000
        ))[1]
        assert price_impact > small_impact

@pytest.mark.parametrize("gas_price", [50, 100, 200])
async def test_profit_analysis_gas_sensitivity(
    profit_analyzer,
    mock_opportunity,
    gas_price
):
    """Test profit analysis with different gas prices."""
    # Modify gas price in mock data
    mock_opportunity["flash_loan_data"]["gasPrice"] = Web3.to_wei(
        gas_price,
        "gwei"
    )
    
    analysis = await profit_analyzer.analyze_opportunity(mock_opportunity)
    
    # Higher gas prices should reduce profitability
    if gas_price > 100:
        assert analysis["expected_profit"] < mock_opportunity["amount_in_usd"] * 0.01

async def test_error_handling(
    network_manager,
    market_data,
    simulator,
    profit_analyzer
):
    """Test error handling in various components."""
    # Test invalid token address
    price = await market_data.get_token_price(
        "0x" + "0" * 40
    )
    assert price is None
    
    # Test invalid RPC request
    result = await network_manager.make_request(
        "invalid_method",
        []
    )
    assert result is None
    
    # Test simulation with invalid data
    result = await simulator.simulate_arbitrage(
        {"invalid": "data"},
        {"invalid": "data"}
    )
    assert result is None or not result["success"]
    
    # Test profit analysis with invalid opportunity
    analysis = await profit_analyzer.analyze_opportunity({})
    assert not analysis["profitable"]
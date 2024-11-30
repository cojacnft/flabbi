from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor

class DEXOptimizer:
    """Optimize cross-DEX arbitrage opportunities."""
    def __init__(
        self,
        web3: Web3,
        chain_id: int,
        market_data_aggregator,
        settings: Dict
    ):
        self.web3 = web3
        self.chain_id = chain_id
        self.market_data = market_data_aggregator
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # DEX configurations
        self.dex_configs = {
            "uniswap_v2": {
                "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "fee": 0.003,  # 0.3%
                "enabled": True,
                "priority": 1
            },
            "sushiswap": {
                "factory": "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                "router": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                "fee": 0.003,
                "enabled": True,
                "priority": 2
            },
            "shibaswap": {
                "factory": "0x115934131916C8b277DD010Ee02de363c09d037c",
                "router": "0x03f7724180AA6b939894B5Ca4314783B0b36b329",
                "fee": 0.003,
                "enabled": True,
                "priority": 3
            },
            "defiswap": {
                "factory": "0x9DEB29c9a4c7A88a3C0257393b7f3335338D9A9D",
                "router": "0x6C9b3A47a28a39B097B5B08D7Aed5d574D2055B0",
                "fee": 0.003,
                "enabled": True,
                "priority": 4
            }
        }
        
        # Pool cache
        self.pool_cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Performance metrics
        self.metrics = {
            "paths_analyzed": 0,
            "opportunities_found": 0,
            "total_profit_found": 0.0,
            "avg_analysis_time_ms": 0.0
        }
        
        # Initialize contracts
        self._init_contracts()
        
        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def _init_contracts(self):
        """Initialize DEX contracts."""
        try:
            self.contracts = {}
            
            for dex_name, config in self.dex_configs.items():
                if not config["enabled"]:
                    continue
                
                # Initialize factory contract
                factory_contract = self.web3.eth.contract(
                    address=config["factory"],
                    abi=self._get_factory_abi()
                )
                
                # Initialize router contract
                router_contract = self.web3.eth.contract(
                    address=config["router"],
                    abi=self._get_router_abi()
                )
                
                self.contracts[dex_name] = {
                    "factory": factory_contract,
                    "router": router_contract
                }
            
            self.logger.info(f"Initialized {len(self.contracts)} DEX contracts")
            
        except Exception as e:
            self.logger.error(f"Error initializing contracts: {str(e)}")
            raise

    def _get_factory_abi(self) -> List:
        """Get factory contract ABI."""
        return [
            {
                "inputs": [
                    {"internalType": "address", "name": "tokenA", "type": "address"},
                    {"internalType": "address", "name": "tokenB", "type": "address"}
                ],
                "name": "getPair",
                "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    def _get_router_abi(self) -> List:
        """Get router contract ABI."""
        return [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {"internalType": "address[]", "name": "path", "type": "address[]"}
                ],
                "name": "getAmountsOut",
                "outputs": [{"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    async def find_optimal_path(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        max_hops: int = 3
    ) -> Optional[Dict]:
        """Find optimal arbitrage path across DEXes."""
        try:
            start_time = datetime.utcnow()
            
            # Get all possible paths
            paths = await self._generate_paths(
                token_in,
                token_out,
                max_hops
            )
            
            # Analyze paths in parallel
            results = await asyncio.gather(*[
                self._analyze_path(path, amount_in)
                for path in paths
            ])
            
            # Filter valid results
            valid_results = [r for r in results if r is not None]
            
            if not valid_results:
                return None
            
            # Sort by profit
            valid_results.sort(
                key=lambda x: x["expected_profit"],
                reverse=True
            )
            
            # Update metrics
            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            self.metrics["paths_analyzed"] += len(paths)
            self.metrics["avg_analysis_time_ms"] = (
                self.metrics["avg_analysis_time_ms"] *
                self.metrics["opportunities_found"] +
                execution_time
            ) / (self.metrics["opportunities_found"] + 1)
            
            best_result = valid_results[0]
            if best_result["expected_profit"] > 0:
                self.metrics["opportunities_found"] += 1
                self.metrics["total_profit_found"] += best_result["expected_profit"]
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Error finding optimal path: {str(e)}")
            return None

    async def _generate_paths(
        self,
        token_in: str,
        token_out: str,
        max_hops: int
    ) -> List[List[Dict]]:
        """Generate all possible paths between tokens."""
        try:
            paths = []
            visited = {token_in}
            
            async def explore_path(current_path: List[Dict], current_token: str):
                if len(current_path) > max_hops:
                    return
                
                # If we reached token_out with at least one hop
                if current_token == token_out and len(current_path) > 0:
                    paths.append(current_path.copy())
                    return
                
                # Get connected tokens
                connected = await self._get_connected_tokens(current_token)
                
                for next_token, pools in connected.items():
                    if next_token not in visited:
                        visited.add(next_token)
                        for pool in pools:
                            current_path.append({
                                "token_in": current_token,
                                "token_out": next_token,
                                "pool": pool["address"],
                                "dex": pool["dex"]
                            })
                            await explore_path(current_path, next_token)
                            current_path.pop()
                        visited.remove(next_token)
            
            await explore_path([], token_in)
            return paths
            
        except Exception as e:
            self.logger.error(f"Error generating paths: {str(e)}")
            return []

    async def _get_connected_tokens(
        self,
        token: str
    ) -> Dict[str, List[Dict]]:
        """Get tokens connected through liquidity pools."""
        try:
            connected: Dict[str, List[Dict]] = {}
            
            # Check each DEX
            for dex_name, contracts in self.contracts.items():
                factory = contracts["factory"]
                
                # Get common pairs from cache or query
                pairs = await self._get_token_pairs(
                    token,
                    dex_name,
                    factory
                )
                
                for pair in pairs:
                    other_token = (
                        pair["token1"]
                        if pair["token0"].lower() == token.lower()
                        else pair["token0"]
                    )
                    
                    if other_token not in connected:
                        connected[other_token] = []
                    
                    connected[other_token].append({
                        "address": pair["address"],
                        "dex": dex_name
                    })
            
            return connected
            
        except Exception as e:
            self.logger.error(f"Error getting connected tokens: {str(e)}")
            return {}

    async def _get_token_pairs(
        self,
        token: str,
        dex_name: str,
        factory_contract
    ) -> List[Dict]:
        """Get token pairs from cache or query."""
        try:
            cache_key = f"{dex_name}_{token}"
            
            # Check cache
            cached = self.pool_cache.get(cache_key)
            if cached and datetime.utcnow() - cached["timestamp"] < self.cache_duration:
                return cached["pairs"]
            
            # Query pairs
            pairs = []
            
            # This would normally use event logs or graph queries
            # For now, using a simplified approach
            common_tokens = await self.market_data.get_common_pairs(token)
            
            for other_token in common_tokens:
                pair_address = await factory_contract.functions.getPair(
                    token,
                    other_token
                ).call()
                
                if pair_address != "0x" + "0" * 40:
                    pairs.append({
                        "address": pair_address,
                        "token0": token,
                        "token1": other_token
                    })
            
            # Update cache
            self.pool_cache[cache_key] = {
                "pairs": pairs,
                "timestamp": datetime.utcnow()
            }
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error getting token pairs: {str(e)}")
            return []

    async def _analyze_path(
        self,
        path: List[Dict],
        amount_in: int
    ) -> Optional[Dict]:
        """Analyze arbitrage path."""
        try:
            current_amount = amount_in
            total_fee = 1.0
            amounts = [amount_in]
            gas_cost = 0
            
            # Simulate trades
            for step in path:
                # Get pool info
                pool_info = await self._get_pool_info(
                    step["pool"],
                    step["dex"]
                )
                
                if not pool_info:
                    return None
                
                # Calculate output amount
                output = await self._calculate_output_amount(
                    current_amount,
                    pool_info,
                    step["dex"]
                )
                
                if output == 0:
                    return None
                
                # Update state
                current_amount = output
                amounts.append(output)
                total_fee *= (1 - self.dex_configs[step["dex"]]["fee"])
                gas_cost += self._estimate_step_gas(step["dex"])
            
            # Calculate metrics
            profit = self._calculate_profit(
                amount_in,
                current_amount,
                path[0]["token_in"]
            )
            
            gas_price = await self.web3.eth.gas_price
            gas_cost_eth = gas_cost * gas_price
            gas_cost_usd = gas_cost_eth * await self._get_eth_price()
            
            net_profit = profit - gas_cost_usd
            
            if net_profit <= 0:
                return None
            
            return {
                "path": path,
                "amounts": amounts,
                "expected_profit": net_profit,
                "gas_cost": gas_cost_usd,
                "total_fee": total_fee,
                "confidence": self._calculate_confidence(
                    path,
                    amounts,
                    pool_info
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing path: {str(e)}")
            return None

    async def _get_pool_info(
        self,
        pool_address: str,
        dex_name: str
    ) -> Optional[Dict]:
        """Get pool information."""
        try:
            # Check cache
            cache_key = f"pool_{pool_address}"
            cached = self.pool_cache.get(cache_key)
            if cached and datetime.utcnow() - cached["timestamp"] < self.cache_duration:
                return cached["info"]
            
            # Get pool contract
            pool = self.web3.eth.contract(
                address=pool_address,
                abi=self._get_pool_abi()
            )
            
            # Get pool data
            reserves = await pool.functions.getReserves().call()
            token0 = await pool.functions.token0().call()
            token1 = await pool.functions.token1().call()
            
            info = {
                "address": pool_address,
                "token0": token0,
                "token1": token1,
                "reserve0": reserves[0],
                "reserve1": reserves[1],
                "fee": self.dex_configs[dex_name]["fee"]
            }
            
            # Update cache
            self.pool_cache[cache_key] = {
                "info": info,
                "timestamp": datetime.utcnow()
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting pool info: {str(e)}")
            return None

    async def _calculate_output_amount(
        self,
        amount_in: int,
        pool_info: Dict,
        dex_name: str
    ) -> int:
        """Calculate output amount for swap."""
        try:
            # Get reserves
            reserve_in = pool_info["reserve0"]
            reserve_out = pool_info["reserve1"]
            
            # Calculate fee
            amount_with_fee = amount_in * (1 - pool_info["fee"])
            
            # Calculate output using constant product formula
            numerator = amount_with_fee * reserve_out
            denominator = reserve_in + amount_with_fee
            
            return numerator // denominator
            
        except Exception as e:
            self.logger.error(f"Error calculating output amount: {str(e)}")
            return 0

    def _estimate_step_gas(self, dex_name: str) -> int:
        """Estimate gas cost for swap step."""
        # Approximate gas costs
        gas_costs = {
            "uniswap_v2": 90000,
            "sushiswap": 90000,
            "shibaswap": 100000,
            "defiswap": 95000
        }
        
        return gas_costs.get(dex_name, 100000)

    def _calculate_confidence(
        self,
        path: List[Dict],
        amounts: List[int],
        pool_info: Dict
    ) -> float:
        """Calculate confidence score for path."""
        try:
            # Base confidence
            confidence = 1.0
            
            # Adjust for path length
            confidence *= max(0.5, 1 - (len(path) * 0.1))
            
            # Adjust for amount relative to liquidity
            for i, step in enumerate(path):
                amount = amounts[i]
                reserve = pool_info["reserve0"]  # Simplified
                ratio = amount / reserve
                if ratio > 0.1:  # More than 10% of liquidity
                    confidence *= 0.8
                elif ratio > 0.05:  # More than 5% of liquidity
                    confidence *= 0.9
            
            # Adjust for DEX reliability
            for step in path:
                dex_priority = self.dex_configs[step["dex"]]["priority"]
                confidence *= (1 - (dex_priority - 1) * 0.1)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    async def prepare_execution(
        self,
        path_result: Dict
    ) -> Optional[Dict]:
        """Prepare path for execution."""
        try:
            execution_steps = []
            
            for i, step in enumerate(path_result["path"]):
                # Get router contract
                router = self.contracts[step["dex"]]["router"]
                
                # Prepare swap data
                swap_data = await self._prepare_swap_data(
                    router,
                    step,
                    path_result["amounts"][i],
                    path_result["amounts"][i+1]
                )
                
                execution_steps.append({
                    "dex": step["dex"],
                    "contract": router.address,
                    "data": swap_data,
                    "value": 0
                })
            
            return {
                "steps": execution_steps,
                "expected_profit": path_result["expected_profit"],
                "gas_estimate": sum(
                    self._estimate_step_gas(step["dex"])
                    for step in path_result["path"]
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing execution: {str(e)}")
            return None

    async def _prepare_swap_data(
        self,
        router,
        step: Dict,
        amount_in: int,
        min_amount_out: int
    ) -> str:
        """Prepare swap transaction data."""
        try:
            # Calculate minimum output with slippage
            min_out = int(min_amount_out * 0.995)  # 0.5% slippage
            
            # Encode swap function
            return router.encodeABI(
                fn_name="swapExactTokensForTokens",
                args=[
                    amount_in,
                    min_out,
                    [step["token_in"], step["token_out"]],
                    self.web3.eth.default_account,
                    int(datetime.utcnow().timestamp() + 300)  # 5 min deadline
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing swap data: {str(e)}")
            return ""

    async def update_pool_state(
        self,
        pool_address: str,
        force: bool = False
    ):
        """Update pool state in cache."""
        try:
            cache_key = f"pool_{pool_address}"
            
            # Check if update needed
            cached = self.pool_cache.get(cache_key)
            if not force and cached and datetime.utcnow() - cached["timestamp"] < self.cache_duration:
                return
            
            # Get updated pool info
            for dex_name in self.dex_configs:
                info = await self._get_pool_info(pool_address, dex_name)
                if info:
                    self.pool_cache[cache_key] = {
                        "info": info,
                        "timestamp": datetime.utcnow()
                    }
                    break
            
        except Exception as e:
            self.logger.error(f"Error updating pool state: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["opportunities_found"] /
                max(1, self.metrics["paths_analyzed"])
            ),
            "avg_profit_per_opportunity": (
                self.metrics["total_profit_found"] /
                max(1, self.metrics["opportunities_found"])
            ),
            "cache_size": len(self.pool_cache),
            "last_update": datetime.utcnow().isoformat()
        }
from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import networkx as nx
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class RouteNode:
    """Node in routing graph."""
    token: str
    dex: str
    pool: str
    reserve_in: int
    reserve_out: int
    fee: float

@dataclass
class Route:
    """Complete arbitrage route."""
    path: List[RouteNode]
    amounts: List[int]
    expected_profit: float
    gas_cost: float
    total_fee: float
    confidence: float

class RouteOptimizer:
    """Optimize arbitrage routes using advanced pathfinding."""
    def __init__(
        self,
        web3: Web3,
        dex_optimizer,
        market_data_aggregator,
        settings: Dict
    ):
        self.web3 = web3
        self.dex = dex_optimizer
        self.market_data = market_data_aggregator
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Routing graph
        self.graph = nx.DiGraph()
        self.last_update = datetime.min
        
        # Route cache
        self.route_cache: Dict[str, Dict] = {}
        self.cache_duration = timedelta(minutes=1)
        
        # Performance metrics
        self.metrics = {
            "routes_analyzed": 0,
            "profitable_routes": 0,
            "total_profit_found": 0.0,
            "avg_analysis_time_ms": 0.0
        }
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def find_arbitrage_routes(
        self,
        token_in: str,
        amount_in: int,
        min_profit: float = 0.0,
        max_hops: int = 4
    ) -> List[Route]:
        """Find profitable arbitrage routes."""
        try:
            start_time = datetime.utcnow()
            
            # Update routing graph
            await self._update_graph()
            
            # Find candidate paths
            paths = self._find_candidate_paths(
                token_in,
                max_hops
            )
            
            # Analyze paths in parallel
            routes = await asyncio.gather(*[
                self._analyze_route(path, amount_in)
                for path in paths
            ])
            
            # Filter profitable routes
            profitable_routes = [
                route for route in routes
                if route and route.expected_profit >= min_profit
            ]
            
            # Sort by expected profit
            profitable_routes.sort(
                key=lambda x: x.expected_profit,
                reverse=True
            )
            
            # Update metrics
            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            self.metrics["routes_analyzed"] += len(paths)
            self.metrics["avg_analysis_time_ms"] = (
                self.metrics["avg_analysis_time_ms"] *
                self.metrics["profitable_routes"] +
                execution_time
            ) / (self.metrics["profitable_routes"] + 1)
            
            if profitable_routes:
                self.metrics["profitable_routes"] += 1
                self.metrics["total_profit_found"] += sum(
                    r.expected_profit for r in profitable_routes
                )
            
            return profitable_routes
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage routes: {str(e)}")
            return []

    async def _update_graph(self):
        """Update routing graph with current market state."""
        try:
            # Check if update needed
            if datetime.utcnow() - self.last_update < timedelta(seconds=30):
                return
            
            # Clear existing graph
            self.graph.clear()
            
            # Get active tokens
            tokens = await self.market_data.get_active_tokens()
            
            # Add nodes and edges
            for token in tokens:
                # Get pools for token
                pools = await self._get_token_pools(token)
                
                for pool in pools:
                    # Add nodes if they don't exist
                    if not self.graph.has_node(pool.token):
                        self.graph.add_node(
                            pool.token,
                            type="token"
                        )
                    
                    # Add edges (both directions)
                    self.graph.add_edge(
                        pool.token,
                        pool.token_out,
                        dex=pool.dex,
                        pool=pool.pool,
                        reserve_in=pool.reserve_in,
                        reserve_out=pool.reserve_out,
                        fee=pool.fee
                    )
            
            self.last_update = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating graph: {str(e)}")

    async def _get_token_pools(self, token: str) -> List[RouteNode]:
        """Get all pools for token."""
        try:
            pools = []
            
            # Get pools from DEX optimizer
            connected = await self.dex._get_connected_tokens(token)
            
            for token_out, token_pools in connected.items():
                for pool_info in token_pools:
                    # Get pool details
                    pool_data = await self.dex._get_pool_info(
                        pool_info["address"],
                        pool_info["dex"]
                    )
                    
                    if pool_data:
                        pools.append(
                            RouteNode(
                                token=token,
                                dex=pool_info["dex"],
                                pool=pool_info["address"],
                                reserve_in=pool_data["reserve0"],
                                reserve_out=pool_data["reserve1"],
                                fee=pool_data["fee"]
                            )
                        )
            
            return pools
            
        except Exception as e:
            self.logger.error(f"Error getting token pools: {str(e)}")
            return []

    def _find_candidate_paths(
        self,
        token_in: str,
        max_hops: int
    ) -> List[List[RouteNode]]:
        """Find candidate arbitrage paths."""
        try:
            paths = []
            
            # Use NetworkX to find cycles
            for path in nx.simple_cycles(self.graph):
                # Filter paths starting with token_in
                if path[0] == token_in and len(path) <= max_hops + 1:
                    # Convert path to RouteNodes
                    route_nodes = []
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(
                            path[i],
                            path[i + 1]
                        )
                        route_nodes.append(
                            RouteNode(
                                token=path[i],
                                dex=edge_data["dex"],
                                pool=edge_data["pool"],
                                reserve_in=edge_data["reserve_in"],
                                reserve_out=edge_data["reserve_out"],
                                fee=edge_data["fee"]
                            )
                        )
                    paths.append(route_nodes)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Error finding candidate paths: {str(e)}")
            return []

    async def _analyze_route(
        self,
        path: List[RouteNode],
        amount_in: int
    ) -> Optional[Route]:
        """Analyze arbitrage route."""
        try:
            # Check cache
            cache_key = self._get_cache_key(path, amount_in)
            cached = self.route_cache.get(cache_key)
            if cached and datetime.utcnow() - cached["timestamp"] < self.cache_duration:
                return cached["route"]
            
            current_amount = amount_in
            amounts = [amount_in]
            total_fee = 1.0
            gas_cost = 0
            
            # Simulate trades
            for node in path:
                # Calculate output
                output = await self._calculate_output(
                    current_amount,
                    node
                )
                
                if output == 0:
                    return None
                
                # Update state
                current_amount = output
                amounts.append(output)
                total_fee *= (1 - node.fee)
                gas_cost += self._estimate_gas(node)
            
            # Calculate metrics
            profit = await self._calculate_profit(
                amount_in,
                current_amount,
                path[0].token
            )
            
            gas_price = await self.web3.eth.gas_price
            gas_cost_eth = gas_cost * gas_price
            gas_cost_usd = gas_cost_eth * await self._get_eth_price()
            
            net_profit = profit - gas_cost_usd
            
            if net_profit <= 0:
                return None
            
            # Create route
            route = Route(
                path=path,
                amounts=amounts,
                expected_profit=net_profit,
                gas_cost=gas_cost_usd,
                total_fee=total_fee,
                confidence=self._calculate_confidence(path, amounts)
            )
            
            # Cache result
            self.route_cache[cache_key] = {
                "route": route,
                "timestamp": datetime.utcnow()
            }
            
            return route
            
        except Exception as e:
            self.logger.error(f"Error analyzing route: {str(e)}")
            return None

    def _get_cache_key(self, path: List[RouteNode], amount_in: int) -> str:
        """Generate cache key for route."""
        return "_".join([
            f"{node.token}_{node.dex}_{node.pool}"
            for node in path
        ]) + f"_{amount_in}"

    async def _calculate_output(
        self,
        amount_in: int,
        node: RouteNode
    ) -> int:
        """Calculate output amount for swap."""
        try:
            # Calculate fee
            amount_with_fee = amount_in * (1 - node.fee)
            
            # Calculate output using constant product formula
            numerator = amount_with_fee * node.reserve_out
            denominator = node.reserve_in + amount_with_fee
            
            return numerator // denominator
            
        except Exception as e:
            self.logger.error(f"Error calculating output: {str(e)}")
            return 0

    def _estimate_gas(self, node: RouteNode) -> int:
        """Estimate gas cost for swap."""
        # Use DEX optimizer's gas estimates
        return self.dex._estimate_step_gas(node.dex)

    async def _calculate_profit(
        self,
        amount_in: int,
        amount_out: int,
        token: str
    ) -> float:
        """Calculate profit in USD."""
        try:
            # Get token prices
            price_in = await self.market_data.get_token_price(token)
            
            if not price_in:
                return 0.0
            
            # Calculate profit
            value_in = amount_in * price_in
            value_out = amount_out * price_in
            
            return value_out - value_in
            
        except Exception as e:
            self.logger.error(f"Error calculating profit: {str(e)}")
            return 0.0

    def _calculate_confidence(
        self,
        path: List[RouteNode],
        amounts: List[int]
    ) -> float:
        """Calculate confidence score for route."""
        try:
            # Base confidence
            confidence = 1.0
            
            # Adjust for path length
            confidence *= max(0.5, 1 - (len(path) * 0.1))
            
            # Adjust for liquidity impact
            for i, node in enumerate(path):
                amount = amounts[i]
                ratio = amount / node.reserve_in
                if ratio > 0.1:  # More than 10% of liquidity
                    confidence *= 0.8
                elif ratio > 0.05:  # More than 5% of liquidity
                    confidence *= 0.9
            
            # Adjust for DEX diversity
            dexes = set(node.dex for node in path)
            confidence *= (1 + len(dexes) * 0.1)  # Bonus for using multiple DEXes
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    async def prepare_execution(
        self,
        route: Route
    ) -> Optional[Dict]:
        """Prepare route for execution."""
        try:
            execution_steps = []
            
            for i, node in enumerate(route.path):
                # Get router contract
                router = self.dex.contracts[node.dex]["router"]
                
                # Prepare swap data
                swap_data = await self._prepare_swap_data(
                    router,
                    node,
                    route.amounts[i],
                    route.amounts[i+1]
                )
                
                execution_steps.append({
                    "dex": node.dex,
                    "contract": router.address,
                    "data": swap_data,
                    "value": 0
                })
            
            return {
                "steps": execution_steps,
                "expected_profit": route.expected_profit,
                "gas_estimate": sum(
                    self._estimate_gas(node)
                    for node in route.path
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing execution: {str(e)}")
            return None

    async def _prepare_swap_data(
        self,
        router,
        node: RouteNode,
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
                    [node.token, node.token_out],
                    self.web3.eth.default_account,
                    int(datetime.utcnow().timestamp() + 300)  # 5 min deadline
                ]
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing swap data: {str(e)}")
            return ""

    async def simulate_route(
        self,
        route: Route,
        block_number: Optional[int] = None
    ) -> Dict:
        """Simulate route execution at specific block."""
        try:
            if block_number:
                # Fork state at block
                self.web3.provider.make_request(
                    "evm_snapshot",
                    [block_number]
                )
            
            results = []
            current_amount = route.amounts[0]
            
            # Simulate each step
            for i, node in enumerate(route.path):
                result = await self._simulate_swap(
                    node,
                    current_amount,
                    route.amounts[i+1]
                )
                
                if not result["success"]:
                    return {
                        "success": False,
                        "error": result["error"],
                        "step": i
                    }
                
                results.append(result)
                current_amount = result["amount_out"]
            
            # Calculate actual metrics
            actual_profit = await self._calculate_profit(
                route.amounts[0],
                current_amount,
                route.path[0].token
            )
            
            return {
                "success": True,
                "actual_profit": actual_profit,
                "expected_profit": route.expected_profit,
                "profit_diff": actual_profit - route.expected_profit,
                "steps": results
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating route: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            if block_number:
                # Revert state
                self.web3.provider.make_request("evm_revert", [])

    async def _simulate_swap(
        self,
        node: RouteNode,
        amount_in: int,
        expected_out: int
    ) -> Dict:
        """Simulate single swap."""
        try:
            # Get pool contract
            pool = self.web3.eth.contract(
                address=node.pool,
                abi=self.dex._get_pool_abi()
            )
            
            # Get current reserves
            reserves = await pool.functions.getReserves().call()
            
            # Calculate actual output
            actual_out = await self._calculate_output(
                amount_in,
                node
            )
            
            return {
                "success": True,
                "amount_in": amount_in,
                "amount_out": actual_out,
                "expected_out": expected_out,
                "slippage": (expected_out - actual_out) / expected_out
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating swap: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["profitable_routes"] /
                max(1, self.metrics["routes_analyzed"])
            ),
            "avg_profit_per_route": (
                self.metrics["total_profit_found"] /
                max(1, self.metrics["profitable_routes"])
            ),
            "cache_size": len(self.route_cache),
            "last_update": datetime.utcnow().isoformat()
        }
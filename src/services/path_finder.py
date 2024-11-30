from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from dataclasses import dataclass
from web3 import Web3

from ..models.token import Token, LiquidityPool
from ..config import SystemConfig
from .token_database import TokenDatabase

@dataclass
class ArbitragePath:
    tokens: List[str]  # List of token addresses in the path
    dexes: List[str]   # List of DEXs to use for each hop
    expected_profit: float
    estimated_gas: int
    min_liquidity: float
    success_probability: float

class PathFinder:
    def __init__(
        self,
        config: SystemConfig,
        web3: Web3,
        token_db: TokenDatabase
    ):
        self.config = config
        self.web3 = web3
        self.token_db = token_db
        self.logger = logging.getLogger(__name__)
        self.max_path_length = 4  # Maximum number of hops (including return to start)

    async def find_arbitrage_paths(
        self,
        start_token: str,
        min_profit_usd: float = 100.0,
        max_paths: int = 10
    ) -> List[ArbitragePath]:
        """Find profitable arbitrage paths starting from a given token."""
        try:
            paths = []
            visited = {start_token.lower()}
            current_path = [start_token]
            current_dexes = []

            await self._find_paths_recursive(
                current_path,
                current_dexes,
                visited,
                min_profit_usd,
                paths
            )

            # Sort paths by expected profit and return top results
            paths.sort(key=lambda x: x.expected_profit, reverse=True)
            return paths[:max_paths]

        except Exception as e:
            self.logger.error(f"Error finding arbitrage paths: {str(e)}")
            return []

    async def _find_paths_recursive(
        self,
        current_path: List[str],
        current_dexes: List[str],
        visited: Set[str],
        min_profit_usd: float,
        results: List[ArbitragePath]
    ):
        """Recursively find arbitrage paths."""
        try:
            # Check if we have a valid cycle
            if len(current_path) > 1 and current_path[0] == current_path[-1]:
                # Evaluate the path
                path_evaluation = await self._evaluate_path(
                    current_path,
                    current_dexes
                )
                if path_evaluation and path_evaluation.expected_profit >= min_profit_usd:
                    results.append(path_evaluation)
                return

            # Stop if path is too long
            if len(current_path) >= self.max_path_length:
                return

            # Get current token
            current_token = await self.token_db.get_token(current_path[-1])
            if not current_token:
                return

            # Find next possible hops
            for pool in current_token.liquidity_pools:
                next_token = pool.token1_address if pool.token0_address.lower() == current_token.address.lower() else pool.token0_address
                next_token = next_token.lower()

                # Skip if we've visited this token (unless it's the start token and we have at least 2 hops)
                if next_token in visited and (next_token != current_path[0].lower() or len(current_path) < 3):
                    continue

                # Check liquidity
                if not await self._check_pool_viability(pool):
                    continue

                # Add to path
                visited.add(next_token)
                current_path.append(next_token)
                current_dexes.append(pool.dex_name)

                # Recurse
                await self._find_paths_recursive(
                    current_path,
                    current_dexes,
                    visited,
                    min_profit_usd,
                    results
                )

                # Backtrack
                visited.remove(next_token)
                current_path.pop()
                current_dexes.pop()

        except Exception as e:
            self.logger.error(f"Error in recursive path finding: {str(e)}")

    async def _check_pool_viability(self, pool: LiquidityPool) -> bool:
        """Check if a pool has sufficient liquidity and activity."""
        try:
            # Minimum liquidity threshold (e.g., $10,000)
            min_liquidity = 10000

            if not pool.total_liquidity_usd or pool.total_liquidity_usd < min_liquidity:
                return False

            # Check if pool has recent activity
            if not pool.volume_24h_usd or pool.volume_24h_usd < min_liquidity / 10:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking pool viability: {str(e)}")
            return False

    async def _evaluate_path(
        self,
        token_path: List[str],
        dex_path: List[str]
    ) -> Optional[ArbitragePath]:
        """Evaluate a potential arbitrage path."""
        try:
            # Simulate the trades
            amount_in = Web3.to_wei(1, 'ether')  # Start with 1 ETH worth
            current_amount = amount_in
            min_liquidity = float('inf')
            total_gas = 0

            # Simulate each hop
            for i in range(len(token_path) - 1):
                token_in = token_path[i]
                token_out = token_path[i + 1]
                dex = dex_path[i]

                # Get protocol instance
                protocol = self._get_protocol_instance(dex)
                if not protocol:
                    return None

                # Check liquidity and get expected output
                amount_out = await protocol.get_amount_out(
                    current_amount,
                    token_in,
                    token_out
                )
                if not amount_out:
                    return None

                # Update current amount
                current_amount = amount_out

                # Track minimum liquidity
                pool = await self._get_pool(token_in, token_out, dex)
                if pool and pool.total_liquidity_usd:
                    min_liquidity = min(min_liquidity, pool.total_liquidity_usd)

                # Add gas estimate for this hop
                total_gas += 100000  # Approximate gas per swap

            # Calculate profit
            profit_wei = current_amount - amount_in
            if profit_wei <= 0:
                return None

            # Convert profit to USD
            eth_price = await self._get_eth_price()
            profit_usd = Web3.from_wei(profit_wei, 'ether') * eth_price

            # Calculate success probability based on liquidity and complexity
            success_prob = self._calculate_success_probability(
                min_liquidity,
                len(token_path),
                profit_usd
            )

            return ArbitragePath(
                tokens=token_path,
                dexes=dex_path,
                expected_profit=profit_usd,
                estimated_gas=total_gas,
                min_liquidity=min_liquidity,
                success_probability=success_prob
            )

        except Exception as e:
            self.logger.error(f"Error evaluating path: {str(e)}")
            return None

    def _get_protocol_instance(self, dex_name: str):
        """Get protocol instance by name."""
        # TODO: Implement protocol instance caching and retrieval
        return None

    async def _get_pool(
        self,
        token_a: str,
        token_b: str,
        dex: str
    ) -> Optional[LiquidityPool]:
        """Get pool information for a token pair on a specific DEX."""
        try:
            token = await self.token_db.get_token(token_a)
            if not token:
                return None

            for pool in token.liquidity_pools:
                if pool.dex_name == dex and (
                    pool.token1_address.lower() == token_b.lower() or
                    pool.token0_address.lower() == token_b.lower()
                ):
                    return pool

            return None

        except Exception as e:
            self.logger.error(f"Error getting pool: {str(e)}")
            return None

    async def _get_eth_price(self) -> float:
        """Get current ETH price in USD."""
        try:
            # Use a price feed (e.g., Chainlink)
            # TODO: Implement price feed
            return 2000.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Error getting ETH price: {str(e)}")
            return 0.0

    def _calculate_success_probability(
        self,
        min_liquidity: float,
        path_length: int,
        profit_usd: float
    ) -> float:
        """Calculate the probability of successful execution."""
        try:
            # Base probability starts at 1.0
            prob = 1.0

            # Reduce probability based on path length
            prob *= 0.9 ** (path_length - 2)  # -2 because we need at least 2 hops

            # Reduce probability based on liquidity
            if min_liquidity < 100000:  # $100k
                prob *= 0.7
            elif min_liquidity < 1000000:  # $1M
                prob *= 0.85

            # Increase probability for very profitable trades
            if profit_usd > 1000:  # $1000
                prob = min(prob * 1.2, 1.0)

            return prob

        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return 0.0
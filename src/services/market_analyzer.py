from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
import aiohttp

from ..models.token import Token
from ..services.token_database import TokenDatabase

class MarketAnalyzer:
    def __init__(self, web3: Web3, token_db: TokenDatabase):
        self.web3 = web3
        self.token_db = token_db
        self.logger = logging.getLogger(__name__)
        
        # Cache for market data
        self.price_cache: Dict[str, Dict] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.liquidity_cache: Dict[str, float] = {}
        
        # Analysis parameters
        self.volatility_window = 24  # hours
        self.price_update_interval = 60  # seconds
        self.min_liquidity_usd = 100000  # $100k
        
        # Initialize monitoring
        self.monitoring_task = None

    async def start_monitoring(self):
        """Start market monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_market())

    async def stop_monitoring(self):
        """Stop market monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

    async def _monitor_market(self):
        """Continuously monitor market conditions."""
        while True:
            try:
                # Get all active tokens
                tokens = await self.token_db.get_active_tokens()
                
                # Update market data for each token
                for token in tokens:
                    await self._update_token_data(token)
                
                # Analyze market conditions
                await self._analyze_market_conditions()
                
                # Sleep before next update
                await asyncio.sleep(self.price_update_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring market: {str(e)}")
                await asyncio.sleep(self.price_update_interval)

    async def _update_token_data(self, token: Token):
        """Update market data for a token."""
        try:
            # Update price data
            price_data = await self._fetch_token_price_data(token.address)
            if price_data:
                self.price_cache[token.address] = price_data
            
            # Update volatility
            volatility = await self._calculate_volatility(token.address)
            if volatility is not None:
                self.volatility_cache[token.address] = volatility
            
            # Update liquidity
            liquidity = await self._calculate_total_liquidity(token)
            if liquidity is not None:
                self.liquidity_cache[token.address] = liquidity

        except Exception as e:
            self.logger.error(f"Error updating token data: {str(e)}")

    async def _fetch_token_price_data(self, token_address: str) -> Optional[Dict]:
        """Fetch token price data from multiple sources."""
        try:
            # Try multiple price sources
            sources = [
                self._fetch_coingecko_price,
                self._fetch_1inch_price,
                self._fetch_chainlink_price
            ]
            
            for source in sources:
                price_data = await source(token_address)
                if price_data:
                    return price_data
            
            return None

        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return None

    async def _fetch_coingecko_price(self, token_address: str) -> Optional[Dict]:
        """Fetch price data from CoinGecko."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.coingecko.com/api/v3/simple/token_price/ethereum",
                    params={
                        "contract_addresses": token_address,
                        "vs_currencies": "usd",
                        "include_24hr_vol": "true",
                        "include_24hr_change": "true"
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get(token_address.lower(), {})
                        if token_data:
                            return {
                                "price": token_data.get("usd", 0),
                                "volume_24h": token_data.get("usd_24h_vol", 0),
                                "price_change_24h": token_data.get("usd_24h_change", 0),
                                "timestamp": datetime.utcnow().isoformat()
                            }
            return None

        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko price: {str(e)}")
            return None

    async def _fetch_1inch_price(self, token_address: str) -> Optional[Dict]:
        """Fetch price data from 1inch API."""
        # TODO: Implement 1inch price fetching
        return None

    async def _fetch_chainlink_price(self, token_address: str) -> Optional[Dict]:
        """Fetch price data from Chainlink oracle."""
        # TODO: Implement Chainlink price fetching
        return None

    async def _calculate_volatility(self, token_address: str) -> Optional[float]:
        """Calculate token price volatility."""
        try:
            price_data = self.price_cache.get(token_address, {})
            if not price_data:
                return None

            # Calculate historical volatility
            price_history = price_data.get("price_history", [])
            if len(price_history) < 2:
                return None

            # Calculate returns
            returns = np.diff(np.log(price_history))
            
            # Calculate annualized volatility
            volatility = np.std(returns) * np.sqrt(365 * 24)
            
            return float(volatility)

        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return None

    async def _calculate_total_liquidity(self, token: Token) -> Optional[float]:
        """Calculate total liquidity across all pools."""
        try:
            total_liquidity = 0.0
            
            for pool in token.liquidity_pools:
                if pool.total_liquidity_usd:
                    total_liquidity += pool.total_liquidity_usd
            
            return total_liquidity if total_liquidity > 0 else None

        except Exception as e:
            self.logger.error(f"Error calculating liquidity: {str(e)}")
            return None

    async def _analyze_market_conditions(self):
        """Analyze overall market conditions."""
        try:
            # Calculate market metrics
            total_liquidity = sum(self.liquidity_cache.values())
            avg_volatility = np.mean(list(self.volatility_cache.values()))
            
            # Update market state
            self.market_state = {
                "total_liquidity": total_liquidity,
                "average_volatility": avg_volatility,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")

    async def get_opportunity_score(
        self,
        token_address: str,
        expected_profit: float
    ) -> float:
        """Calculate opportunity score based on market conditions."""
        try:
            # Get token data
            volatility = self.volatility_cache.get(token_address, 1.0)
            liquidity = self.liquidity_cache.get(token_address, 0.0)
            
            # Calculate base score
            base_score = expected_profit / self.min_liquidity_usd
            
            # Adjust for volatility (lower is better)
            volatility_factor = 1 / (1 + volatility)
            
            # Adjust for liquidity (higher is better)
            liquidity_factor = min(liquidity / self.min_liquidity_usd, 2.0)
            
            # Calculate final score
            score = base_score * volatility_factor * liquidity_factor
            
            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0

    async def get_market_depth(
        self,
        token_address: str,
        amount_usd: float
    ) -> Tuple[float, float]:
        """
        Calculate market depth and price impact.
        Returns (depth_score, price_impact)
        """
        try:
            token = await self.token_db.get_token(token_address)
            if not token:
                return 0.0, 1.0

            total_liquidity = 0.0
            weighted_price_impact = 0.0

            for pool in token.liquidity_pools:
                if pool.total_liquidity_usd:
                    # Calculate pool's contribution to depth
                    pool_depth = pool.total_liquidity_usd
                    total_liquidity += pool_depth

                    # Estimate price impact for this pool
                    price_impact = amount_usd / pool_depth
                    weighted_price_impact += price_impact * (pool_depth / total_liquidity)

            if total_liquidity == 0:
                return 0.0, 1.0

            # Calculate depth score (0 to 1)
            depth_score = min(total_liquidity / (amount_usd * 10), 1.0)
            
            # Normalize price impact (0 to 1)
            normalized_impact = min(weighted_price_impact, 1.0)

            return depth_score, normalized_impact

        except Exception as e:
            self.logger.error(f"Error calculating market depth: {str(e)}")
            return 0.0, 1.0

    def get_market_stats(self) -> Dict:
        """Get current market statistics."""
        try:
            return {
                "total_liquidity": sum(self.liquidity_cache.values()),
                "average_volatility": np.mean(list(self.volatility_cache.values())),
                "active_tokens": len(self.price_cache),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting market stats: {str(e)}")
            return {}
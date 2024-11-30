from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import aiohttp
from web3 import Web3
import json

class MarketDataAggregator:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        
        # Cache settings
        self.price_cache: Dict[str, Dict] = {}
        self.liquidity_cache: Dict[str, Dict] = {}
        self.volume_cache: Dict[str, Dict] = {}
        
        # Cache duration
        self.cache_durations = {
            "price": 10,  # 10 seconds
            "liquidity": 60,  # 1 minute
            "volume": 300  # 5 minutes
        }
        
        # Data sources
        self.sources = {
            "dex": {
                "uniswap_v2": True,
                "sushiswap": True,
                "curve": True
            },
            "aggregator": {
                "1inch": True,
                "0x": True
            },
            "oracle": {
                "chainlink": True,
                "band": False
            }
        }
        
        # Initialize price feeds
        self._init_price_feeds()

    def _init_price_feeds(self):
        """Initialize price feed contracts."""
        try:
            # Load Chainlink price feed addresses
            with open('config/price_feeds.json', 'r') as f:
                self.price_feeds = json.load(f)
            
            # Initialize contracts
            self.price_feed_contracts = {}
            for token, address in self.price_feeds.items():
                self.price_feed_contracts[token] = self.web3.eth.contract(
                    address=address,
                    abi=self._get_chainlink_abi()
                )
                
        except Exception as e:
            self.logger.error(f"Error initializing price feeds: {str(e)}")

    def _get_chainlink_abi(self) -> List:
        """Get Chainlink price feed ABI."""
        return [
            {
                "inputs": [],
                "name": "latestRoundData",
                "outputs": [
                    {"type": "uint80", "name": "roundId"},
                    {"type": "int256", "name": "answer"},
                    {"type": "uint256", "name": "startedAt"},
                    {"type": "uint256", "name": "updatedAt"},
                    {"type": "uint80", "name": "answeredInRound"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    async def get_token_price(
        self,
        token_address: str,
        force_refresh: bool = False
    ) -> Optional[float]:
        """Get token price from multiple sources."""
        try:
            # Check cache
            if not force_refresh and token_address in self.price_cache:
                cache_data = self.price_cache[token_address]
                if datetime.utcnow() - cache_data["timestamp"] < timedelta(
                    seconds=self.cache_durations["price"]
                ):
                    return cache_data["price"]
            
            # Collect prices from all sources
            prices = await asyncio.gather(
                self._get_dex_price(token_address),
                self._get_aggregator_price(token_address),
                self._get_oracle_price(token_address)
            )
            
            # Filter out None values
            valid_prices = [p for p in prices if p is not None]
            
            if not valid_prices:
                return None
            
            # Calculate median price
            median_price = float(np.median(valid_prices))
            
            # Update cache
            self.price_cache[token_address] = {
                "price": median_price,
                "timestamp": datetime.utcnow()
            }
            
            return median_price
            
        except Exception as e:
            self.logger.error(f"Error getting token price: {str(e)}")
            return None

    async def _get_dex_price(self, token_address: str) -> Optional[float]:
        """Get price from DEX liquidity pools."""
        try:
            prices = []
            
            # Get Uniswap V2 price
            if self.sources["dex"]["uniswap_v2"]:
                uni_price = await self._get_uniswap_price(token_address)
                if uni_price:
                    prices.append(uni_price)
            
            # Get SushiSwap price
            if self.sources["dex"]["sushiswap"]:
                sushi_price = await self._get_sushiswap_price(token_address)
                if sushi_price:
                    prices.append(sushi_price)
            
            if not prices:
                return None
            
            # Return volume-weighted average price
            return float(np.average(prices))
            
        except Exception as e:
            self.logger.error(f"Error getting DEX price: {str(e)}")
            return None

    async def _get_aggregator_price(self, token_address: str) -> Optional[float]:
        """Get price from aggregator APIs."""
        try:
            prices = []
            
            # Get 1inch price
            if self.sources["aggregator"]["1inch"]:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.1inch.io/v4.0/1/quote",
                        params={
                            "fromTokenAddress": token_address,
                            "toTokenAddress": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                            "amount": "1000000000000000000"  # 1 token
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            prices.append(float(data["toTokenAmount"]) / 1e18)
            
            # Get 0x price
            if self.sources["aggregator"]["0x"]:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://api.0x.org/swap/v1/quote",
                        params={
                            "sellToken": token_address,
                            "buyToken": "WETH",
                            "sellAmount": "1000000000000000000"  # 1 token
                        }
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            prices.append(float(data["price"]))
            
            if not prices:
                return None
            
            return float(np.median(prices))
            
        except Exception as e:
            self.logger.error(f"Error getting aggregator price: {str(e)}")
            return None

    async def _get_oracle_price(self, token_address: str) -> Optional[float]:
        """Get price from oracle feeds."""
        try:
            # Try Chainlink first
            if self.sources["oracle"]["chainlink"]:
                if token_address in self.price_feed_contracts:
                    contract = self.price_feed_contracts[token_address]
                    data = await contract.functions.latestRoundData().call()
                    return float(data[1]) / 1e8  # Chainlink uses 8 decimals
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting oracle price: {str(e)}")
            return None

    async def get_pool_liquidity(
        self,
        pool_address: str,
        force_refresh: bool = False
    ) -> Optional[float]:
        """Get pool liquidity in USD."""
        try:
            # Check cache
            if not force_refresh and pool_address in self.liquidity_cache:
                cache_data = self.liquidity_cache[pool_address]
                if datetime.utcnow() - cache_data["timestamp"] < timedelta(
                    seconds=self.cache_durations["liquidity"]
                ):
                    return cache_data["liquidity"]
            
            # Get pool data
            pool_data = await self._get_pool_data(pool_address)
            if not pool_data:
                return None
            
            # Get token prices
            token0_price = await self.get_token_price(pool_data["token0"])
            token1_price = await self.get_token_price(pool_data["token1"])
            
            if not token0_price or not token1_price:
                return None
            
            # Calculate total liquidity
            liquidity = (
                float(pool_data["reserve0"]) * token0_price +
                float(pool_data["reserve1"]) * token1_price
            )
            
            # Update cache
            self.liquidity_cache[pool_address] = {
                "liquidity": liquidity,
                "timestamp": datetime.utcnow()
            }
            
            return liquidity
            
        except Exception as e:
            self.logger.error(f"Error getting pool liquidity: {str(e)}")
            return None

    async def _get_pool_data(self, pool_address: str) -> Optional[Dict]:
        """Get pool token addresses and reserves."""
        try:
            # Load pool ABI
            pool_abi = [
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"type": "address"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [],
                    "name": "getReserves",
                    "outputs": [
                        {"type": "uint112", "name": "reserve0"},
                        {"type": "uint112", "name": "reserve1"},
                        {"type": "uint32", "name": "blockTimestampLast"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
            
            # Create contract instance
            pool = self.web3.eth.contract(
                address=pool_address,
                abi=pool_abi
            )
            
            # Get pool data
            token0 = await pool.functions.token0().call()
            token1 = await pool.functions.token1().call()
            reserves = await pool.functions.getReserves().call()
            
            return {
                "token0": token0,
                "token1": token1,
                "reserve0": reserves[0],
                "reserve1": reserves[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pool data: {str(e)}")
            return None

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
            # Get all pools containing the token
            pools = await self._get_token_pools(token_address)
            
            total_liquidity = 0
            weighted_price_impact = 0
            
            for pool in pools:
                # Get pool liquidity
                liquidity = await self.get_pool_liquidity(pool["address"])
                if not liquidity:
                    continue
                
                total_liquidity += liquidity
                
                # Calculate price impact for this pool
                price_impact = amount_usd / liquidity
                weighted_price_impact += price_impact * (liquidity / total_liquidity)
            
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

    async def _get_token_pools(self, token_address: str) -> List[Dict]:
        """Get all liquidity pools containing the token."""
        # TODO: Implement pool discovery logic
        return []

    def get_market_stats(self) -> Dict:
        """Get current market statistics."""
        return {
            "price_cache_size": len(self.price_cache),
            "liquidity_cache_size": len(self.liquidity_cache),
            "volume_cache_size": len(self.volume_cache),
            "sources": self.sources
        }
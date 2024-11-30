from typing import Dict, Optional, Tuple
import asyncio
import logging
from web3 import Web3
import aiohttp
import statistics

class GasOptimizer:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        
        # Gas price history for analysis
        self.price_history: list = []
        self.max_history_size = 1000
        
        # Gas price update interval
        self.update_interval = 15  # seconds
        
        # Initialize gas price monitoring
        self.monitoring_task = None

    async def start_monitoring(self):
        """Start gas price monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_gas_prices())

    async def stop_monitoring(self):
        """Stop gas price monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None

    async def _monitor_gas_prices(self):
        """Monitor gas prices continuously."""
        while True:
            try:
                # Get current gas prices from multiple sources
                gas_prices = await asyncio.gather(
                    self._get_network_gas_price(),
                    self._get_etherscan_gas_price(),
                    self._get_blocknative_gas_price()
                )
                
                # Filter out None values and calculate median
                valid_prices = [p for p in gas_prices if p is not None]
                if valid_prices:
                    median_price = statistics.median(valid_prices)
                    self._update_price_history(median_price)

                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring gas prices: {str(e)}")
                await asyncio.sleep(self.update_interval)

    async def _get_network_gas_price(self) -> Optional[int]:
        """Get gas price from network."""
        try:
            return await self.web3.eth.gas_price
        except Exception as e:
            self.logger.error(f"Error getting network gas price: {str(e)}")
            return None

    async def _get_etherscan_gas_price(self) -> Optional[int]:
        """Get gas price from Etherscan API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.etherscan.io/api",
                    params={
                        "module": "gastracker",
                        "action": "gasoracle",
                        "apikey": "YOUR_ETHERSCAN_API_KEY"  # TODO: Move to config
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return Web3.to_wei(
                            int(data["result"]["SafeGasPrice"]),
                            "gwei"
                        )
            return None
        except Exception as e:
            self.logger.error(f"Error getting Etherscan gas price: {str(e)}")
            return None

    async def _get_blocknative_gas_price(self) -> Optional[int]:
        """Get gas price from Blocknative API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.blocknative.com/gasprices/blockprices",
                    headers={
                        "Authorization": "YOUR_BLOCKNATIVE_API_KEY"  # TODO: Move to config
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return Web3.to_wei(
                            int(data["blockPrices"][0]["estimatedPrices"][0]["price"]),
                            "gwei"
                        )
            return None
        except Exception as e:
            self.logger.error(f"Error getting Blocknative gas price: {str(e)}")
            return None

    def _update_price_history(self, price: int):
        """Update gas price history."""
        self.price_history.append(price)
        if len(self.price_history) > self.max_history_size:
            self.price_history.pop(0)

    async def get_optimal_gas_price(
        self,
        priority: str = "normal"
    ) -> Tuple[int, int]:
        """
        Get optimal gas price and max fee based on priority.
        Returns (base_fee, priority_fee)
        """
        try:
            # Get latest block base fee
            latest_block = await self.web3.eth.get_block('latest')
            base_fee = latest_block['baseFeePerGas']

            # Calculate priority fee based on recent history
            if self.price_history:
                recent_prices = self.price_history[-50:]  # Last 50 prices
                median_priority_fee = statistics.median(recent_prices) - base_fee
            else:
                median_priority_fee = await self.web3.eth.max_priority_fee

            # Adjust based on priority
            priority_multipliers = {
                "low": 0.8,
                "normal": 1.0,
                "high": 1.5,
                "urgent": 2.0
            }
            
            multiplier = priority_multipliers.get(priority, 1.0)
            priority_fee = int(median_priority_fee * multiplier)

            return base_fee, priority_fee

        except Exception as e:
            self.logger.error(f"Error getting optimal gas price: {str(e)}")
            # Fallback to network suggested values
            return (
                await self.web3.eth.gas_price,
                await self.web3.eth.max_priority_fee
            )

    async def estimate_gas_cost(
        self,
        tx_params: Dict,
        priority: str = "normal"
    ) -> Tuple[int, int]:
        """
        Estimate total gas cost for a transaction.
        Returns (estimated_gas, total_cost_wei)
        """
        try:
            # Estimate gas limit
            gas_limit = await self.web3.eth.estimate_gas(tx_params)
            
            # Add safety margin
            safety_margins = {
                "low": 1.1,      # 10% margin
                "normal": 1.2,   # 20% margin
                "high": 1.3,     # 30% margin
                "urgent": 1.5    # 50% margin
            }
            
            margin = safety_margins.get(priority, 1.2)
            gas_limit = int(gas_limit * margin)

            # Get optimal gas price
            base_fee, priority_fee = await self.get_optimal_gas_price(priority)
            
            # Calculate total cost
            total_cost = gas_limit * (base_fee + priority_fee)

            return gas_limit, total_cost

        except Exception as e:
            self.logger.error(f"Error estimating gas cost: {str(e)}")
            return 0, 0

    async def is_gas_price_favorable(
        self,
        expected_profit: int,
        tx_params: Dict,
        min_profit_ratio: float = 0.1  # Minimum 10% profit after gas
    ) -> bool:
        """Check if current gas prices make the transaction profitable."""
        try:
            # Estimate gas cost
            _, total_gas_cost = await self.estimate_gas_cost(tx_params, "normal")
            
            # Calculate profit ratio after gas
            profit_after_gas = expected_profit - total_gas_cost
            profit_ratio = profit_after_gas / expected_profit

            return profit_ratio >= min_profit_ratio

        except Exception as e:
            self.logger.error(f"Error checking gas price favorability: {str(e)}")
            return False

    def get_gas_price_stats(self) -> Dict:
        """Get statistical analysis of recent gas prices."""
        try:
            if not self.price_history:
                return {}

            recent_prices = self.price_history[-50:]  # Last 50 prices
            return {
                "current": recent_prices[-1],
                "mean": statistics.mean(recent_prices),
                "median": statistics.median(recent_prices),
                "min": min(recent_prices),
                "max": max(recent_prices),
                "std_dev": statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting gas price stats: {str(e)}")
            return {}
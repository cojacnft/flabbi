from typing import Dict, List, Optional
import asyncio
import logging
import aiohttp
from web3 import Web3, AsyncWeb3
from web3.providers import AsyncHTTPProvider
import json
import time

class NetworkManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # RPC configuration
        self.rpc_configs = {
            "alchemy": {
                "mainnet": {
                    "http": "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
                    "ws": "wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
                }
            },
            "gelato": {
                "mainnet": {
                    "http": "https://relay.gelato.digital"
                }
            }
        }
        
        # Connection pools
        self.web3_instances: Dict[str, AsyncWeb3] = {}
        self.ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        
        # Rate limiting
        self.request_timestamps: Dict[str, List[float]] = {
            "alchemy": [],
            "gelato": []
        }
        self.rate_limits = {
            "alchemy": {
                "requests_per_second": 25,
                "requests_per_day": 300000
            },
            "gelato": {
                "requests_per_second": 50,
                "requests_per_day": 1000000
            }
        }

    async def initialize(self):
        """Initialize network connections."""
        try:
            # Initialize Alchemy connection
            alchemy_provider = AsyncHTTPProvider(
                self.rpc_configs["alchemy"]["mainnet"]["http"]
            )
            self.web3_instances["alchemy"] = AsyncWeb3(alchemy_provider)
            
            # Initialize WebSocket connection
            await self._setup_websocket()
            
            # Initialize Gelato connection
            gelato_provider = AsyncHTTPProvider(
                self.rpc_configs["gelato"]["mainnet"]["http"]
            )
            self.web3_instances["gelato"] = AsyncWeb3(gelato_provider)
            
            self.logger.info("Network connections initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing network: {str(e)}")
            raise

    async def _setup_websocket(self):
        """Setup WebSocket connection for real-time updates."""
        try:
            session = aiohttp.ClientSession()
            ws = await session.ws_connect(
                self.rpc_configs["alchemy"]["mainnet"]["ws"]
            )
            
            # Subscribe to pending transactions
            await ws.send_json({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_subscribe",
                "params": ["newPendingTransactions"]
            })
            
            self.ws_connections["alchemy"] = ws
            
            # Start WebSocket listener
            asyncio.create_task(self._listen_to_websocket())
            
        except Exception as e:
            self.logger.error(f"Error setting up WebSocket: {str(e)}")

    async def _listen_to_websocket(self):
        """Listen to WebSocket messages."""
        ws = self.ws_connections["alchemy"]
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if "params" in data:
                        await self._handle_pending_tx(data["params"]["result"])
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    break
        except Exception as e:
            self.logger.error(f"Error in WebSocket listener: {str(e)}")
        finally:
            await self._reconnect_websocket()

    async def _reconnect_websocket(self):
        """Reconnect WebSocket on failure."""
        try:
            if "alchemy" in self.ws_connections:
                await self.ws_connections["alchemy"].close()
            await self._setup_websocket()
        except Exception as e:
            self.logger.error(f"Error reconnecting WebSocket: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying
            await self._reconnect_websocket()

    async def _handle_pending_tx(self, tx_hash: str):
        """Handle pending transaction notification."""
        try:
            # Get transaction details through Gelato to avoid rate limits
            tx = await self.web3_instances["gelato"].eth.get_transaction(tx_hash)
            
            # Check if it's a DEX transaction
            if self._is_dex_transaction(tx):
                # Process through Gelato for better execution
                await self._process_dex_transaction(tx)
                
        except Exception as e:
            self.logger.error(f"Error handling pending tx: {str(e)}")

    def _is_dex_transaction(self, tx: Dict) -> bool:
        """Check if transaction is DEX-related."""
        # Add DEX contract addresses
        dex_addresses = {
            "uniswap_v2": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            "sushiswap": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
        }
        return tx["to"].lower() in [addr.lower() for addr in dex_addresses.values()]

    async def _check_rate_limit(self, provider: str) -> bool:
        """Check if we're within rate limits."""
        try:
            now = time.time()
            
            # Clean old timestamps
            self.request_timestamps[provider] = [
                ts for ts in self.request_timestamps[provider]
                if now - ts < 86400  # 24 hours
            ]
            
            # Check daily limit
            if len(self.request_timestamps[provider]) >= self.rate_limits[provider]["requests_per_day"]:
                return False
            
            # Check per-second limit
            recent_requests = len([
                ts for ts in self.request_timestamps[provider]
                if now - ts < 1
            ])
            
            return recent_requests < self.rate_limits[provider]["requests_per_second"]
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return False

    async def make_request(
        self,
        method: str,
        params: List,
        provider: str = "alchemy"
    ) -> Optional[Dict]:
        """Make RPC request with rate limiting."""
        try:
            if not await self._check_rate_limit(provider):
                # Switch to Gelato if Alchemy is rate limited
                if provider == "alchemy":
                    return await self.make_request(method, params, "gelato")
                return None
            
            # Record request
            self.request_timestamps[provider].append(time.time())
            
            # Make request
            web3 = self.web3_instances[provider]
            return await web3.eth.make_request(method, params)
            
        except Exception as e:
            self.logger.error(f"Error making request: {str(e)}")
            return None

    async def get_best_gas_price(self) -> Optional[int]:
        """Get optimal gas price from multiple sources."""
        try:
            # Try Gelato first for better gas estimation
            gelato_price = await self._get_gelato_gas_price()
            if gelato_price:
                return gelato_price
            
            # Fallback to Alchemy
            return await self.web3_instances["alchemy"].eth.gas_price
            
        except Exception as e:
            self.logger.error(f"Error getting gas price: {str(e)}")
            return None

    async def _get_gelato_gas_price(self) -> Optional[int]:
        """Get gas price from Gelato."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.gelato.digital/gas-prices"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return int(data["fast"])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Gelato gas price: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup network connections."""
        try:
            for ws in self.ws_connections.values():
                await ws.close()
            
            # Close any other connections
            for web3 in self.web3_instances.values():
                await web3.provider.close()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def get_provider_status(self) -> Dict:
        """Get current status of providers."""
        return {
            provider: {
                "requests_today": len(timestamps),
                "requests_last_minute": len([
                    ts for ts in timestamps
                    if time.time() - ts < 60
                ])
            }
            for provider, timestamps in self.request_timestamps.items()
        }
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from web3 import Web3, AsyncWeb3
from web3.providers import AsyncHTTPProvider, WebsocketProvider
import aiohttp
import json
from datetime import datetime, timedelta

from ..config.chains import ChainConfig, get_chain_config
from ..models.token import Token

class ChainProvider:
    def __init__(self, chain_id: int):
        self.chain_id = chain_id
        self.config = get_chain_config(chain_id)
        if not self.config:
            raise ValueError(f"Unsupported chain ID: {chain_id}")
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers: Dict[str, AsyncWeb3] = {}
        self.ws_providers: Dict[str, WebsocketProvider] = {}
        self.current_provider_index = 0
        
        # Connection state
        self.last_block = 0
        self.last_provider_switch = datetime.utcnow()
        self.provider_health: Dict[str, bool] = {}
        
        # Initialize connections
        self._init_providers()

    def _init_providers(self):
        """Initialize Web3 providers."""
        try:
            # Initialize HTTP providers
            for i, url in enumerate(self.config.rpc_urls):
                provider_name = f"provider_{i}"
                self.providers[provider_name] = AsyncWeb3(
                    AsyncHTTPProvider(url)
                )
                self.provider_health[provider_name] = True
            
            # Initialize WebSocket providers if available
            if self.config.ws_urls:
                for i, url in enumerate(self.config.ws_urls):
                    ws_name = f"ws_provider_{i}"
                    self.ws_providers[ws_name] = WebsocketProvider(url)
                    self.provider_health[ws_name] = True
            
        except Exception as e:
            self.logger.error(f"Error initializing providers: {str(e)}")
            raise

    async def get_web3(self) -> AsyncWeb3:
        """Get current Web3 provider with automatic failover."""
        try:
            provider_name = list(self.providers.keys())[self.current_provider_index]
            web3 = self.providers[provider_name]
            
            # Check if provider is healthy
            if not await self._check_provider_health(web3, provider_name):
                await self._switch_provider()
                return await self.get_web3()
            
            return web3
            
        except Exception as e:
            self.logger.error(f"Error getting Web3 provider: {str(e)}")
            await self._switch_provider()
            return await self.get_web3()

    async def _check_provider_health(
        self,
        web3: AsyncWeb3,
        provider_name: str
    ) -> bool:
        """Check if a provider is healthy."""
        try:
            # Check connection
            if not await web3.is_connected():
                self.provider_health[provider_name] = False
                return False
            
            # Check block progress
            current_block = await web3.eth.block_number
            if current_block <= self.last_block:
                self.provider_health[provider_name] = False
                return False
            
            self.last_block = current_block
            self.provider_health[provider_name] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Provider health check failed: {str(e)}")
            self.provider_health[provider_name] = False
            return False

    async def _switch_provider(self):
        """Switch to next available provider."""
        try:
            # Only switch if enough time has passed
            if datetime.utcnow() - self.last_provider_switch < timedelta(seconds=30):
                return
            
            # Find next healthy provider
            original_index = self.current_provider_index
            while True:
                self.current_provider_index = (self.current_provider_index + 1) % len(self.providers)
                provider_name = list(self.providers.keys())[self.current_provider_index]
                
                if self.provider_health[provider_name]:
                    break
                
                if self.current_provider_index == original_index:
                    raise Exception("No healthy providers available")
            
            self.last_provider_switch = datetime.utcnow()
            self.logger.info(f"Switched to provider: {provider_name}")
            
        except Exception as e:
            self.logger.error(f"Error switching provider: {str(e)}")
            raise

    async def get_gas_price(self) -> Tuple[int, int]:
        """Get current gas price and priority fee."""
        try:
            web3 = await self.get_web3()
            
            # Get base fee
            block = await web3.eth.get_block('latest')
            base_fee = block.get('baseFeePerGas', 0)
            
            # Get priority fee
            priority_fee = await web3.eth.max_priority_fee
            
            # Apply chain-specific adjustments
            max_gas = Web3.to_wei(self.config.max_gas_price, 'gwei')
            max_priority = Web3.to_wei(self.config.priority_fee, 'gwei')
            
            return (
                min(base_fee, max_gas),
                min(priority_fee, max_priority)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting gas price: {str(e)}")
            return (0, 0)

    async def get_token_balance(
        self,
        token_address: str,
        wallet_address: str
    ) -> int:
        """Get token balance for address."""
        try:
            web3 = await self.get_web3()
            
            # Load ERC20 ABI
            with open('contracts/abi/ERC20.json', 'r') as f:
                abi = json.load(f)
            
            # Create contract instance
            token = web3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=abi
            )
            
            # Get balance
            return await token.functions.balanceOf(
                Web3.to_checksum_address(wallet_address)
            ).call()
            
        except Exception as e:
            self.logger.error(f"Error getting token balance: {str(e)}")
            return 0

    async def get_native_balance(self, address: str) -> int:
        """Get native token balance."""
        try:
            web3 = await self.get_web3()
            return await web3.eth.get_balance(
                Web3.to_checksum_address(address)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting native balance: {str(e)}")
            return 0

    async def estimate_gas(
        self,
        to: str,
        data: str,
        value: int = 0
    ) -> int:
        """Estimate gas for transaction."""
        try:
            web3 = await self.get_web3()
            
            # Estimate gas
            gas = await web3.eth.estimate_gas({
                'to': Web3.to_checksum_address(to),
                'data': data,
                'value': value
            })
            
            # Apply safety multiplier
            return int(gas * self.config.gas_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error estimating gas: {str(e)}")
            return 0

    async def send_transaction(
        self,
        transaction: Dict,
        private_key: str
    ) -> Optional[str]:
        """Send transaction with automatic retries."""
        try:
            web3 = await self.get_web3()
            
            # Sign transaction
            signed = web3.eth.account.sign_transaction(
                transaction,
                private_key
            )
            
            # Send with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    tx_hash = await web3.eth.send_raw_transaction(
                        signed.rawTransaction
                    )
                    return tx_hash.hex()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending transaction: {str(e)}")
            return None

    async def wait_for_transaction(
        self,
        tx_hash: str,
        timeout: int = 180
    ) -> Optional[Dict]:
        """Wait for transaction confirmation."""
        try:
            web3 = await self.get_web3()
            
            # Wait for receipt
            start_time = datetime.utcnow()
            while datetime.utcnow() - start_time < timedelta(seconds=timeout):
                try:
                    receipt = await web3.eth.get_transaction_receipt(tx_hash)
                    if receipt:
                        # Wait for required confirmations
                        current_block = await web3.eth.block_number
                        if current_block - receipt['blockNumber'] >= self.config.confirmations_required:
                            return receipt
                except Exception:
                    pass
                
                await asyncio.sleep(self.config.block_time)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error waiting for transaction: {str(e)}")
            return None

    async def get_contract_events(
        self,
        contract_address: str,
        abi: List,
        event_name: str,
        from_block: int,
        to_block: int
    ) -> List[Dict]:
        """Get contract events."""
        try:
            web3 = await self.get_web3()
            
            # Create contract instance
            contract = web3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=abi
            )
            
            # Get events
            events = await contract.events[event_name].get_logs(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting contract events: {str(e)}")
            return []

    def get_explorer_url(self, tx_hash: str) -> str:
        """Get explorer URL for transaction."""
        if self.config.explorer_url:
            return f"{self.config.explorer_url}/tx/{tx_hash}"
        return ""

    async def get_chain_state(self) -> Dict:
        """Get current chain state."""
        try:
            web3 = await self.get_web3()
            
            return {
                "chain_id": self.chain_id,
                "block_number": await web3.eth.block_number,
                "gas_price": await web3.eth.gas_price,
                "network_id": await web3.net.version,
                "is_syncing": await web3.eth.syncing,
                "provider_health": self.provider_health
            }
            
        except Exception as e:
            self.logger.error(f"Error getting chain state: {str(e)}")
            return {}
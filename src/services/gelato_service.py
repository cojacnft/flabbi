from typing import Dict, List, Optional, Union
import asyncio
import logging
from web3 import Web3
from eth_typing import Address
import aiohttp
import json

class GelatoService:
    def __init__(
        self,
        web3: Web3,
        private_key: str,
        api_key: Optional[str] = None
    ):
        self.web3 = web3
        self.private_key = private_key
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Gelato contract addresses
        self.addresses = {
            "resolver": "0x0630d1b8C2df3F0a68Df578D02075027a6397173",
            "ops": "0x527a819db1eb0e34426297b03bae11F2f8B3A19E"
        }
        
        # Initialize connection
        self._init_gelato()

    def _init_gelato(self):
        """Initialize Gelato connection and contracts."""
        try:
            # Load ABIs
            with open('contracts/abi/GelatoOps.json', 'r') as f:
                self.ops_abi = json.load(f)
            
            # Initialize contracts
            self.ops_contract = self.web3.eth.contract(
                address=self.addresses["ops"],
                abi=self.ops_abi
            )
            
            self.logger.info("Gelato service initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Gelato: {str(e)}")

    async def submit_private_transaction(
        self,
        tx_data: Dict,
        max_gas_price: Optional[int] = None
    ) -> Optional[str]:
        """Submit transaction through Gelato's private mempool."""
        try:
            # Prepare transaction data
            tx = {
                **tx_data,
                "from": self.web3.eth.account.from_key(self.private_key).address,
                "chainId": await self.web3.eth.chain_id
            }
            
            if max_gas_price:
                tx["maxFeePerGas"] = max_gas_price
            
            # Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(
                tx,
                self.private_key
            )
            
            # Submit to Gelato relay
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://relay.gelato.digital/relay/v2/sendPrivateTransaction",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "signedTransaction": signed_tx.rawTransaction.hex(),
                        "chainId": await self.web3.eth.chain_id
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("taskId")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error submitting private transaction: {str(e)}")
            return None

    async def create_arbitrage_task(
        self,
        flash_loan_data: Dict,
        conditions: Dict
    ) -> Optional[str]:
        """Create automated arbitrage task with conditions."""
        try:
            # Prepare resolver data
            resolver_data = self._prepare_resolver_data(conditions)
            
            # Prepare execution data
            execution_data = self._prepare_execution_data(flash_loan_data)
            
            # Create task through Gelato
            task_id = await self._create_task(
                resolver_data,
                execution_data
            )
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error creating arbitrage task: {str(e)}")
            return None

    def _prepare_resolver_data(self, conditions: Dict) -> Dict:
        """Prepare resolver data for task conditions."""
        return {
            "address": self.addresses["resolver"],
            "check": conditions
        }

    def _prepare_execution_data(self, flash_loan_data: Dict) -> Dict:
        """Prepare execution data for flash loan."""
        return {
            "to": flash_loan_data["target"],
            "data": flash_loan_data["data"],
            "value": "0",
            "operation": 0  # Call
        }

    async def _create_task(
        self,
        resolver_data: Dict,
        execution_data: Dict
    ) -> Optional[str]:
        """Create task through Gelato Ops."""
        try:
            # Prepare task creation transaction
            tx_data = self.ops_contract.functions.createTask(
                resolver_data,
                execution_data
            ).build_transaction({
                "from": self.web3.eth.account.from_key(self.private_key).address,
                "nonce": await self.web3.eth.get_transaction_count(
                    self.web3.eth.account.from_key(self.private_key).address
                )
            })
            
            # Submit through private mempool
            return await self.submit_private_transaction(tx_data)
            
        except Exception as e:
            self.logger.error(f"Error creating task: {str(e)}")
            return None

    async def monitor_task(self, task_id: str) -> Dict:
        """Monitor task status and execution."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.gelato.digital/tasks/status/{task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
            return {}
            
        except Exception as e:
            self.logger.error(f"Error monitoring task: {str(e)}")
            return {}

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        try:
            tx_data = self.ops_contract.functions.cancelTask(
                task_id
            ).build_transaction({
                "from": self.web3.eth.account.from_key(self.private_key).address,
                "nonce": await self.web3.eth.get_transaction_count(
                    self.web3.eth.account.from_key(self.private_key).address
                )
            })
            
            task_id = await self.submit_private_transaction(tx_data)
            return task_id is not None
            
        except Exception as e:
            self.logger.error(f"Error canceling task: {str(e)}")
            return False

    async def get_execution_cost(
        self,
        flash_loan_data: Dict
    ) -> Optional[Dict]:
        """Estimate execution cost through Gelato."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.gelato.digital/estimateExecution",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "chainId": await self.web3.eth.chain_id,
                        "data": flash_loan_data["data"],
                        "to": flash_loan_data["target"]
                    }
                ) as response:
                    if response.status == 200:
                        return await response.json()
            return None
            
        except Exception as e:
            self.logger.error(f"Error estimating execution cost: {str(e)}")
            return None

    async def fund_task(
        self,
        task_id: str,
        amount: int
    ) -> bool:
        """Fund a task with ETH for execution."""
        try:
            tx_data = {
                "to": self.addresses["ops"],
                "value": amount,
                "data": self.ops_contract.encodeABI(
                    fn_name="fundTask",
                    args=[task_id]
                )
            }
            
            task_id = await self.submit_private_transaction(tx_data)
            return task_id is not None
            
        except Exception as e:
            self.logger.error(f"Error funding task: {str(e)}")
            return False

    async def get_task_balance(self, task_id: str) -> int:
        """Get remaining balance for a task."""
        try:
            return await self.ops_contract.functions.getTaskBalance(
                task_id
            ).call()
            
        except Exception as e:
            self.logger.error(f"Error getting task balance: {str(e)}")
            return 0
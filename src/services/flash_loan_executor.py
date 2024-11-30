from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import json
from dataclasses import dataclass

@dataclass
class FlashLoanConfig:
    """Flash loan configuration."""
    provider: str
    token_address: str
    amount: int
    fee: float
    router_address: str
    callback_data: str
    gas_limit: int
    priority_fee: int

@dataclass
class ExecutionResult:
    """Flash loan execution result."""
    success: bool
    profit: float
    gas_used: int
    execution_time: float
    error: Optional[str]
    tx_hash: Optional[str]

class FlashLoanExecutor:
    """Advanced flash loan execution optimizer."""
    def __init__(
        self,
        web3: Web3,
        market_data_aggregator,
        strategy_optimizer,
        settings: Dict
    ):
        self.web3 = web3
        self.market_data = market_data_aggregator
        self.strategy = strategy_optimizer
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Flash loan providers
        self.providers = {
            "aave_v2": {
                "address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                "fee": 0.0009,  # 0.09%
                "router": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                "min_amount": Web3.to_wei(0.1, "ether"),
                "max_amount": Web3.to_wei(1000000, "ether"),
                "supported_tokens": [
                    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                    "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
                ]
            },
            "balancer": {
                "address": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                "fee": 0.0,  # No fee
                "router": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                "min_amount": Web3.to_wei(0.1, "ether"),
                "max_amount": Web3.to_wei(500000, "ether"),
                "supported_tokens": [
                    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                    "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
                ]
            },
            "dodo": {
                "address": "0x6D310348d5c12009854DFCf72e0DF9027e8cb4f4",
                "fee": 0.0002,  # 0.02%
                "router": "0x6D310348d5c12009854DFCf72e0DF9027e8cb4f4",
                "min_amount": Web3.to_wei(0.1, "ether"),
                "max_amount": Web3.to_wei(200000, "ether"),
                "supported_tokens": [
                    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"   # USDC
                ]
            }
        }
        
        # Execution history
        self.execution_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_profit": 0.0,
            "total_gas_used": 0,
            "avg_execution_time": 0.0
        }
        
        # Initialize contracts
        self._init_contracts()

    def _init_contracts(self):
        """Initialize flash loan contracts."""
        try:
            self.contracts = {}
            
            for name, provider in self.providers.items():
                # Load contract ABI
                abi = self._load_abi(name)
                
                # Initialize contract
                self.contracts[name] = self.web3.eth.contract(
                    address=provider["address"],
                    abi=abi
                )
            
            self.logger.info(f"Initialized {len(self.contracts)} flash loan contracts")
            
        except Exception as e:
            self.logger.error(f"Error initializing contracts: {str(e)}")
            raise

    def _load_abi(self, provider: str) -> List:
        """Load contract ABI."""
        try:
            abi_map = {
                "aave_v2": "contracts/abi/aave_v2_lending_pool.json",
                "balancer": "contracts/abi/balancer_vault.json",
                "dodo": "contracts/abi/dodo_pool.json"
            }
            
            with open(abi_map[provider], "r") as f:
                return json.load(f)
            
        except Exception as e:
            self.logger.error(f"Error loading ABI: {str(e)}")
            return []

    async def execute_flash_loan(
        self,
        opportunity: Dict,
        context: Dict
    ) -> ExecutionResult:
        """Execute flash loan arbitrage."""
        try:
            start_time = datetime.utcnow()
            
            # Get optimal flash loan configuration
            config = await self._get_optimal_config(
                opportunity,
                context
            )
            
            if not config:
                return ExecutionResult(
                    success=False,
                    profit=0.0,
                    gas_used=0,
                    execution_time=0.0,
                    error="No valid configuration found",
                    tx_hash=None
                )
            
            # Prepare transaction
            tx_data = await self._prepare_transaction(config)
            
            # Execute transaction
            tx_hash = await self._send_transaction(tx_data)
            
            if not tx_hash:
                return ExecutionResult(
                    success=False,
                    profit=0.0,
                    gas_used=0,
                    execution_time=0.0,
                    error="Transaction failed",
                    tx_hash=None
                )
            
            # Wait for confirmation
            result = await self._wait_for_confirmation(tx_hash)
            
            # Calculate execution time
            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            # Update metrics
            await self._update_metrics(
                result,
                execution_time,
                config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing flash loan: {str(e)}")
            return ExecutionResult(
                success=False,
                profit=0.0,
                gas_used=0,
                execution_time=0.0,
                error=str(e),
                tx_hash=None
            )

    async def _get_optimal_config(
        self,
        opportunity: Dict,
        context: Dict
    ) -> Optional[FlashLoanConfig]:
        """Get optimal flash loan configuration."""
        try:
            configs = []
            
            # Check each provider
            for name, provider in self.providers.items():
                # Validate token support
                if opportunity["token_in"] not in provider["supported_tokens"]:
                    continue
                
                # Validate amount limits
                amount = opportunity["amount_in"]
                if (
                    amount < provider["min_amount"] or
                    amount > provider["max_amount"]
                ):
                    continue
                
                # Calculate fees
                fee = amount * provider["fee"]
                
                # Calculate gas limit
                gas_limit = await self._estimate_gas_limit(
                    name,
                    opportunity
                )
                
                # Get priority fee
                priority_fee = await self._get_priority_fee(context)
                
                configs.append(
                    FlashLoanConfig(
                        provider=name,
                        token_address=opportunity["token_in"],
                        amount=amount,
                        fee=fee,
                        router_address=provider["router"],
                        callback_data=opportunity["callback_data"],
                        gas_limit=gas_limit,
                        priority_fee=priority_fee
                    )
                )
            
            if not configs:
                return None
            
            # Sort by total cost (fees + gas)
            configs.sort(
                key=lambda x: x.fee + (x.gas_limit * x.priority_fee)
            )
            
            return configs[0]
            
        except Exception as e:
            self.logger.error(f"Error getting optimal config: {str(e)}")
            return None

    async def _estimate_gas_limit(
        self,
        provider: str,
        opportunity: Dict
    ) -> int:
        """Estimate gas limit for flash loan."""
        try:
            # Base gas costs
            base_costs = {
                "aave_v2": 300000,
                "balancer": 250000,
                "dodo": 200000
            }
            
            # Add gas for each swap
            swap_gas = len(opportunity["path"]) * 100000
            
            # Add safety margin
            total_gas = base_costs[provider] + swap_gas
            return int(total_gas * 1.1)  # 10% safety margin
            
        except Exception as e:
            self.logger.error(f"Error estimating gas: {str(e)}")
            return 500000  # Safe default

    async def _get_priority_fee(self, context: Dict) -> int:
        """Calculate optimal priority fee."""
        try:
            # Get base fee
            base_fee = await self.web3.eth.get_block('latest')
            base_fee = base_fee.get('baseFeePerGas', 0)
            
            # Calculate priority fee based on network load
            network_load = context.get("network_load", 0.5)
            
            if network_load > 0.8:  # High load
                multiplier = 2.0
            elif network_load > 0.5:  # Medium load
                multiplier = 1.5
            else:  # Low load
                multiplier = 1.2
            
            priority_fee = int(base_fee * multiplier)
            
            # Ensure minimum priority fee
            return max(
                priority_fee,
                Web3.to_wei(1, "gwei")  # Minimum 1 gwei
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating priority fee: {str(e)}")
            return Web3.to_wei(2, "gwei")  # Safe default

    async def _prepare_transaction(
        self,
        config: FlashLoanConfig
    ) -> Dict:
        """Prepare flash loan transaction."""
        try:
            contract = self.contracts[config.provider]
            
            if config.provider == "aave_v2":
                return await self._prepare_aave_transaction(
                    contract,
                    config
                )
            elif config.provider == "balancer":
                return await self._prepare_balancer_transaction(
                    contract,
                    config
                )
            elif config.provider == "dodo":
                return await self._prepare_dodo_transaction(
                    contract,
                    config
                )
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
            
        except Exception as e:
            self.logger.error(f"Error preparing transaction: {str(e)}")
            return {}

    async def _prepare_aave_transaction(
        self,
        contract: Web3.eth.Contract,
        config: FlashLoanConfig
    ) -> Dict:
        """Prepare Aave flash loan transaction."""
        try:
            # Encode function call
            data = contract.encodeABI(
                fn_name="flashLoan",
                args=[
                    config.router_address,
                    [config.token_address],
                    [config.amount],
                    [0],  # Interest rate mode
                    config.router_address,
                    config.callback_data,
                    0  # referralCode
                ]
            )
            
            # Prepare transaction
            return {
                "to": contract.address,
                "data": data,
                "gas": config.gas_limit,
                "maxFeePerGas": Web3.to_wei(100, "gwei"),  # Adjust as needed
                "maxPriorityFeePerGas": config.priority_fee
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing Aave transaction: {str(e)}")
            return {}

    async def _prepare_balancer_transaction(
        self,
        contract: Web3.eth.Contract,
        config: FlashLoanConfig
    ) -> Dict:
        """Prepare Balancer flash loan transaction."""
        try:
            # Encode function call
            data = contract.encodeABI(
                fn_name="flashLoan",
                args=[
                    config.router_address,
                    [config.token_address],
                    [config.amount],
                    config.callback_data
                ]
            )
            
            # Prepare transaction
            return {
                "to": contract.address,
                "data": data,
                "gas": config.gas_limit,
                "maxFeePerGas": Web3.to_wei(100, "gwei"),
                "maxPriorityFeePerGas": config.priority_fee
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing Balancer transaction: {str(e)}")
            return {}

    async def _prepare_dodo_transaction(
        self,
        contract: Web3.eth.Contract,
        config: FlashLoanConfig
    ) -> Dict:
        """Prepare DODO flash loan transaction."""
        try:
            # Encode function call
            data = contract.encodeABI(
                fn_name="flashLoan",
                args=[
                    config.amount,
                    config.callback_data,
                    config.router_address
                ]
            )
            
            # Prepare transaction
            return {
                "to": contract.address,
                "data": data,
                "gas": config.gas_limit,
                "maxFeePerGas": Web3.to_wei(100, "gwei"),
                "maxPriorityFeePerGas": config.priority_fee
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing DODO transaction: {str(e)}")
            return {}

    async def _send_transaction(self, tx_data: Dict) -> Optional[str]:
        """Send transaction to network."""
        try:
            # Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(
                tx_data,
                self.settings["private_key"]
            )
            
            # Send transaction
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            return tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Error sending transaction: {str(e)}")
            return None

    async def _wait_for_confirmation(
        self,
        tx_hash: str
    ) -> ExecutionResult:
        """Wait for transaction confirmation."""
        try:
            # Wait for receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=30
            )
            
            if receipt["status"] == 1:
                # Calculate profit
                profit = await self._calculate_profit(receipt)
                
                return ExecutionResult(
                    success=True,
                    profit=profit,
                    gas_used=receipt["gasUsed"],
                    execution_time=0.0,  # Will be set by caller
                    error=None,
                    tx_hash=tx_hash
                )
            else:
                return ExecutionResult(
                    success=False,
                    profit=0.0,
                    gas_used=receipt["gasUsed"],
                    execution_time=0.0,
                    error="Transaction reverted",
                    tx_hash=tx_hash
                )
            
        except Exception as e:
            self.logger.error(f"Error waiting for confirmation: {str(e)}")
            return ExecutionResult(
                success=False,
                profit=0.0,
                gas_used=0,
                execution_time=0.0,
                error=str(e),
                tx_hash=tx_hash
            )

    async def _calculate_profit(self, receipt: Dict) -> float:
        """Calculate actual profit from transaction."""
        try:
            # Parse profit from event logs
            profit = 0.0
            
            for log in receipt["logs"]:
                if log["topics"][0].hex() == self.settings["profit_event_topic"]:
                    profit = int(log["data"], 16) / 1e18
            
            return profit
            
        except Exception as e:
            self.logger.error(f"Error calculating profit: {str(e)}")
            return 0.0

    async def _update_metrics(
        self,
        result: ExecutionResult,
        execution_time: float,
        config: FlashLoanConfig
    ):
        """Update performance metrics."""
        try:
            # Update counters
            self.metrics["total_executions"] += 1
            if result.success:
                self.metrics["successful_executions"] += 1
                self.metrics["total_profit"] += result.profit
            else:
                self.metrics["failed_executions"] += 1
            
            # Update gas metrics
            self.metrics["total_gas_used"] += result.gas_used
            
            # Update execution time
            self.metrics["avg_execution_time"] = (
                self.metrics["avg_execution_time"] *
                (self.metrics["total_executions"] - 1) +
                execution_time
            ) / self.metrics["total_executions"]
            
            # Add to history
            self.execution_history.append({
                "timestamp": datetime.utcnow(),
                "provider": config.provider,
                "amount": config.amount,
                "success": result.success,
                "profit": result.profit,
                "gas_used": result.gas_used,
                "execution_time": execution_time
            })
            
            # Keep fixed window size
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        try:
            return {
                **self.metrics,
                "success_rate": (
                    self.metrics["successful_executions"] /
                    max(1, self.metrics["total_executions"])
                ),
                "avg_profit_per_trade": (
                    self.metrics["total_profit"] /
                    max(1, self.metrics["successful_executions"])
                ),
                "avg_gas_per_trade": (
                    self.metrics["total_gas_used"] /
                    max(1, self.metrics["total_executions"])
                ),
                "execution_history_size": len(self.execution_history),
                "last_update": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {
                "error": str(e),
                "last_update": datetime.utcnow().isoformat()
            }
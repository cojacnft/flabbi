from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address

class FlashLoanOptimizer:
    """Optimize flashloan-based arbitrage strategies."""
    def __init__(
        self,
        web3: Web3,
        chain_id: int,
        market_data_aggregator,
        profit_analyzer
    ):
        self.web3 = web3
        self.chain_id = chain_id
        self.market_data = market_data_aggregator
        self.profit_analyzer = profit_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Flash loan providers
        self.providers: Dict[str, Dict] = {
            "aave_v2": {
                "address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                "fee": 0.0009,  # 0.09%
                "max_tokens": True,  # Supports multiple tokens
                "enabled": True
            },
            "aave_v3": {
                "address": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
                "fee": 0.0005,  # 0.05%
                "max_tokens": True,
                "enabled": True
            },
            "balancer": {
                "address": "0xBA12222222228d8Ba445958a75a0704d566BF2C8",
                "fee": 0.0,  # No fee
                "max_tokens": True,
                "enabled": True
            },
            "dodo": {
                "address": "0x6D310348d5c12009854DFCf72e0DF9027e8cb4f4",
                "fee": 0.0002,  # 0.02%
                "max_tokens": False,  # Single token only
                "enabled": True
            }
        }
        
        # Performance metrics
        self.metrics = {
            "total_flashloans": 0,
            "successful_flashloans": 0,
            "failed_flashloans": 0,
            "total_borrowed_usd": 0.0,
            "total_fees_paid_usd": 0.0,
            "avg_profit_per_loan": 0.0
        }
        
        # Initialize provider-specific data
        self._init_providers()

    def _init_providers(self):
        """Initialize flash loan provider data."""
        try:
            # Load provider ABIs
            self.provider_abis = {
                "aave_v2": self._load_abi("aave_v2_lending_pool"),
                "aave_v3": self._load_abi("aave_v3_pool"),
                "balancer": self._load_abi("balancer_vault"),
                "dodo": self._load_abi("dodo_pool")
            }
            
            # Initialize contracts
            self.provider_contracts = {}
            for name, data in self.providers.items():
                if data["enabled"]:
                    self.provider_contracts[name] = self.web3.eth.contract(
                        address=data["address"],
                        abi=self.provider_abis[name]
                    )
            
            self.logger.info(f"Initialized {len(self.provider_contracts)} flash loan providers")
            
        except Exception as e:
            self.logger.error(f"Error initializing providers: {str(e)}")
            raise

    def _load_abi(self, name: str) -> List:
        """Load contract ABI."""
        try:
            with open(f"contracts/abi/{name}.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading ABI {name}: {str(e)}")
            return []

    async def get_optimal_flashloan(
        self,
        tokens: List[str],
        amounts: List[int],
        min_profit: float = 100.0
    ) -> Optional[Dict]:
        """Get optimal flash loan configuration."""
        try:
            best_config = None
            max_profit = min_profit
            
            # Check each provider
            for name, provider in self.providers.items():
                if not provider["enabled"]:
                    continue
                
                # Validate provider supports token count
                if not provider["max_tokens"] and len(tokens) > 1:
                    continue
                
                # Calculate fees
                fee_amount = sum(amounts) * provider["fee"]
                
                # Get available liquidity
                liquidity = await self._get_provider_liquidity(
                    name,
                    tokens
                )
                
                # Check liquidity
                if not self._validate_liquidity(amounts, liquidity):
                    continue
                
                # Calculate potential profit
                profit = await self._calculate_profit(
                    tokens,
                    amounts,
                    fee_amount
                )
                
                if profit > max_profit:
                    max_profit = profit
                    best_config = {
                        "provider": name,
                        "tokens": tokens,
                        "amounts": amounts,
                        "fee": fee_amount,
                        "expected_profit": profit,
                        "liquidity": liquidity
                    }
            
            return best_config
            
        except Exception as e:
            self.logger.error(f"Error getting optimal flashloan: {str(e)}")
            return None

    async def _get_provider_liquidity(
        self,
        provider: str,
        tokens: List[str]
    ) -> Dict[str, int]:
        """Get provider liquidity for tokens."""
        try:
            liquidity = {}
            contract = self.provider_contracts[provider]
            
            for token in tokens:
                if provider in ["aave_v2", "aave_v3"]:
                    # Aave liquidity
                    reserve_data = await contract.functions.getReserveData(
                        token
                    ).call()
                    liquidity[token] = reserve_data[0]  # Available liquidity
                
                elif provider == "balancer":
                    # Balancer liquidity
                    pool_tokens = await contract.functions.getPoolTokens(
                        self._get_balancer_pool_id(token)
                    ).call()
                    token_index = pool_tokens[0].index(token)
                    liquidity[token] = pool_tokens[1][token_index]
                
                elif provider == "dodo":
                    # DODO liquidity
                    pool_info = await contract.functions.getPoolInfo().call()
                    liquidity[token] = pool_info[1]  # Base token reserve
            
            return liquidity
            
        except Exception as e:
            self.logger.error(f"Error getting provider liquidity: {str(e)}")
            return {}

    def _validate_liquidity(
        self,
        amounts: List[int],
        liquidity: Dict[str, int]
    ) -> bool:
        """Validate sufficient liquidity exists."""
        try:
            for token, amount in zip(liquidity.keys(), amounts):
                if amount > liquidity.get(token, 0) * 0.95:  # 95% max utilization
                    return False
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating liquidity: {str(e)}")
            return False

    async def _calculate_profit(
        self,
        tokens: List[str],
        amounts: List[int],
        fee_amount: float
    ) -> float:
        """Calculate potential profit after fees."""
        try:
            # Get token prices
            prices = {}
            for token in tokens:
                price = await self.market_data.get_token_price(token)
                if price is None:
                    return 0.0
                prices[token] = price
            
            # Calculate borrowed value
            borrowed_value = sum(
                amount * prices[token]
                for token, amount in zip(tokens, amounts)
            )
            
            # Get arbitrage profit
            profit = await self.profit_analyzer.calculate_profit(
                tokens,
                amounts,
                prices
            )
            
            # Subtract fees
            net_profit = profit - (fee_amount * prices[tokens[0]])
            
            return net_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating profit: {str(e)}")
            return 0.0

    async def prepare_flashloan(
        self,
        config: Dict,
        target_contract: str,
        callback_data: str
    ) -> Optional[Dict]:
        """Prepare flash loan transaction."""
        try:
            provider = config["provider"]
            contract = self.provider_contracts[provider]
            
            if provider in ["aave_v2", "aave_v3"]:
                # Prepare Aave flash loan
                tx_data = await self._prepare_aave_flashloan(
                    contract,
                    config,
                    target_contract,
                    callback_data
                )
            
            elif provider == "balancer":
                # Prepare Balancer flash loan
                tx_data = await self._prepare_balancer_flashloan(
                    contract,
                    config,
                    target_contract,
                    callback_data
                )
            
            elif provider == "dodo":
                # Prepare DODO flash loan
                tx_data = await self._prepare_dodo_flashloan(
                    contract,
                    config,
                    target_contract,
                    callback_data
                )
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing flashloan: {str(e)}")
            return None

    async def _prepare_aave_flashloan(
        self,
        contract: Web3.eth.Contract,
        config: Dict,
        target_contract: str,
        callback_data: str
    ) -> Dict:
        """Prepare Aave flash loan transaction."""
        try:
            # Encode params
            params = self.web3.eth.abi.encode_abi(
                ["address", "bytes"],
                [target_contract, callback_data]
            )
            
            # Prepare transaction
            tx_data = contract.functions.flashLoan(
                target_contract,
                config["tokens"],
                config["amounts"],
                [0] * len(config["tokens"]),  # Interest rate modes
                target_contract,
                params,
                0  # referralCode
            ).build_transaction({
                "from": target_contract,
                "gas": 2000000,
                "maxFeePerGas": Web3.to_wei(50, "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(2, "gwei")
            })
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing Aave flashloan: {str(e)}")
            return {}

    async def _prepare_balancer_flashloan(
        self,
        contract: Web3.eth.Contract,
        config: Dict,
        target_contract: str,
        callback_data: str
    ) -> Dict:
        """Prepare Balancer flash loan transaction."""
        try:
            # Encode params
            params = self.web3.eth.abi.encode_abi(
                ["bytes"],
                [callback_data]
            )
            
            # Prepare transaction
            tx_data = contract.functions.flashLoan(
                target_contract,
                config["tokens"],
                config["amounts"],
                params
            ).build_transaction({
                "from": target_contract,
                "gas": 2000000,
                "maxFeePerGas": Web3.to_wei(50, "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(2, "gwei")
            })
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing Balancer flashloan: {str(e)}")
            return {}

    async def _prepare_dodo_flashloan(
        self,
        contract: Web3.eth.Contract,
        config: Dict,
        target_contract: str,
        callback_data: str
    ) -> Dict:
        """Prepare DODO flash loan transaction."""
        try:
            # DODO only supports single token flash loans
            if len(config["tokens"]) > 1:
                raise ValueError("DODO only supports single token flash loans")
            
            # Prepare transaction
            tx_data = contract.functions.flashLoan(
                config["amounts"][0],
                callback_data,
                target_contract
            ).build_transaction({
                "from": target_contract,
                "gas": 2000000,
                "maxFeePerGas": Web3.to_wei(50, "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(2, "gwei")
            })
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing DODO flashloan: {str(e)}")
            return {}

    async def execute_flashloan(
        self,
        tx_data: Dict,
        private_key: str
    ) -> Optional[str]:
        """Execute flash loan transaction."""
        try:
            # Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(
                tx_data,
                private_key
            )
            
            # Send transaction
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Update metrics
            self.metrics["total_flashloans"] += 1
            self.metrics["total_borrowed_usd"] += self._calculate_borrowed_value(
                tx_data
            )
            
            return tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Error executing flashloan: {str(e)}")
            return None

    def _calculate_borrowed_value(self, tx_data: Dict) -> float:
        """Calculate total borrowed value in USD."""
        try:
            # Extract amounts from transaction data
            # This is provider-specific and needs proper decoding
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating borrowed value: {str(e)}")
            return 0.0

    async def update_metrics(
        self,
        success: bool,
        profit: float,
        fees_paid: float
    ):
        """Update performance metrics."""
        try:
            if success:
                self.metrics["successful_flashloans"] += 1
                total_profit = (
                    self.metrics["avg_profit_per_loan"] *
                    (self.metrics["successful_flashloans"] - 1) +
                    profit
                )
                self.metrics["avg_profit_per_loan"] = (
                    total_profit / self.metrics["successful_flashloans"]
                )
            else:
                self.metrics["failed_flashloans"] += 1
            
            self.metrics["total_fees_paid_usd"] += fees_paid
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_flashloans"] /
                max(1, self.metrics["total_flashloans"])
            ),
            "last_update": datetime.utcnow().isoformat()
        }
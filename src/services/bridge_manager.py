from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address

class BridgeProtocol:
    """Base class for bridge protocols."""
    def __init__(
        self,
        name: str,
        source_chain: int,
        dest_chain: int,
        contract_address: str,
        token_address: str
    ):
        self.name = name
        self.source_chain = source_chain
        self.dest_chain = dest_chain
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.token_address = Web3.to_checksum_address(token_address)
        
        # Bridge metrics
        self.metrics = {
            "total_volume": 0.0,
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "avg_transfer_time": 0.0
        }

    async def get_liquidity(self, web3: Web3) -> float:
        """Get bridge liquidity."""
        raise NotImplementedError

    async def get_fees(self, amount: int, web3: Web3) -> int:
        """Get bridge fees for amount."""
        raise NotImplementedError

    async def estimate_transfer_time(self) -> float:
        """Estimate transfer time in seconds."""
        raise NotImplementedError

    async def validate_transfer(
        self,
        amount: int,
        source_chain: int,
        dest_chain: int
    ) -> bool:
        """Validate transfer is possible."""
        raise NotImplementedError

    def get_metrics(self) -> Dict:
        """Get bridge metrics."""
        return {
            **self.metrics,
            "last_update": datetime.utcnow().isoformat()
        }


class HopProtocol(BridgeProtocol):
    """Hop Protocol bridge implementation."""
    def __init__(
        self,
        source_chain: int,
        dest_chain: int,
        contract_address: str,
        token_address: str
    ):
        super().__init__(
            "Hop Protocol",
            source_chain,
            dest_chain,
            contract_address,
            token_address
        )
        
        # Hop-specific settings
        self.min_bonder_fee = Web3.to_wei(0.001, "ether")
        self.max_slippage = 0.005  # 0.5%

    async def get_liquidity(self, web3: Web3) -> float:
        """Get Hop bridge liquidity."""
        try:
            contract = web3.eth.contract(
                address=self.contract_address,
                abi=HOP_BRIDGE_ABI
            )
            
            liquidity = await contract.functions.getAvailableLiquidity().call()
            return float(Web3.from_wei(liquidity, "ether"))
            
        except Exception as e:
            logging.error(f"Error getting Hop liquidity: {str(e)}")
            return 0.0

    async def get_fees(self, amount: int, web3: Web3) -> int:
        """Get Hop bridge fees."""
        try:
            contract = web3.eth.contract(
                address=self.contract_address,
                abi=HOP_BRIDGE_ABI
            )
            
            # Get bonder fee
            bonder_fee = await contract.functions.getBonderFee(amount).call()
            
            # Get LP fee
            lp_fee = amount * 0.0005  # 0.05% LP fee
            
            return bonder_fee + int(lp_fee)
            
        except Exception as e:
            logging.error(f"Error getting Hop fees: {str(e)}")
            return 0

    async def estimate_transfer_time(self) -> float:
        """Estimate Hop transfer time."""
        # Hop transfer times:
        # Ethereum -> L2: ~15 minutes
        # L2 -> Ethereum: ~7 days (challenge period)
        # L2 -> L2: ~15 minutes
        if self.source_chain == 1:  # From Ethereum
            return 900  # 15 minutes
        elif self.dest_chain == 1:  # To Ethereum
            return 604800  # 7 days
        else:  # L2 to L2
            return 900  # 15 minutes

    async def validate_transfer(
        self,
        amount: int,
        source_chain: int,
        dest_chain: int
    ) -> bool:
        """Validate Hop transfer."""
        try:
            # Check chains match
            if (
                source_chain != self.source_chain or
                dest_chain != self.dest_chain
            ):
                return False
            
            # Check minimum amount
            if amount < Web3.to_wei(0.001, "ether"):
                return False
            
            # Check maximum amount (if any)
            if hasattr(self, "max_transfer_amount"):
                if amount > self.max_transfer_amount:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating Hop transfer: {str(e)}")
            return False


class AcrossProtocol(BridgeProtocol):
    """Across Protocol bridge implementation."""
    def __init__(
        self,
        source_chain: int,
        dest_chain: int,
        contract_address: str,
        token_address: str
    ):
        super().__init__(
            "Across Protocol",
            source_chain,
            dest_chain,
            contract_address,
            token_address
        )
        
        # Across-specific settings
        self.min_transfer = Web3.to_wei(0.01, "ether")
        self.max_slippage = 0.003  # 0.3%

    async def get_liquidity(self, web3: Web3) -> float:
        """Get Across bridge liquidity."""
        try:
            contract = web3.eth.contract(
                address=self.contract_address,
                abi=ACROSS_BRIDGE_ABI
            )
            
            liquidity = await contract.functions.pooledTokens().call()
            return float(Web3.from_wei(liquidity, "ether"))
            
        except Exception as e:
            logging.error(f"Error getting Across liquidity: {str(e)}")
            return 0.0

    async def get_fees(self, amount: int, web3: Web3) -> int:
        """Get Across bridge fees."""
        try:
            contract = web3.eth.contract(
                address=self.contract_address,
                abi=ACROSS_BRIDGE_ABI
            )
            
            # Get relayer fee
            relayer_fee = await contract.functions.getRelayerFee(amount).call()
            
            # Get protocol fee (0.1%)
            protocol_fee = amount * 0.001
            
            return relayer_fee + int(protocol_fee)
            
        except Exception as e:
            logging.error(f"Error getting Across fees: {str(e)}")
            return 0

    async def estimate_transfer_time(self) -> float:
        """Estimate Across transfer time."""
        # Across transfer times:
        # Optimistic rollups: ~1 hour
        # ZK rollups: ~15 minutes
        if self.dest_chain in [42161, 10]:  # Arbitrum, Optimism
            return 3600  # 1 hour
        else:
            return 900  # 15 minutes

    async def validate_transfer(
        self,
        amount: int,
        source_chain: int,
        dest_chain: int
    ) -> bool:
        """Validate Across transfer."""
        try:
            # Check chains match
            if (
                source_chain != self.source_chain or
                dest_chain != self.dest_chain
            ):
                return False
            
            # Check minimum amount
            if amount < self.min_transfer:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating Across transfer: {str(e)}")
            return False


class BridgeManager:
    """Manage cross-chain bridges."""
    def __init__(self, web3_providers: Dict[int, Web3]):
        self.web3_providers = web3_providers
        self.logger = logging.getLogger(__name__)
        
        # Initialize bridges
        self.bridges: Dict[str, BridgeProtocol] = {}
        self._init_bridges()
        
        # Bridge states
        self.bridge_states: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            "total_bridges": 0,
            "active_bridges": 0,
            "total_volume_usd": 0.0,
            "successful_transfers": 0,
            "failed_transfers": 0
        }

    def _init_bridges(self):
        """Initialize supported bridges."""
        try:
            # Hop Protocol bridges
            self._add_hop_bridges()
            
            # Across Protocol bridges
            self._add_across_bridges()
            
            self.metrics["total_bridges"] = len(self.bridges)
            self.logger.info(f"Initialized {len(self.bridges)} bridges")
            
        except Exception as e:
            self.logger.error(f"Error initializing bridges: {str(e)}")

    def _add_hop_bridges(self):
        """Add Hop Protocol bridges."""
        hop_bridges = [
            # Ethereum <-> Polygon
            {
                "source_chain": 1,
                "dest_chain": 137,
                "contract": "0x...",  # Hop ETH->Polygon contract
                "token": "0x..."      # ETH token address
            },
            {
                "source_chain": 137,
                "dest_chain": 1,
                "contract": "0x...",  # Hop Polygon->ETH contract
                "token": "0x..."      # MATIC token address
            },
            # Ethereum <-> Arbitrum
            {
                "source_chain": 1,
                "dest_chain": 42161,
                "contract": "0x...",  # Hop ETH->Arbitrum contract
                "token": "0x..."      # ETH token address
            },
            {
                "source_chain": 42161,
                "dest_chain": 1,
                "contract": "0x...",  # Hop Arbitrum->ETH contract
                "token": "0x..."      # ETH token address
            }
        ]
        
        for bridge in hop_bridges:
            bridge_id = f"hop_{bridge['source_chain']}_{bridge['dest_chain']}"
            self.bridges[bridge_id] = HopProtocol(
                bridge["source_chain"],
                bridge["dest_chain"],
                bridge["contract"],
                bridge["token"]
            )

    def _add_across_bridges(self):
        """Add Across Protocol bridges."""
        across_bridges = [
            # Ethereum <-> Optimism
            {
                "source_chain": 1,
                "dest_chain": 10,
                "contract": "0x...",  # Across ETH->Optimism contract
                "token": "0x..."      # ETH token address
            },
            {
                "source_chain": 10,
                "dest_chain": 1,
                "contract": "0x...",  # Across Optimism->ETH contract
                "token": "0x..."      # ETH token address
            },
            # Ethereum <-> Arbitrum
            {
                "source_chain": 1,
                "dest_chain": 42161,
                "contract": "0x...",  # Across ETH->Arbitrum contract
                "token": "0x..."      # ETH token address
            },
            {
                "source_chain": 42161,
                "dest_chain": 1,
                "contract": "0x...",  # Across Arbitrum->ETH contract
                "token": "0x..."      # ETH token address
            }
        ]
        
        for bridge in across_bridges:
            bridge_id = f"across_{bridge['source_chain']}_{bridge['dest_chain']}"
            self.bridges[bridge_id] = AcrossProtocol(
                bridge["source_chain"],
                bridge["dest_chain"],
                bridge["contract"],
                bridge["token"]
            )

    async def get_best_bridge(
        self,
        source_chain: int,
        dest_chain: int,
        amount: int,
        max_time: Optional[float] = None
    ) -> Optional[Tuple[str, Dict]]:
        """Get best bridge for transfer."""
        try:
            valid_bridges = []
            
            # Check all bridges
            for bridge_id, bridge in self.bridges.items():
                # Validate transfer
                if not await bridge.validate_transfer(
                    amount,
                    source_chain,
                    dest_chain
                ):
                    continue
                
                # Get bridge metrics
                liquidity = await bridge.get_liquidity(
                    self.web3_providers[source_chain]
                )
                fees = await bridge.get_fees(
                    amount,
                    self.web3_providers[source_chain]
                )
                time = await bridge.estimate_transfer_time()
                
                # Check time constraint
                if max_time and time > max_time:
                    continue
                
                # Check liquidity
                if liquidity < amount:
                    continue
                
                valid_bridges.append({
                    "bridge_id": bridge_id,
                    "liquidity": liquidity,
                    "fees": fees,
                    "time": time
                })
            
            if not valid_bridges:
                return None
            
            # Sort by fees
            valid_bridges.sort(key=lambda x: x["fees"])
            
            return valid_bridges[0]["bridge_id"], valid_bridges[0]
            
        except Exception as e:
            self.logger.error(f"Error getting best bridge: {str(e)}")
            return None

    async def execute_bridge_transfer(
        self,
        bridge_id: str,
        amount: int,
        recipient: str,
        max_slippage: float = 0.01
    ) -> Optional[str]:
        """Execute bridge transfer."""
        try:
            bridge = self.bridges[bridge_id]
            web3 = self.web3_providers[bridge.source_chain]
            
            # Get contract
            contract = web3.eth.contract(
                address=bridge.contract_address,
                abi=self._get_bridge_abi(bridge_id)
            )
            
            # Prepare transaction
            tx_data = await self._prepare_bridge_tx(
                bridge,
                contract,
                amount,
                recipient,
                max_slippage
            )
            
            # Send transaction
            tx_hash = await self._send_transaction(tx_data, web3)
            
            if tx_hash:
                # Update metrics
                self.metrics["total_volume_usd"] += amount * self._get_token_price(
                    bridge.token_address
                )
                bridge.metrics["total_transfers"] += 1
            
            return tx_hash
            
        except Exception as e:
            self.logger.error(f"Error executing bridge transfer: {str(e)}")
            return None

    def _get_bridge_abi(self, bridge_id: str) -> List:
        """Get bridge contract ABI."""
        if bridge_id.startswith("hop"):
            return HOP_BRIDGE_ABI
        elif bridge_id.startswith("across"):
            return ACROSS_BRIDGE_ABI
        else:
            raise ValueError(f"Unknown bridge: {bridge_id}")

    async def _prepare_bridge_tx(
        self,
        bridge: BridgeProtocol,
        contract: Web3.eth.Contract,
        amount: int,
        recipient: str,
        max_slippage: float
    ) -> Dict:
        """Prepare bridge transaction."""
        try:
            # Get fees
            fees = await bridge.get_fees(
                amount,
                self.web3_providers[bridge.source_chain]
            )
            
            # Calculate minimum received
            min_received = int(amount * (1 - max_slippage))
            
            # Prepare transaction data
            if isinstance(bridge, HopProtocol):
                tx_data = contract.functions.send(
                    recipient,
                    amount,
                    min_received,
                    {"value": fees}
                ).build_transaction({
                    "from": recipient,
                    "gas": 500000,
                    "maxFeePerGas": Web3.to_wei(50, "gwei"),
                    "maxPriorityFeePerGas": Web3.to_wei(2, "gwei")
                })
            elif isinstance(bridge, AcrossProtocol):
                tx_data = contract.functions.deposit(
                    bridge.token_address,
                    amount,
                    bridge.dest_chain,
                    recipient,
                    min_received
                ).build_transaction({
                    "from": recipient,
                    "gas": 500000,
                    "maxFeePerGas": Web3.to_wei(50, "gwei"),
                    "maxPriorityFeePerGas": Web3.to_wei(2, "gwei")
                })
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing bridge tx: {str(e)}")
            return {}

    async def _send_transaction(
        self,
        tx_data: Dict,
        web3: Web3
    ) -> Optional[str]:
        """Send transaction to network."""
        try:
            # Sign transaction
            signed_tx = web3.eth.account.sign_transaction(
                tx_data,
                private_key="YOUR_PRIVATE_KEY"
            )
            
            # Send transaction
            tx_hash = await web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            return tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Error sending transaction: {str(e)}")
            return None

    def _get_token_price(self, token_address: str) -> float:
        """Get token price in USD."""
        # TODO: Implement price fetching
        return 1.0

    async def update_bridge_states(self):
        """Update states for all bridges."""
        try:
            active_bridges = 0
            
            for bridge_id, bridge in self.bridges.items():
                # Get liquidity
                liquidity = await bridge.get_liquidity(
                    self.web3_providers[bridge.source_chain]
                )
                
                # Get 24h volume
                volume = self._get_24h_volume(bridge_id)
                
                # Update state
                self.bridge_states[bridge_id] = {
                    "liquidity": liquidity,
                    "volume_24h": volume,
                    "metrics": bridge.get_metrics(),
                    "last_update": datetime.utcnow()
                }
                
                if liquidity > 0:
                    active_bridges += 1
            
            self.metrics["active_bridges"] = active_bridges
            
        except Exception as e:
            self.logger.error(f"Error updating bridge states: {str(e)}")

    def _get_24h_volume(self, bridge_id: str) -> float:
        """Get 24h volume for bridge."""
        # TODO: Implement volume tracking
        return 0.0

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "bridge_states": self.bridge_states,
            "last_update": datetime.utcnow().isoformat()
        }
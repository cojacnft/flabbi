from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import aiohttp
import json

class MEVProtector:
    """Protect transactions from MEV attacks."""
    def __init__(
        self,
        web3: Web3,
        chain_id: int,
        settings: Dict
    ):
        self.web3 = web3
        self.chain_id = chain_id
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Protection services
        self.services = {
            "flashbots": {
                "enabled": True,
                "url": "https://relay.flashbots.net",
                "min_bribe": 0.1,  # 10% of profit
                "max_bribe": 0.3   # 30% of profit
            },
            "eden": {
                "enabled": True,
                "url": "https://api.edennetwork.io/v1/rpc",
                "min_bribe": 0.05,  # 5% of profit
                "max_bribe": 0.2    # 20% of profit
            },
            "bloxroute": {
                "enabled": True,
                "url": "https://api.bloxroute.com/v1",
                "min_bribe": 0.08,  # 8% of profit
                "max_bribe": 0.25   # 25% of profit
            }
        }
        
        # Bundle history
        self.bundle_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            "bundles_sent": 0,
            "bundles_included": 0,
            "total_bribes_eth": 0.0,
            "sandwiched_txs": 0,
            "frontrun_attempts": 0,
            "backrun_attempts": 0
        }
        
        # Initialize services
        self._init_services()

    def _init_services(self):
        """Initialize MEV protection services."""
        try:
            # Initialize Flashbots
            if self.services["flashbots"]["enabled"]:
                self._init_flashbots()
            
            # Initialize Eden Network
            if self.services["eden"]["enabled"]:
                self._init_eden()
            
            # Initialize bloXroute
            if self.services["bloxroute"]["enabled"]:
                self._init_bloxroute()
            
            self.logger.info("MEV protection services initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing MEV services: {str(e)}")
            raise

    def _init_flashbots(self):
        """Initialize Flashbots connection."""
        try:
            # Load Flashbots signer
            self.flashbots_signer = self.web3.eth.account.from_key(
                self.settings["flashbots_private_key"]
            )
            
            # Set auth header
            self.flashbots_auth = self.web3.eth.sign(
                self.flashbots_signer.address,
                text="flashbots-auth"
            ).hex()
            
        except Exception as e:
            self.logger.error(f"Error initializing Flashbots: {str(e)}")
            self.services["flashbots"]["enabled"] = False

    def _init_eden(self):
        """Initialize Eden Network connection."""
        try:
            # Set Eden API key
            self.eden_api_key = self.settings["eden_api_key"]
            
        except Exception as e:
            self.logger.error(f"Error initializing Eden: {str(e)}")
            self.services["eden"]["enabled"] = False

    def _init_bloxroute(self):
        """Initialize bloXroute connection."""
        try:
            # Set bloXroute auth token
            self.bloxroute_auth = self.settings["bloxroute_auth_token"]
            
        except Exception as e:
            self.logger.error(f"Error initializing bloXroute: {str(e)}")
            self.services["bloxroute"]["enabled"] = False

    async def protect_transaction(
        self,
        tx_data: Dict,
        expected_profit: float,
        max_delay: int = 3  # blocks
    ) -> Optional[Dict]:
        """Protect transaction from MEV attacks."""
        try:
            # Check for MEV risks
            risks = await self._analyze_mev_risks(tx_data)
            
            if not risks["high_risk"]:
                # No significant MEV risk, return original tx
                return tx_data
            
            # Calculate optimal bribe
            bribe = self._calculate_optimal_bribe(
                expected_profit,
                risks["risk_score"]
            )
            
            # Prepare bundles for each service
            bundles = await self._prepare_bundles(
                tx_data,
                bribe,
                risks
            )
            
            # Send bundles to all enabled services
            results = await asyncio.gather(*[
                self._send_bundle(service, bundle)
                for service, bundle in bundles.items()
            ])
            
            # Get best result
            best_result = self._get_best_result(results)
            
            if best_result:
                return best_result
            
            # Fallback: Return original tx with higher gas price
            return self._prepare_fallback_tx(tx_data)
            
        except Exception as e:
            self.logger.error(f"Error protecting transaction: {str(e)}")
            return None

    async def _analyze_mev_risks(self, tx_data: Dict) -> Dict:
        """Analyze transaction for MEV risks."""
        try:
            risks = {
                "high_risk": False,
                "risk_score": 0.0,
                "risk_factors": []
            }
            
            # Check mempool for similar transactions
            mempool_risks = await self._check_mempool(tx_data)
            risks["risk_factors"].extend(mempool_risks)
            
            # Check for sandwich attack risks
            sandwich_risk = await self._check_sandwich_risk(tx_data)
            if sandwich_risk > 0.5:
                risks["risk_factors"].append({
                    "type": "sandwich",
                    "risk": sandwich_risk
                })
            
            # Check for frontrunning risks
            frontrun_risk = await self._check_frontrun_risk(tx_data)
            if frontrun_risk > 0.5:
                risks["risk_factors"].append({
                    "type": "frontrun",
                    "risk": frontrun_risk
                })
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(risks["risk_factors"])
            risks["risk_score"] = risk_score
            risks["high_risk"] = risk_score > 0.7  # 70% threshold
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error analyzing MEV risks: {str(e)}")
            return {"high_risk": True, "risk_score": 1.0, "risk_factors": []}

    async def _check_mempool(self, tx_data: Dict) -> List[Dict]:
        """Check mempool for similar transactions."""
        try:
            risks = []
            
            # Get pending transactions
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.services['bloxroute']['url']}/mempool",
                    headers={"Authorization": self.bloxroute_auth}
                ) as response:
                    if response.status == 200:
                        mempool = await response.json()
                        
                        # Analyze similar transactions
                        for tx in mempool:
                            similarity = self._calculate_tx_similarity(
                                tx_data,
                                tx
                            )
                            if similarity > 0.8:  # 80% similar
                                risks.append({
                                    "type": "similar_tx",
                                    "risk": similarity,
                                    "tx_hash": tx["hash"]
                                })
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error checking mempool: {str(e)}")
            return []

    async def _check_sandwich_risk(self, tx_data: Dict) -> float:
        """Check risk of sandwich attacks."""
        try:
            # Decode transaction
            if not tx_data.get("data"):
                return 0.0
            
            # Check if it's a DEX trade
            if not self._is_dex_trade(tx_data):
                return 0.0
            
            # Get pool state
            pool_data = await self._get_pool_data(tx_data)
            
            # Calculate potential slippage from sandwich
            slippage = self._calculate_sandwich_slippage(
                tx_data,
                pool_data
            )
            
            # Convert slippage to risk score (0-1)
            risk = min(1.0, slippage * 5)  # 20% slippage = max risk
            
            return risk
            
        except Exception as e:
            self.logger.error(f"Error checking sandwich risk: {str(e)}")
            return 1.0

    async def _check_frontrun_risk(self, tx_data: Dict) -> float:
        """Check risk of frontrunning."""
        try:
            # Check gas price
            network_gas = await self.web3.eth.gas_price
            tx_gas = tx_data.get("gasPrice", 0)
            
            if tx_gas < network_gas:
                return 1.0  # High risk if below network gas price
            
            # Check recent blocks for similar patterns
            patterns = await self._analyze_block_patterns(tx_data)
            
            if patterns["frontrun_frequency"] > 0.5:
                return patterns["frontrun_frequency"]
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error checking frontrun risk: {str(e)}")
            return 1.0

    def _calculate_risk_score(self, risk_factors: List[Dict]) -> float:
        """Calculate overall risk score."""
        try:
            if not risk_factors:
                return 0.0
            
            # Weight different risk types
            weights = {
                "similar_tx": 0.3,
                "sandwich": 0.4,
                "frontrun": 0.3
            }
            
            # Calculate weighted average
            total_score = 0.0
            total_weight = 0.0
            
            for factor in risk_factors:
                weight = weights.get(factor["type"], 0.1)
                total_score += factor["risk"] * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            return total_score / total_weight
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            return 1.0

    def _calculate_optimal_bribe(
        self,
        expected_profit: float,
        risk_score: float
    ) -> float:
        """Calculate optimal bribe amount."""
        try:
            # Base bribe as percentage of profit
            base_bribe = expected_profit * 0.1  # Start at 10%
            
            # Adjust based on risk score
            risk_multiplier = 1 + (risk_score * 2)  # Up to 3x for max risk
            
            # Calculate final bribe
            bribe = base_bribe * risk_multiplier
            
            # Apply service-specific limits
            min_bribe = max(
                service["min_bribe"] * expected_profit
                for service in self.services.values()
                if service["enabled"]
            )
            
            max_bribe = min(
                service["max_bribe"] * expected_profit
                for service in self.services.values()
                if service["enabled"]
            )
            
            # Clamp bribe amount
            return max(min_bribe, min(bribe, max_bribe))
            
        except Exception as e:
            self.logger.error(f"Error calculating bribe: {str(e)}")
            return expected_profit * 0.1  # Default to 10%

    async def _prepare_bundles(
        self,
        tx_data: Dict,
        bribe: float,
        risks: Dict
    ) -> Dict[str, Dict]:
        """Prepare bundles for each service."""
        try:
            bundles = {}
            
            # Prepare Flashbots bundle
            if self.services["flashbots"]["enabled"]:
                bundles["flashbots"] = await self._prepare_flashbots_bundle(
                    tx_data,
                    bribe,
                    risks
                )
            
            # Prepare Eden bundle
            if self.services["eden"]["enabled"]:
                bundles["eden"] = await self._prepare_eden_bundle(
                    tx_data,
                    bribe,
                    risks
                )
            
            # Prepare bloXroute bundle
            if self.services["bloxroute"]["enabled"]:
                bundles["bloxroute"] = await self._prepare_bloxroute_bundle(
                    tx_data,
                    bribe,
                    risks
                )
            
            return bundles
            
        except Exception as e:
            self.logger.error(f"Error preparing bundles: {str(e)}")
            return {}

    async def _prepare_flashbots_bundle(
        self,
        tx_data: Dict,
        bribe: float,
        risks: Dict
    ) -> Dict:
        """Prepare Flashbots bundle."""
        try:
            # Convert bribe to ETH
            bribe_wei = Web3.to_wei(bribe, "ether")
            
            # Create coinbase payment transaction
            coinbase_tx = {
                "to": "0x0000000000000000000000000000000000000000",
                "value": bribe_wei,
                "gas": 21000,
                "maxFeePerGas": tx_data.get("maxFeePerGas", 0),
                "maxPriorityFeePerGas": bribe_wei // 21000
            }
            
            # Create bundle
            bundle = {
                "txs": [
                    self.web3.eth.account.sign_transaction(
                        coinbase_tx,
                        self.settings["flashbots_private_key"]
                    ).rawTransaction,
                    tx_data
                ],
                "block_number": "latest",
                "min_timestamp": 0,
                "max_timestamp": 0,
                "revertingTxHashes": []
            }
            
            return bundle
            
        except Exception as e:
            self.logger.error(f"Error preparing Flashbots bundle: {str(e)}")
            return {}

    async def _prepare_eden_bundle(
        self,
        tx_data: Dict,
        bribe: float,
        risks: Dict
    ) -> Dict:
        """Prepare Eden Network bundle."""
        try:
            # Convert bribe to ETH
            bribe_wei = Web3.to_wei(bribe, "ether")
            
            # Prepare bundle
            bundle = {
                "txs": [tx_data],
                "preferences": {
                    "priority": True,
                    "privacy": True,
                    "bribe": {
                        "amount": str(bribe_wei),
                        "token": "ETH"
                    }
                }
            }
            
            return bundle
            
        except Exception as e:
            self.logger.error(f"Error preparing Eden bundle: {str(e)}")
            return {}

    async def _prepare_bloxroute_bundle(
        self,
        tx_data: Dict,
        bribe: float,
        risks: Dict
    ) -> Dict:
        """Prepare bloXroute bundle."""
        try:
            # Convert bribe to ETH
            bribe_wei = Web3.to_wei(bribe, "ether")
            
            # Prepare bundle
            bundle = {
                "transactions": [tx_data],
                "meta": {
                    "bribe": str(bribe_wei),
                    "priority": "high",
                    "privacy": "private"
                }
            }
            
            return bundle
            
        except Exception as e:
            self.logger.error(f"Error preparing bloXroute bundle: {str(e)}")
            return {}

    async def _send_bundle(
        self,
        service: str,
        bundle: Dict
    ) -> Optional[Dict]:
        """Send bundle to MEV protection service."""
        try:
            if service == "flashbots":
                return await self._send_flashbots_bundle(bundle)
            elif service == "eden":
                return await self._send_eden_bundle(bundle)
            elif service == "bloxroute":
                return await self._send_bloxroute_bundle(bundle)
            else:
                raise ValueError(f"Unknown service: {service}")
            
        except Exception as e:
            self.logger.error(f"Error sending bundle to {service}: {str(e)}")
            return None

    async def _send_flashbots_bundle(self, bundle: Dict) -> Optional[Dict]:
        """Send bundle to Flashbots."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.services['flashbots']['url']}/bundle",
                    headers={
                        "X-Flashbots-Signature": self.flashbots_auth,
                        "Content-Type": "application/json"
                    },
                    json=bundle
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update metrics
                        self.metrics["bundles_sent"] += 1
                        if result.get("included"):
                            self.metrics["bundles_included"] += 1
                        
                        return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending Flashbots bundle: {str(e)}")
            return None

    async def _send_eden_bundle(self, bundle: Dict) -> Optional[Dict]:
        """Send bundle to Eden Network."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.services['eden']['url']}/bundle",
                    headers={
                        "X-Eden-Api-Key": self.eden_api_key,
                        "Content-Type": "application/json"
                    },
                    json=bundle
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update metrics
                        self.metrics["bundles_sent"] += 1
                        if result.get("status") == "included":
                            self.metrics["bundles_included"] += 1
                        
                        return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending Eden bundle: {str(e)}")
            return None

    async def _send_bloxroute_bundle(self, bundle: Dict) -> Optional[Dict]:
        """Send bundle to bloXroute."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.services['bloxroute']['url']}/bundle",
                    headers={
                        "Authorization": self.bloxroute_auth,
                        "Content-Type": "application/json"
                    },
                    json=bundle
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update metrics
                        self.metrics["bundles_sent"] += 1
                        if result.get("status") == "success":
                            self.metrics["bundles_included"] += 1
                        
                        return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending bloXroute bundle: {str(e)}")
            return None

    def _get_best_result(self, results: List[Dict]) -> Optional[Dict]:
        """Get best bundle result."""
        try:
            valid_results = [r for r in results if r is not None]
            
            if not valid_results:
                return None
            
            # Sort by inclusion probability and gas price
            valid_results.sort(
                key=lambda x: (
                    x.get("inclusion_probability", 0),
                    -x.get("effective_gas_price", float("inf"))
                ),
                reverse=True
            )
            
            return valid_results[0]
            
        except Exception as e:
            self.logger.error(f"Error getting best result: {str(e)}")
            return None

    def _prepare_fallback_tx(self, tx_data: Dict) -> Dict:
        """Prepare fallback transaction with higher gas price."""
        try:
            # Increase gas price by 20%
            if "maxFeePerGas" in tx_data:
                tx_data["maxFeePerGas"] = int(tx_data["maxFeePerGas"] * 1.2)
                tx_data["maxPriorityFeePerGas"] = int(
                    tx_data["maxPriorityFeePerGas"] * 1.2
                )
            else:
                tx_data["gasPrice"] = int(tx_data["gasPrice"] * 1.2)
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error preparing fallback tx: {str(e)}")
            return tx_data

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["bundles_included"] /
                max(1, self.metrics["bundles_sent"])
            ),
            "last_update": datetime.utcnow().isoformat()
        }
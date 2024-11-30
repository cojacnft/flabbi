from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import aiohttp
import json

class FailureRecovery:
    """Handle and recover from arbitrage execution failures."""
    def __init__(
        self,
        web3: Web3,
        flash_loan_optimizer,
        slippage_optimizer,
        settings: Dict
    ):
        self.web3 = web3
        self.flash_loan = flash_loan_optimizer
        self.slippage = slippage_optimizer
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Failure history
        self.failure_history: List[Dict] = []
        self.recovery_history: List[Dict] = []
        
        # Active recoveries
        self.active_recoveries: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "total_loss_prevented_usd": 0.0,
            "avg_recovery_time_ms": 0.0
        }
        
        # Initialize recovery strategies
        self._init_strategies()

    def _init_strategies(self):
        """Initialize recovery strategies."""
        self.strategies = {
            "flash_loan_failure": {
                "priority": 1,
                "max_attempts": 3,
                "handler": self._handle_flash_loan_failure
            },
            "slippage_failure": {
                "priority": 2,
                "max_attempts": 2,
                "handler": self._handle_slippage_failure
            },
            "gas_failure": {
                "priority": 3,
                "max_attempts": 3,
                "handler": self._handle_gas_failure
            },
            "contract_failure": {
                "priority": 4,
                "max_attempts": 1,
                "handler": self._handle_contract_failure
            }
        }

    async def handle_failure(
        self,
        tx_hash: str,
        error: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """Handle transaction failure."""
        try:
            start_time = datetime.utcnow()
            
            # Analyze failure
            failure_type = await self._analyze_failure(
                tx_hash,
                error,
                context
            )
            
            if not failure_type:
                return None
            
            # Get recovery strategy
            strategy = self.strategies.get(failure_type)
            if not strategy:
                return None
            
            # Create recovery ID
            recovery_id = f"{tx_hash}_{failure_type}"
            
            # Check if already recovering
            if recovery_id in self.active_recoveries:
                return await self._continue_recovery(recovery_id)
            
            # Initialize recovery
            recovery = await self._init_recovery(
                recovery_id,
                failure_type,
                tx_hash,
                error,
                context
            )
            
            # Execute recovery
            result = await self._execute_recovery(recovery)
            
            # Update metrics
            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            self.metrics["avg_recovery_time_ms"] = (
                self.metrics["avg_recovery_time_ms"] *
                self.metrics["total_failures"] +
                execution_time
            ) / (self.metrics["total_failures"] + 1)
            
            self.metrics["total_failures"] += 1
            
            if result["success"]:
                self.metrics["successful_recoveries"] += 1
                if "prevented_loss" in result:
                    self.metrics["total_loss_prevented_usd"] += result["prevented_loss"]
            else:
                self.metrics["failed_recoveries"] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling failure: {str(e)}")
            return None

    async def _analyze_failure(
        self,
        tx_hash: str,
        error: Dict,
        context: Dict
    ) -> Optional[str]:
        """Analyze transaction failure to determine type."""
        try:
            # Get transaction receipt
            receipt = await self.web3.eth.get_transaction_receipt(tx_hash)
            
            if not receipt:
                return None
            
            # Check for revert
            if receipt["status"] == 0:
                # Analyze revert reason
                reason = await self._get_revert_reason(
                    tx_hash,
                    receipt
                )
                
                if "insufficient balance" in reason.lower():
                    return "flash_loan_failure"
                elif "slippage" in reason.lower():
                    return "slippage_failure"
                elif "gas" in reason.lower():
                    return "gas_failure"
                else:
                    return "contract_failure"
            
            # Check for other failures
            if "error" in error:
                if "gas" in error["error"].lower():
                    return "gas_failure"
                elif "execution reverted" in error["error"].lower():
                    return "contract_failure"
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing failure: {str(e)}")
            return None

    async def _get_revert_reason(
        self,
        tx_hash: str,
        receipt: Dict
    ) -> str:
        """Get transaction revert reason."""
        try:
            # Try to get error message from receipt
            if "revertReason" in receipt:
                return receipt["revertReason"]
            
            # Try to decode error from trace
            trace = await self.web3.eth.get_transaction(tx_hash)
            if "input" in trace:
                # Decode input data
                return self._decode_error(trace["input"])
            
            return "Unknown error"
            
        except Exception as e:
            self.logger.error(f"Error getting revert reason: {str(e)}")
            return "Unknown error"

    async def _init_recovery(
        self,
        recovery_id: str,
        failure_type: str,
        tx_hash: str,
        error: Dict,
        context: Dict
    ) -> Dict:
        """Initialize recovery process."""
        try:
            recovery = {
                "id": recovery_id,
                "type": failure_type,
                "tx_hash": tx_hash,
                "error": error,
                "context": context,
                "attempts": 0,
                "max_attempts": self.strategies[failure_type]["max_attempts"],
                "status": "initialized",
                "start_time": datetime.utcnow(),
                "steps": []
            }
            
            # Add to active recoveries
            self.active_recoveries[recovery_id] = recovery
            
            # Add to history
            self.failure_history.append({
                "recovery_id": recovery_id,
                "type": failure_type,
                "tx_hash": tx_hash,
                "timestamp": datetime.utcnow()
            })
            
            return recovery
            
        except Exception as e:
            self.logger.error(f"Error initializing recovery: {str(e)}")
            return {}

    async def _execute_recovery(self, recovery: Dict) -> Dict:
        """Execute recovery strategy."""
        try:
            # Get strategy handler
            handler = self.strategies[recovery["type"]]["handler"]
            
            # Execute recovery steps
            while recovery["attempts"] < recovery["max_attempts"]:
                # Increment attempts
                recovery["attempts"] += 1
                
                # Execute handler
                result = await handler(recovery)
                
                if result["success"]:
                    # Recovery successful
                    await self._finalize_recovery(recovery, result)
                    return result
                
                # Wait before retry
                await asyncio.sleep(1)
            
            # All attempts failed
            await self._finalize_recovery(
                recovery,
                {"success": False, "error": "Max attempts reached"}
            )
            
            return {"success": False, "error": "Max attempts reached"}
            
        except Exception as e:
            self.logger.error(f"Error executing recovery: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_flash_loan_failure(self, recovery: Dict) -> Dict:
        """Handle flash loan failure."""
        try:
            context = recovery["context"]
            
            # Try different flash loan provider
            new_config = await self.flash_loan.get_optimal_flashloan(
                context["tokens"],
                context["amounts"],
                context["min_profit"]
            )
            
            if not new_config:
                return {
                    "success": False,
                    "error": "No alternative flash loan available"
                }
            
            # Prepare new transaction
            tx_data = await self.flash_loan.prepare_flashloan(
                new_config,
                context["target_contract"],
                context["callback_data"]
            )
            
            # Execute transaction
            tx_hash = await self.flash_loan.execute_flashloan(
                tx_data,
                context["private_key"]
            )
            
            if not tx_hash:
                return {
                    "success": False,
                    "error": "Flash loan execution failed"
                }
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "prevented_loss": context["expected_profit"]
            }
            
        except Exception as e:
            self.logger.error(f"Error handling flash loan failure: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_slippage_failure(self, recovery: Dict) -> Dict:
        """Handle slippage failure."""
        try:
            context = recovery["context"]
            
            # Optimize slippage parameters
            new_slippage = await self.slippage.optimize_slippage(
                context["path"],
                context["amounts"],
                context["min_profit"]
            )
            
            if not new_slippage:
                return {
                    "success": False,
                    "error": "Failed to optimize slippage"
                }
            
            # Update transaction data
            tx_data = self._update_slippage_params(
                context["tx_data"],
                new_slippage["slippage"]
            )
            
            # Execute transaction
            tx_hash = await self._send_transaction(
                tx_data,
                context["private_key"]
            )
            
            if not tx_hash:
                return {
                    "success": False,
                    "error": "Transaction execution failed"
                }
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "prevented_loss": context["expected_profit"]
            }
            
        except Exception as e:
            self.logger.error(f"Error handling slippage failure: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_gas_failure(self, recovery: Dict) -> Dict:
        """Handle gas-related failure."""
        try:
            context = recovery["context"]
            
            # Increase gas limit and price
            tx_data = context["tx_data"].copy()
            tx_data["gas"] = int(tx_data["gas"] * 1.2)  # 20% increase
            
            if "maxFeePerGas" in tx_data:
                tx_data["maxFeePerGas"] = int(tx_data["maxFeePerGas"] * 1.2)
                tx_data["maxPriorityFeePerGas"] = int(
                    tx_data["maxPriorityFeePerGas"] * 1.2
                )
            else:
                tx_data["gasPrice"] = int(tx_data["gasPrice"] * 1.2)
            
            # Execute transaction
            tx_hash = await self._send_transaction(
                tx_data,
                context["private_key"]
            )
            
            if not tx_hash:
                return {
                    "success": False,
                    "error": "Transaction execution failed"
                }
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "prevented_loss": context["expected_profit"]
            }
            
        except Exception as e:
            self.logger.error(f"Error handling gas failure: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _handle_contract_failure(self, recovery: Dict) -> Dict:
        """Handle smart contract failure."""
        try:
            context = recovery["context"]
            
            # Analyze contract error
            error_analysis = await self._analyze_contract_error(
                recovery["tx_hash"],
                recovery["error"]
            )
            
            if not error_analysis:
                return {
                    "success": False,
                    "error": "Unable to analyze contract error"
                }
            
            # Try to fix contract call
            fixed_data = await self._fix_contract_call(
                context["tx_data"],
                error_analysis
            )
            
            if not fixed_data:
                return {
                    "success": False,
                    "error": "Unable to fix contract call"
                }
            
            # Execute transaction
            tx_hash = await self._send_transaction(
                fixed_data,
                context["private_key"]
            )
            
            if not tx_hash:
                return {
                    "success": False,
                    "error": "Transaction execution failed"
                }
            
            return {
                "success": True,
                "tx_hash": tx_hash,
                "prevented_loss": context["expected_profit"]
            }
            
        except Exception as e:
            self.logger.error(f"Error handling contract failure: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _continue_recovery(self, recovery_id: str) -> Optional[Dict]:
        """Continue existing recovery process."""
        try:
            recovery = self.active_recoveries.get(recovery_id)
            if not recovery:
                return None
            
            # Check if recovery is still valid
            if (
                datetime.utcnow() - recovery["start_time"] >
                timedelta(minutes=5)
            ):
                # Recovery expired
                await self._finalize_recovery(
                    recovery,
                    {"success": False, "error": "Recovery expired"}
                )
                return None
            
            # Continue recovery
            return await self._execute_recovery(recovery)
            
        except Exception as e:
            self.logger.error(f"Error continuing recovery: {str(e)}")
            return None

    async def _finalize_recovery(
        self,
        recovery: Dict,
        result: Dict
    ):
        """Finalize recovery process."""
        try:
            # Remove from active recoveries
            if recovery["id"] in self.active_recoveries:
                del self.active_recoveries[recovery["id"]]
            
            # Add to recovery history
            self.recovery_history.append({
                "recovery_id": recovery["id"],
                "type": recovery["type"],
                "attempts": recovery["attempts"],
                "success": result["success"],
                "prevented_loss": result.get("prevented_loss", 0),
                "timestamp": datetime.utcnow()
            })
            
            # Cleanup old history
            self._cleanup_history()
            
        except Exception as e:
            self.logger.error(f"Error finalizing recovery: {str(e)}")

    def _cleanup_history(self):
        """Clean up old history entries."""
        try:
            # Keep last 1000 entries
            if len(self.failure_history) > 1000:
                self.failure_history = self.failure_history[-1000:]
            
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up history: {str(e)}")

    async def _send_transaction(
        self,
        tx_data: Dict,
        private_key: str
    ) -> Optional[str]:
        """Send transaction to network."""
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
            
            return tx_hash.hex()
            
        except Exception as e:
            self.logger.error(f"Error sending transaction: {str(e)}")
            return None

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "recovery_rate": (
                self.metrics["successful_recoveries"] /
                max(1, self.metrics["total_failures"])
            ),
            "active_recoveries": len(self.active_recoveries),
            "last_update": datetime.utcnow().isoformat()
        }
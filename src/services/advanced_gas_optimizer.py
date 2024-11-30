from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import aiohttp
import json

class AdvancedGasOptimizer:
    """Advanced gas optimization for arbitrage transactions."""
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
        
        # Gas price history
        self.gas_history: List[Dict] = []
        self.history_window = 200
        
        # Block history
        self.block_history: List[Dict] = []
        self.block_window = 50
        
        # Gas estimations
        self.operation_gas = {
            "flash_loan": 150000,
            "token_swap": 120000,
            "token_transfer": 65000,
            "contract_deployment": 2000000
        }
        
        # Performance metrics
        self.metrics = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "gas_savings_gwei": 0.0,
            "avg_gas_used": 0.0,
            "avg_gas_price_gwei": 0.0
        }
        
        # Initialize ML model
        self._init_model()

    def _init_model(self):
        """Initialize gas prediction model."""
        try:
            # Load model if exists
            model_path = f"models/gas_predictor_{self.chain_id}.joblib"
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(f"{model_path}.scaler")
            except:
                # Train new model
                self._train_initial_model()
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")

    def _train_initial_model(self):
        """Train initial gas prediction model."""
        try:
            # Create simple initial training data
            X = np.random.rand(1000, 5)  # 5 features
            y = np.mean(X, axis=1) * 100  # Simple gas price rule
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            self.logger.error(f"Error training initial model: {str(e)}")

    async def optimize_transaction(
        self,
        tx_data: Dict,
        priority: str = "normal",
        max_wait_time: int = 3  # blocks
    ) -> Dict:
        """Optimize transaction gas settings."""
        try:
            # Get current network state
            network_state = await self._get_network_state()
            
            # Calculate base gas settings
            base_gas = await self._calculate_base_gas(
                tx_data,
                network_state
            )
            
            # Apply priority adjustments
            gas_settings = self._adjust_for_priority(
                base_gas,
                priority,
                network_state
            )
            
            # Predict optimal values
            predictions = await self._predict_gas_values(
                gas_settings,
                network_state,
                max_wait_time
            )
            
            # Apply predictions
            optimized_tx = self._apply_gas_settings(
                tx_data,
                predictions
            )
            
            # Update metrics
            self.metrics["total_optimizations"] += 1
            self.metrics["avg_gas_price_gwei"] = (
                (self.metrics["avg_gas_price_gwei"] *
                 (self.metrics["total_optimizations"] - 1) +
                 predictions["gas_price"]) /
                self.metrics["total_optimizations"]
            )
            
            return optimized_tx
            
        except Exception as e:
            self.logger.error(f"Error optimizing transaction: {str(e)}")
            return tx_data

    async def _get_network_state(self) -> Dict:
        """Get current network state."""
        try:
            # Get latest block
            latest_block = await self.web3.eth.get_block("latest")
            
            # Get base fee
            base_fee = latest_block.get(
                "baseFeePerGas",
                await self.web3.eth.gas_price
            )
            
            # Get network load
            network_load = latest_block["gasUsed"] / latest_block["gasLimit"]
            
            # Get pending transaction count
            pending_count = len(
                await self.web3.eth.get_block("pending")["transactions"]
            )
            
            # Get recent gas prices
            recent_gas = await self._get_recent_gas_prices()
            
            return {
                "base_fee": base_fee,
                "network_load": network_load,
                "pending_count": pending_count,
                "recent_gas": recent_gas,
                "block_number": latest_block["number"],
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting network state: {str(e)}")
            return {}

    async def _get_recent_gas_prices(self) -> List[int]:
        """Get recent gas prices from history."""
        try:
            # Get last 10 blocks
            prices = []
            current_block = await self.web3.eth.block_number
            
            for block_number in range(
                current_block - 9,
                current_block + 1
            ):
                block = await self.web3.eth.get_block(block_number)
                if "baseFeePerGas" in block:
                    prices.append(block["baseFeePerGas"])
                else:
                    # For chains without EIP-1559
                    txs = await self.web3.eth.get_block(
                        block_number,
                        True
                    )["transactions"]
                    if txs:
                        gas_prices = [
                            tx["gasPrice"]
                            for tx in txs
                            if "gasPrice" in tx
                        ]
                        if gas_prices:
                            prices.append(np.median(gas_prices))
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error getting recent gas prices: {str(e)}")
            return []

    async def _calculate_base_gas(
        self,
        tx_data: Dict,
        network_state: Dict
    ) -> Dict:
        """Calculate base gas settings."""
        try:
            # Estimate gas limit
            gas_limit = await self._estimate_gas_limit(tx_data)
            
            # Get base fee
            base_fee = network_state["base_fee"]
            
            # Calculate priority fee
            priority_fee = self._calculate_priority_fee(
                network_state["recent_gas"]
            )
            
            return {
                "gas_limit": gas_limit,
                "base_fee": base_fee,
                "priority_fee": priority_fee
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating base gas: {str(e)}")
            return {}

    async def _estimate_gas_limit(self, tx_data: Dict) -> int:
        """Estimate gas limit for transaction."""
        try:
            # Get operation type
            op_type = self._get_operation_type(tx_data)
            
            # Get base estimate
            base_estimate = self.operation_gas.get(
                op_type,
                21000  # Default gas limit
            )
            
            # Get dynamic estimate
            dynamic_estimate = await self.web3.eth.estimate_gas(tx_data)
            
            # Use maximum of base and dynamic estimates
            gas_limit = max(base_estimate, dynamic_estimate)
            
            # Add safety margin (10%)
            return int(gas_limit * 1.1)
            
        except Exception as e:
            self.logger.error(f"Error estimating gas limit: {str(e)}")
            return 500000  # Safe fallback

    def _get_operation_type(self, tx_data: Dict) -> str:
        """Determine operation type from transaction data."""
        try:
            if not tx_data.get("data"):
                return "token_transfer"
            
            # Check for contract deployment
            if not tx_data.get("to"):
                return "contract_deployment"
            
            # Check for flash loan
            if self._is_flash_loan(tx_data):
                return "flash_loan"
            
            # Check for token swap
            if self._is_token_swap(tx_data):
                return "token_swap"
            
            return "token_transfer"
            
        except Exception as e:
            self.logger.error(f"Error getting operation type: {str(e)}")
            return "token_transfer"

    def _calculate_priority_fee(self, recent_gas: List[int]) -> int:
        """Calculate optimal priority fee."""
        try:
            if not recent_gas:
                return Web3.to_wei(1, "gwei")  # Default 1 gwei
            
            # Calculate percentiles
            percentiles = np.percentile(recent_gas, [25, 50, 75])
            
            # Use median as base priority fee
            base_priority = percentiles[1] - min(recent_gas)
            
            # Adjust based on recent volatility
            volatility = (percentiles[2] - percentiles[0]) / percentiles[1]
            
            if volatility > 0.2:  # High volatility
                return int(base_priority * 1.2)  # 20% increase
            elif volatility < 0.05:  # Low volatility
                return int(base_priority * 0.9)  # 10% decrease
            
            return int(base_priority)
            
        except Exception as e:
            self.logger.error(f"Error calculating priority fee: {str(e)}")
            return Web3.to_wei(1, "gwei")

    def _adjust_for_priority(
        self,
        gas_settings: Dict,
        priority: str,
        network_state: Dict
    ) -> Dict:
        """Adjust gas settings based on priority."""
        try:
            multipliers = {
                "low": 0.8,
                "normal": 1.0,
                "high": 1.3,
                "urgent": 1.5
            }
            
            multiplier = multipliers.get(priority, 1.0)
            
            # Adjust base fee
            gas_settings["base_fee"] = int(
                gas_settings["base_fee"] * multiplier
            )
            
            # Adjust priority fee
            gas_settings["priority_fee"] = int(
                gas_settings["priority_fee"] * multiplier
            )
            
            return gas_settings
            
        except Exception as e:
            self.logger.error(f"Error adjusting for priority: {str(e)}")
            return gas_settings

    async def _predict_gas_values(
        self,
        gas_settings: Dict,
        network_state: Dict,
        max_wait_time: int
    ) -> Dict:
        """Predict optimal gas values."""
        try:
            # Extract features
            features = self._extract_features(
                gas_settings,
                network_state
            )
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate gas price
            gas_price = max(
                gas_settings["base_fee"] + gas_settings["priority_fee"],
                int(prediction)
            )
            
            # Validate against wait time
            while not await self._validate_wait_time(
                gas_price,
                max_wait_time,
                network_state
            ):
                gas_price = int(gas_price * 1.1)  # Increase by 10%
            
            return {
                "gas_price": gas_price,
                "gas_limit": gas_settings["gas_limit"],
                "max_fee_per_gas": int(gas_price * 1.25),  # 25% buffer
                "max_priority_fee_per_gas": gas_settings["priority_fee"]
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting gas values: {str(e)}")
            return gas_settings

    def _extract_features(
        self,
        gas_settings: Dict,
        network_state: Dict
    ) -> np.ndarray:
        """Extract features for prediction."""
        try:
            features = [
                network_state["network_load"],
                network_state["pending_count"] / 1000,  # Normalize
                np.mean(network_state["recent_gas"]) / 1e9,  # Convert to gwei
                np.std(network_state["recent_gas"]) / 1e9,
                gas_settings["gas_limit"] / 1e6  # Normalize
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 5))

    async def _validate_wait_time(
        self,
        gas_price: int,
        max_wait_time: int,
        network_state: Dict
    ) -> bool:
        """Validate if gas price will result in acceptable wait time."""
        try:
            # Get recent blocks
            recent_blocks = self.block_history[-10:]
            
            if not recent_blocks:
                return True
            
            # Calculate acceptance probability
            accepted_txs = 0
            total_txs = 0
            
            for block in recent_blocks:
                for tx in block["transactions"]:
                    if tx.get("gasPrice", 0) <= gas_price:
                        accepted_txs += 1
                    total_txs += 1
            
            if total_txs == 0:
                return True
            
            acceptance_prob = accepted_txs / total_txs
            
            # Calculate expected wait time
            expected_blocks = 1 / acceptance_prob if acceptance_prob > 0 else float("inf")
            
            return expected_blocks <= max_wait_time
            
        except Exception as e:
            self.logger.error(f"Error validating wait time: {str(e)}")
            return True

    def _apply_gas_settings(self, tx_data: Dict, predictions: Dict) -> Dict:
        """Apply optimized gas settings to transaction."""
        try:
            # Check if EIP-1559
            if "maxFeePerGas" in tx_data:
                tx_data["maxFeePerGas"] = predictions["max_fee_per_gas"]
                tx_data["maxPriorityFeePerGas"] = predictions["max_priority_fee_per_gas"]
            else:
                tx_data["gasPrice"] = predictions["gas_price"]
            
            tx_data["gas"] = predictions["gas_limit"]
            
            return tx_data
            
        except Exception as e:
            self.logger.error(f"Error applying gas settings: {str(e)}")
            return tx_data

    async def update_history(self, tx_hash: str, success: bool):
        """Update gas price history with transaction result."""
        try:
            # Get transaction
            tx = await self.web3.eth.get_transaction(tx_hash)
            receipt = await self.web3.eth.get_transaction_receipt(tx_hash)
            
            if not tx or not receipt:
                return
            
            # Add to history
            self.gas_history.append({
                "gas_price": tx.get("gasPrice", 0),
                "gas_used": receipt["gasUsed"],
                "success": success,
                "block_number": receipt["blockNumber"],
                "timestamp": datetime.utcnow()
            })
            
            # Keep fixed window size
            if len(self.gas_history) > self.history_window:
                self.gas_history = self.gas_history[-self.history_window:]
            
            # Update metrics
            if success:
                self.metrics["successful_optimizations"] += 1
                self.metrics["avg_gas_used"] = (
                    (self.metrics["avg_gas_used"] *
                     (self.metrics["successful_optimizations"] - 1) +
                     receipt["gasUsed"]) /
                    self.metrics["successful_optimizations"]
                )
            
            # Retrain model periodically
            if len(self.gas_history) % 100 == 0:
                await self._retrain_model()
            
        except Exception as e:
            self.logger.error(f"Error updating history: {str(e)}")

    async def _retrain_model(self):
        """Retrain gas prediction model."""
        try:
            if len(self.gas_history) < 100:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for i in range(len(self.gas_history) - 10):
                # Get historical window
                window = self.gas_history[i:i+10]
                
                # Extract features
                features = [
                    np.mean([h["gas_price"] for h in window]),
                    np.std([h["gas_price"] for h in window]),
                    np.mean([h["gas_used"] for h in window]),
                    sum(1 for h in window if h["success"]) / len(window),
                    window[-1]["gas_price"] / window[0]["gas_price"]
                ]
                
                X.append(features)
                y.append(window[-1]["gas_price"])
            
            # Convert to arrays
            X = np.array(X)
            y = np.array(y)
            
            # Update scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train new model
            new_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            new_model.fit(X_scaled, y)
            
            # Update if better
            old_score = self.model.score(X_scaled, y)
            new_score = new_model.score(X_scaled, y)
            
            if new_score > old_score:
                self.model = new_model
                self.logger.info(f"Model updated with score: {new_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_optimizations"] /
                max(1, self.metrics["total_optimizations"])
            ),
            "last_update": datetime.utcnow().isoformat()
        }
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
class ExecutionStrategy:
    """Flash loan execution strategy."""
    name: str
    priority: int  # 1 (highest) to 3 (lowest)
    min_profit: float
    max_gas_price: int
    bundle_type: str  # 'flashbots', 'eden', 'standard'
    mev_protection: bool
    backrun_protection: bool
    timeout: int

@dataclass
class ExecutionPlan:
    """Detailed execution plan."""
    strategy: ExecutionStrategy
    transactions: List[Dict]
    estimated_profit: float
    estimated_gas: int
    priority_fee: int
    bundle_data: Optional[Dict]

class ExecutionStrategyOptimizer:
    """Advanced execution strategy optimizer."""
    def __init__(
        self,
        web3: Web3,
        flash_loan_executor,
        market_data_aggregator,
        settings: Dict
    ):
        self.web3 = web3
        self.flash_loan = flash_loan_executor
        self.market_data = market_data_aggregator
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Execution strategies
        self.strategies = {
            "aggressive": ExecutionStrategy(
                name="aggressive",
                priority=1,
                min_profit=50.0,  # Lower profit threshold
                max_gas_price=500,  # Higher gas price tolerance
                bundle_type="flashbots",
                mev_protection=True,
                backrun_protection=True,
                timeout=15
            ),
            "balanced": ExecutionStrategy(
                name="balanced",
                priority=2,
                min_profit=100.0,
                max_gas_price=200,
                bundle_type="eden",
                mev_protection=True,
                backrun_protection=False,
                timeout=30
            ),
            "conservative": ExecutionStrategy(
                name="conservative",
                priority=3,
                min_profit=200.0,  # Higher profit threshold
                max_gas_price=100,  # Lower gas price tolerance
                bundle_type="standard",
                mev_protection=False,
                backrun_protection=False,
                timeout=45
            )
        }
        
        # MEV protection settings
        self.mev_settings = {
            "min_bribe": 0.1,  # 10% of profit
            "max_bribe": 0.3,  # 30% of profit
            "sandwich_threshold": 0.02,  # 2% price impact
            "frontrun_threshold": 0.01,  # 1% price impact
            "backrun_threshold": 0.01  # 1% price impact
        }
        
        # Execution history
        self.execution_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "mev_attacks_prevented": 0,
            "total_profit": 0.0,
            "total_gas_cost": 0.0,
            "avg_execution_time": 0.0
        }
        
        # Initialize bundle builders
        self._init_bundle_builders()

    def _init_bundle_builders(self):
        """Initialize bundle builders for different services."""
        self.bundle_builders = {
            "flashbots": self._build_flashbots_bundle,
            "eden": self._build_eden_bundle,
            "standard": self._build_standard_bundle
        }

    async def optimize_execution(
        self,
        opportunity: Dict,
        context: Dict
    ) -> Optional[ExecutionPlan]:
        """Optimize execution strategy for flash loan arbitrage."""
        try:
            # Get current market conditions
            market_conditions = await self._get_market_conditions()
            
            # Select best strategy
            strategy = await self._select_strategy(
                opportunity,
                market_conditions,
                context
            )
            
            if not strategy:
                return None
            
            # Check for MEV risks
            mev_risks = await self._analyze_mev_risks(
                opportunity,
                strategy
            )
            
            # Prepare execution plan
            plan = await self._prepare_execution_plan(
                opportunity,
                strategy,
                mev_risks
            )
            
            if not plan:
                return None
            
            # Validate plan
            if not await self._validate_execution_plan(plan, context):
                return None
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error optimizing execution: {str(e)}")
            return None

    async def _get_market_conditions(self) -> Dict:
        """Get current market conditions."""
        try:
            # Get mempool state
            mempool = await self._analyze_mempool()
            
            # Get network state
            network = await self._get_network_state()
            
            # Get MEV activity
            mev_activity = await self._analyze_mev_activity()
            
            return {
                "mempool": mempool,
                "network": network,
                "mev_activity": mev_activity,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market conditions: {str(e)}")
            return {}

    async def _analyze_mempool(self) -> Dict:
        """Analyze mempool state."""
        try:
            # Get pending transactions
            pending = await self.web3.eth.get_block('pending')
            
            # Analyze gas prices
            gas_prices = [
                tx.get('gasPrice', 0)
                for tx in pending['transactions']
            ]
            
            # Analyze transaction types
            swap_txs = [
                tx for tx in pending['transactions']
                if self._is_swap_transaction(tx)
            ]
            
            return {
                "pending_count": len(pending['transactions']),
                "avg_gas_price": np.mean(gas_prices) if gas_prices else 0,
                "swap_ratio": len(swap_txs) / max(1, len(pending['transactions'])),
                "congestion_level": self._calculate_congestion(pending)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing mempool: {str(e)}")
            return {}

    def _is_swap_transaction(self, tx: Dict) -> bool:
        """Check if transaction is a swap."""
        try:
            # Common DEX function signatures
            swap_signatures = {
                "0x38ed1739",  # swapExactTokensForTokens
                "0x7c025200",  # swap
                "0x791ac947"   # swapExactTokensForETH
            }
            
            if "input" in tx and len(tx["input"]) >= 10:
                return tx["input"][:10] in swap_signatures
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking swap transaction: {str(e)}")
            return False

    def _calculate_congestion(self, block: Dict) -> float:
        """Calculate network congestion level."""
        try:
            gas_used = sum(
                tx.get('gas', 0)
                for tx in block['transactions']
            )
            gas_limit = block.get('gasLimit', 15000000)
            
            return min(1.0, gas_used / gas_limit)
            
        except Exception as e:
            self.logger.error(f"Error calculating congestion: {str(e)}")
            return 0.5

    async def _get_network_state(self) -> Dict:
        """Get current network state."""
        try:
            # Get latest block
            latest_block = await self.web3.eth.get_block('latest')
            
            # Get base fee
            base_fee = latest_block.get(
                'baseFeePerGas',
                await self.web3.eth.gas_price
            )
            
            return {
                "block_number": latest_block["number"],
                "base_fee": base_fee,
                "timestamp": latest_block["timestamp"],
                "gas_used_ratio": latest_block["gasUsed"] / latest_block["gasLimit"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting network state: {str(e)}")
            return {}

    async def _analyze_mev_activity(self) -> Dict:
        """Analyze MEV activity."""
        try:
            # Get recent blocks
            blocks = []
            current_block = await self.web3.eth.block_number
            
            for block_number in range(current_block - 10, current_block + 1):
                block = await self.web3.eth.get_block(block_number, True)
                blocks.append(block)
            
            # Analyze sandwich attacks
            sandwiches = self._detect_sandwiches(blocks)
            
            # Analyze frontrunning
            frontruns = self._detect_frontruns(blocks)
            
            return {
                "sandwich_count": len(sandwiches),
                "frontrun_count": len(frontruns),
                "mev_ratio": (len(sandwiches) + len(frontruns)) / len(blocks),
                "recent_blocks": len(blocks)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing MEV activity: {str(e)}")
            return {}

    def _detect_sandwiches(self, blocks: List[Dict]) -> List[Dict]:
        """Detect sandwich attacks in blocks."""
        try:
            sandwiches = []
            
            for block in blocks:
                txs = block["transactions"]
                for i in range(len(txs) - 2):
                    if (
                        self._is_swap_transaction(txs[i]) and
                        self._is_swap_transaction(txs[i+1]) and
                        self._is_swap_transaction(txs[i+2]) and
                        txs[i]["from"] == txs[i+2]["from"]
                    ):
                        sandwiches.append({
                            "block": block["number"],
                            "transactions": [txs[i], txs[i+1], txs[i+2]]
                        })
            
            return sandwiches
            
        except Exception as e:
            self.logger.error(f"Error detecting sandwiches: {str(e)}")
            return []

    def _detect_frontruns(self, blocks: List[Dict]) -> List[Dict]:
        """Detect frontrunning in blocks."""
        try:
            frontruns = []
            
            for block in blocks:
                txs = block["transactions"]
                for i in range(len(txs) - 1):
                    if (
                        self._is_swap_transaction(txs[i]) and
                        self._is_swap_transaction(txs[i+1]) and
                        self._are_similar_swaps(txs[i], txs[i+1])
                    ):
                        frontruns.append({
                            "block": block["number"],
                            "transactions": [txs[i], txs[i+1]]
                        })
            
            return frontruns
            
        except Exception as e:
            self.logger.error(f"Error detecting frontruns: {str(e)}")
            return []

    def _are_similar_swaps(self, tx1: Dict, tx2: Dict) -> bool:
        """Check if two swaps are similar (same tokens)."""
        try:
            # Extract token addresses from input data
            tokens1 = self._extract_swap_tokens(tx1)
            tokens2 = self._extract_swap_tokens(tx2)
            
            return bool(set(tokens1) & set(tokens2))
            
        except Exception as e:
            self.logger.error(f"Error comparing swaps: {str(e)}")
            return False

    def _extract_swap_tokens(self, tx: Dict) -> List[str]:
        """Extract token addresses from swap transaction."""
        try:
            # Decode input data
            if len(tx["input"]) >= 138:  # Minimum length for token swap
                # Extract token addresses from common positions
                tokens = []
                data = tx["input"]
                
                # Check common positions for token addresses
                positions = [34, 98]  # Common positions in swap data
                for pos in positions:
                    if len(data) >= pos + 40:
                        token = "0x" + data[pos:pos+40]
                        if Web3.is_address(token):
                            tokens.append(token.lower())
                
                return tokens
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting swap tokens: {str(e)}")
            return []

    async def _select_strategy(
        self,
        opportunity: Dict,
        market_conditions: Dict,
        context: Dict
    ) -> Optional[ExecutionStrategy]:
        """Select best execution strategy."""
        try:
            valid_strategies = []
            
            for strategy in self.strategies.values():
                # Check profit threshold
                if opportunity["expected_profit"] < strategy.min_profit:
                    continue
                
                # Check gas price
                if market_conditions["network"]["base_fee"] > strategy.max_gas_price:
                    continue
                
                # Check MEV conditions
                if strategy.mev_protection and market_conditions["mev_activity"]["mev_ratio"] > 0.3:
                    continue
                
                valid_strategies.append(strategy)
            
            if not valid_strategies:
                return None
            
            # Sort by priority
            valid_strategies.sort(key=lambda x: x.priority)
            
            return valid_strategies[0]
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {str(e)}")
            return None

    async def _analyze_mev_risks(
        self,
        opportunity: Dict,
        strategy: ExecutionStrategy
    ) -> Dict:
        """Analyze MEV risks for opportunity."""
        try:
            risks = {
                "sandwich_risk": 0.0,
                "frontrun_risk": 0.0,
                "backrun_risk": 0.0,
                "total_risk": 0.0
            }
            
            if not strategy.mev_protection:
                return risks
            
            # Calculate sandwich risk
            risks["sandwich_risk"] = await self._calculate_sandwich_risk(
                opportunity
            )
            
            # Calculate frontrun risk
            risks["frontrun_risk"] = await self._calculate_frontrun_risk(
                opportunity
            )
            
            # Calculate backrun risk
            risks["backrun_risk"] = await self._calculate_backrun_risk(
                opportunity
            )
            
            # Calculate total risk
            risks["total_risk"] = (
                risks["sandwich_risk"] * 0.4 +
                risks["frontrun_risk"] * 0.4 +
                risks["backrun_risk"] * 0.2
            )
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error analyzing MEV risks: {str(e)}")
            return {"total_risk": 1.0}

    async def _calculate_sandwich_risk(
        self,
        opportunity: Dict
    ) -> float:
        """Calculate risk of sandwich attack."""
        try:
            # Get pool states
            pool_states = []
            for step in opportunity["path"]:
                state = await self.market_data.get_pool_state(step["pool"])
                if state:
                    pool_states.append(state)
            
            if not pool_states:
                return 1.0
            
            # Calculate price impact
            price_impacts = []
            for state in pool_states:
                impact = opportunity["amount_in"] / state["reserve_in"]
                price_impacts.append(impact)
            
            # Higher impact = higher risk
            max_impact = max(price_impacts)
            if max_impact > self.mev_settings["sandwich_threshold"]:
                return 1.0
            
            return max_impact / self.mev_settings["sandwich_threshold"]
            
        except Exception as e:
            self.logger.error(f"Error calculating sandwich risk: {str(e)}")
            return 1.0

    async def _calculate_frontrun_risk(
        self,
        opportunity: Dict
    ) -> float:
        """Calculate risk of frontrunning."""
        try:
            # Check gas price
            network_gas = await self.web3.eth.gas_price
            tx_gas = opportunity.get("gas_price", network_gas)
            
            if tx_gas < network_gas:
                return 1.0
            
            # Check pool activity
            activity_risk = 0.0
            for step in opportunity["path"]:
                pool_activity = await self.market_data.get_pool_activity(
                    step["pool"]
                )
                if pool_activity > self.mev_settings["frontrun_threshold"]:
                    activity_risk = max(
                        activity_risk,
                        pool_activity / self.mev_settings["frontrun_threshold"]
                    )
            
            return activity_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating frontrun risk: {str(e)}")
            return 1.0

    async def _calculate_backrun_risk(
        self,
        opportunity: Dict
    ) -> float:
        """Calculate risk of backrunning."""
        try:
            # Check pending transactions
            pending_risk = 0.0
            for step in opportunity["path"]:
                pending_swaps = await self.market_data.get_pending_swaps(
                    step["pool"]
                )
                if pending_swaps > self.mev_settings["backrun_threshold"]:
                    pending_risk = max(
                        pending_risk,
                        pending_swaps / self.mev_settings["backrun_threshold"]
                    )
            
            return pending_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating backrun risk: {str(e)}")
            return 1.0

    async def _prepare_execution_plan(
        self,
        opportunity: Dict,
        strategy: ExecutionStrategy,
        mev_risks: Dict
    ) -> Optional[ExecutionPlan]:
        """Prepare detailed execution plan."""
        try:
            # Prepare transactions
            transactions = await self._prepare_transactions(
                opportunity,
                strategy
            )
            
            if not transactions:
                return None
            
            # Calculate gas costs
            estimated_gas = sum(
                tx.get("gas", 0)
                for tx in transactions
            )
            
            # Calculate priority fee
            priority_fee = await self._calculate_priority_fee(
                strategy,
                mev_risks
            )
            
            # Prepare bundle if needed
            bundle_data = None
            if strategy.bundle_type != "standard":
                bundle_data = await self._prepare_bundle(
                    transactions,
                    strategy,
                    opportunity["expected_profit"]
                )
            
            return ExecutionPlan(
                strategy=strategy,
                transactions=transactions,
                estimated_profit=opportunity["expected_profit"],
                estimated_gas=estimated_gas,
                priority_fee=priority_fee,
                bundle_data=bundle_data
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing execution plan: {str(e)}")
            return None

    async def _prepare_transactions(
        self,
        opportunity: Dict,
        strategy: ExecutionStrategy
    ) -> List[Dict]:
        """Prepare transaction sequence."""
        try:
            transactions = []
            
            # Flash loan transaction
            flash_loan_tx = await self.flash_loan._prepare_transaction(
                opportunity["flash_loan_config"]
            )
            if not flash_loan_tx:
                return []
            
            transactions.append(flash_loan_tx)
            
            # Add MEV protection if needed
            if strategy.mev_protection:
                protection_tx = await self._prepare_protection_tx(
                    opportunity,
                    strategy
                )
                if protection_tx:
                    transactions.append(protection_tx)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error preparing transactions: {str(e)}")
            return []

    async def _prepare_protection_tx(
        self,
        opportunity: Dict,
        strategy: ExecutionStrategy
    ) -> Optional[Dict]:
        """Prepare MEV protection transaction."""
        try:
            if strategy.bundle_type == "flashbots":
                return await self._prepare_flashbots_protection(opportunity)
            elif strategy.bundle_type == "eden":
                return await self._prepare_eden_protection(opportunity)
            return None
            
        except Exception as e:
            self.logger.error(f"Error preparing protection: {str(e)}")
            return None

    async def _calculate_priority_fee(
        self,
        strategy: ExecutionStrategy,
        mev_risks: Dict
    ) -> int:
        """Calculate optimal priority fee."""
        try:
            # Get base fee
            base_fee = await self.web3.eth.get_block('latest')
            base_fee = base_fee.get('baseFeePerGas', 0)
            
            # Calculate multiplier based on strategy and risks
            multiplier = 1.0
            
            if strategy.priority == 1:  # Aggressive
                multiplier = 2.0
            elif strategy.priority == 2:  # Balanced
                multiplier = 1.5
            
            # Adjust for MEV risks
            if mev_risks["total_risk"] > 0.5:
                multiplier *= 1.5
            
            priority_fee = int(base_fee * multiplier)
            
            # Ensure minimum fee
            return max(
                priority_fee,
                Web3.to_wei(1, "gwei")
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating priority fee: {str(e)}")
            return Web3.to_wei(2, "gwei")

    async def _prepare_bundle(
        self,
        transactions: List[Dict],
        strategy: ExecutionStrategy,
        expected_profit: float
    ) -> Optional[Dict]:
        """Prepare transaction bundle."""
        try:
            builder = self.bundle_builders.get(strategy.bundle_type)
            if not builder:
                return None
            
            return await builder(
                transactions,
                strategy,
                expected_profit
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing bundle: {str(e)}")
            return None

    async def _build_flashbots_bundle(
        self,
        transactions: List[Dict],
        strategy: ExecutionStrategy,
        expected_profit: float
    ) -> Dict:
        """Build Flashbots bundle."""
        try:
            # Calculate optimal bribe
            bribe = self._calculate_bribe(
                expected_profit,
                strategy
            )
            
            # Create coinbase payment
            coinbase_tx = {
                "to": "0x0000000000000000000000000000000000000000",
                "value": Web3.to_wei(bribe, "ether"),
                "gas": 21000,
                "maxFeePerGas": transactions[0]["maxFeePerGas"],
                "maxPriorityFeePerGas": transactions[0]["maxPriorityFeePerGas"]
            }
            
            return {
                "txs": [coinbase_tx] + transactions,
                "block_number": "latest",
                "min_timestamp": 0,
                "max_timestamp": strategy.timeout,
                "revertingTxHashes": []
            }
            
        except Exception as e:
            self.logger.error(f"Error building Flashbots bundle: {str(e)}")
            return {}

    async def _build_eden_bundle(
        self,
        transactions: List[Dict],
        strategy: ExecutionStrategy,
        expected_profit: float
    ) -> Dict:
        """Build Eden Network bundle."""
        try:
            return {
                "txs": transactions,
                "preferences": {
                    "priority": True,
                    "privacy": True,
                    "bribe": {
                        "amount": str(Web3.to_wei(
                            self._calculate_bribe(expected_profit, strategy),
                            "ether"
                        )),
                        "token": "ETH"
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error building Eden bundle: {str(e)}")
            return {}

    async def _build_standard_bundle(
        self,
        transactions: List[Dict],
        strategy: ExecutionStrategy,
        expected_profit: float
    ) -> Dict:
        """Build standard transaction bundle."""
        try:
            return {
                "transactions": transactions,
                "timeout": strategy.timeout
            }
            
        except Exception as e:
            self.logger.error(f"Error building standard bundle: {str(e)}")
            return {}

    def _calculate_bribe(
        self,
        expected_profit: float,
        strategy: ExecutionStrategy
    ) -> float:
        """Calculate optimal bribe amount."""
        try:
            if strategy.priority == 1:  # Aggressive
                bribe_ratio = self.mev_settings["max_bribe"]
            elif strategy.priority == 2:  # Balanced
                bribe_ratio = (
                    self.mev_settings["min_bribe"] +
                    self.mev_settings["max_bribe"]
                ) / 2
            else:  # Conservative
                bribe_ratio = self.mev_settings["min_bribe"]
            
            return expected_profit * bribe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating bribe: {str(e)}")
            return 0.0

    async def _validate_execution_plan(
        self,
        plan: ExecutionPlan,
        context: Dict
    ) -> bool:
        """Validate execution plan."""
        try:
            # Check gas costs
            gas_cost = plan.estimated_gas * plan.priority_fee
            max_gas_cost = plan.estimated_profit * 0.1  # Max 10% of profit
            
            if gas_cost > max_gas_cost:
                return False
            
            # Check network conditions
            if context.get("network_load", 0) > 0.8 and plan.strategy.priority > 1:
                return False
            
            # Check timing
            if "next_block_time" in context:
                if context["next_block_time"] > plan.strategy.timeout:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating plan: {str(e)}")
            return False

    async def execute_plan(
        self,
        plan: ExecutionPlan
    ) -> Dict:
        """Execute prepared plan."""
        try:
            start_time = datetime.utcnow()
            
            # Execute based on bundle type
            if plan.strategy.bundle_type == "flashbots":
                result = await self._execute_flashbots_bundle(plan)
            elif plan.strategy.bundle_type == "eden":
                result = await self._execute_eden_bundle(plan)
            else:
                result = await self._execute_standard_transactions(plan)
            
            # Calculate execution time
            execution_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            # Update metrics
            await self._update_metrics(
                result,
                execution_time,
                plan
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing plan: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_executions"] /
                max(1, self.metrics["total_executions"])
            ),
            "avg_profit": (
                self.metrics["total_profit"] /
                max(1, self.metrics["successful_executions"])
            ),
            "mev_protection_rate": (
                self.metrics["mev_attacks_prevented"] /
                max(1, self.metrics["total_executions"])
            ),
            "execution_history_size": len(self.execution_history),
            "last_update": datetime.utcnow().isoformat()
        }
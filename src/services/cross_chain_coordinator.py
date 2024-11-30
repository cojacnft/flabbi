from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3

from ..config.chains import (
    ChainConfig,
    get_chain_config,
    get_supported_chains
)
from ..services.chain_provider import ChainProvider
from ..services.market_data import MarketDataAggregator
from ..services.profit_analyzer import ProfitAnalyzer
from ..services.ml_path_finder import MLPathFinder
from ..services.price_predictor import PricePredictor
from ..services.gas_optimizer import GasOptimizer

class CrossChainCoordinator:
    def __init__(
        self,
        enabled_chains: List[int],
        private_key: str,
        settings: Dict
    ):
        self.enabled_chains = enabled_chains
        self.private_key = private_key
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize services per chain
        self.chain_providers: Dict[int, ChainProvider] = {}
        self.market_data: Dict[int, MarketDataAggregator] = {}
        self.profit_analyzers: Dict[int, ProfitAnalyzer] = {}
        self.path_finders: Dict[int, MLPathFinder] = {}
        self.price_predictors: Dict[int, PricePredictor] = {}
        self.gas_optimizers: Dict[int, GasOptimizer] = {}
        
        # Cross-chain state
        self.active_opportunities: Dict[str, Dict] = {}
        self.chain_states: Dict[int, Dict] = {}
        self.bridge_states: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            "opportunities_found": 0,
            "successful_arbitrages": 0,
            "failed_arbitrages": 0,
            "total_profit_usd": 0.0,
            "avg_execution_time_ms": 0.0
        }
        
        # Initialize services
        self._init_services()

    def _init_services(self):
        """Initialize services for each chain."""
        try:
            for chain_id in self.enabled_chains:
                if not get_chain_config(chain_id):
                    continue
                
                # Initialize chain provider
                self.chain_providers[chain_id] = ChainProvider(chain_id)
                
                # Initialize market data service
                self.market_data[chain_id] = MarketDataAggregator(
                    self.chain_providers[chain_id].web3
                )
                
                # Initialize profit analyzer
                self.profit_analyzers[chain_id] = ProfitAnalyzer(
                    self.chain_providers[chain_id].web3,
                    self.market_data[chain_id]
                )
                
                # Initialize path finder
                self.path_finders[chain_id] = MLPathFinder(
                    self.market_data[chain_id],
                    self.profit_analyzers[chain_id]
                )
                
                # Initialize price predictor
                self.price_predictors[chain_id] = PricePredictor(
                    self.market_data[chain_id]
                )
                
                # Initialize gas optimizer
                self.gas_optimizers[chain_id] = GasOptimizer(chain_id)
                
            self.logger.info(f"Initialized services for {len(self.enabled_chains)} chains")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {str(e)}")
            raise

    async def start(self):
        """Start cross-chain monitoring."""
        try:
            # Start monitoring tasks
            tasks = []
            
            for chain_id in self.enabled_chains:
                tasks.extend([
                    asyncio.create_task(
                        self._monitor_chain(chain_id)
                    ),
                    asyncio.create_task(
                        self._monitor_bridges(chain_id)
                    )
                ])
            
            # Start opportunity analyzer
            tasks.append(
                asyncio.create_task(
                    self._analyze_opportunities()
                )
            )
            
            # Wait for all tasks
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            raise

    async def _monitor_chain(self, chain_id: int):
        """Monitor single chain for opportunities."""
        try:
            while True:
                try:
                    # Update chain state
                    self.chain_states[chain_id] = await self._get_chain_state(chain_id)
                    
                    # Find opportunities
                    opportunities = await self._find_chain_opportunities(chain_id)
                    
                    # Process opportunities
                    for opp in opportunities:
                        await self._process_opportunity(chain_id, opp)
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring chain {chain_id}: {str(e)}")
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            self.logger.info(f"Stopped monitoring chain {chain_id}")

    async def _monitor_bridges(self, chain_id: int):
        """Monitor bridge states and liquidity."""
        try:
            while True:
                try:
                    bridges = self._get_chain_bridges(chain_id)
                    
                    for bridge in bridges:
                        state = await self._get_bridge_state(bridge)
                        self.bridge_states[bridge["id"]] = state
                    
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring bridges: {str(e)}")
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            self.logger.info("Stopped monitoring bridges")

    async def _analyze_opportunities(self):
        """Analyze and execute opportunities."""
        try:
            while True:
                try:
                    # Get all active opportunities
                    opportunities = list(self.active_opportunities.values())
                    
                    # Sort by expected profit
                    opportunities.sort(
                        key=lambda x: x["expected_profit"],
                        reverse=True
                    )
                    
                    # Execute best opportunities
                    for opp in opportunities[:3]:  # Top 3 opportunities
                        if await self._validate_opportunity(opp):
                            await self._execute_opportunity(opp)
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing opportunities: {str(e)}")
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            self.logger.info("Stopped analyzing opportunities")

    async def _get_chain_state(self, chain_id: int) -> Dict:
        """Get current chain state."""
        try:
            provider = self.chain_providers[chain_id]
            market_data = self.market_data[chain_id]
            
            # Get basic chain state
            state = await provider.get_chain_state()
            
            # Get market metrics
            state["total_liquidity"] = await market_data.get_total_liquidity()
            state["volume_24h"] = await market_data.get_24h_volume()
            
            # Get gas metrics
            gas_data = await self.gas_optimizers[chain_id].optimize_gas_price()
            state["gas_price"] = gas_data["gas_price"]
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error getting chain state: {str(e)}")
            return {}

    def _get_chain_bridges(self, chain_id: int) -> List[Dict]:
        """Get bridges for chain."""
        bridges = {
            1: [  # Ethereum bridges
                {
                    "id": "hop_eth_poly",
                    "name": "Hop Protocol",
                    "from_chain": 1,
                    "to_chain": 137,
                    "address": "0x..."
                },
                {
                    "id": "hop_eth_arb",
                    "name": "Hop Protocol",
                    "from_chain": 1,
                    "to_chain": 42161,
                    "address": "0x..."
                }
            ],
            137: [  # Polygon bridges
                {
                    "id": "hop_poly_eth",
                    "name": "Hop Protocol",
                    "from_chain": 137,
                    "to_chain": 1,
                    "address": "0x..."
                }
            ],
            42161: [  # Arbitrum bridges
                {
                    "id": "hop_arb_eth",
                    "name": "Hop Protocol",
                    "from_chain": 42161,
                    "to_chain": 1,
                    "address": "0x..."
                }
            ]
        }
        
        return bridges.get(chain_id, [])

    async def _get_bridge_state(self, bridge: Dict) -> Dict:
        """Get bridge state and liquidity."""
        try:
            # Get source chain provider
            provider = self.chain_providers[bridge["from_chain"]]
            
            # Get bridge contract
            contract = provider.web3.eth.contract(
                address=bridge["address"],
                abi=[]  # TODO: Add bridge ABI
            )
            
            # Get liquidity
            liquidity = await contract.functions.getTotalLiquidity().call()
            
            # Get fees
            fees = await contract.functions.getFees().call()
            
            return {
                "liquidity": liquidity,
                "fees": fees,
                "last_update": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bridge state: {str(e)}")
            return {}

    async def _find_chain_opportunities(self, chain_id: int) -> List[Dict]:
        """Find opportunities on single chain."""
        try:
            opportunities = []
            
            # Get path finder
            path_finder = self.path_finders[chain_id]
            
            # Get active tokens
            tokens = await self.market_data[chain_id].get_active_tokens()
            
            # Find opportunities for each token
            for token in tokens:
                paths = await path_finder.find_optimal_path(
                    token,
                    max_hops=4,
                    min_profit_usd=self.settings["min_profit_usd"]
                )
                
                if paths:
                    opportunities.extend(paths)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return []

    async def _process_opportunity(self, chain_id: int, opportunity: Dict):
        """Process and validate new opportunity."""
        try:
            # Generate unique ID
            opp_id = self._generate_opportunity_id(chain_id, opportunity)
            
            # Check if already exists
            if opp_id in self.active_opportunities:
                return
            
            # Validate profitability
            profit_analysis = await self.profit_analyzers[chain_id].analyze_opportunity(
                opportunity
            )
            
            if not profit_analysis["profitable"]:
                return
            
            # Get price predictions
            predictions = {}
            for token in opportunity["path"]:
                pred = await self.price_predictors[chain_id].predict_price(
                    token.address
                )
                if pred:
                    predictions[token.address] = pred
            
            # Optimize gas
            gas_data = await self.gas_optimizers[chain_id].optimize_gas_price()
            
            # Add to active opportunities
            self.active_opportunities[opp_id] = {
                "id": opp_id,
                "chain_id": chain_id,
                "opportunity": opportunity,
                "profit_analysis": profit_analysis,
                "price_predictions": predictions,
                "gas_data": gas_data,
                "timestamp": datetime.utcnow()
            }
            
            self.metrics["opportunities_found"] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing opportunity: {str(e)}")

    def _generate_opportunity_id(self, chain_id: int, opportunity: Dict) -> str:
        """Generate unique opportunity ID."""
        path_str = "-".join(t.address for t in opportunity["path"])
        return f"{chain_id}_{path_str}"

    async def _validate_opportunity(self, opportunity: Dict) -> bool:
        """Validate opportunity is still valid."""
        try:
            chain_id = opportunity["chain_id"]
            
            # Check age
            age = datetime.utcnow() - opportunity["timestamp"]
            if age > timedelta(seconds=30):
                return False
            
            # Revalidate profitability
            profit_analysis = await self.profit_analyzers[chain_id].analyze_opportunity(
                opportunity["opportunity"]
            )
            
            if not profit_analysis["profitable"]:
                return False
            
            # Check price movement
            for token_addr, pred in opportunity["price_predictions"].items():
                current_price = await self.market_data[chain_id].get_token_price(
                    token_addr
                )
                
                if current_price is None:
                    return False
                
                # Check if price moved significantly
                price_diff = abs(current_price - pred["price"]) / pred["price"]
                if price_diff > 0.01:  # 1% movement
                    return False
            
            # Check gas price
            current_gas = await self.gas_optimizers[chain_id].optimize_gas_price()
            if current_gas["gas_price"] > opportunity["gas_data"]["gas_price"] * 1.2:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating opportunity: {str(e)}")
            return False

    async def _execute_opportunity(self, opportunity: Dict):
        """Execute arbitrage opportunity."""
        try:
            start_time = datetime.utcnow()
            chain_id = opportunity["chain_id"]
            
            # Get provider
            provider = self.chain_providers[chain_id]
            
            # Prepare transaction
            tx_data = await self._prepare_transaction(opportunity)
            
            # Execute transaction
            tx_hash = await provider.send_transaction(
                tx_data,
                self.private_key
            )
            
            if not tx_hash:
                self.metrics["failed_arbitrages"] += 1
                return
            
            # Wait for confirmation
            receipt = await provider.wait_for_transaction(tx_hash)
            
            if receipt and receipt["status"] == 1:
                # Success
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                profit = opportunity["profit_analysis"]["expected_profit"]
                
                self.metrics["successful_arbitrages"] += 1
                self.metrics["total_profit_usd"] += profit
                self.metrics["avg_execution_time_ms"] = (
                    self.metrics["avg_execution_time_ms"] *
                    (self.metrics["successful_arbitrages"] - 1) +
                    execution_time
                ) / self.metrics["successful_arbitrages"]
                
                # Update models
                await self._update_models(opportunity, True, profit)
            else:
                self.metrics["failed_arbitrages"] += 1
                await self._update_models(opportunity, False, 0)
            
            # Remove from active opportunities
            del self.active_opportunities[opportunity["id"]]
            
        except Exception as e:
            self.logger.error(f"Error executing opportunity: {str(e)}")
            self.metrics["failed_arbitrages"] += 1

    async def _prepare_transaction(self, opportunity: Dict) -> Dict:
        """Prepare transaction data."""
        try:
            chain_id = opportunity["chain_id"]
            
            # Get latest gas data
            gas_data = await self.gas_optimizers[chain_id].optimize_gas_price()
            
            # Prepare transaction
            return {
                "to": opportunity["opportunity"]["contract_address"],
                "data": opportunity["opportunity"]["data"],
                "value": 0,
                "gasPrice": Web3.to_wei(gas_data["gas_price"], "gwei"),
                "maxFeePerGas": Web3.to_wei(gas_data["max_fee_per_gas"], "gwei"),
                "maxPriorityFeePerGas": Web3.to_wei(
                    gas_data["priority_fee_per_gas"],
                    "gwei"
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing transaction: {str(e)}")
            return {}

    async def _update_models(
        self,
        opportunity: Dict,
        success: bool,
        profit: float
    ):
        """Update ML models with execution results."""
        try:
            chain_id = opportunity["chain_id"]
            
            # Update path finder
            await self.path_finders[chain_id].update_model(
                opportunity["opportunity"],
                success
            )
            
            # Update price predictor
            for token_addr, pred in opportunity["price_predictions"].items():
                current_price = await self.market_data[chain_id].get_token_price(
                    token_addr
                )
                if current_price:
                    await self.price_predictors[chain_id].update_models(
                        token_addr,
                        current_price
                    )
            
            # Update gas optimizer
            await self.gas_optimizers[chain_id].update_model(
                opportunity["gas_data"]["gas_price"],
                success,
                opportunity["profit_analysis"]["execution_time"]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "active_opportunities": len(self.active_opportunities),
            "chain_states": {
                chain_id: state
                for chain_id, state in self.chain_states.items()
            },
            "bridge_states": self.bridge_states,
            "last_update": datetime.utcnow().isoformat()
        }
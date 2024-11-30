from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime
import numpy as np
from web3 import Web3

from ..services.network_manager import NetworkManager
from ..services.gelato_service import GelatoService
from ..services.market_analyzer import MarketAnalyzer
from ..services.token_database import TokenDatabase
from ..models.token import Token
from ..models.settings import SystemSettings

class ArbitrageEngine:
    def __init__(
        self,
        settings: SystemSettings,
        web3: Web3,
        private_key: str
    ):
        self.settings = settings
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.network = NetworkManager()
        self.gelato = GelatoService(web3, private_key)
        self.token_db = TokenDatabase(settings, web3)
        self.market_analyzer = MarketAnalyzer(web3, self.token_db)
        
        # State management
        self.active_tasks: Dict[str, Dict] = {}
        self.recent_opportunities: List[Dict] = []
        self.blacklisted_pairs: Set[str] = set()
        
        # Performance metrics
        self.metrics = {
            "opportunities_found": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_profit_usd": 0,
            "average_execution_time_ms": 0
        }

    async def start(self):
        """Start the arbitrage engine."""
        try:
            # Initialize services
            await self.network.initialize()
            
            # Start monitoring tasks
            monitoring_task = asyncio.create_task(self._monitor_opportunities())
            execution_task = asyncio.create_task(self._manage_executions())
            cleanup_task = asyncio.create_task(self._periodic_cleanup())
            
            # Wait for tasks
            await asyncio.gather(
                monitoring_task,
                execution_task,
                cleanup_task
            )
            
        except Exception as e:
            self.logger.error(f"Error starting arbitrage engine: {str(e)}")
            raise

    async def _monitor_opportunities(self):
        """Monitor for arbitrage opportunities."""
        while True:
            try:
                # Get active tokens
                tokens = await self.token_db.get_active_tokens()
                
                # Process in batches to manage rate limits
                batch_size = 10
                for i in range(0, len(tokens), batch_size):
                    batch = tokens[i:i + batch_size]
                    
                    # Find opportunities for batch
                    opportunities = await self._find_opportunities_batch(batch)
                    
                    # Filter and validate opportunities
                    valid_opportunities = [
                        opp for opp in opportunities
                        if await self._validate_opportunity(opp)
                    ]
                    
                    # Create execution tasks for valid opportunities
                    for opp in valid_opportunities:
                        await self._create_execution_task(opp)
                    
                    # Rate limiting pause
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring opportunities: {str(e)}")
                await asyncio.sleep(5)

    async def _find_opportunities_batch(
        self,
        tokens: List[Token]
    ) -> List[Dict]:
        """Find arbitrage opportunities for a batch of tokens."""
        try:
            opportunities = []
            
            # Get market data for tokens
            market_data = await self._get_market_data(tokens)
            
            # Find opportunities between pairs
            for i, token1 in enumerate(tokens):
                for token2 in tokens[i+1:]:
                    # Skip blacklisted pairs
                    pair_key = self._get_pair_key(token1.address, token2.address)
                    if pair_key in self.blacklisted_pairs:
                        continue
                    
                    # Find opportunities between pair
                    pair_opportunities = await self._analyze_pair(
                        token1,
                        token2,
                        market_data
                    )
                    
                    opportunities.extend(pair_opportunities)
            
            # Sort by expected profit
            opportunities.sort(
                key=lambda x: x["expected_profit"],
                reverse=True
            )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")
            return []

    async def _analyze_pair(
        self,
        token1: Token,
        token2: Token,
        market_data: Dict
    ) -> List[Dict]:
        """Analyze trading pair for opportunities."""
        try:
            opportunities = []
            
            # Get liquidity pools for pair
            pools = self._get_pair_pools(token1, token2)
            
            # Analyze each pool combination
            for i, pool1 in enumerate(pools):
                for pool2 in pools[i+1:]:
                    # Calculate price difference
                    price_diff = self._calculate_price_difference(
                        pool1,
                        pool2,
                        market_data
                    )
                    
                    if price_diff > self.settings.min_price_difference:
                        # Calculate potential profit
                        profit = await self._calculate_potential_profit(
                            pool1,
                            pool2,
                            price_diff,
                            market_data
                        )
                        
                        if profit > self.settings.min_profit_usd:
                            opportunities.append({
                                "token1": token1,
                                "token2": token2,
                                "pool1": pool1,
                                "pool2": pool2,
                                "price_difference": price_diff,
                                "expected_profit": profit,
                                "timestamp": datetime.utcnow()
                            })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair: {str(e)}")
            return []

    async def _validate_opportunity(self, opportunity: Dict) -> bool:
        """Validate arbitrage opportunity."""
        try:
            # Check if opportunity is still valid
            if (datetime.utcnow() - opportunity["timestamp"]).seconds > 30:
                return False
            
            # Verify liquidity
            if not await self._verify_liquidity(opportunity):
                return False
            
            # Check gas costs
            gas_cost = await self._estimate_gas_cost(opportunity)
            if gas_cost >= opportunity["expected_profit"]:
                return False
            
            # Verify price impact
            price_impact = await self._calculate_price_impact(opportunity)
            if price_impact > self.settings.max_price_impact:
                return False
            
            # Calculate success probability
            success_prob = await self._calculate_success_probability(opportunity)
            if success_prob < self.settings.min_success_probability:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating opportunity: {str(e)}")
            return False

    async def _create_execution_task(self, opportunity: Dict):
        """Create Gelato execution task for opportunity."""
        try:
            # Prepare execution data
            execution_data = self._prepare_execution_data(opportunity)
            
            # Set execution conditions
            conditions = {
                "minProfit": opportunity["expected_profit"],
                "maxGasPrice": await self.network.get_best_gas_price(),
                "minLiquidity": self._get_min_liquidity(opportunity),
                "maxSlippage": self.settings.max_slippage
            }
            
            # Create Gelato task
            task_id = await self.gelato.create_arbitrage_task(
                execution_data,
                conditions
            )
            
            if task_id:
                # Track active task
                self.active_tasks[task_id] = {
                    "opportunity": opportunity,
                    "created_at": datetime.utcnow(),
                    "status": "pending"
                }
                
                self.logger.info(f"Created execution task: {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating execution task: {str(e)}")

    async def _manage_executions(self):
        """Manage active execution tasks."""
        while True:
            try:
                # Get active task IDs
                task_ids = list(self.active_tasks.keys())
                
                for task_id in task_ids:
                    task_data = self.active_tasks[task_id]
                    
                    # Check task status
                    status = await self.gelato.monitor_task(task_id)
                    
                    # Update task status
                    task_data["status"] = status["status"]
                    
                    # Handle completed tasks
                    if status["status"] in ["success", "failed"]:
                        await self._handle_completed_task(task_id, status)
                    
                    # Cancel stale tasks
                    elif (datetime.utcnow() - task_data["created_at"]).seconds > 300:
                        await self._cancel_stale_task(task_id)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error managing executions: {str(e)}")
                await asyncio.sleep(5)

    async def _handle_completed_task(self, task_id: str, status: Dict):
        """Handle completed execution task."""
        try:
            task_data = self.active_tasks[task_id]
            opportunity = task_data["opportunity"]
            
            if status["status"] == "success":
                # Update metrics
                self.metrics["successful_executions"] += 1
                self.metrics["total_profit_usd"] += opportunity["expected_profit"]
                
                # Log success
                self.logger.info(
                    f"Successful arbitrage: {opportunity['expected_profit']} USD"
                )
                
            else:
                # Update metrics
                self.metrics["failed_executions"] += 1
                
                # Check if pair should be blacklisted
                await self._check_blacklist_pair(
                    opportunity["token1"].address,
                    opportunity["token2"].address
                )
                
                # Log failure
                self.logger.warning(
                    f"Failed arbitrage: {status.get('error', 'Unknown error')}"
                )
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
        except Exception as e:
            self.logger.error(f"Error handling completed task: {str(e)}")

    async def _periodic_cleanup(self):
        """Perform periodic cleanup tasks."""
        while True:
            try:
                # Clean old opportunities
                self.recent_opportunities = [
                    opp for opp in self.recent_opportunities
                    if (datetime.utcnow() - opp["timestamp"]).seconds < 3600
                ]
                
                # Update blacklist
                await self._update_blacklist()
                
                # Update market data
                await self.market_analyzer.update_market_data()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {str(e)}")
                await asyncio.sleep(60)

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "active_tasks": len(self.active_tasks),
            "blacklisted_pairs": len(self.blacklisted_pairs),
            "recent_opportunities": len(self.recent_opportunities)
        }

    def _get_pair_key(self, token1: str, token2: str) -> str:
        """Get unique key for token pair."""
        return '-'.join(sorted([token1.lower(), token2.lower()]))
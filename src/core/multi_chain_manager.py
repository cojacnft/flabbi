from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime
import json

from ..config.chains import (
    ChainConfig,
    get_chain_config,
    get_supported_chains,
    is_chain_supported
)
from ..services.chain_provider import ChainProvider
from ..services.metrics import MetricsService
from ..services.alert_manager import AlertManager
from .arbitrage_engine import ArbitrageEngine

class MultiChainManager:
    def __init__(
        self,
        private_key: str,
        enabled_chains: Optional[List[int]] = None
    ):
        self.private_key = private_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.metrics = MetricsService()
        self.alert_manager = AlertManager({})  # TODO: Add alert config
        
        # Chain management
        self.enabled_chains = enabled_chains or [1]  # Default to Ethereum
        self.chain_providers: Dict[int, ChainProvider] = {}
        self.arbitrage_engines: Dict[int, ArbitrageEngine] = {}
        
        # State management
        self.active_tasks: Dict[int, asyncio.Task] = {}
        self.chain_states: Dict[int, Dict] = {}
        
        # Initialize chains
        self._init_chains()

    def _init_chains(self):
        """Initialize enabled chains."""
        try:
            for chain_id in self.enabled_chains:
                if not is_chain_supported(chain_id):
                    self.logger.warning(f"Chain {chain_id} is not supported")
                    continue
                
                # Initialize chain provider
                self.chain_providers[chain_id] = ChainProvider(chain_id)
                
                # Initialize arbitrage engine
                config = get_chain_config(chain_id)
                self.arbitrage_engines[chain_id] = ArbitrageEngine(
                    config,
                    self.chain_providers[chain_id].get_web3(),
                    self.private_key
                )
                
                self.logger.info(f"Initialized chain {chain_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing chains: {str(e)}")
            raise

    async def start(self):
        """Start monitoring all enabled chains."""
        try:
            # Start monitoring tasks for each chain
            for chain_id in self.enabled_chains:
                if chain_id not in self.arbitrage_engines:
                    continue
                
                task = asyncio.create_task(
                    self._monitor_chain(chain_id)
                )
                self.active_tasks[chain_id] = task
            
            # Start metrics collection
            asyncio.create_task(self._collect_metrics())
            
            # Start alert monitoring
            asyncio.create_task(self._monitor_alerts())
            
            self.logger.info("Multi-chain monitoring started")
            
            # Wait for all tasks
            await asyncio.gather(*self.active_tasks.values())
            
        except Exception as e:
            self.logger.error(f"Error starting multi-chain monitoring: {str(e)}")
            await self.stop()

    async def stop(self):
        """Stop all monitoring tasks."""
        try:
            # Cancel all tasks
            for task in self.active_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                *self.active_tasks.values(),
                return_exceptions=True
            )
            
            self.logger.info("Multi-chain monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")

    async def _monitor_chain(self, chain_id: int):
        """Monitor a specific chain for opportunities."""
        try:
            engine = self.arbitrage_engines[chain_id]
            provider = self.chain_providers[chain_id]
            
            while True:
                try:
                    # Update chain state
                    self.chain_states[chain_id] = await provider.get_chain_state()
                    
                    # Check for opportunities
                    opportunities = await engine.find_opportunities()
                    
                    # Process opportunities
                    for opp in opportunities:
                        if await engine.validate_opportunity(opp):
                            # Execute arbitrage
                            success = await engine.execute_arbitrage(opp)
                            
                            # Record metrics
                            if success:
                                self.metrics.record_opportunity_executed(
                                    opp.expected_profit,
                                    opp.gas_estimate
                                )
                            
                            # Send alerts
                            await self._handle_execution_alert(
                                chain_id,
                                opp,
                                success
                            )
                    
                    # Sleep between iterations
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(
                        f"Error monitoring chain {chain_id}: {str(e)}"
                    )
                    await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            self.logger.info(f"Stopped monitoring chain {chain_id}")
        except Exception as e:
            self.logger.error(f"Fatal error monitoring chain {chain_id}: {str(e)}")

    async def _collect_metrics(self):
        """Collect system-wide metrics."""
        try:
            while True:
                metrics = {}
                
                # Collect chain metrics
                for chain_id, state in self.chain_states.items():
                    metrics[f"chain_{chain_id}"] = {
                        "block_number": state.get("block_number", 0),
                        "gas_price": state.get("gas_price", 0),
                        "provider_health": state.get("provider_health", {})
                    }
                
                # Update metrics
                self.metrics.record_batch_metrics(metrics)
                
                await asyncio.sleep(15)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")

    async def _monitor_alerts(self):
        """Monitor system for alert conditions."""
        try:
            while True:
                # Get current metrics
                metrics = self.metrics.get_current_metrics()
                
                # Check thresholds
                await self.alert_manager.check_thresholds(metrics)
                
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error monitoring alerts: {str(e)}")

    async def _handle_execution_alert(
        self,
        chain_id: int,
        opportunity: Dict,
        success: bool
    ):
        """Handle alerts for arbitrage execution."""
        try:
            chain_config = get_chain_config(chain_id)
            
            # Create alert message
            alert = {
                "title": f"Arbitrage {'Success' if success else 'Failed'}",
                "message": (
                    f"Chain: {chain_config.name}\n"
                    f"Profit: ${opportunity.expected_profit:.2f}\n"
                    f"Gas: {opportunity.gas_estimate:,}"
                ),
                "type": "execution",
                "level": "info" if success else "warning"
            }
            
            # Send alert
            await self.alert_manager._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error handling execution alert: {str(e)}")

    def get_chain_status(self, chain_id: int) -> Optional[Dict]:
        """Get status for a specific chain."""
        if chain_id not in self.chain_states:
            return None
        
        return {
            "state": self.chain_states[chain_id],
            "metrics": self.metrics.get_current_metrics().get(f"chain_{chain_id}", {}),
            "active": chain_id in self.active_tasks and not self.active_tasks[chain_id].done()
        }

    def get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            "active_chains": len(self.active_tasks),
            "metrics": self.metrics.get_current_metrics(),
            "alerts": self.alert_manager.get_recent_alerts(),
            "chains": {
                chain_id: self.get_chain_status(chain_id)
                for chain_id in self.enabled_chains
            }
        }

    async def add_chain(self, chain_id: int) -> bool:
        """Add a new chain to monitor."""
        try:
            if chain_id in self.enabled_chains:
                return True
            
            if not is_chain_supported(chain_id):
                return False
            
            # Initialize chain
            self.enabled_chains.append(chain_id)
            self._init_chains()
            
            # Start monitoring
            task = asyncio.create_task(
                self._monitor_chain(chain_id)
            )
            self.active_tasks[chain_id] = task
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding chain {chain_id}: {str(e)}")
            return False

    async def remove_chain(self, chain_id: int) -> bool:
        """Remove a chain from monitoring."""
        try:
            if chain_id not in self.enabled_chains:
                return True
            
            # Stop monitoring
            if chain_id in self.active_tasks:
                self.active_tasks[chain_id].cancel()
                await self.active_tasks[chain_id]
                del self.active_tasks[chain_id]
            
            # Cleanup
            self.enabled_chains.remove(chain_id)
            if chain_id in self.chain_providers:
                del self.chain_providers[chain_id]
            if chain_id in self.arbitrage_engines:
                del self.arbitrage_engines[chain_id]
            if chain_id in self.chain_states:
                del self.chain_states[chain_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing chain {chain_id}: {str(e)}")
            return False

    async def update_chain_config(
        self,
        chain_id: int,
        config_updates: Dict
    ) -> bool:
        """Update configuration for a chain."""
        try:
            if chain_id not in self.enabled_chains:
                return False
            
            # Get current config
            config = get_chain_config(chain_id)
            if not config:
                return False
            
            # Update config
            for key, value in config_updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Restart chain monitoring
            await self.remove_chain(chain_id)
            await self.add_chain(chain_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating chain config: {str(e)}")
            return False
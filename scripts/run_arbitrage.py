import asyncio
import click
import logging
from pathlib import Path
import sys
import yaml
from web3 import Web3

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import ConfigManager
from src.services.flash_loan_executor import FlashLoanExecutor
from src.services.execution_strategy import ExecutionStrategyOptimizer
from src.services.market_analyzer import MarketAnalyzer
from src.services.parameter_tuner import ParameterTuner
from src.services.monitoring import MonitoringService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArbitrageBot:
    """Main arbitrage bot class."""
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.running = False
        self.initialize_services()

    def initialize_services(self):
        """Initialize all services."""
        try:
            # Initialize Web3
            self.web3 = Web3(Web3.HTTPProvider(
                self.config.network.rpc_url
            ))
            
            # Initialize services
            self.market_analyzer = MarketAnalyzer(
                self.web3,
                self.config
            )
            
            self.parameter_tuner = ParameterTuner(
                None,  # Will be set after strategy optimizer
                None,  # Will be set after risk manager
                self.config.flash_loan.dict()
            )
            
            self.strategy_optimizer = ExecutionStrategyOptimizer(
                self.web3,
                None,  # Will be set after flash loan executor
                self.market_analyzer,
                self.config.flash_loan.dict()
            )
            
            self.flash_loan_executor = FlashLoanExecutor(
                self.web3,
                self.market_analyzer,
                self.strategy_optimizer,
                self.config.flash_loan.dict()
            )
            
            # Set circular references
            self.strategy_optimizer.flash_loan_executor = self.flash_loan_executor
            self.parameter_tuner.strategy_optimizer = self.strategy_optimizer
            
            # Initialize monitoring
            if self.config.monitoring.enabled:
                self.monitoring = MonitoringService(
                    self.config.monitoring,
                    {
                        "market_analyzer": self.market_analyzer,
                        "strategy_optimizer": self.strategy_optimizer,
                        "flash_loan_executor": self.flash_loan_executor,
                        "parameter_tuner": self.parameter_tuner
                    }
                )
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise

    async def start(self):
        """Start the arbitrage bot."""
        try:
            self.running = True
            logger.info("Starting arbitrage bot...")
            
            # Start monitoring
            if self.config.monitoring.enabled:
                await self.monitoring.start()
            
            # Main loop
            while self.running:
                try:
                    # Get market state
                    market_state = await self.market_analyzer.get_market_state()
                    
                    # Get optimal parameters
                    params = await self.parameter_tuner.get_optimal_parameters(
                        market_state
                    )
                    
                    # Update strategy optimizer
                    self.strategy_optimizer.update_parameters(params)
                    
                    # Find opportunities
                    opportunities = await self.market_analyzer.find_opportunities(
                        params["min_profit_threshold"]
                    )
                    
                    # Process opportunities
                    for opp in opportunities:
                        # Optimize execution
                        plan = await self.strategy_optimizer.optimize_execution(
                            opp,
                            market_state
                        )
                        
                        if plan:
                            # Execute plan
                            result = await self.flash_loan_executor.execute_plan(
                                plan
                            )
                            
                            # Update metrics
                            if self.config.monitoring.enabled:
                                await self.monitoring.record_execution(
                                    plan,
                                    result
                                )
                    
                    # Sleep between iterations
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error starting bot: {str(e)}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the arbitrage bot."""
        try:
            self.running = False
            logger.info("Stopping arbitrage bot...")
            
            # Stop monitoring
            if self.config.monitoring.enabled:
                await self.monitoring.stop()
            
        except Exception as e:
            logger.error(f"Error stopping bot: {str(e)}")

@click.command()
@click.option(
    "--config",
    default="config/arbitrage.yaml",
    help="Path to configuration file"
)
def main(config: str):
    """Run the arbitrage bot."""
    try:
        # Create bot instance
        bot = ArbitrageBot(config)
        
        # Run bot
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
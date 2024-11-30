import asyncio
import logging
from web3 import Web3
from dotenv import load_dotenv
import os

from config import SystemConfig, DEFAULT_CONFIG
from core.engine import ArbitrageEngine
from core.flash_loan import FlashLoanService
from core.protocol import ProtocolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Load environment variables
        load_dotenv()

        # Initialize Web3
        rpc_url = os.getenv('ETHEREUM_RPC_URL', DEFAULT_CONFIG.network.rpc_url)
        web3 = Web3(Web3.HTTPProvider(rpc_url))

        # Verify connection
        if not web3.is_connected():
            raise Exception("Failed to connect to Ethereum network")

        # Load configuration
        config = DEFAULT_CONFIG  # TODO: Load from file/env

        # Initialize core services
        protocol_manager = ProtocolManager(config, web3)
        flash_loan_service = FlashLoanService(config, web3)
        arbitrage_engine = ArbitrageEngine(config, web3)

        logger.info("Starting arbitrage monitoring...")

        while True:
            try:
                # Find opportunities
                opportunities = await arbitrage_engine.find_opportunities()

                for opportunity in opportunities:
                    # Validate and execute profitable opportunities
                    if await arbitrage_engine.validate_opportunity(opportunity):
                        success = await arbitrage_engine.execute_arbitrage(
                            opportunity
                        )
                        if success:
                            logger.info(
                                f"Successfully executed arbitrage: {opportunity}"
                            )

                # Wait before next iteration
                await asyncio.sleep(1)  # 1 second delay

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)  # Longer delay on error

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
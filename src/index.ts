import { ethers } from 'ethers';
import { Logger } from './services/Logger';
import { MetricsCollector } from './services/MetricsCollector';
import { ErrorHandler } from './services/ErrorHandler';
import { ChainManager } from './services/ChainManager';
import { DexRegistry } from './dex/DexRegistry';
import { PriceService } from './services/PriceService';
import { ArbitrageExecutor } from './services/ArbitrageExecutor';
import { ConfigService } from './services/ConfigService';
import { ArbitrageEngine } from './core/ArbitrageEngine';

async function main() {
    try {
        // Initialize configuration
        const config = await ConfigService.getInstance();
        
        // Initialize core services
        const metrics = MetricsCollector.getInstance(config);
        const logger = Logger.getInstance(config, metrics);
        const errorHandler = ErrorHandler.getInstance(logger, metrics, config);

        logger.info('Initializing Flash Loan Arbitrage Bot');

        // Initialize chain manager
        const chainManager = ChainManager.getInstance(
            logger,
            metrics,
            config,
            errorHandler
        );

        // Initialize DEX registry
        const dexRegistry = DexRegistry.getInstance(
            logger,
            metrics,
            errorHandler,
            chainManager
        );

        // Initialize price service
        const priceService = PriceService.getInstance(
            logger,
            metrics,
            chainManager,
            errorHandler
        );

        // Initialize arbitrage executor
        const arbitrageExecutor = ArbitrageExecutor.getInstance(
            logger,
            metrics,
            chainManager,
            errorHandler,
            priceService
        );

        // Initialize arbitrage engine
        const engine = new ArbitrageEngine(
            logger,
            metrics,
            errorHandler,
            chainManager,
            dexRegistry,
            priceService,
            arbitrageExecutor,
            config
        );

        // Handle shutdown
        process.on('SIGINT', async () => {
            logger.info('Shutting down...');
            await engine.stop();
            process.exit(0);
        });

        process.on('SIGTERM', async () => {
            logger.info('Shutting down...');
            await engine.stop();
            process.exit(0);
        });

        // Start the engine
        await engine.start();

        logger.info('Flash Loan Arbitrage Bot started successfully');

    } catch (error) {
        console.error('Fatal error:', error);
        process.exit(1);
    }
}

// Start the application
main().catch(console.error);
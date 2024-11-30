import { ethers } from 'ethers';
import { Logger } from '../services/Logger';
import { MetricsCollector } from '../services/MetricsCollector';
import { ErrorHandler } from '../services/ErrorHandler';
import { ChainManager } from '../services/ChainManager';
import { DexRegistry } from '../dex/DexRegistry';
import { PriceService } from '../services/PriceService';
import { ArbitrageExecutor } from '../services/ArbitrageExecutor';
import { ConfigService } from '../services/ConfigService';

export class ArbitrageEngine {
    private isRunning: boolean = false;
    private executionInterval: NodeJS.Timeout | null = null;

    constructor(
        private readonly logger: Logger,
        private readonly metrics: MetricsCollector,
        private readonly errorHandler: ErrorHandler,
        private readonly chainManager: ChainManager,
        private readonly dexRegistry: DexRegistry,
        private readonly priceService: PriceService,
        private readonly arbitrageExecutor: ArbitrageExecutor,
        private readonly config: ConfigService
    ) {}

    public async start(): Promise<void> {
        if (this.isRunning) {
            this.logger.warn('Arbitrage engine is already running');
            return;
        }

        this.isRunning = true;
        this.logger.info('Starting arbitrage engine');

        try {
            // Initial setup
            await this.initialize();

            // Start main loop
            this.executionInterval = setInterval(
                () => this.mainLoop(),
                this.config.get('execution.interval', 1000) // Default 1 second
            );

            this.logger.info('Arbitrage engine started successfully');
        } catch (error) {
            this.isRunning = false;
            this.logger.error('Failed to start arbitrage engine', error);
            throw error;
        }
    }

    public async stop(): Promise<void> {
        if (!this.isRunning) {
            return;
        }

        this.isRunning = false;
        if (this.executionInterval) {
            clearInterval(this.executionInterval);
            this.executionInterval = null;
        }

        this.logger.info('Arbitrage engine stopped');
    }

    private async initialize(): Promise<void> {
        // Verify all services are ready
        await this.verifyServices();

        // Initialize chain connections
        const activeChains = this.chainManager.getActiveChains();
        this.logger.info('Active chains', { chains: activeChains });

        // Initialize DEX connections
        for (const chainId of activeChains) {
            const dexCount = this.dexRegistry.getDexCount(chainId);
            this.logger.info('DEX initialization', { chainId, dexCount });
        }

        // Warm up price feeds
        await this.priceService.initialize();
    }

    private async mainLoop(): Promise<void> {
        if (!this.isRunning) return;

        try {
            // Get active chains
            const activeChains = this.chainManager.getActiveChains();

            // Process each chain
            for (const chainId of activeChains) {
                if (!this.chainManager.isChainHealthy(chainId)) {
                    continue;
                }

                await this.processChain(chainId);
            }

        } catch (error) {
            this.logger.error('Error in main loop', error);
            // Don't stop the engine on error, just log and continue
        }
    }

    private async processChain(chainId: number): Promise<void> {
        const startTime = Date.now();
        try {
            // Get chain configuration
            const chainConfig = this.chainManager.getChainConfig(chainId);
            
            // Get base tokens to monitor
            const baseTokens = chainConfig.stablecoins;
            
            // Get all DEXes for this chain
            const dexes = this.dexRegistry.getEnabledDexes(chainId);

            // Find opportunities
            const opportunities = await this.findOpportunities(
                chainId,
                baseTokens,
                dexes
            );

            // Execute profitable opportunities
            if (opportunities.length > 0) {
                await this.executeOpportunities(chainId, opportunities);
            }

            // Update metrics
            this.metrics.setGauge('chain_processing_time', Date.now() - startTime, {
                chainId: chainId.toString()
            });

        } catch (error) {
            this.logger.error('Error processing chain', {
                chainId,
                error
            });
        }
    }

    private async findOpportunities(
        chainId: number,
        baseTokens: any[],
        dexes: string[]
    ): Promise<any[]> {
        const opportunities = [];

        for (const baseToken of baseTokens) {
            // Get pairs for base token
            const pairs = await this.getPotentialPairs(chainId, baseToken.address);

            // Check each pair
            for (const pair of pairs) {
                // Get quotes from all DEXes
                const quotes = await this.dexRegistry.getQuotes(
                    chainId,
                    baseToken.address,
                    pair,
                    ethers.utils.parseUnits('1', 18) // 1 token as test amount
                );

                // Find arbitrage opportunities between DEXes
                const arbitrageOpps = this.findArbitrageBetweenDexes(
                    quotes,
                    baseToken,
                    pair
                );

                opportunities.push(...arbitrageOpps);
            }
        }

        return this.filterAndPrioritizeOpportunities(opportunities);
    }

    private async getPotentialPairs(
        chainId: number,
        baseToken: string
    ): Promise<string[]> {
        // Get pairs with significant liquidity
        // This is a simplified version - you'll need to implement proper pair discovery
        return [];
    }

    private findArbitrageBetweenDexes(
        quotes: Record<string, ethers.BigNumber>,
        baseToken: any,
        pair: string
    ): any[] {
        const opportunities = [];
        const dexes = Object.keys(quotes);

        // Compare each DEX pair
        for (let i = 0; i < dexes.length; i++) {
            for (let j = i + 1; j < dexes.length; j++) {
                const dex1 = dexes[i];
                const dex2 = dexes[j];
                const price1 = quotes[dex1];
                const price2 = quotes[dex2];

                // Check if there's a profitable arbitrage
                if (price1.gt(price2.mul(102).div(100))) { // 2% minimum profit
                    opportunities.push({
                        baseToken,
                        pair,
                        buyDex: dex2,
                        sellDex: dex1,
                        priceDiff: price1.sub(price2),
                        estimatedProfit: this.calculateEstimatedProfit(
                            price1,
                            price2,
                            baseToken
                        )
                    });
                } else if (price2.gt(price1.mul(102).div(100))) {
                    opportunities.push({
                        baseToken,
                        pair,
                        buyDex: dex1,
                        sellDex: dex2,
                        priceDiff: price2.sub(price1),
                        estimatedProfit: this.calculateEstimatedProfit(
                            price2,
                            price1,
                            baseToken
                        )
                    });
                }
            }
        }

        return opportunities;
    }

    private calculateEstimatedProfit(
        sellPrice: ethers.BigNumber,
        buyPrice: ethers.BigNumber,
        baseToken: any
    ): ethers.BigNumber {
        // Simplified profit calculation
        // You'll need to account for:
        // - Gas costs
        // - Flash loan fees
        // - DEX fees
        // - Slippage
        return sellPrice.sub(buyPrice);
    }

    private filterAndPrioritizeOpportunities(
        opportunities: any[]
    ): any[] {
        // Sort by estimated profit
        return opportunities.sort((a, b) => 
            b.estimatedProfit.sub(a.estimatedProfit).toNumber()
        );
    }

    private async executeOpportunities(
        chainId: number,
        opportunities: any[]
    ): Promise<void> {
        // Get execution parameters
        const maxConcurrent = this.config.get(
            'execution.maxConcurrentPerChain',
            1
        );

        // Execute top opportunities
        const executions = opportunities
            .slice(0, maxConcurrent)
            .map(opp => this.arbitrageExecutor.executeOpportunity(opp));

        // Wait for all executions
        const results = await Promise.allSettled(executions);

        // Log results
        results.forEach((result, i) => {
            if (result.status === 'fulfilled') {
                this.logger.info('Arbitrage execution successful', {
                    chainId,
                    opportunity: opportunities[i],
                    result: result.value
                });
            } else {
                this.logger.error('Arbitrage execution failed', {
                    chainId,
                    opportunity: opportunities[i],
                    error: result.reason
                });
            }
        });
    }

    private async verifyServices(): Promise<void> {
        // Verify all required services are available and responding
        const checks = [
            this.chainManager.getActiveChains(),
            this.dexRegistry.getAdapterCount(),
            this.priceService.getPrice('USDC', 1), // Test price fetch
        ];

        await Promise.all(checks);
    }

    public getStatus(): any {
        return {
            isRunning: this.isRunning,
            activeChains: this.chainManager.getActiveChains(),
            dexCount: this.dexRegistry.getAdapterCount(),
            metrics: this.metrics.getMetrics()
        };
    }
}
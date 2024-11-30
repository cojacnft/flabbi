import { ethers } from 'ethers';
import { Logger } from './Logger';
import { MetricsCollector } from './MetricsCollector';
import { ConfigService } from './ConfigService';
import { ErrorHandler, ErrorSeverity, ErrorCategory } from './ErrorHandler';
import { ChainConfig } from '../types/chain';

export class ChainManager {
  private static instance: ChainManager;
  private providers: Map<number, ethers.providers.JsonRpcProvider[]> = new Map();
  private currentProviderIndex: Map<number, number> = new Map();
  private chainHealth: Map<number, boolean> = new Map();
  private lastBlockNumber: Map<number, number> = new Map();
  private monitoringInterval: NodeJS.Timeout | null = null;

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly config: ConfigService,
    private readonly errorHandler: ErrorHandler
  ) {
    this.initializeFromConfig();
  }

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    config: ConfigService,
    errorHandler: ErrorHandler
  ): ChainManager {
    if (!ChainManager.instance) {
      ChainManager.instance = new ChainManager(logger, metrics, config, errorHandler);
    }
    return ChainManager.instance;
  }

  private async initializeFromConfig(): Promise<void> {
    try {
      const chains = this.config.get('chains') as Record<number, ChainConfig>;
      
      for (const [chainIdStr, chainConfig] of Object.entries(chains)) {
        const chainId = Number(chainIdStr);
        
        // Initialize providers
        const providers = chainConfig.rpcUrls.map(url => 
          new ethers.providers.JsonRpcProvider(url)
        );
        this.providers.set(chainId, providers);
        this.currentProviderIndex.set(chainId, 0);
        this.chainHealth.set(chainId, true);
        
        // Verify chain connection
        await this.verifyChainConnection(chainId);
        
        this.logger.info(`Initialized chain ${chainId}`, {
          chain: chainConfig.name,
          providers: chainConfig.rpcUrls.length
        });
      }

      // Start monitoring
      this.startChainMonitoring();
      
    } catch (error) {
      this.logger.error('Error initializing chain manager', error);
      throw error;
    }
  }

  private async verifyChainConnection(chainId: number): Promise<void> {
    const provider = this.getProvider(chainId);
    try {
      const network = await provider.getNetwork();
      if (network.chainId !== chainId) {
        throw new Error(`Chain ID mismatch: expected ${chainId}, got ${network.chainId}`);
      }
      
      const blockNumber = await provider.getBlockNumber();
      this.lastBlockNumber.set(chainId, blockNumber);
      
    } catch (error) {
      await this.errorHandler.handleError({
        error: error as Error,
        severity: ErrorSeverity.HIGH,
        category: ErrorCategory.NETWORK,
        context: {
          operation: 'verifyChainConnection',
          chainId,
          timestamp: new Date()
        },
        retryable: true
      });
      throw error;
    }
  }

  public getProvider(chainId: number): ethers.providers.Provider {
    const providers = this.providers.get(chainId);
    if (!providers || providers.length === 0) {
      throw new Error(`No providers available for chain ${chainId}`);
    }

    const currentIndex = this.currentProviderIndex.get(chainId) || 0;
    return providers[currentIndex];
  }

  public async rotateProvider(chainId: number): Promise<void> {
    const providers = this.providers.get(chainId);
    if (!providers || providers.length <= 1) return;

    const currentIndex = this.currentProviderIndex.get(chainId) || 0;
    const nextIndex = (currentIndex + 1) % providers.length;
    
    try {
      // Verify new provider before switching
      const provider = providers[nextIndex];
      await provider.getBlockNumber();
      
      this.currentProviderIndex.set(chainId, nextIndex);
      this.logger.info(`Rotated provider for chain ${chainId}`, {
        previousIndex: currentIndex,
        newIndex: nextIndex
      });
      
    } catch (error) {
      this.logger.error(`Failed to rotate provider for chain ${chainId}`, error);
      // Try next provider
      await this.rotateProvider(chainId);
    }
  }

  public getChainConfig(chainId: number): ChainConfig {
    const chains = this.config.get('chains') as Record<number, ChainConfig>;
    const chainConfig = chains[chainId];
    if (!chainConfig) {
      throw new Error(`No configuration found for chain ${chainId}`);
    }
    return chainConfig;
  }

  public isChainHealthy(chainId: number): boolean {
    return this.chainHealth.get(chainId) || false;
  }

  private async checkChainHealth(chainId: number): Promise<boolean> {
    try {
      const provider = this.getProvider(chainId);
      const [blockNumber, gasPrice] = await Promise.all([
        provider.getBlockNumber(),
        provider.getGasPrice()
      ]);

      const lastBlock = this.lastBlockNumber.get(chainId) || 0;
      const chainConfig = this.getChainConfig(chainId);
      
      // Check if chain is healthy
      const isHealthy = (
        blockNumber > lastBlock && // Blocks are progressing
        gasPrice.lte(ethers.utils.parseUnits(
          chainConfig.gasConfig.maxGwei.toString(),
          'gwei'
        ))
      );

      // Update metrics
      this.metrics.setGauge('chain_health', isHealthy ? 1 : 0, { chainId: chainId.toString() });
      this.metrics.setGauge('chain_gas_price', parseFloat(ethers.utils.formatUnits(gasPrice, 'gwei')), { chainId: chainId.toString() });
      this.metrics.setGauge('chain_block_number', blockNumber, { chainId: chainId.toString() });

      // Update state
      this.lastBlockNumber.set(chainId, blockNumber);
      this.chainHealth.set(chainId, isHealthy);

      return isHealthy;
      
    } catch (error) {
      this.logger.error(`Chain ${chainId} health check failed`, error);
      this.chainHealth.set(chainId, false);
      return false;
    }
  }

  private startChainMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(async () => {
      const chains = this.config.get('chains') as Record<number, ChainConfig>;
      
      for (const chainId of Object.keys(chains).map(Number)) {
        try {
          const wasHealthy = this.isChainHealthy(chainId);
          const isHealthy = await this.checkChainHealth(chainId);

          // Handle health state changes
          if (wasHealthy && !isHealthy) {
            this.logger.warn(`Chain ${chainId} became unhealthy`);
            await this.rotateProvider(chainId);
          } else if (!wasHealthy && isHealthy) {
            this.logger.info(`Chain ${chainId} recovered`);
          }
          
        } catch (error) {
          this.logger.error(`Error monitoring chain ${chainId}`, error);
        }
      }
    }, 5000); // Check every 5 seconds
  }

  public async getGasPrice(chainId: number): Promise<ethers.BigNumber> {
    try {
      const provider = this.getProvider(chainId);
      const gasPrice = await provider.getGasPrice();
      
      // Apply chain-specific multiplier
      const chainConfig = this.getChainConfig(chainId);
      return gasPrice.mul(
        Math.floor(chainConfig.gasConfig.estimateMultiplier * 100)
      ).div(100);
      
    } catch (error) {
      this.logger.error(`Error getting gas price for chain ${chainId}`, error);
      throw error;
    }
  }

  public async estimateGas(
    chainId: number,
    transaction: ethers.providers.TransactionRequest
  ): Promise<ethers.BigNumber> {
    try {
      const provider = this.getProvider(chainId);
      const estimate = await provider.estimateGas(transaction);
      
      // Apply chain-specific multiplier
      const chainConfig = this.getChainConfig(chainId);
      return estimate.mul(
        Math.floor(chainConfig.gasConfig.estimateMultiplier * 100)
      ).div(100);
      
    } catch (error) {
      this.logger.error(`Error estimating gas for chain ${chainId}`, error);
      throw error;
    }
  }

  public stop(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }

  public getActiveChains(): number[] {
    return Array.from(this.chainHealth.entries())
      .filter(([_, isHealthy]) => isHealthy)
      .map(([chainId]) => chainId);
  }

  public async waitForHealthyChain(chainId: number, timeoutMs: number = 30000): Promise<boolean> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
      if (this.isChainHealthy(chainId)) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    return false;
  }
}
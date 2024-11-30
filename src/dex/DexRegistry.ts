import { ethers } from 'ethers';
import { Logger } from '../services/Logger';
import { MetricsCollector } from '../services/MetricsCollector';
import { ErrorHandler } from '../services/ErrorHandler';
import { ChainManager } from '../services/ChainManager';
import { DexAdapter } from './DexAdapter';
import { UniswapV2Adapter } from './UniswapV2Adapter';

interface DexConfig {
  name: string;
  type: 'uniswapv2' | 'uniswapv3';
  routerAddress: string;
  factoryAddress: string;
  initCodeHash: string;
  enabled: boolean;
}

export class DexRegistry {
  private static instance: DexRegistry;
  private adapters: Map<string, DexAdapter> = new Map(); // chainId:dexName -> adapter
  private dexConfigs: Map<number, DexConfig[]> = new Map(); // chainId -> configs

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly errorHandler: ErrorHandler,
    private readonly chainManager: ChainManager
  ) {
    this.initializeRegistry();
  }

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    errorHandler: ErrorHandler,
    chainManager: ChainManager
  ): DexRegistry {
    if (!DexRegistry.instance) {
      DexRegistry.instance = new DexRegistry(
        logger,
        metrics,
        errorHandler,
        chainManager
      );
    }
    return DexRegistry.instance;
  }

  private async initializeRegistry(): Promise<void> {
    try {
      // Load configurations for each chain
      const activeChains = this.chainManager.getActiveChains();
      
      for (const chainId of activeChains) {
        const chainConfig = this.chainManager.getChainConfig(chainId);
        const provider = this.chainManager.getProvider(chainId);

        // Initialize V2 DEXes
        for (const dex of chainConfig.dexes.uniswapV2Like) {
          await this.addDexAdapter({
            name: dex.name,
            type: 'uniswapv2',
            routerAddress: dex.routerAddress,
            factoryAddress: dex.factoryAddress,
            initCodeHash: dex.initCodeHash,
            enabled: true
          }, chainId, provider);
        }

        // Initialize V3 DEXes (to be implemented)
        // for (const dex of chainConfig.dexes.uniswapV3) { ... }
      }

      this.logger.info('DEX registry initialized', {
        adapterCount: this.adapters.size,
        chains: activeChains
      });

    } catch (error) {
      this.logger.error('Error initializing DEX registry', error);
      throw error;
    }
  }

  private async addDexAdapter(
    config: DexConfig,
    chainId: number,
    provider: ethers.providers.Provider
  ): Promise<void> {
    try {
      const key = this.getDexKey(chainId, config.name);
      
      // Create appropriate adapter based on type
      let adapter: DexAdapter;
      switch (config.type) {
        case 'uniswapv2':
          adapter = new UniswapV2Adapter(
            this.logger,
            this.metrics,
            this.errorHandler,
            provider,
            chainId,
            config.routerAddress,
            config.factoryAddress,
            config.initCodeHash
          );
          break;
        // Add other DEX types here
        default:
          throw new Error(`Unsupported DEX type: ${config.type}`);
      }

      // Store adapter and config
      this.adapters.set(key, adapter);
      
      const configs = this.dexConfigs.get(chainId) || [];
      configs.push(config);
      this.dexConfigs.set(chainId, configs);

      // Update metrics
      this.metrics.incrementCounter('dex_adapters_total', {
        chainId: chainId.toString(),
        type: config.type
      });

    } catch (error) {
      this.logger.error(`Error adding DEX adapter: ${config.name}`, error);
      throw error;
    }
  }

  public getAdapter(chainId: number, dexName: string): DexAdapter {
    const key = this.getDexKey(chainId, dexName);
    const adapter = this.adapters.get(key);
    
    if (!adapter) {
      throw new Error(`No adapter found for DEX ${dexName} on chain ${chainId}`);
    }

    return adapter;
  }

  public getEnabledDexes(chainId: number): string[] {
    const configs = this.dexConfigs.get(chainId) || [];
    return configs
      .filter(config => config.enabled)
      .map(config => config.name);
  }

  public async getQuotes(
    chainId: number,
    tokenIn: string,
    tokenOut: string,
    amountIn: ethers.BigNumber
  ): Promise<Record<string, ethers.BigNumber>> {
    const quotes: Record<string, ethers.BigNumber> = {};
    const enabledDexes = this.getEnabledDexes(chainId);

    await Promise.all(
      enabledDexes.map(async (dexName) => {
        try {
          const adapter = this.getAdapter(chainId, dexName);
          const amountOut = await adapter.getAmountOut({
            tokenIn,
            tokenOut,
            amountIn,
            amountOutMin: ethers.constants.Zero,
            to: ethers.constants.AddressZero,
            deadline: Math.floor(Date.now() / 1000) + 3600
          });

          quotes[dexName] = amountOut;

        } catch (error) {
          this.logger.warn(`Error getting quote from ${dexName}`, error);
        }
      })
    );

    return quotes;
  }

  public async getBestQuote(
    chainId: number,
    tokenIn: string,
    tokenOut: string,
    amountIn: ethers.BigNumber
  ): Promise<{
    dex: string;
    amountOut: ethers.BigNumber;
  } | null> {
    try {
      const quotes = await this.getQuotes(
        chainId,
        tokenIn,
        tokenOut,
        amountIn
      );

      let bestDex = '';
      let bestAmount = ethers.constants.Zero;

      for (const [dex, amount] of Object.entries(quotes)) {
        if (amount.gt(bestAmount)) {
          bestDex = dex;
          bestAmount = amount;
        }
      }

      if (bestDex === '') {
        return null;
      }

      return {
        dex: bestDex,
        amountOut: bestAmount
      };

    } catch (error) {
      this.logger.error('Error getting best quote', error);
      return null;
    }
  }

  public async calculatePriceImpacts(
    chainId: number,
    tokenIn: string,
    tokenOut: string,
    amounts: ethers.BigNumber[]
  ): Promise<Record<string, number[]>> {
    const impacts: Record<string, number[]> = {};
    const enabledDexes = this.getEnabledDexes(chainId);

    await Promise.all(
      enabledDexes.map(async (dexName) => {
        try {
          const adapter = this.getAdapter(chainId, dexName);
          const dexImpacts = await Promise.all(
            amounts.map(amount =>
              adapter.calculatePriceImpact({
                tokenIn,
                tokenOut,
                amountIn: amount,
                amountOutMin: ethers.constants.Zero,
                to: ethers.constants.AddressZero,
                deadline: Math.floor(Date.now() / 1000) + 3600
              }).then(result => result.impact)
            )
          );

          impacts[dexName] = dexImpacts;

        } catch (error) {
          this.logger.warn(`Error calculating impact for ${dexName}`, error);
        }
      })
    );

    return impacts;
  }

  private getDexKey(chainId: number, dexName: string): string {
    return `${chainId}:${dexName.toLowerCase()}`;
  }

  public getAdapterCount(): number {
    return this.adapters.size;
  }

  public getDexCount(chainId: number): number {
    return this.dexConfigs.get(chainId)?.length || 0;
  }
}
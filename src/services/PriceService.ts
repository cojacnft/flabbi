import { ethers } from 'ethers';
import { Logger } from './Logger';
import { MetricsCollector } from './MetricsCollector';
import { ChainManager } from './ChainManager';
import { ErrorHandler, ErrorSeverity, ErrorCategory } from './ErrorHandler';

interface PriceData {
  price: number;
  timestamp: number;
  confidence: number;
}

interface TokenMetadata {
  symbol: string;
  decimals: number;
  chainId: number;
}

export class PriceService {
  private static instance: PriceService;
  private prices: Map<string, PriceData> = new Map(); // tokenAddress -> PriceData
  private tokenMetadata: Map<string, TokenMetadata> = new Map(); // tokenAddress -> TokenMetadata
  private updateInterval: NodeJS.Timeout | null = null;
  private readonly PRICE_VALIDITY_MS = 30000; // 30 seconds
  private readonly CONFIDENCE_THRESHOLD = 0.8;

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly chainManager: ChainManager,
    private readonly errorHandler: ErrorHandler
  ) {
    this.initializePriceTracking();
  }

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    chainManager: ChainManager,
    errorHandler: ErrorHandler
  ): PriceService {
    if (!PriceService.instance) {
      PriceService.instance = new PriceService(
        logger,
        metrics,
        chainManager,
        errorHandler
      );
    }
    return PriceService.instance;
  }

  private initializePriceTracking(): void {
    this.updateInterval = setInterval(
      () => this.updateAllPrices(),
      5000 // Update every 5 seconds
    );
  }

  private async updateAllPrices(): Promise<void> {
    const activeChains = this.chainManager.getActiveChains();
    
    await Promise.all(
      activeChains.map(chainId => this.updateChainPrices(chainId))
    );
  }

  private async updateChainPrices(chainId: number): Promise<void> {
    try {
      const chainConfig = this.chainManager.getChainConfig(chainId);
      const provider = this.chainManager.getProvider(chainId);

      // Update stablecoin prices
      await Promise.all(
        chainConfig.stablecoins.map(token =>
          this.updateTokenPrice(token.address, chainId, provider)
        )
      );

      // Update metrics
      this.metrics.setGauge('price_updates', 1, {
        chainId: chainId.toString(),
        status: 'success'
      });

    } catch (error) {
      this.metrics.setGauge('price_updates', 1, {
        chainId: chainId.toString(),
        status: 'failed'
      });

      await this.errorHandler.handleError({
        error: error as Error,
        severity: ErrorSeverity.MEDIUM,
        category: ErrorCategory.PRICE_DATA,
        context: {
          operation: 'updateChainPrices',
          chainId,
          timestamp: new Date()
        },
        retryable: true
      });
    }
  }

  private async updateTokenPrice(
    tokenAddress: string,
    chainId: number,
    provider: ethers.providers.Provider
  ): Promise<void> {
    try {
      // Get price from multiple sources for better accuracy
      const prices = await Promise.all([
        this.getPriceFromDex(tokenAddress, chainId, provider),
        this.getPriceFromOracle(tokenAddress, chainId, provider),
        this.getPriceFromApi(tokenAddress, chainId)
      ]);

      // Filter out invalid prices
      const validPrices = prices.filter(p => p && p > 0);
      
      if (validPrices.length === 0) {
        throw new Error(`No valid prices found for ${tokenAddress}`);
      }

      // Calculate median price
      const sortedPrices = [...validPrices].sort((a, b) => a - b);
      const medianPrice = sortedPrices[Math.floor(sortedPrices.length / 2)];

      // Calculate confidence based on price variance
      const variance = this.calculateVariance(validPrices);
      const confidence = Math.max(0, 1 - variance);

      // Update price data
      this.prices.set(this.getPriceKey(tokenAddress, chainId), {
        price: medianPrice,
        timestamp: Date.now(),
        confidence
      });

      // Update metrics
      this.metrics.setGauge('token_price', medianPrice, {
        token: tokenAddress,
        chainId: chainId.toString()
      });
      this.metrics.setGauge('price_confidence', confidence, {
        token: tokenAddress,
        chainId: chainId.toString()
      });

    } catch (error) {
      this.logger.error(`Error updating price for ${tokenAddress}`, error);
    }
  }

  private async getPriceFromDex(
    tokenAddress: string,
    chainId: number,
    provider: ethers.providers.Provider
  ): Promise<number | null> {
    try {
      const chainConfig = this.chainManager.getChainConfig(chainId);
      
      // Find USDC pair if available
      const usdcAddress = chainConfig.stablecoins.find(
        t => t.symbol === 'USDC'
      )?.address;

      if (!usdcAddress) return null;

      // Get price from largest liquidity pool
      // This is a simplified version - you'll need to implement actual DEX queries
      return null;

    } catch (error) {
      this.logger.error(`Error getting DEX price for ${tokenAddress}`, error);
      return null;
    }
  }

  private async getPriceFromOracle(
    tokenAddress: string,
    chainId: number,
    provider: ethers.providers.Provider
  ): Promise<number | null> {
    try {
      // Implement Chainlink or other oracle price fetching
      return null;
    } catch (error) {
      this.logger.error(`Error getting oracle price for ${tokenAddress}`, error);
      return null;
    }
  }

  private async getPriceFromApi(
    tokenAddress: string,
    chainId: number
  ): Promise<number | null> {
    try {
      // Implement API price fetching (e.g., CoinGecko)
      return null;
    } catch (error) {
      this.logger.error(`Error getting API price for ${tokenAddress}`, error);
      return null;
    }
  }

  private calculateVariance(prices: number[]): number {
    if (prices.length < 2) return 0;
    
    const mean = prices.reduce((a, b) => a + b) / prices.length;
    const squaredDiffs = prices.map(p => Math.pow(p - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b) / prices.length;
    
    return Math.sqrt(variance) / mean; // Return relative standard deviation
  }

  private getPriceKey(tokenAddress: string, chainId: number): string {
    return `${chainId}:${tokenAddress.toLowerCase()}`;
  }

  public async getPrice(
    tokenAddress: string,
    chainId: number
  ): Promise<number | null> {
    const priceKey = this.getPriceKey(tokenAddress, chainId);
    const priceData = this.prices.get(priceKey);

    if (!priceData) {
      return null;
    }

    // Check if price is still valid
    if (Date.now() - priceData.timestamp > this.PRICE_VALIDITY_MS) {
      return null;
    }

    // Check confidence
    if (priceData.confidence < this.CONFIDENCE_THRESHOLD) {
      return null;
    }

    return priceData.price;
  }

  public async convertToUSD(
    amount: ethers.BigNumber,
    tokenAddress: string,
    chainId: number
  ): Promise<number> {
    try {
      // Get token metadata
      const metadata = await this.getTokenMetadata(tokenAddress, chainId);
      if (!metadata) {
        throw new Error(`No metadata for token ${tokenAddress}`);
      }

      // Get price
      const price = await this.getPrice(tokenAddress, chainId);
      if (!price) {
        throw new Error(`No price for token ${tokenAddress}`);
      }

      // Convert to USD
      return parseFloat(ethers.utils.formatUnits(amount, metadata.decimals)) * price;

    } catch (error) {
      this.logger.error(`Error converting to USD`, error);
      return 0;
    }
  }

  private async getTokenMetadata(
    tokenAddress: string,
    chainId: number
  ): Promise<TokenMetadata | null> {
    const key = this.getPriceKey(tokenAddress, chainId);
    
    if (this.tokenMetadata.has(key)) {
      return this.tokenMetadata.get(key) || null;
    }

    try {
      const provider = this.chainManager.getProvider(chainId);
      const tokenContract = new ethers.Contract(
        tokenAddress,
        [
          'function symbol() view returns (string)',
          'function decimals() view returns (uint8)'
        ],
        provider
      );

      const [symbol, decimals] = await Promise.all([
        tokenContract.symbol(),
        tokenContract.decimals()
      ]);

      const metadata = { symbol, decimals, chainId };
      this.tokenMetadata.set(key, metadata);

      return metadata;

    } catch (error) {
      this.logger.error(`Error getting token metadata`, error);
      return null;
    }
  }

  public stop(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }
}
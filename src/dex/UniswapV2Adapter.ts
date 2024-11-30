import { ethers } from 'ethers';
import { Logger } from '../services/Logger';
import { MetricsCollector } from '../services/MetricsCollector';
import { ErrorHandler } from '../services/ErrorHandler';
import { DexAdapter, SwapParams, PoolInfo, PriceImpact } from './DexAdapter';

// UniswapV2 specific constants
const MINIMUM_LIQUIDITY = ethers.BigNumber.from(1000);
const FEE_DENOMINATOR = ethers.BigNumber.from(1000);
const FEE_NUMERATOR = ethers.BigNumber.from(997);

export class UniswapV2Adapter extends DexAdapter {
  private readonly ROUTER_ABI = [
    'function getAmountsOut(uint amountIn, address[] memory path) public view returns (uint[] memory amounts)',
    'function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)',
    'function swapExactTokensForTokensSupportingFeeOnTransfer(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)'
  ];

  private readonly FACTORY_ABI = [
    'function getPair(address tokenA, address tokenB) external view returns (address pair)',
    'function allPairs(uint) external view returns (address pair)',
    'function allPairsLength() external view returns (uint)'
  ];

  private readonly PAIR_ABI = [
    'function token0() external view returns (address)',
    'function token1() external view returns (address)',
    'function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)',
    'function price0CumulativeLast() external view returns (uint)',
    'function price1CumulativeLast() external view returns (uint)'
  ];

  private routerContract: ethers.Contract;
  private factoryContract: ethers.Contract;
  private poolCache: Map<string, PoolInfo> = new Map();
  private lastCacheClean: number = Date.now();
  private readonly CACHE_CLEANUP_INTERVAL = 60000; // 1 minute

  constructor(
    logger: Logger,
    metrics: MetricsCollector,
    errorHandler: ErrorHandler,
    provider: ethers.providers.Provider,
    chainId: number,
    routerAddress: string,
    factoryAddress: string,
    initCodeHash: string
  ) {
    super(
      logger,
      metrics,
      errorHandler,
      provider,
      chainId,
      routerAddress,
      factoryAddress,
      initCodeHash
    );

    this.routerContract = new ethers.Contract(
      routerAddress,
      this.ROUTER_ABI,
      provider
    );

    this.factoryContract = new ethers.Contract(
      factoryAddress,
      this.FACTORY_ABI,
      provider
    );
  }

  public async getPool(
    token0: string,
    token1: string,
    _fee?: number
  ): Promise<PoolInfo> {
    try {
      await this.validateTokens(token0, token1);
      const [tokenA, tokenB] = this.sortTokens(token0, token1);
      const poolKey = this.getPoolKey(tokenA, tokenB);

      // Check cache
      const cachedPool = this.poolCache.get(poolKey);
      if (cachedPool && Date.now() - cachedPool.lastUpdate < 10000) { // 10 seconds cache
        return cachedPool;
      }

      // Get pool address
      const poolAddress = await this.factoryContract.getPair(tokenA, tokenB);
      if (poolAddress === ethers.constants.AddressZero) {
        throw new Error('Pool does not exist');
      }

      // Get pool contract
      const poolContract = new ethers.Contract(
        poolAddress,
        this.PAIR_ABI,
        this.provider
      );

      // Get pool data
      const [reserves, token0Address] = await Promise.all([
        poolContract.getReserves(),
        poolContract.token0()
      ]);

      // Organize reserves based on token order
      const [reserve0, reserve1] = token0Address.toLowerCase() === tokenA.toLowerCase()
        ? [reserves[0], reserves[1]]
        : [reserves[1], reserves[0]];

      const poolInfo: PoolInfo = {
        address: poolAddress,
        token0: tokenA,
        token1: tokenB,
        reserve0: reserve0,
        reserve1: reserve1,
        fee: 3000, // 0.3% for Uniswap V2
        liquidity: this.calculatePoolLiquidity(reserve0, reserve1),
        lastUpdate: Date.now()
      };

      // Update cache
      this.poolCache.set(poolKey, poolInfo);
      this.cleanupCache();

      // Update metrics
      this.updateMetrics('getPool', 'success', {
        liquidity: parseFloat(ethers.utils.formatEther(poolInfo.liquidity))
      });

      return poolInfo;

    } catch (error) {
      this.updateMetrics('getPool', 'failed');
      await this.handleError(error as Error, 'getPool', { token0, token1 });
      throw error;
    }
  }

  public async getAmountOut(params: SwapParams): Promise<ethers.BigNumber> {
    try {
      this.validateSwapParams(params);

      const path = params.path || [params.tokenIn, params.tokenOut];
      const amounts = await this.routerContract.getAmountsOut(
        params.amountIn,
        path
      );

      return amounts[amounts.length - 1];

    } catch (error) {
      await this.handleError(error as Error, 'getAmountOut', params);
      throw error;
    }
  }

  public async buildSwapTransaction(params: SwapParams): Promise<ethers.PopulatedTransaction> {
    try {
      this.validateSwapParams(params);

      const path = params.path || [params.tokenIn, params.tokenOut];
      
      // Check if any tokens have transfer fees
      const hasTransferFee = await this.checkTransferFees(path);

      if (hasTransferFee) {
        return await this.routerContract.populateTransaction
          .swapExactTokensForTokensSupportingFeeOnTransfer(
            params.amountIn,
            params.amountOutMin,
            path,
            params.to,
            params.deadline
          );
      } else {
        return await this.routerContract.populateTransaction
          .swapExactTokensForTokens(
            params.amountIn,
            params.amountOutMin,
            path,
            params.to,
            params.deadline
          );
      }

    } catch (error) {
      await this.handleError(error as Error, 'buildSwapTransaction', params);
      throw error;
    }
  }

  public async calculatePriceImpact(params: SwapParams): Promise<PriceImpact> {
    try {
      const pool = await this.getPool(params.tokenIn, params.tokenOut);
      if (!this.isPoolHealthy(pool)) {
        throw new Error('Unhealthy pool');
      }

      // Calculate spot price
      const spotPrice = this.calculateSpotPrice(
        pool.reserve0,
        pool.reserve1,
        18, // Assuming both tokens have 18 decimals for simplicity
        18
      );

      // Calculate execution price
      const amountOut = await this.getAmountOut(params);
      const executionPrice = parseFloat(ethers.utils.formatEther(amountOut)) /
        parseFloat(ethers.utils.formatEther(params.amountIn));

      // Calculate impact
      const impact = Math.abs((executionPrice - spotPrice) / spotPrice);

      // Calculate fee
      const fee = params.amountIn.mul(3).div(1000); // 0.3% fee

      return {
        impact,
        amountIn: params.amountIn,
        amountOut,
        executionPrice: ethers.utils.parseEther(executionPrice.toFixed(18)),
        fee
      };

    } catch (error) {
      await this.handleError(error as Error, 'calculatePriceImpact', params);
      throw error;
    }
  }

  private getPoolKey(token0: string, token1: string): string {
    return `${this.chainId}:${token0.toLowerCase()}:${token1.toLowerCase()}`;
  }

  private calculatePoolLiquidity(
    reserve0: ethers.BigNumber,
    reserve1: ethers.BigNumber
  ): ethers.BigNumber {
    if (reserve0.isZero() || reserve1.isZero()) {
      return ethers.constants.Zero;
    }

    return ethers.BigNumber.from(
      Math.sqrt(
        reserve0.mul(reserve1).sub(MINIMUM_LIQUIDITY).toString()
      )
    );
  }

  private cleanupCache(): void {
    const now = Date.now();
    if (now - this.lastCacheClean < this.CACHE_CLEANUP_INTERVAL) {
      return;
    }

    for (const [key, pool] of this.poolCache.entries()) {
      if (now - pool.lastUpdate > this.CACHE_CLEANUP_INTERVAL) {
        this.poolCache.delete(key);
      }
    }

    this.lastCacheClean = now;
  }

  private async checkTransferFees(tokens: string[]): Promise<boolean> {
    try {
      // This is a simplified check. In production, you'd want to:
      // 1. Maintain a list of known fee-on-transfer tokens
      // 2. Check token contracts for fee mechanisms
      // 3. Cache results
      return false;
    } catch (error) {
      this.logger.warn('Error checking transfer fees', error);
      return false;
    }
  }

  // Additional helper methods for V2-specific calculations
  private getOptimalAmount(
    reserveIn: ethers.BigNumber,
    reserveOut: ethers.BigNumber
  ): ethers.BigNumber {
    if (reserveIn.isZero() || reserveOut.isZero()) {
      return ethers.constants.Zero;
    }

    const numerator = reserveIn.mul(ethers.BigNumber.from(997));
    const denominator = ethers.BigNumber.from(1000);

    return ethers.BigNumber.from(
      Math.sqrt(
        numerator.mul(numerator)
          .div(denominator)
          .toString()
      )
    ).sub(numerator.div(denominator));
  }

  private getAmountOutForReserves(
    amountIn: ethers.BigNumber,
    reserveIn: ethers.BigNumber,
    reserveOut: ethers.BigNumber
  ): ethers.BigNumber {
    if (amountIn.isZero()) return ethers.constants.Zero;
    if (reserveIn.isZero() || reserveOut.isZero()) return ethers.constants.Zero;

    const amountInWithFee = amountIn.mul(FEE_NUMERATOR);
    const numerator = amountInWithFee.mul(reserveOut);
    const denominator = reserveIn.mul(FEE_DENOMINATOR).add(amountInWithFee);

    return numerator.div(denominator);
  }
}
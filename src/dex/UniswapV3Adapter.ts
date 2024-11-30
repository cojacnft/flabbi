import { ethers } from 'ethers';
import { Logger } from '../services/Logger';
import { MetricsCollector } from '../services/MetricsCollector';
import { ErrorHandler } from '../services/ErrorHandler';
import { DexAdapter, SwapParams, PoolInfo, PriceImpact } from './DexAdapter';
import JSBI from 'jsbi';
import { TickMath, SqrtPriceMath, FullMath, LiquidityMath } from '@uniswap/v3-sdk';

interface Tick {
  index: number;
  liquidityNet: JSBI;
  liquidityGross: JSBI;
}

interface PoolState extends PoolInfo {
  sqrtPriceX96: JSBI;
  tick: number;
  observationIndex: number;
  observationCardinality: number;
  feeProtocol: number;
  unlocked: boolean;
  ticks: Map<number, Tick>;
}

const FEE_TIERS = [100, 500, 3000, 10000]; // 0.01%, 0.05%, 0.3%, 1%

export class UniswapV3Adapter extends DexAdapter {
  private readonly QUOTER_ABI = [
    'function quoteExactInputSingle(address tokenIn, address tokenOut, uint24 fee, uint256 amountIn, uint160 sqrtPriceLimitX96) external returns (uint256 amountOut)',
    'function quoteExactInput(bytes path, uint256 amountIn) external returns (uint256 amountOut)'
  ];

  private readonly POOL_ABI = [
    'function slot0() external view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)',
    'function liquidity() external view returns (uint128)',
    'function ticks(int24 tick) external view returns (uint128 liquidityGross, int128 liquidityNet, uint256 feeGrowthOutside0X128, uint256 feeGrowthOutside1X128, int56 tickCumulativeOutside, uint160 secondsPerLiquidityOutsideX128, uint32 secondsOutside, bool initialized)'
  ];

  private readonly FACTORY_ABI = [
    'function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)'
  ];

  private quoterContract: ethers.Contract;
  private factoryContract: ethers.Contract;
  private poolStates: Map<string, PoolState> = new Map();
  private tickCache: Map<string, Map<number, Tick>> = new Map();
  private lastTickUpdate: Map<string, number> = new Map();
  private readonly TICK_CACHE_DURATION = 60000; // 1 minute

  constructor(
    logger: Logger,
    metrics: MetricsCollector,
    errorHandler: ErrorHandler,
    provider: ethers.providers.Provider,
    chainId: number,
    routerAddress: string,
    factoryAddress: string,
    quoterAddress: string,
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

    this.quoterContract = new ethers.Contract(
      quoterAddress,
      this.QUOTER_ABI,
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
    fee: number = 3000
  ): Promise<PoolInfo> {
    try {
      await this.validateTokens(token0, token1);
      const [tokenA, tokenB] = this.sortTokens(token0, token1);
      const poolKey = this.getPoolKey(tokenA, tokenB, fee);

      // Check cache
      const cachedPool = this.poolStates.get(poolKey);
      if (cachedPool && Date.now() - cachedPool.lastUpdate < 10000) {
        return cachedPool;
      }

      // Get pool address
      const poolAddress = await this.factoryContract.getPool(tokenA, tokenB, fee);
      if (poolAddress === ethers.constants.AddressZero) {
        throw new Error('Pool does not exist');
      }

      // Get pool contract
      const poolContract = new ethers.Contract(
        poolAddress,
        this.POOL_ABI,
        this.provider
      );

      // Get pool data
      const [slot0, liquidity] = await Promise.all([
        poolContract.slot0(),
        poolContract.liquidity()
      ]);

      // Create pool state
      const poolState: PoolState = {
        address: poolAddress,
        token0: tokenA,
        token1: tokenB,
        reserve0: ethers.constants.Zero, // V3 doesn't use reserves directly
        reserve1: ethers.constants.Zero,
        fee,
        liquidity: ethers.BigNumber.from(liquidity),
        lastUpdate: Date.now(),
        sqrtPriceX96: JSBI.BigInt(slot0.sqrtPriceX96.toString()),
        tick: slot0.tick,
        observationIndex: slot0.observationIndex,
        observationCardinality: slot0.observationCardinality,
        feeProtocol: slot0.feeProtocol,
        unlocked: slot0.unlocked,
        ticks: await this.getTicksAround(poolContract, slot0.tick)
      };

      // Update cache
      this.poolStates.set(poolKey, poolState);
      this.updateTickCache(poolKey, poolState.ticks);

      // Update metrics
      this.updateMetrics('getPool', 'success', {
        liquidity: parseFloat(ethers.utils.formatEther(poolState.liquidity))
      });

      return poolState;

    } catch (error) {
      this.updateMetrics('getPool', 'failed');
      await this.handleError(error as Error, 'getPool', { token0, token1, fee });
      throw error;
    }
  }

  public async getAmountOut(params: SwapParams): Promise<ethers.BigNumber> {
    try {
      this.validateSwapParams(params);

      if (params.path) {
        // Multi-hop path
        const encodedPath = this.encodePath(params.path, params.fee || 3000);
        return await this.quoterContract.callStatic.quoteExactInput(
          encodedPath,
          params.amountIn
        );
      } else {
        // Single-hop path
        return await this.quoterContract.callStatic.quoteExactInputSingle(
          params.tokenIn,
          params.tokenOut,
          params.fee || 3000,
          params.amountIn,
          0 // No price limit
        );
      }

    } catch (error) {
      await this.handleError(error as Error, 'getAmountOut', params);
      throw error;
    }
  }

  public async buildSwapTransaction(params: SwapParams): Promise<ethers.PopulatedTransaction> {
    try {
      this.validateSwapParams(params);

      const exactInputParams = {
        tokenIn: params.tokenIn,
        tokenOut: params.tokenOut,
        fee: params.fee || 3000,
        recipient: params.to,
        deadline: params.deadline,
        amountIn: params.amountIn,
        amountOutMinimum: params.amountOutMin,
        sqrtPriceLimitX96: 0 // No price limit
      };

      return {
        to: this.routerAddress,
        data: this.routerContract.interface.encodeFunctionData(
          'exactInputSingle',
          [exactInputParams]
        ),
        value: '0x00'
      };

    } catch (error) {
      await this.handleError(error as Error, 'buildSwapTransaction', params);
      throw error;
    }
  }

  public async calculatePriceImpact(params: SwapParams): Promise<PriceImpact> {
    try {
      const pool = await this.getPool(
        params.tokenIn,
        params.tokenOut,
        params.fee || 3000
      ) as PoolState;

      if (!this.isPoolHealthy(pool)) {
        throw new Error('Unhealthy pool');
      }

      // Calculate spot price from current tick
      const spotPrice = this.getPriceFromSqrtPrice(pool.sqrtPriceX96);

      // Simulate swap to get execution price
      const amountOut = await this.getAmountOut(params);
      const executionPrice = this.calculateExecutionPrice(
        params.amountIn,
        amountOut,
        pool.token0,
        pool.token1,
        params.tokenIn
      );

      // Calculate impact
      const impact = Math.abs((executionPrice - spotPrice) / spotPrice);

      // Calculate fee
      const fee = this.calculateFee(params.amountIn, pool.fee);

      return {
        impact,
        amountIn: params.amountIn,
        amountOut,
        executionPrice: ethers.utils.parseUnits(
          executionPrice.toFixed(18),
          18
        ),
        fee
      };

    } catch (error) {
      await this.handleError(error as Error, 'calculatePriceImpact', params);
      throw error;
    }
  }

  private async getTicksAround(
    poolContract: ethers.Contract,
    currentTick: number,
    range: number = 50
  ): Promise<Map<number, Tick>> {
    const ticks = new Map<number, Tick>();
    const spacing = this.getTickSpacing(poolContract.fee);

    // Get initialized ticks around current tick
    for (let i = -range; i <= range; i++) {
      const tickIndex = Math.floor(currentTick / spacing) * spacing + i * spacing;
      try {
        const tickData = await poolContract.ticks(tickIndex);
        if (tickData.initialized) {
          ticks.set(tickIndex, {
            index: tickIndex,
            liquidityNet: JSBI.BigInt(tickData.liquidityNet.toString()),
            liquidityGross: JSBI.BigInt(tickData.liquidityGross.toString())
          });
        }
      } catch (error) {
        this.logger.warn(`Error fetching tick ${tickIndex}`, error);
      }
    }

    return ticks;
  }

  private getTickSpacing(fee: number): number {
    switch (fee) {
      case 100: return 1;    // 0.01%
      case 500: return 10;   // 0.05%
      case 3000: return 60;  // 0.3%
      case 10000: return 200; // 1%
      default: return 60;
    }
  }

  private updateTickCache(poolKey: string, ticks: Map<number, Tick>): void {
    this.tickCache.set(poolKey, ticks);
    this.lastTickUpdate.set(poolKey, Date.now());

    // Cleanup old cache entries
    for (const [key, lastUpdate] of this.lastTickUpdate.entries()) {
      if (Date.now() - lastUpdate > this.TICK_CACHE_DURATION) {
        this.tickCache.delete(key);
        this.lastTickUpdate.delete(key);
      }
    }
  }

  private getPoolKey(token0: string, token1: string, fee: number): string {
    return `${this.chainId}:${token0.toLowerCase()}:${token1.toLowerCase()}:${fee}`;
  }

  private encodePath(
    path: string[],
    fee: number
  ): string {
    if (path.length < 2) {
      throw new Error('Invalid path');
    }

    const fees = Array(path.length - 1).fill(fee);
    const encoded = ethers.utils.solidityPack(
      ['address', 'uint24', 'address'],
      [path[0], fees[0], path[1]]
    );

    let result = encoded;
    for (let i = 2; i < path.length; i++) {
      result = ethers.utils.solidityPack(
        ['bytes', 'uint24', 'address'],
        [result, fees[i - 1], path[i]]
      );
    }

    return result;
  }

  private getPriceFromSqrtPrice(sqrtPriceX96: JSBI): number {
    const price = JSBI.divide(
      JSBI.multiply(sqrtPriceX96, sqrtPriceX96),
      JSBI.BigInt('2').pow(JSBI.BigInt('192'))
    );
    return parseFloat(price.toString()) / 2 ** 192;
  }

  private calculateExecutionPrice(
    amountIn: ethers.BigNumber,
    amountOut: ethers.BigNumber,
    token0: string,
    token1: string,
    tokenIn: string
  ): number {
    const isToken0In = tokenIn.toLowerCase() === token0.toLowerCase();
    const amountInFloat = parseFloat(ethers.utils.formatEther(amountIn));
    const amountOutFloat = parseFloat(ethers.utils.formatEther(amountOut));

    return isToken0In
      ? amountOutFloat / amountInFloat
      : amountInFloat / amountOutFloat;
  }

  private calculateFee(
    amount: ethers.BigNumber,
    feeBips: number
  ): ethers.BigNumber {
    return amount.mul(feeBips).div(1000000); // feeBips is in 0.0001%
  }

  // Additional V3-specific calculations
  private calculateNextSqrtPrice(
    sqrtPriceX96: JSBI,
    liquidity: JSBI,
    amount: JSBI,
    zeroForOne: boolean
  ): JSBI {
    if (zeroForOne) {
      return SqrtPriceMath.getNextSqrtPriceFromAmount0RoundingUp(
        sqrtPriceX96,
        liquidity,
        amount,
        true
      );
    } else {
      return SqrtPriceMath.getNextSqrtPriceFromAmount1RoundingDown(
        sqrtPriceX96,
        liquidity,
        amount,
        true
      );
    }
  }

  private calculateTickFromPrice(price: number): number {
    const sqrtPrice = Math.sqrt(price);
    return TickMath.getTickAtSqrtRatio(
      JSBI.BigInt(sqrtPrice * 2 ** 96)
    );
  }
}
import { ethers } from 'ethers';
import { Logger } from './Logger';
import { MetricsCollector } from './MetricsCollector';
import { ChainManager } from './ChainManager';
import { ErrorHandler, ErrorSeverity, ErrorCategory } from './ErrorHandler';
import { PriceService } from './PriceService';
import { ChainOpportunityFinder } from './ChainOpportunityFinder';
import { PerformanceOptimizer } from './PerformanceOptimizer';
import { ChainOpportunity, ChainTransaction } from '../types/chain';

interface ExecutionResult {
  success: boolean;
  profit?: string;
  gasUsed?: string;
  error?: string;
  txHash?: string;
}

interface ExecutionContext {
  chainId: number;
  blockNumber: number;
  gasPrice: string;
  nonce: number;
  deadline: number;
}

export class ArbitrageExecutor {
  private static instance: ArbitrageExecutor;
  private executingOpportunities: Set<string> = new Set();
  private lastExecutionTime: Map<number, number> = new Map();
  private successfulStrategies: Map<string, number> = new Map();
  private readonly MIN_EXECUTION_INTERVAL = 2000; // 2 seconds between executions per chain

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly chainManager: ChainManager,
    private readonly errorHandler: ErrorHandler,
    private readonly priceService: PriceService,
    private readonly opportunityFinder: ChainOpportunityFinder,
    private readonly performanceOptimizer: PerformanceOptimizer
  ) {}

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    chainManager: ChainManager,
    errorHandler: ErrorHandler,
    priceService: PriceService,
    opportunityFinder: ChainOpportunityFinder,
    performanceOptimizer: PerformanceOptimizer
  ): ArbitrageExecutor {
    if (!ArbitrageExecutor.instance) {
      ArbitrageExecutor.instance = new ArbitrageExecutor(
        logger,
        metrics,
        chainManager,
        errorHandler,
        priceService,
        opportunityFinder,
        performanceOptimizer
      );
    }
    return ArbitrageExecutor.instance;
  }

  public async executeOpportunity(
    opportunity: ChainOpportunity
  ): Promise<ExecutionResult> {
    const opportunityId = this.getOpportunityId(opportunity);
    
    try {
      // Check if already executing
      if (this.executingOpportunities.has(opportunityId)) {
        return {
          success: false,
          error: 'Already executing this opportunity'
        };
      }

      // Rate limiting per chain
      const lastExecution = this.lastExecutionTime.get(opportunity.chainId) || 0;
      const timeSinceLastExecution = Date.now() - lastExecution;
      if (timeSinceLastExecution < this.MIN_EXECUTION_INTERVAL) {
        return {
          success: false,
          error: 'Rate limit exceeded'
        };
      }

      // Verify chain health
      if (!this.chainManager.isChainHealthy(opportunity.chainId)) {
        return {
          success: false,
          error: 'Chain unhealthy'
        };
      }

      // Start execution
      this.executingOpportunities.add(opportunityId);
      this.lastExecutionTime.set(opportunity.chainId, Date.now());

      // Get execution context
      const context = await this.prepareExecutionContext(opportunity);

      // Validate opportunity is still profitable
      if (!await this.validateOpportunity(opportunity, context)) {
        return {
          success: false,
          error: 'Opportunity no longer profitable'
        };
      }

      // Execute flash loan
      const result = await this.executeFlashLoan(opportunity, context);

      // Update metrics and optimization data
      await this.updateMetrics(opportunity, result);

      return result;

    } catch (error) {
      await this.handleExecutionError(error as Error, opportunity);
      return {
        success: false,
        error: (error as Error).message
      };
    } finally {
      this.executingOpportunities.delete(opportunityId);
    }
  }

  private getOpportunityId(opportunity: ChainOpportunity): string {
    return `${opportunity.chainId}:${opportunity.tokenIn}:${opportunity.path.map(p => p.pool).join(':')}`;
  }

  private async prepareExecutionContext(
    opportunity: ChainOpportunity
  ): Promise<ExecutionContext> {
    const provider = this.chainManager.getProvider(opportunity.chainId);
    const [blockNumber, gasPrice] = await Promise.all([
      provider.getBlockNumber(),
      this.chainManager.getGasPrice(opportunity.chainId)
    ]);

    const wallet = new ethers.Wallet(
      process.env.PRIVATE_KEY!,
      provider
    );
    const nonce = await wallet.getTransactionCount();
    const deadline = Math.floor(Date.now() / 1000) + 300; // 5 minutes

    return {
      chainId: opportunity.chainId,
      blockNumber,
      gasPrice: gasPrice.toString(),
      nonce,
      deadline
    };
  }

  private async validateOpportunity(
    opportunity: ChainOpportunity,
    context: ExecutionContext
  ): Promise<boolean> {
    try {
      // Get current prices
      const currentPrices = await Promise.all(
        opportunity.path.map(async step => ({
          tokenIn: await this.priceService.getPrice(step.tokenIn, opportunity.chainId),
          tokenOut: await this.priceService.getPrice(step.tokenOut, opportunity.chainId)
        }))
      );

      // Calculate current profit
      const expectedProfit = ethers.BigNumber.from(opportunity.expectedProfit);
      const currentProfit = this.calculateCurrentProfit(opportunity, currentPrices);

      // Get minimum profit threshold
      const chainConfig = this.chainManager.getChainConfig(opportunity.chainId);
      const minProfitUSD = chainConfig.minProfit.usd;

      // Calculate gas cost
      const gasCost = ethers.BigNumber.from(context.gasPrice)
        .mul(ethers.BigNumber.from(opportunity.gasEstimate));
      const gasCostUSD = await this.priceService.convertToUSD(
        gasCost,
        'ETH',
        opportunity.chainId
      );

      // Validate profitability
      const currentProfitUSD = await this.priceService.convertToUSD(
        currentProfit,
        opportunity.tokenIn,
        opportunity.chainId
      );

      const netProfitUSD = currentProfitUSD - gasCostUSD;

      // Update metrics
      this.metrics.setGauge('validation_profit_usd', netProfitUSD, {
        chainId: opportunity.chainId.toString()
      });

      return (
        netProfitUSD >= minProfitUSD &&
        currentProfit.gte(expectedProfit.mul(90).div(100)) // Within 10% of expected profit
      );

    } catch (error) {
      this.logger.error('Error validating opportunity', error);
      return false;
    }
  }

  private calculateCurrentProfit(
    opportunity: ChainOpportunity,
    currentPrices: Array<{ tokenIn: number | null; tokenOut: number | null }>
  ): ethers.BigNumber {
    // Implement profit calculation based on current prices
    return ethers.BigNumber.from(0);
  }

  private async executeFlashLoan(
    opportunity: ChainOpportunity,
    context: ExecutionContext
  ): Promise<ExecutionResult> {
    try {
      // Get provider and wallet
      const provider = this.chainManager.getProvider(opportunity.chainId);
      const wallet = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);

      // Get flash loan contract
      const flashLoanContract = await this.getFlashLoanContract(
        opportunity.protocol,
        opportunity.chainId,
        wallet
      );

      // Prepare transaction
      const tx = await this.prepareFlashLoanTransaction(
        opportunity,
        context,
        flashLoanContract
      );

      // Get optimal gas settings
      const gasSettings = await this.getOptimalGasSettings(
        opportunity.chainId,
        context
      );

      // Send transaction
      const response = await wallet.sendTransaction({
        ...tx,
        ...gasSettings
      });

      // Wait for confirmation
      const receipt = await this.waitForConfirmation(
        response,
        opportunity.chainId
      );

      // Parse result
      return this.parseExecutionResult(receipt, opportunity);

    } catch (error) {
      throw this.enhanceError(error as Error, opportunity);
    }
  }

  private async getFlashLoanContract(
    protocol: string,
    chainId: number,
    wallet: ethers.Signer
  ): Promise<ethers.Contract> {
    const chainConfig = this.chainManager.getChainConfig(chainId);
    const flashLoanConfig = chainConfig.flashLoanProviders[protocol];

    if (!flashLoanConfig) {
      throw new Error(`Flash loan provider ${protocol} not configured for chain ${chainId}`);
    }

    // Get appropriate ABI based on protocol
    const abi = await this.getFlashLoanAbi(protocol);

    return new ethers.Contract(
      flashLoanConfig.poolAddress,
      abi,
      wallet
    );
  }

  private async getFlashLoanAbi(protocol: string): Promise<any[]> {
    // Implement ABI loading based on protocol
    return [];
  }

  private async prepareFlashLoanTransaction(
    opportunity: ChainOpportunity,
    context: ExecutionContext,
    contract: ethers.Contract
  ): Promise<ethers.PopulatedTransaction> {
    // Prepare calldata for flash loan callback
    const calldata = this.encodeCallbackData(opportunity);

    // Encode flash loan function call
    return await contract.populateTransaction.flashLoan(
      opportunity.tokenIn,
      opportunity.amount,
      calldata,
      {
        nonce: context.nonce,
        deadline: context.deadline
      }
    );
  }

  private encodeCallbackData(opportunity: ChainOpportunity): string {
    // Implement callback data encoding
    return '0x';
  }

  private async getOptimalGasSettings(
    chainId: number,
    context: ExecutionContext
  ): Promise<any> {
    const chainConfig = this.chainManager.getChainConfig(chainId);
    const baseGasPrice = ethers.BigNumber.from(context.gasPrice);

    // Get network congestion level
    const congestion = await this.getNetworkCongestion(chainId);

    // Calculate optimal gas price
    let multiplier = chainConfig.gasConfig.estimateMultiplier;
    if (congestion > 0.8) {
      multiplier *= 1.2; // Increase by 20% during high congestion
    }

    const gasPrice = baseGasPrice.mul(
      Math.floor(multiplier * 100)
    ).div(100);

    // Calculate priority fee
    const priorityFee = gasPrice.mul(
      Math.floor(chainConfig.gasConfig.priorityFeeMultiplier * 100)
    ).div(100);

    return {
      maxFeePerGas: gasPrice,
      maxPriorityFeePerGas: priorityFee
    };
  }

  private async getNetworkCongestion(chainId: number): Promise<number> {
    try {
      const provider = this.chainManager.getProvider(chainId);
      const block = await provider.getBlock('latest');
      
      return block.gasUsed.mul(100).div(block.gasLimit).toNumber() / 100;
    } catch (error) {
      this.logger.error('Error getting network congestion', error);
      return 0.5; // Default to medium congestion
    }
  }

  private async waitForConfirmation(
    response: ethers.providers.TransactionResponse,
    chainId: number
  ): Promise<ethers.providers.TransactionReceipt> {
    const chainConfig = this.chainManager.getChainConfig(chainId);
    const confirmations = chainConfig.confirmations || 1;

    return await response.wait(confirmations);
  }

  private parseExecutionResult(
    receipt: ethers.providers.TransactionReceipt,
    opportunity: ChainOpportunity
  ): ExecutionResult {
    if (receipt.status === 0) {
      return {
        success: false,
        gasUsed: receipt.gasUsed.toString(),
        error: 'Transaction reverted',
        txHash: receipt.transactionHash
      };
    }

    // Parse profit from events
    const profit = this.parseProfitFromLogs(receipt.logs);

    return {
      success: true,
      profit: profit.toString(),
      gasUsed: receipt.gasUsed.toString(),
      txHash: receipt.transactionHash
    };
  }

  private parseProfitFromLogs(logs: ethers.providers.Log[]): ethers.BigNumber {
    // Implement profit parsing from event logs
    return ethers.BigNumber.from(0);
  }

  private async updateMetrics(
    opportunity: ChainOpportunity,
    result: ExecutionResult
  ): Promise<void> {
    try {
      // Update execution metrics
      this.metrics.incrementCounter('executions_total', {
        chainId: opportunity.chainId.toString(),
        status: result.success ? 'success' : 'failed'
      });

      if (result.success && result.profit) {
        // Update profit metrics
        const profitUSD = await this.priceService.convertToUSD(
          ethers.BigNumber.from(result.profit),
          opportunity.tokenIn,
          opportunity.chainId
        );

        this.metrics.setGauge('profit_usd', profitUSD, {
          chainId: opportunity.chainId.toString()
        });

        // Update successful strategies
        const strategyKey = this.getStrategyKey(opportunity);
        this.successfulStrategies.set(
          strategyKey,
          (this.successfulStrategies.get(strategyKey) || 0) + 1
        );
      }

      if (result.gasUsed) {
        // Update gas metrics
        this.metrics.observeHistogram('gas_used', parseInt(result.gasUsed), {
          chainId: opportunity.chainId.toString()
        });
      }

    } catch (error) {
      this.logger.error('Error updating metrics', error);
    }
  }

  private getStrategyKey(opportunity: ChainOpportunity): string {
    return `${opportunity.chainId}:${opportunity.protocol}:${opportunity.path.map(p => p.dex).join(':')}`;
  }

  private async handleExecutionError(
    error: Error,
    opportunity: ChainOpportunity
  ): Promise<void> {
    await this.errorHandler.handleError({
      error,
      severity: ErrorSeverity.HIGH,
      category: ErrorCategory.EXECUTION,
      context: {
        operation: 'executeFlashLoan',
        chainId: opportunity.chainId,
        opportunity: this.getOpportunityId(opportunity),
        timestamp: new Date()
      },
      retryable: false
    });
  }

  private enhanceError(error: Error, opportunity: ChainOpportunity): Error {
    error.message = `Chain ${opportunity.chainId} - ${error.message}`;
    return error;
  }

  public getExecutionMetrics(): any {
    return {
      activeExecutions: this.executingOpportunities.size,
      successfulStrategies: Object.fromEntries(this.successfulStrategies),
      lastExecutionTimes: Object.fromEntries(this.lastExecutionTime)
    };
  }
}
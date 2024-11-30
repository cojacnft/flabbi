import { expect } from 'chai';
import { ethers } from 'hardhat';
import { ChainManager } from '../../src/services/ChainManager';
import { FlashLoanExecutor } from '../../src/services/FlashLoanExecutor';
import { PerformanceOptimizer } from '../../src/services/PerformanceOptimizer';
import { Logger } from '../../src/services/Logger';
import { MetricsCollector } from '../../src/services/MetricsCollector';
import { ConfigService } from '../../src/services/ConfigService';
import { ErrorHandler } from '../../src/services/ErrorHandler';

describe('Multi-Chain Integration Tests', () => {
  let chainManager: ChainManager;
  let flashLoanExecutor: FlashLoanExecutor;
  let performanceOptimizer: PerformanceOptimizer;
  let logger: Logger;
  let metrics: MetricsCollector;
  let config: ConfigService;
  let errorHandler: ErrorHandler;

  before(async () => {
    // Initialize core services
    config = await ConfigService.getInstance();
    logger = Logger.getInstance(config, metrics);
    metrics = MetricsCollector.getInstance(config);
    errorHandler = ErrorHandler.getInstance(logger, metrics, config);

    // Initialize chain manager
    chainManager = new ChainManager(config.get('chains'));

    // Initialize performance optimizer with chain awareness
    performanceOptimizer = await PerformanceOptimizer.getInstance(
      logger,
      metrics,
      config
    );

    // Initialize flash loan executor with chain support
    flashLoanExecutor = new FlashLoanExecutor(
      chainManager,
      performanceOptimizer,
      logger,
      metrics,
      errorHandler
    );
  });

  describe('Chain Manager Integration', () => {
    it('should properly initialize all configured chains', async () => {
      const chains = config.get('chains');
      for (const chainId of Object.keys(chains)) {
        const provider = chainManager.getProvider(Number(chainId));
        const network = await provider.getNetwork();
        expect(network.chainId).to.equal(Number(chainId));
      }
    });

    it('should handle provider rotation correctly', async () => {
      const testChainId = 1; // Ethereum mainnet
      const initialProvider = chainManager.getProvider(testChainId);
      await chainManager.rotateProvider(testChainId);
      const newProvider = chainManager.getProvider(testChainId);
      
      // If multiple providers configured, they should be different
      const providers = config.get(`chains.${testChainId}.rpcUrls`);
      if (providers.length > 1) {
        expect(initialProvider).to.not.equal(newProvider);
      }
    });

    it('should maintain independent configurations per chain', async () => {
      const chains = config.get('chains');
      for (const chainId of Object.keys(chains)) {
        const gasConfig = chainManager.getGasConfig(Number(chainId));
        const minProfit = chainManager.getMinProfit(Number(chainId));
        
        expect(gasConfig).to.have.property('estimateMultiplier');
        expect(gasConfig).to.have.property('maxGwei');
        expect(minProfit).to.have.property('usd');
        expect(minProfit).to.have.property('native');
      }
    });
  });

  describe('Flash Loan Executor Chain Integration', () => {
    it('should validate opportunities per chain', async () => {
      const testOpportunity = {
        chainId: 1,
        protocol: 'aave',
        tokenIn: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', // WETH
        amount: ethers.utils.parseEther('10'),
        path: [
          {
            dex: 'uniswap_v2',
            tokenIn: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
            tokenOut: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', // USDC
            pool: '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc'
          }
        ]
      };

      const isValid = await flashLoanExecutor.validateOpportunity(testOpportunity);
      expect(isValid).to.be.a('boolean');
    });

    it('should calculate gas costs correctly per chain', async () => {
      const chains = config.get('chains');
      for (const chainId of Object.keys(chains)) {
        const gasPrice = await chainManager.getProvider(Number(chainId)).getGasPrice();
        const gasConfig = chainManager.getGasConfig(Number(chainId));
        
        const estimatedCost = await flashLoanExecutor.estimateGasCost({
          chainId: Number(chainId),
          gasLimit: 500000,
          gasPrice
        });

        expect(estimatedCost.toString()).to.not.equal('0');
      }
    });
  });

  describe('Performance Optimizer Chain Integration', () => {
    it('should maintain separate performance metrics per chain', async () => {
      const testMetrics = {
        chainId: 1,
        executionTime: 1.5,
        gasUsed: 150000,
        memoryUsage: 500,
        cpuUsage: 0.6,
        networkLatency: 100,
        queueLength: 5,
        successRate: 0.95
      };

      await performanceOptimizer.recordMetrics(testMetrics);
      const report = performanceOptimizer.getPerformanceReport(1); // Chain specific report
      
      expect(report).to.have.property('executionTime');
      expect(report).to.have.property('gasUsed');
      expect(report.chainId).to.equal(1);
    });

    it('should optimize parameters per chain', async () => {
      const chains = config.get('chains');
      for (const chainId of Object.keys(chains)) {
        const optimizedParams = await performanceOptimizer.getOptimizedParameters(
          Number(chainId)
        );
        
        expect(optimizedParams).to.have.property('maxGasPrice');
        expect(optimizedParams).to.have.property('minProfit');
        // Parameters should be within chain-specific bounds
        expect(optimizedParams.maxGasPrice).to.be.lte(
          chainManager.getGasConfig(Number(chainId)).maxGwei
        );
      }
    });
  });

  describe('Error Handling Across Chains', () => {
    it('should handle chain-specific errors appropriately', async () => {
      const testError = new Error('Test chain error');
      const chainId = 1;

      await errorHandler.handleError({
        error: testError,
        severity: 'medium',
        category: 'chain',
        context: {
          chainId,
          operation: 'test',
          timestamp: new Date()
        },
        retryable: true
      });

      // Verify error was logged with chain context
      const errorLogs = await logger.query({
        from: new Date(Date.now() - 1000),
        until: new Date(),
        limit: 1,
        order: 'desc'
      });

      expect(errorLogs[0]).to.include({
        chainId: chainId.toString(),
        category: 'chain'
      });
    });

    it('should maintain separate retry counters per chain', async () => {
      const chainId = 1;
      const errorKey = 'test_error';

      // Simulate multiple retries
      for (let i = 0; i < 3; i++) {
        await errorHandler.handleError({
          error: new Error('Test retry error'),
          severity: 'medium',
          category: 'chain',
          context: {
            chainId,
            operation: errorKey,
            timestamp: new Date()
          },
          retryable: true
        });
      }

      const retryCount = errorHandler.getRetryAttempts(`chain:${chainId}:${errorKey}`);
      expect(retryCount).to.equal(3);
    });
  });

  describe('System Integration', () => {
    it('should handle concurrent operations across chains', async () => {
      const chains = [1, 137]; // Test with Ethereum and Polygon
      const operations = chains.map(chainId => 
        flashLoanExecutor.findArbitrageOpportunities(chainId)
      );

      const results = await Promise.all(operations);
      results.forEach((result, index) => {
        expect(result).to.be.an('array');
        if (result.length > 0) {
          expect(result[0].chainId).to.equal(chains[index]);
        }
      });
    });

    it('should maintain system stability under multi-chain load', async () => {
      const startMemory = process.memoryUsage().heapUsed;
      const chains = [1, 137, 42161]; // Ethereum, Polygon, Arbitrum

      // Simulate heavy load
      await Promise.all(chains.map(async chainId => {
        for (let i = 0; i < 5; i++) {
          await flashLoanExecutor.findArbitrageOpportunities(chainId);
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }));

      const endMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (endMemory - startMemory) / 1024 / 1024; // MB

      // Ensure memory usage didn't grow too much
      expect(memoryIncrease).to.be.lessThan(100); // Less than 100MB increase
    });
  });
});
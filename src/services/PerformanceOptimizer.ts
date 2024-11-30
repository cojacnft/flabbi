import { EventEmitter } from 'events';
import { Logger } from './Logger';
import { MetricsCollector } from './MetricsCollector';
import { ConfigService } from './ConfigService';

interface PerformanceMetrics {
  executionTime: number;
  gasUsed: number;
  memoryUsage: number;
  cpuUsage: number;
  networkLatency: number;
  queueLength: number;
  successRate: number;
}

interface OptimizationConfig {
  maxConcurrentExecutions: number;
  batchSize: number;
  timeoutMs: number;
  retryAttempts: number;
  cacheSize: number;
  networkTimeout: number;
}

export class PerformanceOptimizer extends EventEmitter {
  private static instance: PerformanceOptimizer;
  private metricsHistory: PerformanceMetrics[] = [];
  private config: OptimizationConfig;
  private optimizationInterval: NodeJS.Timeout | null = null;

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly configService: ConfigService
  ) {
    super();
    this.config = this.loadConfig();
    this.startOptimization();
  }

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    configService: ConfigService
  ): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer(
        logger,
        metrics,
        configService
      );
    }
    return PerformanceOptimizer.instance;
  }

  private loadConfig(): OptimizationConfig {
    return {
      maxConcurrentExecutions: this.configService.get(
        'performance.maxConcurrentExecutions',
        5
      ),
      batchSize: this.configService.get('performance.batchSize', 10),
      timeoutMs: this.configService.get('performance.timeoutMs', 5000),
      retryAttempts: this.configService.get('performance.retryAttempts', 3),
      cacheSize: this.configService.get('performance.cacheSize', 1000),
      networkTimeout: this.configService.get('performance.networkTimeout', 3000),
    };
  }

  private startOptimization(): void {
    this.optimizationInterval = setInterval(
      () => this.optimize(),
      60000 // Run optimization every minute
    );
  }

  public async recordMetrics(metrics: PerformanceMetrics): Promise<void> {
    this.metricsHistory.push(metrics);
    if (this.metricsHistory.length > 1000) {
      this.metricsHistory = this.metricsHistory.slice(-1000);
    }

    // Update Prometheus metrics
    this.metrics.observeHistogram('execution_time_seconds', metrics.executionTime);
    this.metrics.observeHistogram('gas_used', metrics.gasUsed);
    this.metrics.setGauge('memory_usage', metrics.memoryUsage);
    this.metrics.setGauge('cpu_usage', metrics.cpuUsage);
    this.metrics.observeHistogram('network_latency', metrics.networkLatency);
    this.metrics.setGauge('queue_length', metrics.queueLength);
    this.metrics.setGauge('success_rate', metrics.successRate);
  }

  private async optimize(): Promise<void> {
    try {
      const currentMetrics = await this.analyzeCurrentPerformance();
      const recommendations = this.generateRecommendations(currentMetrics);
      await this.applyOptimizations(recommendations);

      this.logger.info('Performance optimization completed', {
        metrics: currentMetrics,
        recommendations,
      });
    } catch (error) {
      this.logger.error('Error during performance optimization', error);
    }
  }

  private async analyzeCurrentPerformance(): Promise<any> {
    if (this.metricsHistory.length === 0) {
      return null;
    }

    const recentMetrics = this.metricsHistory.slice(-10);

    return {
      avgExecutionTime:
        recentMetrics.reduce((sum, m) => sum + m.executionTime, 0) /
        recentMetrics.length,
      avgGasUsed:
        recentMetrics.reduce((sum, m) => sum + m.gasUsed, 0) /
        recentMetrics.length,
      avgMemoryUsage:
        recentMetrics.reduce((sum, m) => sum + m.memoryUsage, 0) /
        recentMetrics.length,
      avgCpuUsage:
        recentMetrics.reduce((sum, m) => sum + m.cpuUsage, 0) /
        recentMetrics.length,
      avgNetworkLatency:
        recentMetrics.reduce((sum, m) => sum + m.networkLatency, 0) /
        recentMetrics.length,
      avgQueueLength:
        recentMetrics.reduce((sum, m) => sum + m.queueLength, 0) /
        recentMetrics.length,
      avgSuccessRate:
        recentMetrics.reduce((sum, m) => sum + m.successRate, 0) /
        recentMetrics.length,
    };
  }

  private generateRecommendations(metrics: any): any {
    if (!metrics) {
      return null;
    }

    const recommendations: any = {};

    // Concurrency optimization
    if (metrics.avgQueueLength > 5 && metrics.avgSuccessRate > 0.9) {
      recommendations.maxConcurrentExecutions = Math.min(
        this.config.maxConcurrentExecutions + 1,
        10
      );
    } else if (metrics.avgSuccessRate < 0.8) {
      recommendations.maxConcurrentExecutions = Math.max(
        this.config.maxConcurrentExecutions - 1,
        1
      );
    }

    // Batch size optimization
    if (metrics.avgExecutionTime < 1000 && metrics.avgQueueLength > 10) {
      recommendations.batchSize = Math.min(this.config.batchSize + 2, 20);
    } else if (metrics.avgExecutionTime > 2000) {
      recommendations.batchSize = Math.max(this.config.batchSize - 2, 1);
    }

    // Timeout optimization
    if (metrics.avgExecutionTime > this.config.timeoutMs * 0.8) {
      recommendations.timeoutMs = this.config.timeoutMs * 1.2;
    } else if (metrics.avgExecutionTime < this.config.timeoutMs * 0.3) {
      recommendations.timeoutMs = this.config.timeoutMs * 0.8;
    }

    // Cache size optimization
    const memoryUsageRatio = metrics.avgMemoryUsage / process.memoryUsage().heapTotal;
    if (memoryUsageRatio > 0.8) {
      recommendations.cacheSize = Math.floor(this.config.cacheSize * 0.8);
    } else if (memoryUsageRatio < 0.5) {
      recommendations.cacheSize = Math.floor(this.config.cacheSize * 1.2);
    }

    // Network timeout optimization
    if (metrics.avgNetworkLatency > this.config.networkTimeout * 0.8) {
      recommendations.networkTimeout = this.config.networkTimeout * 1.2;
    } else if (metrics.avgNetworkLatency < this.config.networkTimeout * 0.3) {
      recommendations.networkTimeout = this.config.networkTimeout * 0.8;
    }

    return recommendations;
  }

  private async applyOptimizations(recommendations: any): Promise<void> {
    if (!recommendations) {
      return;
    }

    let configChanged = false;

    for (const [key, value] of Object.entries(recommendations)) {
      if (this.config[key as keyof OptimizationConfig] !== value) {
        this.config[key as keyof OptimizationConfig] = value as never;
        configChanged = true;
      }
    }

    if (configChanged) {
      // Update configuration
      await this.configService.set('performance', this.config);

      // Emit optimization event
      this.emit('optimizationApplied', {
        previousConfig: { ...this.config },
        newConfig: recommendations,
        timestamp: new Date(),
      });

      this.logger.info('Applied performance optimizations', {
        recommendations,
        newConfig: this.config,
      });
    }
  }

  public getConfig(): OptimizationConfig {
    return { ...this.config };
  }

  public async updateConfig(updates: Partial<OptimizationConfig>): Promise<void> {
    this.config = {
      ...this.config,
      ...updates,
    };
    await this.configService.set('performance', this.config);
  }

  public getMetricsHistory(): PerformanceMetrics[] {
    return [...this.metricsHistory];
  }

  public stop(): void {
    if (this.optimizationInterval) {
      clearInterval(this.optimizationInterval);
      this.optimizationInterval = null;
    }
  }

  public getPerformanceReport(): any {
    const recentMetrics = this.metricsHistory.slice(-100);
    
    return {
      executionTime: {
        avg: this.calculateAverage(recentMetrics.map(m => m.executionTime)),
        min: Math.min(...recentMetrics.map(m => m.executionTime)),
        max: Math.max(...recentMetrics.map(m => m.executionTime)),
        p95: this.calculatePercentile(recentMetrics.map(m => m.executionTime), 95),
      },
      gasUsed: {
        avg: this.calculateAverage(recentMetrics.map(m => m.gasUsed)),
        min: Math.min(...recentMetrics.map(m => m.gasUsed)),
        max: Math.max(...recentMetrics.map(m => m.gasUsed)),
        p95: this.calculatePercentile(recentMetrics.map(m => m.gasUsed), 95),
      },
      successRate: this.calculateAverage(recentMetrics.map(m => m.successRate)),
      resourceUtilization: {
        memory: this.calculateAverage(recentMetrics.map(m => m.memoryUsage)),
        cpu: this.calculateAverage(recentMetrics.map(m => m.cpuUsage)),
      },
      networkPerformance: {
        latency: this.calculateAverage(recentMetrics.map(m => m.networkLatency)),
        queueLength: this.calculateAverage(recentMetrics.map(m => m.queueLength)),
      },
      config: this.config,
    };
  }

  private calculateAverage(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculatePercentile(values: number[], percentile: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }
}
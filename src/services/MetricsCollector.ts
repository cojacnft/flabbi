import { Registry, Counter, Gauge, Histogram } from 'prom-client';
import { ConfigService } from './ConfigService';

export class MetricsCollector {
  private static instance: MetricsCollector;
  private registry: Registry;
  private counters: Map<string, Counter<string>>;
  private gauges: Map<string, Gauge<string>>;
  private histograms: Map<string, Histogram<string>>;

  private constructor(private readonly config: ConfigService) {
    this.registry = new Registry();
    this.counters = new Map();
    this.gauges = new Map();
    this.histograms = new Map();
    this.initializeDefaultMetrics();
  }

  public static getInstance(config: ConfigService): MetricsCollector {
    if (!MetricsCollector.instance) {
      MetricsCollector.instance = new MetricsCollector(config);
    }
    return MetricsCollector.instance;
  }

  private initializeDefaultMetrics(): void {
    // System metrics
    this.createGauge({
      name: 'system_memory_usage',
      help: 'Memory usage in bytes',
      labelNames: ['type'],
    });

    this.createGauge({
      name: 'system_cpu_usage',
      help: 'CPU usage percentage',
      labelNames: ['type'],
    });

    // Business metrics
    this.createCounter({
      name: 'arbitrage_opportunities_total',
      help: 'Total number of arbitrage opportunities detected',
      labelNames: ['status'],
    });

    this.createCounter({
      name: 'flash_loans_total',
      help: 'Total number of flash loans executed',
      labelNames: ['status', 'provider'],
    });

    this.createGauge({
      name: 'current_gas_price',
      help: 'Current gas price in gwei',
    });

    this.createHistogram({
      name: 'execution_time_seconds',
      help: 'Execution time in seconds',
      labelNames: ['operation'],
      buckets: [0.1, 0.5, 1, 2, 5, 10],
    });

    // Error metrics
    this.createCounter({
      name: 'errors_total',
      help: 'Total number of errors',
      labelNames: ['severity', 'category'],
    });

    // Performance metrics
    this.createGauge({
      name: 'response_time_seconds',
      help: 'Response time in seconds',
      labelNames: ['endpoint'],
    });

    // Business KPIs
    this.createGauge({
      name: 'total_profit_usd',
      help: 'Total profit in USD',
    });

    this.createGauge({
      name: 'success_rate',
      help: 'Success rate of arbitrage executions',
    });
  }

  public createCounter({
    name,
    help,
    labelNames = [],
  }: {
    name: string;
    help: string;
    labelNames?: string[];
  }): void {
    if (!this.counters.has(name)) {
      const counter = new Counter({
        name,
        help,
        labelNames,
        registers: [this.registry],
      });
      this.counters.set(name, counter);
    }
  }

  public createGauge({
    name,
    help,
    labelNames = [],
  }: {
    name: string;
    help: string;
    labelNames?: string[];
  }): void {
    if (!this.gauges.has(name)) {
      const gauge = new Gauge({
        name,
        help,
        labelNames,
        registers: [this.registry],
      });
      this.gauges.set(name, gauge);
    }
  }

  public createHistogram({
    name,
    help,
    labelNames = [],
    buckets = [0.1, 0.5, 1, 2, 5, 10],
  }: {
    name: string;
    help: string;
    labelNames?: string[];
    buckets?: number[];
  }): void {
    if (!this.histograms.has(name)) {
      const histogram = new Histogram({
        name,
        help,
        labelNames,
        buckets,
        registers: [this.registry],
      });
      this.histograms.set(name, histogram);
    }
  }

  public incrementCounter(name: string, labels: Record<string, string> = {}): void {
    const counter = this.counters.get(name);
    if (counter) {
      counter.inc(labels);
    }
  }

  public setGauge(name: string, value: number, labels: Record<string, string> = {}): void {
    const gauge = this.gauges.get(name);
    if (gauge) {
      gauge.set(labels, value);
    }
  }

  public observeHistogram(
    name: string,
    value: number,
    labels: Record<string, string> = {}
  ): void {
    const histogram = this.histograms.get(name);
    if (histogram) {
      histogram.observe(labels, value);
    }
  }

  public async getMetrics(): Promise<string> {
    return await this.registry.metrics();
  }

  public async getMetricValue(name: string): Promise<number | null> {
    try {
      const metrics = await this.registry.getMetricsAsJSON();
      const metric = metrics.find(m => m.name === name);
      if (metric && metric.values && metric.values.length > 0) {
        return metric.values[0].value;
      }
      return null;
    } catch (error) {
      console.error('Error getting metric value:', error);
      return null;
    }
  }

  public resetMetric(name: string): void {
    const counter = this.counters.get(name);
    if (counter) {
      counter.reset();
    }

    const gauge = this.gauges.get(name);
    if (gauge) {
      gauge.reset();
    }

    const histogram = this.histograms.get(name);
    if (histogram) {
      histogram.reset();
    }
  }

  public removeMetric(name: string): void {
    this.registry.removeSingleMetric(name);
    this.counters.delete(name);
    this.gauges.delete(name);
    this.histograms.delete(name);
  }

  public clearMetrics(): void {
    this.registry.clear();
    this.counters.clear();
    this.gauges.clear();
    this.histograms.clear();
    this.initializeDefaultMetrics();
  }

  public startTimer(name: string, labels: Record<string, string> = {}): () => void {
    const histogram = this.histograms.get(name);
    if (histogram) {
      const end = histogram.startTimer(labels);
      return end;
    }
    return () => {};
  }

  public trackMemoryUsage(): void {
    const usage = process.memoryUsage();
    Object.entries(usage).forEach(([type, bytes]) => {
      this.setGauge('system_memory_usage', bytes, { type });
    });
  }

  public trackCpuUsage(): void {
    const startUsage = process.cpuUsage();
    setTimeout(() => {
      const endUsage = process.cpuUsage(startUsage);
      const userUsagePercent = (endUsage.user / 1000000) * 100;
      const systemUsagePercent = (endUsage.system / 1000000) * 100;

      this.setGauge('system_cpu_usage', userUsagePercent, { type: 'user' });
      this.setGauge('system_cpu_usage', systemUsagePercent, { type: 'system' });
    }, 100);
  }

  public getRegistry(): Registry {
    return this.registry;
  }
}
import { EventEmitter } from 'events';
import { Logger } from './Logger';
import { MetricsCollector } from './MetricsCollector';
import { NotificationService } from './NotificationService';
import { ConfigService } from './ConfigService';

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum ErrorCategory {
  NETWORK = 'network',
  CONTRACT = 'contract',
  TRANSACTION = 'transaction',
  VALIDATION = 'validation',
  EXECUTION = 'execution',
  SYSTEM = 'system',
}

export interface ErrorContext {
  operation: string;
  input?: any;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface ErrorDetails {
  error: Error;
  severity: ErrorSeverity;
  category: ErrorCategory;
  context: ErrorContext;
  retryable: boolean;
  recoveryStrategy?: string;
}

export class ErrorHandler extends EventEmitter {
  private static instance: ErrorHandler;
  private retryAttempts: Map<string, number> = new Map();
  private errorCounts: Map<string, number> = new Map();
  private readonly maxRetries: number;
  private readonly retryDelay: number;
  private readonly errorThreshold: number;

  private constructor(
    private readonly logger: Logger,
    private readonly metrics: MetricsCollector,
    private readonly notifications: NotificationService,
    private readonly config: ConfigService
  ) {
    super();
    this.maxRetries = config.get('error.maxRetries', 3);
    this.retryDelay = config.get('error.retryDelay', 1000);
    this.errorThreshold = config.get('error.threshold', 10);
  }

  public static getInstance(
    logger: Logger,
    metrics: MetricsCollector,
    notifications: NotificationService,
    config: ConfigService
  ): ErrorHandler {
    if (!ErrorHandler.instance) {
      ErrorHandler.instance = new ErrorHandler(logger, metrics, notifications, config);
    }
    return ErrorHandler.instance;
  }

  public async handleError(details: ErrorDetails): Promise<void> {
    const errorKey = this.getErrorKey(details);
    
    // Log error
    this.logger.error('Error occurred', {
      error: details.error,
      severity: details.severity,
      category: details.category,
      context: details.context,
    });

    // Update metrics
    this.metrics.incrementCounter('errors_total', {
      severity: details.severity,
      category: details.category,
    });

    // Update error counts
    this.incrementErrorCount(errorKey);

    // Check error threshold
    if (this.shouldTriggerCircuitBreaker(errorKey)) {
      await this.triggerCircuitBreaker(details);
      return;
    }

    // Handle based on severity
    switch (details.severity) {
      case ErrorSeverity.CRITICAL:
        await this.handleCriticalError(details);
        break;
      case ErrorSeverity.HIGH:
        await this.handleHighSeverityError(details);
        break;
      case ErrorSeverity.MEDIUM:
        await this.handleMediumSeverityError(details);
        break;
      case ErrorSeverity.LOW:
        await this.handleLowSeverityError(details);
        break;
    }

    // Attempt recovery if applicable
    if (details.retryable && this.canRetry(errorKey)) {
      await this.retryOperation(details);
    }

    // Emit error event
    this.emit('error', details);
  }

  private getErrorKey(details: ErrorDetails): string {
    return `${details.category}:${details.context.operation}`;
  }

  private incrementErrorCount(key: string): void {
    const count = (this.errorCounts.get(key) || 0) + 1;
    this.errorCounts.set(key, count);

    // Reset count after a period
    setTimeout(() => {
      this.errorCounts.set(key, Math.max(0, (this.errorCounts.get(key) || 0) - 1));
    }, this.config.get('error.countResetDelay', 60000));
  }

  private shouldTriggerCircuitBreaker(key: string): boolean {
    return (this.errorCounts.get(key) || 0) >= this.errorThreshold;
  }

  private async triggerCircuitBreaker(details: ErrorDetails): Promise<void> {
    this.logger.warn('Circuit breaker triggered', {
      category: details.category,
      operation: details.context.operation,
    });

    // Notify about circuit breaker
    await this.notifications.send({
      level: 'critical',
      title: 'Circuit Breaker Triggered',
      message: `Operation ${details.context.operation} has been suspended due to excessive errors`,
      metadata: {
        category: details.category,
        errorCount: this.errorCounts.get(this.getErrorKey(details)),
      },
    });

    // Update metrics
    this.metrics.incrementCounter('circuit_breaker_triggers', {
      category: details.category,
      operation: details.context.operation,
    });

    // Emit circuit breaker event
    this.emit('circuitBreaker', {
      category: details.category,
      operation: details.context.operation,
      timestamp: new Date(),
    });
  }

  private async handleCriticalError(details: ErrorDetails): Promise<void> {
    // Notify immediately
    await this.notifications.send({
      level: 'critical',
      title: 'Critical Error',
      message: details.error.message,
      metadata: details.context,
    });

    // Stop related operations
    this.emit('stopOperations', {
      category: details.category,
      operation: details.context.operation,
    });

    // Collect diagnostic information
    const diagnostics = await this.collectDiagnostics(details);
    await this.logger.error('Critical error diagnostics', diagnostics);
  }

  private async handleHighSeverityError(details: ErrorDetails): Promise<void> {
    // Notify if error persists
    const errorKey = this.getErrorKey(details);
    if ((this.errorCounts.get(errorKey) || 0) > 1) {
      await this.notifications.send({
        level: 'high',
        title: 'High Severity Error',
        message: details.error.message,
        metadata: details.context,
      });
    }

    // Attempt recovery if possible
    if (details.recoveryStrategy) {
      await this.executeRecoveryStrategy(details);
    }
  }

  private async handleMediumSeverityError(details: ErrorDetails): Promise<void> {
    // Log and monitor
    this.logger.warn('Medium severity error', {
      error: details.error,
      context: details.context,
    });

    // Update metrics
    this.metrics.incrementCounter('medium_severity_errors', {
      category: details.category,
      operation: details.context.operation,
    });
  }

  private async handleLowSeverityError(details: ErrorDetails): Promise<void> {
    // Just log and track
    this.logger.info('Low severity error', {
      error: details.error,
      context: details.context,
    });
  }

  private canRetry(errorKey: string): boolean {
    const attempts = this.retryAttempts.get(errorKey) || 0;
    return attempts < this.maxRetries;
  }

  private async retryOperation(details: ErrorDetails): Promise<void> {
    const errorKey = this.getErrorKey(details);
    const attempts = (this.retryAttempts.get(errorKey) || 0) + 1;
    this.retryAttempts.set(errorKey, attempts);

    // Exponential backoff
    const delay = this.retryDelay * Math.pow(2, attempts - 1);
    await new Promise(resolve => setTimeout(resolve, delay));

    this.logger.info('Retrying operation', {
      operation: details.context.operation,
      attempt: attempts,
      maxRetries: this.maxRetries,
    });

    // Emit retry event
    this.emit('retry', {
      ...details,
      attempt: attempts,
      delay,
    });
  }

  private async collectDiagnostics(details: ErrorDetails): Promise<any> {
    try {
      return {
        error: {
          message: details.error.message,
          stack: details.error.stack,
          name: details.error.name,
        },
        context: details.context,
        systemInfo: {
          memory: process.memoryUsage(),
          uptime: process.uptime(),
          timestamp: new Date().toISOString(),
        },
        metrics: await this.metrics.getMetrics(),
      };
    } catch (error) {
      this.logger.error('Error collecting diagnostics', error);
      return null;
    }
  }

  private async executeRecoveryStrategy(details: ErrorDetails): Promise<void> {
    try {
      this.logger.info('Executing recovery strategy', {
        strategy: details.recoveryStrategy,
        operation: details.context.operation,
      });

      // Emit recovery event
      this.emit('recovery', {
        strategy: details.recoveryStrategy,
        context: details.context,
        timestamp: new Date(),
      });

      // Execute strategy-specific recovery logic
      switch (details.recoveryStrategy) {
        case 'resetConnection':
          await this.resetConnection();
          break;
        case 'clearCache':
          await this.clearCache();
          break;
        case 'reloadConfig':
          await this.reloadConfig();
          break;
        default:
          this.logger.warn('Unknown recovery strategy', {
            strategy: details.recoveryStrategy,
          });
      }
    } catch (error) {
      this.logger.error('Recovery strategy failed', {
        error,
        strategy: details.recoveryStrategy,
      });
    }
  }

  private async resetConnection(): Promise<void> {
    // Implementation for resetting connections
    this.emit('resetConnection');
  }

  private async clearCache(): Promise<void> {
    // Implementation for clearing cache
    this.emit('clearCache');
  }

  private async reloadConfig(): Promise<void> {
    // Implementation for reloading configuration
    await this.config.reload();
    this.emit('configReloaded');
  }

  public resetErrorCount(key: string): void {
    this.errorCounts.delete(key);
    this.retryAttempts.delete(key);
  }

  public getErrorCount(key: string): number {
    return this.errorCounts.get(key) || 0;
  }

  public getRetryAttempts(key: string): number {
    return this.retryAttempts.get(key) || 0;
  }
}
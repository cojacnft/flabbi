import winston from 'winston';
import { ElasticsearchTransport } from 'winston-elasticsearch';
import { ConfigService } from './ConfigService';
import { MetricsCollector } from './MetricsCollector';

export class Logger {
  private static instance: Logger;
  private logger: winston.Logger;
  private context: string = '';

  private constructor(
    private readonly config: ConfigService,
    private readonly metrics: MetricsCollector
  ) {
    this.initializeLogger();
  }

  public static getInstance(
    config: ConfigService,
    metrics: MetricsCollector
  ): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger(config, metrics);
    }
    return Logger.instance;
  }

  private initializeLogger(): void {
    const logLevel = this.config.get('log.level', 'info');
    const logFormat = winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json()
    );

    const transports: winston.transport[] = [
      new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple()
        ),
      }),
      new winston.transports.File({
        filename: 'logs/error.log',
        level: 'error',
      }),
      new winston.transports.File({
        filename: 'logs/combined.log',
      }),
    ];

    // Add Elasticsearch transport in production
    if (this.config.get('environment') === 'production') {
      const esConfig = this.config.get('elasticsearch', {});
      if (esConfig.host) {
        transports.push(
          new ElasticsearchTransport({
            level: 'info',
            clientOpts: {
              node: esConfig.host,
              auth: {
                username: esConfig.username,
                password: esConfig.password,
              },
            },
            indexPrefix: 'flashloan-logs',
          })
        );
      }
    }

    this.logger = winston.createLogger({
      level: logLevel,
      format: logFormat,
      defaultMeta: {
        service: 'flashloan-arbitrage',
        environment: this.config.get('environment', 'development'),
      },
      transports,
    });
  }

  public setContext(context: string): void {
    this.context = context;
  }

  private formatMessage(message: string): string {
    return this.context ? `[${this.context}] ${message}` : message;
  }

  private enhanceMetadata(metadata: any = {}): any {
    return {
      ...metadata,
      timestamp: new Date().toISOString(),
      context: this.context || undefined,
    };
  }

  public error(message: string, metadata?: any): void {
    this.metrics.incrementCounter('log_messages_total', {
      level: 'error',
      context: this.context,
    });
    this.logger.error(this.formatMessage(message), this.enhanceMetadata(metadata));
  }

  public warn(message: string, metadata?: any): void {
    this.metrics.incrementCounter('log_messages_total', {
      level: 'warn',
      context: this.context,
    });
    this.logger.warn(this.formatMessage(message), this.enhanceMetadata(metadata));
  }

  public info(message: string, metadata?: any): void {
    this.metrics.incrementCounter('log_messages_total', {
      level: 'info',
      context: this.context,
    });
    this.logger.info(this.formatMessage(message), this.enhanceMetadata(metadata));
  }

  public debug(message: string, metadata?: any): void {
    this.metrics.incrementCounter('log_messages_total', {
      level: 'debug',
      context: this.context,
    });
    this.logger.debug(this.formatMessage(message), this.enhanceMetadata(metadata));
  }

  public async query(options: winston.QueryOptions): Promise<any> {
    return new Promise((resolve, reject) => {
      this.logger.query(options, (err, results) => {
        if (err) {
          reject(err);
        } else {
          resolve(results);
        }
      });
    });
  }

  public async flush(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.logger.end(() => resolve());
    });
  }

  public getLogLevel(): string {
    return this.logger.level;
  }

  public setLogLevel(level: string): void {
    this.logger.level = level;
  }

  public createChildLogger(context: string): Logger {
    const childLogger = new Logger(this.config, this.metrics);
    childLogger.setContext(context);
    return childLogger;
  }

  public async rotate(): Promise<void> {
    // Implement log rotation logic
    try {
      await this.flush();
      this.initializeLogger();
    } catch (error) {
      console.error('Error rotating logs:', error);
    }
  }

  public async cleanup(olderThan: Date): Promise<void> {
    // Implement log cleanup logic
    try {
      const options: winston.QueryOptions = {
        from: new Date(0),
        until: olderThan,
        limit: 100000,
      };

      const results = await this.query(options);
      // Delete old logs
      // Implementation depends on the storage backend
    } catch (error) {
      console.error('Error cleaning up logs:', error);
    }
  }
}
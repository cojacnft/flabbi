import * as fs from 'fs';
import * as path from 'path';
import { EventEmitter } from 'events';
import { ethers } from 'ethers';

export class ConfigService extends EventEmitter {
    private static instance: ConfigService;
    private config: any = {};
    private envVars: Map<string, string> = new Map();
    private readonly configDir = path.join(process.cwd(), 'config');
    private readonly requiredEnvVars = [
        'PRIVATE_KEY',
        'ETH_RPC_URL_1',
        'ARB_RPC_URL_1'
    ];

    private constructor() {
        super();
        this.loadConfig();
        this.watchConfigChanges();
    }

    public static getInstance(): ConfigService {
        if (!ConfigService.instance) {
            ConfigService.instance = new ConfigService();
        }
        return ConfigService.instance;
    }

    private loadConfig(): void {
        try {
            // Load environment variables
            this.loadEnvVariables();

            // Load base config
            const defaultConfig = this.loadConfigFile('default.json');

            // Load environment-specific config
            const env = process.env.NODE_ENV || 'development';
            const envConfig = this.loadConfigFile(`${env}.json`);

            // Merge configs
            this.config = this.deepMerge(defaultConfig, envConfig);

            // Replace environment variables
            this.replaceEnvVariables(this.config);

            // Validate configuration
            this.validateConfig();

        } catch (error) {
            throw new Error(`Failed to load configuration: ${error.message}`);
        }
    }

    private loadEnvVariables(): void {
        // Check required environment variables
        for (const envVar of this.requiredEnvVars) {
            const value = process.env[envVar];
            if (!value) {
                throw new Error(`Required environment variable ${envVar} is not set`);
            }
            this.envVars.set(envVar, value);
        }

        // Load optional environment variables
        Object.entries(process.env).forEach(([key, value]) => {
            if (value && !this.envVars.has(key)) {
                this.envVars.set(key, value);
            }
        });
    }

    private loadConfigFile(filename: string): any {
        const filePath = path.join(this.configDir, filename);
        if (!fs.existsSync(filePath)) {
            if (filename === 'default.json') {
                throw new Error('Default configuration file not found');
            }
            return {};
        }

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            return JSON.parse(content);
        } catch (error) {
            throw new Error(`Error loading config file ${filename}: ${error.message}`);
        }
    }

    private watchConfigChanges(): void {
        fs.watch(this.configDir, (eventType, filename) => {
            if (eventType === 'change' && filename.endsWith('.json')) {
                this.loadConfig();
                this.emit('configChanged', filename);
            }
        });
    }

    private replaceEnvVariables(obj: any): void {
        for (const key in obj) {
            if (typeof obj[key] === 'string' && obj[key].startsWith('${') && obj[key].endsWith('}')) {
                const envVar = obj[key].slice(2, -1);
                const value = this.envVars.get(envVar);
                if (!value) {
                    throw new Error(`Environment variable ${envVar} not found`);
                }
                obj[key] = value;
            } else if (typeof obj[key] === 'object' && obj[key] !== null) {
                this.replaceEnvVariables(obj[key]);
            }
        }
    }

    private validateConfig(): void {
        // Validate chain configurations
        const chains = this.config.chains;
        if (!chains || Object.keys(chains).length === 0) {
            throw new Error('No chains configured');
        }

        for (const [chainId, chainConfig] of Object.entries(chains)) {
            this.validateChainConfig(parseInt(chainId), chainConfig);
        }

        // Validate execution parameters
        const execution = this.config.execution;
        if (!execution) {
            throw new Error('Execution configuration missing');
        }

        if (execution.minProfitUSD <= 0) {
            throw new Error('Invalid minimum profit configuration');
        }

        // Validate security settings
        const security = this.config.security;
        if (!security) {
            throw new Error('Security configuration missing');
        }

        if (security.maxSlippage <= 0 || security.maxSlippage >= 1) {
            throw new Error('Invalid slippage configuration');
        }
    }

    private validateChainConfig(chainId: number, config: any): void {
        if (!config.rpcUrls || config.rpcUrls.length === 0) {
            throw new Error(`No RPC URLs configured for chain ${chainId}`);
        }

        if (!config.dexes || Object.keys(config.dexes).length === 0) {
            throw new Error(`No DEXes configured for chain ${chainId}`);
        }

        if (!config.tokens || Object.keys(config.tokens).length === 0) {
            throw new Error(`No tokens configured for chain ${chainId}`);
        }

        // Validate addresses
        for (const dex of Object.values(config.dexes) as any[]) {
            if (!ethers.utils.isAddress(dex.routerAddress)) {
                throw new Error(`Invalid router address for DEX ${dex.name} on chain ${chainId}`);
            }
        }

        for (const token of Object.values(config.tokens) as any[]) {
            if (!ethers.utils.isAddress(token.address)) {
                throw new Error(`Invalid token address on chain ${chainId}`);
            }
        }
    }

    public get<T>(key: string, defaultValue?: T): T {
        return this.getNestedValue(this.config, key) ?? defaultValue;
    }

    public getChainConfig(chainId: number): any {
        const chainConfig = this.config.chains[chainId];
        if (!chainConfig) {
            throw new Error(`No configuration found for chain ${chainId}`);
        }
        return chainConfig;
    }

    private getNestedValue(obj: any, path: string): any {
        return path.split('.').reduce((current, key) => 
            current && current[key] !== undefined ? current[key] : undefined,
            obj
        );
    }

    private deepMerge(target: any, source: any): any {
        const output = { ...target };
        if (this.isObject(target) && this.isObject(source)) {
            Object.keys(source).forEach(key => {
                if (this.isObject(source[key])) {
                    if (!(key in target)) {
                        Object.assign(output, { [key]: source[key] });
                    } else {
                        output[key] = this.deepMerge(target[key], source[key]);
                    }
                } else {
                    Object.assign(output, { [key]: source[key] });
                }
            });
        }
        return output;
    }

    private isObject(item: any): boolean {
        return item && typeof item === 'object' && !Array.isArray(item);
    }

    public async reload(): Promise<void> {
        this.loadConfig();
        this.emit('configReloaded');
    }

    public getPrivateKey(): string {
        const privateKey = this.envVars.get('PRIVATE_KEY');
        if (!privateKey) {
            throw new Error('Private key not found in environment variables');
        }
        return privateKey;
    }

    public getRpcUrl(chainId: number, index: number = 0): string {
        const chainConfig = this.getChainConfig(chainId);
        if (!chainConfig.rpcUrls || !chainConfig.rpcUrls[index]) {
            throw new Error(`RPC URL not found for chain ${chainId} at index ${index}`);
        }
        return chainConfig.rpcUrls[index];
    }
}
export interface ArbitrageOpportunity {
  id: string;
  tokenIn: string;
  tokenOut: string;
  amountIn: string;
  expectedProfit: string;
  path: {
    dex: string;
    pool: string;
    tokenIn: string;
    tokenOut: string;
  }[];
  timestamp: string;
}

export interface ExecutionResult {
  success: boolean;
  profit: string;
  gasUsed: string;
  executionTime: number;
  error?: string;
  txHash?: string;
}

export interface MetricsSnapshot {
  timestamp: string;
  totalProfit: string;
  totalTrades: number;
  successfulTrades: number;
  failedTrades: number;
  gasSpent: string;
  avgProfitPerTrade: string;
  successRate: number;
  mevAttacksPrevented: number;
  currentGasPrice: number;
  networkLoad: number;
  activeOpportunities: number;
  pendingTransactions: number;
}

export interface Alert {
  id: string;
  level: 'info' | 'warning' | 'error';
  source: string;
  message: string;
  timestamp: string;
}

export interface ChainState {
  chainId: number;
  blockNumber: number;
  gasPrice: string;
  networkId: number;
  isSyncing: boolean;
  providerHealth: Record<string, boolean>;
}

export interface PoolState {
  address: string;
  token0: string;
  token1: string;
  reserve0: string;
  reserve1: string;
  fee: number;
  volume24h: string;
  tvl: string;
}

export interface TokenInfo {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  price: string;
  volume24h: string;
  liquidity: string;
}

export interface SystemStatus {
  status: 'running' | 'paused' | 'error';
  uptime: string;
  lastUpdate: string;
  components: {
    flashLoanExecutor: boolean;
    strategyOptimizer: boolean;
    marketAnalyzer: boolean;
    parameterTuner: boolean;
  };
}
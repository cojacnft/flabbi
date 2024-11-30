import axios from 'axios';
import { 
  ArbitrageOpportunity,
  ExecutionResult,
  MetricsSnapshot,
  Alert,
  ChainState,
  SystemStatus,
  PoolState,
  TokenInfo
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// WebSocket connection
let ws: WebSocket | null = null;
const wsSubscribers: Set<(data: any) => void> = new Set();

export const connectWebSocket = () => {
  if (ws) return;

  ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    wsSubscribers.forEach(callback => callback(data));
  };

  ws.onclose = () => {
    ws = null;
    setTimeout(connectWebSocket, 1000);
  };
};

export const subscribeToWebSocket = (callback: (data: any) => void) => {
  wsSubscribers.add(callback);
  return () => wsSubscribers.delete(callback);
};

// API endpoints
export const apiService = {
  // Metrics
  async getCurrentMetrics(): Promise<MetricsSnapshot> {
    const { data } = await api.get('/api/metrics');
    return data;
  },

  async getMetricsHistory(
    startTime?: string,
    endTime?: string
  ): Promise<MetricsSnapshot[]> {
    const { data } = await api.get('/api/metrics/history', {
      params: { startTime, endTime }
    });
    return data;
  },

  // Opportunities
  async getActiveOpportunities(): Promise<ArbitrageOpportunity[]> {
    const { data } = await api.get('/api/opportunities');
    return data;
  },

  async getOpportunityHistory(
    startTime?: string,
    endTime?: string
  ): Promise<ArbitrageOpportunity[]> {
    const { data } = await api.get('/api/opportunities/history', {
      params: { startTime, endTime }
    });
    return data;
  },

  // Executions
  async getExecutionResults(
    startTime?: string,
    endTime?: string
  ): Promise<ExecutionResult[]> {
    const { data } = await api.get('/api/executions', {
      params: { startTime, endTime }
    });
    return data;
  },

  // Alerts
  async getAlerts(
    level?: 'info' | 'warning' | 'error',
    startTime?: string,
    endTime?: string
  ): Promise<Alert[]> {
    const { data } = await api.get('/api/alerts', {
      params: { level, startTime, endTime }
    });
    return data;
  },

  // Chain state
  async getChainState(): Promise<ChainState> {
    const { data } = await api.get('/api/chain/state');
    return data;
  },

  // System status
  async getSystemStatus(): Promise<SystemStatus> {
    const { data } = await api.get('/api/status');
    return data;
  },

  // Pool data
  async getPoolState(poolAddress: string): Promise<PoolState> {
    const { data } = await api.get(`/api/pools/${poolAddress}`);
    return data;
  },

  async getPoolStates(
    poolAddresses: string[]
  ): Promise<Record<string, PoolState>> {
    const { data } = await api.get('/api/pools', {
      params: { addresses: poolAddresses.join(',') }
    });
    return data;
  },

  // Token data
  async getTokenInfo(tokenAddress: string): Promise<TokenInfo> {
    const { data } = await api.get(`/api/tokens/${tokenAddress}`);
    return data;
  },

  async getTokenInfos(
    tokenAddresses: string[]
  ): Promise<Record<string, TokenInfo>> {
    const { data } = await api.get('/api/tokens', {
      params: { addresses: tokenAddresses.join(',') }
    });
    return data;
  },

  // System control
  async pauseSystem(): Promise<void> {
    await api.post('/api/system/pause');
  },

  async resumeSystem(): Promise<void> {
    await api.post('/api/system/resume');
  },

  async updateSettings(settings: Record<string, any>): Promise<void> {
    await api.post('/api/settings', settings);
  }
};

// Error handling
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      // Server error response
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // No response received
      console.error('Network Error:', error.request);
    } else {
      // Request setup error
      console.error('Request Error:', error.message);
    }
    return Promise.reject(error);
  }
);
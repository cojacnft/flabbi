import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  LocalGasStation as GasIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import { formatUSD, formatEther, formatPercentage } from '../../utils/format';
import { apiService } from '../../services/api';

interface PerformanceMetrics {
  totalProfit: string;
  totalGasSpent: string;
  successRate: number;
  averageExecutionTime: number;
  totalTrades: number;
  failedTrades: number;
  mevAttacksPrevented: number;
  profitHistory: {
    timestamp: string;
    profit: string;
    gasSpent: string;
  }[];
  tokenPerformance: {
    token: string;
    symbol: string;
    profit: string;
    trades: number;
  }[];
  dexPerformance: {
    dex: string;
    profit: string;
    trades: number;
    successRate: number;
  }[];
}

const PerformanceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState('24h');
  const [refreshInterval, setRefreshInterval] = useState(60);

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [timeframe, refreshInterval]);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const data = await apiService.getPerformanceMetrics(timeframe);
      setMetrics(data);
      setError(null);
    } catch (err) {
      console.error('Error loading metrics:', err);
      setError('Failed to load performance metrics');
    } finally {
      setLoading(false);
    }
  };

  const renderMetricCard = (
    title: string,
    value: string | number,
    icon: React.ReactNode,
    color: string,
    subtitle?: string
  ) => (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{
            p: 1,
            borderRadius: 1,
            bgcolor: `${color}20`,
            color: color,
            mr: 2
          }}>
            {icon}
          </Box>
          <Typography variant="h6" color="text.secondary">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" sx={{ color }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (loading && !metrics) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!metrics) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Failed to load performance metrics
        </Alert>
      </Box>
    );
  }

  const profitChartData = {
    labels: metrics.profitHistory.map(h => new Date(h.timestamp).toLocaleString()),
    datasets: [
      {
        label: 'Profit',
        data: metrics.profitHistory.map(h => parseFloat(h.profit)),
        borderColor: '#4caf50',
        tension: 0.1,
      },
      {
        label: 'Gas Cost',
        data: metrics.profitHistory.map(h => parseFloat(h.gasSpent)),
        borderColor: '#f44336',
        tension: 0.1,
      },
    ],
  };

  const tokenChartData = {
    labels: metrics.tokenPerformance.map(t => t.symbol),
    datasets: [
      {
        data: metrics.tokenPerformance.map(t => parseFloat(t.profit)),
        backgroundColor: [
          '#4caf50',
          '#2196f3',
          '#ff9800',
          '#9c27b0',
          '#f44336',
        ],
      },
    ],
  };

  const dexChartData = {
    labels: metrics.dexPerformance.map(d => d.dex),
    datasets: [
      {
        label: 'Profit',
        data: metrics.dexPerformance.map(d => parseFloat(d.profit)),
        backgroundColor: '#4caf50',
      },
      {
        label: 'Success Rate',
        data: metrics.dexPerformance.map(d => d.successRate * 100),
        backgroundColor: '#2196f3',
      },
    ],
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h5">
          Performance Dashboard
        </Typography>
        <Box>
          <FormControl size="small" sx={{ mr: 2, minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              label="Timeframe"
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
              <MenuItem value="30d">30 Days</MenuItem>
            </Select>
          </FormControl>
          <Tooltip title="Refresh">
            <IconButton onClick={loadMetrics} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={3}>
          {renderMetricCard(
            'Total Profit',
            formatUSD(metrics.totalProfit),
            <TrendingUpIcon />,
            '#4caf50',
            `${metrics.totalTrades} trades`
          )}
        </Grid>
        <Grid item xs={12} md={3}>
          {renderMetricCard(
            'Gas Spent',
            formatEther(metrics.totalGasSpent),
            <GasIcon />,
            '#f44336',
            'ETH'
          )}
        </Grid>
        <Grid item xs={12} md={3}>
          {renderMetricCard(
            'Success Rate',
            formatPercentage(metrics.successRate),
            <AssessmentIcon />,
            '#2196f3',
            `${metrics.failedTrades} failed trades`
          )}
        </Grid>
        <Grid item xs={12} md={3}>
          {renderMetricCard(
            'MEV Protected',
            metrics.mevAttacksPrevented,
            <SpeedIcon />,
            '#ff9800',
            'attacks prevented'
          )}
        </Grid>

        {/* Profit History Chart */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Profit History
            </Typography>
            <Box sx={{ height: 300 }}>
              <Line
                data={profitChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  interaction: {
                    mode: 'index',
                    intersect: false,
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'USD',
                      },
                    },
                  },
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Token Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Token Performance
            </Typography>
            <Box sx={{ height: 300 }}>
              <Doughnut
                data={tokenChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'right',
                    },
                  },
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* DEX Performance */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              DEX Performance
            </Typography>
            <Box sx={{ height: 300 }}>
              <Bar
                data={dexChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'USD / Success Rate (%)',
                      },
                    },
                  },
                }}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Performance Details */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Performance Details
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Top Performing Tokens
                </Typography>
                {metrics.tokenPerformance.slice(0, 5).map((token, i) => (
                  <Box key={i} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">
                        {token.symbol}
                      </Typography>
                      <Typography variant="body2" color="success.main">
                        {formatUSD(token.profit)}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={
                        (parseFloat(token.profit) /
                          parseFloat(metrics.tokenPerformance[0].profit)) *
                        100
                      }
                      sx={{ height: 4, borderRadius: 2 }}
                    />
                  </Box>
                ))}
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  DEX Success Rates
                </Typography>
                {metrics.dexPerformance.map((dex, i) => (
                  <Box key={i} sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">
                        {dex.dex}
                      </Typography>
                      <Typography variant="body2">
                        {formatPercentage(dex.successRate)}
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={dex.successRate * 100}
                      sx={{ height: 4, borderRadius: 2 }}
                    />
                  </Box>
                ))}
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PerformanceDashboard;
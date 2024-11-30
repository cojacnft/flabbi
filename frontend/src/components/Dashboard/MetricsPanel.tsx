import React from 'react';
import { Box, Grid, Typography } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { MetricsSnapshot } from '../../types';
import { formatEther, formatUSD } from '../../utils/format';

interface MetricsPanelProps {
  metrics: MetricsSnapshot;
}

const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics }) => {
  const chartData = {
    labels: ['1h ago', '45m ago', '30m ago', '15m ago', 'Now'],
    datasets: [
      {
        label: 'Profit (USD)',
        data: [0, metrics.totalProfit],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
      {
        label: 'Gas Cost (ETH)',
        data: [0, metrics.gasSpent],
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Performance Metrics
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} md={4}>
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary">
              Total Profit
            </Typography>
            <Typography variant="h4" color="success.main">
              {formatUSD(metrics.totalProfit)}
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary">
              Success Rate
            </Typography>
            <Typography variant="h4">
              {(metrics.successRate * 100).toFixed(1)}%
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" color="text.secondary">
              Gas Spent
            </Typography>
            <Typography variant="h4" color="error.main">
              {formatEther(metrics.gasSpent)} ETH
            </Typography>
          </Box>
        </Grid>

        {/* Additional Metrics */}
        <Grid item xs={6} md={3}>
          <Box sx={{ p: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Total Trades
            </Typography>
            <Typography variant="h6">
              {metrics.totalTrades}
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={6} md={3}>
          <Box sx={{ p: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Avg Profit/Trade
            </Typography>
            <Typography variant="h6">
              {formatUSD(metrics.avgProfitPerTrade)}
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={6} md={3}>
          <Box sx={{ p: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              MEV Prevented
            </Typography>
            <Typography variant="h6">
              {metrics.mevAttacksPrevented}
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={6} md={3}>
          <Box sx={{ p: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Active Opportunities
            </Typography>
            <Typography variant="h6">
              {metrics.activeOpportunities}
            </Typography>
          </Box>
        </Grid>

        {/* Chart */}
        <Grid item xs={12}>
          <Box sx={{ height: 300 }}>
            <Line data={chartData} options={chartOptions} />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MetricsPanel;
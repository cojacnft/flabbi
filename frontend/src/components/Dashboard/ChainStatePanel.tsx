import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  CircularProgress,
  LinearProgress,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  CloudQueue as CloudIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { ChainState } from '../../types';
import { apiService } from '../../services/api';
import { formatGwei } from '../../utils/format';

const ChainStatePanel: React.FC = () => {
  const [chainState, setChainState] = useState<ChainState | null>(null);
  const [loading, setLoading] = useState(true);

  const loadChainState = async () => {
    try {
      setLoading(true);
      const data = await apiService.getChainState();
      setChainState(data);
    } catch (error) {
      console.error('Error loading chain state:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadChainState();
    const interval = setInterval(loadChainState, 15000);

    return () => clearInterval(interval);
  }, []);

  if (loading && !chainState) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!chainState) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="error">
          Error loading chain state
        </Typography>
      </Box>
    );
  }

  const getProviderHealth = () => {
    const healthyCount = Object.values(chainState.providerHealth).filter(
      Boolean
    ).length;
    const total = Object.keys(chainState.providerHealth).length;
    return (healthyCount / total) * 100;
  };

  const providerHealth = getProviderHealth();

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">
          Network Status
        </Typography>
        <IconButton size="small" onClick={loadChainState}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      </Box>

      <Grid container spacing={2}>
        {/* Gas Price */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <SpeedIcon color="action" sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              Gas Price
            </Typography>
          </Box>
          <Typography variant="h5">
            {formatGwei(chainState.gasPrice)} gwei
          </Typography>
        </Grid>

        {/* Block Number */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <MemoryIcon color="action" sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              Block Number
            </Typography>
          </Box>
          <Typography variant="h5">
            #{chainState.blockNumber.toLocaleString()}
          </Typography>
        </Grid>

        {/* Provider Health */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <CloudIcon color="action" sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              Provider Health
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box sx={{ flex: 1, mr: 1 }}>
              <Tooltip
                title={
                  Object.entries(chainState.providerHealth)
                    .map(([name, health]) => `${name}: ${health ? '✓' : '✗'}`)
                    .join('\n')
                }
              >
                <LinearProgress
                  variant="determinate"
                  value={providerHealth}
                  color={providerHealth > 80 ? 'success' : 'warning'}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Tooltip>
            </Box>
            <Typography variant="body2" color="text.secondary">
              {providerHealth.toFixed(0)}%
            </Typography>
          </Box>
        </Grid>

        {/* Network Status */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Network
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography>
              Chain ID: {chainState.chainId}
            </Typography>
            <Typography
              color={chainState.isSyncing ? 'warning.main' : 'success.main'}
            >
              {chainState.isSyncing ? 'Syncing...' : 'Synced'}
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ChainStatePanel;
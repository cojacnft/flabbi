import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Button,
  Chip,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { SystemStatus } from '../../types';
import { apiService } from '../../services/api';

interface SystemStatusPanelProps {
  status: SystemStatus;
}

const SystemStatusPanel: React.FC<SystemStatusPanelProps> = ({ status }) => {
  const handleToggleSystem = async () => {
    try {
      if (status.status === 'running') {
        await apiService.pauseSystem();
      } else {
        await apiService.resumeSystem();
      }
    } catch (error) {
      console.error('Error toggling system:', error);
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'running':
        return 'success';
      case 'paused':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = () => {
    switch (status.status) {
      case 'running':
        return <CheckCircleIcon />;
      case 'paused':
        return <WarningIcon />;
      case 'error':
        return <ErrorIcon />;
      default:
        return null;
    }
  };

  const getComponentHealth = () => {
    const total = Object.keys(status.components).length;
    const healthy = Object.values(status.components).filter(Boolean).length;
    return (healthy / total) * 100;
  };

  return (
    <Box>
      <Grid container spacing={2} alignItems="center">
        {/* Status and Controls */}
        <Grid item xs={12} md={4}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              icon={getStatusIcon()}
              label={status.status.toUpperCase()}
              color={getStatusColor()}
            />
            <Button
              variant="contained"
              startIcon={status.status === 'running' ? <PauseIcon /> : <PlayIcon />}
              onClick={handleToggleSystem}
              color={status.status === 'running' ? 'warning' : 'success'}
            >
              {status.status === 'running' ? 'Pause' : 'Resume'}
            </Button>
          </Box>
        </Grid>

        {/* Uptime */}
        <Grid item xs={12} md={4}>
          <Typography variant="body2" color="text.secondary">
            Uptime
          </Typography>
          <Typography variant="h6">
            {status.uptime}
          </Typography>
        </Grid>

        {/* Last Update */}
        <Grid item xs={12} md={4}>
          <Typography variant="body2" color="text.secondary">
            Last Update
          </Typography>
          <Typography variant="h6">
            {new Date(status.lastUpdate).toLocaleTimeString()}
          </Typography>
        </Grid>

        {/* Component Health */}
        <Grid item xs={12}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Component Health
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box sx={{ flex: 1 }}>
              <Tooltip
                title={
                  Object.entries(status.components)
                    .map(([name, health]) => `${name}: ${health ? '✓' : '✗'}`)
                    .join('\n')
                }
              >
                <LinearProgress
                  variant="determinate"
                  value={getComponentHealth()}
                  color={getComponentHealth() === 100 ? 'success' : 'warning'}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Tooltip>
            </Box>
            <Typography variant="body2" color="text.secondary">
              {getComponentHealth().toFixed(0)}% Healthy
            </Typography>
          </Box>
        </Grid>

        {/* Component Details */}
        <Grid item xs={12}>
          <Grid container spacing={1}>
            {Object.entries(status.components).map(([name, health]) => (
              <Grid item xs={6} sm={3} key={name}>
                <Chip
                  label={name.replace(/([A-Z])/g, ' $1').trim()}
                  icon={health ? <CheckCircleIcon /> : <ErrorIcon />}
                  color={health ? 'success' : 'error'}
                  variant="outlined"
                  size="small"
                  sx={{ width: '100%' }}
                />
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemStatusPanel;
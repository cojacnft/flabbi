import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Switch,
  FormControlLabel,
  Alert,
} from '@mui/material';
import { RocketLaunch, Speed } from '@mui/icons-material';

interface ResourceMetrics {
  cpu_usage: {
    current: number;
    average: number;
    max: number;
  };
  memory_usage: {
    current: number;
    average: number;
    max: number;
  };
  thread_pool: {
    size: number;
    active: number;
    tasks_pending: number;
  };
  batch_size: number;
  operations_per_second: number;
}

const PerformanceControl: React.FC = () => {
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [isBoostMode, setIsBoostMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/performance/metrics');
        const data = await response.json();
        setMetrics(data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch metrics');
      }
    };

    const interval = setInterval(fetchMetrics, 1000);
    return () => clearInterval(interval);
  }, []);

  // Handle boost mode toggle
  const handleBoostToggle = async () => {
    try {
      const mode = !isBoostMode ? 'boost' : 'background';
      const response = await fetch(`/api/performance/mode/${mode}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to change performance mode');
      }
      
      setIsBoostMode(!isBoostMode);
      setError(null);
    } catch (err) {
      setError('Failed to change performance mode');
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            System Performance
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={isBoostMode}
                onChange={handleBoostToggle}
                color="primary"
              />
            }
            label={
              <Box display="flex" alignItems="center">
                <RocketLaunch
                  color={isBoostMode ? 'primary' : 'disabled'}
                  sx={{ mr: 1 }}
                />
                Boost Mode
              </Box>
            }
          />
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {metrics && (
          <>
            <Box mb={2}>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                CPU Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.cpu_usage.current}
                color={metrics.cpu_usage.current > 80 ? 'error' : 'primary'}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption" color="textSecondary">
                {`${metrics.cpu_usage.current.toFixed(1)}% (Avg: ${metrics.cpu_usage.average.toFixed(1)}%)`}
              </Typography>
            </Box>

            <Box mb={2}>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Memory Usage
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.memory_usage.current}
                color={metrics.memory_usage.current > 80 ? 'error' : 'primary'}
                sx={{ height: 10, borderRadius: 5 }}
              />
              <Typography variant="caption" color="textSecondary">
                {`${metrics.memory_usage.current.toFixed(1)}% (Avg: ${metrics.memory_usage.average.toFixed(1)}%)`}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2" color="textSecondary">
                Active Threads:
              </Typography>
              <Typography variant="body2">
                {`${metrics.thread_pool.active} / ${metrics.thread_pool.size}`}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2" color="textSecondary">
                Pending Tasks:
              </Typography>
              <Typography variant="body2">
                {metrics.thread_pool.tasks_pending}
              </Typography>
            </Box>

            <Box display="flex" justifyContent="space-between" mb={1}>
              <Typography variant="body2" color="textSecondary">
                Operations/sec:
              </Typography>
              <Typography variant="body2">
                {metrics.operations_per_second}
              </Typography>
            </Box>

            <Box mt={2} display="flex" alignItems="center">
              <Speed color="action" sx={{ mr: 1 }} />
              <Typography variant="body2" color="textSecondary">
                {isBoostMode ? 'High Performance Mode' : 'Background Mode'}
              </Typography>
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default PerformanceControl;
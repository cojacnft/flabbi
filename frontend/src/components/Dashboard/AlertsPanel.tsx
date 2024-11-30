import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Paper,
  Divider,
  Chip,
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { Alert } from '../../types';
import { apiService } from '../../services/api';
import { formatTimeAgo } from '../../utils/format';

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    const loadAlerts = async () => {
      try {
        const data = await apiService.getAlerts();
        setAlerts(data);
      } catch (error) {
        console.error('Error loading alerts:', error);
      }
    };

    loadAlerts();
    const interval = setInterval(loadAlerts, 30000);

    return () => clearInterval(interval);
  }, []);

  const getAlertIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const handleDismiss = (alertId: string) => {
    setAlerts(alerts.filter(alert => alert.id !== alertId));
  };

  const getSourceColor = (source: string) => {
    switch (source) {
      case 'flash_loan':
        return 'primary';
      case 'strategy':
        return 'secondary';
      case 'market':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">
          System Alerts
        </Typography>
        <Box>
          <Chip
            label={`${alerts.filter(a => a.level === 'error').length} Errors`}
            color="error"
            size="small"
            sx={{ mr: 1 }}
          />
          <Chip
            label={`${alerts.filter(a => a.level === 'warning').length} Warnings`}
            color="warning"
            size="small"
          />
        </Box>
      </Box>

      <Paper
        variant="outlined"
        sx={{
          maxHeight: 400,
          overflow: 'auto',
        }}
      >
        <List dense>
          {alerts.map((alert, index) => (
            <React.Fragment key={alert.id}>
              {index > 0 && <Divider />}
              <ListItem
                secondaryAction={
                  <IconButton
                    edge="end"
                    size="small"
                    onClick={() => handleDismiss(alert.id)}
                  >
                    <ClearIcon fontSize="small" />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  {getAlertIcon(alert.level)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {alert.message}
                      <Chip
                        label={alert.source}
                        color={getSourceColor(alert.source)}
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    </Box>
                  }
                  secondary={formatTimeAgo(alert.timestamp)}
                />
              </ListItem>
            </React.Fragment>
          ))}
          {alerts.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No alerts"
                secondary="System is running normally"
              />
            </ListItem>
          )}
        </List>
      </Paper>
    </Box>
  );
};

export default AlertsPanel;
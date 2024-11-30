import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Slider,
  Select,
  MenuItem,
  Alert,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Speed,
  Security,
  LocalGasStation,
  NetworkCheck,
  Route,
  Warning,
  Save,
  Refresh,
  Info,
} from '@mui/icons-material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ padding: '20px 0' }}>
    {value === index && children}
  </div>
);

const SettingsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [settings, setSettings] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Fetch settings
  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/settings');
      const data = await response.json();
      setSettings(data);
      setError(null);
    } catch (err) {
      setError('Failed to load settings');
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        throw new Error('Failed to save settings');
      }

      setSuccess('Settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to save settings');
    }
  };

  const handleReset = async () => {
    try {
      const response = await fetch('/api/settings/reset', {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to reset settings');
      }

      await fetchSettings();
      setSuccess('Settings reset to defaults');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError('Failed to reset settings');
    }
  };

  if (!settings) {
    return <Typography>Loading settings...</Typography>;
  }

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">System Settings</Typography>
          <Box>
            <Button
              startIcon={<Save />}
              variant="contained"
              color="primary"
              onClick={handleSave}
              sx={{ mr: 1 }}
            >
              Save
            </Button>
            <Button
              startIcon={<Refresh />}
              variant="outlined"
              onClick={handleReset}
            >
              Reset
            </Button>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {success}
          </Alert>
        )}

        <Tabs
          value={activeTab}
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab icon={<Speed />} label="Performance" />
          <Tab icon={<Security />} label="MEV Protection" />
          <Tab icon={<LocalGasStation />} label="Gas" />
          <Tab icon={<NetworkCheck />} label="Network" />
          <Tab icon={<Route />} label="Path Finding" />
          <Tab icon={<Warning />} label="Risk Management" />
        </Tabs>

        {/* Performance Settings */}
        <TabPanel value={activeTab} index={0}>
          <FormControlLabel
            control={
              <Select
                value={settings.performance.mode}
                onChange={(e) => setSettings({
                  ...settings,
                  performance: {
                    ...settings.performance,
                    mode: e.target.value,
                  },
                })}
                size="small"
              >
                <MenuItem value="background">Background</MenuItem>
                <MenuItem value="balanced">Balanced</MenuItem>
                <MenuItem value="boost">Boost</MenuItem>
              </Select>
            }
            label="Performance Mode"
            labelPlacement="start"
          />

          <Box mt={2}>
            <Typography gutterBottom>
              Max CPU Usage (%)
              <Tooltip title="Maximum CPU usage allowed">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <Slider
              value={settings.performance.max_cpu_percent}
              onChange={(_, value) => setSettings({
                ...settings,
                performance: {
                  ...settings.performance,
                  max_cpu_percent: value,
                },
              })}
              min={10}
              max={90}
              valueLabelDisplay="auto"
            />
          </Box>

          <Box mt={2}>
            <Typography gutterBottom>
              Max Memory Usage (%)
              <Tooltip title="Maximum memory usage allowed">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <Slider
              value={settings.performance.max_memory_percent}
              onChange={(_, value) => setSettings({
                ...settings,
                performance: {
                  ...settings.performance,
                  max_memory_percent: value,
                },
              })}
              min={20}
              max={90}
              valueLabelDisplay="auto"
            />
          </Box>
        </TabPanel>

        {/* MEV Protection Settings */}
        <TabPanel value={activeTab} index={1}>
          <FormControlLabel
            control={
              <Switch
                checked={settings.mev_protection.enabled}
                onChange={(e) => setSettings({
                  ...settings,
                  mev_protection: {
                    ...settings.mev_protection,
                    enabled: e.target.checked,
                  },
                })}
              />
            }
            label="Enable MEV Protection"
          />

          <Box mt={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.mev_protection.flashbots_enabled}
                  onChange={(e) => setSettings({
                    ...settings,
                    mev_protection: {
                      ...settings.mev_protection,
                      flashbots_enabled: e.target.checked,
                    },
                  })}
                />
              }
              label="Use Flashbots"
            />
          </Box>

          <Box mt={2}>
            <Typography gutterBottom>
              Minimum Bribe (% of profit)
              <Tooltip title="Minimum bribe percentage for Flashbots bundles">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <Slider
              value={settings.mev_protection.min_bribe_percentage * 100}
              onChange={(_, value) => setSettings({
                ...settings,
                mev_protection: {
                  ...settings.mev_protection,
                  min_bribe_percentage: (value as number) / 100,
                },
              })}
              min={1}
              max={20}
              valueLabelDisplay="auto"
            />
          </Box>
        </TabPanel>

        {/* Gas Optimization Settings */}
        <TabPanel value={activeTab} index={2}>
          <FormControlLabel
            control={
              <Switch
                checked={settings.gas_optimization.enabled}
                onChange={(e) => setSettings({
                  ...settings,
                  gas_optimization: {
                    ...settings.gas_optimization,
                    enabled: e.target.checked,
                  },
                })}
              />
            }
            label="Enable Gas Optimization"
          />

          <Box mt={2}>
            <Typography gutterBottom>
              Max Gas Price (Gwei)
              <Tooltip title="Maximum gas price to pay for transactions">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <TextField
              type="number"
              value={settings.gas_optimization.max_gas_price_gwei}
              onChange={(e) => setSettings({
                ...settings,
                gas_optimization: {
                  ...settings.gas_optimization,
                  max_gas_price_gwei: parseInt(e.target.value),
                },
              })}
              size="small"
            />
          </Box>
        </TabPanel>

        {/* Network Settings */}
        <TabPanel value={activeTab} index={3}>
          <FormControlLabel
            control={
              <Switch
                checked={settings.network.websocket_enabled}
                onChange={(e) => setSettings({
                  ...settings,
                  network: {
                    ...settings.network,
                    websocket_enabled: e.target.checked,
                  },
                })}
              />
            }
            label="Enable WebSocket Connections"
          />

          <Box mt={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.network.auto_failover}
                  onChange={(e) => setSettings({
                    ...settings,
                    network: {
                      ...settings.network,
                      auto_failover: e.target.checked,
                    },
                  })}
                />
              }
              label="Automatic Failover"
            />
          </Box>
        </TabPanel>

        {/* Path Finding Settings */}
        <TabPanel value={activeTab} index={4}>
          <Box mt={2}>
            <Typography gutterBottom>
              Maximum Hops
              <Tooltip title="Maximum number of trades in a path">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <Select
              value={settings.path_finding.max_hops}
              onChange={(e) => setSettings({
                ...settings,
                path_finding: {
                  ...settings.path_finding,
                  max_hops: e.target.value,
                },
              })}
              size="small"
            >
              {[2, 3, 4, 5, 6].map((n) => (
                <MenuItem key={n} value={n}>{n}</MenuItem>
              ))}
            </Select>
          </Box>

          <Box mt={2}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.path_finding.ml_prediction_enabled}
                  onChange={(e) => setSettings({
                    ...settings,
                    path_finding: {
                      ...settings.path_finding,
                      ml_prediction_enabled: e.target.checked,
                    },
                  })}
                />
              }
              label="Enable ML Predictions"
            />
          </Box>
        </TabPanel>

        {/* Risk Management Settings */}
        <TabPanel value={activeTab} index={5}>
          <Box mt={2}>
            <Typography gutterBottom>
              Max Position Size (USD)
              <Tooltip title="Maximum position size per trade">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <TextField
              type="number"
              value={settings.risk_management.max_position_size_usd}
              onChange={(e) => setSettings({
                ...settings,
                risk_management: {
                  ...settings.risk_management,
                  max_position_size_usd: parseFloat(e.target.value),
                },
              })}
              size="small"
            />
          </Box>

          <Box mt={2}>
            <Typography gutterBottom>
              Max Daily Loss (USD)
              <Tooltip title="Maximum allowed loss per day">
                <Info fontSize="small" sx={{ ml: 1 }} />
              </Tooltip>
            </Typography>
            <TextField
              type="number"
              value={settings.risk_management.max_daily_loss_usd}
              onChange={(e) => setSettings({
                ...settings,
                risk_management: {
                  ...settings.risk_management,
                  max_daily_loss_usd: parseFloat(e.target.value),
                },
              })}
              size="small"
            />
          </Box>
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default SettingsPanel;
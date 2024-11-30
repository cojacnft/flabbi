import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  Tooltip,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { PoolState } from '../../types';
import { apiService } from '../../services/api';
import { formatUSD, formatEther, shortenAddress } from '../../utils/format';

interface PoolDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (pool: Partial<PoolState>) => void;
  pool?: PoolState;
}

const PoolDialog: React.FC<PoolDialogProps> = ({
  open,
  onClose,
  onSubmit,
  pool,
}) => {
  const [address, setAddress] = useState(pool?.address || '');
  const [token0, setToken0] = useState(pool?.token0 || '');
  const [token1, setToken1] = useState(pool?.token1 || '');
  const [fee, setFee] = useState(pool?.fee || 0.3);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    try {
      setError(null);
      setLoading(true);

      await onSubmit({
        address,
        token0,
        token1,
        fee,
      });

      onClose();
    } catch (err) {
      setError('Failed to save pool');
      console.error('Error saving pool:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {pool ? 'Edit Pool' : 'Add New Pool'}
      </DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Pool Address"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              disabled={!!pool}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Token 0 Address"
              value={token0}
              onChange={(e) => setToken0(e.target.value)}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Token 1 Address"
              value={token1}
              onChange={(e) => setToken1(e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Fee Tier</InputLabel>
              <Select
                value={fee}
                label="Fee Tier"
                onChange={(e) => setFee(Number(e.target.value))}
              >
                <MenuItem value={0.01}>0.01%</MenuItem>
                <MenuItem value={0.05}>0.05%</MenuItem>
                <MenuItem value={0.3}>0.3%</MenuItem>
                <MenuItem value={1}>1%</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading}
          startIcon={loading && <CircularProgress size={20} />}
        >
          {pool ? 'Update' : 'Add'} Pool
        </Button>
      </DialogActions>
    </Dialog>
  );
};

const PoolManagement: React.FC = () => {
  const [pools, setPools] = useState<PoolState[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedPool, setSelectedPool] = useState<PoolState | undefined>();
  const [refreshing, setRefreshing] = useState<Record<string, boolean>>({});
  const [filters, setFilters] = useState({
    dex: '',
    minLiquidity: '',
    search: '',
  });

  useEffect(() => {
    loadPools();
  }, []);

  const loadPools = async () => {
    try {
      setLoading(true);
      const data = await apiService.getPoolStates([]);
      setPools(Object.values(data));
      setError(null);
    } catch (err) {
      setError('Failed to load pools');
      console.error('Error loading pools:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddPool = async (pool: Partial<PoolState>) => {
    try {
      await apiService.addPool(pool.address!, {
        token0: pool.token0!,
        token1: pool.token1!,
        fee: pool.fee!,
      });
      await loadPools();
    } catch (err) {
      console.error('Error adding pool:', err);
      throw err;
    }
  };

  const handleUpdatePool = async (pool: Partial<PoolState>) => {
    try {
      await apiService.updatePool(pool.address!, {
        token0: pool.token0!,
        token1: pool.token1!,
        fee: pool.fee!,
      });
      await loadPools();
    } catch (err) {
      console.error('Error updating pool:', err);
      throw err;
    }
  };

  const handleDeletePool = async (address: string) => {
    try {
      await apiService.deletePool(address);
      await loadPools();
    } catch (err) {
      console.error('Error deleting pool:', err);
    }
  };

  const handleRefreshPool = async (address: string) => {
    try {
      setRefreshing({ ...refreshing, [address]: true });
      await apiService.refreshPool(address);
      await loadPools();
    } catch (err) {
      console.error('Error refreshing pool:', err);
    } finally {
      setRefreshing({ ...refreshing, [address]: false });
    }
  };

  const getHealthStatus = (pool: PoolState) => {
    const tvl = parseFloat(pool.tvl);
    if (tvl > 1000000) {
      return {
        icon: <CheckCircleIcon fontSize="small" />,
        label: 'Healthy',
        color: 'success',
      };
    } else if (tvl > 100000) {
      return {
        icon: <WarningIcon fontSize="small" />,
        label: 'Low Liquidity',
        color: 'warning',
      };
    } else {
      return {
        icon: <ErrorIcon fontSize="small" />,
        label: 'Insufficient Liquidity',
        color: 'error',
      };
    }
  };

  const filteredPools = pools.filter((pool) => {
    if (filters.dex && !pool.address.includes(filters.dex)) return false;
    if (filters.minLiquidity && parseFloat(pool.tvl) < parseFloat(filters.minLiquidity)) return false;
    if (filters.search) {
      const search = filters.search.toLowerCase();
      return (
        pool.address.toLowerCase().includes(search) ||
        pool.token0.toLowerCase().includes(search) ||
        pool.token1.toLowerCase().includes(search)
      );
    }
    return true;
  });

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="h5">Pool Management</Typography>
        <Box>
          <Button
            startIcon={<RefreshIcon />}
            onClick={loadPools}
            sx={{ mr: 1 }}
          >
            Refresh All
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => {
              setSelectedPool(undefined);
              setDialogOpen(true);
            }}
          >
            Add Pool
          </Button>
        </Box>
      </Box>

      {/* Filters */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Search"
            value={filters.search}
            onChange={(e) => setFilters({ ...filters, search: e.target.value })}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            select
            label="DEX"
            value={filters.dex}
            onChange={(e) => setFilters({ ...filters, dex: e.target.value })}
          >
            <MenuItem value="">All DEXs</MenuItem>
            <MenuItem value="uniswap">Uniswap</MenuItem>
            <MenuItem value="sushiswap">Sushiswap</MenuItem>
            <MenuItem value="balancer">Balancer</MenuItem>
          </TextField>
        </Grid>
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Min Liquidity (USD)"
            type="number"
            value={filters.minLiquidity}
            onChange={(e) => setFilters({ ...filters, minLiquidity: e.target.value })}
          />
        </Grid>
      </Grid>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Pool</TableCell>
              <TableCell>Tokens</TableCell>
              <TableCell align="right">TVL</TableCell>
              <TableCell align="right">Volume (24h)</TableCell>
              <TableCell align="right">Fee</TableCell>
              <TableCell align="center">Health</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredPools.map((pool) => {
              const health = getHealthStatus(pool);
              return (
                <TableRow key={pool.address}>
                  <TableCell>
                    <Tooltip title={pool.address}>
                      <Typography variant="body2">
                        {shortenAddress(pool.address)}
                      </Typography>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                      <Typography variant="body2">
                        {shortenAddress(pool.token0)}
                      </Typography>
                      <Typography variant="body2">
                        {shortenAddress(pool.token1)}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    {formatUSD(pool.tvl)}
                  </TableCell>
                  <TableCell align="right">
                    {formatUSD(pool.volume24h)}
                  </TableCell>
                  <TableCell align="right">
                    {pool.fee}%
                  </TableCell>
                  <TableCell align="center">
                    <Chip
                      icon={health.icon}
                      label={health.label}
                      color={health.color}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="Refresh">
                      <IconButton
                        size="small"
                        onClick={() => handleRefreshPool(pool.address)}
                        disabled={refreshing[pool.address]}
                      >
                        {refreshing[pool.address] ? (
                          <CircularProgress size={20} />
                        ) : (
                          <RefreshIcon fontSize="small" />
                        )}
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Edit">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedPool(pool);
                          setDialogOpen(true);
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        onClick={() => handleDeletePool(pool.address)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      <PoolDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={selectedPool ? handleUpdatePool : handleAddPool}
        pool={selectedPool}
      />
    </Box>
  );
};

export default PoolManagement;
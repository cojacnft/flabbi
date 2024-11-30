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
  Collapse,
  Link,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Launch as LaunchIcon,
  ContentCopy as CopyIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { formatDistance } from 'date-fns';
import { apiService } from '../../services/api';
import { formatEther, formatUSD, shortenAddress } from '../../utils/format';

interface Transaction {
  hash: string;
  timestamp: string;
  status: 'success' | 'failed' | 'pending';
  type: 'flashloan' | 'approval' | 'swap';
  from: string;
  to: string;
  value: string;
  gasUsed: string;
  gasPrice: string;
  profit?: string;
  error?: string;
  path?: {
    dex: string;
    tokenIn: string;
    tokenOut: string;
    amountIn: string;
    amountOut: string;
  }[];
}

interface TransactionDetailsProps {
  transaction: Transaction;
  onClose: () => void;
}

const TransactionDetails: React.FC<TransactionDetailsProps> = ({
  transaction,
  onClose,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'failed':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon />;
      case 'failed':
        return <ErrorIcon />;
      default:
        return <WarningIcon />;
    }
  };

  return (
    <Dialog open={true} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        Transaction Details
        <IconButton
          component={Link}
          href={`https://etherscan.io/tx/${transaction.hash}`}
          target="_blank"
          sx={{ ml: 1 }}
        >
          <LaunchIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={2}>
          {/* Status */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                icon={getStatusIcon(transaction.status)}
                label={transaction.status.toUpperCase()}
                color={getStatusColor(transaction.status)}
              />
              {transaction.error && (
                <Typography color="error">
                  Error: {transaction.error}
                </Typography>
              )}
            </Box>
          </Grid>

          {/* Transaction Hash */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Transaction Hash
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                  {transaction.hash}
                </Typography>
                <Tooltip title={copied ? 'Copied!' : 'Copy to clipboard'}>
                  <IconButton
                    size="small"
                    onClick={() => handleCopy(transaction.hash)}
                  >
                    <CopyIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Paper>
          </Grid>

          {/* Basic Info */}
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                From
              </Typography>
              <Typography variant="body2">{transaction.from}</Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                To
              </Typography>
              <Typography variant="body2">{transaction.to}</Typography>
            </Paper>
          </Grid>

          {/* Value and Gas */}
          <Grid item xs={12} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Value
              </Typography>
              <Typography variant="body2">
                {formatEther(transaction.value)} ETH
              </Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Gas Used
              </Typography>
              <Typography variant="body2">
                {formatEther(transaction.gasUsed)} ETH
              </Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Gas Price
              </Typography>
              <Typography variant="body2">
                {formatEther(transaction.gasPrice)} gwei
              </Typography>
            </Paper>
          </Grid>

          {/* Profit (if available) */}
          {transaction.profit && (
            <Grid item xs={12}>
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  bgcolor: 'success.main',
                  color: 'success.contrastText',
                }}
              >
                <Typography variant="subtitle2">Profit</Typography>
                <Typography variant="h5">
                  {formatUSD(transaction.profit)}
                </Typography>
              </Paper>
            </Grid>
          )}

          {/* Path (if available) */}
          {transaction.path && (
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Transaction Path
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>DEX</TableCell>
                      <TableCell>Token In</TableCell>
                      <TableCell>Token Out</TableCell>
                      <TableCell align="right">Amount In</TableCell>
                      <TableCell align="right">Amount Out</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {transaction.path.map((step, index) => (
                      <TableRow key={index}>
                        <TableCell>{step.dex}</TableCell>
                        <TableCell>{shortenAddress(step.tokenIn)}</TableCell>
                        <TableCell>{shortenAddress(step.tokenOut)}</TableCell>
                        <TableCell align="right">
                          {formatEther(step.amountIn)}
                        </TableCell>
                        <TableCell align="right">
                          {formatEther(step.amountOut)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TransactionDetails;
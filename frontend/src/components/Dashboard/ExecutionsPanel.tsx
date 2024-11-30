import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  Link,
} from '@mui/material';
import {
  Launch as LaunchIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { ExecutionResult } from '../../types';
import { apiService } from '../../services/api';
import { formatUSD, formatEther, formatDuration } from '../../utils/format';

const ExecutionsPanel: React.FC = () => {
  const [executions, setExecutions] = useState<ExecutionResult[]>([]);

  useEffect(() => {
    const loadExecutions = async () => {
      try {
        const data = await apiService.getExecutionResults();
        setExecutions(data);
      } catch (error) {
        console.error('Error loading executions:', error);
      }
    };

    loadExecutions();
    const interval = setInterval(loadExecutions, 10000);

    return () => clearInterval(interval);
  }, []);

  const getStatusChip = (success: boolean, error?: string) => {
    if (success) {
      return (
        <Chip
          icon={<CheckCircleIcon fontSize="small" />}
          label="Success"
          color="success"
          size="small"
        />
      );
    }
    return (
      <Tooltip title={error || 'Execution failed'}>
        <Chip
          icon={<ErrorIcon fontSize="small" />}
          label="Failed"
          color="error"
          size="small"
        />
      </Tooltip>
    );
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Recent Executions
      </Typography>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Status</TableCell>
              <TableCell align="right">Profit</TableCell>
              <TableCell align="right">Gas Used</TableCell>
              <TableCell align="right">Time</TableCell>
              <TableCell align="center">Transaction</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {executions.map((execution) => (
              <TableRow key={execution.txHash}>
                <TableCell>
                  {getStatusChip(execution.success, execution.error)}
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    color: execution.success ? 'success.main' : 'text.primary',
                  }}
                >
                  {formatUSD(execution.profit)}
                </TableCell>
                <TableCell align="right">
                  {formatEther(execution.gasUsed)} ETH
                </TableCell>
                <TableCell align="right">
                  {formatDuration(execution.executionTime)}
                </TableCell>
                <TableCell align="center">
                  {execution.txHash && (
                    <Tooltip title="View on Etherscan">
                      <IconButton
                        size="small"
                        component={Link}
                        href={`https://etherscan.io/tx/${execution.txHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <LaunchIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                </TableCell>
              </TableRow>
            ))}
            {executions.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  No executions yet
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ExecutionsPanel;
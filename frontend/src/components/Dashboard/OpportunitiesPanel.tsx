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
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { ArbitrageOpportunity } from '../../types';
import { apiService } from '../../services/api';
import { formatUSD, formatEther, shortenAddress } from '../../utils/format';
import OpportunityDialog from './OpportunityDialog';

const OpportunitiesPanel: React.FC = () => {
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [selectedOpportunity, setSelectedOpportunity] = useState<ArbitrageOpportunity | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    const loadOpportunities = async () => {
      try {
        const data = await apiService.getActiveOpportunities();
        setOpportunities(data);
      } catch (error) {
        console.error('Error loading opportunities:', error);
      }
    };

    loadOpportunities();
    const interval = setInterval(loadOpportunities, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleViewDetails = (opportunity: ArbitrageOpportunity) => {
    setSelectedOpportunity(opportunity);
    setDialogOpen(true);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">
          Active Opportunities
        </Typography>
        <Chip
          label={`${opportunities.length} Found`}
          color="primary"
          size="small"
        />
      </Box>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Path</TableCell>
              <TableCell align="right">Amount In</TableCell>
              <TableCell align="right">Expected Profit</TableCell>
              <TableCell align="right">ROI</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {opportunities.map((opp) => {
              const roi = (
                parseFloat(opp.expectedProfit) /
                parseFloat(opp.amountIn) *
                100
              ).toFixed(2);

              return (
                <TableRow key={opp.id}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TimelineIcon fontSize="small" color="action" />
                      {`${shortenAddress(opp.tokenIn)} → ${shortenAddress(opp.tokenOut)}`}
                    </Box>
                  </TableCell>
                  <TableCell align="right">
                    {formatEther(opp.amountIn)} ETH
                  </TableCell>
                  <TableCell align="right" sx={{ color: 'success.main' }}>
                    {formatUSD(opp.expectedProfit)}
                  </TableCell>
                  <TableCell align="right">
                    {roi}%
                  </TableCell>
                  <TableCell align="center">
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        onClick={() => handleViewDetails(opp)}
                      >
                        <VisibilityIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              );
            })}
            {opportunities.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  No active opportunities
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {selectedOpportunity && (
        <OpportunityDialog
          open={dialogOpen}
          opportunity={selectedOpportunity}
          onClose={() => setDialogOpen(false)}
        />
      )}
    </Box>
  );
};

export default OpportunitiesPanel;
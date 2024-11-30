import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Switch,
  FormControlLabel,
  Slider,
  Card,
  CardContent,
  LinearProgress,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Security as SecurityIcon,
  AccountBalance as AccountBalanceIcon,
  Timeline as TimelineIcon,
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { Line, Radar } from 'react-chartjs-2';
import { formatUSD, formatEther, formatPercentage } from '../../utils/format';
import { apiService } from '../../services/api';
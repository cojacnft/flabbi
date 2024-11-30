import { formatDistance } from 'date-fns';
import { BigNumber } from 'ethers';

export const formatUSD = (value: string | number): string => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(num);
};

export const formatEther = (value: string | number): string => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  return num.toFixed(6);
};

export const formatGwei = (value: string | number): string => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  return num.toFixed(2);
};

export const formatTimeAgo = (timestamp: string): string => {
  return formatDistance(new Date(timestamp), new Date(), { addSuffix: true });
};

export const formatDuration = (seconds: number): string => {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
};

export const shortenAddress = (address: string): string => {
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
};

export const formatNumber = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const parseWei = (value: string | number): BigNumber => {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  return BigNumber.from(num.toString());
};

export const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(2)}%`;
};

export const formatBytes = (bytes: number): string => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  if (bytes === 0) return '0 Byte';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
};
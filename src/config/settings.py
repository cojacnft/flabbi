from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import yaml
import os
from pathlib import Path
from web3 import Web3

class NetworkConfig(BaseModel):
    """Network configuration."""
    rpc_url: str = Field(..., description="RPC endpoint URL")
    ws_url: Optional[str] = Field(None, description="WebSocket endpoint URL")
    chain_id: int = Field(..., description="Chain ID")
    explorer_url: str = Field(..., description="Block explorer URL")
    explorer_api_key: Optional[str] = Field(None, description="Block explorer API key")
    confirmation_blocks: int = Field(1, description="Number of blocks to wait for confirmation")

class FlashLoanConfig(BaseModel):
    """Flash loan configuration."""
    min_profit_usd: float = Field(100.0, description="Minimum profit in USD")
    max_position_size: float = Field(50000.0, description="Maximum position size in USD")
    max_slippage: float = Field(0.01, description="Maximum slippage tolerance")
    min_liquidity: float = Field(100000.0, description="Minimum pool liquidity in USD")
    gas_multiplier: float = Field(1.1, description="Gas estimate multiplier")
    execution_timeout: int = Field(30, description="Execution timeout in seconds")

class MEVConfig(BaseModel):
    """MEV protection configuration."""
    enabled: bool = Field(True, description="Enable MEV protection")
    min_bribe: float = Field(0.1, description="Minimum bribe as fraction of profit")
    max_bribe: float = Field(0.3, description="Maximum bribe as fraction of profit")
    bundle_type: str = Field("flashbots", description="Bundle type (flashbots/eden/standard)")
    backrun_protection: bool = Field(True, description="Enable backrun protection")

class GasConfig(BaseModel):
    """Gas optimization configuration."""
    max_gas_price: int = Field(200, description="Maximum gas price in gwei")
    priority_fee: int = Field(2, description="Priority fee in gwei")
    base_fee_multiplier: float = Field(1.2, description="Base fee multiplier")
    max_gas_limit: int = Field(1000000, description="Maximum gas limit")

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enabled: bool = Field(True, description="Enable monitoring")
    log_level: str = Field("INFO", description="Logging level")
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    alert_enabled: bool = Field(True, description="Enable alerting")
    discord_webhook: Optional[str] = Field(None, description="Discord webhook URL")
    telegram_bot_token: Optional[str] = Field(None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(None, description="Telegram chat ID")

class DatabaseConfig(BaseModel):
    """Database configuration."""
    enabled: bool = Field(True, description="Enable database")
    url: str = Field("sqlite:///data/arbitrage.db", description="Database URL")
    max_entries: int = Field(1000000, description="Maximum database entries")
    backup_enabled: bool = Field(True, description="Enable database backup")
    backup_interval: int = Field(86400, description="Backup interval in seconds")

class ArbitrageConfig(BaseModel):
    """Main configuration."""
    network: NetworkConfig
    flash_loan: FlashLoanConfig
    mev: MEVConfig
    gas: GasConfig
    monitoring: MonitoringConfig
    database: DatabaseConfig
    
    # Contract addresses
    arbitrage_contract: str = Field(..., description="Arbitrage contract address")
    
    # Token addresses
    weth_address: str = Field(
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        description="WETH token address"
    )
    
    # DEX addresses
    approved_dexes: Dict[str, str] = Field(
        default_factory=dict,
        description="Approved DEX router addresses"
    )
    
    # Strategy settings
    max_hops: int = Field(4, description="Maximum hops in arbitrage path")
    min_profit_threshold: float = Field(0.1, description="Minimum profit threshold in ETH")
    max_concurrent_trades: int = Field(3, description="Maximum concurrent trades")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ConfigManager:
    """Configuration manager."""
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv(
            "CONFIG_PATH",
            "config/arbitrage.yaml"
        )
        self.config: Optional[ArbitrageConfig] = None
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            # Load YAML config
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            # Load environment variables
            self._load_env_vars(config_data)
            
            # Create config object
            self.config = ArbitrageConfig(**config_data)
            
            # Validate addresses
            self._validate_addresses()
            
        except Exception as e:
            raise ValueError(f"Error loading config: {str(e)}")

    def _load_env_vars(self, config: Dict):
        """Load environment variables into config."""
        env_vars = {
            "NETWORK_RPC_URL": ("network", "rpc_url"),
            "NETWORK_WS_URL": ("network", "ws_url"),
            "EXPLORER_API_KEY": ("network", "explorer_api_key"),
            "DISCORD_WEBHOOK": ("monitoring", "discord_webhook"),
            "TELEGRAM_BOT_TOKEN": ("monitoring", "telegram_bot_token"),
            "TELEGRAM_CHAT_ID": ("monitoring", "telegram_chat_id"),
            "DATABASE_URL": ("database", "url"),
            "ARBITRAGE_CONTRACT": ("arbitrage_contract",),
        }
        
        for env_var, config_path in env_vars.items():
            value = os.getenv(env_var)
            if value:
                current = config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                current[config_path[-1]] = value

    def _validate_addresses(self):
        """Validate Ethereum addresses."""
        try:
            # Validate contract address
            assert Web3.is_address(self.config.arbitrage_contract), \
                "Invalid arbitrage contract address"
            
            # Validate token addresses
            assert Web3.is_address(self.config.weth_address), \
                "Invalid WETH address"
            
            # Validate DEX addresses
            for name, address in self.config.approved_dexes.items():
                assert Web3.is_address(address), \
                    f"Invalid DEX address for {name}"
            
        except AssertionError as e:
            raise ValueError(f"Address validation failed: {str(e)}")

    def save_config(self):
        """Save configuration to file."""
        try:
            # Convert config to dict
            config_dict = self.config.dict()
            
            # Save to YAML
            with open(self.config_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            
        except Exception as e:
            raise ValueError(f"Error saving config: {str(e)}")

    def update_config(self, updates: Dict):
        """Update configuration values."""
        try:
            # Update config object
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Validate addresses
            self._validate_addresses()
            
            # Save updated config
            self.save_config()
            
        except Exception as e:
            raise ValueError(f"Error updating config: {str(e)}")

    def get_network_config(self) -> NetworkConfig:
        """Get network configuration."""
        return self.config.network

    def get_flash_loan_config(self) -> FlashLoanConfig:
        """Get flash loan configuration."""
        return self.config.flash_loan

    def get_mev_config(self) -> MEVConfig:
        """Get MEV configuration."""
        return self.config.mev

    def get_gas_config(self) -> GasConfig:
        """Get gas configuration."""
        return self.config.gas

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        return self.config.monitoring

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.config.database
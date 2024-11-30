from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_typing import Address
import json
import logging

# AAVE V2 contract addresses on Ethereum mainnet
AAVE_LENDING_POOL = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
AAVE_PROTOCOL_DATA_PROVIDER = "0x057835Ad21a177dbdd3090bB1CAE03EaCF78Fc6d"

# Minimal ABIs for interaction
AAVE_LENDING_POOL_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "asset", "type": "address"},
                    {"internalType": "uint256", "name": "amount", "type": "uint256"}
                ],
                "internalType": "struct ILendingPool.FlashLoanParams[]",
                "name": "assets",
                "type": "tuple[]"
            },
            {"internalType": "address", "name": "receiverAddress", "type": "address"},
            {"internalType": "bytes", "name": "params", "type": "bytes"},
            {"internalType": "uint16", "name": "referralCode", "type": "uint16"}
        ],
        "name": "flashLoan",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getReserveData",
        "outputs": [
            {
                "components": [
                    {
                        "components": [
                            {"internalType": "uint256", "name": "data", "type": "uint256"}
                        ],
                        "internalType": "struct DataTypes.ReserveConfigurationMap",
                        "name": "configuration",
                        "type": "tuple"
                    },
                    {"internalType": "uint128", "name": "liquidityIndex", "type": "uint128"},
                    {"internalType": "uint128", "name": "variableBorrowIndex", "type": "uint128"},
                    {"internalType": "uint128", "name": "currentLiquidityRate", "type": "uint128"},
                    {"internalType": "uint128", "name": "currentVariableBorrowRate", "type": "uint128"},
                    {"internalType": "uint128", "name": "currentStableBorrowRate", "type": "uint128"},
                    {"internalType": "uint40", "name": "lastUpdateTimestamp", "type": "uint40"},
                    {"internalType": "address", "name": "aTokenAddress", "type": "address"},
                    {"internalType": "address", "name": "stableDebtTokenAddress", "type": "address"},
                    {"internalType": "address", "name": "variableDebtTokenAddress", "type": "address"},
                    {"internalType": "address", "name": "interestRateStrategyAddress", "type": "address"},
                    {"internalType": "uint8", "name": "id", "type": "uint8"}
                ],
                "internalType": "struct DataTypes.ReserveData",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

AAVE_PROTOCOL_DATA_PROVIDER_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "asset", "type": "address"}],
        "name": "getReserveConfigurationData",
        "outputs": [
            {"internalType": "uint256", "name": "decimals", "type": "uint256"},
            {"internalType": "uint256", "name": "ltv", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidationThreshold", "type": "uint256"},
            {"internalType": "uint256", "name": "liquidationBonus", "type": "uint256"},
            {"internalType": "uint256", "name": "reserveFactor", "type": "uint256"},
            {"internalType": "bool", "name": "usageAsCollateralEnabled", "type": "bool"},
            {"internalType": "bool", "name": "borrowingEnabled", "type": "bool"},
            {"internalType": "bool", "name": "stableBorrowRateEnabled", "type": "bool"},
            {"internalType": "bool", "name": "isActive", "type": "bool"},
            {"internalType": "bool", "name": "isFrozen", "type": "bool"}
        ],
        "name": "getReserveConfigurationData",
        "stateMutability": "view",
        "type": "function"
    }
]

class AaveV2FlashLoan:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        
        # Initialize contracts
        self.lending_pool = web3.eth.contract(
            address=Web3.to_checksum_address(AAVE_LENDING_POOL),
            abi=AAVE_LENDING_POOL_ABI
        )
        self.data_provider = web3.eth.contract(
            address=Web3.to_checksum_address(AAVE_PROTOCOL_DATA_PROVIDER),
            abi=AAVE_PROTOCOL_DATA_PROVIDER_ABI
        )

    async def check_token_availability(self, token: Address) -> bool:
        """Check if a token is available for flash loans."""
        try:
            config = await self.data_provider.functions.getReserveConfigurationData(
                Web3.to_checksum_address(token)
            ).call()
            
            return config[8]  # isActive
        except Exception as e:
            self.logger.error(f"Error checking token availability: {str(e)}")
            return False

    async def get_flash_loan_fee(self, token: Address) -> Optional[int]:
        """Get the flash loan fee for a token."""
        try:
            reserve_data = await self.lending_pool.functions.getReserveData(
                Web3.to_checksum_address(token)
            ).call()
            
            # AAVE V2 flash loan fee is 0.09%
            return 9  # 9 basis points = 0.09%
        except Exception as e:
            self.logger.error(f"Error getting flash loan fee: {str(e)}")
            return None

    def encode_flash_loan(
        self,
        assets: List[Address],
        amounts: List[int],
        receiver: Address,
        params: bytes
    ) -> bytes:
        """Encode flash loan function call."""
        try:
            assets = [Web3.to_checksum_address(asset) for asset in assets]
            receiver = Web3.to_checksum_address(receiver)
            
            # Create flash loan params struct
            flash_loan_params = []
            for asset, amount in zip(assets, amounts):
                flash_loan_params.append({
                    "asset": asset,
                    "amount": amount
                })

            return self.lending_pool.encodeABI(
                fn_name="flashLoan",
                args=[
                    flash_loan_params,
                    receiver,
                    params,
                    0  # referralCode
                ]
            )
        except Exception as e:
            self.logger.error(f"Error encoding flash loan: {str(e)}")
            return b""

    async def simulate_flash_loan(
        self,
        assets: List[Address],
        amounts: List[int]
    ) -> Tuple[bool, int]:
        """Simulate a flash loan to calculate fees and check availability."""
        try:
            total_fee = 0
            for asset, amount in zip(assets, amounts):
                # Check if token is available
                if not await self.check_token_availability(asset):
                    return False, 0

                # Calculate fee
                fee = await self.get_flash_loan_fee(asset)
                if fee is None:
                    return False, 0
                
                total_fee += (amount * fee) // 10000  # Convert basis points to actual fee

            return True, total_fee

        except Exception as e:
            self.logger.error(f"Error simulating flash loan: {str(e)}")
            return False, 0
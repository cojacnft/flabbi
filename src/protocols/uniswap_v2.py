from typing import Dict, List, Optional, Tuple
from web3 import Web3
from eth_typing import Address
import json
import logging

# Uniswap V2 contract addresses on Ethereum mainnet
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V2_FACTORY = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

# Minimal ABIs for interaction
UNISWAP_V2_ROUTER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"}
        ],
        "name": "getAmountsOut",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint256", "name": "amountOutMin", "type": "uint256"},
            {"internalType": "address[]", "name": "path", "type": "address[]"},
            {"internalType": "address", "name": "to", "type": "address"},
            {"internalType": "uint256", "name": "deadline", "type": "uint256"}
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

UNISWAP_V2_FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenA", "type": "address"},
            {"internalType": "address", "name": "tokenB", "type": "address"}
        ],
        "name": "getPair",
        "outputs": [{"internalType": "address", "name": "pair", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]

UNISWAP_V2_PAIR_ABI = [
    {
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"internalType": "uint112", "name": "reserve0", "type": "uint112"},
            {"internalType": "uint112", "name": "reserve1", "type": "uint112"},
            {"internalType": "uint32", "name": "blockTimestampLast", "type": "uint32"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class UniswapV2Protocol:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        
        # Initialize contracts
        self.router = web3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V2_ROUTER),
            abi=UNISWAP_V2_ROUTER_ABI
        )
        self.factory = web3.eth.contract(
            address=Web3.to_checksum_address(UNISWAP_V2_FACTORY),
            abi=UNISWAP_V2_FACTORY_ABI
        )

    async def get_pair_address(
        self,
        token_a: Address,
        token_b: Address
    ) -> Optional[Address]:
        """Get the pair address for two tokens."""
        try:
            pair_address = await self.factory.functions.getPair(
                Web3.to_checksum_address(token_a),
                Web3.to_checksum_address(token_b)
            ).call()
            
            if pair_address == "0x" + "0" * 40:  # Null address
                return None
                
            return pair_address
        except Exception as e:
            self.logger.error(f"Error getting pair address: {str(e)}")
            return None

    async def get_reserves(
        self,
        pair_address: Address
    ) -> Optional[Tuple[int, int]]:
        """Get reserves for a pair."""
        try:
            pair_contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(pair_address),
                abi=UNISWAP_V2_PAIR_ABI
            )
            reserves = await pair_contract.functions.getReserves().call()
            return (reserves[0], reserves[1])
        except Exception as e:
            self.logger.error(f"Error getting reserves: {str(e)}")
            return None

    async def get_amount_out(
        self,
        amount_in: int,
        token_in: Address,
        token_out: Address
    ) -> Optional[int]:
        """Calculate output amount for a trade."""
        try:
            path = [
                Web3.to_checksum_address(token_in),
                Web3.to_checksum_address(token_out)
            ]
            amounts = await self.router.functions.getAmountsOut(
                amount_in,
                path
            ).call()
            return amounts[-1]
        except Exception as e:
            self.logger.error(f"Error calculating amount out: {str(e)}")
            return None

    async def check_liquidity(
        self,
        token_in: Address,
        token_out: Address,
        amount_in: int
    ) -> bool:
        """Check if there's enough liquidity for a trade."""
        try:
            # Get pair address
            pair_address = await self.get_pair_address(token_in, token_out)
            if not pair_address:
                return False

            # Get reserves
            reserves = await self.get_reserves(pair_address)
            if not reserves:
                return False

            # Check if there's enough liquidity
            # Require at least 2x the input amount as reserve
            token0 = min(token_in, token_out)
            reserve_in = reserves[0] if token0 == token_in else reserves[1]
            
            return reserve_in >= amount_in * 2

        except Exception as e:
            self.logger.error(f"Error checking liquidity: {str(e)}")
            return False

    def encode_swap_data(
        self,
        amount_in: int,
        amount_out_min: int,
        path: List[Address],
        to: Address,
        deadline: int
    ) -> bytes:
        """Encode swap function data."""
        try:
            return self.router.encodeABI(
                fn_name="swapExactTokensForTokens",
                args=[
                    amount_in,
                    amount_out_min,
                    [Web3.to_checksum_address(addr) for addr in path],
                    Web3.to_checksum_address(to),
                    deadline
                ]
            )
        except Exception as e:
            self.logger.error(f"Error encoding swap data: {str(e)}")
            return b""
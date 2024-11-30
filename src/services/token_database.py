from typing import Dict, List, Optional, Set, Tuple
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from web3 import Web3

from ..models.token import Token, TokenMetadata, LiquidityPool
from ..config import SystemConfig

class TokenDatabase:
    def __init__(self, config: SystemConfig, web3: Web3):
        self.config = config
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        self.tokens: Dict[str, Token] = {}
        self.update_lock = asyncio.Lock()
        self._load_tokens()

    def _load_tokens(self):
        """Load tokens from persistent storage."""
        try:
            with open('data/tokens.json', 'r') as f:
                tokens_data = json.load(f)
                for token_data in tokens_data:
                    token = Token.parse_obj(token_data)
                    self.tokens[token.address.lower()] = token
        except FileNotFoundError:
            self.logger.info("No existing token database found. Starting fresh.")
        except Exception as e:
            self.logger.error(f"Error loading tokens: {str(e)}")

    def _save_tokens(self):
        """Save tokens to persistent storage."""
        try:
            with open('data/tokens.json', 'w') as f:
                tokens_data = [token.dict() for token in self.tokens.values()]
                json.dump(tokens_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving tokens: {str(e)}")

    async def add_token(self, address: str) -> Optional[Token]:
        """Add a new token to the database."""
        try:
            address = address.lower()
            if address in self.tokens:
                return self.tokens[address]

            # Get token metadata from blockchain
            metadata = await self._fetch_token_metadata(address)
            if not metadata:
                return None

            # Create token entry
            token = Token(
                address=address,
                metadata=metadata,
                liquidity_pools=[]
            )

            # Update liquidity pools
            await self._update_token_pools(token)

            async with self.update_lock:
                self.tokens[address] = token
                self._save_tokens()

            return token

        except Exception as e:
            self.logger.error(f"Error adding token: {str(e)}")
            return None

    async def _fetch_token_metadata(self, address: str) -> Optional[TokenMetadata]:
        """Fetch token metadata from blockchain and external APIs."""
        try:
            # Load ERC20 ABI
            erc20_abi = [
                {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"type":"function"},
                {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"type":"function"}
            ]

            # Create contract instance
            token_contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=erc20_abi
            )

            # Get basic token info
            name = await token_contract.functions.name().call()
            symbol = await token_contract.functions.symbol().call()
            decimals = await token_contract.functions.decimals().call()
            total_supply = str(await token_contract.functions.totalSupply().call())

            # Get additional info from external API (e.g., CoinGecko)
            market_data = await self._fetch_market_data(address)

            return TokenMetadata(
                name=name,
                symbol=symbol,
                decimals=decimals,
                total_supply=total_supply,
                holders=market_data.get('holders'),
                market_cap_usd=market_data.get('market_cap_usd'),
                volume_24h_usd=market_data.get('volume_24h_usd')
            )

        except Exception as e:
            self.logger.error(f"Error fetching token metadata: {str(e)}")
            return None

    async def _fetch_market_data(self, address: str) -> Dict:
        """Fetch market data from external APIs."""
        try:
            # Example using CoinGecko API
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.coingecko.com/api/v3/simple/token_price/ethereum",
                    params={
                        "contract_addresses": address,
                        "vs_currencies": "usd",
                        "include_market_cap": "true",
                        "include_24hr_vol": "true",
                        "include_24hr_change": "true"
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        token_data = data.get(address.lower(), {})
                        return {
                            "market_cap_usd": token_data.get("usd_market_cap"),
                            "volume_24h_usd": token_data.get("usd_24h_vol"),
                            "price_change_24h": token_data.get("usd_24h_change")
                        }
            return {}

        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {}

    async def _update_token_pools(self, token: Token):
        """Update liquidity pools for a token."""
        try:
            # Get pools from various DEXs
            uniswap_pools = await self._get_uniswap_v2_pools(token.address)
            sushiswap_pools = await self._get_sushiswap_pools(token.address)
            
            # Combine all pools
            all_pools = [*uniswap_pools, *sushiswap_pools]
            
            # Update token's liquidity pools
            token.liquidity_pools = all_pools
            token.updated_at = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Error updating token pools: {str(e)}")

    async def _get_uniswap_v2_pools(self, token_address: str) -> List[LiquidityPool]:
        """Get Uniswap V2 pools for a token."""
        try:
            from ..protocols.uniswap_v2 import UniswapV2Protocol
            
            uniswap = UniswapV2Protocol(self.web3)
            pools = []

            # Common paired tokens (WETH, USDC, USDT, DAI)
            common_pairs = [
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x6B175474E89094C44Da98b954EedeAC495271d0F"   # DAI
            ]

            for pair_token in common_pairs:
                pair_address = await uniswap.get_pair_address(
                    token_address,
                    pair_token
                )
                
                if pair_address:
                    reserves = await uniswap.get_reserves(pair_address)
                    if reserves:
                        pools.append(LiquidityPool(
                            dex_name="Uniswap V2",
                            pool_address=pair_address,
                            token0_address=token_address,
                            token1_address=pair_token,
                            reserves_token0=str(reserves[0]),
                            reserves_token1=str(reserves[1]),
                            fee_tier=0.003  # 0.3%
                        ))

            return pools

        except Exception as e:
            self.logger.error(f"Error getting Uniswap V2 pools: {str(e)}")
            return []

    async def _get_sushiswap_pools(self, token_address: str) -> List[LiquidityPool]:
        """Get SushiSwap pools for a token."""
        # TODO: Implement SushiSwap pool fetching
        return []

    async def get_token(self, address: str) -> Optional[Token]:
        """Get a token from the database."""
        address = address.lower()
        return self.tokens.get(address)

    async def update_token(self, token: Token) -> bool:
        """Update a token in the database."""
        try:
            async with self.update_lock:
                self.tokens[token.address.lower()] = token
                self._save_tokens()
            return True
        except Exception as e:
            self.logger.error(f"Error updating token: {str(e)}")
            return False

    async def remove_token(self, address: str) -> bool:
        """Remove a token from the database."""
        try:
            address = address.lower()
            async with self.update_lock:
                if address in self.tokens:
                    del self.tokens[address]
                    self._save_tokens()
            return True
        except Exception as e:
            self.logger.error(f"Error removing token: {str(e)}")
            return False

    async def get_all_tokens(self) -> List[Token]:
        """Get all tokens in the database."""
        return list(self.tokens.values())

    async def get_active_tokens(self) -> List[Token]:
        """Get all active tokens in the database."""
        return [token for token in self.tokens.values() if token.is_active]

    async def get_verified_tokens(self) -> List[Token]:
        """Get all verified tokens in the database."""
        return [token for token in self.tokens.values() if token.is_verified]

    async def get_tokens_by_tag(self, tag: str) -> List[Token]:
        """Get all tokens with a specific tag."""
        return [
            token for token in self.tokens.values() 
            if tag in token.custom_tags
        ]

    async def update_all_tokens(self):
        """Update metadata and pools for all tokens."""
        try:
            for token in list(self.tokens.values()):
                # Update metadata
                new_metadata = await self._fetch_token_metadata(token.address)
                if new_metadata:
                    token.metadata = new_metadata

                # Update pools
                await self._update_token_pools(token)

            self._save_tokens()

        except Exception as e:
            self.logger.error(f"Error updating all tokens: {str(e)}")

    async def get_top_liquidity_pairs(self, limit: int = 10) -> List[Tuple[Token, Token, float]]:
        """Get top token pairs by liquidity."""
        try:
            pairs = []
            seen_pairs = set()

            for token in self.tokens.values():
                for pool in token.liquidity_pools:
                    pair_key = tuple(sorted([
                        pool.token0_address.lower(),
                        pool.token1_address.lower()
                    ]))

                    if pair_key in seen_pairs:
                        continue

                    seen_pairs.add(pair_key)
                    token1 = self.tokens.get(pool.token0_address.lower())
                    token2 = self.tokens.get(pool.token1_address.lower())

                    if token1 and token2 and pool.total_liquidity_usd:
                        pairs.append((token1, token2, pool.total_liquidity_usd))

            # Sort by liquidity and return top pairs
            pairs.sort(key=lambda x: x[2], reverse=True)
            return pairs[:limit]

        except Exception as e:
            self.logger.error(f"Error getting top liquidity pairs: {str(e)}")
            return []
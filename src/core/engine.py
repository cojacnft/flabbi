from typing import List, Optional
from dataclasses import dataclass
from web3 import Web3
import asyncio
import logging

from ..config import SystemConfig

@dataclass
class ArbitrageOpportunity:
    token_in: str
    token_out: str
    amount_in: int
    expected_profit: int
    route: List[str]
    gas_estimate: int

class ArbitrageEngine:
    def __init__(self, config: SystemConfig, web3: Web3):
        self.config = config
        self.web3 = web3
        self.logger = logging.getLogger(__name__)

    async def find_opportunities(self) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities across configured DEXs."""
        opportunities = []
        try:
            # Get latest block for price data
            block = await self.web3.eth.get_block('latest')
            block_number = block['number']

            # Analyze each enabled protocol pair
            for protocol1 in self.config.protocols.enabled:
                for protocol2 in self.config.protocols.enabled:
                    if protocol1 != protocol2:
                        # Find opportunities between these protocols
                        opportunity = await self._analyze_pair(
                            protocol1, 
                            protocol2, 
                            block_number
                        )
                        if opportunity:
                            opportunities.append(opportunity)

        except Exception as e:
            self.logger.error(f"Error finding opportunities: {str(e)}")

        return opportunities

    async def _analyze_pair(
        self, 
        protocol1: str, 
        protocol2: str, 
        block_number: int
    ) -> Optional[ArbitrageOpportunity]:
        """Analyze trading pair between two protocols for arbitrage opportunities."""
        try:
            # Get protocol instances
            protocol1_instance = self.protocol_manager.get_protocol(protocol1)
            protocol2_instance = self.protocol_manager.get_protocol(protocol2)
            
            if not protocol1_instance or not protocol2_instance:
                return None

            # Common token pairs for testing
            token_pairs = [
                # WETH-USDC
                ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 
                 "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"),
                # WETH-USDT
                ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                 "0xdAC17F958D2ee523a2206206994597C13D831ec7"),
                # WETH-DAI
                ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                 "0x6B175474E89094C44Da98b954EedeAC495271d0F")
            ]

            for token_a, token_b in token_pairs:
                # Check liquidity first
                if not await protocol1_instance.check_liquidity(token_a, token_b, 10**18):
                    continue
                if not await protocol2_instance.check_liquidity(token_b, token_a, 10**18):
                    continue

                # Get prices from both protocols
                amount_in = 10**18  # 1 WETH
                
                # Price check path: token_a -> token_b -> token_a
                amount_out_1 = await protocol1_instance.get_amount_out(
                    amount_in,
                    token_a,
                    token_b
                )
                if not amount_out_1:
                    continue

                amount_out_2 = await protocol2_instance.get_amount_out(
                    amount_out_1,
                    token_b,
                    token_a
                )
                if not amount_out_2:
                    continue

                # Calculate profit
                profit = amount_out_2 - amount_in
                if profit > 0:
                    # Estimate gas cost
                    gas_estimate = 300000  # Approximate gas for flash loan + 2 swaps

                    return ArbitrageOpportunity(
                        token_in=token_a,
                        token_out=token_b,
                        amount_in=amount_in,
                        expected_profit=profit,
                        route=[protocol1, protocol2],
                        gas_estimate=gas_estimate
                    )

            return None

        except Exception as e:
            self.logger.error(
                f"Error analyzing pair {protocol1}-{protocol2}: {str(e)}"
            )
            return None

    async def validate_opportunity(
        self, 
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Validate if an arbitrage opportunity is still valid and profitable."""
        try:
            # Calculate gas costs
            gas_price = await self.web3.eth.gas_price
            gas_cost = opportunity.gas_estimate * gas_price

            # Check if profit exceeds gas cost with minimum threshold
            min_profit = gas_cost * 1.5  # 50% more than gas cost
            return opportunity.expected_profit > min_profit

        except Exception as e:
            self.logger.error(f"Error validating opportunity: {str(e)}")
            return False

    async def execute_arbitrage(
        self, 
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Execute an arbitrage opportunity."""
        if not await self.validate_opportunity(opportunity):
            return False

        try:
            # Get protocol instances
            protocol1 = self.protocol_manager.get_protocol(opportunity.route[0])
            protocol2 = self.protocol_manager.get_protocol(opportunity.route[1])
            
            if not protocol1 or not protocol2:
                return False

            # Get flash loan provider
            flash_loan = self.flash_loan_service.get_provider("AAVE_V2")
            if not flash_loan:
                return False

            # Calculate minimum output amounts with slippage tolerance
            slippage = self.config.protocols.max_slippage
            
            # First swap: token_in -> token_out
            amount_out_1 = await protocol1.get_amount_out(
                opportunity.amount_in,
                opportunity.token_in,
                opportunity.token_out
            )
            if not amount_out_1:
                return False
            
            min_amount_out_1 = int(amount_out_1 * (1 - slippage))

            # Second swap: token_out -> token_in
            amount_out_2 = await protocol2.get_amount_out(
                amount_out_1,
                opportunity.token_out,
                opportunity.token_in
            )
            if not amount_out_2:
                return False
            
            min_amount_out_2 = int(amount_out_2 * (1 - slippage))

            # Prepare flash loan parameters
            deadline = self.web3.eth.get_block('latest')['timestamp'] + 300  # 5 minutes

            # Encode the swap data
            swap1_data = protocol1.encode_swap_data(
                opportunity.amount_in,
                min_amount_out_1,
                [opportunity.token_in, opportunity.token_out],
                self.arbitrage_contract.address,
                deadline
            )

            swap2_data = protocol2.encode_swap_data(
                amount_out_1,
                min_amount_out_2,
                [opportunity.token_out, opportunity.token_in],
                self.arbitrage_contract.address,
                deadline
            )

            # Combine swap data for the flash loan callback
            callback_data = self.arbitrage_contract.encode_callback_data(
                swap1_data,
                swap2_data
            )

            # Execute flash loan
            flash_loan_data = flash_loan.encode_flash_loan(
                [opportunity.token_in],
                [opportunity.amount_in],
                self.arbitrage_contract.address,
                callback_data
            )

            # Send transaction
            tx = await self.arbitrage_contract.functions.executeOperation(
                flash_loan_data
            ).transact({
                'from': self.account.address,
                'gas': opportunity.gas_estimate,
                'gasPrice': await self.web3.eth.gas_price
            })

            # Wait for confirmation
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx)
            return receipt['status'] == 1

        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            return False
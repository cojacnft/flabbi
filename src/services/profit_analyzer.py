from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3

from .market_data import MarketDataAggregator
from .simulator import TransactionSimulator

class ProfitAnalyzer:
    def __init__(
        self,
        web3: Web3,
        market_data: MarketDataAggregator,
        simulator: TransactionSimulator
    ):
        self.web3 = web3
        self.market_data = market_data
        self.simulator = simulator
        self.logger = logging.getLogger(__name__)
        
        # Historical data
        self.profit_history: List[Dict] = []
        self.gas_history: List[Dict] = []
        self.success_history: List[Dict] = []
        
        # Analysis settings
        self.min_profit_threshold = 50  # USD
        self.max_position_size = 100000  # USD
        self.min_success_probability = 0.8
        self.max_gas_cost_ratio = 0.1  # Max 10% of profit

    async def analyze_opportunity(
        self,
        opportunity: Dict,
        simulation_result: Optional[Dict] = None
    ) -> Dict:
        """Analyze profitability of an arbitrage opportunity."""
        try:
            # Get current market conditions
            market_conditions = await self._get_market_conditions(opportunity)
            
            # Calculate base profitability
            base_analysis = self._analyze_base_profitability(
                opportunity,
                market_conditions
            )
            
            # Get simulation results if not provided
            if not simulation_result and base_analysis["profitable"]:
                simulation_result = await self.simulator.simulate_arbitrage(
                    opportunity["flash_loan_data"],
                    opportunity["path_data"]
                )
            
            # Calculate final profitability
            final_analysis = self._calculate_final_profitability(
                base_analysis,
                simulation_result,
                market_conditions
            )
            
            # Update history
            self._update_history(final_analysis)
            
            return final_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunity: {str(e)}")
            return {
                "profitable": False,
                "error": str(e)
            }

    async def _get_market_conditions(self, opportunity: Dict) -> Dict:
        """Get current market conditions for analysis."""
        try:
            # Get token prices
            token_prices = {
                token: await self.market_data.get_token_price(token)
                for token in opportunity["path_data"]["tokens"]
            }
            
            # Get market depth
            depths = {
                token: await self.market_data.get_market_depth(
                    token,
                    opportunity["amount_in_usd"]
                )
                for token in opportunity["path_data"]["tokens"]
            }
            
            # Get gas price
            gas_price = await self.web3.eth.gas_price
            
            # Get network congestion
            block = await self.web3.eth.get_block('latest')
            congestion = block['gasUsed'] / block['gasLimit']
            
            return {
                "token_prices": token_prices,
                "market_depths": depths,
                "gas_price_gwei": gas_price / 1e9,
                "network_congestion": congestion,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market conditions: {str(e)}")
            return {}

    def _analyze_base_profitability(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> Dict:
        """Analyze base profitability before simulation."""
        try:
            # Calculate expected profit
            expected_profit = self._calculate_expected_profit(
                opportunity,
                market_conditions
            )
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(
                opportunity,
                market_conditions
            )
            
            # Calculate success probability
            success_prob = self._calculate_success_probability(
                opportunity,
                risk_factors
            )
            
            # Check if opportunity meets basic criteria
            profitable = (
                expected_profit > self.min_profit_threshold and
                success_prob > self.min_success_probability and
                opportunity["amount_in_usd"] <= self.max_position_size
            )
            
            return {
                "profitable": profitable,
                "expected_profit": expected_profit,
                "risk_factors": risk_factors,
                "success_probability": success_prob,
                "market_conditions": market_conditions
            }
            
        except Exception as e:
            self.logger.error(f"Error in base profitability analysis: {str(e)}")
            return {
                "profitable": False,
                "error": str(e)
            }

    def _calculate_expected_profit(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> float:
        """Calculate expected profit considering market conditions."""
        try:
            # Get relevant prices
            token_prices = market_conditions["token_prices"]
            
            # Calculate input value
            input_value = opportunity["amount_in"] * token_prices[
                opportunity["path_data"]["tokens"][0]
            ]
            
            # Calculate expected output value
            output_value = opportunity["expected_output"] * token_prices[
                opportunity["path_data"]["tokens"][-1]
            ]
            
            # Calculate raw profit
            raw_profit = output_value - input_value
            
            # Adjust for market depth
            depth_adjustment = self._calculate_depth_adjustment(
                opportunity,
                market_conditions
            )
            
            # Adjust for estimated costs
            estimated_costs = self._estimate_costs(
                opportunity,
                market_conditions
            )
            
            return raw_profit * depth_adjustment - estimated_costs
            
        except Exception as e:
            self.logger.error(f"Error calculating expected profit: {str(e)}")
            return 0.0

    def _calculate_depth_adjustment(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> float:
        """Calculate market depth adjustment factor."""
        try:
            depths = market_conditions["market_depths"]
            
            # Get minimum depth across path
            min_depth = min(
                depth[0] for depth in depths.values()
            )
            
            # Calculate adjustment factor (0.5 to 1.0)
            return 0.5 + (min_depth * 0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating depth adjustment: {str(e)}")
            return 0.5

    def _estimate_costs(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> float:
        """Estimate total costs including gas and fees."""
        try:
            # Estimate gas cost
            gas_price_gwei = market_conditions["gas_price_gwei"]
            estimated_gas = 300000  # Base estimate for flash loan + 2 swaps
            gas_cost = (gas_price_gwei * estimated_gas * 1e-9) * market_conditions[
                "token_prices"
            ]["ETH"]
            
            # Flash loan fee (usually 0.09%)
            flash_loan_fee = opportunity["amount_in_usd"] * 0.0009
            
            # DEX fees (0.3% per swap)
            dex_fees = opportunity["amount_in_usd"] * 0.003 * len(
                opportunity["path_data"]["dexes"]
            )
            
            return gas_cost + flash_loan_fee + dex_fees
            
        except Exception as e:
            self.logger.error(f"Error estimating costs: {str(e)}")
            return 0.0

    def _calculate_risk_factors(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> Dict:
        """Calculate various risk factors."""
        try:
            return {
                "market_depth": self._calculate_depth_risk(
                    opportunity,
                    market_conditions
                ),
                "network_congestion": market_conditions["network_congestion"],
                "price_volatility": self._calculate_volatility_risk(
                    opportunity,
                    market_conditions
                ),
                "path_complexity": len(opportunity["path_data"]["tokens"]) / 4
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk factors: {str(e)}")
            return {}

    def _calculate_depth_risk(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> float:
        """Calculate risk based on market depth."""
        try:
            depths = market_conditions["market_depths"]
            
            # Calculate average impact
            impacts = [depth[1] for depth in depths.values()]
            avg_impact = np.mean(impacts)
            
            # Convert to risk score (0 to 1)
            return min(avg_impact * 2, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating depth risk: {str(e)}")
            return 1.0

    def _calculate_volatility_risk(
        self,
        opportunity: Dict,
        market_conditions: Dict
    ) -> float:
        """Calculate risk based on price volatility."""
        # TODO: Implement volatility calculation
        return 0.5

    def _calculate_success_probability(
        self,
        opportunity: Dict,
        risk_factors: Dict
    ) -> float:
        """Calculate probability of successful execution."""
        try:
            # Base probability
            base_prob = 1.0
            
            # Adjust for each risk factor
            for factor, value in risk_factors.items():
                if factor == "market_depth":
                    base_prob *= (1 - value * 0.5)  # Max 50% reduction
                elif factor == "network_congestion":
                    base_prob *= (1 - value * 0.3)  # Max 30% reduction
                elif factor == "price_volatility":
                    base_prob *= (1 - value * 0.4)  # Max 40% reduction
                elif factor == "path_complexity":
                    base_prob *= (1 - value * 0.2)  # Max 20% reduction
            
            # Consider historical success rate
            hist_success = self._get_historical_success_rate(
                opportunity["path_data"]["path_id"]
            )
            if hist_success is not None:
                base_prob = (base_prob + hist_success) / 2
            
            return max(min(base_prob, 1.0), 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return 0.0

    def _calculate_final_profitability(
        self,
        base_analysis: Dict,
        simulation_result: Optional[Dict],
        market_conditions: Dict
    ) -> Dict:
        """Calculate final profitability including simulation results."""
        try:
            if not base_analysis["profitable"]:
                return base_analysis
            
            if simulation_result:
                # Adjust expected profit based on simulation
                simulated_profit = simulation_result["profit"]
                profit_accuracy = min(
                    simulated_profit / base_analysis["expected_profit"],
                    1.0
                )
                
                # Update success probability
                success_prob = base_analysis["success_probability"] * profit_accuracy
                
                # Check if still profitable
                profitable = (
                    simulated_profit > self.min_profit_threshold and
                    success_prob > self.min_success_probability
                )
                
                return {
                    **base_analysis,
                    "profitable": profitable,
                    "simulated_profit": simulated_profit,
                    "profit_accuracy": profit_accuracy,
                    "success_probability": success_prob,
                    "simulation_result": simulation_result
                }
            
            return base_analysis
            
        except Exception as e:
            self.logger.error(f"Error in final profitability analysis: {str(e)}")
            return {
                "profitable": False,
                "error": str(e)
            }

    def _update_history(self, analysis: Dict):
        """Update historical data with analysis results."""
        try:
            # Add to profit history
            self.profit_history.append({
                "timestamp": datetime.utcnow(),
                "expected_profit": analysis["expected_profit"],
                "simulated_profit": analysis.get("simulated_profit"),
                "path_id": analysis.get("path_id")
            })
            
            # Maintain history size
            max_history = 1000
            if len(self.profit_history) > max_history:
                self.profit_history = self.profit_history[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error updating history: {str(e)}")

    def _get_historical_success_rate(self, path_id: str) -> Optional[float]:
        """Get historical success rate for a specific path."""
        try:
            # Filter relevant history
            path_history = [
                h for h in self.success_history
                if h["path_id"] == path_id and
                datetime.utcnow() - h["timestamp"] < timedelta(hours=24)
            ]
            
            if not path_history:
                return None
            
            # Calculate success rate
            successes = sum(1 for h in path_history if h["success"])
            return successes / len(path_history)
            
        except Exception as e:
            self.logger.error(f"Error getting historical success rate: {str(e)}")
            return None

    def get_analytics(self) -> Dict:
        """Get analytics from historical data."""
        try:
            recent_history = [
                h for h in self.profit_history
                if datetime.utcnow() - h["timestamp"] < timedelta(hours=24)
            ]
            
            if not recent_history:
                return {}
            
            profits = [h["expected_profit"] for h in recent_history]
            
            return {
                "total_opportunities": len(recent_history),
                "average_profit": np.mean(profits),
                "median_profit": np.median(profits),
                "max_profit": max(profits),
                "min_profit": min(profits),
                "profit_std": np.std(profits),
                "success_rate": self._calculate_overall_success_rate()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analytics: {str(e)}")
            return {}
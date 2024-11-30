from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class PortfolioTarget:
    """Portfolio target allocation."""
    token: str
    target_weight: float
    min_weight: float
    max_weight: float
    current_weight: float

@dataclass
class RebalanceAction:
    """Portfolio rebalancing action."""
    token: str
    action: str  # "buy" or "sell"
    amount_usd: float
    priority: int  # 1 (high) to 3 (low)

class PortfolioBalancer:
    """Optimize and balance portfolio allocations."""
    def __init__(
        self,
        web3: Web3,
        market_data_aggregator,
        risk_manager,
        settings: Dict
    ):
        self.web3 = web3
        self.market_data = market_data_aggregator
        self.risk_manager = risk_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Portfolio settings
        self.portfolio_settings = {
            "rebalance_threshold": 0.05,  # 5% deviation trigger
            "min_rebalance_amount": 1000.0,  # Minimum $1000 per trade
            "max_rebalance_amount": 50000.0,  # Maximum $50k per trade
            "target_positions": 5,  # Target number of positions
            "max_single_allocation": 0.3,  # Maximum 30% in single token
            "min_single_allocation": 0.05,  # Minimum 5% in single token
            "cash_buffer": 0.1  # 10% cash buffer
        }
        
        # Portfolio state
        self.portfolio_state = {
            "total_value": 0.0,
            "positions": {},
            "cash": 0.0,
            "last_rebalance": datetime.min
        }
        
        # Performance metrics
        self.metrics = {
            "rebalances": 0,
            "total_cost": 0.0,
            "tracking_error": 0.0,
            "turnover_ratio": 0.0
        }
        
        # Initialize portfolio optimizer
        self._init_optimizer()

    def _init_optimizer(self):
        """Initialize portfolio optimization parameters."""
        self.optimization_params = {
            "risk_free_rate": 0.03,  # 3% risk-free rate
            "target_return": 0.15,  # 15% target annual return
            "max_volatility": 0.3,  # 30% max volatility
            "correlation_penalty": 0.1,  # Correlation diversification penalty
            "liquidity_weight": 0.2  # Liquidity importance weight
        }

    async def check_rebalance_needed(self) -> Tuple[bool, List[RebalanceAction]]:
        """Check if portfolio rebalancing is needed."""
        try:
            # Get current allocations
            current_allocations = await self._get_current_allocations()
            
            # Get target allocations
            target_allocations = await self._calculate_target_allocations()
            
            # Calculate deviations
            deviations = self._calculate_deviations(
                current_allocations,
                target_allocations
            )
            
            # Check if rebalance needed
            max_deviation = max(abs(d) for d in deviations.values())
            
            if max_deviation > self.portfolio_settings["rebalance_threshold"]:
                # Calculate rebalance actions
                actions = await self._calculate_rebalance_actions(
                    current_allocations,
                    target_allocations
                )
                
                return True, actions
            
            return False, []
            
        except Exception as e:
            self.logger.error(f"Error checking rebalance: {str(e)}")
            return False, []

    async def _get_current_allocations(self) -> Dict[str, float]:
        """Get current portfolio allocations."""
        try:
            allocations = {}
            total_value = self.portfolio_state["total_value"]
            
            if total_value == 0:
                return {}
            
            # Calculate allocations
            for token, position in self.portfolio_state["positions"].items():
                value = position["amount_usd"]
                allocations[token] = value / total_value
            
            # Add cash allocation
            cash_allocation = self.portfolio_state["cash"] / total_value
            allocations["cash"] = cash_allocation
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error getting allocations: {str(e)}")
            return {}

    async def _calculate_target_allocations(self) -> Dict[str, PortfolioTarget]:
        """Calculate optimal target allocations."""
        try:
            # Get active tokens
            tokens = await self.market_data.get_active_tokens()
            
            # Get historical data
            historical_data = await self._get_historical_data(tokens)
            
            # Calculate optimal weights
            optimal_weights = await self._optimize_portfolio(
                historical_data,
                tokens
            )
            
            # Create targets
            targets = {}
            for token in tokens:
                weight = optimal_weights.get(token, 0.0)
                targets[token] = PortfolioTarget(
                    token=token,
                    target_weight=weight,
                    min_weight=self.portfolio_settings["min_single_allocation"],
                    max_weight=self.portfolio_settings["max_single_allocation"],
                    current_weight=self.portfolio_state["positions"].get(
                        token,
                        {}
                    ).get("weight", 0.0)
                )
            
            # Add cash target
            targets["cash"] = PortfolioTarget(
                token="cash",
                target_weight=self.portfolio_settings["cash_buffer"],
                min_weight=self.portfolio_settings["cash_buffer"],
                max_weight=self.portfolio_settings["cash_buffer"] * 2,
                current_weight=self.portfolio_state["cash"] / self.portfolio_state["total_value"]
            )
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error calculating targets: {str(e)}")
            return {}

    async def _get_historical_data(
        self,
        tokens: List[str]
    ) -> pd.DataFrame:
        """Get historical price data for optimization."""
        try:
            data = {}
            
            for token in tokens:
                prices = await self.market_data.get_price_history(token)
                if prices is not None:
                    data[token] = prices
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()

    async def _optimize_portfolio(
        self,
        historical_data: pd.DataFrame,
        tokens: List[str]
    ) -> Dict[str, float]:
        """Optimize portfolio weights."""
        try:
            # Calculate returns
            returns = historical_data.pct_change().dropna()
            
            # Calculate expected returns and covariance
            expected_returns = returns.mean() * 252  # Annualize
            covariance = returns.cov() * 252
            
            # Get risk metrics
            risk_metrics = await self._get_risk_metrics(tokens)
            
            # Define optimization constraints
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
                {"type": "ineq", "fun": lambda x: x - self.portfolio_settings["min_single_allocation"]},  # Minimum weight
                {"type": "ineq", "fun": lambda x: self.portfolio_settings["max_single_allocation"] - x}  # Maximum weight
            ]
            
            # Define objective function
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
                
                # Add risk adjustments
                risk_adjustment = sum(
                    weights[i] * risk_metrics[token]["risk_score"]
                    for i, token in enumerate(tokens)
                )
                
                # Add correlation penalty
                correlation_penalty = self._calculate_correlation_penalty(
                    weights,
                    returns
                )
                
                # Add liquidity adjustment
                liquidity_adjustment = sum(
                    weights[i] * (1 - risk_metrics[token]["liquidity_score"])
                    for i, token in enumerate(tokens)
                )
                
                # Combine objectives
                return -(
                    portfolio_return / portfolio_risk -  # Sharpe ratio
                    self.optimization_params["correlation_penalty"] * correlation_penalty -
                    self.optimization_params["liquidity_weight"] * liquidity_adjustment
                )
            
            # Initial guess
            n_assets = len(tokens)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                constraints=constraints,
                bounds=[(0, self.portfolio_settings["max_single_allocation"])] * n_assets
            )
            
            # Convert results to dictionary
            optimal_weights = {
                token: weight
                for token, weight in zip(tokens, result.x)
            }
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {str(e)}")
            return {token: 1/len(tokens) for token in tokens}

    async def _get_risk_metrics(
        self,
        tokens: List[str]
    ) -> Dict[str, Dict]:
        """Get risk metrics for tokens."""
        try:
            metrics = {}
            
            for token in tokens:
                # Get volatility
                volatility = await self.market_data.get_volatility(token)
                
                # Get liquidity
                liquidity = await self.market_data.get_total_liquidity(token)
                max_liquidity = max(
                    await self.market_data.get_total_liquidity(t)
                    for t in tokens
                )
                liquidity_score = liquidity / max_liquidity
                
                # Calculate risk score
                risk_score = (
                    volatility * 0.4 +
                    (1 - liquidity_score) * 0.4 +
                    random.random() * 0.2  # Add some randomness
                )
                
                metrics[token] = {
                    "volatility": volatility,
                    "liquidity_score": liquidity_score,
                    "risk_score": risk_score
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {str(e)}")
            return {token: {"risk_score": 0.5, "liquidity_score": 0.5} for token in tokens}

    def _calculate_correlation_penalty(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> float:
        """Calculate correlation penalty for diversification."""
        try:
            correlation = returns.corr()
            weighted_correlation = np.dot(weights.T, np.dot(correlation, weights))
            return weighted_correlation
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation penalty: {str(e)}")
            return 0.0

    def _calculate_deviations(
        self,
        current: Dict[str, float],
        targets: Dict[str, PortfolioTarget]
    ) -> Dict[str, float]:
        """Calculate allocation deviations."""
        try:
            deviations = {}
            
            for token, target in targets.items():
                current_weight = current.get(token, 0.0)
                deviations[token] = current_weight - target.target_weight
            
            return deviations
            
        except Exception as e:
            self.logger.error(f"Error calculating deviations: {str(e)}")
            return {}

    async def _calculate_rebalance_actions(
        self,
        current: Dict[str, float],
        targets: Dict[str, PortfolioTarget]
    ) -> List[RebalanceAction]:
        """Calculate required rebalancing actions."""
        try:
            actions = []
            total_value = self.portfolio_state["total_value"]
            
            for token, target in targets.items():
                current_weight = current.get(token, 0.0)
                deviation = current_weight - target.target_weight
                
                if abs(deviation) > self.portfolio_settings["rebalance_threshold"]:
                    # Calculate amount to trade
                    amount = abs(deviation) * total_value
                    
                    # Apply min/max constraints
                    amount = max(
                        self.portfolio_settings["min_rebalance_amount"],
                        min(
                            amount,
                            self.portfolio_settings["max_rebalance_amount"]
                        )
                    )
                    
                    # Determine action type
                    action_type = "sell" if deviation > 0 else "buy"
                    
                    # Calculate priority
                    priority = self._calculate_action_priority(
                        deviation,
                        target,
                        token
                    )
                    
                    actions.append(
                        RebalanceAction(
                            token=token,
                            action=action_type,
                            amount_usd=amount,
                            priority=priority
                        )
                    )
            
            # Sort by priority
            actions.sort(key=lambda x: x.priority)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error calculating rebalance actions: {str(e)}")
            return []

    def _calculate_action_priority(
        self,
        deviation: float,
        target: PortfolioTarget,
        token: str
    ) -> int:
        """Calculate priority for rebalancing action."""
        try:
            # High priority (1)
            if (
                abs(deviation) > 0.1 or  # Large deviation
                target.current_weight > target.max_weight or  # Over maximum
                target.current_weight < target.min_weight  # Under minimum
            ):
                return 1
            
            # Medium priority (2)
            if (
                abs(deviation) > 0.05 or  # Medium deviation
                token == "cash"  # Cash rebalancing
            ):
                return 2
            
            # Low priority (3)
            return 3
            
        except Exception as e:
            self.logger.error(f"Error calculating priority: {str(e)}")
            return 3

    async def execute_rebalance(
        self,
        actions: List[RebalanceAction]
    ) -> Tuple[bool, List[str]]:
        """Execute rebalancing actions."""
        try:
            executed_actions = []
            total_cost = 0.0
            
            for action in actions:
                # Validate action with risk manager
                validation = await self.risk_manager.validate_trade({
                    "token": action.token,
                    "amount_usd": action.amount_usd,
                    "action": action.action
                })
                
                if not validation["passed"]:
                    continue
                
                # Execute trade
                success, tx_hash = await self._execute_trade(action)
                
                if success:
                    executed_actions.append(tx_hash)
                    
                    # Update portfolio state
                    await self._update_portfolio_state(action)
                    
                    # Calculate and add costs
                    cost = await self._calculate_trade_cost(action)
                    total_cost += cost
            
            # Update metrics
            if executed_actions:
                self.metrics["rebalances"] += 1
                self.metrics["total_cost"] += total_cost
                self.portfolio_state["last_rebalance"] = datetime.utcnow()
            
            return bool(executed_actions), executed_actions
            
        except Exception as e:
            self.logger.error(f"Error executing rebalance: {str(e)}")
            return False, []

    async def _execute_trade(
        self,
        action: RebalanceAction
    ) -> Tuple[bool, Optional[str]]:
        """Execute single rebalancing trade."""
        try:
            # Prepare trade parameters
            trade_params = await self._prepare_trade_params(action)
            
            # Execute trade through DEX
            tx_hash = await self._send_trade_transaction(trade_params)
            
            if tx_hash:
                return True, tx_hash
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False, None

    async def _prepare_trade_params(
        self,
        action: RebalanceAction
    ) -> Dict:
        """Prepare trade parameters."""
        try:
            # Get best route
            route = await self._find_best_route(
                action.token,
                "USDC" if action.action == "buy" else action.token,
                action.amount_usd
            )
            
            if not route:
                raise ValueError("No valid route found")
            
            # Prepare transaction
            return {
                "route": route["path"],
                "amount_in": route["amounts"][0],
                "min_amount_out": int(route["amounts"][-1] * 0.995),  # 0.5% slippage
                "deadline": int(datetime.utcnow().timestamp() + 300)  # 5 min
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing trade params: {str(e)}")
            return {}

    async def _find_best_route(
        self,
        token_in: str,
        token_out: str,
        amount_usd: float
    ) -> Optional[Dict]:
        """Find best trading route."""
        # This would use the RouteOptimizer service
        return None

    async def _send_trade_transaction(
        self,
        params: Dict
    ) -> Optional[str]:
        """Send trade transaction."""
        # This would use the execution service
        return None

    async def _update_portfolio_state(
        self,
        action: RebalanceAction
    ):
        """Update portfolio state after trade."""
        try:
            # Update position
            if action.action == "buy":
                if action.token not in self.portfolio_state["positions"]:
                    self.portfolio_state["positions"][action.token] = {
                        "amount_usd": 0.0,
                        "weight": 0.0
                    }
                self.portfolio_state["positions"][action.token]["amount_usd"] += action.amount_usd
                self.portfolio_state["cash"] -= action.amount_usd
            else:
                self.portfolio_state["positions"][action.token]["amount_usd"] -= action.amount_usd
                self.portfolio_state["cash"] += action.amount_usd
            
            # Update weights
            total_value = self.portfolio_state["total_value"]
            for token, position in self.portfolio_state["positions"].items():
                position["weight"] = position["amount_usd"] / total_value
            
            # Remove empty positions
            self.portfolio_state["positions"] = {
                token: position
                for token, position in self.portfolio_state["positions"].items()
                if position["amount_usd"] > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {str(e)}")

    async def _calculate_trade_cost(
        self,
        action: RebalanceAction
    ) -> float:
        """Calculate trading costs."""
        try:
            # Estimate gas cost
            gas_cost = await self._estimate_gas_cost(action)
            
            # Estimate slippage
            slippage_cost = action.amount_usd * 0.003  # Assume 0.3% slippage
            
            # Add DEX fee
            dex_fee = action.amount_usd * 0.003  # Assume 0.3% fee
            
            return gas_cost + slippage_cost + dex_fee
            
        except Exception as e:
            self.logger.error(f"Error calculating trade cost: {str(e)}")
            return 0.0

    async def _estimate_gas_cost(
        self,
        action: RebalanceAction
    ) -> float:
        """Estimate gas cost for trade."""
        try:
            # Get gas price
            gas_price = await self.web3.eth.gas_price
            
            # Estimate gas usage
            gas_limit = 200000  # Conservative estimate
            
            # Calculate cost in ETH
            cost_eth = gas_price * gas_limit
            
            # Convert to USD
            eth_price = await self.market_data.get_token_price(
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # WETH
            )
            
            return cost_eth * eth_price / 1e18
            
        except Exception as e:
            self.logger.error(f"Error estimating gas cost: {str(e)}")
            return 0.0

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "portfolio_state": self.portfolio_state,
            "last_rebalance": self.portfolio_state["last_rebalance"].isoformat(),
            "total_value": self.portfolio_state["total_value"],
            "cash_ratio": self.portfolio_state["cash"] / self.portfolio_state["total_value"],
            "num_positions": len(self.portfolio_state["positions"]),
            "last_update": datetime.utcnow().isoformat()
        }
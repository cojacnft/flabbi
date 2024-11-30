from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import pandas as pd
from dataclasses import dataclass
from scipy import stats

@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_size: float  # Maximum position size in USD
    max_daily_volume: float  # Maximum daily trading volume in USD
    max_daily_loss: float  # Maximum daily loss in USD
    max_drawdown: float  # Maximum drawdown percentage
    min_profit_threshold: float  # Minimum profit per trade in USD
    max_gas_exposure: float  # Maximum gas cost as percentage of profit
    risk_free_rate: float  # Risk-free rate for Sharpe ratio
    confidence_level: float  # Confidence level for VaR

@dataclass
class PositionRisk:
    """Position risk metrics."""
    value_at_risk: float  # VaR at confidence level
    expected_shortfall: float  # Expected shortfall (CVaR)
    sharpe_ratio: float  # Risk-adjusted return ratio
    profit_factor: float  # Ratio of profits to losses
    win_rate: float  # Percentage of profitable trades
    max_drawdown: float  # Maximum drawdown percentage

class RiskManager:
    """Advanced risk management system."""
    def __init__(
        self,
        web3: Web3,
        market_data_aggregator,
        settings: Dict
    ):
        self.web3 = web3
        self.market_data = market_data_aggregator
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.risk_params = RiskParameters(
            max_position_size=50000.0,  # $50k max position
            max_daily_volume=1000000.0,  # $1M daily volume
            max_daily_loss=10000.0,  # $10k max daily loss
            max_drawdown=0.1,  # 10% max drawdown
            min_profit_threshold=100.0,  # $100 min profit
            max_gas_exposure=0.1,  # 10% max gas cost
            risk_free_rate=0.03,  # 3% risk-free rate
            confidence_level=0.95  # 95% confidence
        )
        
        # Position tracking
        self.active_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        
        # Daily tracking
        self.daily_stats = {
            "volume": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "trades": 0,
            "successful_trades": 0,
            "gas_spent": 0.0
        }
        
        # Risk metrics
        self.metrics = {
            "total_trades": 0,
            "profitable_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        # Initialize portfolio tracker
        self._init_portfolio_tracker()

    def _init_portfolio_tracker(self):
        """Initialize portfolio tracking."""
        self.portfolio = {
            "total_value": 0.0,
            "token_balances": {},
            "token_allocations": {},
            "risk_exposure": 0.0,
            "available_capital": 0.0
        }

    async def validate_trade(
        self,
        trade: Dict,
        context: Dict
    ) -> Tuple[bool, Dict]:
        """Validate trade against risk parameters."""
        try:
            validation = {
                "passed": False,
                "checks": {},
                "risk_metrics": {},
                "warnings": []
            }
            
            # Basic risk checks
            basic_checks = await self._perform_basic_checks(trade, context)
            validation["checks"]["basic"] = basic_checks
            
            if not basic_checks["passed"]:
                return False, validation
            
            # Position risk checks
            position_checks = await self._check_position_risk(trade, context)
            validation["checks"]["position"] = position_checks
            validation["risk_metrics"]["position"] = position_checks["metrics"]
            
            if not position_checks["passed"]:
                return False, validation
            
            # Portfolio risk checks
            portfolio_checks = await self._check_portfolio_risk(trade, context)
            validation["checks"]["portfolio"] = portfolio_checks
            validation["risk_metrics"]["portfolio"] = portfolio_checks["metrics"]
            
            if not portfolio_checks["passed"]:
                return False, validation
            
            # Calculate final risk score
            risk_score = self._calculate_risk_score(
                basic_checks,
                position_checks,
                portfolio_checks
            )
            validation["risk_score"] = risk_score
            
            # Validate against threshold
            validation["passed"] = risk_score <= 0.7  # Max 70% risk score
            
            return validation["passed"], validation
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False, {"passed": False, "error": str(e)}

    async def _perform_basic_checks(
        self,
        trade: Dict,
        context: Dict
    ) -> Dict:
        """Perform basic risk checks."""
        try:
            checks = {
                "passed": False,
                "profit_check": False,
                "size_check": False,
                "gas_check": False,
                "daily_check": False
            }
            
            # Check minimum profit
            expected_profit = trade["expected_profit"]
            checks["profit_check"] = expected_profit >= self.risk_params.min_profit_threshold
            
            # Check position size
            position_size = trade["amount_in_usd"]
            checks["size_check"] = position_size <= self.risk_params.max_position_size
            
            # Check gas costs
            gas_cost = trade["gas_cost"]
            gas_ratio = gas_cost / expected_profit
            checks["gas_check"] = gas_ratio <= self.risk_params.max_gas_exposure
            
            # Check daily limits
            new_volume = self.daily_stats["volume"] + position_size
            checks["daily_check"] = new_volume <= self.risk_params.max_daily_volume
            
            # Overall check
            checks["passed"] = all([
                checks["profit_check"],
                checks["size_check"],
                checks["gas_check"],
                checks["daily_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error in basic checks: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _check_position_risk(
        self,
        trade: Dict,
        context: Dict
    ) -> Dict:
        """Check position-specific risks."""
        try:
            # Calculate position risk metrics
            risk_metrics = await self._calculate_position_risk(trade)
            
            checks = {
                "passed": False,
                "var_check": False,
                "sharpe_check": False,
                "exposure_check": False,
                "metrics": risk_metrics
            }
            
            # Check Value at Risk
            position_size = trade["amount_in_usd"]
            var_limit = position_size * 0.1  # Max 10% VaR
            checks["var_check"] = risk_metrics.value_at_risk <= var_limit
            
            # Check Sharpe ratio
            min_sharpe = 1.5  # Minimum Sharpe ratio
            checks["sharpe_check"] = risk_metrics.sharpe_ratio >= min_sharpe
            
            # Check total exposure
            total_exposure = self._calculate_total_exposure()
            new_exposure = total_exposure + position_size
            max_exposure = self.portfolio["total_value"] * 0.5  # Max 50% exposure
            checks["exposure_check"] = new_exposure <= max_exposure
            
            # Overall check
            checks["passed"] = all([
                checks["var_check"],
                checks["sharpe_check"],
                checks["exposure_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error checking position risk: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _check_portfolio_risk(
        self,
        trade: Dict,
        context: Dict
    ) -> Dict:
        """Check portfolio-level risks."""
        try:
            # Calculate portfolio impact
            portfolio_metrics = await self._calculate_portfolio_impact(trade)
            
            checks = {
                "passed": False,
                "correlation_check": False,
                "concentration_check": False,
                "drawdown_check": False,
                "metrics": portfolio_metrics
            }
            
            # Check correlation with existing positions
            max_correlation = 0.7  # Maximum correlation allowed
            checks["correlation_check"] = portfolio_metrics["max_correlation"] <= max_correlation
            
            # Check concentration
            max_concentration = 0.2  # Maximum 20% in single position
            new_concentration = trade["amount_in_usd"] / self.portfolio["total_value"]
            checks["concentration_check"] = new_concentration <= max_concentration
            
            # Check drawdown
            new_drawdown = portfolio_metrics["projected_drawdown"]
            checks["drawdown_check"] = new_drawdown <= self.risk_params.max_drawdown
            
            # Overall check
            checks["passed"] = all([
                checks["correlation_check"],
                checks["concentration_check"],
                checks["drawdown_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio risk: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _calculate_position_risk(
        self,
        trade: Dict
    ) -> PositionRisk:
        """Calculate position risk metrics."""
        try:
            # Get historical volatility
            volatility = await self._calculate_volatility(
                trade["token_in"],
                trade["token_out"]
            )
            
            # Calculate VaR
            position_size = trade["amount_in_usd"]
            z_score = stats.norm.ppf(self.risk_params.confidence_level)
            value_at_risk = position_size * volatility * z_score
            
            # Calculate Expected Shortfall (CVaR)
            expected_shortfall = value_at_risk * 1.3  # Approximate ES
            
            # Calculate Sharpe ratio
            returns = await self._calculate_historical_returns(trade)
            if returns:
                excess_returns = returns - self.risk_params.risk_free_rate
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
            else:
                sharpe_ratio = 0.0
            
            # Calculate profit factor and win rate
            profit_factor, win_rate = self._calculate_trade_metrics(trade)
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(returns if returns is not None else [])
            
            return PositionRisk(
                value_at_risk=value_at_risk,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                win_rate=win_rate,
                max_drawdown=max_drawdown
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {str(e)}")
            return PositionRisk(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    async def _calculate_portfolio_impact(
        self,
        trade: Dict
    ) -> Dict:
        """Calculate trade's impact on portfolio."""
        try:
            # Calculate correlations
            correlations = await self._calculate_correlations(trade)
            
            # Project new portfolio composition
            new_portfolio = self._project_portfolio(trade)
            
            # Calculate concentration
            concentration = self._calculate_concentration(new_portfolio)
            
            # Project drawdown
            projected_drawdown = await self._project_drawdown(
                trade,
                new_portfolio
            )
            
            return {
                "max_correlation": max(correlations.values(), default=0.0),
                "correlations": correlations,
                "concentration": concentration,
                "projected_drawdown": projected_drawdown,
                "risk_contribution": self._calculate_risk_contribution(
                    trade,
                    new_portfolio
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio impact: {str(e)}")
            return {}

    async def _calculate_volatility(
        self,
        token_in: str,
        token_out: str
    ) -> float:
        """Calculate historical volatility."""
        try:
            # Get price history
            prices_in = await self.market_data.get_price_history(token_in)
            prices_out = await self.market_data.get_price_history(token_out)
            
            if not prices_in or not prices_out:
                return 0.0
            
            # Calculate returns
            returns_in = np.diff(prices_in) / prices_in[:-1]
            returns_out = np.diff(prices_out) / prices_out[:-1]
            
            # Calculate spread volatility
            spread_returns = returns_out - returns_in
            volatility = np.std(spread_returns)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    async def _calculate_historical_returns(
        self,
        trade: Dict
    ) -> Optional[np.ndarray]:
        """Calculate historical returns for similar trades."""
        try:
            similar_trades = [
                t for t in self.position_history
                if (
                    t["token_in"] == trade["token_in"] and
                    t["token_out"] == trade["token_out"]
                )
            ]
            
            if not similar_trades:
                return None
            
            returns = np.array([
                t["profit_loss"] / t["amount_in_usd"]
                for t in similar_trades
            ])
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            return None

    def _calculate_trade_metrics(
        self,
        trade: Dict
    ) -> Tuple[float, float]:
        """Calculate profit factor and win rate."""
        try:
            similar_trades = [
                t for t in self.position_history
                if (
                    t["token_in"] == trade["token_in"] and
                    t["token_out"] == trade["token_out"]
                )
            ]
            
            if not similar_trades:
                return 1.0, 0.5
            
            profits = sum(t["profit_loss"] for t in similar_trades if t["profit_loss"] > 0)
            losses = abs(sum(t["profit_loss"] for t in similar_trades if t["profit_loss"] < 0))
            
            profit_factor = profits / max(losses, 1e-10)
            win_rate = len([t for t in similar_trades if t["profit_loss"] > 0]) / len(similar_trades)
            
            return profit_factor, win_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {str(e)}")
            return 1.0, 0.5

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate drawdowns
            drawdowns = (running_max - cum_returns) / running_max
            
            return np.max(drawdowns)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    async def _calculate_correlations(
        self,
        trade: Dict
    ) -> Dict[str, float]:
        """Calculate correlations with existing positions."""
        try:
            correlations = {}
            
            for position_id, position in self.active_positions.items():
                # Get price histories
                prices1 = await self.market_data.get_price_history(
                    trade["token_in"]
                )
                prices2 = await self.market_data.get_price_history(
                    position["token_in"]
                )
                
                if prices1 is not None and prices2 is not None:
                    # Calculate correlation
                    correlation = np.corrcoef(prices1, prices2)[0, 1]
                    correlations[position_id] = abs(correlation)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {str(e)}")
            return {}

    def _project_portfolio(self, trade: Dict) -> Dict:
        """Project new portfolio composition."""
        try:
            new_portfolio = {
                "total_value": self.portfolio["total_value"],
                "token_balances": self.portfolio["token_balances"].copy(),
                "token_allocations": self.portfolio["token_allocations"].copy(),
                "risk_exposure": self.portfolio["risk_exposure"]
            }
            
            # Add new position
            position_size = trade["amount_in_usd"]
            new_portfolio["total_value"] += position_size
            
            # Update token balances
            for token in [trade["token_in"], trade["token_out"]]:
                if token not in new_portfolio["token_balances"]:
                    new_portfolio["token_balances"][token] = 0.0
                new_portfolio["token_balances"][token] += position_size / 2
            
            # Update allocations
            total_value = new_portfolio["total_value"]
            new_portfolio["token_allocations"] = {
                token: balance / total_value
                for token, balance in new_portfolio["token_balances"].items()
            }
            
            # Update risk exposure
            new_portfolio["risk_exposure"] += position_size
            
            return new_portfolio
            
        except Exception as e:
            self.logger.error(f"Error projecting portfolio: {str(e)}")
            return self.portfolio.copy()

    def _calculate_concentration(self, portfolio: Dict) -> float:
        """Calculate portfolio concentration."""
        try:
            # Use Herfindahl-Hirschman Index (HHI)
            allocations = list(portfolio["token_allocations"].values())
            hhi = sum(x * x for x in allocations)
            
            return hhi
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration: {str(e)}")
            return 0.0

    async def _project_drawdown(
        self,
        trade: Dict,
        portfolio: Dict
    ) -> float:
        """Project potential drawdown."""
        try:
            # Get historical drawdowns
            drawdowns = []
            
            for token, allocation in portfolio["token_allocations"].items():
                prices = await self.market_data.get_price_history(token)
                if prices is not None:
                    returns = np.diff(prices) / prices[:-1]
                    token_drawdown = self._calculate_max_drawdown(returns)
                    drawdowns.append(token_drawdown * allocation)
            
            # Calculate portfolio drawdown
            if drawdowns:
                return sum(drawdowns)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error projecting drawdown: {str(e)}")
            return 0.0

    def _calculate_risk_contribution(
        self,
        trade: Dict,
        portfolio: Dict
    ) -> float:
        """Calculate trade's contribution to portfolio risk."""
        try:
            # Use equal risk contribution approach
            n_positions = len(portfolio["token_allocations"])
            target_contribution = 1.0 / n_positions
            
            new_allocation = trade["amount_in_usd"] / portfolio["total_value"]
            
            return abs(new_allocation - target_contribution)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk contribution: {str(e)}")
            return 0.0

    def _calculate_risk_score(
        self,
        basic_checks: Dict,
        position_checks: Dict,
        portfolio_checks: Dict
    ) -> float:
        """Calculate overall risk score."""
        try:
            scores = []
            
            # Basic risk score
            if basic_checks["passed"]:
                basic_score = 0.3
            else:
                basic_score = 0.7
            scores.append(basic_score)
            
            # Position risk score
            position_metrics = position_checks["metrics"]
            position_score = (
                position_metrics.value_at_risk / position_metrics.expected_shortfall +
                (1 - position_metrics.sharpe_ratio / 3) +
                position_metrics.max_drawdown
            ) / 3
            scores.append(position_score)
            
            # Portfolio risk score
            portfolio_metrics = portfolio_checks["metrics"]
            portfolio_score = (
                portfolio_metrics["max_correlation"] +
                portfolio_metrics["concentration"] +
                portfolio_metrics["projected_drawdown"]
            ) / 3
            scores.append(portfolio_score)
            
            # Calculate weighted average
            weights = [0.3, 0.4, 0.3]  # Basic, Position, Portfolio weights
            risk_score = sum(s * w for s, w in zip(scores, weights))
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            return 1.0

    async def update_position(
        self,
        position_id: str,
        status: str,
        profit_loss: float
    ):
        """Update position status and metrics."""
        try:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                if status == "closed":
                    # Update daily stats
                    self.daily_stats["trades"] += 1
                    if profit_loss > 0:
                        self.daily_stats["profit"] += profit_loss
                        self.daily_stats["successful_trades"] += 1
                    else:
                        self.daily_stats["loss"] += abs(profit_loss)
                    
                    # Update metrics
                    self.metrics["total_trades"] += 1
                    if profit_loss > 0:
                        self.metrics["profitable_trades"] += 1
                        self.metrics["total_profit"] += profit_loss
                    else:
                        self.metrics["total_loss"] += abs(profit_loss)
                    
                    # Update portfolio
                    await self._update_portfolio(position, profit_loss)
                    
                    # Add to history
                    self.position_history.append({
                        **position,
                        "profit_loss": profit_loss,
                        "close_time": datetime.utcnow()
                    })
                    
                    # Remove from active positions
                    del self.active_positions[position_id]
            
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")

    async def _update_portfolio(
        self,
        position: Dict,
        profit_loss: float
    ):
        """Update portfolio state."""
        try:
            # Update total value
            self.portfolio["total_value"] += profit_loss
            
            # Update token balances
            for token in [position["token_in"], position["token_out"]]:
                if token in self.portfolio["token_balances"]:
                    self.portfolio["token_balances"][token] -= position["amount_in_usd"] / 2
                    if self.portfolio["token_balances"][token] <= 0:
                        del self.portfolio["token_balances"][token]
            
            # Update allocations
            total_value = self.portfolio["total_value"]
            self.portfolio["token_allocations"] = {
                token: balance / total_value
                for token, balance in self.portfolio["token_balances"].items()
            }
            
            # Update risk exposure
            self.portfolio["risk_exposure"] -= position["amount_in_usd"]
            
            # Update available capital
            self.portfolio["available_capital"] = (
                self.portfolio["total_value"] -
                self.portfolio["risk_exposure"]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {str(e)}")

    async def reset_daily_stats(self):
        """Reset daily statistics."""
        try:
            self.daily_stats = {
                "volume": 0.0,
                "profit": 0.0,
                "loss": 0.0,
                "trades": 0,
                "successful_trades": 0,
                "gas_spent": 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error resetting daily stats: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "win_rate": (
                self.metrics["profitable_trades"] /
                max(1, self.metrics["total_trades"])
            ),
            "profit_factor": (
                self.metrics["total_profit"] /
                max(1e-10, self.metrics["total_loss"])
            ),
            "active_positions": len(self.active_positions),
            "portfolio": self.portfolio,
            "daily_stats": self.daily_stats,
            "last_update": datetime.utcnow().isoformat()
        }
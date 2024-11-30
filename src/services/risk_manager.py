from typing import Dict, List, Optional, Set, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address

class RiskManager:
    """Manage risk across chains and operations."""
    def __init__(
        self,
        settings: Dict,
        market_data_aggregator,
        bridge_manager
    ):
        self.settings = settings
        self.market_data = market_data_aggregator
        self.bridge_manager = bridge_manager
        self.logger = logging.getLogger(__name__)
        
        # Risk limits
        self.limits = {
            "max_position_size_usd": 50000.0,
            "max_daily_volume_usd": 1000000.0,
            "max_daily_loss_usd": 10000.0,
            "max_slippage": 0.01,
            "min_liquidity_ratio": 10.0,  # Position size must be 10x smaller than liquidity
            "max_gas_cost_ratio": 0.1,    # Max 10% of profit
            "min_profit_usd": 100.0
        }
        
        # Risk metrics
        self.metrics = {
            "daily_volume_usd": 0.0,
            "daily_profit_usd": 0.0,
            "daily_loss_usd": 0.0,
            "current_positions": {},
            "failed_validations": 0,
            "risk_warnings": []
        }
        
        # Historical data
        self.trade_history: List[Dict] = []
        self.risk_events: List[Dict] = []
        
        # Initialize risk models
        self._init_risk_models()

    def _init_risk_models(self):
        """Initialize risk models and parameters."""
        # Volatility calculation parameters
        self.volatility_window = 100
        self.volatility_threshold = 0.02  # 2% threshold
        
        # Correlation parameters
        self.correlation_threshold = 0.7
        
        # Risk scoring weights
        self.risk_weights = {
            "volatility": 0.3,
            "liquidity": 0.3,
            "correlation": 0.2,
            "gas_cost": 0.1,
            "bridge_risk": 0.1
        }

    async def validate_opportunity(
        self,
        opportunity: Dict,
        chain_id: int
    ) -> Tuple[bool, Dict]:
        """Validate arbitrage opportunity against risk parameters."""
        try:
            validation = {
                "passed": False,
                "checks": {},
                "risk_score": 0.0,
                "warnings": []
            }
            
            # Basic checks
            basic_checks = await self._perform_basic_checks(
                opportunity,
                chain_id
            )
            validation["checks"]["basic"] = basic_checks
            
            if not basic_checks["passed"]:
                return False, validation
            
            # Market risk checks
            market_checks = await self._check_market_risks(
                opportunity,
                chain_id
            )
            validation["checks"]["market"] = market_checks
            
            if not market_checks["passed"]:
                return False, validation
            
            # Position risk checks
            position_checks = await self._check_position_risks(
                opportunity,
                chain_id
            )
            validation["checks"]["position"] = position_checks
            
            if not position_checks["passed"]:
                return False, validation
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                basic_checks,
                market_checks,
                position_checks
            )
            validation["risk_score"] = risk_score
            
            # Final validation
            passed = (
                basic_checks["passed"] and
                market_checks["passed"] and
                position_checks["passed"] and
                risk_score <= 0.7  # Max 70% risk score
            )
            
            validation["passed"] = passed
            
            if not passed:
                self.metrics["failed_validations"] += 1
            
            return passed, validation
            
        except Exception as e:
            self.logger.error(f"Error validating opportunity: {str(e)}")
            return False, {"passed": False, "error": str(e)}

    async def _perform_basic_checks(
        self,
        opportunity: Dict,
        chain_id: int
    ) -> Dict:
        """Perform basic validation checks."""
        try:
            checks = {
                "passed": False,
                "profit_check": False,
                "slippage_check": False,
                "gas_check": False
            }
            
            # Check minimum profit
            expected_profit = opportunity["profit_analysis"]["expected_profit"]
            checks["profit_check"] = expected_profit >= self.limits["min_profit_usd"]
            
            # Check slippage
            max_slippage = opportunity["profit_analysis"].get("max_slippage", 1.0)
            checks["slippage_check"] = max_slippage <= self.limits["max_slippage"]
            
            # Check gas costs
            gas_cost = opportunity["gas_data"]["gas_price"] * 500000  # Estimate gas usage
            gas_cost_ratio = gas_cost / expected_profit
            checks["gas_check"] = gas_cost_ratio <= self.limits["max_gas_cost_ratio"]
            
            # Overall check
            checks["passed"] = all([
                checks["profit_check"],
                checks["slippage_check"],
                checks["gas_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error in basic checks: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _check_market_risks(
        self,
        opportunity: Dict,
        chain_id: int
    ) -> Dict:
        """Check market-related risks."""
        try:
            checks = {
                "passed": False,
                "liquidity_check": False,
                "volatility_check": False,
                "correlation_check": False
            }
            
            # Check liquidity
            position_size = opportunity["amount_in_usd"]
            min_liquidity = position_size * self.limits["min_liquidity_ratio"]
            
            liquidity = await self._get_path_liquidity(
                opportunity["path"],
                chain_id
            )
            checks["liquidity_check"] = liquidity >= min_liquidity
            
            # Check volatility
            volatility = await self._calculate_path_volatility(
                opportunity["path"],
                chain_id
            )
            checks["volatility_check"] = volatility <= self.volatility_threshold
            
            # Check correlations
            correlation = await self._calculate_path_correlation(
                opportunity["path"],
                chain_id
            )
            checks["correlation_check"] = correlation <= self.correlation_threshold
            
            # Overall check
            checks["passed"] = all([
                checks["liquidity_check"],
                checks["volatility_check"],
                checks["correlation_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error in market checks: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _check_position_risks(
        self,
        opportunity: Dict,
        chain_id: int
    ) -> Dict:
        """Check position-related risks."""
        try:
            checks = {
                "passed": False,
                "size_check": False,
                "exposure_check": False,
                "daily_limit_check": False
            }
            
            # Check position size
            position_size = opportunity["amount_in_usd"]
            checks["size_check"] = position_size <= self.limits["max_position_size_usd"]
            
            # Check total exposure
            total_exposure = self._calculate_total_exposure(chain_id)
            new_exposure = total_exposure + position_size
            checks["exposure_check"] = new_exposure <= self.limits["max_position_size_usd"] * 2
            
            # Check daily limits
            daily_volume = self.metrics["daily_volume_usd"] + position_size
            checks["daily_limit_check"] = daily_volume <= self.limits["max_daily_volume_usd"]
            
            # Overall check
            checks["passed"] = all([
                checks["size_check"],
                checks["exposure_check"],
                checks["daily_limit_check"]
            ])
            
            return checks
            
        except Exception as e:
            self.logger.error(f"Error in position checks: {str(e)}")
            return {"passed": False, "error": str(e)}

    async def _get_path_liquidity(
        self,
        path: List[str],
        chain_id: int
    ) -> float:
        """Get minimum liquidity along path."""
        try:
            liquidities = []
            
            for token in path:
                liquidity = await self.market_data.get_total_liquidity(token)
                if liquidity:
                    liquidities.append(liquidity)
            
            return min(liquidities) if liquidities else 0
            
        except Exception as e:
            self.logger.error(f"Error getting path liquidity: {str(e)}")
            return 0

    async def _calculate_path_volatility(
        self,
        path: List[str],
        chain_id: int
    ) -> float:
        """Calculate path volatility."""
        try:
            volatilities = []
            
            for token in path:
                # Get price history
                prices = await self.market_data.get_price_history(
                    token,
                    self.volatility_window
                )
                
                if len(prices) >= 2:
                    # Calculate returns
                    returns = np.diff(prices) / prices[:-1]
                    # Calculate volatility
                    volatility = np.std(returns)
                    volatilities.append(volatility)
            
            return max(volatilities) if volatilities else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating path volatility: {str(e)}")
            return 1.0

    async def _calculate_path_correlation(
        self,
        path: List[str],
        chain_id: int
    ) -> float:
        """Calculate path token correlations."""
        try:
            if len(path) < 2:
                return 0
            
            correlations = []
            
            for i in range(len(path)-1):
                for j in range(i+1, len(path)):
                    # Get price histories
                    prices1 = await self.market_data.get_price_history(
                        path[i],
                        self.volatility_window
                    )
                    prices2 = await self.market_data.get_price_history(
                        path[j],
                        self.volatility_window
                    )
                    
                    if len(prices1) == len(prices2) >= 2:
                        # Calculate correlation
                        correlation = np.corrcoef(prices1, prices2)[0, 1]
                        correlations.append(abs(correlation))
            
            return max(correlations) if correlations else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating path correlation: {str(e)}")
            return 0

    def _calculate_total_exposure(self, chain_id: int) -> float:
        """Calculate total position exposure."""
        try:
            exposure = 0.0
            
            for position in self.metrics["current_positions"].values():
                if position["chain_id"] == chain_id:
                    exposure += position["amount_usd"]
            
            return exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating exposure: {str(e)}")
            return 0.0

    def _calculate_risk_score(
        self,
        basic_checks: Dict,
        market_checks: Dict,
        position_checks: Dict
    ) -> float:
        """Calculate overall risk score."""
        try:
            scores = {
                "volatility": 0.0,
                "liquidity": 0.0,
                "correlation": 0.0,
                "gas_cost": 0.0,
                "bridge_risk": 0.0
            }
            
            # Volatility score
            if "volatility_check" in market_checks:
                scores["volatility"] = market_checks["volatility_check"]
            
            # Liquidity score
            if "liquidity_check" in market_checks:
                scores["liquidity"] = market_checks["liquidity_check"]
            
            # Correlation score
            if "correlation_check" in market_checks:
                scores["correlation"] = market_checks["correlation_check"]
            
            # Gas cost score
            if "gas_check" in basic_checks:
                scores["gas_cost"] = basic_checks["gas_check"]
            
            # Calculate weighted score
            risk_score = sum(
                scores[k] * self.risk_weights[k]
                for k in scores
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            return 1.0

    async def validate_bridge_transfer(
        self,
        bridge_id: str,
        amount: int,
        source_chain: int,
        dest_chain: int
    ) -> Tuple[bool, Dict]:
        """Validate bridge transfer."""
        try:
            validation = {
                "passed": False,
                "checks": {},
                "risk_score": 0.0,
                "warnings": []
            }
            
            # Get bridge data
            bridge_data = await self.bridge_manager.get_bridge_state(bridge_id)
            
            if not bridge_data:
                return False, validation
            
            # Check liquidity
            liquidity_check = bridge_data["liquidity"] >= amount * 2
            validation["checks"]["liquidity"] = liquidity_check
            
            # Check volume limits
            volume_check = (
                bridge_data["volume_24h"] + amount <=
                self.limits["max_daily_volume_usd"]
            )
            validation["checks"]["volume"] = volume_check
            
            # Check bridge reliability
            reliability = self._calculate_bridge_reliability(bridge_data)
            reliability_check = reliability >= 0.95  # 95% reliability required
            validation["checks"]["reliability"] = reliability_check
            
            # Calculate risk score
            risk_score = self._calculate_bridge_risk_score(
                bridge_data,
                amount,
                reliability
            )
            validation["risk_score"] = risk_score
            
            # Overall validation
            validation["passed"] = all([
                liquidity_check,
                volume_check,
                reliability_check,
                risk_score <= 0.7
            ])
            
            return validation["passed"], validation
            
        except Exception as e:
            self.logger.error(f"Error validating bridge transfer: {str(e)}")
            return False, {"passed": False, "error": str(e)}

    def _calculate_bridge_reliability(self, bridge_data: Dict) -> float:
        """Calculate bridge reliability score."""
        try:
            if bridge_data["metrics"]["total_transfers"] == 0:
                return 0.0
            
            return (
                bridge_data["metrics"]["successful_transfers"] /
                bridge_data["metrics"]["total_transfers"]
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating bridge reliability: {str(e)}")
            return 0.0

    def _calculate_bridge_risk_score(
        self,
        bridge_data: Dict,
        amount: int,
        reliability: float
    ) -> float:
        """Calculate bridge risk score."""
        try:
            # Liquidity utilization risk
            liquidity_risk = amount / bridge_data["liquidity"]
            
            # Volume risk
            volume_risk = bridge_data["volume_24h"] / self.limits["max_daily_volume_usd"]
            
            # Reliability risk
            reliability_risk = 1 - reliability
            
            # Combine risks
            risk_score = (
                liquidity_risk * 0.4 +
                volume_risk * 0.3 +
                reliability_risk * 0.3
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating bridge risk score: {str(e)}")
            return 1.0

    async def update_position(
        self,
        position_id: str,
        status: str,
        profit_loss: float
    ):
        """Update position status and metrics."""
        try:
            if position_id in self.metrics["current_positions"]:
                position = self.metrics["current_positions"][position_id]
                
                if status == "closed":
                    # Update daily metrics
                    if profit_loss > 0:
                        self.metrics["daily_profit_usd"] += profit_loss
                    else:
                        self.metrics["daily_loss_usd"] += abs(profit_loss)
                    
                    # Add to history
                    self.trade_history.append({
                        **position,
                        "profit_loss": profit_loss,
                        "close_time": datetime.utcnow()
                    })
                    
                    # Remove from current positions
                    del self.metrics["current_positions"][position_id]
                
                elif status == "failed":
                    # Add to risk events
                    self.risk_events.append({
                        **position,
                        "error": "execution_failed",
                        "timestamp": datetime.utcnow()
                    })
                    
                    # Remove from current positions
                    del self.metrics["current_positions"][position_id]
            
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")

    async def reset_daily_metrics(self):
        """Reset daily metrics."""
        try:
            self.metrics["daily_volume_usd"] = 0.0
            self.metrics["daily_profit_usd"] = 0.0
            self.metrics["daily_loss_usd"] = 0.0
            self.metrics["failed_validations"] = 0
            self.metrics["risk_warnings"] = []
            
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "trade_history_size": len(self.trade_history),
            "risk_events": len(self.risk_events),
            "last_update": datetime.utcnow().isoformat()
        }
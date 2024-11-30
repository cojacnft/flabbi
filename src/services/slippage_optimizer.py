from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3
from eth_typing import Address
import aiohttp
import json

class SlippageOptimizer:
    """Optimize slippage parameters for arbitrage trades."""
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
        
        # Slippage history
        self.trade_history: List[Dict] = []
        self.history_window = 1000
        
        # Pool states
        self.pool_states: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "avg_slippage": 0.0,
            "total_savings_usd": 0.0
        }
        
        # Initialize ML model
        self._init_model()

    def _init_model(self):
        """Initialize slippage prediction model."""
        try:
            # Load model if exists
            model_path = "models/slippage_predictor.joblib"
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(f"{model_path}.scaler")
            except:
                # Train new model
                self._train_initial_model()
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")

    def _train_initial_model(self):
        """Train initial slippage prediction model."""
        try:
            # Create simple initial training data
            X = np.random.rand(1000, 8)  # 8 features
            y = np.mean(X, axis=1) * 0.01  # Simple slippage rule
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            self.logger.error(f"Error training initial model: {str(e)}")

    async def optimize_slippage(
        self,
        path: List[Dict],
        amounts: List[int],
        min_profit: float
    ) -> Dict:
        """Optimize slippage parameters for trade path."""
        try:
            # Get current market state
            market_state = await self._get_market_state(path)
            
            # Calculate base slippage
            base_slippage = await self._calculate_base_slippage(
                path,
                amounts,
                market_state
            )
            
            # Predict optimal slippage
            predictions = await self._predict_slippage(
                base_slippage,
                market_state
            )
            
            # Validate predictions
            validated = await self._validate_slippage(
                predictions,
                path,
                amounts,
                min_profit
            )
            
            # Update metrics
            self.metrics["total_trades"] += 1
            self.metrics["avg_slippage"] = (
                (self.metrics["avg_slippage"] *
                 (self.metrics["total_trades"] - 1) +
                 validated["slippage"]) /
                self.metrics["total_trades"]
            )
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error optimizing slippage: {str(e)}")
            return {"slippage": 0.01}  # Default 1% slippage

    async def _get_market_state(self, path: List[Dict]) -> Dict:
        """Get current market state for path."""
        try:
            state = {}
            
            # Get pool states
            for step in path:
                pool_state = await self._get_pool_state(
                    step["pool"],
                    step["token_in"],
                    step["token_out"]
                )
                state[step["pool"]] = pool_state
            
            # Get market metrics
            state["volatility"] = await self._calculate_path_volatility(path)
            state["liquidity"] = await self._calculate_path_liquidity(path)
            state["volume"] = await self._calculate_path_volume(path)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {}

    async def _get_pool_state(
        self,
        pool_address: str,
        token_in: str,
        token_out: str
    ) -> Dict:
        """Get pool state and metrics."""
        try:
            # Get pool contract
            pool = self.web3.eth.contract(
                address=pool_address,
                abi=self._get_pool_abi(pool_address)
            )
            
            # Get reserves
            reserves = await pool.functions.getReserves().call()
            
            # Get token order
            token0 = await pool.functions.token0().call()
            token1 = await pool.functions.token1().call()
            
            # Organize reserves
            if token_in.lower() == token0.lower():
                reserve_in = reserves[0]
                reserve_out = reserves[1]
            else:
                reserve_in = reserves[1]
                reserve_out = reserves[0]
            
            # Calculate metrics
            price = reserve_out / reserve_in
            liquidity = min(
                reserve_in * self._get_token_price(token_in),
                reserve_out * self._get_token_price(token_out)
            )
            
            return {
                "reserve_in": reserve_in,
                "reserve_out": reserve_out,
                "price": price,
                "liquidity": liquidity,
                "last_update": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pool state: {str(e)}")
            return {}

    async def _calculate_base_slippage(
        self,
        path: List[Dict],
        amounts: List[int],
        market_state: Dict
    ) -> Dict:
        """Calculate base slippage parameters."""
        try:
            slippages = []
            
            # Calculate slippage for each step
            for i, step in enumerate(path):
                pool_state = market_state[step["pool"]]
                
                # Calculate price impact
                price_impact = self._calculate_price_impact(
                    amounts[i],
                    pool_state["reserve_in"],
                    pool_state["reserve_out"]
                )
                
                # Calculate optimal slippage
                optimal_slippage = self._calculate_optimal_slippage(
                    price_impact,
                    pool_state["liquidity"],
                    market_state["volatility"]
                )
                
                slippages.append(optimal_slippage)
            
            # Combine slippages
            return {
                "per_step": slippages,
                "total": sum(slippages),
                "max": max(slippages)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating base slippage: {str(e)}")
            return {"per_step": [], "total": 0.01, "max": 0.01}

    def _calculate_price_impact(
        self,
        amount_in: int,
        reserve_in: int,
        reserve_out: int
    ) -> float:
        """Calculate price impact of trade."""
        try:
            # Using constant product formula (x * y = k)
            amount_out = (amount_in * reserve_out) // (reserve_in + amount_in)
            expected_out = (amount_in * reserve_out) // reserve_in
            
            price_impact = (expected_out - amount_out) / expected_out
            return price_impact
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {str(e)}")
            return 0.0

    def _calculate_optimal_slippage(
        self,
        price_impact: float,
        liquidity: float,
        volatility: float
    ) -> float:
        """Calculate optimal slippage tolerance."""
        try:
            # Base slippage from price impact
            base_slippage = price_impact * 1.5  # 50% buffer
            
            # Adjust for liquidity
            liquidity_factor = 1 + (1 / np.log10(max(liquidity, 10)))
            
            # Adjust for volatility
            volatility_factor = 1 + (volatility * 2)
            
            # Combine factors
            optimal_slippage = (
                base_slippage *
                liquidity_factor *
                volatility_factor
            )
            
            # Clamp to reasonable range
            return min(0.05, max(0.001, optimal_slippage))
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal slippage: {str(e)}")
            return 0.01

    async def _predict_slippage(
        self,
        base_slippage: Dict,
        market_state: Dict
    ) -> Dict:
        """Predict optimal slippage using ML model."""
        try:
            # Extract features
            features = self._extract_features(
                base_slippage,
                market_state
            )
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                prediction,
                base_slippage,
                market_state
            )
            
            return {
                "slippage": prediction,
                "confidence": confidence,
                "base_slippage": base_slippage
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting slippage: {str(e)}")
            return {"slippage": base_slippage["total"], "confidence": 0.5}

    def _extract_features(
        self,
        base_slippage: Dict,
        market_state: Dict
    ) -> np.ndarray:
        """Extract features for prediction."""
        try:
            features = [
                base_slippage["total"],
                base_slippage["max"],
                market_state["volatility"],
                np.mean([s["liquidity"] for s in market_state.values()]),
                np.std([s["liquidity"] for s in market_state.values()]),
                market_state["volume"],
                len(base_slippage["per_step"]),
                np.mean(base_slippage["per_step"])
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 8))

    def _calculate_prediction_confidence(
        self,
        prediction: float,
        base_slippage: Dict,
        market_state: Dict
    ) -> float:
        """Calculate confidence in slippage prediction."""
        try:
            # Compare with base slippage
            base_diff = abs(prediction - base_slippage["total"])
            base_confidence = 1 - (base_diff / base_slippage["total"])
            
            # Check market conditions
            market_confidence = 1.0
            if market_state["volatility"] > 0.02:  # High volatility
                market_confidence *= 0.8
            if market_state["volume"] < 100000:  # Low volume
                market_confidence *= 0.9
            
            # Check historical accuracy
            if self.metrics["total_trades"] > 0:
                historical_confidence = (
                    self.metrics["successful_trades"] /
                    self.metrics["total_trades"]
                )
            else:
                historical_confidence = 0.5
            
            # Combine confidence scores
            confidence = (
                base_confidence * 0.4 +
                market_confidence * 0.3 +
                historical_confidence * 0.3
            )
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    async def _validate_slippage(
        self,
        predictions: Dict,
        path: List[Dict],
        amounts: List[int],
        min_profit: float
    ) -> Dict:
        """Validate predicted slippage parameters."""
        try:
            slippage = predictions["slippage"]
            
            # Simulate trade with slippage
            result = await self._simulate_trade(
                path,
                amounts,
                slippage
            )
            
            if not result["success"]:
                # Increase slippage if simulation failed
                slippage *= 1.2
                result = await self._simulate_trade(
                    path,
                    amounts,
                    slippage
                )
            
            # Verify profit remains above minimum
            if result["profit"] < min_profit:
                # Adjust slippage to maintain profit
                max_slippage = self._calculate_max_slippage(
                    result["profit"],
                    min_profit
                )
                slippage = min(slippage, max_slippage)
            
            return {
                "slippage": slippage,
                "simulation": result,
                "confidence": predictions["confidence"]
            }
            
        except Exception as e:
            self.logger.error(f"Error validating slippage: {str(e)}")
            return {"slippage": 0.01, "confidence": 0.5}

    async def _simulate_trade(
        self,
        path: List[Dict],
        amounts: List[int],
        slippage: float
    ) -> Dict:
        """Simulate trade with given slippage."""
        try:
            current_amount = amounts[0]
            total_price_impact = 0
            
            for i, step in enumerate(path):
                # Get pool state
                pool_state = await self._get_pool_state(
                    step["pool"],
                    step["token_in"],
                    step["token_out"]
                )
                
                # Calculate output amount
                output = self._calculate_output_amount(
                    current_amount,
                    pool_state,
                    slippage
                )
                
                if output == 0:
                    return {
                        "success": False,
                        "profit": 0,
                        "price_impact": 1.0
                    }
                
                # Update amount for next step
                current_amount = output
                
                # Calculate price impact
                price_impact = self._calculate_price_impact(
                    amounts[i],
                    pool_state["reserve_in"],
                    pool_state["reserve_out"]
                )
                total_price_impact += price_impact
            
            # Calculate profit
            profit = self._calculate_profit(
                amounts[0],
                current_amount,
                path[0]["token_in"]
            )
            
            return {
                "success": True,
                "profit": profit,
                "price_impact": total_price_impact
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating trade: {str(e)}")
            return {"success": False, "profit": 0, "price_impact": 1.0}

    def _calculate_output_amount(
        self,
        amount_in: int,
        pool_state: Dict,
        slippage: float
    ) -> int:
        """Calculate output amount with slippage."""
        try:
            # Calculate ideal output
            ideal_out = (
                amount_in *
                pool_state["reserve_out"] //
                pool_state["reserve_in"]
            )
            
            # Apply slippage
            min_out = int(ideal_out * (1 - slippage))
            
            return min_out
            
        except Exception as e:
            self.logger.error(f"Error calculating output amount: {str(e)}")
            return 0

    def _calculate_max_slippage(
        self,
        current_profit: float,
        min_profit: float
    ) -> float:
        """Calculate maximum allowable slippage."""
        try:
            if current_profit <= min_profit:
                return 0.001  # Minimum slippage
            
            # Calculate maximum slippage that maintains minimum profit
            profit_buffer = current_profit - min_profit
            max_slippage = profit_buffer / current_profit
            
            # Add safety margin
            return max_slippage * 0.9  # 10% safety margin
            
        except Exception as e:
            self.logger.error(f"Error calculating max slippage: {str(e)}")
            return 0.01

    async def update_history(
        self,
        path: List[Dict],
        amounts: List[int],
        slippage: float,
        actual_output: int,
        success: bool
    ):
        """Update trade history with results."""
        try:
            # Calculate actual slippage
            expected_output = await self._calculate_expected_output(
                path,
                amounts
            )
            
            actual_slippage = (
                expected_output - actual_output
            ) / expected_output
            
            # Add to history
            self.trade_history.append({
                "expected_slippage": slippage,
                "actual_slippage": actual_slippage,
                "success": success,
                "timestamp": datetime.utcnow()
            })
            
            # Keep fixed window size
            if len(self.trade_history) > self.history_window:
                self.trade_history = self.trade_history[-self.history_window:]
            
            # Update metrics
            if success:
                self.metrics["successful_trades"] += 1
                if actual_slippage < slippage:
                    savings = (
                        (slippage - actual_slippage) *
                        self._get_token_price(path[-1]["token_out"]) *
                        actual_output
                    )
                    self.metrics["total_savings_usd"] += savings
            else:
                self.metrics["failed_trades"] += 1
            
            # Retrain model periodically
            if len(self.trade_history) % 100 == 0:
                await self._retrain_model()
            
        except Exception as e:
            self.logger.error(f"Error updating history: {str(e)}")

    async def _calculate_expected_output(
        self,
        path: List[Dict],
        amounts: List[int]
    ) -> int:
        """Calculate expected output without slippage."""
        try:
            current_amount = amounts[0]
            
            for i, step in enumerate(path):
                pool_state = await self._get_pool_state(
                    step["pool"],
                    step["token_in"],
                    step["token_out"]
                )
                
                current_amount = (
                    current_amount *
                    pool_state["reserve_out"] //
                    pool_state["reserve_in"]
                )
            
            return current_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating expected output: {str(e)}")
            return 0

    async def _retrain_model(self):
        """Retrain slippage prediction model."""
        try:
            if len(self.trade_history) < 100:
                return
            
            # Prepare training data
            X = []
            y = []
            
            for i in range(len(self.trade_history) - 10):
                # Get historical window
                window = self.trade_history[i:i+10]
                
                # Extract features
                features = [
                    np.mean([t["expected_slippage"] for t in window]),
                    np.std([t["expected_slippage"] for t in window]),
                    np.mean([t["actual_slippage"] for t in window]),
                    np.std([t["actual_slippage"] for t in window]),
                    sum(1 for t in window if t["success"]) / len(window),
                    window[-1]["expected_slippage"],
                    window[-1]["actual_slippage"],
                    1 if window[-1]["success"] else 0
                ]
                
                X.append(features)
                y.append(window[-1]["actual_slippage"])
            
            # Convert to arrays
            X = np.array(X)
            y = np.array(y)
            
            # Update scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train new model
            new_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            new_model.fit(X_scaled, y)
            
            # Update if better
            old_score = self.model.score(X_scaled, y)
            new_score = new_model.score(X_scaled, y)
            
            if new_score > old_score:
                self.model = new_model
                joblib.dump(self.model, "models/slippage_predictor.joblib")
                joblib.dump(self.scaler, "models/slippage_predictor.joblib.scaler")
                self.logger.info(f"Model updated with score: {new_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_trades"] /
                max(1, self.metrics["total_trades"])
            ),
            "last_update": datetime.utcnow().isoformat()
        }
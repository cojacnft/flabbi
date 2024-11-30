from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
import asyncio

from ..models.token import Token
from ..services.market_data import MarketDataAggregator
from ..services.profit_analyzer import ProfitAnalyzer

class MLPathFinder:
    def __init__(
        self,
        market_data: MarketDataAggregator,
        profit_analyzer: ProfitAnalyzer,
        model_path: str = "models/path_predictor.joblib"
    ):
        self.market_data = market_data
        self.profit_analyzer = profit_analyzer
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # ML components
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Historical data
        self.historical_paths: List[Dict] = []
        self.successful_paths: Set[str] = set()
        
        # Performance metrics
        self.metrics = {
            "predictions_made": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "model_accuracy": 0.0,
            "avg_prediction_time_ms": 0.0
        }
        
        # Initialize ML model
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load existing model or train a new one."""
        try:
            # Try to load existing model
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(f"{self.model_path}.scaler")
            self.logger.info("Loaded existing ML model")
        except:
            self.logger.info("Training new ML model")
            self._train_initial_model()

    def _train_initial_model(self):
        """Train initial model with basic features."""
        try:
            # Create simple initial training data
            X = np.random.rand(1000, 10)  # 10 features
            y = (X[:, 0] * X[:, 1] > 0.5).astype(int)  # Simple rule
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, f"{self.model_path}.scaler")
            
        except Exception as e:
            self.logger.error(f"Error training initial model: {str(e)}")
            raise

    async def find_optimal_path(
        self,
        start_token: Token,
        max_hops: int = 4,
        min_profit_usd: float = 100.0
    ) -> Optional[Dict]:
        """Find optimal arbitrage path using ML predictions."""
        try:
            start_time = datetime.utcnow()
            
            # Get current market state
            market_state = await self._get_market_state(start_token)
            
            # Generate candidate paths
            candidates = await self._generate_candidates(
                start_token,
                max_hops,
                market_state
            )
            
            # Predict profitability
            profitable_paths = await self._predict_profitable_paths(
                candidates,
                market_state
            )
            
            # Validate and rank paths
            validated_paths = await self._validate_paths(
                profitable_paths,
                min_profit_usd
            )
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics["avg_prediction_time_ms"] = (
                self.metrics["avg_prediction_time_ms"] * self.metrics["predictions_made"] +
                execution_time
            ) / (self.metrics["predictions_made"] + 1)
            self.metrics["predictions_made"] += 1
            
            if validated_paths:
                return validated_paths[0]  # Return best path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding optimal path: {str(e)}")
            return None

    async def _get_market_state(self, token: Token) -> Dict:
        """Get current market state features."""
        try:
            # Get token metrics
            price = await self.market_data.get_token_price(token.address)
            volatility = await self.market_data.get_volatility(token.address)
            liquidity = await self.market_data.get_total_liquidity(token)
            
            # Get network metrics
            gas_price = await self.market_data.get_gas_price()
            network_load = await self.market_data.get_network_load()
            
            return {
                "price": price,
                "volatility": volatility,
                "liquidity": liquidity,
                "gas_price": gas_price,
                "network_load": network_load,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {}

    async def _generate_candidates(
        self,
        start_token: Token,
        max_hops: int,
        market_state: Dict
    ) -> List[Dict]:
        """Generate candidate arbitrage paths."""
        try:
            candidates = []
            visited = {start_token.address}
            
            async def explore_path(current_path: List[Token], depth: int):
                if depth >= max_hops:
                    # Check if path returns to start token
                    if current_path[-1].address == start_token.address:
                        candidates.append({
                            "path": current_path.copy(),
                            "features": await self._extract_path_features(
                                current_path,
                                market_state
                            )
                        })
                    return
                
                # Get connected tokens
                connected_tokens = await self._get_connected_tokens(
                    current_path[-1]
                )
                
                for next_token in connected_tokens:
                    if next_token.address not in visited or (
                        depth == max_hops - 1 and
                        next_token.address == start_token.address
                    ):
                        visited.add(next_token.address)
                        current_path.append(next_token)
                        await explore_path(current_path, depth + 1)
                        current_path.pop()
                        visited.remove(next_token.address)
            
            await explore_path([start_token], 1)
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error generating candidates: {str(e)}")
            return []

    async def _get_connected_tokens(self, token: Token) -> List[Token]:
        """Get tokens connected through liquidity pools."""
        try:
            connected = set()
            
            for pool in token.liquidity_pools:
                if pool.token0_address == token.address:
                    connected.add(pool.token1_address)
                else:
                    connected.add(pool.token0_address)
            
            # Get token objects
            tokens = []
            for addr in connected:
                token_obj = await self.market_data.get_token(addr)
                if token_obj:
                    tokens.append(token_obj)
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"Error getting connected tokens: {str(e)}")
            return []

    async def _extract_path_features(
        self,
        path: List[Token],
        market_state: Dict
    ) -> np.ndarray:
        """Extract features for ML prediction."""
        try:
            features = []
            
            # Path-specific features
            features.extend([
                len(path),  # Path length
                len(set(t.address for t in path)),  # Unique tokens
                market_state["gas_price"],  # Current gas price
                market_state["network_load"]  # Network load
            ])
            
            # Token-specific features
            for token in path:
                price = await self.market_data.get_token_price(token.address)
                volatility = await self.market_data.get_volatility(token.address)
                liquidity = await self.market_data.get_total_liquidity(token)
                
                features.extend([
                    price or 0,
                    volatility or 0,
                    liquidity or 0
                ])
            
            # Pad or truncate to fixed length
            target_length = 10  # Adjust based on model requirements
            if len(features) < target_length:
                features.extend([0] * (target_length - len(features)))
            else:
                features = features[:target_length]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 10))

    async def _predict_profitable_paths(
        self,
        candidates: List[Dict],
        market_state: Dict
    ) -> List[Dict]:
        """Predict profitability using ML model."""
        try:
            profitable_paths = []
            
            for candidate in candidates:
                # Scale features
                features_scaled = self.scaler.transform(
                    candidate["features"]
                )
                
                # Make prediction
                probability = self.model.predict_proba(features_scaled)[0][1]
                
                if probability > 0.7:  # High confidence threshold
                    profitable_paths.append({
                        **candidate,
                        "probability": probability
                    })
            
            # Sort by probability
            return sorted(
                profitable_paths,
                key=lambda x: x["probability"],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting profitable paths: {str(e)}")
            return []

    async def _validate_paths(
        self,
        paths: List[Dict],
        min_profit_usd: float
    ) -> List[Dict]:
        """Validate predicted profitable paths."""
        try:
            validated_paths = []
            
            for path in paths:
                # Simulate execution
                profit = await self.profit_analyzer.simulate_path_profit(
                    path["path"]
                )
                
                if profit and profit >= min_profit_usd:
                    validated_paths.append({
                        **path,
                        "expected_profit": profit
                    })
                    
                    # Update model metrics
                    self.metrics["successful_predictions"] += 1
                else:
                    self.metrics["false_positives"] += 1
            
            # Update accuracy
            total_predictions = (
                self.metrics["successful_predictions"] +
                self.metrics["false_positives"]
            )
            if total_predictions > 0:
                self.metrics["model_accuracy"] = (
                    self.metrics["successful_predictions"] /
                    total_predictions
                )
            
            # Sort by expected profit
            return sorted(
                validated_paths,
                key=lambda x: x["expected_profit"],
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Error validating paths: {str(e)}")
            return []

    async def update_model(self, path: Dict, success: bool):
        """Update model with execution results."""
        try:
            # Add to historical data
            self.historical_paths.append({
                **path,
                "success": success,
                "timestamp": datetime.utcnow()
            })
            
            if success:
                self.successful_paths.add(
                    "-".join(t.address for t in path["path"])
                )
            
            # Retrain model periodically
            if len(self.historical_paths) % 100 == 0:  # Every 100 paths
                await self._retrain_model()
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")

    async def _retrain_model(self):
        """Retrain model with updated historical data."""
        try:
            if len(self.historical_paths) < 100:
                return  # Need more data
            
            # Prepare training data
            X = []
            y = []
            
            for path_data in self.historical_paths[-1000:]:  # Last 1000 paths
                X.append(path_data["features"].flatten())
                y.append(1 if path_data["success"] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Update scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train new model
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            new_model.fit(X_scaled, y)
            
            # Evaluate new model
            accuracy = new_model.score(X_scaled, y)
            
            # Update if better
            if accuracy > self.metrics["model_accuracy"]:
                self.model = new_model
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, f"{self.model_path}.scaler")
                self.logger.info(f"Model updated with accuracy: {accuracy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "historical_paths": len(self.historical_paths),
            "successful_paths": len(self.successful_paths),
            "last_update": datetime.utcnow().isoformat()
        }
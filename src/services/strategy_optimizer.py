from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from dataclasses import dataclass

@dataclass
class StrategyParameters:
    """Trading strategy parameters."""
    min_profit_threshold: float
    max_position_size: float
    max_slippage: float
    min_liquidity: float
    gas_multiplier: float
    execution_timeout: int
    confidence_threshold: float

@dataclass
class OpportunityFeatures:
    """Features for opportunity analysis."""
    price_features: np.ndarray
    liquidity_features: np.ndarray
    volume_features: np.ndarray
    gas_features: np.ndarray
    path_features: np.ndarray

class LSTMPredictor(nn.Module):
    """LSTM model for sequence prediction."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        ).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

class OpportunityDataset(Dataset):
    """Dataset for opportunity features."""
    def __init__(
        self,
        features: List[OpportunityFeatures],
        labels: np.ndarray
    ):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # Combine features
        combined = np.concatenate([
            feature.price_features,
            feature.liquidity_features,
            feature.volume_features,
            feature.gas_features,
            feature.path_features
        ])
        
        return torch.FloatTensor(combined), torch.FloatTensor([label])

class StrategyOptimizer:
    """ML-based strategy optimization."""
    def __init__(
        self,
        web3,
        market_data_aggregator,
        risk_manager,
        settings: Dict
    ):
        self.web3 = web3
        self.market_data = market_data_aggregator
        self.risk_manager = risk_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.strategy_params = StrategyParameters(
            min_profit_threshold=100.0,
            max_position_size=50000.0,
            max_slippage=0.01,
            min_liquidity=100000.0,
            gas_multiplier=1.1,
            execution_timeout=30,
            confidence_threshold=0.7
        )
        
        # ML models
        self.models = {
            "opportunity_classifier": None,
            "profit_predictor": None,
            "sequence_predictor": None
        }
        
        # Training data
        self.training_data = {
            "features": [],
            "labels": [],
            "timestamps": []
        }
        
        # Performance metrics
        self.metrics = {
            "predictions_made": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "avg_prediction_error": 0.0
        }
        
        # Initialize models
        self._init_models()

    def _init_models(self):
        """Initialize ML models."""
        try:
            # Load or create opportunity classifier
            try:
                self.models["opportunity_classifier"] = joblib.load(
                    "models/opportunity_classifier.joblib"
                )
                self.opportunity_scaler = joblib.load(
                    "models/opportunity_scaler.joblib"
                )
            except:
                self.models["opportunity_classifier"] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.opportunity_scaler = StandardScaler()
            
            # Load or create profit predictor
            try:
                self.models["profit_predictor"] = joblib.load(
                    "models/profit_predictor.joblib"
                )
                self.profit_scaler = joblib.load(
                    "models/profit_scaler.joblib"
                )
            except:
                self.models["profit_predictor"] = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                self.profit_scaler = StandardScaler()
            
            # Load or create sequence predictor
            try:
                self.models["sequence_predictor"] = torch.load(
                    "models/sequence_predictor.pt"
                )
            except:
                self.models["sequence_predictor"] = LSTMPredictor(
                    input_size=50,  # Combined feature size
                    hidden_size=64,
                    num_layers=2,
                    output_size=1
                )
            
            self.logger.info("ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    async def analyze_opportunity(
        self,
        opportunity: Dict,
        context: Dict
    ) -> Dict:
        """Analyze arbitrage opportunity using ML models."""
        try:
            # Extract features
            features = await self._extract_features(opportunity)
            
            # Classify opportunity
            classification = await self._classify_opportunity(features)
            
            if not classification["is_valid"]:
                return {
                    "valid": False,
                    "reason": classification["reason"]
                }
            
            # Predict profit
            profit_prediction = await self._predict_profit(features)
            
            # Predict sequence success
            sequence_prediction = await self._predict_sequence(features)
            
            # Combine predictions
            prediction = self._combine_predictions(
                classification,
                profit_prediction,
                sequence_prediction
            )
            
            # Validate against strategy parameters
            validation = await self._validate_prediction(
                prediction,
                opportunity
            )
            
            # Update metrics
            self.metrics["predictions_made"] += 1
            if validation["valid"]:
                self.metrics["successful_predictions"] += 1
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunity: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def _extract_features(
        self,
        opportunity: Dict
    ) -> OpportunityFeatures:
        """Extract features for ML models."""
        try:
            # Extract price features
            price_features = await self._extract_price_features(
                opportunity["path"]
            )
            
            # Extract liquidity features
            liquidity_features = await self._extract_liquidity_features(
                opportunity["path"]
            )
            
            # Extract volume features
            volume_features = await self._extract_volume_features(
                opportunity["path"]
            )
            
            # Extract gas features
            gas_features = self._extract_gas_features(opportunity)
            
            # Extract path features
            path_features = self._extract_path_features(opportunity)
            
            return OpportunityFeatures(
                price_features=price_features,
                liquidity_features=liquidity_features,
                volume_features=volume_features,
                gas_features=gas_features,
                path_features=path_features
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None

    async def _extract_price_features(
        self,
        path: List[Dict]
    ) -> np.ndarray:
        """Extract price-related features."""
        try:
            features = []
            
            for step in path:
                # Get price history
                prices = await self.market_data.get_price_history(
                    step["token_in"]
                )
                
                if prices is not None:
                    # Calculate returns
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Calculate features
                    features.extend([
                        np.mean(returns),
                        np.std(returns),
                        np.percentile(returns, 10),
                        np.percentile(returns, 90),
                        np.mean(prices[-10:]) / np.mean(prices[-30:]) - 1
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting price features: {str(e)}")
            return np.zeros(5 * len(path))

    async def _extract_liquidity_features(
        self,
        path: List[Dict]
    ) -> np.ndarray:
        """Extract liquidity-related features."""
        try:
            features = []
            
            for step in path:
                # Get pool liquidity
                liquidity = await self.market_data.get_pool_liquidity(
                    step["pool"]
                )
                
                if liquidity:
                    # Calculate features
                    features.extend([
                        liquidity,
                        liquidity / self.strategy_params.min_liquidity,
                        step["amount_in"] / liquidity if liquidity > 0 else 1
                    ])
                else:
                    features.extend([0, 0, 1])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting liquidity features: {str(e)}")
            return np.zeros(3 * len(path))

    async def _extract_volume_features(
        self,
        path: List[Dict]
    ) -> np.ndarray:
        """Extract volume-related features."""
        try:
            features = []
            
            for step in path:
                # Get volume history
                volumes = await self.market_data.get_volume_history(
                    step["pool"]
                )
                
                if volumes is not None:
                    # Calculate features
                    features.extend([
                        np.mean(volumes),
                        np.std(volumes),
                        volumes[-1] / np.mean(volumes) if len(volumes) > 0 else 0
                    ])
                else:
                    features.extend([0, 0, 0])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting volume features: {str(e)}")
            return np.zeros(3 * len(path))

    def _extract_gas_features(
        self,
        opportunity: Dict
    ) -> np.ndarray:
        """Extract gas-related features."""
        try:
            return np.array([
                opportunity["gas_price"],
                opportunity["gas_cost"],
                opportunity["gas_cost"] / opportunity["expected_profit"]
            ])
            
        except Exception as e:
            self.logger.error(f"Error extracting gas features: {str(e)}")
            return np.zeros(3)

    def _extract_path_features(
        self,
        opportunity: Dict
    ) -> np.ndarray:
        """Extract path-related features."""
        try:
            return np.array([
                len(opportunity["path"]),
                opportunity["amount_in_usd"],
                opportunity["expected_profit"],
                opportunity["total_fee"]
            ])
            
        except Exception as e:
            self.logger.error(f"Error extracting path features: {str(e)}")
            return np.zeros(4)

    async def _classify_opportunity(
        self,
        features: OpportunityFeatures
    ) -> Dict:
        """Classify arbitrage opportunity."""
        try:
            # Combine features
            combined = np.concatenate([
                features.price_features,
                features.liquidity_features,
                features.volume_features,
                features.gas_features,
                features.path_features
            ])
            
            # Scale features
            scaled = self.opportunity_scaler.transform(
                combined.reshape(1, -1)
            )
            
            # Make prediction
            probability = self.models["opportunity_classifier"].predict_proba(
                scaled
            )[0][1]
            
            # Validate prediction
            is_valid = probability >= self.strategy_params.confidence_threshold
            
            return {
                "is_valid": is_valid,
                "probability": probability,
                "reason": self._get_invalid_reason(features)
                if not is_valid else None
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying opportunity: {str(e)}")
            return {"is_valid": False, "probability": 0.0, "reason": str(e)}

    def _get_invalid_reason(
        self,
        features: OpportunityFeatures
    ) -> str:
        """Get reason for invalid classification."""
        try:
            # Check liquidity
            if np.min(features.liquidity_features) < self.strategy_params.min_liquidity:
                return "Insufficient liquidity"
            
            # Check gas cost
            if features.gas_features[2] > self.strategy_params.gas_multiplier:
                return "Gas cost too high"
            
            # Check path complexity
            if features.path_features[0] > 4:
                return "Path too complex"
            
            return "Low confidence prediction"
            
        except Exception as e:
            self.logger.error(f"Error getting invalid reason: {str(e)}")
            return "Unknown reason"

    async def _predict_profit(
        self,
        features: OpportunityFeatures
    ) -> Dict:
        """Predict arbitrage profit."""
        try:
            # Combine features
            combined = np.concatenate([
                features.price_features,
                features.liquidity_features,
                features.volume_features,
                features.gas_features,
                features.path_features
            ])
            
            # Scale features
            scaled = self.profit_scaler.transform(
                combined.reshape(1, -1)
            )
            
            # Make prediction
            prediction = self.models["profit_predictor"].predict(scaled)[0]
            
            # Calculate confidence interval
            std = np.std([
                tree.predict(scaled)[0]
                for tree in self.models["profit_predictor"].estimators_
            ])
            
            return {
                "predicted_profit": prediction,
                "confidence_interval": (prediction - 2*std, prediction + 2*std),
                "std": std
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting profit: {str(e)}")
            return {"predicted_profit": 0.0, "confidence_interval": (0.0, 0.0), "std": 0.0}

    async def _predict_sequence(
        self,
        features: OpportunityFeatures
    ) -> Dict:
        """Predict sequence success probability."""
        try:
            # Prepare sequence data
            sequence = self._prepare_sequence(features)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.models["sequence_predictor"](
                    sequence
                ).item()
            
            return {
                "success_probability": prediction,
                "recommended_timeout": self._calculate_timeout(prediction)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting sequence: {str(e)}")
            return {"success_probability": 0.0, "recommended_timeout": 30}

    def _prepare_sequence(
        self,
        features: OpportunityFeatures
    ) -> torch.Tensor:
        """Prepare sequence data for LSTM."""
        try:
            # Combine features
            combined = np.concatenate([
                features.price_features,
                features.liquidity_features,
                features.volume_features,
                features.gas_features,
                features.path_features
            ])
            
            # Reshape for LSTM
            sequence = combined.reshape(1, -1, 1)
            
            return torch.FloatTensor(sequence)
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence: {str(e)}")
            return torch.zeros(1, 50, 1)

    def _calculate_timeout(
        self,
        success_probability: float
    ) -> int:
        """Calculate recommended execution timeout."""
        try:
            # Base timeout
            base_timeout = self.strategy_params.execution_timeout
            
            # Adjust based on probability
            if success_probability > 0.9:
                return int(base_timeout * 0.8)  # Faster for high probability
            elif success_probability < 0.7:
                return int(base_timeout * 1.2)  # Slower for low probability
            
            return base_timeout
            
        except Exception as e:
            self.logger.error(f"Error calculating timeout: {str(e)}")
            return self.strategy_params.execution_timeout

    def _combine_predictions(
        self,
        classification: Dict,
        profit_prediction: Dict,
        sequence_prediction: Dict
    ) -> Dict:
        """Combine all predictions."""
        try:
            return {
                "valid": classification["is_valid"],
                "confidence": classification["probability"],
                "predicted_profit": profit_prediction["predicted_profit"],
                "profit_confidence": (
                    profit_prediction["confidence_interval"][1] -
                    profit_prediction["confidence_interval"][0]
                ) / profit_prediction["predicted_profit"],
                "success_probability": sequence_prediction["success_probability"],
                "recommended_timeout": sequence_prediction["recommended_timeout"]
            }
            
        except Exception as e:
            self.logger.error(f"Error combining predictions: {str(e)}")
            return {
                "valid": False,
                "confidence": 0.0,
                "predicted_profit": 0.0,
                "profit_confidence": 0.0,
                "success_probability": 0.0,
                "recommended_timeout": self.strategy_params.execution_timeout
            }

    async def _validate_prediction(
        self,
        prediction: Dict,
        opportunity: Dict
    ) -> Dict:
        """Validate prediction against strategy parameters."""
        try:
            validation = {
                "valid": False,
                "checks": {},
                "prediction": prediction
            }
            
            # Check profit threshold
            profit_check = (
                prediction["predicted_profit"] >=
                self.strategy_params.min_profit_threshold
            )
            validation["checks"]["profit"] = profit_check
            
            # Check position size
            size_check = (
                opportunity["amount_in_usd"] <=
                self.strategy_params.max_position_size
            )
            validation["checks"]["size"] = size_check
            
            # Check confidence
            confidence_check = (
                prediction["confidence"] >=
                self.strategy_params.confidence_threshold
            )
            validation["checks"]["confidence"] = confidence_check
            
            # Check success probability
            probability_check = prediction["success_probability"] >= 0.7
            validation["checks"]["probability"] = probability_check
            
            # Overall validation
            validation["valid"] = all([
                profit_check,
                size_check,
                confidence_check,
                probability_check
            ])
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def update_models(
        self,
        opportunity: Dict,
        result: Dict
    ):
        """Update ML models with execution results."""
        try:
            # Extract features
            features = await self._extract_features(opportunity)
            
            if not features:
                return
            
            # Update training data
            self.training_data["features"].append(features)
            self.training_data["labels"].append(
                1 if result["success"] else 0
            )
            self.training_data["timestamps"].append(
                datetime.utcnow()
            )
            
            # Update metrics
            if result["success"]:
                self.metrics["successful_predictions"] += 1
            else:
                if opportunity["expected_profit"] > 0:
                    self.metrics["false_positives"] += 1
                else:
                    self.metrics["false_negatives"] += 1
            
            # Retrain models periodically
            if len(self.training_data["features"]) % 100 == 0:
                await self._retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    async def _retrain_models(self):
        """Retrain ML models."""
        try:
            # Prepare training data
            X = np.array([
                np.concatenate([
                    f.price_features,
                    f.liquidity_features,
                    f.volume_features,
                    f.gas_features,
                    f.path_features
                ])
                for f in self.training_data["features"]
            ])
            y = np.array(self.training_data["labels"])
            
            # Create time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Retrain opportunity classifier
            best_classifier_score = self.models["opportunity_classifier"].score(
                X,
                y
            )
            
            new_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                new_classifier.fit(X_train, y_train)
                score = new_classifier.score(X_test, y_test)
                
                if score > best_classifier_score:
                    self.models["opportunity_classifier"] = new_classifier
                    joblib.dump(
                        new_classifier,
                        "models/opportunity_classifier.joblib"
                    )
            
            # Retrain profit predictor
            best_predictor_score = self.models["profit_predictor"].score(X, y)
            
            new_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                new_predictor.fit(X_train, y_train)
                score = new_predictor.score(X_test, y_test)
                
                if score > best_predictor_score:
                    self.models["profit_predictor"] = new_predictor
                    joblib.dump(
                        new_predictor,
                        "models/profit_predictor.joblib"
                    )
            
            # Retrain sequence predictor
            dataset = OpportunityDataset(
                self.training_data["features"],
                self.training_data["labels"]
            )
            
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True
            )
            
            new_sequence_predictor = await self._train_lstm(
                loader,
                epochs=50
            )
            
            if new_sequence_predictor:
                self.models["sequence_predictor"] = new_sequence_predictor
                torch.save(
                    new_sequence_predictor,
                    "models/sequence_predictor.pt"
                )
            
            self.logger.info("Models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")

    async def _train_lstm(
        self,
        loader: DataLoader,
        epochs: int
    ) -> Optional[LSTMPredictor]:
        """Train LSTM model."""
        try:
            model = LSTMPredictor(
                input_size=50,
                hidden_size=64,
                num_layers=2,
                output_size=1
            )
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            best_loss = float("inf")
            best_model = None
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(loader)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model = model.state_dict()
            
            if best_model:
                model.load_state_dict(best_model)
                return model
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error training LSTM: {str(e)}")
            return None

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "accuracy": (
                self.metrics["successful_predictions"] /
                max(1, self.metrics["predictions_made"])
            ),
            "false_positive_rate": (
                self.metrics["false_positives"] /
                max(1, self.metrics["predictions_made"])
            ),
            "false_negative_rate": (
                self.metrics["false_negatives"] /
                max(1, self.metrics["predictions_made"])
            ),
            "training_samples": len(self.training_data["features"]),
            "last_update": datetime.utcnow().isoformat()
        }
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
import asyncio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PricePredictor:
    def __init__(
        self,
        market_data_aggregator,
        model_path: str = "models/price_predictor.joblib",
        window_size: int = 100,
        prediction_horizon: int = 10
    ):
        self.market_data = market_data_aggregator
        self.model_path = model_path
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.gb_model: Optional[GradientBoostingRegressor] = None
        self.lstm_model: Optional[PriceLSTM] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Price history
        self.price_history: Dict[str, List[float]] = {}
        self.timestamp_history: Dict[str, List[datetime]] = {}
        
        # Performance metrics
        self.metrics = {
            "predictions_made": 0,
            "mse": 0.0,
            "mae": 0.0,
            "correct_direction": 0,
            "avg_prediction_time_ms": 0.0
        }
        
        # Initialize models
        self._load_or_train_models()

    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        try:
            # Try to load existing models
            self.gb_model = joblib.load(f"{self.model_path}_gb")
            self.lstm_model = torch.load(f"{self.model_path}_lstm")
            self.scaler = joblib.load(f"{self.model_path}_scaler")
            self.logger.info("Loaded existing price prediction models")
        except:
            self.logger.info("Training new price prediction models")
            self._train_initial_models()

    def _train_initial_models(self):
        """Train initial models with basic data."""
        try:
            # Create simple initial training data
            X = np.random.rand(1000, self.window_size)
            y = np.mean(X, axis=1) + np.random.randn(1000) * 0.1
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Gradient Boosting model
            self.gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.gb_model.fit(X_scaled, y)
            
            # Train LSTM model
            self.lstm_model = PriceLSTM(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                output_size=self.prediction_horizon
            )
            
            # Convert data for LSTM
            X_lstm = torch.FloatTensor(X_scaled).unsqueeze(-1)
            y_lstm = torch.FloatTensor(y)
            
            # Train LSTM
            self._train_lstm(X_lstm, y_lstm)
            
            # Save models
            joblib.dump(self.gb_model, f"{self.model_path}_gb")
            torch.save(self.lstm_model, f"{self.model_path}_lstm")
            joblib.dump(self.scaler, f"{self.model_path}_scaler")
            
        except Exception as e:
            self.logger.error(f"Error training initial models: {str(e)}")
            raise

    def _train_lstm(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """Train LSTM model."""
        try:
            # Create data loader
            dataset = PriceDataset(X, y)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                self.lstm_model.parameters(),
                lr=0.001
            )
            
            # Training loop
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in loader:
                    # Forward pass
                    outputs = self.lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch [{epoch+1}/{epochs}], "
                        f"Loss: {total_loss/len(loader):.4f}"
                    )
            
        except Exception as e:
            self.logger.error(f"Error training LSTM: {str(e)}")

    async def predict_price(
        self,
        token_address: str,
        time_horizon: int = None
    ) -> Optional[Dict]:
        """Predict future token price."""
        try:
            start_time = datetime.utcnow()
            
            # Get price history
            await self._update_price_history(token_address)
            
            if not self.price_history.get(token_address):
                return None
            
            # Prepare features
            features = self._prepare_features(token_address)
            if features is None:
                return None
            
            # Make predictions with both models
            gb_pred = self._predict_gb(features)
            lstm_pred = self._predict_lstm(features)
            
            # Combine predictions
            prediction = self._combine_predictions(gb_pred, lstm_pred)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                prediction,
                gb_pred,
                lstm_pred
            )
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics["avg_prediction_time_ms"] = (
                self.metrics["avg_prediction_time_ms"] * self.metrics["predictions_made"] +
                execution_time
            ) / (self.metrics["predictions_made"] + 1)
            self.metrics["predictions_made"] += 1
            
            return {
                "price": prediction,
                "confidence": confidence,
                "timestamp": datetime.utcnow(),
                "horizon": time_horizon or self.prediction_horizon
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting price: {str(e)}")
            return None

    async def _update_price_history(self, token_address: str):
        """Update price history for token."""
        try:
            # Get current price
            current_price = await self.market_data.get_token_price(token_address)
            if not current_price:
                return
            
            # Initialize history if needed
            if token_address not in self.price_history:
                self.price_history[token_address] = []
                self.timestamp_history[token_address] = []
            
            # Add new price
            self.price_history[token_address].append(current_price)
            self.timestamp_history[token_address].append(datetime.utcnow())
            
            # Keep fixed window size
            if len(self.price_history[token_address]) > self.window_size:
                self.price_history[token_address] = (
                    self.price_history[token_address][-self.window_size:]
                )
                self.timestamp_history[token_address] = (
                    self.timestamp_history[token_address][-self.window_size:]
                )
            
        except Exception as e:
            self.logger.error(f"Error updating price history: {str(e)}")

    def _prepare_features(self, token_address: str) -> Optional[np.ndarray]:
        """Prepare features for prediction."""
        try:
            if len(self.price_history[token_address]) < self.window_size:
                return None
            
            # Get price history
            prices = np.array(self.price_history[token_address])
            
            # Calculate technical indicators
            features = []
            
            # Price changes
            price_changes = np.diff(prices) / prices[:-1]
            features.extend([
                np.mean(price_changes),
                np.std(price_changes),
                np.min(price_changes),
                np.max(price_changes)
            ])
            
            # Moving averages
            ma_5 = np.mean(prices[-5:])
            ma_20 = np.mean(prices[-20:])
            features.extend([
                ma_5/prices[-1] - 1,
                ma_20/prices[-1] - 1
            ])
            
            # Volatility
            volatility = np.std(price_changes) * np.sqrt(252)
            features.append(volatility)
            
            # Momentum
            momentum = prices[-1] / prices[-10] - 1
            features.append(momentum)
            
            # Scale features
            features = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            return features_scaled
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None

    def _predict_gb(self, features: np.ndarray) -> np.ndarray:
        """Make prediction using Gradient Boosting model."""
        try:
            return self.gb_model.predict(features)
        except Exception as e:
            self.logger.error(f"Error in GB prediction: {str(e)}")
            return np.zeros(self.prediction_horizon)

    def _predict_lstm(self, features: np.ndarray) -> np.ndarray:
        """Make prediction using LSTM model."""
        try:
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.lstm_model(features_tensor)
            
            return prediction.numpy()
            
        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {str(e)}")
            return np.zeros(self.prediction_horizon)

    def _combine_predictions(
        self,
        gb_pred: np.ndarray,
        lstm_pred: np.ndarray
    ) -> float:
        """Combine predictions from both models."""
        try:
            # Use weighted average
            weights = [0.4, 0.6]  # Give more weight to LSTM
            combined = weights[0] * gb_pred + weights[1] * lstm_pred
            return float(np.mean(combined))
            
        except Exception as e:
            self.logger.error(f"Error combining predictions: {str(e)}")
            return 0.0

    def _calculate_confidence(
        self,
        prediction: float,
        gb_pred: np.ndarray,
        lstm_pred: np.ndarray
    ) -> float:
        """Calculate confidence score for prediction."""
        try:
            # Calculate agreement between models
            diff = abs(np.mean(gb_pred) - np.mean(lstm_pred))
            max_val = max(abs(np.mean(gb_pred)), abs(np.mean(lstm_pred)))
            
            if max_val == 0:
                return 0.5
            
            agreement = 1 - (diff / max_val)
            
            # Calculate prediction stability
            gb_std = np.std(gb_pred)
            lstm_std = np.std(lstm_pred)
            stability = 1 - min(1, (gb_std + lstm_std) / 2)
            
            # Combine metrics
            confidence = (agreement * 0.6 + stability * 0.4)
            return min(1, max(0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    async def update_models(self, token_address: str, actual_price: float):
        """Update models with actual price data."""
        try:
            # Get previous prediction
            prev_predictions = [
                p for p in self.price_history.get(token_address, [])
                if p is not None
            ]
            
            if not prev_predictions:
                return
            
            # Calculate error
            last_prediction = prev_predictions[-1]
            error = actual_price - last_prediction
            
            # Update metrics
            self.metrics["mse"] = (
                self.metrics["mse"] * self.metrics["predictions_made"] +
                error ** 2
            ) / (self.metrics["predictions_made"] + 1)
            
            self.metrics["mae"] = (
                self.metrics["mae"] * self.metrics["predictions_made"] +
                abs(error)
            ) / (self.metrics["predictions_made"] + 1)
            
            # Update direction accuracy
            if len(prev_predictions) >= 2:
                pred_direction = last_prediction > prev_predictions[-2]
                actual_direction = actual_price > prev_predictions[-2]
                
                if pred_direction == actual_direction:
                    self.metrics["correct_direction"] += 1
            
            # Retrain models periodically
            if self.metrics["predictions_made"] % 1000 == 0:
                await self._retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")

    async def _retrain_models(self):
        """Retrain models with updated data."""
        try:
            # Collect training data from all tokens
            X = []
            y = []
            
            for token_address in self.price_history:
                if len(self.price_history[token_address]) >= self.window_size:
                    features = self._prepare_features(token_address)
                    if features is not None:
                        X.append(features)
                        y.append(self.price_history[token_address][-1])
            
            if not X:
                return
            
            # Convert to arrays
            X = np.vstack(X)
            y = np.array(y)
            
            # Retrain GB model
            self.gb_model.fit(X, y)
            
            # Retrain LSTM model
            X_lstm = torch.FloatTensor(X).unsqueeze(-1)
            y_lstm = torch.FloatTensor(y)
            self._train_lstm(X_lstm, y_lstm, epochs=50)
            
            # Save updated models
            joblib.dump(self.gb_model, f"{self.model_path}_gb")
            torch.save(self.lstm_model, f"{self.model_path}_lstm")
            
            self.logger.info("Models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        return {
            **self.metrics,
            "tokens_tracked": len(self.price_history),
            "last_update": datetime.utcnow().isoformat()
        }


class PriceLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
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


class PriceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
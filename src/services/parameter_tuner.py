from typing import Dict, List, Optional, Tuple, Set
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

@dataclass
class ParameterSpace:
    """Parameter space for optimization."""
    name: str
    min_value: float
    max_value: float
    current_value: float
    step_size: float
    importance: float  # 0 to 1

@dataclass
class ParameterSet:
    """Set of parameters with performance metrics."""
    parameters: Dict[str, float]
    performance: float
    timestamp: datetime
    context: Dict

class ParameterTuner:
    """Dynamic parameter optimization system."""
    def __init__(
        self,
        strategy_optimizer,
        risk_manager,
        settings: Dict
    ):
        self.strategy = strategy_optimizer
        self.risk = risk_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Parameter spaces
        self.parameter_spaces = {
            "profit_threshold": ParameterSpace(
                name="profit_threshold",
                min_value=50.0,
                max_value=500.0,
                current_value=100.0,
                step_size=10.0,
                importance=0.8
            ),
            "position_size": ParameterSpace(
                name="position_size",
                min_value=1000.0,
                max_value=100000.0,
                current_value=50000.0,
                step_size=1000.0,
                importance=0.9
            ),
            "slippage_tolerance": ParameterSpace(
                name="slippage_tolerance",
                min_value=0.001,
                max_value=0.02,
                current_value=0.01,
                step_size=0.001,
                importance=0.7
            ),
            "gas_multiplier": ParameterSpace(
                name="gas_multiplier",
                min_value=1.0,
                max_value=2.0,
                current_value=1.1,
                step_size=0.05,
                importance=0.6
            ),
            "execution_timeout": ParameterSpace(
                name="execution_timeout",
                min_value=10,
                max_value=60,
                current_value=30,
                step_size=5,
                importance=0.5
            )
        }
        
        # Performance history
        self.performance_history: List[ParameterSet] = []
        self.optimization_history: List[Dict] = []
        
        # Optimization settings
        self.optimization_settings = {
            "window_size": 100,  # Number of samples to consider
            "min_samples": 20,  # Minimum samples before optimization
            "update_interval": 60,  # Seconds between updates
            "exploration_rate": 0.2,  # Probability of random exploration
            "performance_threshold": 0.8  # Minimum performance improvement required
        }
        
        # Initialize models
        self._init_models()
        
        # Load historical data
        self._load_history()

    def _init_models(self):
        """Initialize optimization models."""
        try:
            # Gaussian Process for parameter optimization
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0] * len(self.parameter_spaces))
            
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                random_state=42
            )
            
            self.logger.info("Optimization models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _load_history(self):
        """Load historical performance data."""
        try:
            try:
                with open("data/parameter_history.json", "r") as f:
                    data = json.load(f)
                    
                    for entry in data:
                        self.performance_history.append(
                            ParameterSet(
                                parameters=entry["parameters"],
                                performance=entry["performance"],
                                timestamp=datetime.fromisoformat(entry["timestamp"]),
                                context=entry["context"]
                            )
                        )
                    
                    self.logger.info(f"Loaded {len(data)} historical entries")
            except FileNotFoundError:
                self.logger.info("No historical data found")
            
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")

    async def get_optimal_parameters(
        self,
        context: Dict
    ) -> Dict[str, float]:
        """Get optimal parameters for current context."""
        try:
            # Check if update needed
            if not self._should_update(context):
                return self._get_current_parameters()
            
            # Get relevant historical data
            relevant_history = self._get_relevant_history(context)
            
            if len(relevant_history) < self.optimization_settings["min_samples"]:
                return self._get_exploration_parameters()
            
            # Optimize parameters
            optimal_params = await self._optimize_parameters(
                relevant_history,
                context
            )
            
            # Validate parameters
            validated_params = self._validate_parameters(optimal_params)
            
            # Update current values
            self._update_current_values(validated_params)
            
            return validated_params
            
        except Exception as e:
            self.logger.error(f"Error getting optimal parameters: {str(e)}")
            return self._get_current_parameters()

    def _should_update(self, context: Dict) -> bool:
        """Check if parameters should be updated."""
        try:
            # Check last update time
            if not self.optimization_history:
                return True
            
            last_update = self.optimization_history[-1]["timestamp"]
            time_since_update = (datetime.utcnow() - last_update).total_seconds()
            
            if time_since_update < self.optimization_settings["update_interval"]:
                return False
            
            # Check context changes
            if self.optimization_history:
                last_context = self.optimization_history[-1]["context"]
                context_change = self._calculate_context_change(
                    last_context,
                    context
                )
                
                if context_change > 0.2:  # 20% change threshold
                    return True
            
            # Check performance
            recent_performance = self._get_recent_performance()
            if recent_performance < self.optimization_settings["performance_threshold"]:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking update: {str(e)}")
            return False

    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values."""
        return {
            name: space.current_value
            for name, space in self.parameter_spaces.items()
        }

    def _get_exploration_parameters(self) -> Dict[str, float]:
        """Get parameters for exploration."""
        try:
            params = {}
            
            for name, space in self.parameter_spaces.items():
                if np.random.random() < self.optimization_settings["exploration_rate"]:
                    # Random exploration
                    params[name] = np.random.uniform(
                        space.min_value,
                        space.max_value
                    )
                else:
                    # Use current value
                    params[name] = space.current_value
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error getting exploration parameters: {str(e)}")
            return self._get_current_parameters()

    def _get_relevant_history(
        self,
        context: Dict
    ) -> List[ParameterSet]:
        """Get relevant historical data for current context."""
        try:
            relevant = []
            
            for entry in self.performance_history[-self.optimization_settings["window_size"]:]:
                similarity = self._calculate_context_similarity(
                    entry.context,
                    context
                )
                
                if similarity > 0.7:  # 70% similarity threshold
                    relevant.append(entry)
            
            return relevant
            
        except Exception as e:
            self.logger.error(f"Error getting relevant history: {str(e)}")
            return []

    async def _optimize_parameters(
        self,
        history: List[ParameterSet],
        context: Dict
    ) -> Dict[str, float]:
        """Optimize parameters using Gaussian Process."""
        try:
            # Prepare training data
            X = np.array([
                [entry.parameters[p] for p in self.parameter_spaces.keys()]
                for entry in history
            ])
            
            y = np.array([entry.performance for entry in history])
            
            # Fit GP model
            self.gp_model.fit(X, y)
            
            # Define objective function
            def objective(x):
                x_reshaped = x.reshape(1, -1)
                mean, std = self.gp_model.predict(x_reshaped, return_std=True)
                return -(mean[0] + 0.1 * std[0])  # Maximize mean and explore uncertainty
            
            # Define bounds
            bounds = [
                (space.min_value, space.max_value)
                for space in self.parameter_spaces.values()
            ]
            
            # Optimize
            result = minimize(
                objective,
                x0=np.array([space.current_value for space in self.parameter_spaces.values()]),
                bounds=bounds,
                method="L-BFGS-B"
            )
            
            # Convert result to parameters
            optimal_params = {
                name: value
                for name, value in zip(self.parameter_spaces.keys(), result.x)
            }
            
            return optimal_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {str(e)}")
            return self._get_current_parameters()

    def _validate_parameters(
        self,
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """Validate and adjust parameters."""
        try:
            validated = {}
            
            for name, value in parameters.items():
                space = self.parameter_spaces[name]
                
                # Clamp to bounds
                validated[name] = max(
                    space.min_value,
                    min(value, space.max_value)
                )
                
                # Round to step size
                if space.step_size > 0:
                    validated[name] = round(
                        validated[name] / space.step_size
                    ) * space.step_size
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return self._get_current_parameters()

    def _update_current_values(self, parameters: Dict[str, float]):
        """Update current parameter values."""
        try:
            for name, value in parameters.items():
                if name in self.parameter_spaces:
                    self.parameter_spaces[name].current_value = value
            
            # Save optimization history
            self.optimization_history.append({
                "parameters": parameters,
                "timestamp": datetime.utcnow(),
                "context": self._get_current_context()
            })
            
            # Save to file periodically
            if len(self.optimization_history) % 10 == 0:
                self._save_history()
            
        except Exception as e:
            self.logger.error(f"Error updating parameters: {str(e)}")

    def _calculate_context_change(
        self,
        context1: Dict,
        context2: Dict
    ) -> float:
        """Calculate relative change between contexts."""
        try:
            changes = []
            
            for key in context1:
                if key in context2:
                    if isinstance(context1[key], (int, float)):
                        old_val = context1[key]
                        new_val = context2[key]
                        if old_val != 0:
                            change = abs(new_val - old_val) / abs(old_val)
                            changes.append(change)
            
            return np.mean(changes) if changes else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating context change: {str(e)}")
            return 1.0

    def _calculate_context_similarity(
        self,
        context1: Dict,
        context2: Dict
    ) -> float:
        """Calculate similarity between contexts."""
        try:
            similarities = []
            
            for key in context1:
                if key in context2:
                    if isinstance(context1[key], (int, float)):
                        # Numerical similarity
                        max_val = max(abs(context1[key]), abs(context2[key]))
                        if max_val > 0:
                            similarity = 1 - abs(
                                context1[key] - context2[key]
                            ) / max_val
                            similarities.append(similarity)
                    else:
                        # String equality
                        similarities.append(
                            1.0 if context1[key] == context2[key] else 0.0
                        )
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _get_recent_performance(self) -> float:
        """Get recent average performance."""
        try:
            if not self.performance_history:
                return 0.0
            
            recent = self.performance_history[-10:]  # Last 10 entries
            return np.mean([entry.performance for entry in recent])
            
        except Exception as e:
            self.logger.error(f"Error getting performance: {str(e)}")
            return 0.0

    def _get_current_context(self) -> Dict:
        """Get current market context."""
        try:
            return {
                "gas_price": self.strategy.current_gas_price,
                "network_load": self.strategy.network_load,
                "volatility": self.strategy.market_volatility,
                "liquidity": self.strategy.average_liquidity,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting context: {str(e)}")
            return {"timestamp": datetime.utcnow().isoformat()}

    async def update_performance(
        self,
        parameters: Dict[str, float],
        performance: float,
        context: Dict
    ):
        """Update performance history."""
        try:
            # Add to history
            self.performance_history.append(
                ParameterSet(
                    parameters=parameters,
                    performance=performance,
                    timestamp=datetime.utcnow(),
                    context=context
                )
            )
            
            # Keep fixed window size
            if len(self.performance_history) > self.optimization_settings["window_size"]:
                self.performance_history = self.performance_history[
                    -self.optimization_settings["window_size"]:
                ]
            
            # Save periodically
            if len(self.performance_history) % 10 == 0:
                self._save_history()
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {str(e)}")

    def _save_history(self):
        """Save performance history to file."""
        try:
            data = [
                {
                    "parameters": entry.parameters,
                    "performance": entry.performance,
                    "timestamp": entry.timestamp.isoformat(),
                    "context": entry.context
                }
                for entry in self.performance_history
            ]
            
            with open("data/parameter_history.json", "w") as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get current metrics and statistics."""
        try:
            recent_history = self.performance_history[-100:]  # Last 100 entries
            
            if not recent_history:
                return {
                    "samples": 0,
                    "avg_performance": 0.0,
                    "parameters": self._get_current_parameters(),
                    "last_update": datetime.utcnow().isoformat()
                }
            
            performances = [entry.performance for entry in recent_history]
            
            return {
                "samples": len(self.performance_history),
                "avg_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "max_performance": max(performances),
                "min_performance": min(performances),
                "parameters": self._get_current_parameters(),
                "last_update": recent_history[-1].timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            return {
                "error": str(e),
                "last_update": datetime.utcnow().isoformat()
            }
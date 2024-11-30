from typing import Dict, Optional
import time
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server
)
import logging
from datetime import datetime

class MetricsService:
    def __init__(self, port: int = 8000):
        self.logger = logging.getLogger(__name__)
        self.port = port
        
        # Initialize metrics
        self._init_metrics()
        
        # Start Prometheus HTTP server
        start_http_server(port)
        self.logger.info(f"Metrics server started on port {port}")

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Opportunity metrics
        self.opportunities_found = Counter(
            'arbitrage_opportunities_found_total',
            'Total number of arbitrage opportunities found'
        )
        self.opportunities_validated = Counter(
            'arbitrage_opportunities_validated_total',
            'Total number of validated opportunities'
        )
        self.opportunities_executed = Counter(
            'arbitrage_opportunities_executed_total',
            'Total number of executed opportunities'
        )
        
        # Profit metrics
        self.total_profit = Counter(
            'arbitrage_profit_total_usd',
            'Total profit in USD'
        )
        self.profit_per_trade = Histogram(
            'arbitrage_profit_per_trade_usd',
            'Profit per trade in USD',
            buckets=[10, 50, 100, 500, 1000, 5000]
        )
        
        # Gas metrics
        self.gas_used = Histogram(
            'arbitrage_gas_used',
            'Gas used per transaction',
            buckets=[100000, 200000, 300000, 400000, 500000]
        )
        self.gas_price = Gauge(
            'arbitrage_gas_price_gwei',
            'Current gas price in Gwei'
        )
        
        # Performance metrics
        self.execution_time = Histogram(
            'arbitrage_execution_time_seconds',
            'Time taken to execute arbitrage',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.success_rate = Gauge(
            'arbitrage_success_rate',
            'Success rate of arbitrage executions'
        )
        
        # System metrics
        self.active_tasks = Gauge(
            'arbitrage_active_tasks',
            'Number of active arbitrage tasks'
        )
        self.memory_usage = Gauge(
            'arbitrage_memory_usage_bytes',
            'Memory usage in bytes'
        )
        self.cpu_usage = Gauge(
            'arbitrage_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Network metrics
        self.rpc_requests = Counter(
            'arbitrage_rpc_requests_total',
            'Total number of RPC requests',
            ['provider']
        )
        self.rpc_errors = Counter(
            'arbitrage_rpc_errors_total',
            'Total number of RPC errors',
            ['provider']
        )
        self.network_latency = Histogram(
            'arbitrage_network_latency_seconds',
            'Network request latency',
            ['provider'],
            buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        # Market metrics
        self.token_prices = Gauge(
            'arbitrage_token_price_usd',
            'Token price in USD',
            ['token_address']
        )
        self.pool_liquidity = Gauge(
            'arbitrage_pool_liquidity_usd',
            'Pool liquidity in USD',
            ['pool_address']
        )
        self.market_depth = Gauge(
            'arbitrage_market_depth_score',
            'Market depth score',
            ['token_pair']
        )

    def record_opportunity_found(self):
        """Record found opportunity."""
        self.opportunities_found.inc()

    def record_opportunity_validated(self):
        """Record validated opportunity."""
        self.opportunities_validated.inc()

    def record_opportunity_executed(self, profit: float, gas_used: int):
        """Record executed opportunity."""
        self.opportunities_executed.inc()
        self.total_profit.inc(profit)
        self.profit_per_trade.observe(profit)
        self.gas_used.observe(gas_used)

    def update_gas_price(self, price_gwei: float):
        """Update current gas price."""
        self.gas_price.set(price_gwei)

    def record_execution_time(self, seconds: float):
        """Record execution time."""
        self.execution_time.observe(seconds)

    def update_success_rate(self, rate: float):
        """Update success rate."""
        self.success_rate.set(rate)

    def update_active_tasks(self, count: int):
        """Update number of active tasks."""
        self.active_tasks.set(count)

    def update_system_metrics(self, memory_bytes: int, cpu_percent: float):
        """Update system metrics."""
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)

    def record_rpc_request(
        self,
        provider: str,
        success: bool,
        latency: float
    ):
        """Record RPC request metrics."""
        self.rpc_requests.labels(provider=provider).inc()
        if not success:
            self.rpc_errors.labels(provider=provider).inc()
        self.network_latency.labels(provider=provider).observe(latency)

    def update_token_price(self, token_address: str, price_usd: float):
        """Update token price."""
        self.token_prices.labels(token_address=token_address).set(price_usd)

    def update_pool_liquidity(self, pool_address: str, liquidity_usd: float):
        """Update pool liquidity."""
        self.pool_liquidity.labels(pool_address=pool_address).set(liquidity_usd)

    def update_market_depth(self, token_pair: str, depth_score: float):
        """Update market depth score."""
        self.market_depth.labels(token_pair=token_pair).set(depth_score)

    def record_batch_metrics(self, metrics: Dict):
        """Record multiple metrics at once."""
        try:
            # Update system metrics
            if "system" in metrics:
                self.update_system_metrics(
                    metrics["system"].get("memory_bytes", 0),
                    metrics["system"].get("cpu_percent", 0)
                )
            
            # Update market metrics
            if "market" in metrics:
                for token, price in metrics["market"].get("prices", {}).items():
                    self.update_token_price(token, price)
                
                for pool, liquidity in metrics["market"].get("liquidity", {}).items():
                    self.update_pool_liquidity(pool, liquidity)
            
            # Update performance metrics
            if "performance" in metrics:
                self.update_success_rate(
                    metrics["performance"].get("success_rate", 0)
                )
                self.update_active_tasks(
                    metrics["performance"].get("active_tasks", 0)
                )
            
            # Update network metrics
            if "network" in metrics:
                for provider, stats in metrics["network"].items():
                    self.record_rpc_request(
                        provider,
                        stats.get("success", True),
                        stats.get("latency", 0)
                    )
            
        except Exception as e:
            self.logger.error(f"Error recording batch metrics: {str(e)}")

    def get_current_metrics(self) -> Dict:
        """Get current metrics values."""
        return {
            "opportunities": {
                "found": self.opportunities_found._value.get(),
                "validated": self.opportunities_validated._value.get(),
                "executed": self.opportunities_executed._value.get()
            },
            "profit": {
                "total": self.total_profit._value.get(),
                "per_trade_avg": self.profit_per_trade.describe()["avg"]
            },
            "gas": {
                "price": self.gas_price._value.get(),
                "used_avg": self.gas_used.describe()["avg"]
            },
            "performance": {
                "success_rate": self.success_rate._value.get(),
                "execution_time_avg": self.execution_time.describe()["avg"],
                "active_tasks": self.active_tasks._value.get()
            },
            "system": {
                "memory_usage": self.memory_usage._value.get(),
                "cpu_usage": self.cpu_usage._value.get()
            }
        }
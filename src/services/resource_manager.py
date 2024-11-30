import psutil
import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

@dataclass
class ResourceLimits:
    max_cpu_percent: float = 70.0  # Maximum CPU usage percentage
    max_memory_percent: float = 80.0  # Maximum memory usage percentage
    cpu_cores: int = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    thread_pool_size: int = 4  # Default thread pool size
    batch_size: int = 100  # Number of operations to batch
    check_interval: float = 1.0  # Resource check interval in seconds

class ResourceManager:
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.logger = logging.getLogger(__name__)
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.limits.thread_pool_size,
            thread_name_prefix="arbitrage_worker"
        )
        
        # Resource monitoring
        self.monitoring = False
        self.monitoring_task = None
        
        # Performance metrics
        self.metrics: Dict = {
            "cpu_usage": [],
            "memory_usage": [],
            "active_threads": 0,
            "operations_per_second": 0
        }

    async def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitor_resources())
            self.logger.info("Resource monitoring started")

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Resource monitoring stopped")

    async def _monitor_resources(self):
        """Monitor system resources and adjust accordingly."""
        while self.monitoring:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Update metrics
                self.metrics["cpu_usage"].append(cpu_percent)
                self.metrics["memory_usage"].append(memory_percent)
                
                # Keep only recent history
                max_history = 60  # 1 minute of history
                self.metrics["cpu_usage"] = self.metrics["cpu_usage"][-max_history:]
                self.metrics["memory_usage"] = self.metrics["memory_usage"][-max_history:]
                
                # Adjust resources if needed
                await self._adjust_resources(cpu_percent, memory_percent)
                
                # Log resource usage if high
                if cpu_percent > self.limits.max_cpu_percent:
                    self.logger.warning(f"High CPU usage: {cpu_percent}%")
                if memory_percent > self.limits.max_memory_percent:
                    self.logger.warning(f"High memory usage: {memory_percent}%")
                
                await asyncio.sleep(self.limits.check_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
                await asyncio.sleep(self.limits.check_interval)

    async def _adjust_resources(self, cpu_percent: float, memory_percent: float):
        """Adjust resource usage based on current utilization."""
        try:
            # Calculate resource pressure (0 to 1)
            cpu_pressure = cpu_percent / 100
            memory_pressure = memory_percent / 100
            
            # Calculate overall system pressure
            system_pressure = max(cpu_pressure, memory_pressure)
            
            # Adjust thread pool size
            optimal_threads = self._calculate_optimal_threads(system_pressure)
            current_threads = self.thread_pool._max_workers
            
            if optimal_threads != current_threads:
                # Create new thread pool with adjusted size
                new_pool = ThreadPoolExecutor(
                    max_workers=optimal_threads,
                    thread_name_prefix="arbitrage_worker"
                )
                
                # Replace old pool
                old_pool = self.thread_pool
                self.thread_pool = new_pool
                
                # Shutdown old pool gracefully
                old_pool.shutdown(wait=False)
                
                self.logger.info(f"Adjusted thread pool size to {optimal_threads}")

            # Adjust batch size based on pressure
            self.limits.batch_size = self._calculate_optimal_batch_size(system_pressure)

        except Exception as e:
            self.logger.error(f"Error adjusting resources: {str(e)}")

    def _calculate_optimal_threads(self, system_pressure: float) -> int:
        """Calculate optimal number of threads based on system pressure."""
        # Base number of threads (leave one core free)
        base_threads = self.limits.cpu_cores
        
        # Reduce threads under high pressure
        if system_pressure > 0.8:
            return max(1, base_threads - 2)
        elif system_pressure > 0.6:
            return max(1, base_threads - 1)
        else:
            return base_threads

    def _calculate_optimal_batch_size(self, system_pressure: float) -> int:
        """Calculate optimal batch size based on system pressure."""
        # Base batch size
        base_size = 100
        
        # Adjust based on pressure
        if system_pressure > 0.8:
            return base_size // 4
        elif system_pressure > 0.6:
            return base_size // 2
        else:
            return base_size

    async def execute_batch(self, tasks: list) -> list:
        """Execute a batch of tasks with resource management."""
        try:
            # Split into smaller batches if needed
            batch_size = self.limits.batch_size
            results = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Execute batch using thread pool
                batch_futures = [
                    self.thread_pool.submit(task) for task in batch
                ]
                
                # Gather results
                batch_results = [
                    future.result() for future in batch_futures
                ]
                results.extend(batch_results)
                
                # Check resource pressure and adjust if needed
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                await self._adjust_resources(cpu_percent, memory_percent)
                
                # Add small delay if system is under pressure
                if cpu_percent > self.limits.max_cpu_percent:
                    await asyncio.sleep(0.1)

            return results

        except Exception as e:
            self.logger.error(f"Error executing batch: {str(e)}")
            return []

    def get_resource_metrics(self) -> Dict:
        """Get current resource metrics."""
        try:
            return {
                "cpu_usage": {
                    "current": psutil.cpu_percent(),
                    "average": np.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
                    "max": max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0
                },
                "memory_usage": {
                    "current": psutil.virtual_memory().percent,
                    "average": np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
                    "max": max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
                },
                "thread_pool": {
                    "size": self.thread_pool._max_workers,
                    "active": len([t for t in self.thread_pool._threads if t.is_alive()]),
                    "tasks_pending": self.thread_pool._work_queue.qsize()
                },
                "batch_size": self.limits.batch_size,
                "operations_per_second": self.metrics["operations_per_second"]
            }

        except Exception as e:
            self.logger.error(f"Error getting resource metrics: {str(e)}")
            return {}

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            self.logger.info("Resource manager cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    async def optimize_for_background(self):
        """Optimize settings for background operation."""
        try:
            # Reduce resource limits
            self.limits.max_cpu_percent = 40.0  # Lower CPU usage
            self.limits.max_memory_percent = 60.0  # Lower memory usage
            self.limits.thread_pool_size = max(2, self.limits.cpu_cores // 2)
            self.limits.batch_size = 50  # Smaller batches
            
            # Adjust current thread pool
            await self._adjust_resources(
                psutil.cpu_percent(),
                psutil.virtual_memory().percent
            )
            
            self.logger.info("Optimized for background operation")
            
        except Exception as e:
            self.logger.error(f"Error optimizing for background: {str(e)}")

    async def optimize_for_active(self):
        """Optimize settings for active operation."""
        try:
            # Restore resource limits
            self.limits.max_cpu_percent = 70.0
            self.limits.max_memory_percent = 80.0
            self.limits.thread_pool_size = self.limits.cpu_cores
            self.limits.batch_size = 100
            
            # Adjust current thread pool
            await self._adjust_resources(
                psutil.cpu_percent(),
                psutil.virtual_memory().percent
            )
            
            self.logger.info("Optimized for active operation")
            
        except Exception as e:
            self.logger.error(f"Error optimizing for active: {str(e)}")

    def set_priority(self, priority: str):
        """Set process priority."""
        try:
            process = psutil.Process()
            
            if priority == "low":
                if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(10)  # Unix-like systems
            elif priority == "normal":
                if hasattr(psutil, "NORMAL_PRIORITY_CLASS"):
                    process.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(0)
            elif priority == "high":
                if hasattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS"):
                    process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(-10)
            
            self.logger.info(f"Process priority set to {priority}")
            
        except Exception as e:
            self.logger.error(f"Error setting priority: {str(e)}")
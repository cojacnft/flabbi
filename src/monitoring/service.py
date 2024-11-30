import asyncio
import logging
from typing import Dict, List, Optional
import aiohttp
from datetime import datetime
import json

class MonitoringService:
    """Service for monitoring and alerting."""
    def __init__(
        self,
        config: Dict,
        services: Dict
    ):
        self.config = config
        self.services = services
        self.logger = logging.getLogger(__name__)
        
        # Initialize dashboard
        from .dashboard import DashboardService
        self.dashboard = DashboardService(
            flash_loan_executor=services["flash_loan_executor"],
            strategy_optimizer=services["strategy_optimizer"],
            market_analyzer=services["market_analyzer"],
            parameter_tuner=services["parameter_tuner"],
            config=config
        )
        
        # Alert history
        self.alert_history: List[Dict] = []
        self.max_alert_history = 1000

    async def start(self):
        """Start monitoring service."""
        try:
            # Start dashboard
            asyncio.create_task(self.dashboard.start())
            
            # Start alert monitoring
            asyncio.create_task(self._monitor_alerts())
            
            self.logger.info("Monitoring service started")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")

    async def stop(self):
        """Stop monitoring service."""
        try:
            # Stop dashboard
            await self.dashboard.app.shutdown()
            
            self.logger.info("Monitoring service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")

    async def _monitor_alerts(self):
        """Monitor system for alerts."""
        try:
            while True:
                alerts = []
                
                # Check flash loan executor
                flash_metrics = self.services["flash_loan_executor"].get_metrics()
                flash_alerts = self._check_flash_loan_alerts(flash_metrics)
                alerts.extend(flash_alerts)
                
                # Check strategy optimizer
                strategy_metrics = self.services["strategy_optimizer"].get_metrics()
                strategy_alerts = self._check_strategy_alerts(strategy_metrics)
                alerts.extend(strategy_alerts)
                
                # Check market analyzer
                market_metrics = await self.services["market_analyzer"].get_metrics()
                market_alerts = self._check_market_alerts(market_metrics)
                alerts.extend(market_alerts)
                
                # Send alerts if any
                if alerts:
                    await self._send_alerts(alerts)
                
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Error monitoring alerts: {str(e)}")

    def _check_flash_loan_alerts(self, metrics: Dict) -> List[Dict]:
        """Check flash loan executor metrics for alerts."""
        alerts = []
        
        # Check success rate
        success_rate = metrics["successful_executions"] / max(1, metrics["total_executions"])
        if success_rate < 0.8:
            alerts.append({
                "level": "warning",
                "source": "flash_loan",
                "message": f"Low success rate: {success_rate:.1%}"
            })
        
        # Check failed executions
        if metrics["failed_executions"] > 5:
            alerts.append({
                "level": "error",
                "source": "flash_loan",
                "message": f"High number of failed executions: {metrics['failed_executions']}"
            })
        
        return alerts

    def _check_strategy_alerts(self, metrics: Dict) -> List[Dict]:
        """Check strategy optimizer metrics for alerts."""
        alerts = []
        
        # Check MEV attacks
        if metrics["mev_attacks_prevented"] > 10:
            alerts.append({
                "level": "warning",
                "source": "strategy",
                "message": f"High MEV activity: {metrics['mev_attacks_prevented']} attacks prevented"
            })
        
        # Check profit ratio
        if metrics["avg_profit_per_trade"] < 50:
            alerts.append({
                "level": "warning",
                "source": "strategy",
                "message": f"Low profit per trade: ${metrics['avg_profit_per_trade']:.2f}"
            })
        
        return alerts

    def _check_market_alerts(self, metrics: Dict) -> List[Dict]:
        """Check market analyzer metrics for alerts."""
        alerts = []
        
        # Check gas price
        if metrics["current_gas_price"] > 200:
            alerts.append({
                "level": "warning",
                "source": "market",
                "message": f"High gas price: {metrics['current_gas_price']} gwei"
            })
        
        # Check network load
        if metrics["network_load"] > 0.8:
            alerts.append({
                "level": "warning",
                "source": "market",
                "message": f"High network load: {metrics['network_load']:.1%}"
            })
        
        return alerts

    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts to configured channels."""
        try:
            # Add to history
            self.alert_history.extend(alerts)
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
            
            # Send to Discord
            if self.config.get("discord_webhook"):
                await self._send_discord_alerts(alerts)
            
            # Send to Telegram
            if self.config.get("telegram_bot_token"):
                await self._send_telegram_alerts(alerts)
            
            # Log alerts
            for alert in alerts:
                log_level = logging.WARNING if alert["level"] == "warning" else logging.ERROR
                self.logger.log(log_level, f"{alert['source']}: {alert['message']}")
            
        except Exception as e:
            self.logger.error(f"Error sending alerts: {str(e)}")

    async def _send_discord_alerts(self, alerts: List[Dict]):
        """Send alerts to Discord."""
        try:
            webhook_url = self.config["discord_webhook"]
            
            async with aiohttp.ClientSession() as session:
                for alert in alerts:
                    # Create Discord embed
                    embed = {
                        "title": f"Alert: {alert['source'].title()}",
                        "description": alert['message'],
                        "color": 16711680 if alert["level"] == "error" else 16776960,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    payload = {
                        "embeds": [embed]
                    }
                    
                    async with session.post(
                        webhook_url,
                        json=payload
                    ) as response:
                        if response.status != 204:
                            self.logger.error(
                                f"Error sending Discord alert: {response.status}"
                            )
            
        except Exception as e:
            self.logger.error(f"Error sending Discord alerts: {str(e)}")

    async def _send_telegram_alerts(self, alerts: List[Dict]):
        """Send alerts to Telegram."""
        try:
            bot_token = self.config["telegram_bot_token"]
            chat_id = self.config["telegram_chat_id"]
            
            async with aiohttp.ClientSession() as session:
                for alert in alerts:
                    # Create message
                    emoji = "🔴" if alert["level"] == "error" else "⚠️"
                    message = (
                        f"{emoji} *{alert['source'].title()} Alert*\n"
                        f"{alert['message']}"
                    )
                    
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    params = {
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "Markdown"
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            self.logger.error(
                                f"Error sending Telegram alert: {response.status}"
                            )
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alerts: {str(e)}")

    async def record_execution(
        self,
        plan: Dict,
        result: Dict
    ):
        """Record execution result."""
        try:
            # Update dashboard metrics
            await self.dashboard._broadcast_metrics({
                "type": "execution",
                "data": {
                    "plan": plan,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            # Check for alerts
            if not result["success"]:
                await self._send_alerts([{
                    "level": "error",
                    "source": "execution",
                    "message": f"Execution failed: {result.get('error', 'Unknown error')}"
                }])
            
        except Exception as e:
            self.logger.error(f"Error recording execution: {str(e)}")

    def get_metrics(self) -> Dict:
        """Get monitoring metrics."""
        return {
            "alerts_sent": len(self.alert_history),
            "recent_alerts": self.alert_history[-10:],
            "last_update": datetime.utcnow().isoformat()
        }
from typing import Dict, List, Optional
import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
import json

class AlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds
        self.thresholds = {
            "profit": {
                "min_profit_usd": 50.0,
                "high_profit_usd": 500.0
            },
            "gas": {
                "max_gas_gwei": 100,
                "critical_gas_gwei": 200
            },
            "performance": {
                "min_success_rate": 0.8,
                "min_execution_speed_ms": 500
            },
            "system": {
                "max_cpu_percent": 80,
                "max_memory_percent": 80
            }
        }
        
        # Alert history
        self.alert_history: List[Dict] = []
        self.max_history = 1000
        
        # Rate limiting
        self.rate_limits = {
            "telegram": timedelta(minutes=1),
            "discord": timedelta(minutes=1),
            "email": timedelta(minutes=5)
        }
        self.last_alerts: Dict[str, datetime] = {}

    async def check_thresholds(self, metrics: Dict):
        """Check metrics against thresholds and generate alerts."""
        try:
            alerts = []
            
            # Check profit alerts
            if "profit" in metrics:
                profit = metrics["profit"].get("last_profit", 0)
                if profit > self.thresholds["profit"]["high_profit_usd"]:
                    alerts.append({
                        "level": "info",
                        "title": "High Profit Opportunity",
                        "message": f"Profit of ${profit:.2f} detected",
                        "type": "profit"
                    })
                elif profit < self.thresholds["profit"]["min_profit_usd"]:
                    alerts.append({
                        "level": "warning",
                        "title": "Low Profit Alert",
                        "message": f"Profit of ${profit:.2f} below threshold",
                        "type": "profit"
                    })
            
            # Check gas alerts
            if "gas" in metrics:
                gas_price = metrics["gas"].get("current_gas_gwei", 0)
                if gas_price > self.thresholds["gas"]["critical_gas_gwei"]:
                    alerts.append({
                        "level": "critical",
                        "title": "Critical Gas Price",
                        "message": f"Gas price at {gas_price} Gwei",
                        "type": "gas"
                    })
                elif gas_price > self.thresholds["gas"]["max_gas_gwei"]:
                    alerts.append({
                        "level": "warning",
                        "title": "High Gas Price",
                        "message": f"Gas price at {gas_price} Gwei",
                        "type": "gas"
                    })
            
            # Check performance alerts
            if "performance" in metrics:
                success_rate = metrics["performance"].get("success_rate", 1)
                if success_rate < self.thresholds["performance"]["min_success_rate"]:
                    alerts.append({
                        "level": "warning",
                        "title": "Low Success Rate",
                        "message": f"Success rate at {success_rate*100:.1f}%",
                        "type": "performance"
                    })
            
            # Check system alerts
            if "system" in metrics:
                cpu_percent = metrics["system"].get("cpu_percent", 0)
                memory_percent = metrics["system"].get("memory_percent", 0)
                
                if cpu_percent > self.thresholds["system"]["max_cpu_percent"]:
                    alerts.append({
                        "level": "warning",
                        "title": "High CPU Usage",
                        "message": f"CPU usage at {cpu_percent}%",
                        "type": "system"
                    })
                
                if memory_percent > self.thresholds["system"]["max_memory_percent"]:
                    alerts.append({
                        "level": "warning",
                        "title": "High Memory Usage",
                        "message": f"Memory usage at {memory_percent}%",
                        "type": "system"
                    })
            
            # Process alerts
            for alert in alerts:
                await self._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking thresholds: {str(e)}")

    async def _process_alert(self, alert: Dict):
        """Process and send alert through configured channels."""
        try:
            # Add timestamp
            alert["timestamp"] = datetime.utcnow()
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]
            
            # Check rate limits
            for channel in ["telegram", "discord", "email"]:
                if await self._check_rate_limit(channel, alert["type"]):
                    await self._send_alert(channel, alert)
            
        except Exception as e:
            self.logger.error(f"Error processing alert: {str(e)}")

    async def _check_rate_limit(self, channel: str, alert_type: str) -> bool:
        """Check if alert can be sent based on rate limits."""
        try:
            key = f"{channel}_{alert_type}"
            now = datetime.utcnow()
            
            if key in self.last_alerts:
                time_since_last = now - self.last_alerts[key]
                if time_since_last < self.rate_limits[channel]:
                    return False
            
            self.last_alerts[key] = now
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {str(e)}")
            return False

    async def _send_alert(self, channel: str, alert: Dict):
        """Send alert through specified channel."""
        try:
            if channel == "telegram" and self.config.get("telegram_token"):
                await self._send_telegram_alert(alert)
            elif channel == "discord" and self.config.get("discord_webhook"):
                await self._send_discord_alert(alert)
            elif channel == "email" and self.config.get("email_settings"):
                await self._send_email_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error sending {channel} alert: {str(e)}")

    async def _send_telegram_alert(self, alert: Dict):
        """Send alert through Telegram."""
        try:
            message = self._format_telegram_message(alert)
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage",
                    json={
                        "chat_id": self.config["telegram_chat_id"],
                        "text": message,
                        "parse_mode": "HTML"
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {str(e)}")

    async def _send_discord_alert(self, alert: Dict):
        """Send alert through Discord."""
        try:
            embed = self._format_discord_embed(alert)
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.config["discord_webhook"],
                    json={
                        "embeds": [embed]
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error sending Discord alert: {str(e)}")

    async def _send_email_alert(self, alert: Dict):
        """Send alert through email."""
        try:
            subject, body = self._format_email_message(alert)
            
            # TODO: Implement email sending
            pass
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")

    def _format_telegram_message(self, alert: Dict) -> str:
        """Format alert for Telegram."""
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        
        return (
            f"{emoji_map.get(alert['level'], '❗')} "
            f"<b>{alert['title']}</b>\n\n"
            f"{alert['message']}\n\n"
            f"Type: {alert['type']}\n"
            f"Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    def _format_discord_embed(self, alert: Dict) -> Dict:
        """Format alert for Discord."""
        color_map = {
            "info": 0x3498db,
            "warning": 0xf1c40f,
            "critical": 0xe74c3c
        }
        
        return {
            "title": alert["title"],
            "description": alert["message"],
            "color": color_map.get(alert["level"], 0x95a5a6),
            "fields": [
                {
                    "name": "Type",
                    "value": alert["type"],
                    "inline": True
                },
                {
                    "name": "Level",
                    "value": alert["level"],
                    "inline": True
                }
            ],
            "timestamp": alert["timestamp"].isoformat()
        }

    def _format_email_message(self, alert: Dict) -> Tuple[str, str]:
        """Format alert for email."""
        subject = f"[{alert['level'].upper()}] {alert['title']}"
        
        body = f"""
        Alert Details:
        --------------
        Title: {alert['title']}
        Message: {alert['message']}
        Type: {alert['type']}
        Level: {alert['level']}
        Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        return subject, body

    def get_recent_alerts(
        self,
        alert_type: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent alerts with optional filtering."""
        try:
            alerts = self.alert_history
            
            if alert_type:
                alerts = [a for a in alerts if a["type"] == alert_type]
            
            if level:
                alerts = [a for a in alerts if a["level"] == level]
            
            return sorted(
                alerts,
                key=lambda x: x["timestamp"],
                reverse=True
            )[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {str(e)}")
            return []

    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds."""
        try:
            for category, values in new_thresholds.items():
                if category in self.thresholds:
                    self.thresholds[category].update(values)
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {str(e)}")

    async def cleanup_history(self):
        """Clean up old alerts from history."""
        try:
            cutoff = datetime.utcnow() - timedelta(days=7)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert["timestamp"] > cutoff
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up history: {str(e)}")

    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        try:
            stats = {
                "total_alerts": len(self.alert_history),
                "by_type": {},
                "by_level": {},
                "last_24h": 0
            }
            
            cutoff = datetime.utcnow() - timedelta(hours=24)
            
            for alert in self.alert_history:
                # Count by type
                alert_type = alert["type"]
                if alert_type not in stats["by_type"]:
                    stats["by_type"][alert_type] = 0
                stats["by_type"][alert_type] += 1
                
                # Count by level
                alert_level = alert["level"]
                if alert_level not in stats["by_level"]:
                    stats["by_level"][alert_level] = 0
                stats["by_level"][alert_level] += 1
                
                # Count last 24h
                if alert["timestamp"] > cutoff:
                    stats["last_24h"] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting alert stats: {str(e)}")
            return {}
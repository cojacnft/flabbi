from typing import Dict, List, Optional
import asyncio
import aiohttp
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import json

from ..config import SystemConfig
from .path_finder import ArbitragePath
from .simulator import SimulationResult

class NotificationService:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._load_notification_config()

    def _load_notification_config(self):
        """Load notification configuration."""
        try:
            with open('config/notifications.json', 'r') as f:
                self.notification_config = json.load(f)
        except FileNotFoundError:
            self.notification_config = {
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_address": "",
                    "to_addresses": []
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "",
                    "chat_ids": []
                },
                "discord": {
                    "enabled": False,
                    "webhook_url": ""
                },
                "thresholds": {
                    "min_profit_alert": 100.0,  # USD
                    "high_profit_alert": 1000.0,  # USD
                    "gas_price_alert": 100,  # Gwei
                    "min_success_probability": 0.8
                }
            }
            self._save_notification_config()

    def _save_notification_config(self):
        """Save notification configuration."""
        try:
            with open('config/notifications.json', 'w') as f:
                json.dump(self.notification_config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving notification config: {str(e)}")

    async def notify_opportunity(
        self,
        path: ArbitragePath,
        simulation: Optional[SimulationResult] = None
    ):
        """Notify about a profitable arbitrage opportunity."""
        try:
            if path.expected_profit < self.notification_config["thresholds"]["min_profit_alert"]:
                return

            message = self._format_opportunity_message(path, simulation)

            # Send notifications through configured channels
            tasks = []
            
            if self.notification_config["email"]["enabled"]:
                tasks.append(self._send_email(
                    "Arbitrage Opportunity Alert",
                    message
                ))
            
            if self.notification_config["telegram"]["enabled"]:
                tasks.append(self._send_telegram(message))
            
            if self.notification_config["discord"]["enabled"]:
                tasks.append(self._send_discord(message))

            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Error sending opportunity notification: {str(e)}")

    async def notify_execution(
        self,
        path: ArbitragePath,
        success: bool,
        profit_usd: float,
        error: Optional[str] = None
    ):
        """Notify about arbitrage execution results."""
        try:
            message = self._format_execution_message(
                path,
                success,
                profit_usd,
                error
            )

            tasks = []
            
            if self.notification_config["email"]["enabled"]:
                tasks.append(self._send_email(
                    "Arbitrage Execution Alert",
                    message
                ))
            
            if self.notification_config["telegram"]["enabled"]:
                tasks.append(self._send_telegram(message))
            
            if self.notification_config["discord"]["enabled"]:
                tasks.append(self._send_discord(message))

            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Error sending execution notification: {str(e)}")

    async def notify_error(self, error: str, severity: str = "medium"):
        """Notify about system errors."""
        try:
            message = self._format_error_message(error, severity)

            tasks = []
            
            if self.notification_config["email"]["enabled"]:
                tasks.append(self._send_email(
                    f"Arbitrage System Error - {severity.upper()}",
                    message
                ))
            
            if self.notification_config["telegram"]["enabled"]:
                tasks.append(self._send_telegram(message))
            
            if self.notification_config["discord"]["enabled"]:
                tasks.append(self._send_discord(message))

            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")

    def _format_opportunity_message(
        self,
        path: ArbitragePath,
        simulation: Optional[SimulationResult] = None
    ) -> str:
        """Format opportunity notification message."""
        try:
            message = [
                "🚨 Arbitrage Opportunity Detected 🚨",
                "",
                f"Expected Profit: ${path.expected_profit:.2f}",
                f"Success Probability: {path.success_probability * 100:.1f}%",
                "",
                "Path:",
            ]

            # Add path details
            for i in range(len(path.tokens) - 1):
                message.append(
                    f"{i+1}. {path.tokens[i]} -> {path.tokens[i+1]} "
                    f"(via {path.dexes[i]})"
                )

            # Add simulation results if available
            if simulation:
                message.extend([
                    "",
                    "Simulation Results:",
                    f"Actual Profit: ${simulation.actual_profit_usd:.2f}",
                    f"Gas Used: {simulation.gas_used:,}",
                    f"Slippage: {simulation.slippage * 100:.2f}%",
                    f"Execution Time: {simulation.execution_time_ms}ms"
                ])

            return "\n".join(message)

        except Exception as e:
            self.logger.error(f"Error formatting opportunity message: {str(e)}")
            return "Error formatting opportunity message"

    def _format_execution_message(
        self,
        path: ArbitragePath,
        success: bool,
        profit_usd: float,
        error: Optional[str] = None
    ) -> str:
        """Format execution notification message."""
        try:
            status = "✅ Success" if success else "❌ Failed"
            message = [
                f"Arbitrage Execution {status}",
                "",
                f"Actual Profit: ${profit_usd:.2f}",
                "",
                "Path:"
            ]

            for i in range(len(path.tokens) - 1):
                message.append(
                    f"{i+1}. {path.tokens[i]} -> {path.tokens[i+1]} "
                    f"(via {path.dexes[i]})"
                )

            if error:
                message.extend([
                    "",
                    "Error:",
                    error
                ])

            return "\n".join(message)

        except Exception as e:
            self.logger.error(f"Error formatting execution message: {str(e)}")
            return "Error formatting execution message"

    def _format_error_message(self, error: str, severity: str) -> str:
        """Format error notification message."""
        try:
            severity_emoji = {
                "low": "ℹ️",
                "medium": "⚠️",
                "high": "🚨"
            }

            return (
                f"{severity_emoji.get(severity, '⚠️')} System Error - "
                f"{severity.upper()}\n\n"
                f"Error: {error}\n"
                f"Time: {datetime.utcnow().isoformat()}"
            )

        except Exception as e:
            self.logger.error(f"Error formatting error message: {str(e)}")
            return "Error formatting error message"

    async def _send_email(self, subject: str, body: str):
        """Send email notification."""
        try:
            config = self.notification_config["email"]
            if not config["enabled"]:
                return

            msg = MIMEMultipart()
            msg["From"] = config["from_address"]
            msg["To"] = ", ".join(config["to_addresses"])
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
                server.starttls()
                server.login(config["username"], config["password"])
                server.send_message(msg)

        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")

    async def _send_telegram(self, message: str):
        """Send Telegram notification."""
        try:
            config = self.notification_config["telegram"]
            if not config["enabled"]:
                return

            async with aiohttp.ClientSession() as session:
                for chat_id in config["chat_ids"]:
                    url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
                    async with session.post(url, json={
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "HTML"
                    }) as response:
                        if response.status != 200:
                            self.logger.error(
                                f"Telegram API error: {await response.text()}"
                            )

        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")

    async def _send_discord(self, message: str):
        """Send Discord notification."""
        try:
            config = self.notification_config["discord"]
            if not config["enabled"]:
                return

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["webhook_url"],
                    json={"content": message}
                ) as response:
                    if response.status != 204:
                        self.logger.error(
                            f"Discord webhook error: {await response.text()}"
                        )

        except Exception as e:
            self.logger.error(f"Error sending Discord message: {str(e)}")

    async def update_notification_config(self, new_config: Dict) -> bool:
        """Update notification configuration."""
        try:
            self.notification_config.update(new_config)
            self._save_notification_config()
            return True
        except Exception as e:
            self.logger.error(f"Error updating notification config: {str(e)}")
            return False
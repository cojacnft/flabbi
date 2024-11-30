import json
import logging
from typing import Dict, Optional
from pathlib import Path
import asyncio
from datetime import datetime

from ..models.settings import SystemSettings, DEFAULT_SETTINGS

class SettingsManager:
    def __init__(self, settings_file: str = "config/settings.json"):
        self.settings_file = settings_file
        self.logger = logging.getLogger(__name__)
        self.settings: SystemSettings = DEFAULT_SETTINGS
        self.settings_lock = asyncio.Lock()
        self._load_settings()
        
        # Keep track of settings changes
        self.last_updated = datetime.utcnow()
        self.update_callbacks = []

    def _load_settings(self):
        """Load settings from file."""
        try:
            path = Path(self.settings_file)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.settings = SystemSettings.parse_obj(data)
                    self.logger.info("Settings loaded successfully")
            else:
                self._save_settings()
                self.logger.info("Default settings created")
        except Exception as e:
            self.logger.error(f"Error loading settings: {str(e)}")
            self.settings = DEFAULT_SETTINGS

    def _save_settings(self):
        """Save settings to file."""
        try:
            path = Path(self.settings_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(
                    self.settings.dict(),
                    f,
                    indent=2,
                    default=str
                )
            self.last_updated = datetime.utcnow()
            self.logger.info("Settings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")

    async def update_settings(self, new_settings: Dict) -> bool:
        """Update settings with new values."""
        try:
            async with self.settings_lock:
                # Create new settings object
                updated_settings = SystemSettings.parse_obj({
                    **self.settings.dict(),
                    **new_settings
                })
                
                # Validate settings
                if not self._validate_settings(updated_settings):
                    return False
                
                # Apply settings
                self.settings = updated_settings
                self._save_settings()
                
                # Notify callbacks
                await self._notify_updates()
                
                return True
        except Exception as e:
            self.logger.error(f"Error updating settings: {str(e)}")
            return False

    def _validate_settings(self, settings: SystemSettings) -> bool:
        """Validate settings for consistency and safety."""
        try:
            # Performance validation
            if settings.performance.max_cpu_percent > 90:
                self.logger.warning("CPU usage limit too high")
                return False
            
            if settings.performance.max_memory_percent > 90:
                self.logger.warning("Memory usage limit too high")
                return False
            
            # Risk management validation
            if settings.risk_management.max_position_size_usd > 100000:
                self.logger.warning("Position size limit too high")
                return False
            
            if settings.risk_management.max_daily_loss_usd > 10000:
                self.logger.warning("Daily loss limit too high")
                return False
            
            # Path finding validation
            if settings.path_finding.max_hops > 6:
                self.logger.warning("Max hops too high")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating settings: {str(e)}")
            return False

    def register_callback(self, callback):
        """Register a callback for settings updates."""
        if callback not in self.update_callbacks:
            self.update_callbacks.append(callback)

    def unregister_callback(self, callback):
        """Unregister a settings update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    async def _notify_updates(self):
        """Notify all registered callbacks of settings updates."""
        for callback in self.update_callbacks:
            try:
                await callback(self.settings)
            except Exception as e:
                self.logger.error(f"Error in settings callback: {str(e)}")

    def get_settings(self) -> SystemSettings:
        """Get current settings."""
        return self.settings

    def reset_to_defaults(self) -> bool:
        """Reset settings to defaults."""
        try:
            self.settings = DEFAULT_SETTINGS
            self._save_settings()
            return True
        except Exception as e:
            self.logger.error(f"Error resetting settings: {str(e)}")
            return False

    async def apply_performance_mode(self):
        """Apply current performance mode settings."""
        try:
            from .resource_manager import ResourceManager
            
            resource_manager = ResourceManager()
            
            if self.settings.performance.mode == "boost":
                await resource_manager.optimize_for_active()
            elif self.settings.performance.mode == "background":
                await resource_manager.optimize_for_background()
            else:  # balanced
                # Custom balanced settings
                await resource_manager.optimize_for_background()
                resource_manager.limits.max_cpu_percent = 55.0
                resource_manager.limits.max_memory_percent = 70.0
                
            return True
        except Exception as e:
            self.logger.error(f"Error applying performance mode: {str(e)}")
            return False

    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(
                    self.settings.dict(),
                    f,
                    indent=2,
                    default=str
                )
            return True
        except Exception as e:
            self.logger.error(f"Error exporting settings: {str(e)}")
            return False

    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                new_settings = SystemSettings.parse_obj(data)
                
                if self._validate_settings(new_settings):
                    self.settings = new_settings
                    self._save_settings()
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error importing settings: {str(e)}")
            return False
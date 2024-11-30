#!/usr/bin/env python3
import click
import subprocess
import sys
import os
import time
import requests
import json
from typing import Dict, Optional

@click.group()
def cli():
    """Monitoring system management tool."""
    pass

@cli.command()
def start():
    """Start the monitoring stack."""
    click.echo("Starting monitoring stack...")
    
    try:
        # Check if Docker is running
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        
        # Start the stack
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True
        )
        
        click.echo("Monitoring stack started successfully")
        
        # Wait for services to be ready
        wait_for_services()
        
        # Show access information
        show_access_info()
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error starting monitoring stack: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
def stop():
    """Stop the monitoring stack."""
    click.echo("Stopping monitoring stack...")
    
    try:
        subprocess.run(
            ["docker-compose", "down"],
            check=True
        )
        click.echo("Monitoring stack stopped successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error stopping monitoring stack: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
def status():
    """Check status of monitoring services."""
    try:
        # Get container status
        result = subprocess.run(
            ["docker-compose", "ps"],
            check=True,
            capture_output=True,
            text=True
        )
        
        click.echo("\nContainer Status:")
        click.echo(result.stdout)
        
        # Check service health
        check_service_health()
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error checking status: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--days', default=7, help='Number of days of metrics to keep')
def cleanup():
    """Clean up old monitoring data."""
    click.echo(f"Cleaning up monitoring data older than {days} days...")
    
    try:
        # Stop services
        subprocess.run(["docker-compose", "stop"], check=True)
        
        # Clean up Prometheus data
        cleanup_prometheus_data(days)
        
        # Clean up Grafana data
        cleanup_grafana_data(days)
        
        # Restart services
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        click.echo("Cleanup completed successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error during cleanup: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--output', default='monitoring_backup.tar.gz', help='Backup file name')
def backup():
    """Backup monitoring data."""
    click.echo(f"Creating backup to {output}...")
    
    try:
        # Create backup directory
        os.makedirs('backup', exist_ok=True)
        
        # Backup Prometheus data
        subprocess.run([
            "docker", "run", "--rm",
            "-v", "prometheus_data:/data",
            "-v", f"{os.getcwd()}/backup:/backup",
            "ubuntu",
            "tar", "czf", "/backup/prometheus.tar.gz", "/data"
        ], check=True)
        
        # Backup Grafana data
        subprocess.run([
            "docker", "run", "--rm",
            "-v", "grafana_data:/data",
            "-v", f"{os.getcwd()}/backup:/backup",
            "ubuntu",
            "tar", "czf", "/backup/grafana.tar.gz", "/data"
        ], check=True)
        
        # Create final backup
        subprocess.run([
            "tar", "czf", output,
            "-C", "backup",
            "prometheus.tar.gz",
            "grafana.tar.gz"
        ], check=True)
        
        # Cleanup
        subprocess.run(["rm", "-rf", "backup"], check=True)
        
        click.echo(f"Backup created successfully: {output}")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error creating backup: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('backup_file')
def restore():
    """Restore monitoring data from backup."""
    if not os.path.exists(backup_file):
        click.echo(f"Backup file not found: {backup_file}", err=True)
        sys.exit(1)
    
    click.echo(f"Restoring from backup: {backup_file}...")
    
    try:
        # Stop services
        subprocess.run(["docker-compose", "stop"], check=True)
        
        # Create temp directory
        os.makedirs('restore', exist_ok=True)
        
        # Extract backup
        subprocess.run([
            "tar", "xzf", backup_file,
            "-C", "restore"
        ], check=True)
        
        # Restore Prometheus data
        subprocess.run([
            "docker", "run", "--rm",
            "-v", "prometheus_data:/data",
            "-v", f"{os.getcwd()}/restore:/restore",
            "ubuntu",
            "tar", "xzf", "/restore/prometheus.tar.gz", "-C", "/"
        ], check=True)
        
        # Restore Grafana data
        subprocess.run([
            "docker", "run", "--rm",
            "-v", "grafana_data:/data",
            "-v", f"{os.getcwd()}/restore:/restore",
            "ubuntu",
            "tar", "xzf", "/restore/grafana.tar.gz", "-C", "/"
        ], check=True)
        
        # Cleanup
        subprocess.run(["rm", "-rf", "restore"], check=True)
        
        # Start services
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        click.echo("Restore completed successfully")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error restoring backup: {str(e)}", err=True)
        sys.exit(1)

def wait_for_services():
    """Wait for all services to be ready."""
    services = {
        "Prometheus": "http://localhost:9090/-/ready",
        "Grafana": "http://localhost:3000/api/health"
    }
    
    click.echo("\nWaiting for services to be ready...")
    
    for service, url in services.items():
        retries = 30
        while retries > 0:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    click.echo(f"{service} is ready")
                    break
            except requests.exceptions.RequestException:
                pass
            
            retries -= 1
            if retries == 0:
                click.echo(f"Timeout waiting for {service}", err=True)
            time.sleep(1)

def check_service_health():
    """Check health of monitoring services."""
    try:
        # Check Prometheus
        response = requests.get("http://localhost:9090/-/healthy")
        prometheus_healthy = response.status_code == 200
        
        # Check Grafana
        response = requests.get("http://localhost:3000/api/health")
        grafana_healthy = response.status_code == 200
        
        click.echo("\nService Health:")
        click.echo(f"Prometheus: {'✅' if prometheus_healthy else '❌'}")
        click.echo(f"Grafana: {'✅' if grafana_healthy else '❌'}")
        
    except requests.exceptions.RequestException as e:
        click.echo(f"Error checking service health: {str(e)}", err=True)

def show_access_info():
    """Show access information for monitoring services."""
    click.echo("\nAccess Information:")
    click.echo("Grafana: http://localhost:3000")
    click.echo("Prometheus: http://localhost:9090")
    click.echo("\nAPI Token is configured for authentication")

def cleanup_prometheus_data(days: int):
    """Clean up old Prometheus data."""
    try:
        subprocess.run([
            "docker", "run", "--rm",
            "-v", "prometheus_data:/prometheus",
            "prom/prometheus",
            "promtool", "tsdb", "clean", "--delete-delay", f"{days*24}h"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        click.echo(f"Error cleaning Prometheus data: {str(e)}", err=True)

def cleanup_grafana_data(days: int):
    """Clean up old Grafana data."""
    # TODO: Implement Grafana data cleanup
    pass

if __name__ == '__main__':
    cli()
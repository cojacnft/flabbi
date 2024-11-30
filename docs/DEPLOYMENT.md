# Deployment Guide

This guide covers the deployment of the Flash Loan Arbitrage system in a production environment.

## System Requirements

### Hardware Requirements
- CPU: 4+ cores
- RAM: 16GB+ (32GB recommended)
- Storage: 100GB+ SSD
- Network: High-speed internet connection (1Gbps recommended)

### Software Requirements
- Ubuntu 20.04 LTS or later
- Python 3.8+
- Node.js 16+
- PostgreSQL 13+
- Redis 6+

### Network Requirements
- Ethereum node access (Infura, Alchemy, or private node)
- Low-latency connection to node
- Stable internet connection
- Firewall access for required ports

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    nginx \
    postgresql \
    postgresql-contrib \
    redis-server

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install -y nodejs

# Create application user
sudo useradd -m -s /bin/bash arbitrage
sudo usermod -aG sudo arbitrage
```

### 2. Application Setup

```bash
# Switch to application user
sudo su - arbitrage

# Clone repository
git clone https://github.com/yourusername/flash-loan-arbitrage.git
cd flash-loan-arbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### 3. Database Setup

```bash
# Create database
sudo -u postgres psql

postgres=# CREATE DATABASE arbitrage;
postgres=# CREATE USER arbitrage WITH PASSWORD 'your_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE arbitrage TO arbitrage;
postgres=# \q

# Initialize database
python scripts/init_db.py
```

### 4. Configuration

```bash
# Copy example configs
cp config/arbitrage.example.yaml config/arbitrage.yaml
cp config/monitoring.example.yaml config/monitoring.yaml

# Create environment file
cat > .env << EOL
# Network
NETWORK_RPC_URL=your_node_url
NETWORK_WS_URL=your_websocket_url
EXPLORER_API_KEY=your_etherscan_key

# Database
DATABASE_URL=postgresql://arbitrage:your_password@localhost/arbitrage

# Monitoring
DISCORD_WEBHOOK=your_discord_webhook
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Security
PRIVATE_KEY=your_private_key
API_KEY=your_api_key
EOL

# Set permissions
chmod 600 .env
```

### 5. Contract Deployment

```bash
# Compile contracts
npx hardhat compile

# Deploy to mainnet
python scripts/deploy.py --network mainnet

# Add contract address to .env
echo "ARBITRAGE_CONTRACT=deployed_contract_address" >> .env
```

### 6. Service Setup

#### Arbitrage Service
```bash
# Create service file
sudo tee /etc/systemd/system/arbitrage.service << EOL
[Unit]
Description=Flash Loan Arbitrage Bot
After=network.target postgresql.service redis-server.service

[Service]
User=arbitrage
WorkingDirectory=/home/arbitrage/flash-loan-arbitrage
Environment=PYTHONPATH=/home/arbitrage/flash-loan-arbitrage
EnvironmentFile=/home/arbitrage/flash-loan-arbitrage/.env
ExecStart=/home/arbitrage/flash-loan-arbitrage/venv/bin/python scripts/run_arbitrage.py
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
EOL
```

#### Monitoring Service
```bash
# Create service file
sudo tee /etc/systemd/system/arbitrage-monitor.service << EOL
[Unit]
Description=Flash Loan Arbitrage Monitor
After=network.target postgresql.service

[Service]
User=arbitrage
WorkingDirectory=/home/arbitrage/flash-loan-arbitrage
Environment=PYTHONPATH=/home/arbitrage/flash-loan-arbitrage
EnvironmentFile=/home/arbitrage/flash-loan-arbitrage/.env
ExecStart=/home/arbitrage/flash-loan-arbitrage/venv/bin/python scripts/run_monitoring.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL
```

### 7. Web Server Setup

```bash
# Create Nginx config
sudo tee /etc/nginx/sites-available/arbitrage << EOL
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOL

# Enable site
sudo ln -s /etc/nginx/sites-available/arbitrage /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL (optional)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

### 8. Start Services

```bash
# Start services
sudo systemctl daemon-reload
sudo systemctl enable arbitrage arbitrage-monitor
sudo systemctl start arbitrage arbitrage-monitor

# Check status
sudo systemctl status arbitrage
sudo systemctl status arbitrage-monitor
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Configure rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw allow 8000  # Dashboard port

# Enable firewall
sudo ufw enable
```

### 2. Private Key Management

```bash
# Create secure directory
sudo mkdir /opt/arbitrage/keys
sudo chown arbitrage:arbitrage /opt/arbitrage/keys
sudo chmod 700 /opt/arbitrage/keys

# Store private key
echo "your_private_key" | sudo tee /opt/arbitrage/keys/eth_key
sudo chmod 600 /opt/arbitrage/keys/eth_key

# Update service to use key file
sudo sed -i 's|PRIVATE_KEY=.*|PRIVATE_KEY=$(cat /opt/arbitrage/keys/eth_key)|' \
    /home/arbitrage/flash-loan-arbitrage/.env
```

### 3. Rate Limiting

```bash
# Update Nginx config
sudo tee /etc/nginx/conf.d/rate_limit.conf << EOL
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=ws:10m rate=1r/s;
EOL

# Apply to site config
sudo sed -i '/location \/ {/a\    limit_req zone=api burst=20;' \
    /etc/nginx/sites-available/arbitrage
sudo sed -i '/location \/ws {/a\    limit_req zone=ws burst=5;' \
    /etc/nginx/sites-available/arbitrage

sudo systemctl restart nginx
```

### 4. Database Security

```bash
# Update PostgreSQL config
sudo tee -a /etc/postgresql/13/main/postgresql.conf << EOL
ssl = on
ssl_cert_file = '/etc/ssl/certs/ssl-cert-snakeoil.pem'
ssl_key_file = '/etc/ssl/private/ssl-cert-snakeoil.key'
EOL

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## Monitoring Setup

### 1. Prometheus Setup

```bash
# Install Prometheus
sudo apt install -y prometheus

# Create config
sudo tee /etc/prometheus/prometheus.yml << EOL
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arbitrage'
    static_configs:
      - targets: ['localhost:8000']
EOL

# Restart Prometheus
sudo systemctl restart prometheus
```

### 2. Grafana Setup

```bash
# Install Grafana
sudo apt install -y grafana

# Start Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Import dashboard
curl -X POST \
    -H "Content-Type: application/json" \
    -d @config/grafana/dashboard.json \
    http://admin:admin@localhost:3000/api/dashboards/db
```

### 3. Alert Configuration

```bash
# Create alert rules
sudo tee /etc/prometheus/rules/arbitrage.yml << EOL
groups:
  - name: arbitrage
    rules:
      - alert: HighFailureRate
        expr: rate(arbitrage_failed_trades[5m]) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High trade failure rate
EOL

# Update Prometheus config
sudo systemctl restart prometheus
```

## Backup Setup

### 1. Database Backup

```bash
# Create backup script
sudo tee /opt/arbitrage/backup.sh << EOL
#!/bin/bash
BACKUP_DIR="/opt/arbitrage/backups"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
pg_dump arbitrage > \$BACKUP_DIR/db_\$TIMESTAMP.sql
find \$BACKUP_DIR -type f -mtime +7 -delete
EOL

# Make executable
sudo chmod +x /opt/arbitrage/backup.sh

# Add to crontab
echo "0 0 * * * /opt/arbitrage/backup.sh" | sudo tee -a /var/spool/cron/crontabs/arbitrage
```

### 2. Configuration Backup

```bash
# Create backup script
sudo tee /opt/arbitrage/config_backup.sh << EOL
#!/bin/bash
BACKUP_DIR="/opt/arbitrage/backups/config"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
tar -czf \$BACKUP_DIR/config_\$TIMESTAMP.tar.gz /home/arbitrage/flash-loan-arbitrage/config
find \$BACKUP_DIR -type f -mtime +30 -delete
EOL

# Make executable
sudo chmod +x /opt/arbitrage/config_backup.sh

# Add to crontab
echo "0 0 * * 0 /opt/arbitrage/config_backup.sh" | sudo tee -a /var/spool/cron/crontabs/arbitrage
```

## Maintenance

### 1. Log Rotation

```bash
# Create logrotate config
sudo tee /etc/logrotate.d/arbitrage << EOL
/var/log/arbitrage/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 640 arbitrage arbitrage
}
EOL
```

### 2. System Updates

```bash
# Create update script
sudo tee /opt/arbitrage/update.sh << EOL
#!/bin/bash
cd /home/arbitrage/flash-loan-arbitrage
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart arbitrage arbitrage-monitor
EOL

# Make executable
sudo chmod +x /opt/arbitrage/update.sh
```

### 3. Health Checks

```bash
# Create health check script
sudo tee /opt/arbitrage/health_check.sh << EOL
#!/bin/bash
curl -f http://localhost:8000/health || systemctl restart arbitrage
curl -f http://localhost:8000/metrics || systemctl restart arbitrage-monitor
EOL

# Make executable
sudo chmod +x /opt/arbitrage/health_check.sh

# Add to crontab
echo "*/5 * * * * /opt/arbitrage/health_check.sh" | sudo tee -a /var/spool/cron/crontabs/arbitrage
```

## Troubleshooting

### 1. Service Issues

```bash
# Check service status
sudo systemctl status arbitrage
sudo systemctl status arbitrage-monitor

# View logs
sudo journalctl -u arbitrage -f
sudo journalctl -u arbitrage-monitor -f

# Check application logs
tail -f /var/log/arbitrage/app.log
tail -f /var/log/arbitrage/error.log
```

### 2. Database Issues

```bash
# Check database connection
psql -U arbitrage -d arbitrage -c "\dt"

# Check database size
psql -U arbitrage -d arbitrage -c "SELECT pg_size_pretty(pg_database_size('arbitrage'));"

# Vacuum database
psql -U arbitrage -d arbitrage -c "VACUUM ANALYZE;"
```

### 3. Network Issues

```bash
# Check node connection
curl -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
    $NETWORK_RPC_URL

# Check WebSocket connection
wscat -c $NETWORK_WS_URL
```

### 4. Performance Issues

```bash
# Check system resources
top -u arbitrage
free -m
df -h

# Check network latency
ping etherscan.io
mtr $NETWORK_RPC_URL
```

## Scaling

### 1. Horizontal Scaling

```bash
# Load balancer setup
sudo apt install haproxy

# HAProxy config
sudo tee /etc/haproxy/haproxy.cfg << EOL
frontend arbitrage
    bind *:80
    default_backend arbitrage_backend

backend arbitrage_backend
    server arb1 localhost:8000 check
    server arb2 localhost:8001 check
EOL
```

### 2. Database Scaling

```bash
# Enable connection pooling
sudo apt install pgbouncer

# PgBouncer config
sudo tee /etc/pgbouncer/pgbouncer.ini << EOL
[databases]
arbitrage = host=127.0.0.1 port=5432 dbname=arbitrage

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
EOL
```

### 3. Caching

```bash
# Redis config
sudo tee /etc/redis/redis.conf << EOL
maxmemory 2gb
maxmemory-policy allkeys-lru
EOL

# Update application config
sed -i 's/CACHE_URL=.*/CACHE_URL=redis:\/\/localhost:6379\/0/' .env
```

## Monitoring

### 1. System Metrics

```bash
# Install node exporter
sudo apt install prometheus-node-exporter

# Add to Prometheus config
sudo tee -a /etc/prometheus/prometheus.yml << EOL
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOL
```

### 2. Application Metrics

```bash
# Enable metrics endpoint
sed -i 's/METRICS_ENABLED=.*/METRICS_ENABLED=true/' .env

# Add custom metrics
python scripts/add_custom_metrics.py
```

### 3. Alert Rules

```bash
# Create alert rules
sudo tee /etc/prometheus/rules/arbitrage_alerts.yml << EOL
groups:
  - name: arbitrage
    rules:
      - alert: HighGasPrice
        expr: ethereum_gas_price > 200
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High gas price detected
EOL
```

## Recovery Procedures

### 1. Service Recovery

```bash
# Restart services
sudo systemctl restart arbitrage arbitrage-monitor

# Check logs
sudo journalctl -u arbitrage -n 100
```

### 2. Database Recovery

```bash
# Restore from backup
pg_restore -U arbitrage -d arbitrage /opt/arbitrage/backups/db_latest.sql

# Check data integrity
psql -U arbitrage -d arbitrage -c "SELECT count(*) FROM trades;"
```

### 3. Configuration Recovery

```bash
# Restore configs
tar -xzf /opt/arbitrage/backups/config/config_latest.tar.gz -C /tmp
cp /tmp/config/* /home/arbitrage/flash-loan-arbitrage/config/
```

## Support

For support:

1. Check logs:
```bash
tail -f /var/log/arbitrage/*.log
```

2. Check metrics:
```bash
curl http://localhost:8000/metrics
```

3. Contact support:
```
Email: support@your-domain.com
Discord: https://discord.gg/your-server
```
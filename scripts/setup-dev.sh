#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up development environment...${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from .env.example${NC}"
    cp .env.example .env
    echo -e "${RED}Please update .env with your actual values${NC}"
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    yarn install
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p logs data deploy/prometheus deploy/grafana/dashboards

# Build TypeScript
echo -e "${YELLOW}Building TypeScript...${NC}"
yarn build

# Setup Hardhat
echo -e "${YELLOW}Setting up Hardhat...${NC}"
if [ ! -f "hardhat.config.ts" ]; then
    echo -e "${RED}Please create hardhat.config.ts${NC}"
    exit 1
fi

# Start development environment
echo -e "${YELLOW}Starting development environment...${NC}"
docker-compose -f docker-compose.yml up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if services are running
echo -e "${YELLOW}Checking services...${NC}"
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}Development environment is ready!${NC}"
    echo -e "\nAvailable services:"
    echo -e "- Arbitrage Bot API: http://localhost:3000"
    echo -e "- Prometheus: http://localhost:9091"
    echo -e "- Grafana: http://localhost:3001 (admin/admin)"
    echo -e "- Redis: localhost:6379"
    echo -e "- Hardhat Node: http://localhost:8545"
else
    echo -e "${RED}Some services failed to start. Please check docker-compose logs${NC}"
    exit 1
fi

# Setup git hooks
echo -e "${YELLOW}Setting up git hooks...${NC}"
if [ -d ".git" ]; then
    cp scripts/pre-commit .git/hooks/
    chmod +x .git/hooks/pre-commit
fi

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Don't forget to:${NC}"
echo -e "1. Update .env with your actual values"
echo -e "2. Configure your IDE for TypeScript"
echo -e "3. Run tests with 'yarn test'"
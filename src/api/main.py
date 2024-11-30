from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import asyncio
import logging
from web3 import Web3

from ..config import SystemConfig, DEFAULT_CONFIG
from ..models.token import Token
from ..services.token_database import TokenDatabase
from ..services.path_finder import PathFinder, ArbitragePath
from ..services.simulator import ArbitrageSimulator, SimulationResult
from ..services.notifications import NotificationService

app = FastAPI(title="Flash Loan Arbitrage API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
config = DEFAULT_CONFIG
web3 = Web3(Web3.HTTPProvider(config.network.rpc_url))
token_db = TokenDatabase(config, web3)
path_finder = PathFinder(config, web3, token_db)
simulator = ArbitrageSimulator(config, web3, token_db)
notification_service = NotificationService(config)

# WebSocket connections
active_connections: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logging.basicConfig(level=logging.INFO)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_opportunity(opportunity: ArbitragePath):
    """Broadcast opportunity to all connected clients."""
    message = {
        "type": "opportunity",
        "data": {
            "tokens": opportunity.tokens,
            "dexes": opportunity.dexes,
            "expected_profit": opportunity.expected_profit,
            "success_probability": opportunity.success_probability
        }
    }
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logging.error(f"Error broadcasting message: {str(e)}")

# Token Management Endpoints

@app.post("/tokens/")
async def add_token(address: str):
    """Add a new token to the database."""
    token = await token_db.add_token(address)
    if not token:
        raise HTTPException(status_code=400, detail="Failed to add token")
    return token

@app.get("/tokens/")
async def get_tokens(
    active_only: bool = False,
    verified_only: bool = False,
    tag: Optional[str] = None
):
    """Get tokens from the database with optional filters."""
    if tag:
        return await token_db.get_tokens_by_tag(tag)
    elif active_only:
        return await token_db.get_active_tokens()
    elif verified_only:
        return await token_db.get_verified_tokens()
    else:
        return await token_db.get_all_tokens()

@app.get("/tokens/{address}")
async def get_token(address: str):
    """Get a specific token's details."""
    token = await token_db.get_token(address)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    return token

@app.put("/tokens/{address}")
async def update_token(address: str, token: Token):
    """Update token details."""
    if not await token_db.update_token(token):
        raise HTTPException(status_code=400, detail="Failed to update token")
    return token

@app.delete("/tokens/{address}")
async def remove_token(address: str):
    """Remove a token from the database."""
    if not await token_db.remove_token(address):
        raise HTTPException(status_code=400, detail="Failed to remove token")
    return {"status": "success"}

# Arbitrage Endpoints

@app.get("/opportunities/{token_address}")
async def find_opportunities(
    token_address: str,
    min_profit: float = 100.0,
    max_paths: int = 10
):
    """Find arbitrage opportunities for a token."""
    paths = await path_finder.find_arbitrage_paths(
        token_address,
        min_profit,
        max_paths
    )
    return paths

@app.post("/simulate")
async def simulate_path(path: ArbitragePath, amount_usd: float = 1000.0):
    """Simulate an arbitrage path execution."""
    result = await simulator.simulate_path(path, amount_usd)
    return result

@app.get("/pairs/top")
async def get_top_pairs(limit: int = 10):
    """Get top token pairs by liquidity."""
    return await token_db.get_top_liquidity_pairs(limit)

# Notification Endpoints

@app.post("/notifications/test")
async def test_notification(
    message: str,
    channels: List[str] = ["email", "telegram", "discord"]
):
    """Test notification channels."""
    tasks = []
    if "email" in channels:
        tasks.append(notification_service._send_email(
            "Test Notification",
            message
        ))
    if "telegram" in channels:
        tasks.append(notification_service._send_telegram(message))
    if "discord" in channels:
        tasks.append(notification_service._send_discord(message))
    
    await asyncio.gather(*tasks)
    return {"status": "success"}

@app.get("/notifications/config")
async def get_notification_config():
    """Get notification configuration."""
    return notification_service.notification_config

@app.put("/notifications/config")
async def update_notification_config(config: dict):
    """Update notification configuration."""
    if not await notification_service.update_notification_config(config):
        raise HTTPException(
            status_code=400,
            detail="Failed to update notification config"
        )
    return {"status": "success"}

# System Configuration Endpoints

@app.get("/config")
async def get_system_config():
    """Get current system configuration."""
    return config

@app.put("/config")
async def update_system_config(new_config: SystemConfig):
    """Update system configuration."""
    global config
    config = new_config
    return config
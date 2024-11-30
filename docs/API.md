# Flash Loan Arbitrage API Documentation

## Core Services

### Flash Loan Executor

The Flash Loan Executor service handles flash loan execution and management.

#### Methods

##### `execute_flash_loan`
```python
async def execute_flash_loan(
    opportunity: Dict,
    context: Dict
) -> ExecutionResult:
    """
    Execute a flash loan arbitrage opportunity.
    
    Args:
        opportunity (Dict): {
            "token_in": str,            # Input token address
            "amount_in": int,           # Input amount in wei
            "path": List[Dict],         # Trading path
            "expected_profit": float,   # Expected profit in USD
            "flash_loan_config": Dict   # Flash loan configuration
        }
        context (Dict): {
            "network_load": float,      # Current network load (0-1)
            "gas_price": int           # Current gas price in gwei
        }
    
    Returns:
        ExecutionResult: {
            "success": bool,           # Execution success
            "profit": float,           # Actual profit in USD
            "gas_used": int,           # Gas used
            "execution_time": float,   # Execution time in seconds
            "error": Optional[str],    # Error message if failed
            "tx_hash": Optional[str]   # Transaction hash if successful
        }
    """
```

##### `find_opportunity`
```python
async def find_opportunity(
    path: List[Dict],
    amount_in: int
) -> Optional[Dict]:
    """
    Find arbitrage opportunity in given path.
    
    Args:
        path (List[Dict]): List of pools to trade through
        amount_in (int): Input amount in wei
    
    Returns:
        Optional[Dict]: Opportunity details if found
    """
```

##### `validate_opportunity`
```python
async def validate_opportunity(
    opportunity: Dict,
    context: Dict
) -> bool:
    """
    Validate arbitrage opportunity.
    
    Args:
        opportunity (Dict): Opportunity details
        context (Dict): Current market context
    
    Returns:
        bool: True if opportunity is valid
    """
```

### Strategy Optimizer

The Strategy Optimizer service handles execution strategy optimization.

#### Methods

##### `optimize_execution`
```python
async def optimize_execution(
    opportunity: Dict,
    context: Dict
) -> Optional[ExecutionPlan]:
    """
    Optimize execution strategy for opportunity.
    
    Args:
        opportunity (Dict): Opportunity details
        context (Dict): Current market context
    
    Returns:
        Optional[ExecutionPlan]: Optimized execution plan
    """
```

##### `analyze_mev_risks`
```python
async def analyze_mev_risks(
    opportunity: Dict,
    strategy: ExecutionStrategy
) -> Dict:
    """
    Analyze MEV risks for opportunity.
    
    Args:
        opportunity (Dict): Opportunity details
        strategy (ExecutionStrategy): Execution strategy
    
    Returns:
        Dict: Risk analysis results
    """
```

##### `prepare_execution_plan`
```python
async def prepare_execution_plan(
    opportunity: Dict,
    strategy: ExecutionStrategy,
    mev_risks: Dict
) -> Optional[ExecutionPlan]:
    """
    Prepare detailed execution plan.
    
    Args:
        opportunity (Dict): Opportunity details
        strategy (ExecutionStrategy): Execution strategy
        mev_risks (Dict): MEV risk analysis
    
    Returns:
        Optional[ExecutionPlan]: Detailed execution plan
    """
```

### Market Analyzer

The Market Analyzer service handles market analysis and opportunity detection.

#### Methods

##### `find_opportunities`
```python
async def find_opportunities(
    min_profit: float = 100.0
) -> List[Dict]:
    """
    Find arbitrage opportunities.
    
    Args:
        min_profit (float): Minimum profit threshold in USD
    
    Returns:
        List[Dict]: List of opportunities
    """
```

##### `get_market_state`
```python
async def get_market_state() -> Dict:
    """
    Get current market state.
    
    Returns:
        Dict: {
            "gas_price": int,          # Current gas price in gwei
            "network_load": float,     # Network load (0-1)
            "block_number": int,       # Current block number
            "timestamp": str           # Current timestamp
        }
    """
```

##### `get_pool_state`
```python
async def get_pool_state(
    pool_address: str
) -> Optional[Dict]:
    """
    Get pool state.
    
    Args:
        pool_address (str): Pool contract address
    
    Returns:
        Optional[Dict]: Pool state if available
    """
```

### Parameter Tuner

The Parameter Tuner service handles dynamic parameter optimization.

#### Methods

##### `get_optimal_parameters`
```python
async def get_optimal_parameters(
    context: Dict
) -> Dict[str, float]:
    """
    Get optimal parameters for current context.
    
    Args:
        context (Dict): Current market context
    
    Returns:
        Dict[str, float]: Optimal parameters
    """
```

##### `update_performance`
```python
async def update_performance(
    parameters: Dict[str, float],
    performance: float,
    context: Dict
):
    """
    Update performance history.
    
    Args:
        parameters (Dict[str, float]): Parameters used
        performance (float): Performance metric
        context (Dict): Market context
    """
```

### Monitoring Service

The Monitoring Service handles system monitoring and alerting.

#### Methods

##### `record_execution`
```python
async def record_execution(
    plan: Dict,
    result: Dict
):
    """
    Record execution result.
    
    Args:
        plan (Dict): Execution plan
        result (Dict): Execution result
    """
```

##### `get_metrics`
```python
def get_metrics() -> Dict:
    """
    Get monitoring metrics.
    
    Returns:
        Dict: Current metrics
    """
```

## Data Types

### ExecutionPlan
```python
@dataclass
class ExecutionPlan:
    """Detailed execution plan."""
    strategy: ExecutionStrategy
    transactions: List[Dict]
    estimated_profit: float
    estimated_gas: int
    priority_fee: int
    bundle_data: Optional[Dict]
```

### ExecutionStrategy
```python
@dataclass
class ExecutionStrategy:
    """Flash loan execution strategy."""
    name: str
    priority: int  # 1 (highest) to 3 (lowest)
    min_profit: float
    max_gas_price: int
    bundle_type: str  # 'flashbots', 'eden', 'standard'
    mev_protection: bool
    backrun_protection: bool
    timeout: int
```

### MetricsSnapshot
```python
@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics."""
    timestamp: str
    total_profit: float
    total_trades: int
    successful_trades: int
    failed_trades: int
    gas_spent: float
    avg_profit_per_trade: float
    success_rate: float
    mev_attacks_prevented: int
    current_gas_price: int
    network_load: float
    active_opportunities: int
    pending_transactions: int
```

## WebSocket API

### Endpoints

#### `/ws`
```python
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    Messages:
        metrics: {
            "type": "metrics",
            "data": MetricsSnapshot
        }
        alerts: {
            "type": "alerts",
            "data": List[Dict]
        }
    """
```

### Message Types

#### Metrics Update
```json
{
    "type": "metrics",
    "data": {
        "timestamp": "2024-01-01T00:00:00Z",
        "total_profit": 1000.0,
        "total_trades": 100,
        "successful_trades": 95,
        "failed_trades": 5,
        "gas_spent": 10.5,
        "avg_profit_per_trade": 10.0,
        "success_rate": 0.95,
        "mev_attacks_prevented": 10,
        "current_gas_price": 50,
        "network_load": 0.6,
        "active_opportunities": 5,
        "pending_transactions": 2
    }
}
```

#### Alert Message
```json
{
    "type": "alerts",
    "data": [
        {
            "level": "warning",
            "source": "flash_loan",
            "message": "Low success rate: 80%"
        }
    ]
}
```

## Error Handling

### Error Types

#### ExecutionError
```python
class ExecutionError(Exception):
    """Flash loan execution error."""
    pass
```

#### ValidationError
```python
class ValidationError(Exception):
    """Opportunity validation error."""
    pass
```

#### MarketError
```python
class MarketError(Exception):
    """Market analysis error."""
    pass
```

### Error Responses

#### HTTP Errors
```json
{
    "error": "string",
    "detail": "string",
    "timestamp": "string"
}
```

#### WebSocket Errors
```json
{
    "type": "error",
    "error": "string",
    "detail": "string"
}
```

## Rate Limiting

- Maximum 10 requests per second per IP
- WebSocket connections limited to 100 per IP
- API key required for higher limits

## Authentication

### API Key
```http
Authorization: Bearer <api_key>
```

### WebSocket Authentication
```json
{
    "type": "auth",
    "api_key": "string"
}
```

## Examples

### Execute Flash Loan
```python
# Initialize services
flash_loan = FlashLoanExecutor(web3, market_data, strategy_optimizer)

# Create opportunity
opportunity = {
    "token_in": WETH,
    "amount_in": Web3.to_wei(10, "ether"),
    "path": [
        {
            "token_in": WETH,
            "token_out": USDC,
            "pool": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
        }
    ],
    "expected_profit": 100.0,
    "flash_loan_config": {
        "provider": "aave_v2",
        "token_address": WETH,
        "amount": Web3.to_wei(10, "ether"),
        "fee": 0.0009
    }
}

# Execute
result = await flash_loan.execute_flash_loan(
    opportunity,
    {"network_load": 0.6, "gas_price": 50}
)
```

### Monitor Executions
```python
# Initialize monitoring
monitoring = MonitoringService(config, services)

# Connect to WebSocket
async with websockets.connect("ws://localhost:8000/ws") as ws:
    while True:
        message = await ws.recv()
        data = json.loads(message)
        
        if data["type"] == "metrics":
            print(f"Profit: ${data['data']['total_profit']:.2f}")
        elif data["type"] == "alerts":
            for alert in data["data"]:
                print(f"Alert: {alert['message']}")
```
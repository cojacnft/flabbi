// Dashboard configuration
const config = {
    maxDataPoints: 100,
    updateInterval: 1000,
    chartOptions: {
        animation: false,
        responsive: true,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute'
                }
            }
        }
    }
};

// Initialize charts
const profitChart = new Chart(
    document.getElementById('profit-chart'),
    {
        type: 'line',
        data: {
            datasets: [{
                label: 'Total Profit',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            ...config.chartOptions,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Profit (USD)'
                    }
                }
            }
        }
    }
);

const networkChart = new Chart(
    document.getElementById('network-chart'),
    {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Gas Price (gwei)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    yAxisID: 'y1'
                },
                {
                    label: 'Network Load',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    yAxisID: 'y2'
                }
            ]
        },
        options: {
            ...config.chartOptions,
            scales: {
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Gas Price (gwei)'
                    }
                },
                y2: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Network Load'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    }
);

// WebSocket connection
let ws = null;
function connectWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = function(event) {
        const message = JSON.parse(event.data);
        
        if (message.type === 'metrics') {
            updateMetrics(message.data);
        } else if (message.type === 'alerts') {
            updateAlerts(message.data);
        }
    };
    
    ws.onclose = function() {
        setTimeout(connectWebSocket, 1000);
    };
}

// Update metrics
function updateMetrics(metrics) {
    // Update key metrics
    document.getElementById('total-profit').textContent = 
        `$${metrics.total_profit.toFixed(2)}`;
    document.getElementById('success-rate').textContent = 
        `${(metrics.success_rate * 100).toFixed(1)}%`;
    document.getElementById('active-opportunities').textContent = 
        metrics.active_opportunities;
    document.getElementById('gas-price').textContent = 
        `${metrics.current_gas_price} gwei`;
    
    // Update status metrics
    document.getElementById('total-trades').textContent = metrics.total_trades;
    document.getElementById('failed-trades').textContent = metrics.failed_trades;
    document.getElementById('mev-prevented').textContent = metrics.mev_attacks_prevented;
    
    // Update charts
    updateCharts(metrics);
}

function updateCharts(metrics) {
    const timestamp = new Date(metrics.timestamp);
    
    // Update profit chart
    profitChart.data.datasets[0].data.push({
        x: timestamp,
        y: metrics.total_profit
    });
    
    if (profitChart.data.datasets[0].data.length > config.maxDataPoints) {
        profitChart.data.datasets[0].data.shift();
    }
    
    profitChart.update('none');
    
    // Update network chart
    networkChart.data.datasets[0].data.push({
        x: timestamp,
        y: metrics.current_gas_price
    });
    
    networkChart.data.datasets[1].data.push({
        x: timestamp,
        y: metrics.network_load
    });
    
    if (networkChart.data.datasets[0].data.length > config.maxDataPoints) {
        networkChart.data.datasets[0].data.shift();
        networkChart.data.datasets[1].data.shift();
    }
    
    networkChart.update('none');
}

function updateAlerts(alerts) {
    const alertContainer = document.getElementById('alerts');
    
    alerts.forEach(alert => {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${alert.level} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        
        alertDiv.innerHTML = `
            ${alert.message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        alertContainer.prepend(alertDiv);
    });
    
    // Remove old alerts
    while (alertContainer.children.length > 10) {
        alertContainer.removeChild(alertContainer.lastChild);
    }
}

// Initialize dashboard
async function initDashboard() {
    try {
        // Get initial metrics
        const response = await fetch('/api/metrics');
        const metrics = await response.json();
        updateMetrics(metrics);
        
        // Get system status
        const statusResponse = await fetch('/api/status');
        const status = await statusResponse.json();
        
        document.getElementById('uptime').textContent = status.uptime;
        
        // Update status badge
        const statusBadge = document.getElementById('status-badge').querySelector('.badge');
        statusBadge.className = `badge bg-${status.status === 'running' ? 'success' : 'danger'}`;
        statusBadge.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
        
        // Connect WebSocket
        connectWebSocket();
        
    } catch (error) {
        console.error('Error initializing dashboard:', error);
    }
}

// Start dashboard
document.addEventListener('DOMContentLoaded', initDashboard);
#!/bin/bash

echo "=== Fraud Detection MVP - Environment Check ==="

# Docker
echo -e "\n[1/4] Docker Containers:"
docker ps --filter "name=fraud" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# GPU
echo -e "\n[2/4] GPU Status:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Python
echo -e "\n[3/4] Python Environment:"
cd ~/fraud-detection-mvp
source .venv/bin/activate 2>/dev/null
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except:
    print('PyTorch not installed')
"

# Services
echo -e "\n[4/4] Service Health:"
echo -n "Redpanda: "; curl -s -o /dev/null -w "%{http_code}" http://localhost:9644/v1/status 2>/dev/null || echo "DOWN"
echo -n "Prometheus: "; curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/-/healthy 2>/dev/null || echo "DOWN"
echo -n "Grafana: "; curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null || echo "DOWN"

echo -e "\n=== URLs ==="
echo "Grafana:      http://$(curl -s ifconfig.me):3000 (admin/admin)"
echo "Prometheus:   http://$(curl -s ifconfig.me):9090"
echo "Redpanda UI:  http://$(curl -s ifconfig.me):9644"

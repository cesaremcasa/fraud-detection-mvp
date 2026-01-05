Real-Time Fraud Detection with Deep Learning
Show Image
Show Image
Show Image
Show Image
About
Traditional fraud detection systems operate in batch mode, processing transactions hours after they occur. This delay creates a critical window for financial losses as fraudulent transactions accumulate undetected.
This system detects fraud in taxi transactions (NYC Dataset) in real-time. Transactions arrive via API, flow through Kafka (Redpanda), and are analyzed by a GPU-accelerated PyTorch worker using an Autoencoder. Normal rides reconstruct with low error, while fraudulent patterns (impossible distances, extreme fares) produce high reconstruction error and trigger alerts. The system achieves 2.5ms average latency.
Architecture
FastAPI → Redpanda (Kafka) → PyTorch Worker (GPU) → Grafana
Data Flow:

Transactions arrive via FastAPI endpoint
Events stream through Redpanda
PyTorch worker runs GPU inference using Autoencoder
High reconstruction error indicates fraud
Metrics flow to Prometheus and Grafana

Technology Stack:

FastAPI
Redpanda (Kafka)
PyTorch with CUDA
NVIDIA L4 GPU
Prometheus + Grafana
Docker Compose

Performance Metrics

Average Latency: 2.5ms (target: < 100ms)
Throughput: Thousands of transactions per second
Detection Method: Unsupervised Autoencoder (no labels required)
GPU: NVIDIA L4

Prerequisites

Docker & Docker Compose
NVIDIA GPU with CUDA support (or CPU for debug)
Python 3.10+

Quick Start
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/real-time-fraud-detection.git
cd real-time-fraud-detection

# Start all services
docker-compose up -d

# Run smoke test
docker-compose exec api python tests/smoke_test.py

# Access Grafana dashboard
open http://localhost:3000
Project Structure
real-time-fraud-detection/
├── src/
│   ├── api/
│   ├── worker/
│   └── training/
├── docker/
│   └── docker-compose.yml
├── infra/
│   ├── prometheus/
│   └── grafana/
├── tests/
└── README.md
License
MIT License
Copyright (c) 2025 Cesar Augusto
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Cesar Augusto
Contact: cesardonahill3@gmail.com
Location: Winter Garden, Florida

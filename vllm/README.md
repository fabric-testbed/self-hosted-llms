# Multi-Model vLLM Gateway with LiteLLM and NGINX

> Deploy multiple vLLM model servers behind a unified HTTPS gateway with LiteLLM routing on NVIDIA DGX Spark

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Management](#management)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is this?

This setup provides a production-ready multi-model inference gateway that runs multiple vLLM model servers with LiteLLM as the model routing and management layer, behind an NGINX reverse proxy for HTTPS termination. Each model runs in its own container with dedicated resources:

- **NGINX**: HTTPS termination and SSL/TLS certificates
- **LiteLLM**: Model routing, load balancing, caching, cost tracking, and API management
- **vLLM**: High-performance model inference servers

Key features:
- **Unified HTTPS endpoint** with SSL/TLS certificates via NGINX
- **Intelligent model routing** via LiteLLM to different models (`gpt-oss-20b`, `gpt-oss-120b`, `qwen-30b`, `glm-4.7-flash`)
- **Load balancing** across multiple model instances
- **Request caching** with Redis for repeated queries
- **Cost tracking** and rate limiting per API key
- **Admin UI** for monitoring and configuration
- **Streaming optimization** for real-time inference

### What you'll accomplish

- Deploy multiple LLM inference servers concurrently on DGX Spark
- Configure HTTPS access with SSL/TLS certificates
- Route requests to different models through LiteLLM's unified API
- Monitor and manage multiple model servers from LiteLLM's Admin UI

### What to know before starting

- Experience with Docker Compose for multi-container applications
- Understanding of reverse proxy and API gateway concepts
- Familiarity with OpenAI-compatible API endpoints
- Knowledge of SSL/TLS certificates (self-signed and production)
- Experience with environment variable configuration

## Prerequisites

- **DGX Spark** with Blackwell GPU architecture (128GB unified memory)
- **Docker** installed and configured: `docker --version` succeeds
- **Docker Compose** v2.0+: `docker compose version` succeeds
- **NVIDIA Container Toolkit** installed
- **HuggingFace account** with access tokens for gated models
- **Network access** to pull container images and models
- **Sufficient GPU memory** for running multiple models concurrently

### GPU Memory Planning

Approximate memory requirements per model:
- **GPT-OSS-20B**: ~20GB GPU memory
- **GPT-OSS-120B**: ~60GB GPU memory (with FP8 KV cache)
- **Qwen-30B**: ~30GB GPU memory (with FP8 KV cache)
- **GLM-4.7-Flash**: ~62GB GPU memory (31B MoE, only 3B active per token)

DGX Spark's 128GB unified memory allows running 1-2 large models or 2-3 smaller models concurrently.

## Time & Risk

- **Duration**: 15-20 minutes for initial setup
- **Risks**: Minimal - all services run in isolated containers
- **Rollback**: Simple container removal, no system-level changes
- **Last Updated**: 2025-12-19

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     NGINX Gateway                           │
│  HTTPS (443) / HTTP (80)                                    │
│  - SSL/TLS termination                                      │
│  - Proxies all requests to LiteLLM                          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  LiteLLM Proxy (Port 4000)                  │
│  - Model routing (gpt-oss-20b, gpt-oss-120b, qwen-30b)     │
│  - Load balancing across instances                          │
│  - Request caching (Redis)                                  │
│  - Cost tracking (PostgreSQL)                               │
│  - Rate limiting & API key management                       │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼──────┐  ┌──────────▼─────────┐  ┌──────▼────────┐
│ vLLM Server  │  │  vLLM Server       │  │ vLLM Server   │
│ GPT-OSS-20B  │  │  GPT-OSS-120B      │  │  Qwen-30B     │
│ Port: 8001   │  │  Port: 8002        │  │  Port: 8003   │
│ Container:   │  │  Container:        │  │  Container:   │
│ GPU access   │  │  GPU access        │  │  GPU access   │
└──────────────┘  └────────────────────┘  └───────────────┘
```

### Network Architecture

All containers join the `vllm_network` bridge network, allowing:
- Inter-container communication by hostname
- NGINX to proxy requests to LiteLLM
- LiteLLM to route requests to backend vLLM servers
- Shared HuggingFace model cache across containers

**Note**: The `vllm_network` is created automatically when you start the first model server. For explicit control, you can create it manually:
```bash
docker network create vllm_network
```

### Core Components

**LiteLLM Proxy** (`litellm/`): The central API gateway that provides:
- **Model routing**: Routes requests to appropriate vLLM backend based on model name
- **Load balancing**: Distributes requests across multiple model instances
- **Request caching**: Redis-based caching for repeated queries
- **Cost tracking**: PostgreSQL-based analytics and usage monitoring
- **Rate limiting**: Per API key rate limiting
- **Admin UI**: Web-based dashboard for monitoring and configuration

See `litellm/README.md` for detailed configuration options.

### File Structure

```
llms/vllm/
├── README.md                    # This file
├── docker-compose.yml           # NGINX gateway configuration
├── certs.sh                     # SSL certificate generation script
├── test-endpoints.sh            # Endpoint testing script
├── sync-env.sh                  # Sync environment variables to model directories
├── .env.example                 # Environment variable template
├── .env                         # Your environment configuration (create from .env.example)
├── ssl/                         # Generated SSL certificates (gitignored)
│   ├── public.pem              # Self-signed certificate
│   └── private.pem             # Private key
├── nginx/
│   └── multi-model.conf        # NGINX config (proxies to LiteLLM)
├── gpt-oss-20b/
│   ├── docker-compose.yml      # GPT-OSS-20B vLLM server (default)
│   ├── docker-compose-gpu.yml  # GPT-OSS-20B with dedicated GPU 0
│   └── chat.template           # Chat template for GPT-OSS models
├── gpt-oss-120b/
│   ├── docker-compose.yml      # GPT-OSS-120B vLLM server (default)
│   ├── docker-compose-gpu.yml  # GPT-OSS-120B with dedicated GPUs 0,1,2
│   └── chat.template           # Chat template for GPT-OSS models
├── qwen-30b/
│   ├── docker-compose.yml      # Qwen-30B vLLM server (default)
│   ├── docker-compose-gpu.yml  # Qwen-30B with dedicated GPUs 1,2
│   └── chat.template           # Chat template for Qwen models
├── glm-4.7-flash/
│   ├── docker-compose.yml      # GLM-4.7-Flash vLLM server (default)
│   └── docker-compose-gpu.yml  # GLM-4.7-Flash with dedicated GPU 0
└── litellm/                    # LiteLLM API gateway (handles model routing)
    ├── README.md               # LiteLLM documentation
    ├── docker-compose.yml      # LiteLLM proxy + Redis + PostgreSQL
    ├── config.yaml             # Model routing and caching configuration
    ├── quick-start.sh          # LiteLLM deployment script
    └── test-litellm.sh         # LiteLLM endpoint tests
```

---

## Manual Deployment

This section provides step-by-step manual deployment instructions for DGX Spark systems.

### Step 1. Create Docker Network

Create the shared Docker network that all vLLM services will use:

```bash
# Create the network
docker network create vllm_network

# Verify network was created
docker network ls | grep vllm_network
```

### Step 2. Generate SSL Certificates

Create self-signed certificates for HTTPS access:

```bash
cd llms/vllm
./certs.sh
```

This creates `ssl/public.pem` and `ssl/private.pem` with 10-year validity for localhost.

**For production**: Replace self-signed certificates with CA-issued certificates:
```bash
# Replace generated files with your production certificates
cp /path/to/your/cert.pem ssl/public.pem
cp /path/to/your/key.pem ssl/private.pem
```

### Step 3. Configure HuggingFace Authentication

Create a `.env` file from the template and set your HuggingFace token:

```bash
cd llms/vllm

# Copy the example environment file
cp .env.example .env

# Edit .env and set your HF_TOKEN
# Get your token from: https://huggingface.co/settings/tokens
nano .env  # or use your preferred editor
```

Your `.env` file should look like:
```bash
# llms/vllm/.env
HF_TOKEN=hf_your_actual_token_here
VLLM_LOGGING_LEVEL=INFO
NGINX_HTTPS_PORT=443
NGINX_HTTP_PORT=80
```

### Step 4. Sync Environment Variables

Propagate environment variables to each model directory:

```bash
cd llms/vllm
./sync-env.sh
```

This creates synchronized `.env` files in each model directory (`gpt-oss-120b/.env`, `qwen-30b/.env`, etc.) that are auto-generated and should not be edited directly.

### Step 5. Start Model Servers

Launch the models you want to deploy. Each model directory contains two docker-compose files:
- **`docker-compose.yml`**: Default configuration for DGX Spark (uses all available GPUs)
- **`docker-compose-gpu.yml`**: Explicit GPU assignment for multi-GPU systems

**DGX Spark Deployment (Recommended)**

Use the default `docker-compose.yml` which leverages DGX Spark's unified memory architecture:

```bash
cd llms/vllm

# Option A: Start all models (requires ~110GB+ GPU memory)
cd gpt-oss-20b && docker compose up -d && cd ..
cd gpt-oss-120b && docker compose up -d && cd ..
cd qwen-30b && docker compose up -d && cd ..
cd glm-4.7-flash && docker compose up -d && cd ..

# Option B: Start specific model (e.g., GPT-OSS-120B only - recommended)
cd gpt-oss-120b
docker compose up -d
cd ..
```

**Multi-GPU Systems with Explicit GPU Assignment**

For systems where you want to assign specific GPUs to each model, use `docker-compose-gpu.yml`:

```bash
cd llms/vllm

# Start GPT-OSS-120B on GPUs 0,1,2 with tensor parallelism
cd gpt-oss-120b
docker compose -f docker-compose-gpu.yml up -d
cd ..

# Start Qwen-30B on GPUs 1,2
cd qwen-30b
docker compose -f docker-compose-gpu.yml up -d
cd ..

# Start GPT-OSS-20B on GPU 0
cd gpt-oss-20b
docker compose -f docker-compose-gpu.yml up -d
cd ..
```

**Monitor Model Loading**

Models take 2-5 minutes to load into memory. Monitor progress:

```bash
# Watch GPT-OSS-120B loading
docker logs -f vllm-gpt-oss-120b

# Check GPU memory usage
nvidia-smi

# Wait for model to be ready (shows "Application startup complete" when ready)
```

### Step 6. Start LiteLLM Gateway

After at least one model server is running and healthy, start LiteLLM:

```bash
cd llms/vllm/litellm
./quick-start.sh

# Or manually:
docker compose up -d
```

This starts the LiteLLM proxy that routes requests to your model servers.

### Step 7. Start NGINX Gateway

After LiteLLM is running, start the NGINX reverse proxy:

```bash
cd llms/vllm
docker compose up -d
```

This starts NGINX which provides HTTPS termination and proxies all requests to LiteLLM.

### Step 8. Verify Deployment

Check all services are healthy:

```bash
# Check all running containers
docker ps

# Verify containers are healthy (STATUS column should show "healthy")
docker ps --filter "name=vllm-"

# Check NGINX logs
docker logs vllm-nginx-proxy

# Verify health endpoint via NGINX
curl -k https://localhost/health

# Verify LiteLLM directly
curl http://localhost:4000/health
```

Expected output: `healthy`

### Step 9. Test Model Inference

Test the gateway with a sample request:

```bash
# Test using the provided script
cd llms/vllm
./test-endpoints.sh

# Or manually test GPT-OSS-120B via HTTPS (through NGINX → LiteLLM)
curl -k https://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Step 10. Shutdown (When Done)

Gracefully stop all services in reverse order:

```bash
cd llms/vllm

# Stop NGINX gateway first
docker compose down

# Stop LiteLLM
cd litellm && docker compose down && cd ..

# Stop model servers
cd gpt-oss-120b && docker compose down && cd ..
cd gpt-oss-20b && docker compose down && cd ..
cd qwen-30b && docker compose down && cd ..
cd glm-4.7-flash && docker compose down && cd ..

# Optional: Remove network (only if no other services are using it)
docker network rm vllm_network

# Optional: Remove volumes to free disk space (WARNING: deletes cached models)
docker volume rm gpt-oss-120b_huggingface_cache
docker volume rm gpt-oss-20b_huggingface_cache
docker volume rm qwen-30b_huggingface_cache
docker volume rm glm-4.7-flash_huggingface_cache
```

---

## Configuration

### Docker Compose Files

Each model directory contains two docker-compose files:

#### docker-compose.yml (Default - DGX Spark)

The default configuration for DGX Spark systems:
- Uses all available GPUs (`count: all`)
- Leverages DGX Spark's Unified Memory Architecture (UMA)
- Recommended for single-system deployments
- Simplest configuration

Example GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all available GPUs
          capabilities: [gpu]
```

#### docker-compose-gpu.yml (Explicit GPU Assignment)

For multi-GPU systems requiring explicit GPU assignment:
- Assigns specific GPU IDs to each model
- Useful for isolating workloads
- Required when running multiple models with different GPU requirements
- Uses tensor parallelism for models across multiple GPUs

Example GPU configuration:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1', '2']  # Specific GPUs
          capabilities: [gpu]
```

**When to use docker-compose-gpu.yml:**
- Running multiple models on separate GPUs
- Need guaranteed GPU isolation between models
- Using tensor parallelism with specific GPU topology
- Debugging GPU-specific issues

**Usage:**
```bash
# Default (DGX Spark)
docker compose up -d

# Explicit GPU assignment
docker compose -f docker-compose-gpu.yml up -d
```

### Model-Specific Settings

Each model directory contains model-specific configuration:

#### GPT-OSS-120B Configuration

Located in `gpt-oss-120b/docker-compose.yml`:

```yaml
command: >
  vllm serve openai/gpt-oss-120b
    --enable-auto-tool-choice      # Enable automatic tool selection
    --tool-call-parser openai      # Use OpenAI tool call format
    --chat-template /root/chat.template
    --tensor-parallel-size 1       # Single GPU (change for multi-GPU)
    --max-model-len 4096          # Maximum context length
    --kv-cache-dtype fp8          # Use FP8 for KV cache compression
    --trust-remote-code           # Allow remote code execution
```

**Key parameters to adjust:**
- `--max-model-len`: Reduce if memory constrained (default: 4096)
- `--tensor-parallel-size`: Increase for multi-GPU setups
- `--kv-cache-dtype`: Options are `auto`, `fp8`, `fp16` (FP8 saves ~50% memory)
- `--gpu-memory-utilization`: Default 0.9, increase to 0.95 for maximum throughput

#### GLM-4.7-Flash Configuration

Located in `glm-4.7-flash/docker-compose.yml`:

```yaml
command: >
  vllm serve zai-org/GLM-4.7-Flash
    --enable-auto-tool-choice        # Enable automatic tool selection
    --tool-call-parser glm47         # GLM-4.7 tool call format
    --reasoning-parser glm45         # GLM reasoning/thinking mode
    --speculative-config.method mtp  # Multi-token prediction for speed
    --speculative-config.num_speculative_tokens 1
    --tensor-parallel-size 1         # Single GPU
    --max-model-len 8192            # Maximum context length
    --kv-cache-dtype fp8            # Use FP8 for KV cache compression
    --trust-remote-code             # Allow remote code execution
```

**Key parameters to adjust:**
- `--max-model-len`: Increase up to 131072 for full context (needs more memory)
- `--gpu-memory-utilization`: Default 0.85, increase to 0.95 for maximum throughput
- Remove `--speculative-config.*` flags if you hit compatibility issues

#### Qwen-30B Configuration

Located in `qwen-30b/docker-compose.yml`:

```yaml
command: >
  vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct
    --enable-auto-tool-choice
    --tool-call-parser qwen3_code  # Qwen-specific parser
    --chat-template /root/chat.template
    --tensor-parallel-size 1
    --max-model-len 8192          # Higher context for code tasks
    --kv-cache-dtype fp8
    --trust-remote-code
```

### NGINX Configuration

The NGINX configuration (`nginx/multi-model.conf`) provides HTTPS termination and proxies all requests to LiteLLM:

#### Proxy to LiteLLM

```nginx
upstream litellm-backend {
    server vllm-litellm-proxy:4000;
    keepalive 32;
}

location / {
    proxy_pass http://litellm-backend;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

**Note**: Model routing is handled by LiteLLM, not NGINX. To add or modify model routes, edit `litellm/config.yaml`.

#### Streaming Optimization

```nginx
proxy_buffering off;           # Disable buffering for streaming
proxy_http_version 1.1;        # HTTP/1.1 for keepalive
proxy_set_header Connection "";# Persistent connections
```

#### Timeout Settings

Default timeouts are set to 30 minutes (1800 seconds) to accommodate:
- Long-running tool-calling sequences
- Complex reasoning tasks
- Large context processing

```nginx
proxy_read_timeout 1800;
```

### Environment Variables

Create `.env` file in `llms/vllm/` directory:

```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional - NGINX ports
NGINX_HTTPS_PORT=443
NGINX_HTTP_PORT=80

# Optional - vLLM logging
VLLM_LOGGING_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR

# Optional - vLLM optimizations (set in docker-compose.yml)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_USE_V1=1  # Enable vLLM v1 engine
```

### GPU Assignment

By default, all models use all available GPUs. To assign specific GPUs:

Edit the model's `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Uncomment and specify GPU ID
          capabilities: [gpu]
```

### VM Deployment with Dedicated GPU Allocation

For VM deployments where you need to run multiple models concurrently with dedicated GPU assignments, use the `docker-compose-gpu.yml` files instead of the standard `docker-compose.yml` files.

#### Available Configurations

Each model directory contains a `docker-compose-gpu.yml` with pre-configured GPU assignments:

**gpt-oss-20b/docker-compose-gpu.yml**:
- GPU Assignment: GPU 0
- Tensor Parallel Size: 1
- Memory: ~20GB

**gpt-oss-120b/docker-compose-gpu.yml**:
- GPU Assignment: GPUs 0, 1, 2
- Tensor Parallel Size: 3
- Memory: ~60GB (distributed across 3 GPUs)

**qwen-30b/docker-compose-gpu.yml**:
- GPU Assignment: GPUs 1, 2
- Tensor Parallel Size: 2
- Memory: ~30GB (distributed across 2 GPUs)

#### Deployment Scenarios

The GPU allocations support two non-overlapping deployment scenarios:

**Scenario 1: GPT-OSS-120B alone** (for reasoning-heavy workloads):
```bash
cd llms/vllm/gpt-oss-120b
docker compose -f docker-compose-gpu.yml up -d
```
Uses: GPUs 0, 1, 2

**Scenario 2: GPT-OSS-20B + Qwen-30B together** (for diverse workload mix):
```bash
cd llms/vllm/gpt-oss-20b
docker compose -f docker-compose-gpu.yml up -d

cd ../qwen-30b
docker compose -f docker-compose-gpu.yml up -d
```
Uses: GPU 0 (20B model) + GPUs 1, 2 (30B model) - no conflicts

#### When to Use docker-compose-gpu.yml

Use `docker-compose-gpu.yml` when:
- Running on VMs with multiple GPUs (not DGX Spark with UMA)
- You need to run multiple models concurrently
- You want explicit GPU resource isolation between models
- Running in environments where GPU sharing is not optimal

Use standard `docker-compose.yml` when:
- Running on DGX Spark with Unified Memory Architecture
- Running only one model at a time
- You want dynamic GPU allocation

#### Starting Services with GPU Configs

```bash
cd llms/vllm

# Sync environment variables first
./sync-env.sh

# Start NGINX gateway
docker compose up -d

# Start models with GPU assignment (choose your scenario)

# Scenario 1: 120B model alone
cd gpt-oss-120b
docker compose -f docker-compose-gpu.yml up -d

# OR Scenario 2: 20B + 30B models together
cd gpt-oss-20b
docker compose -f docker-compose-gpu.yml up -d
cd ../qwen-30b
docker compose -f docker-compose-gpu.yml up -d
```

#### Verifying GPU Allocation

Check that each container is using its assigned GPUs:

```bash
# View GPU allocation for each container
docker exec vllm-gpt-oss-120b nvidia-smi

# Check all GPU usage across system
nvidia-smi

# View which GPUs are assigned to each container
docker inspect vllm-gpt-oss-120b | grep -A 10 DeviceRequests
```

---

## Usage Examples

All requests go through a single unified endpoint. Model selection is done via the `model` field in the request body, and LiteLLM routes to the appropriate backend.

### List Available Models

```bash
curl -k https://localhost/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq
```

Response:
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-oss-20b", "object": "model"},
    {"id": "gpt-oss-120b", "object": "model"},
    {"id": "qwen-30b", "object": "model"},
    {"id": "glm-4.7-flash", "object": "model"}
  ]
}
```

### Chat Completion

```bash
curl -k https://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq .
```

### Streaming Response

```bash
curl -k https://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "qwen-30b",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
    ],
    "stream": true,
    "max_tokens": 200
  }'
```

### Tool Calling (Function Calling)

```bash
curl -k https://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather in a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | jq .
```

### Python Client Example

```python
from openai import OpenAI
import os

# Configure client to use local gateway via HTTPS
client = OpenAI(
    base_url="https://localhost/v1",
    api_key=os.environ.get("LITELLM_MASTER_KEY", "sk-your-key"),
)

# Disable SSL verification for self-signed certificates
import httpx
client._client = httpx.Client(verify=False)

# Make request - specify model in the request
response = client.chat.completions.create(
    model="gpt-oss-120b",  # LiteLLM routes to correct backend
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Docker in simple terms."}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### Using Different Models

Switch models by changing the `model` field in your request:

```python
from openai import OpenAI
import os

# Single client for all models
client = OpenAI(
    base_url="https://localhost/v1",
    api_key=os.environ.get("LITELLM_MASTER_KEY"),
)

# Use GPT-OSS-120B (best for reasoning and tool calling)
response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[{"role": "user", "content": "Solve this math problem..."}]
)

# Use Qwen-30B (best for code generation)
response = client.chat.completions.create(
    model="qwen-30b",
    messages=[{"role": "user", "content": "Write a Python function..."}]
)

# Use GPT-OSS-20B (fastest, smaller model)
response = client.chat.completions.create(
    model="gpt-oss-20b",
    messages=[{"role": "user", "content": "Quick question..."}]
)
```

---

## Management

### View Logs

```bash
# NGINX gateway logs
docker logs vllm-nginx-proxy -f

# LiteLLM proxy logs
docker logs vllm-litellm-proxy -f

# Model server logs
docker logs vllm-gpt-oss-120b -f
docker logs vllm-qwen-30b -f
docker logs vllm-gpt-oss-20b -f
docker logs vllm-glm-4-7-flash -f
```

### Monitor Resource Usage

```bash
# GPU memory usage
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Container resource usage
docker stats

# Detailed GPU metrics per container
docker exec vllm-gpt-oss-120b nvidia-smi
```

### Stop Services

Stop services in reverse order (NGINX first, then LiteLLM, then models):

```bash
cd llms/vllm

# Stop NGINX gateway first
docker compose down

# Stop LiteLLM
cd litellm && docker compose down && cd ..

# Stop model servers (default docker-compose.yml)
cd gpt-oss-120b && docker compose down && cd ..
cd qwen-30b && docker compose down && cd ..
cd gpt-oss-20b && docker compose down && cd ..

# If using docker-compose-gpu.yml files
cd gpt-oss-120b && docker compose -f docker-compose-gpu.yml down && cd ..
cd qwen-30b && docker compose -f docker-compose-gpu.yml down && cd ..
cd gpt-oss-20b && docker compose -f docker-compose-gpu.yml down && cd ..
cd glm-4.7-flash && docker compose -f docker-compose-gpu.yml down && cd ..

# Or stop specific services without removing containers
docker stop vllm-nginx-proxy
docker stop vllm-litellm-proxy
docker stop vllm-gpt-oss-120b
docker stop vllm-qwen-30b
docker stop vllm-gpt-oss-20b
docker stop vllm-glm-4-7-flash

# Optional: Remove network (only if no other services using it)
docker network rm vllm_network

# Optional: Remove volumes to free disk space (WARNING: deletes cached models)
docker volume rm gpt-oss-120b_huggingface_cache
docker volume rm gpt-oss-20b_huggingface_cache
docker volume rm qwen-30b_huggingface_cache
```

### Restart Services

```bash
# Restart NGINX (for config changes)
docker restart vllm-nginx-proxy

# Restart model server
docker restart vllm-gpt-oss-120b
```

### Update Model Configuration

1. Edit the model's `docker-compose.yml`
2. Recreate the container:

```bash
cd llms/vllm/gpt-oss-120b
docker compose up -d --force-recreate
```

### Add New Model

1. Create new model directory:

```bash
mkdir llms/vllm/my-model
cp llms/vllm/gpt-oss-120b/docker-compose.yml llms/vllm/my-model/
```

2. Edit `llms/vllm/my-model/docker-compose.yml`:
   - Change container name (e.g., `vllm-my-model`)
   - Update port mapping if needed
   - Modify vLLM serve command with new model

3. Add the model to LiteLLM's `litellm/config.yaml`:

```yaml
model_list:
  # ... existing models ...

  - model_name: my-model
    litellm_params:
      model: openai/my-model-name
      api_base: http://vllm-my-model:8000/v1
      api_key: "not-used"
```

4. Start the model server and restart LiteLLM:

```bash
# Start the new model
cd llms/vllm/my-model && docker compose up -d && cd ..

# Restart LiteLLM to pick up config changes
cd litellm && docker compose restart litellm && cd ..
```

**Note**: NGINX configuration does not need to change when adding models. All model routing is handled by LiteLLM.

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `curl: (60) SSL certificate problem` | Self-signed certificate not trusted | Use `-k` flag with curl, or install certificate in system trust store |
| `502 Bad Gateway` from NGINX | LiteLLM or vLLM server not running | Check `docker ps`, verify LiteLLM and model containers are running |
| `504 Gateway Timeout` | Model loading or inference taking too long | Increase NGINX timeout in `multi-model.conf`, check vLLM logs for errors |
| `401 Unauthorized` | Invalid or missing LiteLLM API key | Check `Authorization: Bearer` header matches `LITELLM_MASTER_KEY` in .env |
| `No models available` | All vLLM servers down | Verify vLLM containers are running: `docker ps \| grep vllm` |
| `CUDA Out of Memory` | Multiple large models exceeding GPU capacity | Stop some models, reduce `--max-model-len`, or use `--kv-cache-dtype fp8` |
| `Network vllm_network not found` | Network not created | Create network: `docker network create vllm_network` |
| `Cannot access gated repo` | HuggingFace token invalid or missing | Run `./sync-env.sh` to propagate HF_TOKEN, verify token in .env file |
| Container fails to start | Port already in use | Change port mapping in docker-compose.yml or stop conflicting service |
| Model download fails | Network issues or authentication | Check HF_TOKEN in .env, run `./sync-env.sh`, verify internet connectivity |
| `HF_TOKEN not passed to containers` | Environment variable not synchronized | Run `./sync-env.sh` to create .env files in model directories |
| LiteLLM connection refused | Model container hostname mismatch | Verify model container names in `litellm/config.yaml` match actual container names |

### Memory Management

DGX Spark uses Unified Memory Architecture (UMA). If encountering memory issues:

```bash
# Flush buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Check actual memory usage
free -h
nvidia-smi
```

### Health Check Debugging

```bash
# Check NGINX health (proxied through LiteLLM)
curl -k https://localhost/health

# Check LiteLLM health directly
curl http://localhost:4000/health

# Check individual model server health
curl http://localhost:8001/health  # GPT-OSS-20B direct port
curl http://localhost:8002/health  # GPT-OSS-120B direct port
curl http://localhost:8003/health  # Qwen-30B direct port

# Check from inside LiteLLM container
docker exec -it vllm-litellm-proxy python3 -c "import urllib.request; print(urllib.request.urlopen('http://vllm-gpt-oss-20b:8000/v1/models').read().decode())"
```

### View Container Details

```bash
# Inspect NGINX configuration
docker exec vllm-nginx-proxy cat /etc/nginx/conf.d/default.conf

# Check NGINX error logs
docker exec vllm-nginx-proxy cat /var/log/nginx/error.log

# Check LiteLLM configuration
docker exec vllm-litellm-proxy cat /app/config.yaml

# Verify network connectivity
docker exec -it vllm-litellm-proxy python3 -c "import urllib.request; print(urllib.request.urlopen('http://vllm-gpt-oss-20b:8000/v1/models').read().decode())"
```

### Certificate Issues

If HTTPS is not working:

```bash
# Verify certificates exist
ls -la llms/vllm/ssl/

# Regenerate certificates
cd llms/vllm
./certs.sh

# Check certificate details
openssl x509 -in ssl/public.pem -text -noout

# Restart NGINX to reload certificates
docker restart vllm-nginx-proxy
```

### Performance Tuning

For optimal performance:

1. **Memory optimization**: Use FP8 KV cache (`--kv-cache-dtype fp8`)
2. **Context length**: Set `--max-model-len` based on your needs (lower = more memory for batching)
3. **GPU utilization**: Increase `--gpu-memory-utilization` to 0.95 for maximum throughput
4. **Batching**: Adjust `--max-num-batched-tokens` for your workload
5. **Connection pooling**: NGINX keepalive already configured (32 connections per backend)

---

## Production Deployment Considerations

### Security

- **Replace self-signed certificates** with CA-issued certificates from Let's Encrypt or your organization
- **Change default credentials**: Set strong `LITELLM_MASTER_KEY` and `LITELLM_UI_PASSWORD` in .env
- **API key management**: Use LiteLLM's Admin UI to create per-user API keys with rate limits
- **Firewall rules**: Restrict access to port 443/80 to authorized networks
- **Secrets management**: Use Docker secrets or external secret management (e.g., Vault)

### Monitoring

- **LiteLLM Admin UI**: Access `https://localhost/ui` for real-time monitoring and cost tracking
- **Metrics collection**: LiteLLM exposes Prometheus metrics at `/metrics` endpoint
- **Log aggregation**: Ship NGINX, LiteLLM, and vLLM logs to centralized logging (e.g., ELK stack)
- **Alerting**: Configure alerts for health check failures, high latency, or OOM errors

### High Availability

- **Load balancing**: Deploy multiple DGX Spark nodes behind a load balancer
- **Health checks**: Configure upstreams to mark failed backends as down
- **Graceful shutdown**: Use `docker compose stop` to allow in-flight requests to complete

### Model Updates

- **Blue-green deployment**: Start new model version on different port, update NGINX config, then stop old version
- **Rolling updates**: Update models one at a time to maintain availability
- **Model versioning**: Use explicit model version tags in HuggingFace paths

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [NGINX Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)
- [Docker Compose Networking](https://docs.docker.com/compose/networking/)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Support

For issues related to:
- **vLLM**: https://github.com/vllm-project/vllm/issues
- **NGINX**: https://nginx.org/en/docs/
- **DGX Spark**: Contact NVIDIA Enterprise Support

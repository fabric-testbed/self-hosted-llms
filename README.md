# LLM Inference Services

This directory contains production-ready LLM inference deployments optimized for NVIDIA DGX Spark and multi-GPU VMs.

## Contents

### [vLLM Multi-Model Gateway](vllm/)

A production-ready multi-model inference gateway that runs multiple vLLM model servers behind an NGINX reverse proxy with HTTPS support.

**Features:**
- Multiple models served concurrently, on one host or spread across several
- Unified HTTPS endpoint with OpenAI-compatible API
- Tool calling and reasoning-trace separation configured per model
- Health monitoring and load balancing
- Support for DGX Spark (UMA), multi-GPU VMs, and multi-node tensor parallelism

**Model configurations included:**

| directory | model | weights | nodes |
|---|---|---|---|
| [`vllm/qwen3.5-122b`](vllm/qwen3.5-122b) | Qwen3.5-122B-A10B NVFP4 | 75.6 GB | 1 |
| [`vllm/minimax-m2.7`](vllm/minimax-m2.7) | MiniMax-M2.7 230B-A10B NVFP4 | 135.5 GB | **2 (TP=2)** |
| [`vllm/glm-4.7-flash`](vllm/glm-4.7-flash) | GLM-4.7-Flash 30B-A3B NVFP4 | 20.4 GB | 1 |
| [`vllm/gemma-4-27b`](vllm/gemma-4-27b) | Gemma 4 26B-A4B (multimodal) | ~50 GB | 1 |
| [`vllm/nemotron-nano-30b`](vllm/nemotron-nano-30b) | Nemotron 3 Nano 30B-A3B | ~60 GB | 1 |
| `vllm/gpt-oss-20b`, `gpt-oss-120b`, `qwen-30b`, `qwen-14b-awq`, `phi-2` | earlier configs | varies | 1 |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              NGINX (HTTPS) → LiteLLM router              │
│        one OpenAI-compatible endpoint, many models       │
└─────────────────────────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────────────┐
      │                   │                           │
┌─────▼─────┐   ┌─────────▼────────┐   ┌──────────────▼──────────────┐
│ vLLM on   │   │  vLLM on         │   │  vLLM tensor-parallel        │
│ this host │   │  another node    │   │  across TWO nodes via Ray    │
│ (bridge)  │   │  (host:8000)     │   │  e.g. MiniMax-M2.7 at TP=2   │
└───────────┘   └──────────────────┘   └──────────────────────────────┘
```

LiteLLM reaches co-located models by container name and remote ones by
`http://<node-ip>:8000/v1`, so models can live on any mix of hosts.

## Prerequisites

- **Hardware**: DGX Spark or multi-GPU VM (minimum 3 GPUs recommended)
- **Software**:
  - Docker & Docker Compose v2.0+
  - NVIDIA Container Toolkit
  - CUDA 13.0+
- **Access**: HuggingFace account with token for gated models

## Getting Started

1. **Build the base image** (required — the model compose files reference it):
   ```bash
   cd vllm/base-image && docker build -t local/vllm:26.07-xg024 .
   ```
   This is NGC vLLM 26.07 plus xgrammar 0.2.4. Without it, tool calls return
   HTTP 500 — see [vllm/base-image/README.md](vllm/base-image/README.md).
   For multi-node models also build [`vllm/base-image-ray`](vllm/base-image-ray).

2. **Start a model**, e.g.:
   ```bash
   cd vllm/qwen3.5-122b && docker compose up -d
   ```
   Multi-node models use a cluster script instead of compose — see
   [vllm/minimax-m2.7/README.md](vllm/minimax-m2.7/README.md).

3. **Start the gateway** (LiteLLM + NGINX):
   ```bash
   cd vllm/litellm && docker compose up -d
   cd .. && ./certs.sh && docker compose up -d
   ```

4. **Access the gateway:**
   ```bash
   curl -k https://localhost/v1/models
   ```

## Documentation

- **[vLLM Multi-Model Gateway Documentation](vllm/README.md)** - Complete setup, configuration, and usage guide
- **[DGX Spark Deployment Notes](vllm/DGX-SPARK-DEPLOYMENT-NOTES.md)** - NVFP4 sizing, per-image requirements, multi-node networking, and the failure modes worth knowing before you hit them
- **[Multi-Node Models (Ray)](vllm/README.md#multi-node-models-ray--tensor-parallelism)** - running one model tensor-parallel across two nodes
- **[Two-Spark Cluster Guide](vllm/TWO-SPARK-CLUSTER.md)** - background on pairing two Sparks
- **[vLLM Official Docs](https://docs.vllm.ai/)** - vLLM framework documentation
- **[OpenAI API Reference](https://platform.openai.com/docs/api-reference)** - API compatibility reference

## Support

For issues or questions:
- Check the [vLLM Troubleshooting Guide](vllm/README.md#troubleshooting)
- Review [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- For DGX Spark specific issues, contact NVIDIA Enterprise Support
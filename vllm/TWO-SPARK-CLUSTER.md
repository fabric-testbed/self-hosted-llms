# Two-Spark Cluster Guide

> Using two NVIDIA DGX Sparks as a cluster for distributed model inference

## Table of Contents

- [Overview](#overview)
- [Physical Setup](#physical-setup)
- [Option 1: Single Large Model Spanning Both Sparks](#option-1-single-large-model-spanning-both-sparks)
- [Option 2: Multiple Independent Models (One Per Spark)](#option-2-multiple-independent-models-one-per-spark)
- [Option Comparison](#option-comparison)
- [FAQ: Can I Cluster 2 Sparks and Still Run Multiple Models?](#faq-can-i-cluster-2-sparks-and-still-run-multiple-models)
- [Known Issues](#known-issues)
- [References](#references)

---

## Overview

Two DGX Sparks can be used together in two ways:

1. **Tensor-parallel cluster** — Ray stitches both Sparks into a single compute pool; vLLM splits model weights across both GPUs using NCCL over the QSFP cable.
2. **Independent per-Spark** — Each Spark runs its own model(s); LiteLLM routes requests to the appropriate Spark.

## Physical Setup

Both Sparks must be connected via the **QSFP cable** (high-speed direct connect). Verify connectivity:

```bash
# From Spark 1, ping Spark 2's QSFP interface IP
ping <SPARK_2_QSFP_IP>
```

---

## Option 1: Single Large Model Spanning Both Sparks

With 2x 128GB = **256GB unified memory**, you can serve models that don't fit on one Spark.

### Step 1: Download the Ray Cluster Script (Both Nodes)

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```

### Step 2: Pull the vLLM Container (Both Nodes)

```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.11-py3
```

### Step 3: Start the Head Node (Spark 1)

```bash
export MN_IF_NAME=enp1s0f1np1   # your QSFP interface name
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

### Step 4: Join the Worker Node (Spark 2)

```bash
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<SPARK_1_QSFP_IP>

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

### Step 5: Verify the Cluster

```bash
VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status
# Should show 2 nodes
```

### Step 6: Serve a Model with Tensor Parallelism

```bash
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9'
```

### Step 7: Test Inference

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

### Ray Dashboard

Access at `http://<head-node-ip>:8265` for cluster monitoring.

### Models That Work Well Spanning 2 Sparks

| Model | Size | Notes |
|-------|------|-------|
| Llama 3.3 70B | ~140GB FP16 | Works well |
| DeepSeek-R1 70B | ~140GB FP16 | Works well |
| Qwen 72B | ~144GB FP16 | Works well |
| Llama 3.1 405B (AWQ-INT4) | ~200GB | Quantized, fits in 256GB |

---

## Option 2: Multiple Independent Models (One Per Spark)

Run different models on each Spark independently, with LiteLLM routing requests.

### Architecture

```
                    ┌──────────────────────┐
                    │   LiteLLM Gateway    │
                    │   (runs on Spark 1)  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
    ┌─────────▼──────────┐           ┌──────────▼─────────┐
    │     Spark 1        │           │     Spark 2        │
    │  gpt-oss-120b      │           │  qwen-30b          │
    │  (port 8002)       │           │  gpt-oss-20b       │
    │  ~60GB GPU         │           │  (ports 8003/8001) │
    └────────────────────┘           └────────────────────┘
```

### Deployment

**Spark 1** — Deploy your existing setup as-is (e.g., `gpt-oss-120b`):

```bash
cd vllm/gpt-oss-120b
docker compose up -d
```

**Spark 2** — Deploy smaller models:

```bash
cd vllm/qwen-30b
docker compose up -d

cd ../gpt-oss-20b
docker compose up -d
```

### LiteLLM Configuration

Update `litellm/config.yaml` to point at both Sparks:

```yaml
model_list:
  - model_name: gpt-oss-120b
    litellm_params:
      model: openai/gpt-oss-120b
      api_base: http://<SPARK_1_IP>:8002/v1
      api_key: "not-used"

  - model_name: qwen-30b
    litellm_params:
      model: openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
      api_base: http://<SPARK_2_IP>:8003/v1
      api_key: "not-used"

  - model_name: gpt-oss-20b
    litellm_params:
      model: openai/gpt-oss-20b
      api_base: http://<SPARK_2_IP>:8001/v1
      api_key: "not-used"
```

Then restart LiteLLM:

```bash
cd vllm/litellm
docker compose restart litellm
```

---

## Option Comparison

| Approach | What you get | Memory pool | Models |
|---|---|---|---|
| **Tensor-parallel cluster** | 1 large model spanning both | 256GB combined | 70B+ models in full precision, 405B quantized |
| **Independent per-Spark** | Multiple smaller models | 128GB each | Run 2-3 models simultaneously across both |
| **Hybrid** | Mix of both | Flexible | Large model on one, small models on the other |

---

## FAQ: Can I Cluster 2 Sparks and Still Run Multiple Models?

**Short answer: Technically possible, but not efficient and not recommended.**

### Why Splitting gpu-memory-utilization on a Ray Cluster Doesn't Work Well

When you run a model with `--tensor-parallel-size 2` via Ray, the vLLM process **claims the GPU on both Sparks**. Setting `--gpu-memory-utilization 0.5` reserves only 50% of each GPU, but:

- The Ray container still **owns the entire GPU device**
- You can't run a second vLLM instance inside the same Ray cluster with a different model
- Running a second vLLM container *outside* Ray on the same GPU leads to memory contention and CUDA conflicts

### What Happens with Split Utilization

```
Spark 1 (128GB)                    Spark 2 (128GB)
┌──────────────────────┐           ┌──────────────────────┐
│ Ray TP=2 Model A     │           │ Ray TP=2 Model A     │
│ gpu-mem-util: 0.5    │           │ gpu-mem-util: 0.5    │
│ (64GB used)          │           │ (64GB used)          │
├──────────────────────┤           ├──────────────────────┤
│ 64GB "free" but...   │           │ 64GB "free" but...   │
│ - CUDA context clash │           │ - CUDA context clash │
│ - No isolation       │           │ - No isolation       │
│ - OOM risk           │           │ - OOM risk           │
└──────────────────────┘           └──────────────────────┘
```

The "free" 64GB on each Spark is unreliable because:

- **No GPU isolation** — DGX Spark has 1 GPU per node, not multiple discrete GPUs
- **CUDA memory fragmentation** — two processes sharing one GPU fight over memory
- **KV cache starvation** — lower `gpu-memory-utilization` means smaller KV cache, which means fewer concurrent requests and shorter context lengths for *both* models
- **UMA complications** — Spark's unified memory makes memory accounting unpredictable when two processes compete

### Efficient Approaches (Ranked)

#### 1. Independent per-Spark (Best for multiple models)

Each Spark runs its own models independently. No Ray cluster needed.

```
Spark 1 (128GB)                    Spark 2 (128GB)
┌──────────────────────┐           ┌──────────────────────┐
│ gpt-oss-120b         │           │ qwen-30b  (~30GB)    │
│ gpu-mem-util: 0.9    │           │ gpu-mem-util: 0.45   │
│ (~60GB with FP8 KV)  │           ├──────────────────────┤
│                      │           │ gpt-oss-20b (~20GB)  │
│                      │           │ gpu-mem-util: 0.45   │
└──────────────────────┘           └──────────────────────┘
         LiteLLM routes to both Sparks
```

- Full GPU ownership per model
- Each model gets dedicated KV cache
- LiteLLM handles routing transparently

#### 2. Ray Cluster for One Big Model (Best for large models)

Dedicate both Sparks entirely to one model that needs >128GB.

```
Spark 1 (128GB) ◄──QSFP──► Spark 2 (128GB)
┌─────────────────────────────────────────────┐
│        Llama-3.3-70B (TP=2, FP16)           │
│        gpu-mem-util: 0.9                     │
│        256GB combined, full KV cache         │
└─────────────────────────────────────────────┘
```

- Maximum performance for one model
- Full memory available for KV cache = more concurrent users

#### 3. Hybrid — No Ray (Practical compromise)

One Spark runs a large model (quantized to fit), the other runs smaller models. No Ray needed.

```
Spark 1 (128GB)                    Spark 2 (128GB)
┌──────────────────────┐           ┌──────────────────────┐
│ Llama-3.3-70B        │           │ qwen-30b             │
│ (AWQ-INT4, ~35GB)    │           │ gpt-oss-20b          │
│ gpu-mem-util: 0.9    │           │ gpu-mem-util: 0.45   │
│ TP=1, single node    │           │ each                 │
└──────────────────────┘           └──────────────────────┘
```

### Summary Table

| Approach | Multiple models? | Efficient? | Recommended? |
|---|---|---|---|
| Ray TP=2 + split gpu-mem-util | Unreliable | No — memory contention, KV cache starved | **No** |
| Independent per-Spark | Yes | Yes — full GPU per model | **Yes** |
| Ray TP=2, single model | No (one only) | Yes — full 256GB | **Yes, for large models** |
| Hybrid (no Ray) | Yes | Yes | **Yes** |

**Recommendation:** Use independent per-Spark for multiple models. Only use the Ray cluster when you have a single model that *needs* more than 128GB. Splitting `gpu-memory-utilization` across a Ray cluster gives the worst of both worlds — reduced performance for every model with added instability.

---

## Known Issues

### Ray Cluster Stability

The two-Spark Ray cluster can be unstable. Some users report:
- One node dropping while the other's GPU goes to 100% forever
- Engine initialization failures with certain models and `--tensor-parallel-size 2`

### Workarounds

- The community [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) project has nightly-tested vLLM wheels and workarounds for stability issues.
- Ensure `NCCL_SOCKET_IFNAME` is set correctly to the QSFP interface.
- Use `RAY_memory_monitor_refresh_ms=0` to disable Ray's memory monitor which can cause false OOM kills on unified memory systems.

### QSFP Interface Name

The interface name may vary between Spark units. Common names:
- `enp1s0f1np1`
- `enP2p1s0f1np1`

Check with: `ip link show | grep -i 'enp\|enP'`

---

## References

- [NVIDIA DGX Spark - Stacked Sparks vLLM Playbook](https://build.nvidia.com/spark/vllm/stacked-sparks)
- [Community spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- [Two-Spark Cluster Stability Issue (NVIDIA Forums)](https://forums.developer.nvidia.com/t/two-spark-cluster-with-vllm-using-tensor-parallel-size-2-causes-one-node-to-drop-while-the-others-gpu-goes-100-forever/358755)
- [DeepSeek-R1-70B on Dual Sparks (Medium)](https://medium.com/@sarankannan2002/taming-giants-serving-deepseek-r1-70b-across-dual-nvidia-dgx-spark-nodes-0b3aca0c4988)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Ray Documentation](https://docs.ray.io/)

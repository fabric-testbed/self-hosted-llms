# MiniMax-M2.7-NVFP4 — two-node (TP=2) on DGX Spark

230B total / ~10B active, NVFP4, **135.5 GB of weights**. Too large for one Spark's
121 GB usable memory, so it runs tensor-parallel across **two** Sparks joined by the
direct QSFP cable. Verified serving 2026-08-09 with a 563,088-token KV cache at
196,608 context.

There is no `docker-compose.yml` here: this is a Ray cluster spanning two hosts, so it
is started with `start-cluster.sh` rather than compose.

## Prerequisites

1. Build the Ray-enabled image on **both** nodes (see [../base-image-ray](../base-image-ray)):
   ```bash
   cd vllm/base-image && docker build -t local/vllm:26.07-xg024 .
   cd ../base-image-ray && docker build -t local/vllm:26.07-xg024-ray .
   ```
2. Pre-download the weights on **both** nodes *before* stopping whatever they
   currently serve — it takes a long time and there is no reason to be down for it:
   ```bash
   docker run --rm -v $HOME/.cache/huggingface:/root/.cache/huggingface \
     -e HF_HOME=/root/.cache/huggingface --entrypoint python3 local/vllm:26.07-xg024-ray \
     -c 'from huggingface_hub import snapshot_download; snapshot_download("lukealonso/MiniMax-M2.7-NVFP4", max_workers=8)'
   ```
3. Firewall: Ray needs `--network host`, and **host networking is subject to ufw**
   (unlike Docker published ports, which bypass INPUT via DNAT/FORWARD). On each node:
   ```bash
   # on head: allow the worker, and the LiteLLM gateway to reach the API port
   sudo ufw allow from <WORKER_QSFP_IP>
   sudo ufw allow from <GATEWAY_IP> to any port 8000 proto tcp
   # on worker: allow the head
   sudo ufw allow from <HEAD_QSFP_IP>
   ```
   Ray uses a wide range of dynamic ports, which is why the peer rules are not
   port-scoped.

## Start

```bash
./start-cluster.sh          # run on the HEAD node
```

## Flags that matter

| flag / env | why |
|---|---|
| `GLOO_SOCKET_IFNAME` | **Required.** Gloo does the CPU-side rendezvous *before* NCCL. Without it, it resolves the host name to `127.0.0.1` and the worker fails with `Gloo connectFullMesh failed ... remote=[127.0.0.1]`. Setting only `NCCL_SOCKET_IFNAME` is not enough. |
| `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME` | pin GPU collectives / TP traffic to the QSFP link |
| **unset `VLLM_HOST_IP`** | Setting it redirects the API-server↔EngineCore ZMQ handshake; the engine reports ready and the API server waits forever in `wait_for_engine_startup`. The `*_SOCKET_IFNAME` vars already pin the interface. |
| `--reasoning-parser minimax_m2` | **Not** `minimax_m2_append_think` (as some recipes show). The `append_think` variant injects `<think>` into the prompt, but this model emits its own, so the parser never matches and raw `<think>` tags leak into `content` while `reasoning_content` stays empty. |
| `--tool-call-parser minimax_m2` | verified emitting correct `tool_calls` |
| `--kv-cache-dtype fp8`, `--moe-backend cutlass`, `--attention-backend flashinfer` | from the NVIDIA DGX Spark community recipe |

Note `VLLM_NVFP4_GEMM_BACKEND` and `VLLM_USE_FLASHINFER_MOE_FP4` appear in that recipe
but are rejected as unknown by vLLM 0.24 (`Unknown vLLM environment variable detected`).
They are ignored, not fatal.

## Restarting

Always confirm Ray has released the GPUs before relaunching, or the new engine waits
several minutes for a placement group and looks hung:

```bash
docker exec ray-head pkill -f 'vllm serve'
until docker exec ray-head ray status | grep -q '0.0/2.0 GPU'; do sleep 5; done
```

Startup takes roughly 20 minutes from launch to `Application startup complete`
(weight load + compile + CUDA graph capture). Do not call it hung early.

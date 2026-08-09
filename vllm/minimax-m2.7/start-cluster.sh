#!/usr/bin/env bash
# Start MiniMax-M2.7-NVFP4 tensor-parallel across two DGX Sparks.
# Run this on the HEAD node. See README.md for prerequisites.
set -euo pipefail

HEAD_IP="${HEAD_IP:-23.134.235.228}"      # head QSFP address (enp1s0f1np1)
WORKER_IP="${WORKER_IP:-23.134.235.229}"  # worker QSFP address
WORKER_SSH="${WORKER_SSH:-spark4}"        # ssh alias for the worker
IFACE="${IFACE:-enp1s0f1np1}"             # direct-cable interface
IMAGE="${IMAGE:-local/vllm:26.07-xg024-ray}"
MODEL="${MODEL:-lukealonso/MiniMax-M2.7-NVFP4}"
SERVED_NAME="${SERVED_NAME:-minimax-m2.7}"

# NOTE: GLOO_SOCKET_IFNAME is required, not optional - Gloo runs the CPU-side
# rendezvous before NCCL and otherwise falls back to 127.0.0.1.
# VLLM_HOST_IP is deliberately NOT set: it breaks the APIServer<->EngineCore handshake.
ENV_ARGS=(
  -e HF_HOME=/root/.cache/huggingface
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e OMP_NUM_THREADS=8
  -e NCCL_IB_DISABLE=0
  -e NCCL_P2P_DISABLE=1
  -e NCCL_SOCKET_IFNAME="$IFACE"
  -e GLOO_SOCKET_IFNAME="$IFACE"
  -e TP_SOCKET_IFNAME="$IFACE"
)

DOCKER_ARGS=(
  --network host --gpus all --shm-size 16g --ipc host
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)

echo "==> starting Ray head on $HEAD_IP"
docker rm -f ray-head >/dev/null 2>&1 || true
docker run -d --name ray-head "${DOCKER_ARGS[@]}" "${ENV_ARGS[@]}" \
  --entrypoint bash "$IMAGE" \
  -c "ray start --head --node-ip-address=$HEAD_IP --port=6379 --block" >/dev/null
sleep 20

echo "==> starting Ray worker on $WORKER_IP (via $WORKER_SSH)"
ssh "$WORKER_SSH" "docker rm -f ray-worker >/dev/null 2>&1 || true
docker run -d --name ray-worker --network host --gpus all --shm-size 16g --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e OMP_NUM_THREADS=8 -e NCCL_IB_DISABLE=0 -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE -e TP_SOCKET_IFNAME=$IFACE \
  --entrypoint bash $IMAGE \
  -c 'ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --block' >/dev/null"
sleep 25

echo "==> cluster status (expect 2 nodes / 2 GPUs)"
docker exec ray-head ray status | sed -n '1,16p'

echo "==> launching vLLM (TP=2)"
docker exec -d -e VLLM_HOST_IP= ray-head bash -lc "vllm serve $MODEL \
  --served-model-name $SERVED_NAME \
  --trust-remote-code --kv-cache-dtype fp8 \
  --moe-backend cutlass --attention-backend flashinfer \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 -tp 2 \
  --distributed-executor-backend ray \
  --max-model-len 196608 --max-num-seqs 5 \
  --load-format fastsafetensors \
  --enable-auto-tool-choice --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  > /tmp/vllm-minimax.log 2>&1"

echo "==> launched. Startup takes ~20 min; watch with:"
echo "    docker exec ray-head tail -f /tmp/vllm-minimax.log"
echo "    curl -s -o /dev/null -w '%{http_code}\\n' http://localhost:8000/health"

# DGX Spark deployment notes

Field notes from running this stack on a 4-node DGX Spark fleet (GB10 Grace-Blackwell,
aarch64, ~121 GB usable unified memory per node). Recorded 2026-08-09.

## Pick NVFP4, not FP8

GB10 is Blackwell, so **NVFP4 is the hardware-native 4-bit format**. Sizing tables
based on FP8 will tell you a 120B-class model needs two Sparks; at NVFP4 it fits one.

| model | NVFP4 size | Sparks |
|---|---|---|
| Qwen3.5-122B-A10B | 75.6 GB | 1 |
| GLM-4.7-Flash (30B-A3B) | 20.4 GB | 1 |
| Nemotron-3-Super-120B-A12B | ~64 GB | 1 |
| GLM-4.7 (358B) | ~183 GB | 2 (TP=2) |
| Qwen3.5-397B-A17B | ~200 GB | 2 (TP=2) |
| MiniMax M2.7 (230B) | 130.6 GB | 2 — over one node's 121 GB |
| Kimi K2.x / K3 | 591–639 GB+ | 8+ — not viable on 4 |

## vLLM reserves memory up front

`--gpu-memory-utilization 0.85` reserves ~103 GB of a 121 GB node **regardless of
how large the weights are**. A 30 GB model still consumes the whole box, so two
models can never share a Spark. Stopping one frees all of it.

## Launch failures worth knowing

1. **`model type 'glm4_moe_lite' but Transformers does not recognize this architecture`**
   → needs transformers 5.x (image 26.07). `--trust-remote-code` does *not* help:
   it only runs custom config code shipped in the model repo, and NVFP4 quant
   repos ship none.

2. **`In Mamba cache align mode, block_size (4176) must be <= max_num_batched_tokens (2048)`**
   → vLLM 0.24 enforces this for **any hybrid Mamba model** (Qwen3.5-122B-A10B,
   Nemotron-3-Nano/Super). Fix: `--max-num-batched-tokens 8192`. Set it
   preemptively when moving a Mamba model onto 0.24; vendor model cards omit it.

3. **`Reasoning parser 'X' not found`** → parser *filenames* under
   `vllm/reasoning/` do **not** match registered names (e.g. the file is
   `glm47_moe_reasoning_parser.py`, the registered name is `glm47`; the file is
   `nemotron_v3_engine_reasoning_parser.py`, the name is `nemotron_v3`).
   Read the names from the `KeyError`, which lists them all.

4. **`cannot import name 'normalize_tool_choice' from 'xgrammar'` (HTTP 500 on
   any tool call)** → see [base-image/README.md](base-image/README.md). Not
   fixable with flags.

5. **Reasoning models leak chain-of-thought into `content`** without
   `--reasoning-parser`. GLM-4.7-Flash answered "Reply with exactly: glm online"
   with `"1. **Analyze the Request:** ..."` until the parser was set.

## Entrypoints differ between image families

- `vllm/vllm-openai:*` — entrypoint already runs `vllm serve`, so `command:`
  starts with the **model name**.
- `nvcr.io/nvidia/vllm:*` — it does **not**, so `command:` must start with an
  explicit **`vllm serve`**.

Switching image families without adjusting this silently changes what runs.

## Multi-host LiteLLM routing

`litellm/config.yaml` in this repo uses container names, which only works when
vLLM and LiteLLM share a host. To front models running on *other* nodes, point
`api_base` at that node's IP:

```yaml
  - model_name: qwen3.5-122b
    litellm_params:
      model: hosted_vllm/qwen3.5-122b
      api_base: http://<node-ip>:8000/v1
```

Prefer a stable, directly-connected address over a hostname that may resolve to
a management interface. If the nodes are multihomed, the serving node needs a
route back to the gateway **out of the same interface the request arrived on** —
otherwise the SYN arrives, the container answers, and the SYN-ACK leaves via the
default route and is dropped upstream. This looks exactly like a firewall block
but is not; diagnose with `tcpdump -i any` and watch which interface the SYN-ACK
leaves on. Note that Docker's published ports bypass `ufw`'s INPUT chain via
DNAT/FORWARD, so `ufw status` is misleading here.

## Multi-node (TP=2) gotchas

Running one model across two Sparks with Ray surfaces problems single-node never does.
All four of these cost a failed startup before being found:

1. **`GLOO_SOCKET_IFNAME` is required, not just `NCCL_SOCKET_IFNAME`.** Gloo runs the
   CPU-side rendezvous *before* NCCL touches the GPUs. Without it, the head advertises
   `127.0.0.1` (from `/etc/hosts`) and the worker dies with
   `Gloo connectFullMesh failed ... Connection refused, remote=[127.0.0.1]`.

2. **Do not set `VLLM_HOST_IP`.** It redirects the APIServer↔EngineCore ZMQ handshake:
   the engine logs `init engine ... took N s` and then the API server waits forever in
   `wait_for_engine_startup` (confirm with `py-spy dump`). The `*_SOCKET_IFNAME`
   variables already pin the interface for distributed traffic.

3. **`--network host` is subject to ufw; published ports are not.** This is the single
   most confusing interaction on these boxes. A bridge-networked vLLM with
   `-p 8000:8000` is reachable through a `DROP`-policy firewall because Docker's DNAT
   rules bypass the INPUT chain. The moment the same service runs with `--network host`
   (which Ray requires), ufw applies and the port goes dark. Two identical-looking
   deployments will behave differently.

4. **Free the GPUs before relaunching.** Killing `vllm serve` does not immediately
   release the Ray placement group; a new engine then sits in
   `Waiting for creating a placement group ... 310 seconds` and looks hung.
   Wait for `ray status` to report `0.0/2.0 GPU` first.

Also: a 230B model at TP=2 takes ~20 minutes from launch to
`Application startup complete`. Budget for that before declaring a hang — and when a
client times out, check whether the *client's* timeout is the limit rather than the
server's.

## Broken healthchecks are expected

`vllm-litellm-proxy` probes with `curl`, which is absent from the LiteLLM image;
the nginx healthcheck resolves `localhost` to `::1` while nginx listens on IPv4.
Both containers report `unhealthy` while serving correctly — verify with an
external request instead of trusting `docker ps`.

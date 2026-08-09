# Patched vLLM base image (`local/vllm:26.07-xg024`)

All model `docker-compose.yml` files in this repo reference `local/vllm:26.07-xg024`.
Build it once on each node before starting a model:

```bash
cd vllm/base-image
docker build -t local/vllm:26.07-xg024 .
```

## Why this exists

`nvcr.io/nvidia/vllm:26.07-py3` ships vLLM 0.24.0, which **pins `xgrammar==0.2.0`
but calls `xgrammar.normalize_tool_choice()` — a function only added in 0.2.4**.
The mismatch is invisible at startup: the server comes up healthy and answers
plain chat requests, then every request carrying `tools` fails with:

```
HTTP 500  cannot import name 'normalize_tool_choice' from 'xgrammar'
```

The import lives in `vllm/tool_parsers/abstract_tool_parser.py` (the base class),
so it affects **every** tool-call parser and **cannot** be worked around with
`--structured-outputs-config.backend`. Overriding just that one package fixes it.
`--no-deps` keeps every other pinned dependency untouched.

## Why 26.07 and not an older tag

| image | vLLM | transformers | notes |
|---|---|---|---|
| `nvcr.io/nvidia/vllm:26.03-py3` | 0.17.1 | 4.57.5 | too old for post-Q1-2026 architectures |
| `nvcr.io/nvidia/vllm:26.04-py3` | 0.19.0 | 4.57.6 | no `glm4_moe_lite` |
| `vllm/vllm-openai:gemma4-cu130` | 0.19.1.dev | 5.5.0 | Gemma-4-specific; no longer needed |
| **`nvcr.io/nvidia/vllm:26.07-py3`** | **0.24.0** | **5.6.1** | base for this image |

transformers 5.x is required for newer architectures such as `glm4_moe_lite`
(GLM-4.7-Flash). Note that `--trust-remote-code` does **not** substitute for it:
that flag only executes custom config code shipped inside a model repo, and the
NVFP4 quantized repos ship none.

## Driver warning is expected

On DGX Spark (driver 580.x) these images print:

```
ERROR: This container was built for NVIDIA Driver Release 595.58 or later,
       but version 580.126.09 was detected and compatibility mode is UNAVAILABLE.
```

This is non-fatal — the images run correctly.

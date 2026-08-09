FROM local/vllm:26.07-xg024
# vLLM multi-node tensor parallelism needs Ray; the NGC image ships without it.
RUN pip install --no-cache-dir ray

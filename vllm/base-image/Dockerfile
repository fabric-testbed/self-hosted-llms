FROM nvcr.io/nvidia/vllm:26.07-py3
# vLLM 0.24.0 pins xgrammar==0.2.0 but calls xgrammar.normalize_tool_choice,
# which only exists in >=0.2.4. Without this, any tool call returns HTTP 500.
RUN pip install --no-deps --no-cache-dir xgrammar==0.2.4

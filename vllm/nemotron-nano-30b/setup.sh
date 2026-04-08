#!/bin/bash
# Download the custom reasoning parser required by Nemotron 3 Nano
# This file must be present before starting the container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
NC='\033[0m'

echo "Downloading Nemotron 3 Nano reasoning parser..."

wget -q -O nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

echo -e "${GREEN}✓ Downloaded nano_v3_reasoning_parser.py${NC}"
echo ""
echo "You can now start the model with:"
echo "  docker compose up -d"

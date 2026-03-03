#!/bin/bash

# Sync environment variables to model directories
# This ensures HF_TOKEN and other variables are available to all docker-compose files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "Syncing environment variables to model directories..."
echo ""

# Check if parent .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found in llms/vllm/${NC}"
    echo "Creating .env from .env.example..."

    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from template${NC}"
        echo ""
        echo -e "${YELLOW}IMPORTANT: Edit .env and add your HF_TOKEN${NC}"
        echo "  export HF_TOKEN=your_token_here"
        echo "  OR edit llms/vllm/.env"
        echo ""
    else
        echo -e "${RED}Error: .env.example not found${NC}"
        exit 1
    fi
fi

# Read .env and create/update model .env files
source .env 2>/dev/null || true

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}Warning: HF_TOKEN not set in .env${NC}"
    echo "Some models may require authentication."
    echo ""
fi

# Model directories
MODEL_DIRS=("gpt-oss-20b" "gpt-oss-120b" "qwen-30b" "qwen-14b-awq" "phi-3.5-mini")

for dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Syncing to $dir/..."

        # Create or update .env in model directory
        cat > "$dir/.env" <<EOF
# Auto-generated from parent .env
# DO NOT EDIT - changes will be overwritten
# Edit llms/vllm/.env instead and run ./sync-env.sh

HF_TOKEN=${HF_TOKEN:-}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
EOF
        echo -e "${GREEN}  ✓ Updated $dir/.env${NC}"
    fi
done

echo ""
echo -e "${GREEN}Environment sync complete!${NC}"
echo ""

if [ -n "$HF_TOKEN" ]; then
    # Mask the token for display
    MASKED_TOKEN="${HF_TOKEN:0:4}...${HF_TOKEN: -4}"
    echo "HF_TOKEN: $MASKED_TOKEN"
else
    echo -e "${YELLOW}HF_TOKEN: Not set${NC}"
fi

echo ""

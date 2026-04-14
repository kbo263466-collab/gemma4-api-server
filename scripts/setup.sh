#!/bin/bash
set -euo pipefail
# Quick setup script: install deps, download model, build dylib
# Usage: bash scripts/setup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Gemma4 API Server Setup ==="
echo ""

# 1. Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "  Python: $(python3 --version)"

# 2. Install Python deps
echo "[2/4] Installing Python dependencies..."
pip3 install -r "$PROJECT_DIR/requirements.txt" --quiet 2>/dev/null || \
    pip3 install -r "$PROJECT_DIR/requirements.txt"
echo "  Done"

# 3. Build dylib
echo "[3/4] Building liblitertlm.dylib..."
if [ -f "$PROJECT_DIR/lib/liblitertlm.dylib" ]; then
    echo "  Already exists, skipping. Rebuild with: bash scripts/build_dylib.sh"
else
    bash "$PROJECT_DIR/scripts/build_dylib.sh"
fi

# 4. Download Metal accelerator
echo "[4/4] Setting up Metal GPU accelerator..."
LITERT_DIR="${LITERT_LM_PATH:-../LiteRT-LM}"
PREBUILT="$LITERT_DIR/prebuilt/macos_arm64"
ACCEL_FILES=(
    "libLiteRtMetalAccelerator.dylib"
    "libGemmaModelConstraintProvider.dylib"
)
for f in "${ACCEL_FILES[@]}"; do
    if [ -f "$PREBUILT/$f" ] && [ ! -f "$PROJECT_DIR/lib/$f" ]; then
        cp "$PREBUILT/$f" "$PROJECT_DIR/lib/"
        echo "  Copied $f"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Download a Gemma 4 model (.litertlm format)"
echo "  2. Run: MODEL_PATH=/path/to/model.litertlm python3 server.py"
echo ""
echo "Or use the API to hot-load:"
echo "  python3 server.py  # starts with no model"
echo "  curl -X POST http://localhost:8080/v1/model/load -H Content-Type: application/json -d {"model_path": "/path/to/model.litertlm"}"

#!/bin/bash
set -euo pipefail
# Build liblitertlm.dylib from LiteRT-LM source
# Usage: bash scripts/build_dylib.sh [LITERT_LM_PATH]
# Requires: bazel, clang++ (Xcode Command Line Tools)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LITERT_DIR="${1:-../LiteRT-LM}"

echo "=== Building liblitertlm.dylib ==="
echo "LiteRT-LM source: $LITERT_DIR"
echo "Output:           $PROJECT_DIR/lib/"

# Check prerequisites
if [ ! -d "$LITERT_DIR" ]; then
    echo "ERROR: LiteRT-LM directory not found: $LITERT_DIR"
    echo "Clone it first: git clone https://github.com/google-ai-edge/LiteRT-LM.git"
    exit 1
fi

if ! command -v bazel &> /dev/null; then
    echo "ERROR: bazel not found. Install: brew install bazelisk"
    exit 1
fi

if ! command -v clang++ &> /dev/null; then
    echo "ERROR: clang++ not found. Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

mkdir -p "$PROJECT_DIR/lib"

# Build
echo ""
echo "[1/3] Building C API (bazel build //c:engine)..."
cd "$LITERT_DIR"
bazel build //c:engine --config=macos_arm64 -c opt

# Collect archives
echo "[2/3] Collecting archive files..."
ARCHIVES=$(find bazel-bin -name "*.a" -type f | sort)
ARCHIVE_COUNT=$(echo "$ARCHIVES" | grep -c "^" || echo "0")
echo "  Found $ARCHIVE_COUNT archive files"

if [ "$ARCHIVE_COUNT" -eq 0 ]; then
    echo "ERROR: No .a files found in bazel-bin"
    exit 1
fi

# Link
echo "[3/3] Linking dylib..."
clang++ -dynamiclib \
  -install_name @rpath/liblitertlm.dylib \
  -o "$PROJECT_DIR/lib/liblitertlm.dylib" \
  -Wl,-all_load \
  $ARCHIVES \
  -framework Foundation \
  -framework Metal \
  -framework Accelerate \
  -framework CoreGraphics \
  -framework CoreVideo \
  -framework MetalPerformanceShaders \
  -lstdc++ \
  -lc++

echo ""
echo "=== Done! ==="
echo "Output: $PROJECT_DIR/lib/liblitertlm.dylib"
file "$PROJECT_DIR/lib/liblitertlm.dylib"
ls -lh "$PROJECT_DIR/lib/liblitertlm.dylib"

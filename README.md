# Gemma4 API Server

<p align="center">
  <strong>OpenAI-Compatible Local API Server for Gemma 4 Models</strong><br>
  Metal GPU Acceleration on macOS | Multimodal (Text + Image + Audio) | Single File
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-macOS%20ARM64-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Backend-Metal%20GPU-green" alt="GPU">
  <img src="https://img.shields.io/badge/Model-Gemma%204%20E4B-orange" alt="Model">
  <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-purple" alt="API">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

A lightweight, single-file API server that runs [Gemma 4](https://ai.google.dev/gemma) models locally on macOS with Metal GPU acceleration. Fully compatible with the OpenAI Chat Completions API — works as a drop-in replacement for ChatGPT in any app that supports custom OpenAI endpoints.

Built on [Google LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), the official on-device inference framework for Gemma models.

## Why This Project?

| | Gemma4 API Server | Ollama | llama.cpp |
|---|---|---|---|
| **Framework** | LiteRT-LM (Google official) | Custom | Custom |
| **GPU on macOS** | Metal (native) | Metal | Metal |
| **Multimodal** | Text + Image + Audio | Text + Image | Text + Image |
| **OpenAI API** | Built-in | Built-in | Needs server |
| **Mobile support** | iOS + Android (same framework) | No | Limited |
| **Hot reload** | Swap models without restart | Yes | No |
| **Dependencies** | 1 Python file + 1 dylib | Large binary | C++ build |

## Features

- **Metal GPU Acceleration** — 2-3x faster than CPU on Apple Silicon
- **OpenAI Compatible** — Drop-in replacement for ChatGPT API
- **Multimodal** — Text, image (base64/URL), and audio support
- **Streaming** — Server-Sent Events (SSE) for real-time output
- **Hot Reload** — Load/unload models via API without restarting
- **Multi-turn** — Full conversation context support
- **Single File** — Entire server in one `server.py` (512 lines)
- **Zero Cloud** — 100% local, no data leaves your Mac

## Quick Start

### Prerequisites

- macOS 12+ (Apple Silicon recommended)
- Python 3.9+
- Xcode Command Line Tools (`xcode-select --install`)

### Install

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/gemma4-api-server.git
cd gemma4-api-server

# Install Python dependencies
pip3 install -r requirements.txt

# Build liblitertlm.dylib (requires bazel)
bash scripts/build_dylib.sh /path/to/LiteRT-LM

# Or use pre-built dylib from releases
```

### Run

```bash
# Start server with a model
MODEL_PATH=/path/to/model.litertlm python3 server.py

# Custom port and backend
PORT=9000 BACKEND=cpu python3 server.py /path/to/model.litertlm
```

Server starts at `http://localhost:8080` with OpenAI-compatible API.

### Test

```bash
# Text chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Image Analysis

```bash
# Send image as base64 (OpenAI vision format)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BASE64_STRING"}}
      ]
    }]
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a story"}],"stream":true}'
```

### Use with Any OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-4b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

Works with: ChatGPT clients, Continue.dev, Cursor, LobeChat, Open WebUI, and any app supporting custom OpenAI endpoints.

## API Reference

### Chat Completions

`POST /v1/chat/completions`

OpenAI-compatible. Supports `messages`, `temperature`, `max_tokens`, `stream`.

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/model/load` | POST | Load or hot-swap model |
| `/v1/model/unload` | POST | Unload model, free memory |
| `/v1/model/status` | GET | Current model info and params |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Hot Reload Example

```bash
# Unload current model
curl -X POST http://localhost:8080/v1/model/unload

# Load a different model with custom params
curl -X POST http://localhost:8080/v1/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_path":"/path/to/model.litertlm", "backend":"cpu", "temperature":0.5}'
```

## Project Structure

```
gemma4-api-server/
├── server.py              # Main server (single file, 512 lines)
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
├── README.md
├── lib/
│   ├── liblitertlm.dylib                  # LiteRT-LM C API library
│   ├── libLiteRtMetalAccelerator.dylib    # Metal GPU accelerator
│   └── libGemmaModelConstraintProvider.dylib
├── scripts/
│   ├── setup.sh         # One-click setup
│   └── build_dylib.sh   # Build from LiteRT-LM source
└── docs/
    └── API.md           # Detailed API documentation
```

## Building from Source

### Build liblitertlm.dylib

Requires [Bazel](https://bazel.build/install) and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM):

```bash
# Clone LiteRT-LM
git clone https://github.com/google-ai-edge/LiteRT-LM.git

# Build
bash scripts/build_dylib.sh ./LiteRT-LM
```

### Get GPU Accelerator Libraries

```bash
cp LiteRT-LM/prebuilt/macos_arm64/libLiteRtMetalAccelerator.dylib lib/
cp LiteRT-LM/prebuilt/macos_arm64/libGemmaModelConstraintProvider.dylib lib/
```

## Model Format

This server uses `.litertlm` model format from Google LiteRT-LM. To get Gemma 4 models in this format, use the [LiteRT-LM tools](https://github.com/google-ai-edge/LiteRT-LM) to convert from HuggingFace checkpoints.

## Performance

Tested on MacBook Pro (Apple M1, 16GB RAM) with Gemma 4 E4B:

| Mode | First Token | Tokens/sec |
|------|------------|------------|
| GPU (Metal) | ~0.8s | ~15 tok/s |
| CPU | ~2.5s | ~5 tok/s |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to .litertlm model file |
| `BACKEND` | `gpu` | Inference backend: `gpu` or `cpu` |
| `PORT` | `8080` | Server port |

## Roadmap

- [ ] Docker support
- [ ] GPU sampler integration (faster token generation)
- [ ] Token usage tracking
- [ ] Embeddings API
- [ ] Function calling / tools support
- [ ] Model download helper script

## Acknowledgments

- [Google LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) — On-device inference framework
- [Gemma](https://ai.google.dev/gemma) — Google open language models
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework

## License

[MIT](LICENSE)

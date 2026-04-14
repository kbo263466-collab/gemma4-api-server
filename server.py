#!/usr/bin/env python3
"""Gemma4 API Server - OpenAI Compatible Local Inference Server

A lightweight OpenAI-compatible API server for Gemma 4 models using
Google LiteRT-LM framework with Metal GPU acceleration on macOS.

Usage:
    python3 server.py                          # Use MODEL_PATH env or prompt
    python3 server.py /path/to/model.litertlm  # Specify model path
    MODEL_PATH=... BACKEND=cpu PORT=9000 python3 server.py
"""

from __future__ import annotations

import os
import ctypes
import json
import time
import uuid
import threading
import socket
import sys
import base64
from pathlib import Path
from typing import Optional, List, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ============ C API Bindings ============

class SamplerParams(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("seed", ctypes.c_int32),
    ]

STREAM_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_char_p
)


class LiteRTLMEngine:
    """Wrapper around LiteRT-LM C API for Gemma model inference."""

    def __init__(self, lib_path: str, model_path: str, backend: str = "gpu"):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self.model_path = model_path
        self.backend = backend
        self._engine = None
        self._conversation = None
        self._settings = None
        self._conv_config = None
        self._lock = threading.Lock()
        self._default_temperature = 0.7
        self._default_top_k = 40
        self._default_top_p = 0.95
        self._default_max_tokens = 2048
        self._loaded = False

    def _setup_functions(self):
        lib = self.lib
        lib.litert_lm_set_min_log_level.argtypes = [ctypes.c_int]
        lib.litert_lm_set_min_log_level.restype = None

        lib.litert_lm_engine_settings_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.litert_lm_engine_settings_create.restype = ctypes.c_void_p
        lib.litert_lm_engine_settings_delete.argtypes = [ctypes.c_void_p]
        lib.litert_lm_engine_settings_delete.restype = None

        lib.litert_lm_engine_create.argtypes = [ctypes.c_void_p]
        lib.litert_lm_engine_create.restype = ctypes.c_void_p
        lib.litert_lm_engine_delete.argtypes = [ctypes.c_void_p]
        lib.litert_lm_engine_delete.restype = None

        lib.litert_lm_engine_settings_set_max_num_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.litert_lm_engine_settings_set_max_num_tokens.restype = None
        lib.litert_lm_engine_settings_set_cache_dir.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.litert_lm_engine_settings_set_cache_dir.restype = None
        lib.litert_lm_engine_settings_set_parallel_file_section_loading.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.litert_lm_engine_settings_set_parallel_file_section_loading.restype = None

        lib.litert_lm_session_config_create.argtypes = []
        lib.litert_lm_session_config_create.restype = ctypes.c_void_p
        lib.litert_lm_session_config_set_max_output_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.litert_lm_session_config_set_sampler_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(SamplerParams)]
        lib.litert_lm_session_config_delete.argtypes = [ctypes.c_void_p]

        lib.litert_lm_conversation_config_create.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool
        ]
        lib.litert_lm_conversation_config_create.restype = ctypes.c_void_p
        lib.litert_lm_conversation_config_delete.argtypes = [ctypes.c_void_p]

        lib.litert_lm_conversation_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.litert_lm_conversation_create.restype = ctypes.c_void_p
        lib.litert_lm_conversation_delete.argtypes = [ctypes.c_void_p]

        lib.litert_lm_conversation_send_message.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.litert_lm_conversation_send_message.restype = ctypes.c_void_p

        lib.litert_lm_conversation_send_message_stream.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
            STREAM_CALLBACK, ctypes.c_void_p
        ]
        lib.litert_lm_conversation_send_message_stream.restype = ctypes.c_int

        lib.litert_lm_conversation_cancel_process.argtypes = [ctypes.c_void_p]

        lib.litert_lm_json_response_delete.argtypes = [ctypes.c_void_p]
        lib.litert_lm_json_response_get_string.argtypes = [ctypes.c_void_p]
        lib.litert_lm_json_response_get_string.restype = ctypes.c_char_p

    @property
    def is_loaded(self):
        return self._loaded

    def _create_session_config(self, temperature=None, max_tokens=None):
        session_config = self.lib.litert_lm_session_config_create()
        mt = max_tokens if max_tokens is not None else self._default_max_tokens
        self.lib.litert_lm_session_config_set_max_output_tokens(session_config, mt)
        temp = temperature if temperature is not None else self._default_temperature
        params = SamplerParams(type=2, top_k=self._default_top_k, top_p=self._default_top_p, temperature=temp, seed=0)
        self.lib.litert_lm_session_config_set_sampler_params(session_config, ctypes.byref(params))
        return session_config

    def _create_conversation(self):
        session_config = self._create_session_config()
        conv_config = self.lib.litert_lm_conversation_config_create(
            self._engine, session_config, None, None, None, False
        )
        self.lib.litert_lm_session_config_delete(session_config)
        if not conv_config:
            raise RuntimeError("Failed to create conversation config")
        self._conv_config = conv_config
        self._conversation = self.lib.litert_lm_conversation_create(self._engine, conv_config)
        if not self._conversation:
            raise RuntimeError("Failed to create conversation")

    def load(self, model_path=None, backend=None, temperature=None, top_k=None,
             top_p=None, max_tokens=None):
        if self._loaded:
            self.unload()
        if model_path is not None:
            self.model_path = model_path
        if backend is not None:
            self.backend = backend
        if temperature is not None:
            self._default_temperature = temperature
        if top_k is not None:
            self._default_top_k = top_k
        if top_p is not None:
            self._default_top_p = top_p
        if max_tokens is not None:
            self._default_max_tokens = max_tokens

        self.lib.litert_lm_set_min_log_level(1)

        # Audio backend uses CPU (audio models do not support GPU)
        settings = self.lib.litert_lm_engine_settings_create(
            self.model_path.encode(), self.backend.encode(), self.backend.encode(), b"cpu"
        )
        if not settings:
            raise RuntimeError("Failed to create engine settings")
        self._settings = settings

        self._engine = self.lib.litert_lm_engine_create(settings)
        if not self._engine:
            self._settings = None
            self.lib.litert_lm_engine_settings_delete(settings)
            raise RuntimeError("Failed to create engine. Check model path and backend.")

        self._create_conversation()
        self._loaded = True
        print("[OK] Model loaded: %s (backend=%s)" % (self.model_path, self.backend))

    def _convert_openai_to_conversation_format(self, messages):
        """Convert OpenAI format messages to LiteRT conversation format.

        OpenAI: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        LiteRT:  {"type": "image", "blob": "<base64>"}
        LiteRT:  {"type": "image", "path": "/path/to/file"}
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype == "text":
                            new_content.append({"type": "text", "text": part.get("text", "")})
                        elif ptype == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:image"):
                                b64_part = url.split(";base64,", 1)
                                if len(b64_part) == 2:
                                    new_content.append({"type": "image", "blob": b64_part[1]})
                            elif url.startswith("http"):
                                pass  # URL images not supported
                            else:
                                new_content.append({"type": "image", "path": url})
                        elif ptype in ("image", "audio"):
                            new_content.append(part)
                converted.append({"role": role, "content": new_content})
        return converted

    def generate_sync(self, messages, temperature=None, max_tokens=None):
        with self._lock:
            if not self._loaded:
                raise RuntimeError("Engine not loaded")

            conv_messages = self._convert_openai_to_conversation_format(messages)
            message_json = json.dumps(conv_messages)

            response_ptr = self.lib.litert_lm_conversation_send_message(
                self._conversation, message_json.encode(), None
            )
            if not response_ptr:
                raise RuntimeError("Failed to generate response")

            response_str = self.lib.litert_lm_json_response_get_string(response_ptr)
            result = json.loads(response_str.decode())
            self.lib.litert_lm_json_response_delete(response_ptr)

            text = ""
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and "text" in item:
                        text += item["text"]
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and "content" in item:
                        for sub in item["content"]:
                            if isinstance(sub, dict) and "text" in sub:
                                text += sub["text"]
            return text

    def generate_stream(self, messages, temperature=None, max_tokens=None):
        with self._lock:
            if not self._loaded:
                raise RuntimeError("Engine not loaded")

            conv_messages = self._convert_openai_to_conversation_format(messages)
            message_json = json.dumps(conv_messages)

            chunks = []
            done_event = threading.Event()
            error_holder = [None]

            @STREAM_CALLBACK
            def callback(cb_data, chunk, is_final, error_msg):
                if error_msg:
                    error_holder[0] = error_msg.decode() if error_msg else "Unknown error"
                    done_event.set()
                    return
                if is_final:
                    done_event.set()
                    return
                if chunk:
                    chunks.append(chunk.decode())

            ret = self.lib.litert_lm_conversation_send_message_stream(
                self._conversation, message_json.encode(), None,
                callback, None
            )
            if ret != 0:
                raise RuntimeError("Failed to start stream (code: %d)" % ret)

            idx = 0
            while not done_event.is_set():
                done_event.wait(timeout=0.1)
                while idx < len(chunks):
                    yield chunks[idx]
                    idx += 1

            while idx < len(chunks):
                yield chunks[idx]
                idx += 1

            if error_holder[0]:
                raise RuntimeError(error_holder[0])

    def unload(self):
        if self._conversation:
            self.lib.litert_lm_conversation_delete(self._conversation)
            self._conversation = None
        if self._conv_config:
            self.lib.litert_lm_conversation_config_delete(self._conv_config)
            self._conv_config = None
        if self._engine:
            self.lib.litert_lm_engine_delete(self._engine)
            self._engine = None
        if self._settings:
            self.lib.litert_lm_engine_settings_delete(self._settings)
            self._settings = None
        self._loaded = False
        print("[OK] Model unloaded")


# ============ OpenAI-Compatible Models ============

class ChatMessage(BaseModel):
    role: str
    content: Union[str, list]

class ChatCompletionRequest(BaseModel):
    model: str = "gemma-4-4b"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

class LoadModelRequest(BaseModel):
    model_path: Optional[str] = None
    backend: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


# ============ FastAPI App ============

app = FastAPI(title="Gemma4 API Server", version="1.3.0",
              description="OpenAI-compatible API server for Gemma 4 models with Metal GPU acceleration")
engine: Optional[LiteRTLMEngine] = None


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.on_event("startup")
async def startup():
    global engine

    model_path = os.environ.get("MODEL_PATH", "")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not model_path or not Path(model_path).exists():
        print("[ERROR] Model path not specified or file not found.")
        print("Usage: python3 server.py <model_path>")
        print("       MODEL_PATH=/path/to/model.litertlm python3 server.py")
        sys.exit(1)

    backend = os.environ.get("BACKEND", "gpu")
    lib_path = Path(__file__).parent / "lib" / "liblitertlm.dylib"
    if not lib_path.exists():
        lib_path = Path(__file__).parent / "liblitertlm.dylib"
    if not lib_path.exists():
        print("[ERROR] liblitertlm.dylib not found in %s or lib/" % Path(__file__).parent)
        sys.exit(1)

    engine = LiteRTLMEngine(str(lib_path), model_path, backend)
    engine.load()

    port = int(os.environ.get("PORT", 8080))
    print("")
    print("=" * 40)
    print("  Gemma4 API Server v1.3.0")
    print("  Model:   %s" % model_path)
    print("  Backend: %s" % backend)
    print("  Local:   http://localhost:%d" % port)
    print("  LAN:     http://%s:%d" % (get_local_ip(), port))
    print("  API:     http://%s:%d/v1/chat/completions" % (get_local_ip(), port))
    print("=" * 40)
    print("")


@app.on_event("shutdown")
async def shutdown():
    if engine:
        engine.unload()


@app.post("/v1/model/load")
async def load_model(req: LoadModelRequest = LoadModelRequest()):
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    try:
        engine.load(
            model_path=req.model_path,
            backend=req.backend,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
        return {"status": "ok", "model": engine.model_path, "backend": engine.backend}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/model/unload")
async def unload_model():
    if not engine:
        raise HTTPException(500, "Engine not initialized")
    engine.unload()
    return {"status": "ok", "message": "Model unloaded"}


@app.get("/v1/model/status")
async def model_status():
    if not engine:
        return {"loaded": False}
    return {
        "loaded": engine.is_loaded,
        "model": engine.model_path,
        "backend": engine.backend,
        "temperature": engine._default_temperature,
        "top_k": engine._default_top_k,
        "top_p": engine._default_top_p,
        "max_tokens": engine._default_max_tokens,
    }


@app.get("/v1/models")
async def list_models():
    if engine and engine.is_loaded:
        return {
            "object": "list",
            "data": [{
                "id": "gemma-4-4b",
                "object": "model",
                "owned_by": "local",
                "backend": engine.backend,
            }]
        }
    return {"object": "list", "data": []}


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": engine.is_loaded if engine else False}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not engine or not engine.is_loaded:
        raise HTTPException(503, "Model not loaded. POST /v1/model/load first.")

    messages = [m.model_dump() for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_response(messages, req.model, req.temperature, req.max_tokens),
            media_type="text/event-stream",
        )

    try:
        text = engine.generate_sync(messages, temperature=req.temperature, max_tokens=req.max_tokens)
        return {
            "id": "chatcmpl-%s" % uuid.uuid4().hex[:8],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


def _stream_response(messages, model, temperature, max_tokens):
    chunk_id = "chatcmpl-%s" % uuid.uuid4().hex[:8]
    created = int(time.time())

    try:
        for chunk_text in engine.generate_stream(messages, temperature=temperature, max_tokens=max_tokens):
            data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }],
            }
            yield "data: %s\n\n" % json.dumps(data)

        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield "data: %s\n\n" % json.dumps(data)
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield "data: %s\n\n" % json.dumps({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

#!/usr/bin/env python3
"""Gemma 4 API 服务 - OpenAI 兼容接口 (llama-cpp-python + GGUF)
规范建档：github.com/moonhwm/- ｜ data_cutoff: 2026-08-29"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def _maybe_preload_cuda():
    """n_gpu_layers>0 时，在 import llama_cpp 前预载 pip 版 cublas/cudart DLL。
    llama_cpp 以 winmode=RTLD_GLOBAL 加载原生库，os.add_dll_directory 对其无效，
    必须用 ctypes.CDLL 显式预载（gpu-test/test_gpu_load.py 实测结论）。"""
    layers = os.environ.get("GEMMA4_N_GPU_LAYERS")
    if layers is None:
        cfg_file = Path(__file__).parent / "server.config.json"
        if cfg_file.exists():
            try:
                layers = str(json.loads(cfg_file.read_text(encoding="utf-8")).get("n_gpu_layers", 0))
            except Exception:
                layers = "0"
    try:
        if not layers or int(layers) <= 0:
            return
    except ValueError:
        return
    import ctypes
    sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    rt, cu = sp / "cuda_runtime" / "bin", sp / "cublas" / "bin"
    for d in (rt, cu):
        if d.exists():
            os.add_dll_directory(str(d))
    for dll in (rt / "cudart64_12.dll", cu / "cublas64_12.dll", cu / "cublasLt64_12.dll"):
        if dll.exists():
            ctypes.CDLL(str(dll))
            print(f"[*] 预载 CUDA DLL: {dll.name}")


_maybe_preload_cuda()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uvicorn

try:
    from llama_cpp import Llama
except ImportError as e:
    print(f"错误: 缺少 llama-cpp-python: {e}")
    sys.exit(1)

_CONFIG_PATH = Path(__file__).parent / "server.config.json"
_FILE_CFG: dict = {}
if _CONFIG_PATH.exists():
    try:
        _FILE_CFG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        print(f"[*] 已加载配置文件: {_CONFIG_PATH}")
    except Exception as e:
        print(f"[!] 配置文件解析失败，忽略: {e}")


def _cfg(env_name: str, key: str, default, cast=str):
    if env_name in os.environ:
        return cast(os.environ[env_name])
    if key in _FILE_CFG:
        return cast(_FILE_CFG[key])
    return cast(default)


REPO_ID = _cfg("GEMMA4_REPO", "repo", "ggml-org/gemma-4-E4B-it-GGUF")
MODEL_FILE = _cfg("GEMMA4_MODEL_FILE", "model_file", "gemma-4-E4B-it-BF16.gguf")
PORT = _cfg("GEMMA4_PORT", "port", 8000, int)
N_CTX = _cfg("GEMMA4_N_CTX", "n_ctx", 4096, int)
N_THREADS = _cfg("GEMMA4_N_THREADS", "n_threads", 6, int)
N_GPU_LAYERS = _cfg("GEMMA4_N_GPU_LAYERS", "n_gpu_layers", 0, int)
FLASH_ATTN = _cfg("GEMMA4_FLASH_ATTN", "flash_attn", "false").lower() in ("1", "true", "yes")

MODELS_DIR = Path(_cfg("GEMMA4_MODELS_DIR", "models_dir", r"C:\temp\g4models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MMPROJ_FILE = _cfg("GEMMA4_MMPROJ_FILE", "mmproj_file", "mmproj-gemma-4-E4B-it-BF16.gguf")


def ensure_model() -> str:
    local_path = MODELS_DIR / MODEL_FILE
    if local_path.exists():
        print(f"[*] 使用本地模型: {local_path}")
        return str(local_path)
    print(f"[*] 模型文件未找到，开始下载: {REPO_ID}/{MODEL_FILE}")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, local_dir=str(MODELS_DIR), local_dir_use_symlinks=False)
        print(f"[*] 模型下载完成: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[!] 下载失败: {e}")
        sys.exit(1)


model_path = ensure_model()

print(f"[*] 加载模型: {model_path}")
print(f"[*] 上下文长度: {N_CTX} | 线程数: {N_THREADS} | GPU 卸载层数: {N_GPU_LAYERS} | FlashAttn: {FLASH_ATTN}")

chat_handler = None
if MMPROJ_FILE.lower() not in ("", "none"):
    mmproj_path = MODELS_DIR / MMPROJ_FILE
    if mmproj_path.exists():
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        print(f"[*] 启用视觉功能: {mmproj_path}")
        chat_handler = Gemma4ChatHandler(clip_model_path=str(mmproj_path), verbose=False, use_gpu=False)

llm = Llama(
    model_path=model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,
    flash_attn=FLASH_ATTN,
    use_mmap=True,
    use_mlock=False,
    chat_handler=chat_handler,
    verbose=False,
)

app = FastAPI(title="Gemma 4 API", version="1.0-llamacpp")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def webchat_page():
    page = Path(__file__).parent / "webchat.html"
    return FileResponse(str(page), media_type="text/html; charset=utf-8")


@app.get("/claw-local-assistant.apk", include_in_schema=False)
def download_apk():
    apk = Path(__file__).parent.parent / "claw-local-assistant.apk"
    return FileResponse(str(apk), media_type="application/vnd.android.package-archive", filename="claw-local-assistant.apk")


@app.get("/healthz", include_in_schema=False)
def healthz():
    import llama_cpp
    return {
        "ok": True,
        "model": MODEL_FILE,
        "llama_cpp_python": llama_cpp.__version__,
        "config": {"n_ctx": N_CTX, "n_threads": N_THREADS, "n_gpu_layers": N_GPU_LAYERS, "flash_attn": FLASH_ATTN, "vision": chat_handler is not None, "port": PORT},
        "config_file": str(_CONFIG_PATH),
        "config_file_loaded": bool(_FILE_CFG),
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "gemma4", "object": "model", "created": int(time.time()), "owned_by": "google"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    gen_kwargs = {
        "messages": body.get("messages", []),
        "max_tokens": int(body.get("max_tokens", 1024)),
        "temperature": float(body.get("temperature", 0.7)),
        "top_p": float(body.get("top_p", 0.9)),
        "top_k": int(body.get("top_k", 40)),
        "repeat_penalty": float(body.get("repetition_penalty", 1.0)),
        "stream": bool(body.get("stream", False)),
    }
    if gen_kwargs["stream"]:
        return StreamingResponse(_stream_chat(**gen_kwargs), media_type="text/event-stream")
    return JSONResponse(llm.create_chat_completion(**gen_kwargs))


async def _stream_chat(**kwargs):
    for chunk in llm.create_chat_completion(**kwargs):
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

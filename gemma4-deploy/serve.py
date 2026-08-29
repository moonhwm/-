#!/usr/bin/env python3
"""Gemma 4 API 服务 - OpenAI 兼容接口 (llama-cpp-python + GGUF)"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def _maybe_preload_cuda():
    """n_gpu_layers>0 时，在 import llama_cpp 前预载 pip 版 cublas/cudart DLL。

    llama_cpp 以 winmode=RTLD_GLOBAL 加载原生库，os.add_dll_directory 对其无效，
    必须用 ctypes.CDLL 显式预载（gpu-test/test_gpu_load.py 实测结论）。
    """
    layers = os.environ.get("GEMMA4_N_GPU_LAYERS")
    if layers is None:
        cfg_file = Path(__file__).parent / "server.config.json"
        if cfg_file.exists():
            try:
                layers = str(json.loads(cfg_file.read_text(encoding="utf-8")).get("n_gpu_layers", 0))
            except Exception:
                layers = "0"
    try:
        if not layers or int(layers) == 0:  # 0=纯 CPU；负数为"全层上卡"，仍需 CUDA
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

# 配置：优先级 环境变量 > server.config.json > 默认值
# server.config.json 是面向未来调优/开发的接口文件，所有键都可用同名环境变量覆盖
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


# 注意：HuggingFace 上实际存在的 Gemma-4 模型为 E4B（非 4b），已自动适配
REPO_ID = _cfg("GEMMA4_REPO", "repo", "ggml-org/gemma-4-E4B-it-GGUF")
MODEL_FILE = _cfg("GEMMA4_MODEL_FILE", "model_file", "gemma-4-E4B-it-BF16.gguf")
PORT = _cfg("GEMMA4_PORT", "port", 8000, int)
# BF16 全量包 14GB > 本机可用内存，上下文收窄以降低 KV cache 占用（可用 GEMMA4_N_CTX 覆盖）
N_CTX = _cfg("GEMMA4_N_CTX", "n_ctx", 4096, int)
# i7-12700H = 6P+8E；P 核 6 个是甜点，超线程在纯 CPU 推理上增益有限
N_THREADS = _cfg("GEMMA4_N_THREADS", "n_threads", 6, int)
# GPU 卸载层数（需 CUDA 版 llama-cpp-python；CPU 版下自动忽略）
N_GPU_LAYERS = _cfg("GEMMA4_N_GPU_LAYERS", "n_gpu_layers", 0, int)
# Flash Attention（省 KV 显存/内存；模型不支持时 llama.cpp 自动回退）
FLASH_ATTN = _cfg("GEMMA4_FLASH_ATTN", "flash_attn", "false").lower() in ("1", "true", "yes")

# 使用短 ASCII 路径存放模型，避免 Windows C++ 库潜在路径问题
MODELS_DIR = Path(_cfg("GEMMA4_MODELS_DIR", "models_dir", r"C:\temp\g4models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 多模态视觉附件（mmproj）。设 GEMMA4_MMPROJ_FILE=none 可关闭视觉功能
MMPROJ_FILE = _cfg("GEMMA4_MMPROJ_FILE", "mmproj_file", "mmproj-gemma-4-E4B-it-BF16.gguf")

# 公网暴露时的访问密钥：设置后 /v1/* 必须带 Authorization: Bearer <key>
# 生成：python -c "import secrets; print(secrets.token_urlsafe(24))"，只写进本地 server.config.json，不入库不入档
API_KEY = _cfg("GEMMA4_API_KEY", "api_key", "")


def ensure_model() -> str:
    """确保 GGUF 模型文件已下载，返回本地绝对路径"""
    local_path = MODELS_DIR / MODEL_FILE
    if local_path.exists():
        print(f"[*] 使用本地模型: {local_path}")
        return str(local_path)

    print(f"[*] 模型文件未找到，开始下载: {REPO_ID}/{MODEL_FILE}")
    print(f"[*] 镜像: {os.environ['HF_ENDPOINT']}")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[*] 模型下载完成: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[!] 下载失败: {e}")
        sys.exit(1)


model_path = ensure_model()

print(f"[*] 加载模型: {model_path}")
print(f"[*] 上下文长度: {N_CTX}")
print(f"[*] 线程数: {N_THREADS}")
print(f"[*] GPU 卸载层数: {N_GPU_LAYERS}（0=纯 CPU）")
print(f"[*] Flash Attention: {FLASH_ATTN}")
print(f"[*] 服务端口: {PORT}")
print("[*] 首次加载需要初始化内存映射，请稍等...")

# 视觉 chat handler：mmproj 存在时启用，接口自动支持 OpenAI image_url 消息格式
chat_handler = None
if MMPROJ_FILE.lower() not in ("", "none"):
    mmproj_path = MODELS_DIR / MMPROJ_FILE
    if mmproj_path.exists():
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        print(f"[*] 启用视觉功能: {mmproj_path}")
        chat_handler = Gemma4ChatHandler(
            clip_model_path=str(mmproj_path), verbose=False, use_gpu=False
        )
    else:
        print(f"[!] 未找到 mmproj 文件，视觉功能关闭: {mmproj_path}")

# 加载模型（mmap 惰性分页：物理内存不足时按需从磁盘换页，避免一次性占满内存）
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

print(f"[*] 模型加载完成")
print(f"[*] 服务启动: http://localhost:{PORT}")

app = FastAPI(title="Gemma 4 API", version="1.0-llamacpp")

# 允许局域网 APK / WebView 跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def webchat_page():
    """浏览器版聊天界面（手机/平板直接访问，与 API 同源免跨域）"""
    page = Path(__file__).parent / "webchat.html"
    return FileResponse(str(page), media_type="text/html; charset=utf-8")


@app.get("/claw-local-assistant.apk", include_in_schema=False)
def download_apk():
    """APK 安装包下载（局域网手机浏览器直接下载）"""
    apk = Path(__file__).parent.parent / "claw-local-assistant.apk"
    return FileResponse(
        str(apk),
        media_type="application/vnd.android.package-archive",
        filename="claw-local-assistant.apk",
    )


@app.get("/healthz", include_in_schema=False)
def healthz():
    """运行自检：配置回显 + 版本信息，供监控与未来调优工具使用"""
    import llama_cpp
    return {
        "ok": True,
        "model": MODEL_FILE,
        "llama_cpp_python": llama_cpp.__version__,
        "config": {
            "n_ctx": N_CTX,
            "n_threads": N_THREADS,
            "n_gpu_layers": N_GPU_LAYERS,
            "flash_attn": FLASH_ATTN,
            "vision": chat_handler is not None,
            "port": PORT,
        },
        "config_file": str(_CONFIG_PATH),
        "config_file_loaded": bool(_FILE_CFG),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gemma4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 设置了 api_key 时校验 Bearer 头；未设置则放行（纯局域网模式）
    if API_KEY:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {API_KEY}":
            return JSONResponse({"error": {"message": "unauthorized", "type": "auth_error"}}, status_code=401)
    body = await request.json()
    messages = body.get("messages", [])
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 1024))
    stream = bool(body.get("stream", False))
    top_p = float(body.get("top_p", 0.9))
    top_k = int(body.get("top_k", 40))
    repeat_penalty = float(body.get("repetition_penalty", 1.0))

    gen_kwargs = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "stream": stream,
    }

    if stream:
        return StreamingResponse(
            _stream_chat(**gen_kwargs),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(llm.create_chat_completion(**gen_kwargs))


async def _stream_chat(**kwargs):
    for chunk in llm.create_chat_completion(**kwargs):
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

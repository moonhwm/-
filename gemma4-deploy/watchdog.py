# -*- coding: utf-8 -*-
"""Gemma-4 BF16 服务看门狗：每 120 秒探活 /healthz，死了就经 start_bf16.py 拉起。
由计划任务 Gemma4-BF16-Watchdog（登录触发）托管运行，与主服务相互独立。"""
import os
import time
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LOG = HERE / "watchdog.log"
HEALTH = "http://127.0.0.1:8000/healthz"
INTERVAL = 120
BOOT_GRACE = 180  # 登录/启动后宽限期，模型加载需要时间
HEALTH_TIMEOUT = 25  # 推理时 GIL 被占，healthz 可能慢，25s 容忍
FAIL_THRESHOLD = 3  # 连续 3 次失败才拉起，防长推理期误杀（实测 2026-08-28 23:39 长生成致 healthz 超时）

PY_GPU = r"C:\temp\g4env-gpu\Scripts\python.exe"
PY_CPU = r"C:\temp\g4env\Scripts\python.exe"
PY = PY_GPU if os.path.exists(PY_GPU) else PY_CPU


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    with open(LOG, "ab") as f:
        f.write(line.encode("utf-8", errors="replace"))


def alive() -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=HEALTH_TIMEOUT) as r:
            return r.status == 200
    except Exception:
        return False


def relaunch():
    DETACHED = 0x00000008
    NEWGROUP = 0x00000200
    lf = open(HERE / "server_bf16.log", "ab")
    subprocess.Popen([PY, str(HERE / "serve.py")], stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE), creationflags=DETACHED | NEWGROUP)


def main():
    log("watchdog started")
    time.sleep(BOOT_GRACE)
    fail_streak = 0
    while True:
        if alive():
            fail_streak = 0
        else:
            fail_streak += 1
            log(f"health check failed ({fail_streak})")
            if fail_streak >= FAIL_THRESHOLD:
                log("relaunching serve.py")
                relaunch()
                fail_streak = 0
                time.sleep(BOOT_GRACE)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

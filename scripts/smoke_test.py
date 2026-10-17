"""Day-0 smoke test for Crusoe Managed Inference.

Checks, per model: basic chat, latency, usage fields, rate-limit headers,
native tool calling, JSON mode, the thinking toggle, and image input.

    CRUSOE_API_KEY=... python scripts/smoke_test.py [model ...]
"""
import base64
import json
import os
import struct
import sys
import time
import zlib

import httpx

BASE = os.environ.get("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")
KEY = os.environ["CRUSOE_API_KEY"]
DEFAULT_MODELS = [
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3.5-Lightning-30B-A3B",
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B",
    "Qwen/Qwen3.8-27B",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "gpu_metrics",
        "description": "Get DCGM metrics for one GPU.",
        "parameters": {
            "type": "object",
            "properties": {"node": {"type": "string"}, "gpu": {"type": "integer"}},
            "required": ["node", "gpu"],
        },
    },
}]

def _png(w, h, rgb):
    """Encode a solid-color PNG (no external deps)."""
    row = b"\x00" + bytes(rgb) * w
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))


RED_PNG = base64.b64encode(_png(16, 16, (220, 20, 20))).decode()


def call(model, body, timeout=120):
    t0 = time.time()
    r = httpx.post(f"{BASE}/chat/completions", json={"model": model, **body},
                   headers={"Authorization": f"Bearer {KEY}"}, timeout=timeout)
    dt = time.time() - t0
    rl = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text[:300]}
    return r.status_code, dt, data, rl


def msg(data):
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}


def check(model):
    out = {"model": model}
    # 1. basic chat + headers + usage
    code, dt, data, rl = call(model, {
        "messages": [{"role": "user", "content": "Reply with exactly: capy ok"}],
        "max_tokens": 400})
    m = msg(data)
    out["basic"] = {"http": code, "sec": round(dt, 2),
                    "content": (m.get("content") or "")[:80],
                    "has_reasoning": bool(m.get("reasoning_content") or m.get("reasoning")),
                    "usage": data.get("usage"), "ratelimit": rl}
    if code != 200:
        out["basic"]["error"] = str(data)[:300]
        return out
    # 2. thinking toggle
    code, dt, data, _ = call(model, {
        "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
        "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}})
    m = msg(data)
    out["no_think"] = {"http": code, "sec": round(dt, 2), "content": (m.get("content") or "")[:60],
                       "has_reasoning": bool(m.get("reasoning_content") or m.get("reasoning")),
                       "completion_tokens": (data.get("usage") or {}).get("completion_tokens")}
    # 3. native tool calling
    code, dt, data, _ = call(model, {
        "messages": [{"role": "user", "content": "Check GPU 3 on node np-17 using the tool."}],
        "tools": TOOLS, "tool_choice": "auto", "max_tokens": 600})
    m = msg(data)
    tc = m.get("tool_calls") or []
    out["tools"] = {"http": code, "sec": round(dt, 2), "n_calls": len(tc),
                    "call": (tc[0]["function"] if tc else None),
                    "content": (m.get("content") or "")[:80]}
    # 4. JSON mode
    code, dt, data, _ = call(model, {
        "messages": [{"role": "user", "content": 'Return a JSON object {"suspect": <node string>, "confidence": <0-1>} for node np-17 at 0.9.'}],
        "response_format": {"type": "json_object"}, "max_tokens": 400})
    m = msg(data)
    content = m.get("content") or ""
    try:
        json.loads(content)
        ok = True
    except ValueError:
        ok = False
    out["json"] = {"http": code, "sec": round(dt, 2), "parses": ok, "content": content[:80]}
    # 5. image input
    code, dt, data, _ = call(model, {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What color is this image? One word."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED_PNG}"}}]}],
        "max_tokens": 300})
    m = msg(data)
    out["image"] = {"http": code, "sec": round(dt, 2), "content": (m.get("content") or "")[:60],
                    "error": (str(data)[:160] if code != 200 else None)}
    return out


if __name__ == "__main__":
    models = sys.argv[1:] or DEFAULT_MODELS
    for mdl in models:
        print(json.dumps(check(mdl), indent=1, default=str), flush=True)

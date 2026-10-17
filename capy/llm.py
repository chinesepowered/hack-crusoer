"""Thin async client for Crusoe Managed Inference (OpenAI-compatible).

Handles retries on 429/503 (shared serverless capacity), per-model
concurrency, and token/cost accounting per role, including the prompt
cache hits that Crusoe's MemoryAlloy fabric reports.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import httpx

BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")

# Model per role. Everything is Nemotron except the skeptic, which is
# deliberately a different model family so its blind spots differ.
MODELS = {
    "lead": "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
    "contract": "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
    "field": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "archivist": "nvidia/Nemotron-3.5-Lightning-30B-A3B",
    "skeptic": "Qwen/Qwen3.8-27B",
    "baseline": "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
}
FALLBACK = {
    "nvidia/Nemotron-3.5-Lightning-30B-A3B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Qwen/Qwen3.8-27B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
}
DISPLAY = {
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL": "Nemotron 3.5 Super",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": "Nemotron 3 Nano",
    "nvidia/Nemotron-3.5-Lightning-30B-A3B": "Nemotron 3.5 Lightning",
    "Qwen/Qwen3.8-27B": "Qwen 3.8",
}
# $ per 1M tokens (prompt, completion, cached prompt), from GET /v1/models
PRICING = {
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL": (0.10, 0.40, 0.08),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": (0.30, 2.40, 0.15),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.05, 0.20, 0.03),
    "nvidia/Nemotron-3.5-Lightning-30B-A3B": (0.05, 0.20, 0.03),
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B": (0.30, 1.83, 0.30),
    "Qwen/Qwen3.8-27B": (0.25, 2.00, 0.15),
}


def load_env():
    """Load KEY=VALUE lines from $CAPY_ENV_FILE or ./.env if the key isn't set."""
    for path in (os.environ.get("CAPY_ENV_FILE"), ".env"):
        if path and Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, api_key: str | None = None, concurrency: int = 16):
        load_env()
        self.key = api_key or os.environ.get("CRUSOE_API_KEY")
        if not self.key:
            raise LLMError("CRUSOE_API_KEY is not set")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0))
        self._sems: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(concurrency))
        self.usage: dict[str, dict] = defaultdict(lambda: defaultdict(float))

    async def close(self):
        await self._client.aclose()

    async def chat(self, role: str, messages: list[dict], *, tools: list | None = None,
                   tool_choice=None, max_tokens: int = 2048, thinking: bool = True,
                   json_mode: bool = False, temperature: float = 0.3, model: str | None = None,
                   meter: str | None = None, usage: dict | None = None) -> dict:
        """One chat completion. Returns the assistant message plus `_model` and `_usage`."""
        model = model or MODELS[role]
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        delay = 1.0
        last_err = None
        for attempt in range(7):
            try:
                async with self._sems[model]:
                    t0 = time.time()
                    r = await self._client.post(f"{BASE_URL}/chat/completions", json=body,
                                                headers={"Authorization": f"Bearer {self.key}"})
                    dt = time.time() - t0
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = f"transport: {e}"
                r = None
            if r is not None and r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                self._account(self.usage, meter or role, model, data.get("usage") or {}, dt)
                if usage is not None:
                    self._account(usage, meter or role, model, data.get("usage") or {}, dt)
                msg["_model"] = model
                msg["_usage"] = data.get("usage") or {}
                msg["_finish"] = data["choices"][0].get("finish_reason")
                return msg
            if r is not None:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404, 422):
                    break
            if attempt >= 2 and model in FALLBACK:
                model = FALLBACK[model]
                body["model"] = model
            await asyncio.sleep(delay + random.random() * 0.5)
            delay = min(delay * 2, 20.0)
        raise LLMError(f"{role} ({model}) failed: {last_err}")

    @staticmethod
    def _account(store, role, model, usage, dt):
        if role not in store:
            store[role] = defaultdict(float)
        u = store[role]
        pt = usage.get("prompt_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or 0
        cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                     or usage.get("reasoning_tokens") or 0)
        price = PRICING.get(model, (0.3, 1.0, 0.3))
        cost = ((pt - cached) * price[0] + cached * price[2] + ct * price[1]) / 1e6
        u["calls"] += 1
        u["prompt_tokens"] += pt
        u["completion_tokens"] += ct
        u["cached_tokens"] += cached
        u["reasoning_tokens"] += reasoning
        u["cost_usd"] += cost
        u["latency_s"] += dt
        u["model"] = model

    def totals(self) -> dict:
        tot = defaultdict(float)
        for u in self.usage.values():
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    tot[k] += v
        return dict(tot)

    def snapshot(self) -> dict:
        return {role: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in u.items()}
                for role, u in self.usage.items()}

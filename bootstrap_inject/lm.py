"""Minimal LM client — an OpenAI-compatible chat callable with call accounting.

Self-contained (only httpx) so the package installs and runs with zero heavy
deps. The default backend is local ollama; any OpenAI-compatible endpoint works
via base_url / api_key. Behaviour is identical to the engine used for the
verified bet-0002 BBH runs (temperature 0, greedy, leading <think> stripped).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

OLLAMA_BASE = "http://localhost:11434/v1"


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LM:
    """Greedy OpenAI-compatible chat client. `lm(messages)` returns the reply text.

    Every call increments `usage.calls` — the rollout count is the efficiency
    metric the operator optimises (bootstrap-only = `len(trainset)` rollouts,
    no selection search on top).
    """

    model: str = "qwen2.5:3b-instruct"
    base_url: str = OLLAMA_BASE
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = None
    timeout: float = 300.0
    usage: Usage = field(default_factory=Usage)

    def __call__(self, messages: list[dict], **overrides) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
        }
        if self.seed is not None:
            body["seed"] = self.seed
        headers = {}
        key = self.api_key or os.environ.get("BI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(
            f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        self.usage.calls += 1
        u = data.get("usage") or {}
        self.usage.prompt_tokens += u.get("prompt_tokens", 0)
        self.usage.completion_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]


def strip_think(text: str) -> str:
    """Drop a leading <think>...</think> block (qwen3-style reasoning models)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()

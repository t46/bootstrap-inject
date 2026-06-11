"""Default task adapter: BBH-style exact-match (one canonical answer per item).

These three callables are the only task-specific surface BootstrapInject needs;
pass your own to `compile(...)` for other tasks/metrics. Kept identical to the
scorer used for the verified bet-0002 BBH runs so results reproduce.
"""

from __future__ import annotations

import re
import string

_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*", re.IGNORECASE)


def build_messages(system_prompt: str, ex: dict, no_think: bool = False) -> list[dict]:
    user = ex["input"]
    if no_think:
        user += " /no_think"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def extract_answer(completion: str) -> str:
    """Text after the last 'answer is'/'answer:' marker, else the last non-empty line."""
    parts = _ANSWER_RE.split(completion)
    if len(parts) > 1:
        tail = parts[-1].strip()
    else:
        lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
        tail = lines[-1] if lines else ""
    return tail.split("\n")[0].strip().rstrip(".").strip()


def normalize(s: str) -> str:
    s = s.strip().lower()
    m = re.fullmatch(r"\(?\s*([a-z])\s*\)?", s)
    if m:
        return m.group(1)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def score(pred: str, target: str) -> float:
    """Exact match after normalization; also credits a bare letter matching '(X)'."""
    np_, nt = normalize(pred), normalize(target)
    if np_ == nt:
        return 1.0
    tm = re.fullmatch(r"\(([a-z])\)", target.strip().lower())
    if tm and np_ == tm.group(1):
        return 1.0
    return 0.0


def evaluate(lm, prompt: str, examples: list[dict], *, no_think: bool = False) -> float:
    """Exact-match accuracy of `prompt` over `examples`. Used by ceiling_probe and tests."""
    from .lm import strip_think

    correct = 0
    for ex in examples:
        text = strip_think(lm(build_messages(prompt, ex, no_think)))
        correct += score(extract_answer(text), ex["target"]) >= 1.0
    return correct / max(1, len(examples))

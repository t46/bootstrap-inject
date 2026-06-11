"""Bootstrap-Inject: zero-search few-shot demo injection (the verified bet-0002 operator).

The contribution is a SUBTRACTION from BootstrapFewShot[WithRandomSearch]:

  BootstrapFewShot          : bootstrap correct traces, attach the first k.
  ...WithRandomSearch       : ALSO search over many candidate demo-sets, pick
                              the best by a validation score.        <-- deleted
  BootstrapInject (this)    : bootstrap correct traces, inject k at RANDOM, stop.

Why deleting the selection search is safe (verified, not assumed): at 3B scale
on BBH the validation-gate score of a demo-set is ~uncorrelated with its held-out
accuracy (gate-dev decorrelation, anomaly bet-0002-gate-dev-decorrelation). A
gate-0.45 demo set scored dev 0.67 while a gate-0.80 set scored 0.63 — the gate
ranks noise. So ALL the gain is in bootstrap+inject; the search only spends
rollouts. Deleting it costs ~0.2x the rollouts of random search at equal quality
(M1 VERIFIED, protocol p-3083b999 / p-c795222b: none 40ro dev 0.643 vs
random-search 200ro dev 0.650, equal on frozen test).

Pipeline:
  1. bootstrap_pool: run the seed prompt over the trainset; keep items the model
     gets RIGHT. Each is a self-generated worked exemplar (its own correct trace).
     Cost = len(trainset) rollouts, once. No leakage: only the trainset is touched.
  2. inject: prepend k pool exemplars (sampled with the run seed) to the seed
     prompt. No scoring, no gate, no candidate loop.

Use `ceiling_probe` (ceiling.py) FIRST as a pre-flight scope guard — injection
only helps when the task is elicitation-bound, not capability-bound.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .lm import LM, strip_think


@dataclass
class Demo:
    """A self-generated exemplar: a trainset input + the model's own correct trace."""

    id: str
    input: str
    trace: str
    gold: str


# --- step 1: bootstrap a pool of self-generated correct traces -------------------


def bootstrap_pool(
    lm: LM,
    seed_prompt: str,
    trainset: list[dict],
    *,
    pool_n: int | None = None,
    scorer=None,
    build_messages=None,
    extract_answer=None,
    no_think: bool = False,
    verbose: bool = False,
) -> list[Demo]:
    """Run `seed_prompt` over the first `pool_n` trainset items; keep the correct
    ones as self-generated exemplars.

    `trainset` items are dicts with keys: id, input, target. `scorer(pred, target)
    -> float` (>=1.0 == correct), `extract_answer(text) -> str`, and
    `build_messages(prompt, item, no_think) -> messages` adapt the package to a
    task. Defaults use the bundled BBH-style exact-match helpers (see tasks.py).
    """
    from . import tasks  # default task adapters

    scorer = scorer or tasks.score
    extract_answer = extract_answer or tasks.extract_answer
    build_messages = build_messages or tasks.build_messages

    items = trainset if pool_n is None else trainset[:pool_n]
    pool: list[Demo] = []
    for i, ex in enumerate(items):
        text = strip_think(lm(build_messages(seed_prompt, ex, no_think))).strip()
        pred = extract_answer(text)
        correct = scorer(pred, ex["target"]) >= 1.0
        if verbose:
            print(f"  bootstrap {i + 1}/{len(items)} correct={correct} rollouts={lm.usage.calls}",
                  flush=True)
        if correct:
            pool.append(Demo(id=ex["id"], input=ex["input"], trace=text, gold=ex["target"]))
    return pool


# --- step 2: inject k demos at random (no search) --------------------------------


def build_prompt(seed_prompt: str, demos: list[Demo]) -> str:
    """Prepend worked exemplars between the seed instructions and the new task.

    Exemplars are the model's own correct traces (already in the required output
    format, ending in the canonical answer line), so no reformatting is needed.
    """
    if not demos:
        return seed_prompt
    blocks = [
        f"Example {j}:\n{d.input.strip()}\n\n{d.trace.strip()}" for j, d in enumerate(demos, 1)
    ]
    return (
        seed_prompt.rstrip()
        + "\n\nHere are worked examples of the same kind of problem. Solve the new "
        "problem the same way, then finish with the required answer line.\n\n"
        + "\n\n".join(blocks)
    )


@dataclass
class CompileResult:
    prompt: str
    demos: list[Demo]
    pool_size: int
    rollouts_used: int
    seed: int
    history: list[dict] = field(default_factory=list)


class BootstrapInject:
    """Zero-search few-shot demo injection.

    >>> op = BootstrapInject(k=3, pool_n=40, seed=0)
    >>> res = op.compile(lm, seed_prompt, trainset)
    >>> res.prompt          # seed prompt + 3 injected self-generated exemplars
    >>> res.rollouts_used   # == number of bootstrap items run (no search rollouts)
    """

    def __init__(self, *, k: int = 3, pool_n: int = 40, seed: int = 0, no_think: bool = False):
        self.k = k
        self.pool_n = pool_n
        self.seed = seed
        self.no_think = no_think

    def compile(
        self,
        lm: LM,
        seed_prompt: str,
        trainset: list[dict],
        *,
        scorer=None,
        build_messages=None,
        extract_answer=None,
        verbose: bool = False,
    ) -> CompileResult:
        rng = random.Random(self.seed)
        pool = bootstrap_pool(
            lm, seed_prompt, trainset, pool_n=self.pool_n, scorer=scorer,
            build_messages=build_messages, extract_answer=extract_answer,
            no_think=self.no_think, verbose=verbose,
        )
        if not pool:
            raise ValueError(
                "bootstrap pool empty — the seed prompt gets nothing right on the "
                "trainset, so there is nothing to inject. The task is likely "
                "capability-bound (run ceiling_probe first)."
            )
        demos = rng.sample(pool, min(self.k, len(pool)))
        prompt = build_prompt(seed_prompt, demos)
        history = [
            {"event": "bootstrap", "pool_size": len(pool), "pool_n": self.pool_n,
             "rollouts": lm.usage.calls},
            {"event": "inject", "demo_ids": [d.id for d in demos], "search_rollouts": 0},
        ]
        return CompileResult(
            prompt=prompt, demos=demos, pool_size=len(pool),
            rollouts_used=lm.usage.calls, seed=self.seed, history=history,
        )

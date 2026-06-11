# bootstrap-inject

**Zero-search few-shot demo injection for small LMs.** Bootstrap correct traces,
inject *k* at random, stop. No validation/selection search.

It is a *subtraction* from DSPy's `BootstrapFewShot[WithRandomSearch]`, plus a
cheap pre-flight that tells you whether to bother. On 3B-class models it matches
random-search demo optimization at **~0.2× the rollouts** — because the search
it deletes was ranking noise.

```python
from bootstrap_inject import LM, BootstrapInject, ceiling_probe

lm = LM(model="qwen2.5:3b-instruct")              # local ollama by default

# 1. pre-flight: will demos help this task at all?  (scope guard, ~2*n rollouts)
v = ceiling_probe(lm, seed_prompt, heavy_scaffold_prompt, trainset, n=50)
print(v)   # -> PROCEED (elicitation-bound)  |  SKIP (capability-bound)

# 2. the operator: bootstrap correct traces, inject k at random, no search
if v.predicted_helps:
    res = BootstrapInject(k=3, pool_n=40, seed=0).compile(lm, seed_prompt, trainset)
    print(res.prompt)          # seed prompt + 3 self-generated worked exemplars
    print(res.rollouts_used)   # == pool_n; search_rollouts == 0
```

DSPy users get a drop-in teleprompter (`pip install bootstrap-inject[dspy]`):

```python
from bootstrap_inject.dspy_teleprompter import BootstrapInjectTeleprompter
compiled = BootstrapInjectTeleprompter(metric, k=3, max_pool=40).compile(student, trainset=trainset)
```

## The one idea

DSPy's few-shot optimizers do two things: (1) **bootstrap** — run a teacher over
the trainset and keep demonstrations it gets right; (2) **select** — search over
many candidate demo-sets and keep the one that scores best on a validation set.
`BootstrapFewShotWithRandomSearch` is mostly step (2).

On small models, **step (2) is noise.** We measured the validation-gate score of
a demo-set against its held-out accuracy and found them ~uncorrelated: a
gate-0.45 demo-set scored dev **0.67** while a gate-0.80 set scored **0.63**. The
gate ranks demo-sets no better than chance, so every rollout spent searching is
wasted. Delete it. Bootstrap a pool of correct self-generated traces, inject *k*
of them at random, and ship.

| optimizer | bootstrap | selection search | rollouts (this setup) |
|---|---|---|---|
| `BootstrapFewShot` | ✓ | first-k, no search | ~`max_pool` |
| `BootstrapFewShotWithRandomSearch` | ✓ | search demo-sets on valset | **~200** |
| **`BootstrapInject`** | ✓ | **none** | **~40 (`pool_n`)** |

## When it helps — and when it doesn't (read this)

Injection is **conditional**, and the condition is *not* "low baseline." It helps
when a task is **elicitation-bound** (the model can do it but the naive prompt
under-elicits the reasoning) and not when it is **capability-bound** (the model
cannot do it; demos only distract) or when the **bootstrap pool is thin**
(self-demos drawn only from easy items don't transfer to the hard ones).

Verified on BBH subtasks, `qwen2.5:3b-instruct`, dev[:100], 3 seeds:

| task | structure | competent baseline | injection (k=3, 40 ro) | Δ | verdict |
|---|---|---|---|---|---|
| logical_deduction_three_objects | 3-obj ordering | 0.55 | 0.643 | **+0.093** | helps |
| date_understanding | temporal 5-way MC | 0.44 | 0.513 | **+0.073** | helps ✓ verified |
| logical_deduction_five_objects | 5-obj ordering | 0.41 | 0.413 | +0.003 | flat (thin pool 12/40) |
| navigate | spatial yes/no | 0.74 | 0.670 | −0.07 | regresses (capability-bound) |

The win generalizes across *structure* (ordering → temporal arithmetic), so it is
not memorized to one task family. But `navigate` regresses and `five_objects` is
flat — baseline level alone (0.74 vs 0.41) does not tell you which.

### The scope guard

`ceiling_probe` operationalizes the discriminator with two cheap evals: the
competent seed vs a **heavy-scaffolding** ("do it step-by-step like this") prompt
on a small slice.

- `ceiling_acc > baseline_acc` → headroom is elicitation → **injection predicted to help**.
- `ceiling_acc ≤ baseline_acc` → the model can't do it even when told exactly how
  → **capability-bound, skip.**

This flagged `navigate` *before* any seed sweep: its heavy-scaffold prompt scored
**0.57 < 0.74** baseline — explicit coordinate tracking *hurts* a 3B model. It is
a predictor, not a proof: pool representativeness is a second condition the probe
does not see (`five_objects` passes the probe but stays flat on a 12/40 pool).

## Evidence

All numbers come from the AAD harness (pre-registered protocol, re-executed from
a clean checkout of the recorded commit). bet-0002, `qwen2.5:3b-instruct`:

- **M1 efficiency** — protocol `p-3083b999`: `rollouts_used == 40` reproduced ×3
  seeds (`r-a15a971f3a`, `r-414130a605`, `r-8554146a2a`). vs random-search 200.
- **M1 quality** — protocol `p-c795222b`: mean dev **0.643** reproduced
  (`r-1d28105667`), vs random-search 0.650 — equal at 0.2× rollouts; on frozen
  test, injection ≥ random every seed.
- **M2 generalization** — protocol `p-29bcb252`: date_understanding mean dev
  **0.513** (Δ+0.073) reproduced from clean checkout (`r-de42201882`, verified).
- **gate–dev decorrelation** (why selection is deletable): `r-9070d71cfa`
  (gate-0.45 → dev 0.67) vs `r-cd8122be8f` (gate-0.80 → dev 0.63).

This is a small-model, prompt-elicitation result. It is **not** claimed for large
models, for tasks with multi-answer/partial-credit metrics, or beyond the scope
the probe gates. Rigor over reach.

## Install / test

```bash
pip install bootstrap-inject            # core (httpx only)
pip install bootstrap-inject[dspy]      # + DSPy teleprompter
pytest                                  # offline tests (no network)
uv run --with dspy-ai --with httpx python examples/dspy_integration.py  # live DSPy wiring
```

## License

MIT.

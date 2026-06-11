"""Ceiling probe — the cheap pre-flight scope guard for BootstrapInject.

The verified scope condition (bet-0002 M2, claim
bet-0002-m2-injection-generalizes-conditionally): zero-search injection helps
iff the task is ELICITATION-bound, not CAPABILITY-bound. Baseline level alone
does NOT predict this (five_objects 0.41 flat; date 0.44 helps; navigate 0.74
regresses). The discriminator is whether the model CAN do the task when handed a
heavy-scaffolding prompt:

  ceiling_acc  = accuracy of a heavy step-by-step "do-it-this-way" scaffold prompt
  baseline_acc = accuracy of the competent seed

  ceiling_acc <= baseline_acc  ==>  capability-bound: scaffold/demos can't lift it,
                                    injection is predicted NOT to help. SKIP.
  ceiling_acc >  baseline_acc  ==>  headroom is elicitation; injection is predicted
                                    to help. PROCEED.

This correctly flagged navigate before any seed sweep (ceiling 0.57 < baseline
0.74, r-491076f4af) and is the charter triage precondition made into one cheap
call-bounded check. It is a PREDICTOR, not a guarantee — pool representativeness
is a second condition (a thin bootstrap pool, e.g. five_objects 12/40, can still
leave injection flat even when the probe says proceed).
"""

from __future__ import annotations

from dataclasses import dataclass

from .lm import LM
from . import tasks


@dataclass
class CeilingVerdict:
    baseline_acc: float
    ceiling_acc: float
    n: int
    rollouts_used: int
    predicted_helps: bool

    @property
    def margin(self) -> float:
        return self.ceiling_acc - self.baseline_acc

    def __str__(self) -> str:
        verdict = "PROCEED (elicitation-bound)" if self.predicted_helps else "SKIP (capability-bound)"
        return (f"ceiling_probe: baseline={self.baseline_acc:.3f} ceiling={self.ceiling_acc:.3f} "
                f"margin={self.margin:+.3f} n={self.n} -> {verdict}")


def ceiling_probe(
    lm: LM,
    seed_prompt: str,
    ceiling_prompt: str,
    probe_set: list[dict],
    *,
    n: int = 50,
    evaluate=None,
    no_think: bool = False,
) -> CeilingVerdict:
    """Compare a competent seed against a heavy-scaffolding ceiling prompt on a small
    probe slice (default first 50 trainset items, 2*n rollouts total). Returns a
    verdict whose `predicted_helps` is the go/no-go for running BootstrapInject.
    """
    evaluate = evaluate or tasks.evaluate
    items = probe_set[:n]
    baseline = evaluate(lm, seed_prompt, items, no_think=no_think)
    ceiling = evaluate(lm, ceiling_prompt, items, no_think=no_think)
    return CeilingVerdict(
        baseline_acc=baseline,
        ceiling_acc=ceiling,
        n=len(items),
        rollouts_used=lm.usage.calls,
        predicted_helps=ceiling > baseline,
    )

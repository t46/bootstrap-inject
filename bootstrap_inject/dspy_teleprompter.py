"""DSPy-compatible teleprompter — the drop-in integration path.

`BootstrapInjectTeleprompter` is a `dspy.teleprompt.Teleprompter` subclass that
implements the same SUBTRACTION as the standalone core: it bootstraps correct
demonstrations with a teacher, then injects `k` of them AT RANDOM into every
predictor — with NO validation/selection search over candidate demo-sets.

It is the rollout-cheaper sibling of DSPy's own optimizers:

  BootstrapFewShot               attach first k bootstrapped demos
  BootstrapFewShotWithRandomSearch   + search many demo-sets, keep best on valset
  BootstrapInjectTeleprompter (this) bootstrap, inject k at random, STOP   <-- here

The random-search valset selection is what this deletes. On 3B-class models the
valset gate is ~uncorrelated with held-out accuracy (gate-dev decorrelation,
bet-0002), so the search spends rollouts for no quality — deleting it gives
~0.2x the rollouts at equal accuracy (M1 VERIFIED on the standalone path).

dspy is an OPTIONAL dependency: `pip install bootstrap-inject[dspy]`. The import
is lazy so the standalone core works without it. Run `ceiling_probe` first.
"""

from __future__ import annotations

import random


def _require_dspy():
    try:
        import dspy  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "BootstrapInjectTeleprompter needs DSPy. Install with "
            "`pip install bootstrap-inject[dspy]`."
        ) from e
    return dspy


def make_teleprompter():
    """Build the Teleprompter subclass against the installed dspy (lazy)."""
    dspy = _require_dspy()
    from dspy.teleprompt import Teleprompter

    class BootstrapInjectTeleprompter(Teleprompter):
        """Bootstrap correct demos, inject k at random, no selection search.

        Args:
            metric:  dspy metric ``(example, prediction, trace=None) -> bool/float``.
            k:       demos injected per predictor (default 3).
            max_pool: trainset items to attempt for the bootstrap pool (default 40);
                      this is the rollout cost — there is no search on top of it.
            seed:    RNG seed for which k pool demos are injected.
        """

        def __init__(self, metric, *, k: int = 3, max_pool: int = 40, seed: int = 0):
            self.metric = metric
            self.k = k
            self.max_pool = max_pool
            self.seed = seed

        def compile(self, student, *, trainset, teacher=None):
            teacher = teacher or student.reset_copy()
            rng = random.Random(self.seed)

            # --- step 1: bootstrap a pool of correct, fully-traced demonstrations ---
            pool = []  # list[dict predictor_name -> dspy.Example]
            for ex in trainset[: self.max_pool]:
                try:
                    with dspy.context(trace=[]):
                        pred = teacher(**ex.inputs())
                        trace = dspy.settings.trace
                    ok = self.metric(ex, pred, trace)
                except Exception:
                    ok = False
                    trace = []
                if not ok:
                    continue
                demo_by_pred = {}
                for step_pred, inputs, outputs in trace:
                    name = _predictor_name(teacher, step_pred)
                    demo_by_pred[name] = dspy.Example(augmented=True, **inputs, **outputs)
                if demo_by_pred:
                    pool.append(demo_by_pred)

            if not pool:
                raise ValueError(
                    "bootstrap pool empty — the student gets nothing right on the "
                    "trainset; the task is likely capability-bound (run ceiling_probe)."
                )

            # --- step 2: inject k pooled demos at random into every predictor ----
            chosen = rng.sample(pool, min(self.k, len(pool)))
            for name, predictor in student.named_predictors():
                predictor.demos = [d[name] for d in chosen if name in d]

            student._compiled = True
            return student

    def _predictor_name(program, predictor) -> str:
        for name, p in program.named_predictors():
            if p is predictor:
                return name
        return "self"

    return BootstrapInjectTeleprompter


def BootstrapInjectTeleprompter(*args, **kwargs):
    """Convenience constructor: builds and instantiates the teleprompter."""
    return make_teleprompter()(*args, **kwargs)

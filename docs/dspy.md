# DSPy integration

`bootstrap-inject` exposes a DSPy-compatible teleprompter for users who already
have a DSPy module, trainset, and metric. It bootstraps correct teacher traces,
samples up to `k` successful traces, and attaches them as predictor demos. It
does not run validation-set random search over candidate demo sets.

## Install

```bash
pip install "bootstrap-inject[dspy]"
```

For local checkout testing:

```bash
uv run --with "bootstrap-inject[dspy]" python your_script.py
```

## Minimal use

```python
from bootstrap_inject.dspy_teleprompter import BootstrapInjectTeleprompter

teleprompter = BootstrapInjectTeleprompter(metric, k=3, max_pool=40, seed=0)
compiled = teleprompter.compile(student, trainset=trainset)
```

`student` is a normal DSPy module with predictors. `trainset` is a list of DSPy
examples with inputs marked through `.with_inputs(...)`. `metric` should return
true only when the prediction is usable as a demonstration.

## Offline wiring smoke

The adapter can be smoke-tested without calling a live LM:

1. Instantiate `BootstrapInjectTeleprompter` with `k=2` and `max_pool=6`.
2. Compile a small DSPy module over six examples.
3. Assert that compile marks the student as compiled, attaches two demos to the
   target predictor, and calls the metric once per pool example.

Expected result:

```text
compiled._compiled == True
len(dict(compiled.named_predictors())["predict"].demos) == 2
metric_calls == 6
```

## Scope

This adapter preserves the core `BootstrapInject` behavior for DSPy users: no
search over demo sets, only bootstrapping and random injection from successful
traces. The smoke test validates adapter wiring. It is not evidence of live
`dspy-ai` model behavior, task quality gains, or package publication.

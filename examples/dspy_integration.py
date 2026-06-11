"""Live DSPy integration example: optimize a tiny program with BootstrapInject.

Run (needs the optional dspy extra + a local ollama):
  uv run --with dspy-ai --with httpx python examples/dspy_integration.py

Proves the teleprompter bootstraps correct traces and attaches k demos to the
predictor with NO valset/selection call. Not a numeric claim — a wiring check.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy

from bootstrap_inject.dspy_teleprompter import BootstrapInjectTeleprompter


def main() -> None:
    # point DSPy at the local ollama OpenAI-compatible endpoint
    lm = dspy.LM("openai/qwen2.5:3b-instruct", api_base="http://localhost:11434/v1",
                 api_key="x", temperature=0.0, max_tokens=512)
    dspy.configure(lm=lm)

    class QA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.solve(question=question)

    # tiny arithmetic trainset (the model gets these right -> non-empty pool)
    pairs = [("2+2", "4"), ("3+5", "8"), ("10-4", "6"), ("6+1", "7"),
             ("9-3", "6"), ("4+4", "8")]
    trainset = [dspy.Example(question=q, answer=a).with_inputs("question") for q, a in pairs]

    def metric(ex, pred, trace=None):
        return str(ex.answer).strip() in str(pred.answer)

    tp = BootstrapInjectTeleprompter(metric, k=2, max_pool=6, seed=0)
    compiled = tp.compile(QA(), trainset=trainset)

    name, predictor = next(iter(compiled.named_predictors()))
    demos = predictor.demos
    print(f"attached demos to predictor {name!r}: {len(demos)}")
    for d in demos:
        print(f"  q={d.get('question')!r} answer={d.get('answer')!r} augmented={d.get('augmented')}")
    assert len(demos) == 2, "expected k=2 injected demos"
    assert all(d.get("augmented") for d in demos), "demos should be bootstrapped (augmented)"
    print("OK: BootstrapInject teleprompter bootstrapped + injected k=2 demos, zero search.")


if __name__ == "__main__":
    main()

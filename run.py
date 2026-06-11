"""End-to-end demo / smoke: ceiling_probe -> BootstrapInject -> evaluate, on a
frozen BBH split. This is the shippable operator exercised exactly as a user
would, and the harness-recorded T0 smoke that proves the packaged code
reproduces the verified mechanism.

Usage (via the AAD harness):
  aad exp run --bet bet-0002 --tier t0 -- \
    uv run --frozen --with httpx --no-project python bootstrap_inject/run.py \
      --data-dir data/bbh --subtask logical_deduction_three_objects \
      --seed-prompt prompts/competent_seed.txt --k 3 --pool-n 20 \
      --probe-n 0 --eval-n 20 --model qwen2.5:3b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_inject import LM, BootstrapInject, ceiling_probe  # noqa: E402
from bootstrap_inject.tasks import evaluate  # noqa: E402


def load_split(data_dir: Path, subtask: str, name: str) -> list[dict]:
    path = data_dir / subtask / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--seed-prompt", required=True)
    ap.add_argument("--ceiling-prompt", default=None,
                    help="heavy-scaffold prompt for the pre-flight probe; skip probe if omitted")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool-n", type=int, default=40)
    ap.add_argument("--probe-n", type=int, default=0, help="ceiling probe slice (0 = skip)")
    ap.add_argument("--eval-n", type=int, default=50, help="dev-eval slice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    opt = load_split(data_dir, args.subtask, "opt")
    dev = load_split(data_dir, args.subtask, "dev")
    seed_prompt = Path(args.seed_prompt).read_text()

    lm = LM(model=args.model, max_tokens=args.max_tokens)
    metrics: dict = {"subtask": args.subtask, "k": args.k, "pool_n": args.pool_n,
                     "seed": args.seed, "model": args.model}

    # pre-flight scope guard (optional)
    if args.probe_n and args.ceiling_prompt:
        verdict = ceiling_probe(lm, seed_prompt, Path(args.ceiling_prompt).read_text(),
                                opt, n=args.probe_n)
        print(str(verdict), flush=True)
        metrics.update(probe_baseline=round(verdict.baseline_acc, 4),
                       probe_ceiling=round(verdict.ceiling_acc, 4),
                       probe_predicts_helps=int(verdict.predicted_helps))

    # competent baseline for delta (on the eval slice)
    base_lm = LM(model=args.model, max_tokens=args.max_tokens)
    baseline_acc = evaluate(base_lm, seed_prompt, dev[: args.eval_n])

    # the operator
    op = BootstrapInject(k=args.k, pool_n=args.pool_n, seed=args.seed)
    res = op.compile(lm, seed_prompt, opt, verbose=True)
    inj_lm = LM(model=args.model, max_tokens=args.max_tokens)
    injected_acc = evaluate(inj_lm, res.prompt, dev[: args.eval_n])

    metrics.update(
        baseline_acc=round(baseline_acc, 4),
        injected_acc=round(injected_acc, 4),
        delta=round(injected_acc - baseline_acc, 4),
        pool_size=res.pool_size,
        rollouts_used=res.rollouts_used,
        search_rollouts=0,
        n_demos=len(res.demos),
        eval_n=len(dev[: args.eval_n]),
    )
    print(json.dumps(metrics, indent=2))

    mp = os.environ.get("AAD_METRICS_PATH")
    if mp:
        Path(mp).write_text(json.dumps(metrics) + "\n")


if __name__ == "__main__":
    main()

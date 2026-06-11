"""bootstrap-inject: zero-search few-shot demo injection for small LMs.

The verified bet-0002 operator. Two public pieces:

  ceiling_probe(...)      pre-flight scope guard — predicts whether injection helps
  BootstrapInject(...)    the operator — bootstrap correct traces, inject k at random

DSPy integration (optional): `from bootstrap_inject.dspy_teleprompter import
BootstrapInjectTeleprompter`.
"""

from .lm import LM, Usage, strip_think
from .core import BootstrapInject, CompileResult, Demo, bootstrap_pool, build_prompt
from .ceiling import CeilingVerdict, ceiling_probe

__all__ = [
    "LM", "Usage", "strip_think",
    "BootstrapInject", "CompileResult", "Demo", "bootstrap_pool", "build_prompt",
    "CeilingVerdict", "ceiling_probe",
]

__version__ = "0.1.0"

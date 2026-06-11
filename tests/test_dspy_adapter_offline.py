"""Offline DSPy adapter test.

This uses a tiny in-process fake `dspy` module so CI can verify the adapter
without installing DSPy or calling a live LM.
"""

from __future__ import annotations

import copy
import sys
import types


def test_dspy_teleprompter_bootstraps_and_attaches_demos(monkeypatch):
    dspy = types.ModuleType("dspy")

    class _Settings:
        trace = []

    dspy.settings = _Settings()

    class _Context:
        def __init__(self, trace=None):
            self.trace = [] if trace is None else trace

        def __enter__(self):
            dspy.settings.trace = self.trace
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def context(trace=None):
        return _Context(trace)

    class Prediction:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Example:
        def __init__(self, **kwargs):
            self._input_keys = set()
            self.__dict__.update(kwargs)

        def with_inputs(self, *keys):
            self._input_keys = set(keys)
            return self

        def inputs(self):
            return {key: getattr(self, key) for key in self._input_keys}

    class Predict:
        def __init__(self, signature):
            self.signature = signature
            self.demos = []

        def __call__(self, **kwargs):
            prediction = Prediction(answer="a")
            dspy.settings.trace.append((self, kwargs, {"answer": prediction.answer}))
            return prediction

    class Module:
        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def reset_copy(self):
            return copy.deepcopy(self)

        def named_predictors(self):
            for name, value in self.__dict__.items():
                if isinstance(value, Predict):
                    yield name, value

    dspy.context = context
    dspy.Example = Example
    dspy.Module = Module
    dspy.Predict = Predict

    teleprompt = types.ModuleType("dspy.teleprompt")

    class Teleprompter:
        pass

    teleprompt.Teleprompter = Teleprompter

    monkeypatch.setitem(sys.modules, "dspy", dspy)
    monkeypatch.setitem(sys.modules, "dspy.teleprompt", teleprompt)

    from bootstrap_inject.dspy_teleprompter import BootstrapInjectTeleprompter

    class Echo(Module):
        def __init__(self):
            self.predict = Predict("question -> answer")

        def forward(self, question):
            return self.predict(question=question)

    trainset = [
        Example(question=f"q{idx}", answer="a").with_inputs("question")
        for idx in range(6)
    ]
    metric_calls = 0

    def metric(example, prediction, trace=None):
        nonlocal metric_calls
        metric_calls += 1
        return prediction.answer == example.answer and bool(trace)

    student = Echo()
    teleprompter = BootstrapInjectTeleprompter(metric, k=2, max_pool=6, seed=0)
    compiled = teleprompter.compile(student, trainset=trainset)

    predictors = dict(compiled.named_predictors())
    demos = predictors["predict"].demos

    assert compiled is student
    assert compiled._compiled is True
    assert metric_calls == 6
    assert len(demos) == 2
    assert all(demo.augmented is True for demo in demos)
    assert {demo.answer for demo in demos} == {"a"}

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for loading GGUF weights into layers with a legacy weight loader.

``quantization/replicated.py`` has no vLLM dependency, so these tests run
without vLLM (the module is loaded by path in that case). The real
``ReplicatedLinear`` is covered in ``test_plugin.py``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

try:
    from vllm_gguf_plugin.quantization import replicated
except ImportError:  # no vLLM on this machine: load the pure module by path
    _path = (
        Path(__file__).resolve().parents[1]
        / "vllm_gguf_plugin"
        / "quantization"
        / "replicated.py"
    )
    _spec = importlib.util.spec_from_file_location("gguf_replicated", _path)
    replicated = importlib.util.module_from_spec(_spec)
    sys.modules["gguf_replicated"] = replicated
    _spec.loader.exec_module(replicated)


class FakeGGUFParameter(UninitializedParameter):
    """Stand-in for GGUFUninitializedWeightParameter (materialized on _store)."""

    cls_to_become = torch.nn.Parameter

    def _store(self, loaded_weight):
        self.materialize(tuple(loaded_weight.shape), dtype=loaded_weight.dtype)
        self.data.copy_(loaded_weight)


def replicated_linear_weight_loader(param, loaded_weight):
    """vLLM ReplicatedLinear.weight_loader (vllm/model_executor/layers/linear.py)."""
    if len(loaded_weight.shape) == 0:
        loaded_weight = loaded_weight.reshape(1)
    assert param.size() == loaded_weight.size(), (
        f"Tried to load weights of size {loaded_weight.size()}"
        f"to a parameter of size {param.size()}"
    )
    param.data.copy_(loaded_weight)


def test_unwrapped_legacy_loader_fails_on_uninitialized_gguf_weight():
    param = FakeGGUFParameter(requires_grad=False)
    assert param.size() == torch.Size([0])
    with pytest.raises(AssertionError, match="Tried to load weights"):
        replicated_linear_weight_loader(param, torch.ones((8, 34), dtype=torch.uint8))


def test_wrapped_legacy_loader_stores_packed_weight():
    loader = replicated.wrap_legacy_weight_loader(replicated_linear_weight_loader)
    param = FakeGGUFParameter(requires_grad=False)
    packed = torch.arange(8 * 34, dtype=torch.uint8).reshape(8, 34)

    loader(param, packed)

    assert param.shape == (8, 34) and param.dtype == torch.uint8
    assert torch.equal(param.data, packed)


def test_wrapped_legacy_loader_stores_weight_type():
    loader = replicated.wrap_legacy_weight_loader(replicated_linear_weight_loader)
    param = FakeGGUFParameter(requires_grad=False)
    loader(param, torch.tensor(8, dtype=torch.uint8).reshape(1))
    assert param.tolist() == [8]


def test_wrapped_legacy_loader_leaves_plain_parameters_to_the_base_loader():
    loader = replicated.wrap_legacy_weight_loader(replicated_linear_weight_loader)
    bias = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    loader(bias, torch.arange(4, dtype=torch.float32))
    assert bias.tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(AssertionError):  # base loader checks are kept
        loader(bias, torch.zeros(5))


def test_wrapping_is_idempotent():
    once = replicated.wrap_legacy_weight_loader(replicated_linear_weight_loader)
    assert replicated.wrap_legacy_weight_loader(once) is once

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loading for layers that only have a legacy ``weight_loader``.

vLLM's ``ReplicatedLinear`` has no ``weight_loader_v2``. Its ``weight_loader``
asserts ``param.size() == loaded_weight.size()`` and copies into ``param.data``,
which cannot work for a GGUF weight: the parameter is an uninitialized one
(size ``[0]``) that is materialized by the first loaded tensor, and the loaded
tensor is the packed ``[rows, bytes_per_row]`` data. This module is free of
vLLM imports so that it can be tested without vLLM.
"""

from collections.abc import Callable

import torch

_WRAPPED_ATTR = "_gguf_legacy_loader_wrapped"


def wrap_legacy_weight_loader(base_loader: Callable) -> Callable:
    """Make a ``weight_loader(param, loaded_weight)`` GGUF-aware.

    GGUF parameters (anything with a ``_store`` method) store the loaded tensor
    themselves, which materializes them; other parameters (e.g. a bias) go to
    ``base_loader`` unchanged. Replicated layers are not sharded, so the tensor
    is stored as is.
    """
    if getattr(base_loader, _WRAPPED_ATTR, False):
        return base_loader

    def gguf_legacy_weight_loader(
        param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs
    ):
        if hasattr(param, "_store") and not args and not kwargs:
            param._store(loaded_weight)
            return
        return base_loader(param, loaded_weight, *args, **kwargs)

    setattr(gguf_legacy_weight_loader, _WRAPPED_ATTR, True)
    return gguf_legacy_weight_loader

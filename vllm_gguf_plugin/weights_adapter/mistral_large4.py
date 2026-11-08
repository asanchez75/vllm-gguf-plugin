# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for Mistral Large 4 GGUF files (``general.architecture`` is
``mistral-large4``).

The GGUF tensors are renamed to the Mistral consolidated checkpoint names that
``MistralLarge4ForCausalLM.load_weights`` consumes, so the model's own loader
(stacked shards, per-expert mapping, ``lerope``, missing-parameter audit) does
the placement. The pure logic lives in :mod:`.mistral_large4_mapping`; this
class only plugs it into the plugin hooks and converts between torch and numpy.

Experts: the stacked ``[E, rows, bytes]`` tensors are delivered whole under the
name of expert 0; ``RoutedExperts.weight_loader`` has a 3-D "full load" branch
and the plugin's ``_materialize_gguf_moe_param`` sizes the parameter from a 3-D
tensor, which turns ~27,000 per-expert loader calls into 213. Set
``VLLM_GGUF_MISTRAL_SPLIT_EXPERTS=1`` to fall back to per-expert views
(zero-copy ``unbind``), the path the other MoE adapters use.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from ..gguf_files import GGUFModelFiles
from .base import BaseGGUFWeightsAdapter, GGUFWeight
from .mistral_large4_mapping import (
    GGUF_ARCHITECTURE,
    HF_MODEL_TYPE,
    VLLM_ARCHITECTURE,
    StreamTransformer,
    audit_inventory,
    expert_names,
    expert_quant_types,
    read_gguf_architecture,
    read_gguf_inventory,
    split_stacked,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

#: Deliver routed experts one by one (views of the stacked tensor).
SPLIT_EXPERTS_ENV = "VLLM_GGUF_MISTRAL_SPLIT_EXPERTS"


def _is_mistral_large4(config: object) -> bool:
    if getattr(config, "model_type", None) == HF_MODEL_TYPE:
        return True
    return VLLM_ARCHITECTURE in (getattr(config, "architectures", None) or ())


def _to_numpy(tensor: torch.Tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


class MistralLarge4GGUFAdapter(BaseGGUFWeightsAdapter):
    """Adapter for the ``mistral-large4`` GGUF architecture."""

    def __init__(self) -> None:
        self._inventory = None

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        return _is_mistral_large4(config)

    @classmethod
    def architecture(cls, config: PretrainedConfig) -> str | None:
        del config
        return VLLM_ARCHITECTURE

    def _get_inventory(self, files: GGUFModelFiles):
        if self._inventory is None:
            self._inventory = read_gguf_inventory(files.backbone)
        return self._inventory

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        architecture = read_gguf_architecture(files.primary_backbone)
        if architecture != GGUF_ARCHITECTURE:
            raise ValueError(
                f"{files.primary_backbone} has general.architecture="
                f"{architecture!r}, expected {GGUF_ARCHITECTURE!r} for a "
                f"{HF_MODEL_TYPE} model config."
            )
        return hf_config

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        inventory = self._get_inventory(files)
        text_config = model_config.hf_config.get_text_config()
        # Raises with the full list of problems if the GGUF and the model
        # disagree (missing/unexpected tensors, shapes, gate/up types, vocab).
        name_map = audit_inventory(inventory, text_config)
        groups: dict[tuple[int, ...], int] = {}
        for types in expert_quant_types(inventory).values():
            groups[types] = groups.get(types, 0) + 1
        logger.info(
            "Mistral Large 4 GGUF: %d tensors mapped; expert quantization types "
            "(gate, up, down) -> layers: %s",
            len(name_map),
            groups,
        )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        dtype = model_config.dtype
        split_experts = os.environ.get(SPLIT_EXPERTS_ENV) == "1"
        stream = StreamTransformer()
        for name, weight in weights:
            if name.endswith(".weight_type") and weight.ndim == 0:
                if not stream.consume_type(name, int(weight)):
                    yield name, weight
            elif stream.handles(name):
                for item in stream.feed(name, _to_numpy(weight)):
                    tensor = torch.from_numpy(item.array)
                    yield item.name, tensor if item.keep_float32 else tensor.to(dtype)
            elif split_experts and weight.ndim == 3 and ".experts.0." in name:
                yield from zip(
                    expert_names(name, weight.shape[0]), split_stacked(weight)
                )
            else:
                yield name, weight
        stream.finish()

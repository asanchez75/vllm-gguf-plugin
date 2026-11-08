# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from .weights_adapter import get_adapter_architecture
from .weights_adapter.mistral_large4_mapping import (
    is_mistral_large4_params,
    strip_params_for_gguf,
)


def _parse_mistral_large4_params(
    config_dir: str | Path,
) -> tuple[dict, PretrainedConfig] | None:
    """Build the Mistral Large 4 config from ``params.json`` next to the GGUF.

    The HF ``config.json`` of this model cannot be used (transformers does not
    know ``mistral_large4`` and the file lacks the MoE/SSMax/LeRoPE fields), so
    vLLM's Mistral-format conversion is used instead. The FP8 ``quantization``
    block and the vision blocks are dropped in memory (the file on disk is not
    touched) and no ``config.json`` defaults are merged: they would bring the
    FP8 ``quantization_config`` back.

    Returns ``None`` if ``config_dir`` is not a local directory with such a
    ``params.json``.
    """
    params_file = Path(config_dir) / "params.json"
    if not params_file.is_file():
        return None
    with open(params_file, encoding="utf-8") as f:
        params = json.load(f)
    if not is_mistral_large4_params(params):
        return None

    from vllm.transformers_utils.configs.mistral import adapt_config_dict

    config_dict = strip_params_for_gguf(params)
    if "max_position_embeddings" not in config_dict:
        config_dict["max_position_embeddings"] = _read_max_positions(config_dir)
    config = adapt_config_dict(config_dict, defaults={})
    return config_dict, config


def _read_max_positions(config_dir: str | Path) -> int:
    """``max_position_embeddings`` from ``config.json`` if present, else 128000.

    Only this one number is taken from ``config.json`` (vLLM's default for a
    ``params.json`` without it is 128000).
    """
    try:
        with open(Path(config_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        return int((cfg.get("text_config") or cfg)["max_position_embeddings"])
    except (OSError, KeyError, TypeError, ValueError):
        return 128_000


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        if (parsed := _parse_mistral_large4_params(resolved_model)) is not None:
            return parsed
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        architecture = get_adapter_architecture(config)
        if (
            architecture is None
            and config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        ):
            architecture = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        if architecture is None:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        config_dict["architectures"] = [architecture]
        config.update({"architectures": [architecture]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model

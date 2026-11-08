# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure (vLLM-free) logic of the Mistral Large 4 GGUF adapter.

Nothing in this module imports vLLM, the plugin package or torch, so it can be
unit tested on a machine without vLLM. It only needs ``numpy`` and ``gguf``.

Design ("Option A"): GGUF tensors of the ``mistral-large4`` architecture are
renamed to the Mistral *consolidated* checkpoint names that the model's own
``load_weights`` (vLLM ``models/mistral/mistral_large4/nvidia/model.py``,
``_CHECKPOINT_NAME_MAP`` / ``_STACKED_PARAMS`` / ``_load_weights``) consumes.
Quantized tensors keep their packed bytes and are delivered through the plugin's
usual ``weight_type`` + ``weight`` convention. Modules that the model builds
without a quantization config get dense tensors instead:

* ``attn_kv_a_mqa`` (Q8_0)  -> ``attention.wkv_a_with_mqa`` dequantized
* ``attn_k_b`` + ``attn_v_b`` (Q8_0) -> ``attention.wkv_b`` fused + dequantized
* ``indexer.attn_k`` (Q8_0) -> ``attention.indexer.wk`` dequantized
* ``rope_freqs`` (F32)      -> ``lerope.alpha = -log(rope_freqs)``

The tensor names and shapes are those of llama.cpp's ``mistral-large4`` GGUF
(``conversion/mistral4.py``, ``gguf-py`` constants). The installed ``gguf``
package may not know this architecture, so the table below is explicit and no
``gguf.MODEL_ARCH_NAMES`` / ``get_tensor_name_map`` lookup is used.
"""

from __future__ import annotations

import enum
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize

GGUF_ARCHITECTURE = "mistral-large4"
#: Debug switch: accept a GGUF that has more layers than the config (a
#: truncated debug model with a reduced ``num_hidden_layers``). The tensors of
#: the deeper layers are left out of the name map, so they are never read. The
#: model's own loader skips them as well.
ALLOW_TRUNCATED_ENV = "VLLM_GGUF_MISTRAL_ALLOW_TRUNCATED"
HF_MODEL_TYPE = "mistral_large_4"
VLLM_ARCHITECTURE = "MistralLarge4ForCausalLM"

#: Types that carry plain floats (no ``weight_type`` companion is streamed).
FLOAT_QTYPES = frozenset(
    {
        int(GGMLQuantizationType.F32),
        int(GGMLQuantizationType.F16),
        int(GGMLQuantizationType.BF16),
    }
)

# ---------------------------------------------------------------------------
# Name table
# ---------------------------------------------------------------------------


class Kind(enum.Enum):
    """How a GGUF tensor reaches the model."""

    #: Renamed only; bytes and ``weight_type`` are passed through.
    RENAME = "rename"
    #: Dequantized to a dense tensor (module is built without quant config).
    DEQUANT = "dequant"
    #: Half of the fused ``wkv_b`` (``attn_k_b`` / ``attn_v_b``).
    KV_B_K = "kv_b_k"
    KV_B_V = "kv_b_v"
    #: Stacked ``[E, rows, bytes]`` routed expert tensor.
    EXPERTS = "experts"
    #: ``rope_freqs`` -> ``lerope.alpha``.
    LEROPE = "lerope"


@dataclass(frozen=True, slots=True)
class Rule:
    #: Consolidated target name; ``{i}`` is the layer index.
    target: str
    kind: Kind = Kind.RENAME


_A = "layers.{i}.attention"
_IDX = "layers.{i}.attention.indexer"

#: Tensors outside ``blk.N.``.
GLOBAL_RULES: dict[str, Rule] = {
    "token_embd.weight": Rule("tok_embeddings.weight"),
    "output_norm.weight": Rule("norm.weight"),
    "output.weight": Rule("output.weight"),
    "rope_freqs.weight": Rule("lerope.alpha", Kind.LEROPE),
}

#: Suffixes after ``blk.{i}.``. The ``wkv_b.{k_b,v_b}`` targets are internal
#: names for the two halves of the fused matrix: they never reach the model.
LAYER_RULES: dict[str, Rule] = {
    "attn_norm.weight": Rule("layers.{i}.attention_norm.weight"),
    "ffn_norm.weight": Rule("layers.{i}.ffn_norm.weight"),
    "attn_q.weight": Rule(f"{_A}.wq.weight"),
    "attn_kv_a_mqa.weight": Rule(f"{_A}.wkv_a_with_mqa.weight", Kind.DEQUANT),
    "attn_kv_a_norm.weight": Rule(f"{_A}.kv_a_norm.weight"),
    "attn_k_b.weight": Rule(f"{_A}.wkv_b.k_b.weight", Kind.KV_B_K),
    "attn_v_b.weight": Rule(f"{_A}.wkv_b.v_b.weight", Kind.KV_B_V),
    "attn_output.weight": Rule(f"{_A}.wo.weight"),
    # SSMaxScaling parameters are called ``s`` and ``b`` (nvidia/ssmax.py).
    "attn_ssmax.weight": Rule(f"{_A}.ssmax_scaling.s"),
    "attn_ssmax.bias": Rule(f"{_A}.ssmax_scaling.b"),
    # Indexer. ``wk`` + ``weights_proj`` are the two shards of
    # ``wk_weights_proj`` (``_STACKED_PARAMS``), ``wk`` is dequantized.
    "indexer.attn_k.weight": Rule(f"{_IDX}.wk.weight", Kind.DEQUANT),
    "indexer.proj.weight": Rule(f"{_IDX}.weights_proj.weight"),
    "indexer.attn_q_b.weight": Rule(f"{_IDX}.wq_b.weight"),
    "indexer.q_norm.weight": Rule(f"{_IDX}.q_input_norm.weight"),
    "indexer.q_norm.bias": Rule(f"{_IDX}.q_input_norm.bias"),
    "indexer.k_norm.weight": Rule(f"{_IDX}.k_norm.weight"),
    "indexer.k_norm.bias": Rule(f"{_IDX}.k_norm.bias"),
    "indexer.ssmax.weight": Rule(f"{_IDX}.ssmax_scaling.s"),
    "indexer.ssmax.bias": Rule(f"{_IDX}.ssmax_scaling.b"),
    # Dense FFN (layer 0 only)
    "ffn_gate.weight": Rule("layers.{i}.feed_forward.w1.weight"),
    "ffn_up.weight": Rule("layers.{i}.feed_forward.w3.weight"),
    "ffn_down.weight": Rule("layers.{i}.feed_forward.w2.weight"),
    # MoE
    "ffn_gate_inp.weight": Rule("layers.{i}.gate.weight"),
    "ffn_gate_shexp.weight": Rule("layers.{i}.shared_experts.w1.weight"),
    "ffn_up_shexp.weight": Rule("layers.{i}.shared_experts.w3.weight"),
    "ffn_down_shexp.weight": Rule("layers.{i}.shared_experts.w2.weight"),
    # The whole stacked tensor is delivered under the name of expert 0.
    "ffn_gate_exps.weight": Rule("layers.{i}.experts.0.w1.weight", Kind.EXPERTS),
    "ffn_up_exps.weight": Rule("layers.{i}.experts.0.w3.weight", Kind.EXPERTS),
    "ffn_down_exps.weight": Rule("layers.{i}.experts.0.w2.weight", Kind.EXPERTS),
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass(frozen=True, slots=True)
class MappedTensor:
    gguf_name: str
    target: str
    kind: Kind
    layer: int | None
    suffix: str


def map_tensor_name(name: str) -> MappedTensor | None:
    """Map one GGUF tensor name, or return ``None`` if it is unknown."""
    if (rule := GLOBAL_RULES.get(name)) is not None:
        return MappedTensor(name, rule.target, rule.kind, None, name)
    match = _BLK_RE.match(name)
    if match is None:
        return None
    layer, suffix = int(match.group(1)), match.group(2)
    if (rule := LAYER_RULES.get(suffix)) is None:
        return None
    return MappedTensor(name, rule.target.format(i=layer), rule.kind, layer, suffix)


def build_name_map(names: Iterable[str]) -> dict[str, str]:
    """GGUF name -> target name for every tensor, raising on unknown ones.

    Vision (``v.*``) and projector (``mm.*``) tensors are not part of the text
    model and are skipped (they are absent from the name map). Everything else
    must be known: silently dropping a weight would produce wrong text.
    """
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for name in names:
        mapped = map_tensor_name(name)
        if mapped is not None:
            mapping[name] = mapped.target
        elif not name.startswith(("v.", "mm.")):
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"{len(unknown)} GGUF tensors are not known to the {GGUF_ARCHITECTURE} "
            f"adapter, first: {sorted(unknown)[:8]}"
        )
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        dup = sorted({t for t in targets if targets.count(t) > 1})
        raise ValueError(f"Several GGUF tensors map to the same target: {dup}")
    return mapping


# ---------------------------------------------------------------------------
# Per-layer structure (port of models/mistral/common/layer_spec.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayerSpec:
    layer_idx: int
    use_rope: bool
    shares_kv: bool
    has_indexer: bool
    is_moe: bool


def get_layer_specs(config: Any) -> list[LayerSpec]:
    """Per-layer variants, resolved like the model's ``get_layer_specs``."""
    n = config.num_hidden_layers
    nope = getattr(config, "nope_layer_flags", None) or [False] * n
    kv_sources = getattr(config, "kv_source_layers", None) or list(range(n))
    index_sharing = getattr(config, "index_sharing", None)
    use_sparse = getattr(config, "index_topk", None) is not None
    n_routed = getattr(config, "n_routed_experts", None)
    first_dense = getattr(config, "first_k_dense_replace", 0)
    freq = getattr(config, "moe_layer_freq", 1)
    return [
        LayerSpec(
            layer_idx=i,
            use_rope=not nope[i],
            shares_kv=kv_sources[i] != i,
            has_indexer=use_sparse
            and (index_sharing is None or index_sharing[i] == "full"),
            is_moe=bool(n_routed) and i >= first_dense and i % freq == 0,
        )
        for i in range(n)
    ]


def expected_layer_suffixes(spec: LayerSpec, config: Any) -> set[str]:
    """GGUF suffixes that the model needs in layer ``spec.layer_idx``."""
    s = {
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q.weight",
        "attn_k_b.weight",
        "attn_v_b.weight",
        "attn_output.weight",
    }
    if getattr(config, "ssmax_scaling", False):
        s |= {"attn_ssmax.weight", "attn_ssmax.bias"}
    if not spec.shares_kv:
        s |= {"attn_kv_a_mqa.weight", "attn_kv_a_norm.weight"}
    if spec.has_indexer:
        s |= {
            "indexer.attn_k.weight",
            "indexer.proj.weight",
            "indexer.attn_q_b.weight",
            "indexer.q_norm.weight",
            "indexer.q_norm.bias",
            "indexer.k_norm.weight",
            "indexer.k_norm.bias",
        }
        if getattr(config, "indexer_ssmax_scaling", False):
            s |= {"indexer.ssmax.weight", "indexer.ssmax.bias"}
    if spec.is_moe:
        s |= {
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        }
    else:
        s |= {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}
    return s


# ---------------------------------------------------------------------------
# params.json handling
# ---------------------------------------------------------------------------

#: ``params.json`` blocks that must not reach the model config for a GGUF run:
#: the FP8 ``quantization`` block (it would become an FP8 ``quantization_config``
#: next to the GGUF one) and the vision/audio blocks (the GGUF has no mmproj
#: here, and a ``vision_encoder`` turns the architecture into Pixtral).
PARAMS_KEYS_TO_DROP = ("quantization", "vision_encoder", "multimodal")


def is_mistral_large4_params(params: dict) -> bool:
    """Whether a ``params.json`` dict describes a Mistral DSA/MLA/MoE model."""
    return bool(params.get("dsa")) and bool(params.get("moe"))


def strip_params_for_gguf(params: dict) -> dict:
    """Copy of ``params`` without :data:`PARAMS_KEYS_TO_DROP` (in memory only)."""
    return {k: v for k, v in params.items() if k not in PARAMS_KEYS_TO_DROP}


# ---------------------------------------------------------------------------
# Inventory (names, types, logical shapes) and its audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    qtype: int
    #: Logical shape in numpy order, i.e. the reverse of the GGUF ``ne``.
    shape: tuple[int, ...]


def read_gguf_inventory(paths: Iterable[str]) -> list[TensorInfo]:
    """Read tensor names, types and logical shapes (no tensor data)."""
    out: list[TensorInfo] = []
    for path in paths:
        reader = GGUFReader(path)
        out.extend(
            TensorInfo(
                t.name,
                int(t.tensor_type),
                tuple(int(d) for d in reversed(t.shape.tolist())),
            )
            for t in reader.tensors
        )
    return out


def read_gguf_architecture(path: str) -> str | None:
    """Return ``general.architecture`` of a GGUF file."""
    reader = GGUFReader(path)
    field = reader.fields.get("general.architecture")
    if field is None:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def expert_quant_types(inventory: Iterable[TensorInfo]) -> dict[int, tuple[int, ...]]:
    """Layer -> (gate, up, down) quantization types of the stacked experts."""
    by_layer: dict[int, dict[str, int]] = defaultdict(dict)
    for info in inventory:
        mapped = map_tensor_name(info.name)
        if mapped is not None and mapped.kind is Kind.EXPERTS:
            by_layer[mapped.layer][mapped.suffix] = info.qtype  # type: ignore[index]
    return {
        layer: (
            t["ffn_gate_exps.weight"],
            t["ffn_up_exps.weight"],
            t["ffn_down_exps.weight"],
        )
        for layer, t in sorted(by_layer.items())
        if len(t) == 3
    }


def check_gate_up_types(layer: int, gate: int, up: int) -> None:
    """The MoE method keeps one type per ``w13``: gate and up must agree."""
    if gate != up:
        raise ValueError(
            f"Layer {layer}: ffn_gate_exps and ffn_up_exps use different GGUF "
            f"quantization types ({gate} vs {up}); they are fused into one w13 "
            "weight with a single type, which this adapter cannot represent."
        )


def _expected_shapes(
    mapped: MappedTensor, spec: LayerSpec | None, cfg: Any
) -> tuple[int, ...] | None:
    def g(name: str) -> int | None:
        return getattr(cfg, name, None)

    hidden, heads = g("hidden_size"), g("num_attention_heads")
    kv_lora, nope, rope = (
        g("kv_lora_rank"),
        g("qk_nope_head_dim"),
        g("qk_rope_head_dim"),
    )
    v_dim = g("v_head_dim")
    sfx = mapped.suffix
    if spec is None:
        if sfx in ("token_embd.weight", "output.weight"):
            vocab = g("vocab_size")
            return (vocab, hidden) if vocab and hidden else None
        if sfx == "rope_freqs.weight":
            return (rope // 2,) if rope else None
        return None
    if None in (hidden, heads, kv_lora, nope, rope, v_dim):
        return None
    if sfx == "attn_k_b.weight":
        return (heads, kv_lora, nope if spec.use_rope else nope + rope)
    if sfx == "attn_v_b.weight":
        return (heads, v_dim, kv_lora)
    if sfx == "attn_kv_a_mqa.weight":
        return (kv_lora + (rope if spec.use_rope else 0), hidden)
    if sfx == "attn_q.weight":
        return (heads * (nope + rope), hidden)
    if sfx == "attn_output.weight":
        return (hidden, heads * v_dim)
    if sfx == "indexer.attn_k.weight" and g("index_head_dim"):
        return (g("index_head_dim"), hidden)
    if sfx == "indexer.proj.weight" and g("index_n_heads"):
        return (g("index_n_heads"), hidden)
    n_exp, inter = g("n_routed_experts"), g("moe_intermediate_size")
    if sfx in ("ffn_gate_exps.weight", "ffn_up_exps.weight") and n_exp and inter:
        return (n_exp, inter, hidden)
    if sfx == "ffn_down_exps.weight" and n_exp and inter:
        return (n_exp, hidden, inter)
    return None


def audit_inventory(inventory: Sequence[TensorInfo], config: Any) -> dict[str, str]:
    """Check a GGUF tensor inventory against the model structure.

    Returns the GGUF -> target name map. Raises ``ValueError`` listing every
    problem found: unknown tensors, missing or unexpected tensors per layer,
    shape mismatches, and gate/up quantization type mismatches.
    """
    problems: list[str] = []
    if os.environ.get(ALLOW_TRUNCATED_ENV) == "1":
        depth = config.num_hidden_layers
        inventory = [
            info
            for info in inventory
            if (mapped := map_tensor_name(info.name)) is None
            or mapped.layer is None
            or mapped.layer < depth
        ]
    name_map = build_name_map(info.name for info in inventory)

    specs = get_layer_specs(config)
    present: dict[int, set[str]] = defaultdict(set)
    for info in inventory:
        mapped = map_tensor_name(info.name)
        if mapped is None:
            continue
        spec = None if mapped.layer is None else _spec_or_none(specs, mapped.layer)
        if mapped.layer is not None:
            if spec is None:
                problems.append(f"{info.name}: layer {mapped.layer} out of range")
                continue
            present[mapped.layer].add(mapped.suffix)
        want = _expected_shapes(mapped, spec, config)
        if want is not None and tuple(info.shape) != tuple(want):
            problems.append(f"{info.name}: shape {info.shape}, expected {want}")

    for spec in specs:
        expected = expected_layer_suffixes(spec, config)
        have = present.get(spec.layer_idx, set())
        if missing := sorted(expected - have):
            problems.append(f"layer {spec.layer_idx}: missing {missing}")
        if extra := sorted(have - expected):
            problems.append(f"layer {spec.layer_idx}: unexpected {extra}")

    for global_name in ("token_embd.weight", "output_norm.weight", "output.weight"):
        if global_name not in name_map:
            problems.append(f"missing {global_name}")
    if getattr(config, "lerope", False) and "rope_freqs.weight" not in name_map:
        problems.append("config has lerope but the GGUF has no rope_freqs.weight")

    for layer, (gate, up, _down) in expert_quant_types(inventory).items():
        try:
            check_gate_up_types(layer, gate, up)
        except ValueError as error:
            problems.append(str(error))

    if problems:
        raise ValueError(
            "GGUF inventory does not match the Mistral Large 4 model:\n  "
            + "\n  ".join(problems[:40])
            + (f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else "")
        )
    return name_map


def _spec_or_none(specs: Sequence[LayerSpec], layer: int) -> LayerSpec | None:
    return specs[layer] if 0 <= layer < len(specs) else None


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def dequantize_to_float32(data: np.ndarray, qtype: int | None) -> np.ndarray:
    """Dequantize raw GGUF bytes (``[..., bytes_per_row]`` uint8) to float32.

    Float arrays (F32/F16) are returned as float32. Raw BF16 bytes are widened.
    Quantized types use ``gguf.quants.dequantize`` on CPU, the independent
    reference implementation. ``qtype`` is required for uint8 data.
    """
    if data.dtype != np.uint8:
        return np.asarray(data, dtype=np.float32)
    if qtype is None:
        raise ValueError("uint8 tensor data without a GGUF quantization type")
    qtype = GGMLQuantizationType(int(qtype))
    if qtype == GGMLQuantizationType.BF16:
        wide = np.ascontiguousarray(data).view(np.uint16).astype(np.uint32) << 16
        return wide.view(np.float32)
    return dequantize(np.ascontiguousarray(data), qtype)


def fuse_kv_b(k_b: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Rebuild the model's ``wkv_b`` from llama.cpp's split ``k_b`` / ``v_b``.

    ``k_b``: ``(heads, kv_lora, nope)``, ``v_b``: ``(heads, v_dim, kv_lora)``.
    Result: ``(heads * (nope + v_dim), kv_lora)`` with, per head, the ``nope``
    rows of ``W_UK`` followed by the ``v_dim`` rows of ``W_UV``. This inverts
    ``conversion/mistral4.py``: ``kv_b.view(H, nope + v, kv_lora)`` split into
    ``k_part`` / ``v_part``, ``k_b = k_part.transpose(1, 2)``, ``v_b = v_part``.
    """
    if k_b.ndim != 3 or v_b.ndim != 3:
        raise ValueError(f"k_b/v_b must be 3-D, got {k_b.shape} and {v_b.shape}")
    heads, kv_lora, nope = k_b.shape
    if v_b.shape[0] != heads or v_b.shape[2] != kv_lora:
        raise ValueError(f"Incompatible k_b {k_b.shape} and v_b {v_b.shape}")
    fused = np.concatenate([k_b.transpose(0, 2, 1), v_b], axis=1)
    return fused.reshape(heads * (nope + v_b.shape[1]), kv_lora)


def lerope_alpha_from_rope_freqs(rope_freqs: np.ndarray) -> np.ndarray:
    """``lerope.alpha`` from the GGUF ``rope_freqs``.

    The converter writes ``rope_freqs = exp(-alpha)`` (ggml divides theta by
    the freq factors, LeRoPE multiplies inverse frequencies by ``exp(alpha)``).
    """
    return (-np.log(np.asarray(rope_freqs, dtype=np.float64))).astype(np.float32)


# ---------------------------------------------------------------------------
# Stacked experts
# ---------------------------------------------------------------------------

_EXPERT0_RE = re.compile(r"\.experts\.0\.(w[123])\.weight$")


def expert_names(target: str, num_experts: int) -> list[str]:
    """Per-expert names for a stacked target (``...experts.0.w1.weight``)."""
    if _EXPERT0_RE.search(target) is None:
        raise ValueError(f"Not a stacked expert target: {target}")
    return [target.replace(".experts.0.", f".experts.{e}.") for e in range(num_experts)]


def split_stacked(array: Any) -> list[Any]:
    """Per-expert views of a stacked ``[E, rows, ...]`` tensor, without copies.

    The expert axis is the slowest axis, so ``array[e]`` is the contiguous byte
    range ``[e * n, (e + 1) * n)`` of the packed data. Works for numpy arrays
    and torch tensors (both return views).
    """
    if len(array.shape) != 3:
        raise ValueError(f"Expected a 3-D stacked expert tensor, got {array.shape}")
    return [array[e] for e in range(array.shape[0])]


# ---------------------------------------------------------------------------
# Streaming transformation
# ---------------------------------------------------------------------------

_TARGET_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.attention\."
    r"(?P<what>wkv_a_with_mqa|indexer\.wk|wkv_b\.k_b|wkv_b\.v_b)\.weight(?P<type>_type)?$"
)
_EXPERT_TYPE_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.experts\.0\.(?P<proj>w[123])\.weight_type$"
)


@dataclass(frozen=True, slots=True)
class Emitted:
    name: str
    array: np.ndarray
    #: ``True`` for tensors that must stay float32 (``lerope.alpha``).
    keep_float32: bool = False


class StreamTransformer:
    """Stateful transformation of the plugin's ``(name, tensor)`` stream.

    The plugin streams, for each quantized tensor, ``("<name>.weight_type",
    qtype)`` followed by ``("<name>.weight", bytes)``; float tensors come with
    no type entry. This class consumes the entries of the dequantized targets
    (see :data:`LAYER_RULES`) and emits dense float32 arrays for them. All
    other entries are passed through by the caller.
    """

    def __init__(self) -> None:
        self._types: dict[str, int] = {}
        self._kv_b: dict[int, dict[str, np.ndarray]] = {}
        self._expert_types: dict[int, dict[str, int]] = defaultdict(dict)

    # -- type entries -------------------------------------------------------
    def consume_type(self, name: str, qtype: int) -> bool:
        """Record a ``*.weight_type`` entry; ``True`` if it must not be forwarded."""
        if (m := _EXPERT_TYPE_RE.match(name)) is not None:
            layer = int(m["layer"])
            seen = self._expert_types[layer]
            seen[m["proj"]] = int(qtype)
            if "w1" in seen and "w3" in seen:
                check_gate_up_types(layer, seen["w1"], seen["w3"])
            return False
        if _TARGET_RE.match(name) is not None:
            self._types[name.removesuffix("_type")] = int(qtype)
            return True
        return False

    # -- data entries -------------------------------------------------------
    def handles(self, name: str) -> bool:
        m = _TARGET_RE.match(name)
        return (m is not None and m["type"] is None) or name == "lerope.alpha"

    def feed(self, name: str, data: np.ndarray) -> list[Emitted]:
        """Transform one handled data entry; may emit zero or one tensors."""
        if name == "lerope.alpha":
            return [Emitted(name, lerope_alpha_from_rope_freqs(data), True)]
        m = _TARGET_RE.match(name)
        assert m is not None and m["type"] is None, name
        dense = dequantize_to_float32(data, self._types.pop(name, None))
        what = m["what"]
        if what not in ("wkv_b.k_b", "wkv_b.v_b"):
            return [Emitted(name, dense)]
        layer = int(m["layer"])
        parts = self._kv_b.setdefault(layer, {})
        parts[what.split(".")[1]] = dense
        if "k_b" not in parts or "v_b" not in parts:
            return []
        del self._kv_b[layer]
        fused = fuse_kv_b(parts["k_b"], parts["v_b"])
        return [Emitted(f"layers.{layer}.attention.wkv_b.weight", fused)]

    def finish(self) -> None:
        """Raise if the stream ended with half-processed state."""
        if self._kv_b:
            raise ValueError(
                f"Incomplete attn_k_b/attn_v_b pair for layers {sorted(self._kv_b)}"
            )
        if self._types:
            raise ValueError(f"weight_type without data for {sorted(self._types)}")

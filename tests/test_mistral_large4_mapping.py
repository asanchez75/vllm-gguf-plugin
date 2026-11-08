# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the vLLM-free part of the Mistral Large 4 GGUF adapter.

They only need numpy, torch and gguf: the mapping module is loaded straight
from its file when ``vllm`` (and with it the plugin package) is not importable.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize, quantize

try:
    from vllm_gguf_plugin.weights_adapter import mistral_large4_mapping as m
except ImportError:  # no vLLM on this machine: load the pure module by path
    _path = (
        Path(__file__).resolve().parents[1]
        / "vllm_gguf_plugin"
        / "weights_adapter"
        / "mistral_large4_mapping.py"
    )
    _spec = importlib.util.spec_from_file_location("mistral_large4_mapping", _path)
    m = importlib.util.module_from_spec(_spec)
    sys.modules["mistral_large4_mapping"] = m
    _spec.loader.exec_module(m)

# ---------------------------------------------------------------------------
# Real Mistral Large 4 structure (spike section 14.2 / ladder-0-2 section 5)
# ---------------------------------------------------------------------------

N_LAYERS = 72
NOPE_LAYERS = set(range(3, 72, 4))  # 3, 7, ..., 71
KV_SHARED_LAYERS = set(range(39, 72, 4))  # 39, 43, ..., 71 (source 35)
INDEXER_LAYERS = set(range(0, 17)) | {
    i for i in range(17, 72) if (i - 17) % 4 in (2, 3)
}


def real_config():
    kv_sources = [35 if i in KV_SHARED_LAYERS else i for i in range(N_LAYERS)]
    return SimpleNamespace(
        num_hidden_layers=N_LAYERS,
        hidden_size=6144,
        num_attention_heads=64,
        kv_lora_rank=512,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=6144,
        intermediate_size=24576,
        n_routed_experts=128,
        first_k_dense_replace=1,
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        ssmax_scaling=True,
        indexer_ssmax_scaling=True,
        lerope=True,
        vocab_size=204800,
        nope_layer_flags=[i in NOPE_LAYERS for i in range(N_LAYERS)],
        kv_source_layers=kv_sources,
        index_sharing=["full" if i in INDEXER_LAYERS else "shared" for i in range(72)],
    )


def expert_types(layer: int) -> tuple[T, T, T]:
    if layer in range(1, 7) or layer == 71:
        return T.IQ2_XXS, T.IQ2_XXS, T.IQ2_XXS
    if layer in range(7, 29):
        return T.IQ1_S, T.IQ1_S, T.IQ2_XXS
    return T.IQ1_S, T.IQ1_S, T.IQ1_S


def real_inventory(cfg=None) -> list:
    """Tensor inventory of the real GGUF as listed in the spike (1602 tensors)."""
    cfg = cfg or real_config()
    h, nh, kv = cfg.hidden_size, cfg.num_attention_heads, cfg.kv_lora_rank
    inv = []

    def add(name, qtype, *shape):
        inv.append(m.TensorInfo(name, int(qtype), tuple(shape)))

    add("token_embd.weight", T.Q6_K, cfg.vocab_size, h)
    add("output_norm.weight", T.F32, h)
    add("output.weight", T.Q6_K, cfg.vocab_size, h)
    add("rope_freqs.weight", T.F32, cfg.qk_rope_head_dim // 2)
    for i in range(N_LAYERS):
        b = f"blk.{i}."
        rope = i not in NOPE_LAYERS
        nope = cfg.qk_nope_head_dim + (0 if rope else cfg.qk_rope_head_dim)
        add(b + "attn_norm.weight", T.F32, h)
        add(b + "ffn_norm.weight", T.F32, h)
        add(b + "attn_q.weight", T.Q6_K, nh * (cfg.qk_nope_head_dim + 64), h)
        add(b + "attn_output.weight", T.Q6_K, h, nh * cfg.v_head_dim)
        add(b + "attn_k_b.weight", T.Q8_0, nh, kv, nope)
        add(b + "attn_v_b.weight", T.Q8_0, nh, cfg.v_head_dim, kv)
        add(b + "attn_ssmax.weight", T.F32, nh)
        add(b + "attn_ssmax.bias", T.F32, nh)
        if i not in KV_SHARED_LAYERS:
            add(b + "attn_kv_a_mqa.weight", T.Q8_0, kv + (64 if rope else 0), h)
            add(b + "attn_kv_a_norm.weight", T.F32, kv)
        if i in INDEXER_LAYERS:
            add(b + "indexer.attn_k.weight", T.Q8_0, 128, h)
            add(b + "indexer.proj.weight", T.F32, 32, h)
            add(b + "indexer.attn_q_b.weight", T.Q6_K, 4096, h)
            for n, d in (("q_norm", h), ("k_norm", 128), ("ssmax", 32)):
                add(b + f"indexer.{n}.weight", T.F32, d)
                add(b + f"indexer.{n}.bias", T.F32, d)
        if i == 0:
            add(b + "ffn_gate.weight", T.Q8_0, 24576, h)
            add(b + "ffn_up.weight", T.Q8_0, 24576, h)
            add(b + "ffn_down.weight", T.Q8_0, h, 24576)
        else:
            g, u, d = expert_types(i)
            add(b + "ffn_gate_inp.weight", T.F32, 128, h)
            for p in ("gate", "up"):
                add(b + f"ffn_{p}_shexp.weight", T.Q8_0, 6144, h)
            add(b + "ffn_down_shexp.weight", T.Q8_0, h, 6144)
            add(b + "ffn_gate_exps.weight", g, 128, 6144, h)
            add(b + "ffn_up_exps.weight", u, 128, 6144, h)
            add(b + "ffn_down_exps.weight", d, 128, h, 6144)
    return inv


# ---------------------------------------------------------------------------
# What the model's load_weights does with a consolidated name.
# Copied as data/logic from vllm models/mistral/mistral_large4/nvidia/model.py
# (PR 64534 @ 28a4495d): _CHECKPOINT_NAME_MAP (:296-316), _STACKED_PARAMS
# (:319-325) and the name handling in _load_weights (:362-427).
# ---------------------------------------------------------------------------

PR_CHECKPOINT_NAME_MAP = [
    (r"^tok_embeddings(?=\.|$)", "model.embed_tokens"),
    (r"^norm(?=\.|$)", "model.norm"),
    (r"^output(?=\.|$)", "lm_head"),
    (r"^layers\.", "model.layers."),
    (r"\.attention_norm(?=\.|$)", ".input_layernorm"),
    (r"\.ffn_norm(?=\.|$)", ".post_attention_layernorm"),
    (r"\.attention\.indexer\.ssmax_scaling(?=\.|$)", ".self_attn.indexer.ssmax"),
    (r"\.attention\.ssmax_scaling(?=\.|$)", ".self_attn.ssmax"),
    (r"\.attention\.wq(?=\.|$)", ".self_attn.q_proj"),
    (r"\.attention\.wkv_a_with_mqa(?=\.|$)", ".self_attn.kv_a_proj_with_mqa"),
    (r"\.attention\.kv_a_norm(?=\.|$)", ".self_attn.kv_a_layernorm"),
    (r"\.attention\.wkv_b(?=\.|$)", ".self_attn.kv_b_proj"),
    (r"\.attention\.wo(?=\.|$)", ".self_attn.o_proj"),
    (r"\.attention\.indexer(?=\.|$)", ".self_attn.indexer"),
    (r"\.feed_forward(?=\.|$)", ".mlp"),
    (r"\.shared_experts(?=\.|$)", ".mlp.shared_experts"),
    (r"\.gate(?=\.|$)", ".mlp.gate"),
    (r"\.experts(?=\.|$)", ".mlp.experts"),
    (r"\.qscale_weight$", ".weight_scale_inv"),
]
PR_STACKED_PARAMS = [
    (".gate_up_proj.", ".w1.", 0),
    (".gate_up_proj.", ".w3.", 1),
    (".down_proj.", ".w2.", -1),
    (".wk_weights_proj.", ".wk.", 0),
    (".wk_weights_proj.", ".weights_proj.", 1),
]


def pr_param_for(ckpt_name: str) -> str:
    """The model parameter a consolidated checkpoint name ends up in."""
    name = ckpt_name
    for pattern, repl in PR_CHECKPOINT_NAME_MAP:
        name = re.sub(pattern, repl, name)
    if ".mlp.experts." in name:
        # fused_moe_make_expert_params_mapping(ckpt w1/w2/w3, prefix routed_experts)
        mt = re.search(r"experts\.(\d+)\.(w[123])\.", name)
        assert mt is not None, name
        stem = "w2_" if mt[2] == "w2" else "w13_"
        return name.replace(mt[0], f"experts.routed_experts.{stem}")
    for param_part, ckpt_part, _ in PR_STACKED_PARAMS:
        if ckpt_part in name:
            return name.replace(ckpt_part, param_part)
    return name


# Parameters of the PR model per layer type (ladder-0-2 section 5.2, 1342 total).
def pr_params(cfg, quantized_types: bool) -> set[str]:
    """Model parameters (+ GGUF ``weight_type`` twins if ``quantized_types``)."""
    out: set[str] = set()
    specs = m.get_layer_specs(cfg)

    def add(name, *, typed=False):
        out.add(name)
        if typed and quantized_types:
            out.add(name.removesuffix(".weight") + ".weight_type")

    add("model.embed_tokens.weight", typed=True)
    add("lm_head.weight", typed=True)
    add("model.norm.weight")
    for s in specs:
        p = f"model.layers.{s.layer_idx}."
        add(p + "input_layernorm.weight")
        add(p + "post_attention_layernorm.weight")
        add(p + "self_attn.q_proj.weight", typed=True)
        add(p + "self_attn.o_proj.weight", typed=True)
        add(p + "self_attn.ssmax.s")
        add(p + "self_attn.ssmax.b")
        add(p + "self_attn.kv_b_proj.weight")
        if not s.shares_kv:
            add(p + "self_attn.kv_a_proj_with_mqa.weight")
            add(p + "self_attn.kv_a_layernorm.weight")
        if s.has_indexer:
            q = p + "self_attn.indexer."
            for n in ("q_input_norm", "k_norm"):
                add(q + n + ".weight")
                add(q + n + ".bias")
            add(q + "wq_b.weight", typed=True)
            add(q + "wk_weights_proj.weight")
            add(q + "ssmax.s")
            add(q + "ssmax.b")
        if s.is_moe:
            add(p + "mlp.gate.weight")
            add(p + "mlp.shared_experts.gate_up_proj.weight", typed=True)
            add(p + "mlp.shared_experts.down_proj.weight", typed=True)
            for stem in ("w13_", "w2_"):
                add(p + f"mlp.experts.routed_experts.{stem}weight", typed=True)
                # the MoE twin is named w13_weight_type, not w13_weight.weight_type
        else:
            add(p + "mlp.gate_up_proj.weight", typed=True)
            add(p + "mlp.down_proj.weight", typed=True)
    # expert twins: "...w13_weight" -> "...w13_weight_type"
    return {re.sub(r"(w13_|w2_)weight\.weight_type$", r"\1weight_type", n) for n in out}


def simulate_stream(inventory) -> list[str]:
    """Names the plugin streams to the transformer (type entries + data)."""
    name_map = m.build_name_map(i.name for i in inventory)
    names = []
    for info in inventory:
        target = name_map[info.name]
        if info.qtype not in m.FLOAT_QTYPES:
            names.append(target.replace("weight", "weight_type"))
        names.append(target)
    return names


# ---------------------------------------------------------------------------
# (a) completeness
# ---------------------------------------------------------------------------


def test_real_structure_counts():
    cfg = real_config()
    specs = m.get_layer_specs(cfg)
    assert len(real_inventory(cfg)) == 1602
    assert sum(not s.use_rope for s in specs) == 18
    assert sum(s.shares_kv for s in specs) == 9
    assert sum(s.has_indexer for s in specs) == 44
    assert [s.is_moe for s in specs] == [False] + [True] * 71


def test_every_tensor_maps_to_exactly_one_target():
    inv = real_inventory()
    name_map = m.build_name_map(i.name for i in inv)
    assert len(name_map) == len(inv) == 1602
    assert len(set(name_map.values())) == 1602


def test_audit_accepts_real_inventory():
    assert len(m.audit_inventory(real_inventory(), real_config())) == 1602


def truncated_config(depth: int):
    cfg = real_config()
    cfg.num_hidden_layers = depth
    cfg.nope_layer_flags = cfg.nope_layer_flags[:depth]
    cfg.kv_source_layers = cfg.kv_source_layers[:depth]
    cfg.index_sharing = cfg.index_sharing[:depth]
    return cfg


def test_truncated_model_is_rejected_unless_explicitly_allowed(monkeypatch):
    cfg = truncated_config(4)
    inv = real_inventory()
    monkeypatch.delenv(m.ALLOW_TRUNCATED_ENV, raising=False)
    with pytest.raises(ValueError, match="layer 4 out of range"):
        m.audit_inventory(inv, cfg)

    monkeypatch.setenv(m.ALLOW_TRUNCATED_ENV, "1")
    name_map = m.audit_inventory(inv, cfg)
    layers = {m.map_tensor_name(n).layer for n in name_map}
    assert layers == {None, 0, 1, 2, 3}
    assert "rope_freqs.weight" in name_map and "output.weight" in name_map
    # the deeper layers are not in the map, so the weight iterator never reads them
    assert not any(n.startswith("blk.4.") for n in name_map)
    # a GGUF with fewer layers than the config is still an error
    short = [i for i in inv if not i.name.startswith("blk.3.")]
    with pytest.raises(ValueError, match="layer 3: missing"):
        m.audit_inventory(short, cfg)


def test_targets_reach_exactly_the_model_parameters():
    """Model-side names after the PR renames == the 1342 model parameters.

    Parts of fused tensors (k_b/v_b) disappear into ``wkv_b``; ``lerope.alpha``
    is not a parameter (loaded separately).
    """
    cfg = real_config()
    inv = real_inventory(cfg)
    targets = [t for t in m.build_name_map(i.name for i in inv).values()]
    final = []
    for t in targets:
        if t == "lerope.alpha":
            continue
        mt = re.match(r"layers\.(\d+)\.attention\.wkv_b\.(k_b|v_b)\.weight$", t)
        if mt:
            final.append(f"layers.{mt[1]}.attention.wkv_b.weight")
        else:
            final.append(t)
    # k_b and v_b collapse to one wkv_b per layer
    assert len(final) - len(set(final)) == N_LAYERS
    reached = {pr_param_for(n) for n in set(final)}
    # every expert tensor of a layer lands in w13 (gate+up) or w2
    assert reached == pr_params(cfg, quantized_types=False)
    assert len(reached) == 1342


def test_stream_covers_all_parameters_including_weight_types():
    """Adding the ``weight_type`` entries, 1342 + 476 parameters are reached.

    476 = 474 (ladder-0-2 5.5) + the embedding and the LM head, whose GGUF
    methods are installed by ``recursive_replace_vocab_modules``.
    """
    cfg = real_config()
    inv = real_inventory(cfg)
    stream = m.StreamTransformer()
    forwarded = []
    for name in simulate_stream(inv):
        if name.endswith(".weight_type"):
            if not stream.consume_type(name, 0):
                forwarded.append(name)
        elif not stream.handles(name):
            forwarded.append(name)
    reached = {pr_param_for(n) for n in forwarded}
    # transformed outputs are emitted under these names
    for layer in range(N_LAYERS):
        reached.add(pr_param_for(f"layers.{layer}.attention.wkv_b.weight"))
        if layer not in KV_SHARED_LAYERS:
            n = f"layers.{layer}.attention.wkv_a_with_mqa.weight"
            reached.add(pr_param_for(n))
        if layer in INDEXER_LAYERS:
            reached.add(pr_param_for(f"layers.{layer}.attention.indexer.wk.weight"))
    expected = pr_params(cfg, quantized_types=True)
    assert reached == expected
    assert len(expected) == 1342 + 476


def test_stream_handles_exactly_the_dense_kinds():
    stream = m.StreamTransformer()
    rules = {**m.GLOBAL_RULES, **m.LAYER_RULES}
    for suffix, rule in rules.items():
        target = rule.target.format(i=3)
        dense = rule.kind in (m.Kind.DEQUANT, m.Kind.KV_B_K, m.Kind.KV_B_V)
        dense = dense or rule.kind is m.Kind.LEROPE
        assert stream.handles(target) == dense, suffix


def test_dequantized_targets_are_never_forwarded_with_a_type():
    stream = m.StreamTransformer()
    consumed = [
        "layers.5.attention.wkv_a_with_mqa.weight_type",
        "layers.5.attention.wkv_b.k_b.weight_type",
        "layers.5.attention.wkv_b.v_b.weight_type",
        "layers.5.attention.indexer.wk.weight_type",
    ]
    assert all(stream.consume_type(n, int(T.Q8_0)) for n in consumed)
    for n in (
        "layers.5.attention.wq.weight_type",
        "layers.5.attention.indexer.wq_b.weight_type",
        "layers.5.experts.0.w2.weight_type",
        "layers.0.feed_forward.w1.weight_type",
    ):
        assert not stream.consume_type(n, int(T.Q6_K))


def test_layer_variants_follow_the_spike():
    cfg = real_config()
    inv = {i.name: i for i in real_inventory(cfg)}
    # shared-KV layers: no kv_a, no kv_a_norm; NoPE kv_a has 512 rows
    assert "blk.39.attn_kv_a_mqa.weight" not in inv
    assert "blk.39.attn_kv_a_norm.weight" not in inv
    assert inv["blk.3.attn_kv_a_mqa.weight"].shape == (512, 6144)
    assert inv["blk.4.attn_kv_a_mqa.weight"].shape == (576, 6144)
    assert inv["blk.3.attn_k_b.weight"].shape == (64, 512, 128)
    assert inv["blk.4.attn_k_b.weight"].shape == (64, 512, 64)
    # indexer layers 17/18 shared, 19/20 present, 71 present
    assert "blk.17.indexer.attn_k.weight" not in inv
    assert "blk.19.indexer.attn_k.weight" in inv
    assert "blk.71.indexer.attn_k.weight" in inv


def test_expert_type_groups_by_layer():
    groups = {}
    for layer, types in m.expert_quant_types(real_inventory()).items():
        groups.setdefault(types, []).append(layer)
    iq1, iq2 = int(T.IQ1_S), int(T.IQ2_XXS)
    assert groups[(iq2, iq2, iq2)] == [1, 2, 3, 4, 5, 6, 71]
    assert groups[(iq1, iq1, iq2)] == list(range(7, 29))
    assert groups[(iq1, iq1, iq1)] == list(range(29, 71))


def test_audit_reports_missing_unexpected_and_shape_problems():
    cfg = real_config()
    inv = real_inventory(cfg)
    dropped = [i for i in inv if i.name != "blk.39.attn_k_b.weight"]
    with pytest.raises(ValueError, match="layer 39: missing"):
        m.audit_inventory(dropped, cfg)

    extra = [
        *inv,
        m.TensorInfo("blk.39.attn_kv_a_mqa.weight", int(T.Q8_0), (512, 6144)),
    ]
    with pytest.raises(ValueError, match="layer 39: unexpected"):
        m.audit_inventory(extra, cfg)

    bad = [
        m.TensorInfo(i.name, i.qtype, (64, 512, 64))
        if i.name == "blk.3.attn_k_b.weight"
        else i
        for i in inv
    ]
    with pytest.raises(ValueError, match=r"blk\.3\.attn_k_b\.weight: shape"):
        m.audit_inventory(bad, cfg)

    small_vocab = [
        m.TensorInfo(i.name, i.qtype, (1000, 6144))
        if i.name == "token_embd.weight"
        else i
        for i in inv
    ]
    with pytest.raises(ValueError, match="token_embd.weight: shape"):
        m.audit_inventory(small_vocab, cfg)


def test_unknown_tensor_is_an_error_but_vision_is_skipped():
    with pytest.raises(ValueError, match="not known"):
        m.build_name_map(["token_embd.weight", "blk.0.attn_unknown.weight"])
    mapping = m.build_name_map(
        ["token_embd.weight", "v.blk.0.attn_q.weight", "mm.0.weight"]
    )
    assert mapping == {"token_embd.weight": "tok_embeddings.weight"}


def test_global_and_layer_names():
    mp = m.build_name_map
    assert mp(["output_norm.weight"]) == {"output_norm.weight": "norm.weight"}
    names = {
        "blk.7.attn_ssmax.weight": "layers.7.attention.ssmax_scaling.s",
        "blk.7.attn_ssmax.bias": "layers.7.attention.ssmax_scaling.b",
        "blk.7.indexer.ssmax.weight": "layers.7.attention.indexer.ssmax_scaling.s",
        "blk.7.indexer.q_norm.bias": "layers.7.attention.indexer.q_input_norm.bias",
        "blk.7.indexer.k_norm.weight": "layers.7.attention.indexer.k_norm.weight",
        "blk.7.indexer.attn_q_b.weight": "layers.7.attention.indexer.wq_b.weight",
        "blk.7.indexer.attn_k.weight": "layers.7.attention.indexer.wk.weight",
        "blk.7.indexer.proj.weight": "layers.7.attention.indexer.weights_proj.weight",
        "blk.0.ffn_gate.weight": "layers.0.feed_forward.w1.weight",
        "blk.0.ffn_up.weight": "layers.0.feed_forward.w3.weight",
        "blk.0.ffn_down.weight": "layers.0.feed_forward.w2.weight",
        "blk.8.ffn_gate_inp.weight": "layers.8.gate.weight",
        "blk.8.ffn_up_shexp.weight": "layers.8.shared_experts.w3.weight",
        "blk.8.ffn_gate_exps.weight": "layers.8.experts.0.w1.weight",
        "blk.8.ffn_down_exps.weight": "layers.8.experts.0.w2.weight",
        "rope_freqs.weight": "lerope.alpha",
    }
    assert mp(list(names)) == names


# ---------------------------------------------------------------------------
# (b) kv_b fusion
# ---------------------------------------------------------------------------


def converter_split(kv_b: np.ndarray, n_head: int, nope: int, v_dim: int):
    """llama.cpp conversion/mistral4.py:111-126, in numpy (the tensors to invert)."""
    kv = kv_b.reshape(n_head, nope + v_dim, kv_b.shape[-1])
    k_part, v_part = kv[:, :nope], kv[:, nope:]
    return k_part.transpose(0, 2, 1), v_part  # k_b (H, kv_lora, nope), v_b


@pytest.mark.parametrize("nope", [64, 128])  # RoPE layers / NoPE layers
def test_kv_b_fusion_inverts_the_converter(nope):
    rng = np.random.default_rng(nope)
    n_head, v_dim, kv_lora = 4, 16, 32
    kv_b = rng.standard_normal((n_head * (nope + v_dim), kv_lora)).astype(np.float32)
    k_b, v_b = converter_split(kv_b, n_head, nope, v_dim)
    fused = m.fuse_kv_b(np.ascontiguousarray(k_b), np.ascontiguousarray(v_b))
    assert fused.shape == (n_head * (nope + v_dim), kv_lora)
    np.testing.assert_array_equal(fused, kv_b)


def test_kv_b_fusion_reproduces_the_absorbed_products():
    """Expanded attention (kv_b) == llama.cpp's absorbed k_b / v_b products."""
    rng = np.random.default_rng(0)
    n_head, nope, v_dim, kv_lora = 3, 8, 6, 16
    w_uk = rng.standard_normal((n_head, nope, kv_lora)).astype(np.float32)
    w_uv = rng.standard_normal((n_head, v_dim, kv_lora)).astype(np.float32)
    k_b = np.ascontiguousarray(w_uk.transpose(0, 2, 1))  # GGUF attn_k_b
    v_b = w_uv  # GGUF attn_v_b
    fused = m.fuse_kv_b(k_b, v_b).reshape(n_head, nope + v_dim, kv_lora)

    latent = rng.standard_normal(kv_lora).astype(np.float32)
    q_nope = rng.standard_normal((n_head, nope)).astype(np.float32)
    for h in range(n_head):
        expanded = fused[h] @ latent  # what the model computes with kv_b_proj
        np.testing.assert_allclose(expanded[:nope], w_uk[h] @ latent, rtol=1e-5)
        np.testing.assert_allclose(expanded[nope:], v_b[h] @ latent, rtol=1e-5)
        # score with the expanded key == score with the absorbed query
        np.testing.assert_allclose(
            q_nope[h] @ expanded[:nope], (k_b[h] @ q_nope[h]) @ latent, rtol=1e-4
        )


def test_kv_b_fusion_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        m.fuse_kv_b(np.zeros((2, 8, 4)), np.zeros((3, 4, 8)))
    with pytest.raises(ValueError):
        m.fuse_kv_b(np.zeros((8, 4)), np.zeros((2, 4, 8)))


# ---------------------------------------------------------------------------
# (c) lerope
# ---------------------------------------------------------------------------


def test_lerope_round_trip():
    rng = np.random.default_rng(1)
    alpha = rng.uniform(-0.9, 0.9, 32).astype(np.float32)
    rope_freqs = np.exp(-alpha.astype(np.float32)).astype(np.float32)  # the converter
    back = m.lerope_alpha_from_rope_freqs(rope_freqs)
    assert back.dtype == np.float32 and back.shape == (32,)
    np.testing.assert_allclose(back, alpha, atol=2e-6)
    # exp(alpha) * rope_freqs == 1 (ggml divides theta by rope_freqs)
    np.testing.assert_allclose(np.exp(back) * rope_freqs, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# (d) stacked experts: byte contiguity and shapes for the three quant types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qtype", [T.IQ1_S, T.IQ2_XXS, T.Q8_0])
def test_expert_slices_are_contiguous_views(qtype):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    n_exp, rows, cols = 5, 7, 2 * max(block, 256)
    row_bytes = cols // block * size
    stacked = np.random.default_rng(3).integers(
        0, 256, (n_exp, rows, row_bytes), dtype=np.uint8
    )
    base = stacked.__array_interface__["data"][0]
    slices = m.split_stacked(stacked)
    assert len(slices) == n_exp
    for e, s in enumerate(slices):
        assert s.shape == (rows, row_bytes)
        assert s.flags.c_contiguous and np.shares_memory(s, stacked)
        assert s.__array_interface__["data"][0] - base == e * rows * row_bytes
        np.testing.assert_array_equal(s, stacked[e])

    # the same holds for the torch tensor the plugin streams (no copies)
    t = torch.from_numpy(stacked)
    tbase = t.data_ptr()
    for e, s in enumerate(m.split_stacked(t)):
        assert s.is_contiguous() and s.data_ptr() - tbase == e * rows * row_bytes


@pytest.mark.parametrize("qtype", [T.IQ1_S, T.IQ2_XXS, T.Q8_0])
def test_expert_slice_dequantizes_like_the_stacked_tensor(qtype):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    n_exp, rows, cols = 3, 4, 256
    if qtype == T.Q8_0:
        x = np.random.default_rng(4).standard_normal((n_exp, rows, cols), np.float32)
        stacked = quantize(x, qtype)
    else:
        raw = np.random.default_rng(4).integers(
            0, 256, (n_exp, rows, cols // block * size), dtype=np.uint8
        )
        # keep the fp16 scale finite so equality is meaningful
        raw[..., :2] = np.frombuffer(np.float16(0.25).tobytes(), dtype=np.uint8)
        stacked = raw
    whole = dequantize(stacked, qtype)
    assert whole.shape == (n_exp, rows, cols)
    for e, s in enumerate(m.split_stacked(stacked)):
        np.testing.assert_array_equal(dequantize(s, qtype), whole[e])


def test_expert_names():
    names = m.expert_names("layers.9.experts.0.w2.weight", 128)
    assert names[0] == "layers.9.experts.0.w2.weight"
    assert names[127] == "layers.9.experts.127.w2.weight"
    assert len(set(names)) == 128
    with pytest.raises(ValueError):
        m.expert_names("layers.9.attention.wq.weight", 4)
    with pytest.raises(ValueError):
        m.split_stacked(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# (e) gate == up
# ---------------------------------------------------------------------------


def test_gate_up_type_mismatch_is_rejected():
    m.check_gate_up_types(7, int(T.IQ1_S), int(T.IQ1_S))
    with pytest.raises(ValueError, match="different GGUF quantization types"):
        m.check_gate_up_types(7, int(T.IQ1_S), int(T.IQ2_XXS))

    cfg = real_config()
    inv = [
        m.TensorInfo(i.name, int(T.IQ2_XXS), i.shape)
        if i.name == "blk.30.ffn_up_exps.weight"
        else i
        for i in real_inventory(cfg)
    ]
    with pytest.raises(ValueError, match="Layer 30"):
        m.audit_inventory(inv, cfg)

    stream = m.StreamTransformer()
    assert not stream.consume_type("layers.30.experts.0.w1.weight_type", int(T.IQ1_S))
    with pytest.raises(ValueError, match="Layer 30"):
        stream.consume_type("layers.30.experts.0.w3.weight_type", int(T.IQ2_XXS))


# ---------------------------------------------------------------------------
# (f) dequantization helpers
# ---------------------------------------------------------------------------


def test_dequantize_q8_0_matches_gguf_reference_3d():
    x = np.random.default_rng(5).standard_normal((4, 6, 64)).astype(np.float32)
    q = quantize(x, T.Q8_0)
    out = m.dequantize_to_float32(q, int(T.Q8_0))
    assert out.dtype == np.float32 and out.shape == x.shape
    np.testing.assert_array_equal(out, dequantize(q, T.Q8_0))
    assert np.max(np.abs(out - x)) < 0.05  # Q8_0 quantization error


def test_dequantize_float_inputs_pass_through():
    f32 = np.random.default_rng(6).standard_normal((3, 5)).astype(np.float32)
    np.testing.assert_array_equal(m.dequantize_to_float32(f32, int(T.F32)), f32)
    np.testing.assert_array_equal(m.dequantize_to_float32(f32, None), f32)
    f16 = f32.astype(np.float16)
    np.testing.assert_array_equal(
        m.dequantize_to_float32(f16, int(T.F16)), f16.astype(np.float32)
    )
    bf16 = torch.from_numpy(f32).to(torch.bfloat16)
    raw = bf16.view(torch.uint16).numpy().view(np.uint8)
    np.testing.assert_array_equal(
        m.dequantize_to_float32(raw, int(T.BF16)), bf16.float().numpy()
    )
    with pytest.raises(ValueError):
        m.dequantize_to_float32(np.zeros((2, 34), np.uint8), None)


# ---------------------------------------------------------------------------
# End to end on a small synthetic GGUF: write, read, stream, transform.
# ---------------------------------------------------------------------------

MINI = dict(
    hidden=256,
    heads=2,
    kv_lora=256,
    nope=64,
    rope=64,
    v_dim=128,
    experts=3,
    moe_inter=256,
    dense_inter=256,
    idx_heads=2,
    idx_dim=128,
    vocab=256,
    layers=6,
)
MINI_NOPE = {2, 5}
MINI_SHARED_KV = {5}  # attends to layer 2's cache
MINI_NO_INDEXER = {3}


def mini_config():
    n = MINI["layers"]
    return SimpleNamespace(
        num_hidden_layers=n,
        hidden_size=MINI["hidden"],
        num_attention_heads=MINI["heads"],
        kv_lora_rank=MINI["kv_lora"],
        qk_nope_head_dim=MINI["nope"],
        qk_rope_head_dim=MINI["rope"],
        v_head_dim=MINI["v_dim"],
        moe_intermediate_size=MINI["moe_inter"],
        n_routed_experts=MINI["experts"],
        first_k_dense_replace=1,
        index_topk=16,
        index_n_heads=MINI["idx_heads"],
        index_head_dim=MINI["idx_dim"],
        ssmax_scaling=True,
        indexer_ssmax_scaling=True,
        lerope=True,
        vocab_size=MINI["vocab"],
        nope_layer_flags=[i in MINI_NOPE for i in range(n)],
        kv_source_layers=[2 if i in MINI_SHARED_KV else i for i in range(n)],
        index_sharing=["shared" if i in MINI_NO_INDEXER else "full" for i in range(n)],
    )


def build_mini_gguf(path, rng):
    """Mini GGUF with llama.cpp's layout; also returns the float ground truth."""
    c = MINI
    h, nh, kv = c["hidden"], c["heads"], c["kv_lora"]
    truth: dict[str, np.ndarray] = {}
    writer = gguf.GGUFWriter(str(path), "mistral-large4")

    def f32(name, *shape, value=None):
        x = value if value is not None else rng.standard_normal(shape)
        x = np.asarray(x, np.float32)
        writer.add_tensor(name, x)
        truth[name] = x

    def q8(name, x):
        x = np.asarray(x, np.float32)
        writer.add_tensor(name, quantize(x, T.Q8_0), raw_dtype=T.Q8_0)
        truth[name] = x

    def raw(name, qtype, *shape):
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        data = rng.integers(0, 256, (*shape[:-1], shape[-1] // block * size), np.uint8)
        data[..., :2] = np.frombuffer(np.float16(0.5).tobytes(), np.uint8)
        writer.add_tensor(name, data, raw_dtype=qtype)

    alpha = rng.uniform(-0.5, 0.5, c["rope"] // 2).astype(np.float32)
    raw("token_embd.weight", T.Q6_K, c["vocab"], h)
    f32("output_norm.weight", h)
    raw("output.weight", T.Q6_K, c["vocab"], h)
    f32("rope_freqs.weight", value=np.exp(-alpha))
    truth["alpha"] = alpha
    for i in range(c["layers"]):
        b = f"blk.{i}."
        nope = c["nope"] + (c["rope"] if i in MINI_NOPE else 0)
        f32(b + "attn_norm.weight", h)
        f32(b + "ffn_norm.weight", h)
        raw(b + "attn_q.weight", T.Q6_K, nh * (c["nope"] + c["rope"]), h)
        raw(b + "attn_output.weight", T.Q6_K, h, nh * c["v_dim"])
        # ground-truth kv_b, split exactly like the converter
        kv_b = rng.standard_normal((nh * (nope + c["v_dim"]), kv)).astype(np.float32)
        k_b, v_b = converter_split(kv_b, nh, nope, c["v_dim"])
        truth[f"kv_b.{i}"] = kv_b
        writer.add_tensor(
            b + "attn_k_b.weight",
            quantize(np.ascontiguousarray(k_b), T.Q8_0),
            raw_dtype=T.Q8_0,
        )
        writer.add_tensor(
            b + "attn_v_b.weight",
            quantize(np.ascontiguousarray(v_b), T.Q8_0),
            raw_dtype=T.Q8_0,
        )
        f32(b + "attn_ssmax.weight", nh)
        f32(b + "attn_ssmax.bias", nh)
        if i not in MINI_SHARED_KV:
            rows = kv + (0 if i in MINI_NOPE else c["rope"])
            q8(b + "attn_kv_a_mqa.weight", rng.standard_normal((rows, h)))
            f32(b + "attn_kv_a_norm.weight", kv)
        if i not in MINI_NO_INDEXER:
            q8(b + "indexer.attn_k.weight", rng.standard_normal((c["idx_dim"], h)))
            f32(b + "indexer.proj.weight", c["idx_heads"], h)
            raw(b + "indexer.attn_q_b.weight", T.Q6_K, c["idx_heads"] * c["idx_dim"], h)
            for n, d in (
                ("q_norm", h),
                ("k_norm", c["idx_dim"]),
                ("ssmax", c["idx_heads"]),
            ):
                f32(b + f"indexer.{n}.weight", d)
                f32(b + f"indexer.{n}.bias", d)
        if i == 0:
            raw(b + "ffn_gate.weight", T.Q8_0, c["dense_inter"], h)
            raw(b + "ffn_up.weight", T.Q8_0, c["dense_inter"], h)
            raw(b + "ffn_down.weight", T.Q8_0, h, c["dense_inter"])
        else:
            f32(b + "ffn_gate_inp.weight", c["experts"], h)
            raw(b + "ffn_gate_shexp.weight", T.Q8_0, c["moe_inter"], h)
            raw(b + "ffn_up_shexp.weight", T.Q8_0, c["moe_inter"], h)
            raw(b + "ffn_down_shexp.weight", T.Q8_0, h, c["moe_inter"])
            g, u, d = (
                (T.IQ2_XXS, T.IQ2_XXS, T.IQ2_XXS)
                if i < 3
                else (T.IQ1_S, T.IQ1_S, T.IQ2_XXS)
            )
            raw(b + "ffn_gate_exps.weight", g, c["experts"], c["moe_inter"], h)
            raw(b + "ffn_up_exps.weight", u, c["experts"], c["moe_inter"], h)
            raw(b + "ffn_down_exps.weight", d, c["experts"], h, c["moe_inter"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return truth


def plugin_stream(paths, name_map):
    """Replica of weight_utils.gguf_quant_weights_iterator_multi (cited:
    vllm_gguf_plugin/weight_utils.py:158-199) for float / quantized tensors."""
    for path in paths:
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            if tensor.name not in name_map:
                continue
            name = name_map[tensor.name]
            if tensor.tensor_type.name not in ("F32", "BF16", "F16"):
                yield (
                    name.replace("weight", "weight_type"),
                    torch.tensor(tensor.tensor_type),
                )
            yield name, torch.tensor(tensor.data)


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    path = tmp_path_factory.mktemp("mini") / "mini-mistral-large4.gguf"
    truth = build_mini_gguf(path, np.random.default_rng(7))
    return path, truth


def run_stream(path, cfg, *, split_experts=False):
    inv = m.read_gguf_inventory([str(path)])
    name_map = m.audit_inventory(inv, cfg)
    stream = m.StreamTransformer()
    out: list[tuple[str, torch.Tensor]] = []
    for name, w in plugin_stream([str(path)], name_map):
        if name.endswith(".weight_type") and w.ndim == 0:
            if not stream.consume_type(name, int(w)):
                out.append((name, w))
        elif stream.handles(name):
            for item in stream.feed(name, w.numpy()):
                t = torch.from_numpy(item.array)
                out.append(
                    (item.name, t if item.keep_float32 else t.to(torch.bfloat16))
                )
        elif split_experts and w.ndim == 3 and ".experts.0." in name:
            out.extend(zip(m.expert_names(name, w.shape[0]), m.split_stacked(w)))
        else:
            out.append((name, w))
    stream.finish()
    return inv, out


def test_mini_gguf_arch_and_inventory(mini):
    path, _ = mini
    assert m.read_gguf_architecture(str(path)) == "mistral-large4"
    inv = {i.name: i for i in m.read_gguf_inventory([str(path)])}
    assert inv["blk.2.attn_k_b.weight"].shape == (2, 256, 128)  # (H, kv_lora, nope)
    assert inv["blk.1.ffn_gate_exps.weight"].qtype == int(T.IQ2_XXS)
    assert inv["blk.4.ffn_gate_exps.weight"].qtype == int(T.IQ1_S)
    assert m.audit_inventory(list(inv.values()), mini_config())


def test_mini_end_to_end_reaches_exactly_the_model_parameters(mini):
    path, _ = mini
    cfg = mini_config()
    _, out = run_stream(path, cfg)
    names = [n for n, _ in out]
    assert len(names) == len(set(names))
    reached = {pr_param_for(n) for n in names if n != "lerope.alpha"}
    assert reached == pr_params(cfg, quantized_types=True)
    assert "lerope.alpha" in names


def test_mini_end_to_end_values(mini):
    path, truth = mini
    out = dict(run_stream(path, mini_config())[1])
    # kv_b: fused + dequantized == the matrix the converter started from
    for i in range(MINI["layers"]):
        fused = out[f"layers.{i}.attention.wkv_b.weight"]
        assert fused.dtype == torch.bfloat16
        np.testing.assert_allclose(
            fused.float().numpy(), truth[f"kv_b.{i}"], atol=0.06, rtol=0.02
        )
    # kv_a (Q8_0) and indexer wk (Q8_0) are dequantized to dense bf16
    kv_a = out["layers.4.attention.wkv_a_with_mqa.weight"]
    assert kv_a.shape == (MINI["kv_lora"] + MINI["rope"], MINI["hidden"])
    np.testing.assert_allclose(
        kv_a.float().numpy(), truth["blk.4.attn_kv_a_mqa.weight"], atol=0.06, rtol=0.02
    )
    assert out["layers.2.attention.wkv_a_with_mqa.weight"].shape[0] == MINI["kv_lora"]
    wk = out["layers.1.attention.indexer.wk.weight"]
    np.testing.assert_allclose(
        wk.float().numpy(), truth["blk.1.indexer.attn_k.weight"], atol=0.06, rtol=0.02
    )
    # weights_proj stays float32, ssmax renamed, lerope inverted
    np.testing.assert_array_equal(
        out["layers.1.attention.indexer.weights_proj.weight"].numpy(),
        truth["blk.1.indexer.proj.weight"],
    )
    np.testing.assert_array_equal(
        out["layers.1.attention.ssmax_scaling.s"].numpy(),
        truth["blk.1.attn_ssmax.weight"],
    )
    alpha = out["lerope.alpha"]
    assert alpha.dtype == torch.float32
    np.testing.assert_allclose(alpha.numpy(), truth["alpha"], atol=2e-6)
    # no raw k_b / v_b halves or types of dequantized tensors leak out
    assert not [n for n in out if ".k_b." in n or ".v_b." in n]
    assert "layers.4.attention.wkv_a_with_mqa.weight_type" not in out
    assert "layers.1.attention.indexer.wk.weight_type" not in out


def test_mini_experts_stay_packed_and_typed(mini):
    path, _ = mini
    out = dict(run_stream(path, mini_config())[1])
    c = MINI
    w1 = out["layers.4.experts.0.w1.weight"]
    assert w1.dtype == torch.uint8
    assert w1.shape == (c["experts"], c["moe_inter"], c["hidden"] // 256 * 50)  # IQ1_S
    assert int(out["layers.4.experts.0.w1.weight_type"]) == int(T.IQ1_S)
    assert int(out["layers.4.experts.0.w3.weight_type"]) == int(T.IQ1_S)
    assert int(out["layers.4.experts.0.w2.weight_type"]) == int(T.IQ2_XXS)
    assert int(out["layers.1.experts.0.w2.weight_type"]) == int(T.IQ2_XXS)
    w2 = out["layers.4.experts.0.w2.weight"]
    assert w2.shape == (c["experts"], c["hidden"], c["moe_inter"] // 256 * 66)


def test_mini_split_experts_fallback_is_zero_copy(mini):
    path, _ = mini
    _, whole = run_stream(path, mini_config())
    _, split = run_stream(path, mini_config(), split_experts=True)
    whole = dict(whole)
    names = [n for n, _ in split]
    assert len(names) == len(set(names))
    assert "layers.4.experts.2.w3.weight" in names
    assert "layers.4.experts.0.w3.weight_type" in names
    assert "layers.4.experts.2.w3.weight_type" not in names  # one type per tensor
    stacked = whole["layers.4.experts.0.w1.weight"]
    for name, t in split:
        if name.startswith("layers.4.experts.") and name.endswith(".w1.weight"):
            torch.testing.assert_close(t, stacked[int(name.split(".")[3])])


def test_stream_finish_detects_a_lone_k_b():
    stream = m.StreamTransformer()
    stream.consume_type("layers.3.attention.wkv_b.k_b.weight_type", int(T.Q8_0))
    k_b = quantize(np.zeros((2, 32, 64), np.float32), T.Q8_0)
    assert stream.feed("layers.3.attention.wkv_b.k_b.weight", k_b) == []
    with pytest.raises(ValueError, match="Incomplete"):
        stream.finish()


# ---------------------------------------------------------------------------
# params.json handling
# ---------------------------------------------------------------------------


def test_params_json_is_stripped_in_memory_only():
    params = {
        "dim": 6144,
        "dsa": {"topk": 2048},
        "moe": {"num_experts": 128},
        "quantization": {"qformat_weight": "fp8_e4m3"},
        "vision_encoder": {"image_size": 1540},
        "multimodal": {"x": 1},
        "lerope": True,
    }
    assert m.is_mistral_large4_params(params)
    stripped = m.strip_params_for_gguf(params)
    assert set(stripped) == {"dim", "dsa", "moe", "lerope"}
    assert "quantization" in params and "vision_encoder" in params  # input untouched
    assert not m.is_mistral_large4_params({"dim": 1, "moe": {"a": 1}})
    assert not m.is_mistral_large4_params({"dsa": {"a": 1}})

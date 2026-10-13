"""LoFi-patched LLaMA attention.

Replaces each decoder layer's self-attention with a variant that runs eager
attention (the prefill probabilities are the importance signal), compresses the
cache once after prefill, and applies the mass-correction logit bias during
decode.

Modified from the LLaMA attention in HuggingFace Transformers v4.33, via the
D2O reference pipeline (https://github.com/AIoT-MLSys-Lab/d2o).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from third_party.transformers_v433.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaForCausalLM,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRotaryEmbedding,
)

from ..config import LoFiConfig
from ..core import compress_kv_cache, mass_logit_bias


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last axis, the HF RoPE pairing convention."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, H_kv, S, D) to (B, H_kv * n_rep, S, D) for grouped-query attention."""
    b, n_kv, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    return hidden[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def _causal_mask(bsz: int, tgt_len: int, past_len: int, dtype: torch.dtype,
                 device: torch.device) -> torch.Tensor:
    """Additive causal mask of shape (B, 1, tgt_len, past_len + tgt_len)."""
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    cols = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(cols < (cols + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_len > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_len, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_len)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to (B, H, S, D) at explicit `positions`.

    Positions are passed in rather than inferred, because after compression the
    cache length no longer equals the number of tokens the model has seen.
    """
    c = cos.squeeze(1).squeeze(0)[positions].unsqueeze(1)
    s = sin.squeeze(1).squeeze(0)[positions].unsqueeze(1)
    return (x * c) + (_rotate_half(x) * s)


class LoFiLlamaAttention(nn.Module):
    """LLaMA self-attention that compresses its KV cache once after prefill."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        kv_out = self.num_key_value_heads * self.head_dim
        self.k_proj = nn.Linear(self.hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        lofi_config = getattr(config, "lofi_config", None)
        self.lofi_config: LoFiConfig = lofi_config if lofi_config is not None else LoFiConfig()

        self.compressed = False
        self.merge_counts: torch.Tensor | None = None
        self.stats = None

    def _init_rope(self) -> None:
        scaling = getattr(self.config, "rope_scaling", None)
        if scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            return
        kind = scaling["type"]
        factor = scaling["factor"]
        cls = (LlamaLinearScalingRotaryEmbedding if kind == "linear"
               else LlamaDynamicNTKScalingRotaryEmbedding)
        self.rotary_emb = cls(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=factor,
            base=self.rope_theta,
        )

    def reset_cache(self) -> None:
        """Clear per-sequence state. Must be called between eval examples."""
        self.compressed = False
        self.merge_counts = None
        self.stats = None

    def forward(self, hidden_states, attention_mask=None, position_ids=None,  # noqa: ARG002
                past_key_value=None, output_attentions=False, use_cache=False):
        # attention_mask is part of the caller's signature but unused: the
        # mask is rebuilt here because the cache length no longer matches the
        # number of tokens seen once compression has run.
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        has_past = past_key_value is not None and past_key_value[0] is not None
        past_len = past_key_value[0].shape[-2] if has_past else 0
        mask = _causal_mask(bsz, q_len, past_len, query.dtype, query.device)

        # RoPE tables must cover the true position, which outruns the cache
        # length once the cache has been compressed.
        rope_len = key.shape[-2] + past_len
        if not position_ids.nelement() > 1:
            rope_len = max(rope_len, position_ids.item() + 1)
        cos, sin = self.rotary_emb(value, seq_len=rope_len)
        query = _apply_rope(query, cos, sin, position_ids)
        key = _apply_rope(key, cos, sin, position_ids)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        if has_past:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        logits = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            logits = logits + mask

        if (self.compressed and self.merge_counts is not None
                and self.lofi_config.mass_mode == "logn"):
            bias = mass_logit_bias(self.merge_counts, logits.shape[-1])
            logits = logits + bias.to(logits.dtype)

        # Softmax accumulates in fp16 to match the reference pipeline the
        # reported numbers come from; widening it changes results slightly.
        probs = nn.functional.softmax(logits, dim=-1, dtype=torch.float16).to(query.dtype)
        out = torch.matmul(probs, value).transpose(1, 2).contiguous()
        out = self.o_proj(out.reshape(bsz, q_len, self.hidden_size))

        present = (key, value) if use_cache else None
        if (present is not None and q_len > 1 and not self.compressed
                and not self.lofi_config.disabled):
            result = compress_kv_cache(
                key, value, self.rope_theta, self.lofi_config,
                attn_weights=probs,
            )
            present = (result.keys, result.values)
            self.compressed = True
            self.merge_counts = result.merge_counts
            self.stats = result.stats

        return out, probs if output_attentions else None, present


class LoFiLlamaForCausalLM(LlamaForCausalLM):
    """LLaMA-family causal LM with every self-attention replaced by LoFi's.

    Attach a :class:`~lofi.config.LoFiConfig` as ``config.lofi_config`` before
    calling ``from_pretrained``; otherwise the defaults are used.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer in self.model.layers:
            layer.self_attn = LoFiLlamaAttention(config)


def reset_caches(model: nn.Module) -> None:
    """Reset LoFi state on every layer. Call once per evaluation example."""
    for module in model.modules():
        if isinstance(module, LoFiLlamaAttention):
            module.reset_cache()

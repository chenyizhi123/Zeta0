"""Versioned, parameter-shape-compatible attention for new predictor runs."""

import contextlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

LEGACY_STABILITY = "legacy_v1"
QK_STABILITY = "qk_norm_v1"
STABILITY_VERSIONS = (LEGACY_STABILITY, QK_STABILITY)


def normalize_qk(x):
    """Non-affine per-head RMS normalization in FP32, without square overflow.

    No learned gain/temperature can undo the logit bound sqrt(head_dim).
    The rescaled epsilon preserves RMSNorm(x, eps=1e-6).
    """
    x = x.float()
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled = x / scale
    return scaled * torch.rsqrt(scaled.square().mean(dim=-1, keepdim=True)
                                + (1e-6 / scale) / scale)


class QKNormAttention(nn.MultiheadAttention):
    """MHA state-dict ABI; Q/K norm after projection, V amplitude preserved.

    Short state history uses FP32 projections and math SDPA. Longer visual /
    action attention retains mixed-precision SDPA after bounded Q/K. Never
    delegate to MHA's inference fast path (which would bypass Q/K norm).
    """

    def __init__(self, *args, fp32_math=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.batch_first or not self._qkv_same_embed_dim:
            raise ValueError("QKNormAttention requires batch-first equal-width Q/K/V")
        self.fp32_math = fp32_math

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if need_weights or is_causal or self.bias_k is not None or self.add_zero_attn:
            raise ValueError("unsupported stable attention mode")
        output_dtype = query.dtype
        precision = (torch.autocast(query.device.type, enabled=False)
                     if self.fp32_math else contextlib.nullcontext())
        with precision:
            weights = self.in_proj_weight.chunk(3)
            biases = (self.in_proj_bias.chunk(3) if self.in_proj_bias is not None
                      else (None, None, None))
            projected = []
            for x, w, b in zip((query, key, value), weights, biases):
                if self.fp32_math:
                    x, w, b = x.float(), w.float(), None if b is None else b.float()
                x = F.linear(x, w, b)
                projected.append(x.reshape(x.shape[0], x.shape[1], self.num_heads,
                                           self.head_dim).transpose(1, 2))
            q, k, v = projected
            observer = getattr(self, "stability_observer", None)
            if observer is not None:
                observer(q, k, v)
            q, k = normalize_qk(q).to(v.dtype), normalize_qk(k).to(v.dtype)
            mask = None
            if key_padding_mask is not None:
                padding = key_padding_mask[:, None, None, :]
                mask = (torch.zeros_like(padding, dtype=q.dtype).masked_fill(padding, -torch.inf)
                        if padding.dtype == torch.bool else padding.to(q.dtype))
            if attn_mask is not None:
                other = attn_mask
                if other.ndim == 3:
                    other = other.reshape(q.shape[0], self.num_heads, q.shape[2], k.shape[2])
                if other.dtype == torch.bool:
                    other = torch.zeros_like(other, dtype=q.dtype).masked_fill(other, -torch.inf)
                other = other.to(q.dtype)
                mask = other if mask is None else mask + other
            backend = sdpa_kernel(SDPBackend.MATH) if self.fp32_math else contextlib.nullcontext()
            with backend:
                result = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            result = result.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.embed_dim)
            result = F.linear(result, self.out_proj.weight.float() if self.fp32_math else self.out_proj.weight,
                              self.out_proj.bias.float() if self.fp32_math and self.out_proj.bias is not None
                              else self.out_proj.bias)
        return (result.to(output_dtype) if self.fp32_math else result), None


class QKNormEncoderLayer(nn.TransformerEncoderLayer):
    """Explicit pre-norm forward: torch's fused encoder bypasses custom MHA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.self_attn
        self.self_attn = QKNormAttention(old.embed_dim, old.num_heads, dropout=old.dropout,
                                        batch_first=True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if not self.norm_first:
            raise ValueError("stable encoder requires pre-norm")
        x = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal)
        return x + self._ff_block(self.norm2(x))

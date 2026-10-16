# Part of a minimal torch-SDPA stand-in for the parts of xformers (BSD-3-Clause, facebookresearch/xformers) that
# upstream SoulX-FlashTalk imports. Written for this repository; it contains no xformers code, only the same
# import names. server.py puts models/selfhost/soulx_flashtalk/shims first on sys.path.
import torch
import torch.nn.functional as F

from . import fmha  # noqa: F401
from .fmha.attn_bias import BlockDiagonalMask


def _sdpa(q, k, v, scale):
    # xformers layout [B, M, H, K] -> SDPA [B, H, M, K]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale)
    return o.transpose(1, 2)


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, op=None, **_):
    assert not p, "inference shim: dropout not supported"
    if attn_bias is None:
        return _sdpa(query, key, value, scale)
    assert isinstance(attn_bias, BlockDiagonalMask), type(attn_bias)
    assert query.shape[0] == 1 and key.shape[0] == 1, "block-diagonal masks pack sequences along batch-1"
    qs = torch.split(query, attn_bias.q_seqlen, dim=1)
    ks = torch.split(key, attn_bias.kv_seqlen, dim=1)
    vs = torch.split(value, attn_bias.kv_seqlen, dim=1)
    return torch.cat([_sdpa(q, k, v, scale) for q, k, v in zip(qs, ks, vs)], dim=1)

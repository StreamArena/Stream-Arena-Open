"""flash_attn_func / flash_attn_qkvpacked_func / flash_attn_varlen_func on torch SDPA.

Layouts as flash-attn 2: q [b, sq, hq, d], k/v [b, sk, hk, d] (hq a multiple of hk, GQA); varlen q [total_q, hq, d]
with cu_seqlens. No dropout at inference (dropout_p must be 0); causal / softmax_scale honoured."""
import torch
import torch.nn.functional as F


def _sdpa(q, k, v, softmax_scale=None, causal=False):
    # [b, s, h, d] -> [b, h, s, d]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale,
                                       enable_gqa=k.shape[1] != q.shape[1])
    return o.transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1),
                    alibi_slopes=None, deterministic=False, return_attn_probs=False, **_):
    assert not dropout_p, "inference shim: dropout not supported"
    assert tuple(window_size) == (-1, -1) and alibi_slopes is None
    return _sdpa(q, k, v, softmax_scale, causal)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, **kw)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, dropout_p=0.0,
                           softmax_scale=None, causal=False, deterministic=False, **_):
    assert not dropout_p, "inference shim: dropout not supported"
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for i in range(len(cq) - 1):
        if cq[i + 1] == cq[i]:
            continue
        o = _sdpa(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]], softmax_scale, causal)
        out[cq[i]:cq[i + 1]] = o[0]
    return out

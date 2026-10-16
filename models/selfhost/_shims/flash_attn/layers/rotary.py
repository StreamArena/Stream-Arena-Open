"""flash_attn.layers.rotary.apply_rotary_emb (non-interleaved / GPT-NeoX convention) in torch.

x [b, s, h, d]; cos / sin [s, rotary_dim / 2]. The first rotary_dim channels are rotated as two halves, the rest pass
through, as flash-attn's kernel does."""
import torch


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None):
    assert not interleaved and cu_seqlens is None and not seqlen_offsets
    ro = cos.shape[-1] * 2
    s = x.shape[1]
    cos = cos[:s, None, :].to(torch.float32)
    sin = sin[:s, None, :].to(torch.float32)
    xr = x[..., :ro].to(torch.float32)
    x1, x2 = xr[..., : ro // 2], xr[..., ro // 2:]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
    return torch.cat([rot, x[..., ro:]], dim=-1) if ro < x.shape[-1] else rot

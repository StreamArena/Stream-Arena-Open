"""Minimal stand-in for xformers (BSD-3-Clause, facebookresearch/xformers), written for this repository; it contains
no xformers code. Upstream SoulX-FlashTalk imports xformers; this module provides a portable SDPA implementation. Only ops.memory_efficient_attention with attn_bias None or
fmha.attn_bias.BlockDiagonalMask is provided, computed exactly with torch SDPA. server.py puts
models/selfhost/soulx_flashtalk/shims first on sys.path."""
from . import ops  # noqa: F401

__version__ = "0.0.31+torch-sdpa-shim"

# Part of a minimal torch-SDPA stand-in for the parts of xformers (BSD-3-Clause, facebookresearch/xformers) that
# upstream SoulX-FlashTalk imports. Written for this repository; it contains no xformers code, only the same
# import names. server.py puts models/selfhost/soulx_flashtalk/shims first on sys.path.
from . import attn_bias  # noqa: F401

"""Portable flash_attn entry points implemented with exact softmax attention through torch SDPA.
Servers enable this module by adding models/selfhost/_shims to sys.path.
"""
from .flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func  # noqa: F401

__version__ = "2.4.2+torch-sdpa-shim"

# Part of a minimal torch-SDPA stand-in for the parts of xformers (BSD-3-Clause, facebookresearch/xformers) that
# upstream SoulX-FlashTalk imports. Written for this repository; it contains no xformers code, only the same
# import names. server.py puts models/selfhost/soulx_flashtalk/shims first on sys.path.
class BlockDiagonalMask:
    """Query block i attends only to key/value block i (sequences packed along dim 1)."""

    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_seqlen = [int(x) for x in (kv_seqlen if kv_seqlen is not None else q_seqlen)]
        assert len(self.q_seqlen) == len(self.kv_seqlen)

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None, **_):
        return cls(q_seqlen, kv_seqlen)

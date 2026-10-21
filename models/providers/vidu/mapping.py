"""Edit conditions on Vidu S2 Editing (vidu_s2_editing_all on reference_edit, vidu_s2_editing_t2i on
instruction_edit).

S2 Editing holds exactly one reference image and one editing_type at a time (create, then WS type 13 switch_prompt).
The benchmark's edit state can hold several edited targets, patch one of them or cancel an edit. The mapping sends
what S2 can hold and nothing else:

* the newest active target wins: S2 gets that target's reference image; a target worn by a person goes out as
  virtual_tryon, any other target as subject_replacement;
* when no target is active (an edit was cancelled, or no instruction has arrived yet) S2 gets style_transfer with
  the current source frame as its reference, the closest S2 has to "show the source".

The mapping is fixed before a session and reads the released inputs only. Conditions S2 cannot hold (two edits at
once, a restore) are sampled anyway and judged like any other output.

"Worn by a person" is decided from the target's own text: the reference scope of its slot, or the instruction's
target phrase (GARMENT). On the v1 roster this rule gives exactly the garment sets the published runs used.

Instructions: the composed route turns an instruction into a reference image, so it has to know which target the
instruction edits and how. It understands the six instruction templates of the v1 roster (INSTRUCTIONS); any other
wording, e.g. the free-form instructions of the development roster, has no mapping and is released as unsupported
without a provider call.
"""
from __future__ import annotations

import re

GARMENT = re.compile(r"\bworn\b|\bdraped over [^.;]*\b(?:shoulders|back)\b", re.I)
NO_TARGET = "no_active_target:current_source_frame"
T2I_MODEL = "qwen-image"
T2I_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
T2I_PARAMETERS = {"size": "1024*1024"}
T2I_TIMEOUT_S = 60
T2I_PROMPT = ("Studio product photo of a single {kind}: {target}, now with this look: {look}. {kind_cap} only, "
              "centered on a plain white background. No person, no text, no logo.")

# the instruction templates of the v1 roster
INSTRUCTIONS = (
    ("set", re.compile(r"^Change only (?P<target>.+) to: (?P<look>.+)\. Keep everything else, the current action "
                       r"and the camera unchanged\.$")),
    ("set", re.compile(r"^Update only (?P<target>.+) to: (?P<look>.+); everything else unchanged\.$")),
    ("set", re.compile(r"^Update only (?P<target>.+) to: (?P<look>.+); keep the second target's edit\.$")),
    ("patch", re.compile(r"^Additionally change (?P<target>.+) to: (?P<look>.+); keep the current edit on the "
                         r"first target\.$")),
    ("patch", re.compile(r"^Also change (?P<target>.+) to: (?P<look>.+); keep the first edit\.$")),
    ("restore", re.compile(r"^Cancel the edit: restore (?P<target>.+) to its current original appearance in the "
                           r"live source, without replaying earlier frames\.$")),
)


def garment(text: str) -> bool:
    return bool(GARMENT.search(text or ""))


def editing_type(is_garment: bool) -> str:
    return "virtual_tryon" if is_garment else "subject_replacement"


class EditState:
    """The benchmark's active edits, oldest first: target -> what S2 would be given for it, plus whether the target
    is worn."""

    def __init__(self):
        self.active: dict[str, object] = {}
        self.worn: dict[str, bool] = {}

    def assign(self, target: str, value, worn: bool) -> None:
        self.active.pop(target, None)  # a re-assigned target becomes the newest
        self.active[target] = value
        self.worn[target] = worn

    def remove(self, target: str) -> None:
        self.active.pop(target, None)

    def clear(self) -> None:
        self.active.clear()

    def newest(self):
        """(target, value, worn) of the newest active target, or None when nothing is edited."""
        if not self.active:
            return None
        target = next(reversed(self.active))
        return target, self.active[target], self.worn[target]

    def basis(self) -> str:
        newest = self.newest()
        return NO_TARGET if newest is None else "newest_active_target:" + newest[0]


def apply_reference(state: EditState, payload: dict) -> str:
    """Apply one reference_control payload (initialization or event); returns its operation. Reset and set start
    from an empty state."""
    operation = payload["operation"]
    if operation in ("set", "reset"):
        state.clear()
    for slot in payload.get("remove") or []:
        state.remove(slot)
    for reference in payload.get("references") or []:
        state.assign(reference["target_slot"], reference["asset_id"], garment(reference.get("scope")))
    return operation


def parse_instruction(text: str) -> dict:
    """{"op": "set" | "patch" | "restore", "target", "look"} of one released instruction ("look" is absent for a
    restore). An instruction outside the templates is refused, never guessed."""
    for op, pattern in INSTRUCTIONS:
        match = pattern.match(text)
        if match:
            return {"op": op, **match.groupdict()}
    raise ValueError("instruction outside the v1 templates: " + text[:80])


def apply_instruction(state: EditState, parsed: dict) -> None:
    """A "set" on a target that is not active yet replaces the whole state (first instruction); on an active target
    it updates that target and keeps the others; "patch" adds a target; "restore" removes one."""
    if parsed["op"] == "restore":
        state.remove(parsed["target"])
        return
    if parsed["op"] == "set" and parsed["target"] not in state.active:
        state.clear()
    state.assign(parsed["target"], parsed["look"], garment(parsed["target"]))


def reference_prompt(target: str, look: str) -> tuple[str, str]:
    """(category, text-to-image prompt) of the reference image generated for one instructed look."""
    category = "garment" if garment(target) else "object"
    return category, T2I_PROMPT.format(kind=category, target=target, look=look, kind_cap=category.capitalize())


def generate_reference(prompt: str, api_key: str) -> bytes:
    """One qwen-image reference (PNG bytes) for an instructed look. One request, no retry: a failed or slow
    generation fails the attempt (a client-side failure); it is never replaced by another input."""
    import requests
    reply = requests.post(T2I_URL, headers={"Authorization": "Bearer " + api_key}, timeout=T2I_TIMEOUT_S,
                          json={"model": T2I_MODEL, "parameters": T2I_PARAMETERS,
                                "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]}})
    try:
        url = reply.json()["output"]["choices"][0]["message"]["content"][0]["image"]
    except (KeyError, IndexError, TypeError, ValueError):
        raise RuntimeError(f"qwen-image gave no image (HTTP {reply.status_code})") from None
    image = requests.get(url, timeout=T2I_TIMEOUT_S).content
    if not image.startswith(b"\x89PNG") or len(image) > 4 * 1024 * 1024:
        raise RuntimeError("qwen-image reference is not a PNG within the 4 MiB inline limit")
    return image

"""Full-state planning for editing clients whose only call is ``set(prompt, image)`` (one reference slot).

Such a client (XMax X2.0) cannot apply a reference *operation*; every event becomes the complete
next input. Only a trusted ReleasedInput and a bytes resolver for released assets enter here; no schedule, no SDK
call. "State" is the intended next input, never the state the model is known to have applied.

    text_control   replace the explicit text; resend the retained reference
    set            replace the explicit text and the reference with this event's reference
    patch          merge into the current reference; more than one slot is unsupported
    remove         drop the named slots, keep the explicit text; an empty input is unsupported
    reset          unsupported (no neutral prompt, reconnect or memory-clear is invented)

The provider prompt of a reference state is a frozen template over the authored scope texts only
(``{reference_scope}`` = references[].scope, ``{operation_scope}`` = payload.scope).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import time
from typing import Callable

from ..benchmark.reference import validate_payload
from ..benchmark.tracks import TRACKS
from ..runtime.scheduler import ReleasedInput, UnsupportedInput

TEXT_PROFILES = frozenset({TRACKS["instruction_edit"]["input_profile"]})
REFERENCE_PROFILES = frozenset({TRACKS["reference_edit"]["input_profile"]})


@dataclass(frozen=True)
class ReferenceState:
    asset_id: str
    target_slot: str
    scope: str
    sha256: str
    data: bytes = field(repr=False)

    def metadata(self) -> dict:
        return {"asset_id": self.asset_id, "target_slot": self.target_slot, "scope": self.scope,
                "sha256": self.sha256, "bytes": len(self.data)}


@dataclass(frozen=True)
class EditState:
    explicit_prompt: str | None = None
    reference: ReferenceState | None = None
    reference_operation_scope: str | None = None

    def metadata(self) -> dict:
        return {"explicit_prompt": self.explicit_prompt,
                "reference": self.reference.metadata() if self.reference else None,
                "reference_operation_scope": self.reference_operation_scope}


@dataclass(frozen=True)
class InputPlan:
    event_id: str
    profile: str
    released_at_ns: int
    planned_at_ns: int
    operation: str
    before: EditState
    after: EditState
    provider_prompt: str
    enhance: bool

    def metadata(self) -> dict:
        return {"event_id": self.event_id, "profile": self.profile, "released_at_ns": self.released_at_ns,
                "planned_at_ns": self.planned_at_ns, "operation": self.operation,
                "before": self.before.metadata(), "after": self.after.metadata(),
                "provider_prompt": self.provider_prompt,
                "provider_prompt_sha256": hashlib.sha256(self.provider_prompt.encode()).hexdigest(),
                "enhance": self.enhance}


class EditStatePlanner:
    """Plan one released event at a time; the state changes only when planning succeeds.

    ``asset_bytes_resolver(asset_id)`` returns the verified bytes of a released asset; an asset id may not change
    bytes during the planner's lifetime. A dispatch that fails afterwards does not undo the plan: the planner's
    state is the intended input, not proof of the model's state."""

    def __init__(self, *, asset_bytes_resolver: Callable[[str], bytes], epoch_ns: int, reference_template: str,
                 enhance: bool = False, clock: Callable[[], int] = time.monotonic_ns):
        if "{reference_scope}" not in reference_template:
            raise ValueError("the reference template must bind {reference_scope}")
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise ValueError("invalid session epoch")
        self._resolver, self._clock, self._enhance = asset_bytes_resolver, clock, bool(enhance)
        self._template = reference_template
        self._last_planned_ns = self._last_released_ns = epoch_ns
        self._seen: set[str] = set()
        self._hashes: dict[str, str] = {}
        self._state = EditState()

    @property
    def state(self) -> EditState:
        return self._state

    def plan(self, event: ReleasedInput) -> InputPlan:
        if not isinstance(event, ReleasedInput):
            raise TypeError("a trusted ReleasedInput is required")
        now = self._clock()
        if now < self._last_planned_ns:
            raise ValueError("the clock went backwards")
        if not self._last_released_ns <= event.released_at_ns <= now:
            raise ValueError("event not released yet, or release order regressed")
        if event.event_id in self._seen:
            raise ValueError("event already planned (no implicit retry)")
        payload = deepcopy(event.payload)
        before, operation_scope, new_hashes = self._state, None, {}
        if payload.get("type") == "text_control":
            if event.profile not in TEXT_PROFILES:
                raise UnsupportedInput("text_control_requires_edit_instruction_condition")
            if set(payload) != {"type", "text"} or not isinstance(payload["text"], str) or not payload["text"].strip():
                raise ValueError("text_control needs a non-empty text")
            operation = "text_control"
            after = EditState(payload["text"], before.reference, before.reference_operation_scope)
        elif payload.get("type") == "reference_control":
            if event.profile not in REFERENCE_PROFILES:
                raise UnsupportedInput("reference_control_requires_reference_edit_condition")
            validate_payload(payload)
            operation = payload["operation"]
            if operation == "reset":
                raise UnsupportedInput("empty_state_reset_has_no_expressible_input")
            operation_scope = payload["scope"]
            references = {before.reference.target_slot: before.reference} if before.reference else {}
            explicit = before.explicit_prompt
            if operation == "set":
                references, explicit = {}, None
            if operation in {"set", "patch"}:
                # no bytes are resolved until the whole target shape is supported
                if len(set(references) | {r["target_slot"] for r in payload["references"]}) > 1:
                    raise UnsupportedInput("multiple_independent_reference_slots_not_supported")
                for ref in payload["references"]:
                    data = self._resolver(ref["asset_id"])
                    if type(data) is not bytes or not data:
                        raise ValueError("the resolver must return verified non-empty image bytes")
                    sha = hashlib.sha256(data).hexdigest()
                    if self._hashes.get(ref["asset_id"], sha) != sha:
                        raise ValueError("released asset bytes changed for the same id")
                    new_hashes[ref["asset_id"]] = sha
                    references[ref["target_slot"]] = ReferenceState(ref["asset_id"], ref["target_slot"],
                                                                    ref["scope"], sha, data)
            else:  # remove: keep every slot the event does not name
                for slot in payload["remove"]:
                    references.pop(slot, None)
            reference = next(iter(references.values()), None)
            if reference is None and explicit is None:
                raise UnsupportedInput("empty_state_restore_has_no_expressible_input")
            after = EditState(explicit, reference, operation_scope if reference else None)
        else:
            raise UnsupportedInput("only_text_or_reference_control_supported")
        prompt = compose_prompt(after, operation_scope, self._template)
        planned = self._clock()
        result = InputPlan(event.event_id, event.profile, event.released_at_ns, planned, operation, before, after,
                           prompt, self._enhance)
        self._state = after
        self._hashes.update(new_hashes)
        self._seen.add(event.event_id)
        self._last_planned_ns, self._last_released_ns = planned, event.released_at_ns
        return result


def compose_prompt(state: EditState, operation_scope: str | None, template: str) -> str:
    sections = [state.explicit_prompt] if state.explicit_prompt is not None else []
    scope = operation_scope or (state.reference_operation_scope if state.reference is not None else None)
    if state.reference is not None:
        sections.append(template.format(reference_scope=state.reference.scope, operation_scope=scope or "").strip())
    elif operation_scope:
        sections.append(operation_scope)
    if not sections:
        raise UnsupportedInput("cannot_invent_neutral_prompt_for_empty_state")
    return "\n\n".join(sections)

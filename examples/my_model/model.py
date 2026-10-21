"""Template of an in-process model: a live instruction editor (track instruction_edit).

Copy this folder, replace ``MyModel.edit`` (and ``load``) with your model, adjust system.json, then:

    streameval run --system my_model_instruction --case sa.v1.001.instruction_edit     # one session
    streameval evaluate --system my_model_instruction                                  # the development-roster score

The skeleton is what matters, not the toy edit:

* ``open`` returns at once; loading and the ``ready`` message happen on the model's own thread;
* the harness pushes the source from before the activity epoch (the first frame is held), so the model outputs
  frames before any instruction arrives (an unedited frame is fine);
* the worker always edits the newest source frame and drops older ones: a model slower than the source falls
  behind in frame rate, never in latency;
* every event is acked; an instruction the model cannot carry out is acked "unsupported" (scored as a failure,
  which is the honest outcome).
"""
from __future__ import annotations

import io
import threading
import time

from streameval.models.base import OutputSink, ReferenceEvent, SessionInit, StreamingModel, TextEvent


class MyModel(StreamingModel):
    def __init__(self, device: str = "cpu"):
        self.device = device                    # system.json "kwargs" arrive here
        self.stopped = threading.Event()
        self.new_frame = threading.Condition()
        self.worker: threading.Thread | None = None

    def describe(self) -> dict:
        """Recorded in binding.json: anything that identifies the run (weights, steps, device)."""
        return {"transport": "in-process", "model": "my_model", "device": self.device}

    # ---- the model ---------------------------------------------------------------------------------
    def load(self) -> None:
        """Load weights here (called once per session on the worker thread, before ready)."""

    def edit(self, image, instruction: str | None):
        """One output frame from one source frame (PIL images) under the active instruction. Replace this.

        The toy rule: "black and white" / "grayscale" turns the picture gray; anything else passes through.
        """
        if instruction and any(k in instruction.lower() for k in ("black and white", "grayscale", "greyscale")):
            return image.convert("L").convert("RGB")
        return image

    def understands(self, instruction: str) -> bool:
        """Whether the model can carry out an instruction at all (else it is acked "unsupported")."""
        return True

    # ---- session -----------------------------------------------------------------------------------
    def open(self, init: SessionInit, sink: OutputSink) -> None:
        self.init, self.sink = init, sink
        self.source: tuple[int, bytes] | None = None      # newest (pts_ns, jpeg) not yet edited
        self.instruction: str | None = None
        self.stopped.clear()
        self.worker = threading.Thread(target=self._run, name="my-model", daemon=True)
        self.worker.start()

    def _run(self):
        from PIL import Image
        self.load()
        self.sink.ready(model="my_model", version="0.1", width=self.init.width, height=self.init.height,
                        fps=self.init.fps, config={"seed": self.init.seed, "device": self.device})
        while not self.stopped.is_set():
            with self.new_frame:
                while self.source is None and not self.stopped.is_set():
                    self.new_frame.wait(0.1)
                if self.stopped.is_set():
                    return
                pts_ns, jpeg = self.source
                self.source = None
            image = self.edit(Image.open(io.BytesIO(jpeg)).convert("RGB"), self.instruction)
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=90)
            self.sink.video_frame(buf.getvalue(), time.monotonic_ns())

    # ---- inputs ------------------------------------------------------------------------------------
    def push_source_frame(self, pts_ns: int, jpeg: bytes) -> None:
        with self.new_frame:
            self.source = (pts_ns, jpeg)
            self.new_frame.notify()

    def send_event(self, event: TextEvent | ReferenceEvent) -> None:
        if not isinstance(event, TextEvent) or event.kind != "text_control":
            self.sink.ack(event.event_id, "unsupported", "my_model edits from text instructions only")
            return
        if not self.understands(event.text):
            self.sink.ack(event.event_id, "unsupported", "instruction outside the model's edit vocabulary")
            return
        self.instruction = event.text          # the next edited frame follows it
        self.sink.ack(event.event_id, "applied")

    def close(self) -> bool:
        self.stopped.set()
        with self.new_frame:
            self.new_frame.notify()
        if self.worker is not None:
            self.worker.join(5)
        return self.worker is None or not self.worker.is_alive()

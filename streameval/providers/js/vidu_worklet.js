// StreamEval v1: capture of the received avatar audio inside the page's
// AudioContext, plus an online voice-activity segmenter used only to release
// conditional user turns (the offline scorer recomputes speech from the PCM).
//
// Every render quantum is recorded; frames are indexed by the context's
// currentFrame, so the recording is gapless and its clock is the context clock.
const BLOCK = 4096;          // samples per posted PCM block (~85 ms at 48 kHz)
const WINDOW_S = 0.02;       // VAD window
const VOICED_DBFS = -45;     // absolute level; received TTS speech is far above it
const MIN_VOICED_S = 0.3;    // a segment counts once it has this much voiced time
const MAX_GAP_S = 0.25;      // voiced windows closer than this belong to one segment

class ArenaCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BLOCK); this.n = 0; this.blockStart = null;
    this.win = Math.round(sampleRate * WINDOW_S); this.acc = 0; this.cnt = 0; this.winStart = null;
    this.segStart = null; this.segVoiced = 0; this.lastVoicedEnd = null; this.reported = false;
    this.lastQualifiedEnd = null;
  }

  process(inputs) {
    const input = inputs[0];
    const ch = input && input.length ? input[0] : null;
    const frames = ch ? ch.length : 128;
    for (let i = 0; i < frames; i++) {
      const frame = currentFrame + i;
      const v = ch ? ch[i] : 0;
      if (this.blockStart === null) this.blockStart = frame;
      this.buf[this.n++] = v;
      if (this.n === BLOCK) {
        this.port.postMessage({type: 'pcm', start: this.blockStart, data: this.buf}, [this.buf.buffer]);
        this.buf = new Float32Array(BLOCK); this.n = 0; this.blockStart = null;
      }
      if (this.winStart === null) this.winStart = frame;
      this.acc += v * v; this.cnt++;
      if (this.cnt === this.win) {
        this.window(this.winStart, Math.sqrt(this.acc / this.cnt));
        this.acc = 0; this.cnt = 0; this.winStart = null;
      }
    }
    return true;
  }

  window(frame, rms) {
    const db = 20 * Math.log10(Math.max(rms, 1e-9));
    if (db <= VOICED_DBFS) {
      if (this.speaking && frame - this.lastVoicedEnd > MAX_GAP_S * sampleRate) {
        this.speaking = false;
        this.port.postMessage({type: 'offset', end: this.lastVoicedEnd});
      }
      return;
    }
    const gap = this.lastVoicedEnd === null ? Infinity : frame - this.lastVoicedEnd;
    if (this.segStart === null || gap > MAX_GAP_S * sampleRate) {
      this.segStart = frame; this.segVoiced = 0; this.reported = false;
    }
    this.segVoiced += this.win;
    this.lastVoicedEnd = frame + this.win;
    if (this.segVoiced >= MIN_VOICED_S * sampleRate) {
      this.speaking = true;
      if (!this.reported) {
        this.reported = true;
        this.port.postMessage({type: 'onset', start: this.segStart, previous_end: this.lastQualifiedEnd,
                               detected: this.lastVoicedEnd});
      }
      this.lastQualifiedEnd = this.lastVoicedEnd;
    }
  }
}

registerProcessor('arena-capture', ArenaCapture);

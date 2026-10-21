// StreamEval v1: in-page RTC client for Vidu S2 (AliRTC Web SDK 7.1.9).
// Driven only by commands from vidu_browser.mjs. All page times are
// performance.now() milliseconds; audio times are AudioContext frames.
//
// Chromium runs with fake media devices, so no physical microphone or camera
// can be opened. User audio is a WebAudio track fed from benchmark WAVs; the
// editing source is a canvas track fed frame-by-frame from the source video.
const $ = s => document.querySelector(s);
const V = window.vidu = {};
const S = {mode: null, zero: null, ended: false, sampleMs: 500, nextBin: 0, remoteUid: null,
           frames: 0, audioBlocks: 0, sourceFrames: 0, sampleWrites: [], lines: {}, writes: []};

function emit(event, extra = {}) { window.emitEvent({event, page_ms: performance.now(), ...extra}); }
function line(name, obj) { (S.lines[name] ||= []).push(JSON.stringify(obj)); }
function flushLines() {
  for (const [name, rows] of Object.entries(S.lines)) {
    if (rows.length) { S.writes.push(window.saveLines(name, rows.splice(0))); }
  }
}
setInterval(flushLines, 500);

// WebRTC receive/send statistics every 2 s: what arrived, was decoded, dropped or frozen.
const STAT_KEYS = ['kind', 'framesReceived', 'framesDecoded', 'framesDropped', 'framesPerSecond', 'frameWidth',
  'frameHeight', 'freezeCount', 'totalFreezesDuration', 'pauseCount', 'totalPausesDuration', 'packetsReceived',
  'packetsLost', 'jitter', 'jitterBufferDelay', 'jitterBufferEmittedCount', 'bytesReceived', 'decoderImplementation',
  'framesEncoded', 'framesSent', 'bytesSent', 'targetBitrate', 'qualityLimitationReason', 'encoderImplementation',
  'totalSamplesReceived', 'concealedSamples', 'totalAudioEnergy', 'audioLevel'];
setInterval(async () => {
  for (const [i, pc] of (window.__arenaPCs || []).entries()) {
    if (pc.connectionState === 'closed') continue;
    try {
      const report = await pc.getStats();
      report.forEach(r => {
        if (r.type !== 'inbound-rtp' && r.type !== 'outbound-rtp') return;
        const row = {page_ms: performance.now(), pc: i, type: r.type};
        for (const k of STAT_KEYS) if (r[k] !== undefined) row[k] = r[k];
        line('webrtc_stats.jsonl', row);
      });
    } catch (e) { /* closed while sampling */ }
  }
}, 2000);

function b64(bytes) {
  let s = ''; const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  return btoa(s);
}

// ---- clocks -------------------------------------------------------------
// Context-time -> page-time pairs, sampled continuously; the nearest pair maps
// a context frame to page ms. Both ends of every latency live on this clock.
function pair() {
  const a = performance.now(), t = S.ctx.currentTime, b = performance.now();
  const out = S.ctx.getOutputTimestamp ? S.ctx.getOutputTimestamp() : {};
  const p = {ctx_s: t, page_ms: (a + b) / 2, read_ms: b - a,
             out_ctx_s: out.contextTime ?? null, out_page_ms: out.performanceTime ?? null};
  S.pairs.push(p); line('clock_pairs.jsonl', p);
  return p;
}
function ctxToPage(seconds) {
  let best = S.pairs[S.pairs.length - 1];
  for (let i = S.pairs.length - 1; i >= 0 && i >= S.pairs.length - 4000; i--) {
    if (Math.abs(S.pairs[i].ctx_s - seconds) < Math.abs(best.ctx_s - seconds)) best = S.pairs[i];
  }
  return best.page_ms + (seconds - best.ctx_s) * 1000;
}

V.clock = () => ({page_ms: performance.now(), time_origin_ms: performance.timeOrigin});

// ---- init -----------------------------------------------------------------
V.init = async ({mode, sample_ms, user_audio = {}, source = null}) => {
  S.mode = mode; S.sampleMs = sample_ms;
  const devices = await navigator.mediaDevices.enumerateDevices();
  S.ctx = new AudioContext({sampleRate: 48000, latencyHint: 'interactive', sinkId: {type: 'none'}});
  await S.ctx.audioWorklet.addModule('/worklet.js');
  await S.ctx.resume();
  S.pairs = []; pair(); S.pairTimer = setInterval(pair, 200);
  S.dest = S.ctx.createMediaStreamDestination();
  S.dest.channelCount = 1;
  S.buffers = {};
  for (const [eventId, url] of Object.entries(user_audio)) {
    const bytes = await (await fetch(url)).arrayBuffer();
    S.buffers[eventId] = await S.ctx.decodeAudioData(bytes);
  }
  if (source) {
    const video = $('#source');
    video.src = source.url; video.muted = true; video.loop = false; video.preload = 'auto';
    await new Promise((resolve, reject) => { video.oncanplaythrough = resolve; video.onerror = () => reject(new Error('source load failed')); });
    const canvas = $('#sourceCanvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    S.sourceCtx = canvas.getContext('2d');
    S.sourceCtx.drawImage(video, 0, 0);
    S.sourceTrack = canvas.captureStream(0).getVideoTracks()[0];
    S.sourceTrack.requestFrame();
    S.source = source;
    // Playback state changes of the source element (stalls show up as waiting/playing pairs); recorded so that a
    // source that did not play in real time can be told from source_events.jsonl together with source_frames.jsonl.
    for (const name of ['playing', 'waiting', 'pause', 'ended']) {
      video.addEventListener(name, () => line('source_events.jsonl', {event: name, page_ms: performance.now(),
                                                                      media_time: video.currentTime}));
    }
  }
  return {sample_rate: S.ctx.sampleRate, base_latency_s: S.ctx.baseLatency, output_latency_s: S.ctx.outputLatency ?? null,
          sink: 'none', devices: devices.map(d => ({kind: d.kind, label: d.label})),
          user_audio_s: Object.fromEntries(Object.entries(S.buffers).map(([k, b]) => [k, b.duration])),
          source: source ? {width: $('#source').videoWidth, height: $('#source').videoHeight,
                            duration_s: $('#source').duration} : null};
};

// ---- receiving --------------------------------------------------------------
function onRemoteFrame(now, md) {
  if (S.ended) return;
  S.frames++;
  const page = md.presentationTime ?? now;
  line('remote_video_frames.jsonl', {n: S.frames, presentation_ms: md.presentationTime, expected_display_ms: md.expectedDisplayTime,
    callback_ms: now, media_time: md.mediaTime, presented_frames: md.presentedFrames, receive_ms: md.receiveTime ?? null,
    rtp_timestamp: md.rtpTimestamp ?? null, width: md.width, height: md.height});
  if (S.frames === 1) { emit('first_remote_frame', {presentation_ms: page, width: md.width, height: md.height}); startRecorder(); }
  if (S.zero !== null) {
    const caseMs = page - S.zero;
    const bin = Math.floor(caseMs / S.sampleMs);
    if (caseMs >= 0 && bin >= S.nextBin && bin < S.binLimit) {
      S.nextBin = bin + 1;
      const video = $('#remote'), canvas = $('#sample');
      canvas.width = md.width || video.videoWidth; canvas.height = md.height || video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const meta = {sample_bin: bin, scheduled_bin_start_ms: bin * S.sampleMs, case_ms: caseMs, presentation_ms: page,
                    media_time: md.mediaTime, presented_frames: md.presentedFrames, width: canvas.width, height: canvas.height,
                    selection_basis: 'content_independent'};
      S.sampleWrites.push(new Promise(resolve => canvas.toBlob(async blob => {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        resolve(window.saveFrame(meta, b64(bytes)));
      }, 'image/jpeg', 0.9)));
    }
  }
  $('#remote').requestVideoFrameCallback(onRemoteFrame);
}

async function attachRemote(uid) {
  // uid_hash is the uid's length, not a hash (the uid itself is never recorded); the key name is kept for artifacts.
  if (S.remoteUid && S.remoteUid !== uid) { emit('second_remote_publisher_ignored', {uid_hash: uid.length}); return; }
  S.remoteUid = uid;
  S.engine.setRemoteViewConfig($('#remote'), uid, 1);
  const video = $('#remote');
  video.muted = true;
  if (!S.frameLoop) { S.frameLoop = true; video.requestVideoFrameCallback(onRemoteFrame); }
  try { await video.play(); } catch (e) { emit('remote_play_error', {name: e.name}); }
  startRecorder();
}

async function attachRemoteAudio(uid) {
  if (S.audioAttached) return;
  const track = await S.engine.getAudioTrack(uid);
  if (!track) { emit('remote_audio_track_missing'); return; }
  S.audioAttached = true; S.remoteAudioTrack = track;
  const source = S.ctx.createMediaStreamSource(new MediaStream([track]));
  const node = new AudioWorkletNode(S.ctx, 'arena-capture', {numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1,
                                                             channelCountMode: 'explicit'});
  const mute = S.ctx.createGain(); mute.gain.value = 0;
  source.connect(node); node.connect(mute); mute.connect(S.ctx.destination);
  node.port.onmessage = ({data}) => {
    if (data.type === 'pcm') {
      const pcm = new Int16Array(data.data.length);
      for (let i = 0; i < pcm.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(data.data[i] * 32767)));
      S.audioBlocks++;
      const meta = {first_frame: data.start, samples: pcm.length, page_ms: ctxToPage(data.start / S.ctx.sampleRate)};
      line('remote_audio_blocks.jsonl', meta);
      S.writes.push(window.saveBlob('remote_audio.s16le', b64(new Uint8Array(pcm.buffer))));
    } else if (data.type === 'onset') {
      const rate = S.ctx.sampleRate;
      const onset = {start_frame: data.start, previous_end_frame: data.previous_end, detected_frame: data.detected,
                     start_page_ms: ctxToPage(data.start / rate),
                     previous_end_page_ms: data.previous_end === null ? null : ctxToPage(data.previous_end / rate),
                     detected_page_ms: ctxToPage(data.detected / rate)};
      line('vad_online.jsonl', onset);
      emit('speech_onset', onset);
    } else if (data.type === 'offset') {
      const offset = {end_frame: data.end, end_page_ms: ctxToPage(data.end / S.ctx.sampleRate)};
      line('vad_online.jsonl', offset);
      emit('speech_offset', offset);
    }
  };
  S.captureNode = node;
  emit('remote_audio_attached', {sample_rate: S.ctx.sampleRate});
  startRecorder();
}

function startRecorder(force = false) {
  const video = $('#remote');
  const tracks = [...(video.srcObject ? video.srcObject.getVideoTracks() : [])];
  if (S.remoteAudioTrack) tracks.push(S.remoteAudioTrack);
  const wantAudio = S.mode !== 'editing' && !force;
  if (S.recorder || !tracks.some(t => t.kind === 'video') || (wantAudio && !tracks.some(t => t.kind === 'audio'))) return;
  const mime = tracks.some(t => t.kind === 'audio') ? 'video/webm;codecs=vp8,opus' : 'video/webm;codecs=vp8';
  S.recorder = new MediaRecorder(new MediaStream(tracks), {mimeType: mime, videoBitsPerSecond: 4_000_000});
  S.recorder.ondataavailable = ev => {
    if (!ev.data.size) return;
    S.writes.push(ev.data.arrayBuffer().then(buf => window.saveBlob('recording.webm', b64(new Uint8Array(buf)))));
  };
  S.recorder.start(1000);
  emit('recorder_started', {mime});
}

// ---- join -------------------------------------------------------------------
V.join = async ({rtc, render_uid, kind, publish_audio, encoder}) => {
  const Engine = window.AliRtcEngine;
  const support = await Engine.isSupported();
  if (!support.support) throw new Error('AliRTC web SDK reports this browser unsupported');
  const engine = S.engine = Engine.getInstance();
  engine.on('remoteUserOnLineNotify', uid => emit('rtc_remote_online', {kind: /^live-(bot|video-push)-/.test(uid) ? 'avatar_publisher' : uid === render_uid ? 'render_uid' : 'other'}));
  engine.on('remoteUserOffLineNotify', uid => emit('rtc_remote_offline', {is_output: uid === S.remoteUid}));
  engine.on('bye', code => emit('rtc_bye', {code}));
  engine.on('connectionStatusChange', (status, reason) => emit('rtc_connection', {status, reason}));
  engine.on('localDeviceException', (type, code, msg) => emit('rtc_local_device_exception', {type, code, msg}));
  engine.on('audioPublishStateChanged', (o, n) => emit('rtc_audio_publish', {old: o, state: n}));
  engine.on('videoPublishStateChanged', (o, n) => emit('rtc_video_publish', {old: o, state: n}));
  const wanted = uid => kind === 'editing' ? uid === render_uid : /^live-(bot|video-push)-/.test(uid);
  engine.on('videoSubscribeStateChanged', (uid, o, n) => {
    emit('rtc_video_subscribe', {wanted: wanted(uid), old: o, state: n});
    if (n === 3 && wanted(uid)) attachRemote(uid);
  });
  engine.on('audioSubscribeStateChanged', (uid, o, n) => {
    emit('rtc_audio_subscribe', {wanted: wanted(uid), old: o, state: n});
    if (n === 3 && wanted(uid) && kind !== 'editing') attachRemoteAudio(uid);
  });
  await engine.setChannelProfile('communication');
  await engine.setAudioOnlyMode(false);
  await engine.setDefaultSubscribeAllRemoteAudioStreams(kind !== 'editing');
  await engine.setDefaultSubscribeAllRemoteVideoStreams(true);
  engine.setPlayoutVolume(0);
  await engine.publishLocalVideoStream(false);
  await engine.publishLocalAudioStream(false);
  if (encoder) await engine.setCameraCapturerConfiguration(encoder);
  await engine.joinChannel(rtc.token, rtc.user_id);
  emit('rtc_joined');
  if (publish_audio) {
    await engine.startAudioCapture();
    await engine.switchMicrophone(undefined, S.dest.stream.getAudioTracks()[0]);
    await engine.publishLocalAudioStream(true);
    const published = await engine.getAudioTrack();
    emit('publishing_audio', {custom_track: published === S.dest.stream.getAudioTracks()[0],
                              label: published ? published.label : null});
  }
  if (S.sourceTrack) {
    // Before the epoch the service receives the held first source frame at the
    // source frame rate (a canvas track emits nothing unless frames are requested).
    const held = () => { S.sourceCtx.drawImage($('#source'), 0, 0); S.sourceTrack.requestFrame(); S.heldFrames++; };
    S.heldFrames = 0;
    held();
    S.preroll = setInterval(held, 1000 / S.source.fps);
    await engine.startPreview();
    await engine.switchCamera(undefined, S.sourceTrack);
    await engine.publishLocalVideoStream(true);
    emit('publishing_source', {source_fps: S.source.fps, track_kind: S.sourceTrack.kind});
  }
  return {joined: true};
};

// ---- activity -----------------------------------------------------------------
V.start = async ({zero_page_ms, duration_ms}) => {
  S.zero = zero_page_ms; S.binLimit = Math.ceil(duration_ms / S.sampleMs);
  S.nextBin = 0;
  startRecorder(true);
  if (S.source) {
    clearInterval(S.preroll);
    line('source_frames.jsonl', {held_first_frame_repeats_before_epoch: S.heldFrames});
    const video = $('#source');
    // From the epoch the sender carries the media element's own capture track, which
    // emits every decoded source frame at the native rate (no canvas, no compositor).
    // replaceTrack keeps the negotiated sender and encoder settings. rVFC only logs
    // the presented source PTS for alignment.
    const log = (now, md) => {
      if (S.ended) return;
      S.sourceFrames++;
      line('source_frames.jsonl', {n: S.sourceFrames, presentation_ms: md.presentationTime, callback_ms: now,
                                   media_time: md.mediaTime, presented_frames: md.presentedFrames});
      video.requestVideoFrameCallback(log);
    };
    video.requestVideoFrameCallback(log);
    await video.play();
    const capture = video.captureStream().getVideoTracks()[0];
    const senders = (window.__arenaPCs || []).flatMap(pc => pc.connectionState === 'closed' ? [] : pc.getSenders());
    const sender = senders.find(x => x.track && x.track.kind === 'video');
    if (!sender) throw new Error('no video sender to carry the source');
    await sender.replaceTrack(capture);
    S.captureTrack = capture;
    emit('source_started', {sender_track: 'media_element_capture', settings: capture.getSettings()});
  }
  return {zero_page_ms};
};

// The source frame currently shown by the source element, as JPEG (quality 0.9) at the source's own size. Used as
// the style_transfer reference when no edit target is active (models/providers/vidu/mapping.py).
V.snapshot_source = async () => {
  const video = $('#source');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const meta = {media_time: video.currentTime, page_ms: performance.now(), width: canvas.width, height: canvas.height};
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
  return {...meta, jpeg_base64: b64(new Uint8Array(await blob.arrayBuffer()))};
};

V.play = async ({event_id}) => {
  const buffer = S.buffers[event_id];
  if (!buffer) throw new Error('unknown user audio ' + event_id);
  const node = S.ctx.createBufferSource();
  node.buffer = buffer; node.connect(S.dest);
  const when = S.ctx.currentTime + 0.05;
  node.start(when);
  const pairNow = pair();
  return {event_id, ctx_start_s: when, ctx_end_s: when + buffer.duration, duration_s: buffer.duration,
          page_start_ms: ctxToPage(when), page_end_ms: ctxToPage(when + buffer.duration), pair: pairNow};
};

V.finish = async () => {
  S.ended = true;
  if (S.source) { $('#source').pause(); S.captureTrack?.stop(); }
  if (S.recorder && S.recorder.state !== 'inactive') {
    await new Promise(resolve => { S.recorder.onstop = resolve; S.recorder.stop(); });
  }
  await new Promise(r => setTimeout(r, 300));
  await Promise.all(S.sampleWrites);
  flushLines();
  await Promise.all(S.writes);
  clearInterval(S.pairTimer);
  const stats = {remote_frames: S.frames, audio_blocks: S.audioBlocks, source_frames: S.sourceFrames,
                 samples: S.nextBin, recorder: !!S.recorder};
  if (S.source) {
    const quality = v => { const q = v.getVideoPlaybackQuality(); return {total: q.totalVideoFrames, dropped: q.droppedVideoFrames}; };
    stats.playback_quality = {source: quality($('#source')), remote: quality($('#remote'))};
  }
  try { await S.engine?.leaveChannel(); } catch (e) { stats.leave_error = String(e); }
  try { await S.engine?.destroy(); } catch (e) { stats.destroy_error = String(e); }
  return stats;
};

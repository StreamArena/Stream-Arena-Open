// StreamEval v1 editing tracks: in-page harness of the vendor pages (edit_page_xmax.mjs).
// Driven only by commands from edit_browser.mjs. All page times are performance.now()
// milliseconds; streameval/providers/browser.py maps them to the host clock (min-RTT sync).
//
// Source: a canvas track at the pushed size, redrawn once per decoded source frame
// (requestVideoFrameCallback) and emitted with requestFrame(); before the activity
// epoch the first frame is held at the pushed frame rate. Output: the provider's remote
// video; the first presented frame in each activity bin is saved as JPEG (content
// independent), every presented frame is logged, and the stream is recorded to WebM.
const $ = s => document.querySelector(s);

export function harness() {
  const A = {ready: false};
  const S = {zero: null, ended: false, sampleMs: 250, nextBin: 0, binLimit: 0, frames: 0, sourceFrames: 0,
             held: 0, sampleWrites: [], writes: [], lines: {}, recorder: null, remote: null};
  const emit = (event, extra = {}) => window.emitEvent({event, page_ms: performance.now(), ...extra});
  const line = (name, obj) => { (S.lines[name] ||= []).push(JSON.stringify(obj)); };
  const flushLines = () => {
    for (const [name, rows] of Object.entries(S.lines)) if (rows.length) S.writes.push(window.saveLines(name, rows.splice(0)));
  };
  setInterval(flushLines, 500);
  const STAT_KEYS = ['kind', 'framesReceived', 'framesDecoded', 'framesDropped', 'framesPerSecond', 'frameWidth',
    'frameHeight', 'freezeCount', 'totalFreezesDuration', 'packetsReceived', 'packetsLost', 'jitter', 'bytesReceived',
    'framesEncoded', 'framesSent', 'bytesSent', 'targetBitrate', 'qualityLimitationReason', 'encoderImplementation',
    'decoderImplementation'];
  setInterval(async () => {
    for (const [i, pc] of (window.__arenaPCs || []).entries()) {
      if (pc.connectionState === 'closed') continue;
      try {
        (await pc.getStats()).forEach(r => {
          if (r.type !== 'inbound-rtp' && r.type !== 'outbound-rtp') return;
          const row = {page_ms: performance.now(), pc: i, type: r.type};
          for (const k of STAT_KEYS) if (r[k] !== undefined) row[k] = r[k];
          line('webrtc_stats.jsonl', row);
        });
      } catch (e) { /* closed while sampling */ }
    }
  }, 2000);
  const b64 = bytes => {
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  };
  const assets = {};
  const asset = async id => {
    if (!assets[id]) {
      const response = await fetch('/asset/' + id);
      if (!response.ok) throw new Error('asset unavailable ' + id);
      assets[id] = await response.blob();
    }
    return assets[id];
  };

  A.clock = () => ({page_ms: performance.now(), time_origin_ms: performance.timeOrigin});

  A.init = async ({sample_ms, source, push}) => {
    S.sampleMs = sample_ms; S.push = push;
    const video = $('#source');
    video.src = source.url; video.muted = true; video.loop = false; video.preload = 'auto';
    await new Promise((resolve, reject) => { video.oncanplaythrough = resolve; video.onerror = () => reject(new Error('source load failed')); });
    const canvas = $('#sourceCanvas');
    canvas.width = push.width; canvas.height = push.height;
    S.ctx = canvas.getContext('2d');
    S.ctx.imageSmoothingQuality = 'high';
    S.draw = () => S.ctx.drawImage(video, 0, 0, push.width, push.height);
    S.draw();
    S.track = canvas.captureStream(0).getVideoTracks()[0];
    S.track.requestFrame();
    S.stream = new MediaStream([S.track]);
    S.hold = setInterval(() => { S.draw(); S.track.requestFrame(); S.held++; }, 1000 / push.fps);
    return {source: {width: video.videoWidth, height: video.videoHeight, duration_s: video.duration},
            pushed: {width: canvas.width, height: canvas.height, fps: push.fps}};
  };

  function onRemoteFrame(now, md) {
    if (S.ended) return;
    S.frames++;
    const page = md.presentationTime ?? now;
    line('remote_video_frames.jsonl', {n: S.frames, presentation_ms: md.presentationTime, expected_display_ms: md.expectedDisplayTime,
      callback_ms: now, media_time: md.mediaTime, presented_frames: md.presentedFrames, receive_ms: md.receiveTime ?? null,
      rtp_timestamp: md.rtpTimestamp ?? null, width: md.width, height: md.height});
    if (S.frames === 1) emit('first_remote_frame', {presentation_ms: page, width: md.width, height: md.height});
    if (S.zero !== null) {
      const caseMs = page - S.zero;
      const bin = Math.floor(caseMs / S.sampleMs);
      if (caseMs >= 0 && bin >= S.nextBin && bin < S.binLimit) {
        S.nextBin = bin + 1;
        const video = $('#remote'), canvas = $('#sample');
        canvas.width = md.width || video.videoWidth; canvas.height = md.height || video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const meta = {sample_bin: bin, scheduled_bin_start_ms: bin * S.sampleMs, case_ms: caseMs, presentation_ms: page,
                      media_time: md.mediaTime, presented_frames: md.presentedFrames, width: canvas.width,
                      height: canvas.height, selection_basis: 'content_independent'};
        S.sampleWrites.push(new Promise(resolve => canvas.toBlob(async blob => {
          resolve(window.saveFrame(meta, b64(new Uint8Array(await blob.arrayBuffer()))));
        }, 'image/jpeg', 0.9)));
      }
    }
    $('#remote').requestVideoFrameCallback(onRemoteFrame);
  }

  // Called by the provider page whenever its SDK hands over (or re-binds) the remote stream.
  function attachRemote(stream) {
    const video = $('#remote');
    video.srcObject = stream; video.muted = true;
    emit('remote_stream', {video_tracks: stream.getVideoTracks().length, rebind: S.remote !== null});
    S.remote = stream;
    if (!S.frameLoop) { S.frameLoop = true; video.requestVideoFrameCallback(onRemoteFrame); }
    video.play().catch(e => emit('remote_play_error', {name: e.name}));
    if (!S.recorder && stream.getVideoTracks().length) {
      S.recorder = new MediaRecorder(new MediaStream(stream.getVideoTracks()), {mimeType: 'video/webm;codecs=vp8',
                                                                                 videoBitsPerSecond: 4_000_000});
      S.recorder.ondataavailable = ev => {
        if (ev.data.size) S.writes.push(ev.data.arrayBuffer().then(buf => window.saveBlob('recording.webm', b64(new Uint8Array(buf)))));
      };
      S.recorder.start(1000);
      emit('recorder_started', {mime: 'video/webm;codecs=vp8'});
    }
  }

  A.start = async ({zero_page_ms, duration_ms}) => {
    S.zero = zero_page_ms; S.binLimit = Math.ceil(duration_ms / S.sampleMs); S.nextBin = 0;
    clearInterval(S.hold);
    line('source_frames.jsonl', {held_first_frame_repeats_before_epoch: S.held});
    const video = $('#source');
    const paint = (now, md) => {
      if (S.ended) return;
      S.draw(); S.track.requestFrame();
      S.sourceFrames++;
      line('source_frames.jsonl', {n: S.sourceFrames, presentation_ms: md.presentationTime, callback_ms: now,
                                   media_time: md.mediaTime, presented_frames: md.presentedFrames});
      video.requestVideoFrameCallback(paint);
    };
    video.requestVideoFrameCallback(paint);
    await video.play();
    emit('source_started', {pushed: S.push});
    return {zero_page_ms};
  };

  // Stops sampling/recording and flushes every write; the provider page disconnects around it.
  A.finishMedia = async () => {
    S.ended = true;
    $('#source').pause();
    if (S.recorder && S.recorder.state !== 'inactive') {
      await new Promise(resolve => { S.recorder.onstop = resolve; S.recorder.stop(); });
    }
    await new Promise(r => setTimeout(r, 300));
    await Promise.all(S.sampleWrites);
    flushLines();
    await Promise.all(S.writes);
    S.track?.stop();
    return {remote_frames: S.frames, source_frames: S.sourceFrames, samples: S.nextBin, recorder: !!S.recorder};
  };

  window.arena = A;
  return {A, S, emit, asset, attachRemote};
}

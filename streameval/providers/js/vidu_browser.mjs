#!/usr/bin/env node
// StreamEval v1: isolated browser RTC client for Vidu S2 (AliRTC Web SDK 7.1.9).
//
// models/providers/vidu/backend.py owns session creation, the WS control channel, the
// trusted scheduler and hangup. This process owns only the browser: it joins the RTC
// channel with the credentials Vidu returned, publishes the evaluator's synthetic
// tracks and records what was received, into a private run folder. The RTC join token
// arrives with the 'join' command on stdin, is never written to disk and is redacted
// from every log line.
//
// Isolation: Chromium runs with --use-fake-device-for-media-stream and a silent
// fake-capture file (launch.flags), so no physical microphone or camera can be opened;
// user audio is a WebAudio track fed from benchmark WAVs, and audio output is muted.
//
// Protocol (streameval/providers/browser.py): stdin {id, op, args}; stdout {event, ...};
// command results come back as {event: "reply", id, ok, result | error}.
//
//   node streameval/providers/js/vidu_browser.mjs LAUNCH.json
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import readline from 'node:readline';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const launch = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = launch.out;
const require = createRequire(path.join(launch.runtime, 'package.json'));
const {chromium} = require('playwright');

const emit = obj => process.stdout.write(JSON.stringify(obj) + '\n');
const sha256 = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const LINES = new Set(['remote_video_frames.jsonl', 'remote_audio_blocks.jsonl', 'source_frames.jsonl',
                       'source_events.jsonl', 'clock_pairs.jsonl', 'vad_online.jsonl', 'webrtc_stats.jsonl']);
const BLOBS = new Set(['recording.webm', 'remote_audio.s16le']);

let secret = null;  // the RTC join token of the created live, redacted from every log line
const redact = text => (secret ? String(text).split(secret).join('<redacted>') : String(text));

// Serve only verified files: the pinned SDK, this client's scripts and the job's assets.
const files = {
  '/sdk.js': {file: launch.sdk.path, sha256: launch.sdk.sha256, mime: 'text/javascript'},
  '/page.js': {file: path.join(here, 'vidu_page.js'), mime: 'text/javascript'},
  '/worklet.js': {file: path.join(here, 'vidu_worklet.js'), mime: 'text/javascript'},
};
for (const [name, asset] of Object.entries(launch.assets || {})) {
  files['/asset/' + name] = {file: asset.path, sha256: asset.sha256, mime: asset.mime};
}
for (const [route, entry] of Object.entries(files)) {
  if (entry.sha256 && sha256(entry.file) !== entry.sha256) {
    emit({event: 'fatal', error: 'hash mismatch for ' + route});
    process.exit(3);
  }
}
// Elements are shown at 4x4 CSS px: frames are read from the media, not the layout,
// and full-size compositing of a 1080p source would starve the presentation loop.
const html = '<!doctype html><meta charset="utf-8"><title>arena</title>' +
  '<style>video,canvas{width:4px;height:4px}</style>' +
  '<video id="remote" autoplay muted playsinline></video><video id="source" muted playsinline></video>' +
  '<canvas id="sourceCanvas"></canvas><canvas id="sample"></canvas>' +
  '<script src="/sdk.js"></script><script type="module" src="/page.js"></script>';

function serve(req, res) {
  const url = req.url.split('?')[0];
  if (url === '/') { res.writeHead(200, {'Content-Type': 'text/html'}); res.end(html); return; }
  const entry = files[url];
  if (!entry) { res.writeHead(404); res.end(); return; }
  const size = fs.statSync(entry.file).size;
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
  if (m) {
    const start = m[1] ? Number(m[1]) : 0, end = m[2] ? Math.min(Number(m[2]), size - 1) : size - 1;
    res.writeHead(206, {'Content-Type': entry.mime, 'Accept-Ranges': 'bytes',
                        'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': end - start + 1});
    fs.createReadStream(entry.file, {start, end}).pipe(res);
  } else {
    res.writeHead(200, {'Content-Type': entry.mime, 'Accept-Ranges': 'bytes', 'Content-Length': size});
    fs.createReadStream(entry.file).pipe(res);
  }
}

const server = http.createServer(serve);
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));

// STREAMEVAL_CHROMIUM (via launch.chromium) pins the executable; unset uses Playwright's own build.
const browser = await chromium.launch({headless: true, executablePath: launch.chromium || undefined, args: launch.flags});
const page = await browser.newPage();
// Keep a handle on every peer connection the SDK creates, for getStats().
await page.addInitScript(() => {
  const Native = window.RTCPeerConnection;
  window.__arenaPCs = [];
  function Wrapped(...args) { const pc = new Native(...args); window.__arenaPCs.push(pc); return pc; }
  Wrapped.prototype = Native.prototype;
  Object.setPrototypeOf(Wrapped, Native);
  window.RTCPeerConnection = Wrapped;
});
fs.mkdirSync(path.join(out, 'frames'), {recursive: true, mode: 0o700});
const pageEvents = fs.openSync(path.join(out, 'page_events.jsonl'), 'a', 0o600);
await page.exposeFunction('emitEvent', event => {
  const safe = JSON.parse(redact(JSON.stringify(event)));
  fs.writeSync(pageEvents, JSON.stringify(safe) + '\n');
  emit(safe);
});
await page.exposeFunction('saveLines', (name, rows) => {
  if (!LINES.has(name)) throw new Error('unexpected line file');
  fs.appendFileSync(path.join(out, name), rows.join('\n') + '\n', {mode: 0o600});
});
await page.exposeFunction('saveBlob', (name, data) => {
  if (!BLOBS.has(name)) throw new Error('unexpected blob file');
  fs.appendFileSync(path.join(out, name), Buffer.from(data, 'base64'), {mode: 0o600});
});
await page.exposeFunction('saveFrame', (meta, data) => {
  const bytes = Buffer.from(data, 'base64');
  const relative = `frames/s_${String(meta.sample_bin).padStart(6, '0')}.jpg`;
  fs.writeFileSync(path.join(out, relative), bytes, {flag: 'wx', mode: 0o600});
  const receipt = {...meta, path: relative, sha256: createHash('sha256').update(bytes).digest('hex')};
  fs.appendFileSync(path.join(out, 'frame_samples.jsonl'), JSON.stringify(receipt) + '\n', {mode: 0o600});
});
page.on('console', msg => {
  if (msg.type() === 'error') fs.writeSync(pageEvents, JSON.stringify({event: 'console_error', text: redact(msg.text()).slice(0, 500)}) + '\n');
});
page.on('pageerror', err => emit({event: 'page_error', error: redact(err).slice(0, 500)}));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.waitForFunction(() => window.vidu && window.AliRtcEngine);
emit({event: 'browser_ready', chromium: browser.version()});

let closing = false;
async function shutdown(code) {
  if (closing) return;
  closing = true;
  try { await browser.close(); } catch {}
  server.close();
  fs.closeSync(pageEvents);
  emit({event: 'closed'});
  process.exit(code);
}

const rl = readline.createInterface({input: process.stdin});
rl.on('line', async raw => {
  let cmd;
  try { cmd = JSON.parse(raw); } catch { return; }
  if (cmd.op === 'close') { await shutdown(0); return; }
  const args = cmd.args || {};
  if (cmd.op === 'join') secret = (args.rtc && args.rtc.token) || null;
  try {
    const result = await page.evaluate(([op, a]) => window.vidu[op](a), [cmd.op, args]);
    emit({event: 'reply', id: cmd.id, ok: true, result: JSON.parse(redact(JSON.stringify(result ?? null)))});
  } catch (error) {
    emit({event: 'reply', id: cmd.id, ok: false, error: redact(error).slice(0, 1000)});
  }
});
rl.on('close', () => shutdown(0));

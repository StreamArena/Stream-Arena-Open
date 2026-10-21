#!/usr/bin/env node
// StreamEval v1: isolated browser client for the editing tracks (XMax X2.0).
//
// streameval/providers/edit.py owns credentials, the trusted scheduler, state planning and
// cleanup. This process bundles one provider page (edit_page_<provider>.mjs, which
// imports the provider's JS SDK from the pinned runtime), serves only the job's verified
// assets, and forwards JSON-line commands to window.arena[op](args). Short-lived provider
// credentials arrive with the 'connect' command on stdin and are never written to disk.
//
// Chromium runs with fake devices and muted output: no physical camera or microphone.
// Protocol (streameval/providers/browser.py): stdin {id, op, args}; stdout {event, ...};
// command results come back as {event: "reply", id, ok, result | error}.
//
//   node streameval/providers/js/edit_browser.mjs LAUNCH.json
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
const {build} = require('esbuild');

const emit = obj => process.stdout.write(JSON.stringify(obj) + '\n');
const sha256 = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const LINES = new Set(['remote_video_frames.jsonl', 'source_frames.jsonl', 'webrtc_stats.jsonl']);
const BLOBS = new Set(['recording.webm']);
const PAGES = {xmax_x2: 'edit_page_xmax.mjs'};

let secret = null;  // the short-lived provider credential, redacted from every log line
const redact = text => (secret ? String(text).split(secret).join('<redacted>') : String(text));

if (!PAGES[launch.provider]) { emit({event: 'fatal', error: 'unknown provider page'}); process.exit(3); }
const bundle = path.join(out, 'bundle.js');
// launch.alias maps an imported module to another installed package (XMax: the regional SDK build).
await build({entryPoints: [path.join(here, PAGES[launch.provider])], bundle: true, format: 'esm', outfile: bundle,
             nodePaths: [path.join(launch.runtime, 'node_modules')], logLevel: 'silent',
             ...(launch.alias ? {alias: launch.alias} : {})});
const files = {'/bundle.js': {file: bundle, mime: 'text/javascript'}};
for (const [name, asset] of Object.entries(launch.assets || {})) {
  files['/asset/' + name] = {file: asset.path, sha256: asset.sha256, mime: asset.mime};
}
for (const [route, entry] of Object.entries(files)) {
  if (entry.sha256 && sha256(entry.file) !== entry.sha256) {
    emit({event: 'fatal', error: 'hash mismatch for ' + route});
    process.exit(3);
  }
}
const html = '<!doctype html><meta charset="utf-8"><title>arena</title>' +
  '<style>video,canvas{width:4px;height:4px}</style>' +
  '<video id="remote" autoplay muted playsinline></video><video id="source" muted playsinline></video>' +
  '<canvas id="sourceCanvas"></canvas><canvas id="sample"></canvas>' +
  '<script type="module" src="/bundle.js"></script>';

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
await page.waitForFunction(() => window.arena && window.arena.ready);
emit({event: 'browser_ready', chromium: browser.version(), bundle_sha256: sha256(bundle)});

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
  if (cmd.op === 'connect') secret = args.key || args.token || null;
  try {
    const result = await page.evaluate(([op, a]) => window.arena[op](a), [cmd.op, args]);
    emit({event: 'reply', id: cmd.id, ok: true, result: JSON.parse(redact(JSON.stringify(result ?? null)))});
  } catch (error) {
    emit({event: 'reply', id: cmd.id, ok: false, error: redact(error).slice(0, 1000)});
  }
});
rl.on('close', () => shutdown(0));

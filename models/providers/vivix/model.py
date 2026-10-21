"""Vivix A1 audio dialogue: portrait upload, REST session, Agora media and control WebSocket.

Ported from the evaluated provider adapter; the shared open harness supplies timing and failure accounting.
No historical per-provider retry/void policy is imported. Provider credentials stay in memory or local cache.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import urllib.parse
import uuid
from PIL import Image
from streameval.config import env
from streameval.providers.rtc import RtcAvatar
from streameval.providers.agora import AgoraRoom
from streameval.util import private_json, scrub

API = "https://api.vivix.ai"
MODEL, VOICE, ASR = "vivix-a1-stream", "loongeva_v3.6", "nova-3"
TRANSPORT_RETRIES, CONTROL_OPEN_S, PING_S = 4, 30, 20
RECONNECTS, RECONNECT_WAIT_S = 4, 20


class VivixError(RuntimeError):
    def __init__(self, status: int, code, message: str, retry_after=None):
        super().__init__(f"vivix_http_{status}: code {code}: {message}")
        self.status, self.code, self.retry_after = status, code, retry_after


class Client:
    def __init__(self, key: str):
        self.key = key

    def request(self, method: str, path: str, body=None, files=None, retries: int = TRANSPORT_RETRIES) -> dict:
        """Connection-level failures (TLS EOF, reset, timeout) are retried: GET/close are idempotent, an image
        upload only duplicates an unlisted file, and session creation carries an idempotency_key."""
        import requests
        headers = {"Authorization": "Bearer " + self.key, "User-Agent": "streameval/1.0"}
        for attempt in range(retries + 1):
            try:
                response = requests.request(method, API + path, json=body if files is None else None, files=files,
                                            headers=headers, timeout=60)
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt == retries:
                    raise
                time.sleep(2 + 2 * attempt)
        try:
            data = response.json()
        except ValueError:
            data = {"code": None, "message": response.text[:500]}
        if response.status_code >= 400 or data.get("code") not in (0, None):
            message = scrub(json.dumps({k: data.get(k) for k in ("message", "data")}))[:500].replace(self.key, "<redacted>")
            raise VivixError(response.status_code, data.get("code"), message, response.headers.get("Retry-After"))
        return data.get("data") or {}


class Control:
    """The session's control WebSocket: a receiver thread logs every server event (no credentials).

    A dropped connection is reconnected with the same session credential (documented: reconnecting before the
    disconnected timeout keeps the session); only a session.closed event, or RECONNECTS failed attempts, end it.
    The credential stays in memory only."""

    def __init__(self, url: str, secret: str, note, connect=None, sleep=time.sleep):
        parts = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qsl(parts.query) + [("token", secret)]
        self._url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
        self.url = None
        self.note, self.connect_fn, self.sleep = note, connect, sleep
        self.events, self.lock = [], threading.Lock()
        self.on_event = None
        self.closed = None
        self.ws = None
        self.stopping = False
        self.connected = threading.Event()
        self.reconnects = 0

    def open(self, timeout: float = CONTROL_OPEN_S):
        if self.connect_fn is None:
            from websockets.sync.client import connect
            # client pings every 20 s keep the channel from being dropped as idle (no client frames for minutes
            # made the provider close long sessions with disconnected_timeout), but a late pong never closes it:
            # the provider's control channel can stall its frames for >20 s
            self.connect_fn = lambda url: connect(url, open_timeout=timeout, max_size=None, ping_interval=PING_S,
                                                  ping_timeout=None)
        self.ws = self.connect_fn(self._url)
        self.connected.set()
        threading.Thread(target=self._receive, name="vivix-control", daemon=True).start()

    def _drain(self):
        for raw in self.ws:
            at = time.monotonic_ns()
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            row = summarize(event, at)
            with self.lock:
                self.events.append(row)
            if event.get("type") == "session.closed":
                self.closed = "session.closed: " + str(event.get("reason") or event.get("close_reason") or "")
            if self.on_event is not None:
                self.on_event(event, at)

    def _receive(self):
        try:
            while True:
                try:
                    self._drain()
                    reason = "closed by peer"
                except Exception as exc:  # noqa: BLE001 - a dropped connection, reconnected below
                    reason = f"{type(exc).__name__}: {scrub(str(exc))[:200]}"
                if self.stopping or self.closed is not None:
                    return
                self.connected.clear()
                self.note("control_dropped", detail=reason)
                for attempt in range(RECONNECTS):
                    self.sleep(min(8, 1 + 2 * attempt))
                    try:
                        self.ws = self.connect_fn(self._url)
                    except Exception as exc:  # noqa: BLE001
                        self.note("control_reconnect_failed", attempt=attempt + 1, error=type(exc).__name__)
                        continue
                    self.reconnects += 1
                    self.note("control_reconnected", attempt=attempt + 1)
                    self.connected.set()
                    break
                else:
                    return
        finally:
            self.connected.set()   # release any sender waiting for a reconnection
            self.closed = self.closed or "control_websocket_closed"

    def send(self, event: dict) -> int:
        if not self.connected.wait(RECONNECT_WAIT_S) or self.closed is not None:
            raise RuntimeError("control channel unavailable: " + str(self.closed or "reconnecting"))
        with self.lock:
            self.ws.send(json.dumps(event))
        return time.monotonic_ns()

    def close(self):
        self.stopping = True
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001
                pass


def summarize(event: dict, at: int) -> dict:
    """What result.json keeps of a server event: type, ids, status, error and final texts."""
    row = {"host_ns": at, "type": event.get("type")}
    for key in ("event_id", "response_id", "item_id", "status"):
        if event.get(key) is not None:
            row[key] = event[key]
    response = event.get("response") if isinstance(event.get("response"), dict) else {}
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    if response.get("id"):
        row["response_id"] = response["id"]
    if response.get("status"):
        row["status"] = response["status"]
    if item.get("id"):
        row["item_id"] = item["id"]
    if event.get("type") == "error":
        row["error"] = scrub(json.dumps(event.get("error") or event))[:500]
    if event.get("type") in ("response.output_text.done", "conversation.item.input_audio_transcription.completed"):
        row["text"] = str(event.get("text") or event.get("transcript") or "")[:2000]
    return row


_image_lock = threading.Lock()


def ensure_image(client, portrait: bytes, cache: Path) -> str:
    fingerprint = hashlib.sha256(portrait).hexdigest()
    with _image_lock:
        path = cache / (fingerprint + '.json')
        if path.exists():
            return json.loads(path.read_text())['url']
        mime = 'image/png' if portrait.startswith(b'\x89PNG') else 'image/jpeg'
        created = client.request('POST', '/v1/files/images',
                                 files={'file':(fingerprint[:16]+('.png' if mime=='image/png' else '.jpg'),portrait,mime)})
        if not str(created.get('url','')).startswith('https://'):
            raise RuntimeError('Vivix portrait upload returned no HTTPS URL')
        private_json(path, {'file_id':created.get('file_id'), 'url':created['url'], 'portrait_sha256':fingerprint})
        return created['url']


class SinkBridge:
    """Convert Agora RGB frames on its worker thread into the common output sink."""
    def __init__(self, sink):
        self.sink = sink
        self.audio_ready = threading.Event()

    def attach_audio(self):
        self.audio_ready.set()

    def video(self, width, height, pixels, mode):
        image = Image.frombytes(mode, (width,height), pixels())
        output = io.BytesIO()
        image.save(output, 'JPEG', quality=92)
        self.sink.video_frame(output.getvalue())

    def audio(self, pcm):
        self.sink.audio(pcm)


class VivixA1(RtcAvatar):
    credentials_env = {'api_key':'VIVIX_API_KEY'}
    session_cap_s = 300
    client_transport = 'Agora Python Server SDK 2.4.9, in-process; control WebSocket'
    audio_transport = 'Agora custom PCM track (48 kHz mono, 10 ms blocks)'
    cost = {'unit':'session_minutes','basis':'No public price; duration plus 90 seconds of session overhead, capped at 300 seconds'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = self.sid = self.session = self.control = None

    def estimate(self, case):
        return min(300, case['duration_ms']/1000+90)/60

    def system_config(self, system_id, spec):
        return {**super().system_config(system_id,spec), 'model':MODEL, 'api_origin':API,
                'avatar':{'voice':VOICE,'asr':ASR,'portrait':'exact benchmark portrait, cached upload to /v1/files/images',
                          'llm':'provider default','persona':'initial persona plus known facts'},
                'output':{'aspect_ratio':'1:1','resolution':'720p'}, 'session_cap_ms':300000}

    def preflight(self):
        # Import native/runtime dependencies before uploading media or creating a paid session.
        try:
            from agora.rtc.agora_service import AgoraService  # noqa: F401
            from websockets.sync.client import connect  # noqa: F401
        except (ImportError, OSError) as exc:
            raise RuntimeError('Vivix requires a supported Linux host and pip install -e ".[vivix]"') from exc

    def create(self, init):
        self.preflight()
        if not init.image:
            raise ValueError('Vivix A1 requires the benchmark portrait')
        self.client = Client(self.credentials()['api_key'])
        cache = Path(env('STREAMEVAL_VIVIX_IMAGES','~/.cache/streameval/providers/vivix_images')).expanduser()
        image_url = ensure_image(self.client, init.image, cache)
        request = {'model':MODEL,'output':{'aspect_ratio':'1:1','resolution':'720p'},
            'avatars':[{'avatar_id':'host','instructions':init.persona,
                        'visual':{'source_images':[{'source_image_id':'portrait','url':image_url,
                                  'media_type':'image/png' if init.image.startswith(b'\x89PNG') else 'image/jpeg'}]}}],
            'pipeline_config':{'tts_config':{'tts_voice_id':VOICE},'asr_config':{'provider':ASR}},
            'max_duration_seconds':int(min(300, (init.duration_ms or 15000)/1000+90)),
            'auto_close':{'disconnected_timeout_seconds':30},'recording_mode':'off',
            'delivery':{'media':{'transport':'agora'}},'idempotency_key':'streameval-'+uuid.uuid4().hex}
        self.record['status'] = 'submission_unknown'
        try:
            self.session = self.client.request('POST','/v1/realtime-avatar/sessions',request)
        except VivixError as exc:
            if exc.status < 500:
                self.record['status'] = 'prepared'
            raise
        self.sid = self.session['session_id']
        self.record.update(status='created',session_id=self.sid)
        return {'session_id':self.sid,'output':self.session.get('output'),
                'max_duration_s':request['max_duration_seconds']}

    def join(self):
        media = self.session['delivery']['media']
        if media.get('transport') != 'agora' or not media.get('agora'):
            raise RuntimeError('Vivix returned no Agora delivery credentials')
        agora, control = media['agora'], self.session['control']
        self.control = Control(control['url'],control['client_secret'],self.note)
        self.control.open()
        bridge = SinkBridge(self.sink)
        self.room = AgoraRoom(bridge,publisher=str(agora['publisher_user_id']),note=self.note,publish_audio=True)
        self.room.audio_ready = bridge.audio_ready
        self.room.connect(agora['app_id'],agora['channel_name'],agora.get('token'),str(agora['user_id']))
        self.session = None
        threading.Thread(target=self._monitor,name='vivix-session-monitor',daemon=True).start()

    def _monitor(self):
        while not self.closing:
            reason = self.room.disconnected or self.control.closed
            if reason:
                self.ended(reason)
                return
            time.sleep(.2)

    def end(self):
        errors = []
        if self.sid:
            try:
                self.client.request('POST',f'/v1/realtime-avatar/sessions/{self.sid}/close',{})
            except Exception as exc:
                errors.append(exc)
        if self.control:
            self.control.close()
        if self.room:
            try:
                self.note('agora_stats',**self.room.stats())
                self.room.close()
            except Exception as exc:
                errors.append(exc)
        if not self.sid:
            return self.record['status']=='prepared'
        final = {}
        for _ in range(15):
            try:
                final = self.client.request('GET',f'/v1/realtime-avatar/sessions/{self.sid}')
            except Exception as exc:
                errors.append(exc)
                break
            if final.get('status') in ('closed','failed'):
                break
            time.sleep(2)
        self.record['provider_final_status'] = final.get('status')
        if errors:
            raise errors[0]
        return final.get('status')=='closed'


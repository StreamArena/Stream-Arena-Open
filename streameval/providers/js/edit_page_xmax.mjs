// StreamEval v1: XMax X2.0 page (@xmaxai/sdk-global, pinned by the runtime).
// connect: client.realtime.connect(canvas source stream) with the pushed stream setting;
// for reference_edit the preloaded reference A is uploaded and applied before the epoch.
// set: one session.set({prompt, refImageUrl}) per released event; a released image is
// uploaded with the SDK file client at dispatch time (part of the measured response).
import {createXmaxClient, models} from '@xmaxai/sdk-global';
import {harness} from './edit_page_common.mjs';

const {A, S, emit, asset, attachRemote} = harness();
const uploads = {};

async function upload(assetId) {
  if (!uploads[assetId]) {
    const blob = await asset(assetId);
    const type = blob.type || 'image/png';
    const file = new File([blob], 'reference.' + (type.split('/')[1] || 'png'), {type});
    uploads[assetId] = (await S.client.files.uploadImage(file)).url;
  }
  return uploads[assetId];
}

A.connect = async ({key, model, context, auto_start, stream}) => {
  S.client = createXmaxClient({apiKey: key, telemetry: false});
  let initial;
  if (context) {
    const t0 = performance.now();
    initial = {prompt: context.prompt, refImageUrl: await upload(context.ref_asset)};
    emit('initial_reference_uploaded', {upload_ms: performance.now() - t0});
  }
  S.session = await S.client.realtime.connect(S.stream, {
    model: models.realtime(model), autoStart: auto_start, stream, audio: {publish: false, subscribe: false},
    ...(initial ? {context: initial} : {}),
    onRemoteStream: attachRemote,
    onRemoteVideoFirstFrame: info => emit('provider_first_frame', {width: info.width, height: info.height}),
    onStateChange: state => emit('provider_state', {state}),
    onError: (message, error) => emit('provider_error', {message: String(message).slice(0, 300), code: error?.code}),
    onNotice: notice => emit('provider_notice', {code: notice?.code, message: String(notice?.message).slice(0, 300)}),
    onDisconnect: reason => { S.disconnected = true; emit('provider_disconnect', {reason: String(reason).slice(0, 200)}); },
    onRoomEvent: e => emit('provider_room_event', {name: e?.event ?? null}),
  });
  const media = S.session.media;
  return {session_uid: S.session.getSessionUid(), stream_setting: {...media.streamSetting},
          source_setting: {...media.sourceSetting}, state: S.session.state};
};

A.set = async ({event_id, prompt, ref_asset, clear_ref}) => {
  const start = performance.now();
  const context = {prompt};
  if (ref_asset) context.refImageUrl = await upload(ref_asset);
  else if (clear_ref) context.refImageUrl = null;
  const uploaded = performance.now();
  await S.session.set(context);
  const resolved = performance.now();
  emit('input_sent', {event_id});
  return {page_start_ms: start, page_upload_done_ms: uploaded, page_resolved_ms: resolved,
          reference: ref_asset ? 'uploaded' : clear_ref ? 'cleared' : 'unchanged', state: S.session.state};
};

A.finish = async () => {
  const stats = {};
  try { await S.session?.stopGeneration(); stats.generation_stopped = true; } catch (e) { stats.stop_error = String(e).slice(0, 300); }
  Object.assign(stats, await A.finishMedia());
  try { await S.session?.disconnect(); } catch (e) { stats.disconnect_error = String(e).slice(0, 300); }
  stats.disconnected = !!S.disconnected || S.session?.state === 'disconnected';
  return stats;
};

A.ready = true;

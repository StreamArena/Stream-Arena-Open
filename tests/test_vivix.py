"""Offline checks for the released Vivix REST/Agora adapter; never call a provider."""
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from streameval.models import registry
from streameval.models.base import SessionInit
from streameval.providers.agora import i420_to_rgb


def adapter():
    return registry.build_model(registry.spec_of('vivix_a1_audio')['entry'])


def portrait():
    out = io.BytesIO()
    Image.new('RGB',(8,8)).save(out,'PNG')
    return out.getvalue()


def test_registered_paper_input_and_audio_conversion():
    spec = registry.spec_of('vivix_a1_audio')
    config = registry.system_config('vivix_a1_audio',spec)
    assert config['group']=='digital_human' and config['portrait_policy']=='benchmark_portrait'
    assert config['avatar']['asr']=='nova-3' and config['session_cap_ms']==300000
    model = adapter()
    pcm = (np.ones(160)*2000).astype(np.int16).tobytes()
    assert len(model.upsample(pcm)) == 480*2


def test_session_request_cache_and_confirmed_close(tmp_path,monkeypatch):
    model = adapter()
    module = __import__(type(model).__module__,fromlist=['Client'])
    monkeypatch.setattr(model,'preflight',lambda:None)
    calls = []
    class Client:
        def __init__(self,key):
            assert key=='fixture-secret'
        def request(self,method,path,body=None,files=None):
            calls.append((method,path,body,files))
            if path=='/v1/files/images':
                assert files['file'][1]==portrait()
                return {'url':'https://example.test/portrait.png','file_id':'fixture-image'}
            if path=='/v1/realtime-avatar/sessions':
                return {'session_id':'fixture-session','output':{'resolution':'720p'}}
            return {'status':'closed'}
    monkeypatch.setattr(module,'Client',Client)
    monkeypatch.setenv('VIVIX_API_KEY','fixture-secret')
    monkeypatch.setenv('STREAMEVAL_VIVIX_IMAGES',str(tmp_path))
    init = SessionInit('digital_human_audio','fixture',720,720,24,0,{},persona='Guide',image=portrait(),duration_ms=300000)
    record = model.create(init)
    body = next(x[2] for x in calls if x[1]=='/v1/realtime-avatar/sessions')
    assert body['max_duration_seconds']==300 and body['pipeline_config']['asr_config']['provider']=='nova-3'
    assert body['delivery']['media']['transport']=='agora' and body['idempotency_key']
    assert 'fixture-secret' not in json.dumps(record)
    closed=[]
    model.control=SimpleNamespace(close=lambda:closed.append('control'))
    model.room=SimpleNamespace(close=lambda:closed.append('room'),stats=lambda:{})
    assert model.end() and closed==['control','room']
    model.create(init)
    assert sum(x[1]=='/v1/files/images' for x in calls)==1


def test_agora_i420_conversion_handles_strides():
    # Limited-range black with neutral chroma, and unused padding bytes.
    rgb=i420_to_rgb(2,2,bytes([16,16,255,16,16,255]),bytes([128,0]),bytes([128,0]),3,2,2)
    assert rgb==bytes(12)

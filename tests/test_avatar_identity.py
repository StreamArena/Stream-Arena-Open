import io
from types import SimpleNamespace

import pytest
from PIL import Image

from streameval.judge import units
from streameval.judge.prompts import IDENTITY_FEATURES, AVATAR_QUALITY


@pytest.mark.parametrize('benchmark_portrait', [True, False])
@pytest.mark.parametrize('second_status,expected', [('completed', 75), ('failed', None)])
def test_avatar_identity_uses_declared_reference_and_both_judges(tmp_path, benchmark_portrait, second_status, expected):
    image = io.BytesIO()
    Image.new('RGB', (8, 8), 'blue').save(image, format='JPEG')
    reference = image.getvalue()
    run = SimpleNamespace(benchmark_portrait=benchmark_portrait,
                          reference_frame=lambda: reference,
                          near=lambda t, lo, hi: {'path':'out.jpg','case_ms':t},
                          image=lambda sample: reference)
    seen=[]
    def judge(status, changed):
        def ask(text, images, kind):
            seen.append(images[0][1])
            identity = dict.fromkeys(IDENTITY_FEATURES, 'same')
            identity.update(same_person='yes')
            if changed: identity['hair']='slightly_different'
            return {'status':status,'reply':{'quality':dict.fromkeys(AVATAR_QUALITY,'none'),'temporal':{'appearance_jump':'none','background_jump':'none'},'identity':identity}}
        return SimpleNamespace(ask=ask)
    # Only identity is varied; all quality and temporal categories report no defects.
    rel=SimpleNamespace(asset_jpeg=lambda asset: b'benchmark portrait')
    units.judge_avatar_clip(run,rel,'c0',[0,5000],'photorealistic','portrait',tmp_path,
                           {'primary':judge('completed',False),'second':judge(second_status,True)})
    path=tmp_path/'units/c0.json'
    record=units.read_json(path)
    assert seen == [b'benchmark portrait' if benchmark_portrait else units.jpeg(Image.open(io.BytesIO(reference)))]*2
    assert 'not_applicable' not in record['identity']
    plan={'metrics':{'identity':{'units':[{'unit_id':'identity.c0','applicable':True}]}}}
    decision=units.clip_decisions(plan,{'c0':(record,path)},tmp_path)['identity'][0]
    assert decision['status'] == ('observed' if expected is not None else 'undetermined')
    assert decision.get('value') == expected

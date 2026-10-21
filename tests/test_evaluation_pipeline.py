"""Run the actual CLI pipeline and exact replay with synthetic media and scripted observations.

These fixtures test software plumbing, not judge accuracy or model capability.
"""
from types import SimpleNamespace

import pytest

from streameval import cli
from streameval.config import Endpoint
from streameval.judge import client
from streameval.judge.prompts import GEN_QUALITY, GEN_TEMPORAL, GEN_CONTINUITY, valid
from streameval.util import read_json, write_json
import synthetic


class ScriptedVision(client.VisionJudge):
    def ask(self, text, images, kind, frame_ids=()):
        if kind == 'gen_clip':
            reply = {'quality':dict.fromkeys(GEN_QUALITY,'none'), 'temporal':dict.fromkeys(GEN_TEMPORAL,'none'),
                     'continuity':dict.fromkeys(GEN_CONTINUITY,'none'), 'evidence':'Synthetic software test fixture.'}
            reply['quality']['detail_loss'] = 'minor'
        elif kind == 'gen_follow':
            reply = {'expected_per_frame':dict.fromkeys(frame_ids['frames'],'no'),
                     'protected_violation':'none','evidence':'Synthetic software test fixture.'}
        else:
            raise AssertionError(f'unexpected request kind {kind}')
        assert valid(reply, kind, frame_ids)
        return {**self.record(self.body(text, images), images, kind), 'status':'completed', 'reply':reply}


def scripted_judges():
    endpoint = Endpoint('http://invalid.test', 'scripted-software-test', 'not-a-key')
    return ({'primary':ScriptedVision(endpoint), 'second':ScriptedVision(endpoint),
             'text':SimpleNamespace(model='unused',base_url='')}, SimpleNamespace(model='unused',base_url=''))


@pytest.mark.slow
def test_cli_pipeline_and_recorded_request_replay(tmp_path, monkeypatch):
    bench = synthetic.build(tmp_path/'benchmark')
    registry = tmp_path/'registry'
    write_json(registry/'system.json', {'system_id':'testgen','model_id':'testgen','name':'Synthetic test',
        'provider':'test','system_kind':'native','tracks':['general_interaction'],'portrait_policy':None,
        'python':'streameval.models.mock:MockModel'})
    monkeypatch.setattr(client,'live_judges',scripted_judges)
    out = tmp_path/'run'
    common = ['--benchmark',str(bench.root),'--data',str(bench.data)]
    assert cli.main(['evaluate','--system','testgen_t2v','--registry',str(registry),'--out',str(out),
                     '--population','development','--repetitions','1',*common]) == 0
    job = synthetic.CASE['general_interaction']+'.r01'
    original = read_json(out/'eval/testgen_t2v'/job/'score.json')
    assert original['metrics']['visual_quality']['score'] == 75
    assert original['metrics']['stability']['score'] == 100
    assert original['metrics']['following']['score'] == 0
    board = read_json(out/'leaderboard_development.json')
    assert board['entries'][0]['metrics']['overall']['value'] == 13.75
    assert cli.main(['judge','--runs',str(out),'--out',str(out/'replay'),'--replay',str(out/'eval'),*common]) == 0
    replayed = read_json(out/'replay/testgen_t2v'/job/'score.json')
    assert replayed['metrics'] == original['metrics']
    assert cli.main(['assemble','--runs',str(out),'--eval',str(out/'replay'),'--out',str(out/'replayed-results'),*common]) == 0


def test_empty_judging_is_not_a_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cli,'_judge_set',lambda a: SimpleNamespace())
    monkeypatch.setattr(client,'live_judges',lambda: pytest.fail('no request should be made'))
    assert cli.main(['judge','--runs',str(tmp_path),'--out',str(tmp_path/'eval')]) == 1


def test_unsupported_batch_keeps_frozen_repeats_without_judge_requests(tmp_path, monkeypatch):
    bench = synthetic.build(tmp_path/'benchmark')
    registry = tmp_path/'registry'
    write_json(registry/'system.json', {'system_id':'unsupported','model_id':'unsupported','name':'Unsupported fixture',
        'provider':'test','system_kind':'native','tracks':['general_interaction'],'portrait_policy':None,
        'ops':[], 'unsupported_ops':['user_prompt','camera_control'],
        'python':'streameval.models.mock:MockModel'})
    # A frozen admission imported from an external runner can reject a complete case.
    # The ordinary open runner instead admits cases with unsupported events.
    from streameval.runtime import batch
    admit = batch.admission
    def rejected(*args):
        result = admit(*args)
        for row in result['cases']:
            row.update(admitted=False,reasons=['fixture interface unavailable'])
        result['admitted'] = 0
        return result
    monkeypatch.setattr(batch,'admission',rejected)
    monkeypatch.setattr(client,'live_judges',lambda: pytest.fail('unsupported cases need no judge request'))
    out = tmp_path/'run'
    common = ['--benchmark',str(bench.root),'--data',str(bench.data)]
    assert cli.main(['evaluate','--system','unsupported_t2v','--registry',str(registry),'--out',str(out),
                     '--population','development','--repetitions','2',*common]) == 0
    scores = sorted((out/'eval/unsupported_t2v').glob('*/score.json'))
    assert len(scores) == 2
    for score_path in scores:
        score = read_json(score_path)
        assert score['metrics']['visual_quality']['score'] == 0
    assert read_json(out/'leaderboard_development.json')['entries'][0]['metrics']['overall']['value'] == 0
    plan = out/'unsupported_t2v/development/batch_plan.json'
    tampered = read_json(plan)
    tampered['system_config']['name'] = 'tampered'
    write_json(plan,tampered)
    with pytest.raises(ValueError,match='frozen batch'):
        cli.main(['judge','--runs',str(out),'--out',str(out/'eval'),*common])

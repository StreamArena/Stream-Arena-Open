"""Independent examples of the paper's arithmetic and fixed metric denominators."""
import json

import pytest

from stream_arena import REPO
from stream_arena.judge.prompts import defect_rating, identity_rating, picture_stable, stable, IDENTITY_FEATURES
from stream_arena.scoring.calculation import aggregate_roster, summarize_units
from stream_arena.scoring.leaderboard import markdown, overall_metric, public_board


@pytest.mark.parametrize('defects,score', [([],100), (['minor'],75), (['minor','minor'],50),
                                        (['major','minor'],25), (['major','major'],0)])
def test_quality_scale(defects, score):
    assert 25 * (defect_rating(defects) - 1) == score


def test_stability_requires_temporal_and_quality_gates():
    assert stable(['none', 'minor']) and not stable(['minor', 'minor'])
    assert picture_stable({'stability': {'stable': True}, 'visual_quality': {'rating': 3}})
    assert not picture_stable({'stability': {'stable': True}, 'visual_quality': {'rating': 2}})
    assert picture_stable({'stability': {'stable': True}}) is None


def test_identity_clear_changes_and_uncertainty():
    same = {**dict.fromkeys(IDENTITY_FEATURES, 'same'), 'same_person': 'yes'}
    assert identity_rating(same) == 5
    assert identity_rating({**same, 'same_person': 'unclear'}) == 3
    assert identity_rating({**same, 'eyes': 'clearly_different', 'nose': 'clearly_different'}) == 2


def test_failed_output_and_missing_judgment_have_different_bounds():
    units = [{'unit_id': n, 'kind': 'binary', 'weight': 1, 'applicable': True} for n in ('a', 'b')]
    evidence = {'artifact_ref': 'fixture.json', 'sha256': '0' * 64}
    first = {'unit_id': 'a', 'status': 'pass', 'evidence': evidence}
    unknown = summarize_units(units, [first])
    failed = summarize_units(units, [first, {'unit_id': 'b', 'status': 'output_missing', 'evidence': evidence}])
    assert (unknown['score'], unknown['lower'], unknown['upper']) == (None, 50, 100)
    assert (failed['score'], failed['lower'], failed['upper']) == (50, 50, 50)


def test_equal_duration_weights_with_unequal_case_counts():
    roster, results = [], []
    for i, duration in enumerate([15000,30000,60000,120000,300000]):
        for j in range(2 if i == 0 else 1):
            cid = f'{i}-{j}'
            roster.append(dict(case_id=cid, duration_ms=duration, split='public_dev', group='g',
                               input_profile='p', source_family_id=cid))
            for repeat in (1,2,3):
                score = 100 if i == 0 else 0
                results.append(dict(case_id=cid, repeat=repeat, status='complete', score=score, lower=score, upper=score))
    aggregate = aggregate_roster(roster, results, duration_strata_ms=[15000,30000,60000,120000,300000], repetitions=3)
    assert aggregate['score'] == 20  # not 200/6: each duration, not each condition, has equal final weight


def test_unavailable_metric_does_not_improve_overall():
    metrics = {'quality': {'status':'complete','score':80,'lower':80,'upper':80},
               'av_sync': {'status':'not_established','score':None,'lower':None,'upper':None},
               'identity': {'status':'not_applicable','score':None,'lower':None,'upper':None}}
    score = overall_metric(metrics, {})
    assert score['components'] == ['quality', 'av_sync']
    assert (score['score'], score['lower'], score['upper']) == (None, 40, 90)


def test_paper_scores_and_common_metric_table():
    board = json.loads((REPO/'leaderboard/v1_leaderboard.json').read_text())
    paper = json.loads((REPO/'leaderboard/paper_results.json').read_text())
    core = {r['systemId']:r for r in public_board(board)['entries'] if r['population']=='core'}
    assert len(core) == 49
    for row in paper['entries']:
        for metric, value in row['metrics'].items():
            assert core[row['systemId']]['metrics'][metric]['value'] == value, (row['systemId'], metric)
    table = markdown(board)
    assert '| Common-4' in table and '<u>' not in table and '**Bold**' not in table
    assert '85.83' in table and '85.01' in table

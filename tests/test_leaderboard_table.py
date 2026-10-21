"""Local table rendering uses synthetic values, with no bundled result file."""
import json

from streameval.cli import main
from streameval.scoring import leaderboard


def board():
    def row(sid, name, population, value):
        return {'systemId': sid, 'modelId': sid, 'name': name, 'group': 'general_interaction',
                'population': population,
                'metrics': {m: {'status': 'complete', 'score': value, 'lower': value, 'upper': value}
                            for m in ('visual_quality', 'stability', 'following', 'continuity', 'overall')}}
    return {'entries': [row('synthetic-z', 'Z fixture', 'core', 25),
                        row('synthetic-a', 'A fixture', 'core', 75),
                        row('synthetic-dev', 'Development fixture', 'development', 50)]}


def test_local_table_preserves_rows_and_sorts_by_name():
    text = leaderboard.markdown(board())
    assert text.index('| A fixture') < text.index('| Z fixture')
    assert '| A fixture | 75.00 | 75.00 | 75.00 | 75.00 | 75.00 |' in text
    assert '| Z fixture | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 |' in text
    assert 'Development fixture' in text
    assert '| Common-4' not in text and '<u>' not in text


def test_local_aggregate_does_not_export_execution_evidence():
    source = board()
    source['entries'][0]['private_diagnostics'] = {'account': 'synthetic-secret'}
    source['entries'][0]['metrics']['latency'] = {'p50_s': 12.345}
    source['entries'][0]['metrics']['visual_quality']['ci95'] = [0, 100]
    exported = leaderboard.encode(source)
    assert b'synthetic-secret' not in exported and b'12.345' not in exported and b'ci95' not in exported
    assert len(json.loads(exported)['entries']) == 3


def test_cli_renders_explicit_local_file(tmp_path, capsys):
    path = tmp_path/'own-results.json'
    path.write_bytes(leaderboard.encode(board()))
    assert main(['leaderboard', '--table', '--out', str(path)]) == 0
    assert 'Development fixture' in capsys.readouterr().out


def test_cli_requires_explicit_path(capsys):
    assert main(['leaderboard', '--table']) == 2
    assert 'give --out FILE' in capsys.readouterr().err
    assert main(['leaderboard']) == 2
    assert 'give --out FILE' in capsys.readouterr().err


def test_cli_reports_missing_local_file(tmp_path, capsys):
    assert main(['leaderboard', '--table', '--out', str(tmp_path/'missing.json')]) == 2
    assert 'local board file not found' in capsys.readouterr().err


def test_missing_calibration_keeps_metrics_provisional(tmp_path, monkeypatch):
    monkeypatch.setattr(leaderboard, 'AV_SYNC_VALIDATION', tmp_path/'absent-av.json')
    monkeypatch.setattr(leaderboard, 'PROTOCOL', tmp_path)
    assert leaderboard.av_sync_meta(tmp_path, 'digital_human', [])['provisional'] is True
    assert leaderboard.content_validated({}) is False

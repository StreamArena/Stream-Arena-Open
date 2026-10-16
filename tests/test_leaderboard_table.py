"""Publication boundary: paper cells survive; unreported observations do not."""
import copy
import json

import pytest

from stream_arena import REPO
from stream_arena.scoring import leaderboard


def board():
    return json.loads((REPO / "leaderboard/v1_leaderboard.json").read_text())


def test_table_is_current():
    tables = leaderboard.markdown(board())
    assert (REPO / "leaderboard/v1_leaderboard.md").read_text().endswith(tables + "\n")


def test_publication_removes_nonpaper_rows_and_diagnostics():
    source = board()
    expected = leaderboard.paper_board(source)
    extra = copy.deepcopy(source['entries'][0])
    extra['population'] = 'extension'
    source['entries'].append(extra)
    other = copy.deepcopy(extra)
    other.update(population='core', systemId='unpublished-fixture', modelId='unpublished-fixture')
    source['entries'].append(other)
    for row in source['entries']:
        row['metrics']['latency'] = {'p50_s': 12.345, 'coverage_percent': 67.8}
        row['metrics']['private_probe'] = {'value': 98.765}
        row['private_diagnostics'] = {'failures': 123}
        row['metrics']['visual_quality']['ci95'] = [0.123, 99.876]
    actual = leaderboard.paper_board(source)
    assert actual == expected
    text = leaderboard.markdown(actual)
    assert 'P50' not in text and 'Coverage' not in text
    assert '12.345' not in text and '67.8' not in text
    assert len(actual['entries']) == 49


def test_publication_rejects_changed_or_missing_paper_values():
    source = board()
    source['entries'][0]['metrics']['visual_quality']['value'] += 1
    with pytest.raises(ValueError, match='Paper score differs'):
        leaderboard.paper_board(source)
    source = board()
    source['entries'].pop()
    with pytest.raises(ValueError, match='Missing paper configurations'):
        leaderboard.paper_board(source)


def test_local_evaluation_of_other_populations_remains_available():
    source = board()
    source['entries'] = [source['entries'][0]]
    source['entries'][0]['population'] = 'development'
    assert leaderboard.public_board(source)['entries'][0]['population'] == 'development'

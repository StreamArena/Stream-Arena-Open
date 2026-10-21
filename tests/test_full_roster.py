"""The default benchmark retains the complete set through runtime preparation and aggregation."""
from collections import Counter
from streameval.benchmark.dataset import Benchmark
from streameval.benchmark.roster import Roster
from streameval.runtime.batch import sampling_jobs
from streameval.scoring.leaderboard import population_ids


def test_full_benchmark_default_retains_every_condition(tmp_path):
    bench = Benchmark()
    assert bench.root.name == 'full'
    assert len(bench.cases) == 300
    assert set(bench.cases) == set(bench.oracles)
    assert Counter((c['group'], c['duration_ms']) for c in bench.cases.values()) == {
        (g, seconds * 1000): 12
        for g in ('digital_human', 'avatar_performance', 'general_interaction', 'instruction_edit', 'reference_edit')
        for seconds in (15, 30, 60, 120, 300)}
    for case in bench.cases.values():
        assert set(case['asset_ids']) <= set(bench.assets)
    jobs = sampling_jobs(bench, tmp_path / 'full', population='full')
    assert len(jobs) == 900
    assert Counter(j['case_id'] for j in jobs) == dict.fromkeys(bench.cases, 3)
    roster = Roster(tmp_path / 'full')
    assert len(roster.plans('full')) == 300
    selected = set()
    for group, profile in {(c['group'], c['input_profile']) for c in bench.cases.values()}:
        ids = population_ids('full', group, profile, roster)
        expected = {c['case_id'] for c in bench.cases.values() if (c['group'], c['input_profile']) == (group, profile)}
        assert set(ids) == expected
        assert not selected.intersection(ids)
        selected.update(ids)
    assert selected == set(bench.cases)

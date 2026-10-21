"""Check release contents, scoring policy and frozen executable source hashes."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Maintainer: refresh executable-file hashes after review')
    args = parser.parse_args()
    for name in ('leaderboard', 'results', 'runs', 'recordings', 'samples', 'outputs', 'attempts',
                 'traces', 'replays', 'logs', 'videos', '.work', 'local-reports', 'audit'):
        require(not (ROOT/name).exists(), f'Local evaluation artifacts in release: {name}')
    for path in (ROOT/'benchmark').rglob('*.json*'):
        text = path.read_text()
        require('.work/' not in text and '/home/' not in text and '/Users/' not in text,
                'Private construction-review path in benchmark: ' + str(path.relative_to(ROOT)))
    core = ROOT/'benchmark/v1/full'
    cases = [json.loads(line) for line in (core/'cases.jsonl').read_text().splitlines() if line.strip()]
    oracles = [json.loads(line) for line in (core/'oracles.jsonl').read_text().splitlines() if line.strip()]
    groups = ('digital_human', 'avatar_performance', 'general_interaction', 'instruction_edit', 'reference_edit')
    expected = {(group, seconds * 1000): 12 for group in groups for seconds in (15,30,60,120,300)}
    require(Counter((row['group'], row['duration_ms']) for row in cases) == expected,
            'Full benchmark must include every task and duration stratum')
    ids = {row['case_id'] for row in cases}
    require(len(ids) == len(cases) == len(oracles) == 300 and ids == {row['case_id'] for row in oracles},
            'Full cases and oracles must match one-to-one')
    require(not (ROOT/'benchmark/v1/excluded').exists(), 'All core conditions belong in the core roster')
    protocol = ROOT/'evaluation/protocol'
    require(not list(protocol.glob('*validation*.json')), 'Historical validation results must stay local')
    policy = json.loads((protocol/'evaluation_v2.json').read_text())
    scoring = json.loads((protocol/'scoring_v2.json').read_text())
    require(policy['publication']['numerical_scope'] ==
            'no_evaluation_results; runtime_parameters_and_benchmark_definitions_only', 'Publication scope differs')
    require(policy['aggregation']['table_highlights'] == 'none', 'Table policy differs')
    require(scoring['metrics']['stability']['quality_rating_min'] == 3, 'Stability quality gate differs')
    require(policy['aggregation']['repetitions_per_condition'] == 3, 'Repetition policy differs')
    require(policy['aggregation']['duration_strata_s'] == [15,30,60,120,300], 'Duration weights differ')
    from streameval.models.registry import systems
    registered = systems()
    require(bool(registered), 'No model adapters registered')
    from streameval.judge import prompts
    from streameval.judge.av_sync import AV_VERSION
    version = {'visual':prompts.JUDGE_VERSION, 'extension':prompts.EXT_VERSION,
               'stability':prompts.STABILITY_RULE, 'av_sync':AV_VERSION,
               'visual_prompts_sha256':prompts.PROMPTS_SHA256,
               'extension_prompts_sha256':prompts.EXT_PROMPTS_SHA256}
    frozen_path = protocol/'release_manifest.json'
    sources = [p for base in ('streameval','models','examples','evaluation/protocol','scripts')
               for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.mjs','.js','.sh')
               and p != frozen_path and not {'__pycache__', 'node_modules'}.intersection(p.parts)]
    sources += [ROOT/'pyproject.toml']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    manifest = {'schema':'streameval.executable-release/1', 'runtime':version,
                'validation':'Matching source hashes do not establish measurement validity; validate the configured judge independently.',
                'files':hashes}
    if args.freeze:
        frozen_path.write_text(json.dumps(manifest,indent=2)+'\n')
    require(frozen_path.exists() and json.loads(frozen_path.read_text()) == manifest,
            'Executable sources changed; review and freeze the release before publishing')
    print(json.dumps({'ok':True,'frozen_files':len(hashes),'bundled_evaluation_results':False}))


if __name__ == '__main__':
    main()

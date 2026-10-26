"""Check stored measurements without a GPU, network access, or generated-code execution."""
import hashlib
import json
import math
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for row in json.loads((root / 'results/comparison.json').read_text()):
    directory = root / 'results' / row['variant']
    runs = []
    for case in ('prose', 'code', 'long_prose'):
        group = [json.loads((directory / f'{case}_r{i}.json').read_text()) for i in (1, 2, 3)]
        for run in group:
            assert not run['held_during_reply']
            assert run['usage']['completion_tokens'] == 256
            chunks = run['token_chunks']
            assert sum(len(c[1]) for c in chunks) == 256
            decode = sum(len(c[1]) for c in chunks[1:]) / (chunks[-1][0] - chunks[0][0])
            assert math.isclose(decode, run['decode_tokens_per_s'])
            assert math.isclose(256 / run['elapsed_s'], run['end_to_end_tokens_per_s'])
        assert math.isclose(statistics.median(r['decode_tokens_per_s'] for r in group), row['cases'][case]['decode_median'])
        runs.extend(group)
    score = sum(r['usage']['completion_tokens'] for r in runs) / sum(r['elapsed_s'] for r in runs)
    assert math.isclose(score, row['aggregate_request_tokens_per_s'])
    before = json.loads((directory / 'warmup-counters.json').read_text())
    after = json.loads((directory / 'code_r3-counters.json').read_text())
    for key in before:
        assert after[key] - before[key] == row['draft_metrics'][key]
    quality = json.loads((directory / 'quality-checks.json').read_text())
    assert quality == row['quality']
    assert all(v['passed'] if isinstance(v, dict) else v for v in quality.values())
    code = json.loads((directory / 'rawcode-verification.json').read_text())
    assert code == row['code_checks'] and code['passed'] and len(code['checks']) == 13
    print(f"{row['variant']}: {score:.4f} tok/s; stored checks pass")
manifest = json.loads((root / 'chisel/source-manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / 'chisel' / name).read_bytes()).hexdigest() == expected
print('PASS: 63 stored timing requests, recorded quality results, and Chisel source hashes')

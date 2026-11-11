import csv
import hashlib
import json
from pathlib import Path

output = Path(__file__).resolve().parent
root = output.parent
prior = root / 'CORRECTED_RESULT_TABLE_20261104'
deliveries = root / 'D4P_CORRECTED_TEST_FOUR_DOMAINS_20261104'
thresholds = (1, 3, 5, 10)
methods = {
    'original': 'Original', 'presence': 'Presence', 'rank_fusion': 'Rank Fusion',
    'fsr': 'FSR', 'scar': 'SCAR',
}
domains = {
    'D4P_PANDA': ('pandaset', 2000, 25, 4),
    'D4P_NUSC': ('nuscenesfull', 7403, 210, 16),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (output / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


main_input = prior / 'RESULTS_SOURCE.csv'
long_input = prior / 'SCAR_MULTITHRESHOLD_LONG_AFTER_C.csv'
previous_hashes = prior / 'SOURCE_HASHES.json'
hashes = read_json(previous_hashes)
inputs = [main_input, long_input, previous_hashes]
main = read_csv(main_input)
long = read_csv(long_input)
main_by_id = {row['id']: row for row in main}
assert len(main) == len(main_by_id) == 25
assert len(long) == len({row['row_id'] for row in long}) == 125
assert {row['id'] for row in main if row['formal_status'] == 'PENDING'} == domains.keys()

for config, (domain, queries, clusters, epoch) in domains.items():
    folder = deliveries / domain
    manifest_path = folder / 'SHA256SUMS.txt'
    manifest = {}
    for line in manifest_path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            sha, name = line.split(maxsplit=1)
            manifest[name.lstrip('*').replace(chr(92), '/').removeprefix('./')] = sha.lower()

    metrics_path = folder / 'test/eval/metrics.json'
    verify_path = folder / 'test/VERIFY_TEST.json'
    for path in (metrics_path, verify_path):
        assert digest(path) == manifest[path.relative_to(folder).as_posix()], path
    inputs.extend((manifest_path, metrics_path, verify_path))

    metrics = read_json(metrics_path)
    verify = read_json(verify_path)
    invariants = metrics['invariants']
    assert verify['status'] == 'PASS' and not verify['failures'] and len(verify['checks']) == 154
    assert verify['domain'] == metrics['protocol']['domain'] == domain
    assert verify['split'] == metrics['protocol']['split'] == 'test'
    assert verify['n_queries'] == invariants['n_queries'] == queries
    assert verify['n_clusters'] == invariants['n_clusters'] == clusters
    assert metrics['main_head']['formal_epoch'] == epoch
    assert metrics['exploratory_not_confirmatory']
    assert invariants['same_pool'] and invariants['same_coordinates']

    notes = f'修正版追加探索性 TEST，s0 ep{epoch}，VERIFY PASS。{queries} 查询 / {clusters} 查询场景簇。'
    if domain == 'nuscenesfull':
        notes += '候选库覆盖 213 个场景。'
    row = main_by_id[config]
    row.update(source=str(metrics_path), notes=notes, formal_status='AVAILABLE')
    long_rows = {item['method_key']: item for item in long if item['config_id'] == config}
    assert set(long_rows) == methods.keys()
    for key, label in methods.items():
        result_key = 'scar_s0' if key == 'scar' else key
        loc = metrics['results_loc'][result_key]
        row[f'{label}_top1'] = str(100 * loc['Top1@5m'])
        row[f'{label}_cell'] = str(100 * metrics['results_cell'][result_key]['R@1'])
        item = long_rows[key]
        assert int(item['n_queries']) == queries
        rates = [loc[f'Top1@{t}m'] for t in thresholds]
        assert sorted(rates) == rates and all(0 <= rate <= 1 for rate in rates)
        for t, rate in zip(thresholds, rates):
            assert item[f'status_{t}m'] == 'PENDING'
            hits = round(rate * queries)
            assert abs(hits / queries - rate) < 1e-12
            item[f'top1_{t}m_pct'] = f'{100 * rate:.4f}'
            item[f'hits_{t}m'] = str(hits)
            item[f'status_{t}m'] = 'REUSED'
            item[f'source_{t}m'] = f'{metrics_path}#results_loc/{result_key}/Top1@{t}m'
        item.update(
            monotonic_ok='TRUE', reconcile_5m='MATCH_SOURCE',
            source_sha_1m=manifest['test/eval/metrics.json'],
            source_sha_3m=manifest['test/eval/metrics.json'],
            input_ids='', notes=notes,
        )
        if key == 'scar':
            item['scar_head'] = f"s0 ep{epoch} {metrics['main_head']['sha256']}"

for item in long:
    assert all(item[f'status_{t}m'] != 'PENDING' for t in thresholds)
    values = [float(item[f'top1_{t}m_pct']) for t in thresholds]
    assert 0 <= values[0] <= values[1] <= values[2] <= values[3] <= 100
    label = methods[item['method_key']]
    value_5m = float(main_by_id[item['config_id']][f'{label}_top1'])
    assert abs(values[2] - value_5m) < 0.0051, item['row_id']
assert all(row['formal_status'] != 'PENDING' for row in main)
assert sum(row['formal_status'] == 'AVAILABLE_WITH_PROTOCOL_CAVEAT' for row in main) == 1

write_csv('RESULTS_SOURCE.csv', main)
write_csv('SCAR_MULTITHRESHOLD_LONG.csv', long)
for metric, name in [('top1', 'TOP1_5M.csv'), ('cell', 'CELL_R1.csv')]:
    columns = ('Original', 'Presence', 'Rank Fusion', 'FSR', 'Historical A', 'SCAR')
    write_csv(name, [
        {
            'method': row['method'], 'dataset': row['dataset'], 'split': row['split'],
            **{label: row[f'{label}_{metric}'] for label in columns},
            'status': row['formal_status'], 'notes': row['notes'],
        }
        for row in main
    ])

hashes.update({str(path): digest(path) for path in inputs})
(output / 'SOURCE_HASHES.json').write_text(
    json.dumps(hashes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
)
print('25/25 configurations; 125/125 method rows; 500/500 threshold values.')
print('125/125 Top-1@5m matches across tables within display precision.')


"""Grouped outer cross-validation with a separate inner validation split. No OCR rerun."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

from prepare_production import prepare


def partitions(groups, folds, seed):
    groups = sorted(set(groups))
    if folds < 2 or len(groups) < folds * 3:
        raise ValueError('Need at least two folds and three groups per fold')
    random.Random(seed).shuffle(groups)
    outer = {g: i % folds for i, g in enumerate(groups)}
    result = []
    for fold in range(folds):
        remaining = [g for g in groups if outer[g] != fold]
        random.Random(seed + fold + 1).shuffle(remaining)
        validation = set(remaining[:max(1, round(len(remaining) * .1))])
        result.append({g: 'test' if outer[g] == fold else 'validation' if g in validation else 'train' for g in groups})
    return result


def select_clean(rows, assignments, kind, split):
    selected = [dict(r, split=split) for r in rows if r['kind'] == kind and assignments[r['group']] == split]
    targets = defaultdict(set)
    for r in selected:
        targets[r['raw']].add(r['clean'])
    seen, output = set(), []
    for r in selected:
        # Only training supervision can be filtered using training conflicts.
        if split == 'train' and len(targets[r['raw']]) > 1:
            continue
        key = r['raw'], r['clean']
        if key not in seen:
            if len(r['raw']) > 512:
                raise ValueError('Cleaner input exceeds the configured length')
            seen.add(key)
            output.append(r)
    if not output:
        raise ValueError(f'Empty {kind}/{split}')
    return output


def materialize(data, output, folds, seed):
    original = json.loads((data / 'prepared/splits.json').read_text())
    allowed = {key: group for key, (split, group) in original.items() if split in ('train', 'validation')}
    excluded = {group for split, group in original.values() if split in ('test', 'llm_holdout')}
    assert not set(allowed.values()) & excluded
    assignments = partitions(allowed.values(), folds, seed)
    docs = [d for d in json.loads((data / 'prepared/layout.json').read_text()) if d['id'] in allowed]
    rows = []
    with gzip.open(data / 'records.jsonl.gz', 'rt') as source:
        next(source)
        for line in source:
            record = json.loads(line)
            if record['_id'] not in allowed:
                continue
            _, clean, _ = prepare(record)
            rows.extend(dict(r, file=record['_id'], group=allowed[record['_id']]) for r in clean)
    summary = {'folds': folds, 'seed': seed, 'development_records': len(allowed),
               'development_groups': len(set(allowed.values())), 'layout_documents': len(docs),
               'original_test_and_llm_holdout_excluded': True, 'inner_validation_fraction': .1,
               'source_sha256': hashlib.sha256((data / 'records.jsonl.gz').read_bytes()).hexdigest(), 'counts': []}
    for fold, assignment in enumerate(assignments):
        root = output / f'fold-{fold}'
        (root / 'prepared').mkdir(parents=True, exist_ok=True)
        (root / 'records.jsonl.gz').symlink_to(data / 'records.jsonl.gz')
        (root / 'assignments.json').write_text(json.dumps(assignment))
        rewritten, counts = [], Counter()
        for doc in docs:
            split = assignment[doc['group']]
            rewritten.append(dict(doc, split=split))
            dest = root / 'layout' / split
            dest.mkdir(parents=True, exist_ok=True)
            for suffix in ('.jpeg', '-ocr.json', '-label.json'):
                source = data / 'layout' / doc['split'] / (doc['id'] + suffix)
                assert source.is_file(), source
                (dest / source.name).symlink_to(source)
            counts['layout_' + split] += 1
        assert all(counts['layout_' + split] for split in ('train', 'validation', 'test'))
        (root / 'prepared/layout.json').write_text(json.dumps(rewritten))
        for kind in ('track', 'sku', 'recipient'):
            dest = root / 'clean' / kind
            dest.mkdir(parents=True, exist_ok=True)
            for split in ('train', 'validation', 'test'):
                selected = select_clean(rows, assignment, kind, split)
                with (dest / (split + '.csv')).open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['file', 'group', 'split', 'kind', 'raw', 'clean', 'ocr_indices'])
                    writer.writeheader()
                    writer.writerows(selected)
                counts[kind + '_' + split] = len(selected)
        summary['counts'].append(dict(counts))
    (output / 'data-audit.json').write_text(json.dumps(summary, indent=2))
    return summary


def run(command, cwd, log):
    print('RUN', log, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1')
    with log.open('w') as stream:
        subprocess.run([str(c) for c in command], cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def aggregate(output, folds):
    summaries = [json.loads((output / f'fold-{i}/evaluation/summary.json').read_text()) for i in range(folds)]
    result = {'folds': folds, 'scope': 'Grouped development-set CV on fully OCR-aligned weak labels; excludes original temporal test and LLM holdout', 'pipeline': {}, 'conditional_cleaning': {}}
    for mode in ('production', 'old_layout_new_clean', 'new_pipeline'):
        result['pipeline'][mode] = {}
        for field in ('track', 'recipient', 'skus', 'items', 'all_fields'):
            rates = [s['metrics'][mode][field] / s['documents'] for s in summaries]
            correct = sum(s['metrics'][mode][field] for s in summaries)
            count = sum(s['documents'] for s in summaries)
            result['pipeline'][mode][field] = {'correct': correct, 'count': count, 'pooled_rate': correct / count,
                'fold_rates': rates, 'fold_mean': statistics.mean(rates), 'fold_std': statistics.stdev(rates)}
    cleaners = [json.loads((output / f'fold-{i}/models/clean/evaluation.json').read_text()) for i in range(folds)]
    for kind in ('track', 'sku', 'recipient'):
        values = [c[kind]['test'] for c in cleaners]
        rates = [v['exact'] / v['count'] for v in values]
        result['conditional_cleaning'][kind] = {key: sum(v[key] for v in values) for key in ('count', 'exact', 'identity_baseline_exact', 'unseen_raw_count', 'unseen_raw_exact')}
        result['conditional_cleaning'][kind].update(fold_rates=rates, fold_mean=statistics.mean(rates), fold_std=statistics.stdev(rates))
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--layout-python', type=Path, required=True)
    parser.add_argument('--clean-python', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    repo = Path(__file__).resolve().parents[1]
    clean_repo = repo / 'packages/clean/ngn_fba_clean'
    layout_python, clean_python = args.layout_python.resolve(), args.clean_python.resolve()
    audit = materialize(data, output, args.folds, args.seed)
    print(json.dumps(audit, indent=2), flush=True)
    for fold in range(args.folds):
        root = output / f'fold-{fold}'
        logs = root / 'logs'; logs.mkdir()
        clean_run = root / 'models/clean'
        for kind in ('track', 'sku', 'recipient'):
            run([clean_python, 'train.py', '--data', root / f'clean/{kind}/train.csv', '--val_data', root / f'clean/{kind}/validation.csv',
                 '--file_col', 'group', '--outdir', clean_run / kind, '--epochs', 20, '--batch_size', 128, '--max_len', 512,
                 '--accelerator', 'gpu', '--devices', 1, '--min_cov_clean', 1, '--seed', 42 + fold], clean_repo, logs / f'clean-{kind}.log')
        run([clean_python, 'evaluate.py', '--run', clean_run, '--data', root / 'clean', '--splits', 'validation', 'test'], clean_repo, logs / 'clean-evaluate.log')
        run([layout_python, repo / 'packages/layout/ngn_fba_layout/train.py', '--data_dir', root / 'layout/train', '--val_dir', root / 'layout/validation',
             '--model_name', args.base_model.resolve(), '--output_dir', root / 'models/layout', '--epochs', 5,
             '--train_bs', 4, '--eval_bs', 4, '--seed', 42 + fold], repo, logs / 'layout.log')
        run([clean_python, repo / 'training/evaluate_pipeline.py', '--data', root, '--model', root / 'models/layout', '--clean-run', clean_run,
             '--clean-code', clean_repo / 'runtime.py', '--postprocess', data / 'legacy_postprocess.py',
             '--output', root / 'evaluation', '--split', 'test'], repo, logs / 'pipeline-evaluate.log')
        print('FOLD_COMPLETE', fold, flush=True)
    print(json.dumps(aggregate(output, args.folds), indent=2), flush=True)


if __name__ == '__main__':
    main()

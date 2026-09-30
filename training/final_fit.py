"""Fit deployment candidates on development data using inner-CV epoch counts only."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from cross_validate import run, select_clean
from prepare_production import prepare


def selected_epochs(cv):
    choices = {kind: [] for kind in ('layout', 'track', 'sku', 'recipient')}
    for fold in range(5):
        models = cv / f'fold-{fold}/models'
        states = [json.loads(p.read_text()) for p in (models / 'layout').glob('checkpoint-*/trainer_state.json')]
        state = max(states, key=lambda s: s['global_step'])
        step = int(state['best_model_checkpoint'].rsplit('-', 1)[1])
        epoch = next(h['epoch'] for h in state['log_history'] if h.get('step') == step and 'eval_loss' in h)
        if not float(epoch).is_integer() or epoch <= 0:
            raise ValueError('Expected positive, complete training epochs')
        choices['layout'].append(int(epoch))
        for kind in ('track', 'sku', 'recipient'):
            checkpoints = list((models / f'clean/{kind}/checkpoints').glob('best-*.ckpt'))
            if len(checkpoints) != 1:
                raise ValueError(f'Expected one inner-validation checkpoint: {kind}/{fold}')
            choices[kind].append(int(checkpoints[0].stem.split('-')[1]) + 1)
    return choices, {kind: int(statistics.median(values)) for kind, values in choices.items()}


def materialize(data, output):
    splits = json.loads((data / 'prepared/splits.json').read_text())
    allowed = {key: group for key, (split, group) in splits.items() if split in ('train', 'validation')}
    excluded = {group for split, group in splits.values() if split in ('test', 'llm_holdout')}
    if set(allowed.values()) & excluded:
        raise ValueError('Development and held-out groups overlap')
    layout = output / 'layout'
    layout.mkdir()
    count = 0
    for doc in json.loads((data / 'prepared/layout.json').read_text()):
        if doc['id'] not in allowed:
            continue
        if doc['split'] not in ('train', 'validation') or doc['group'] != allowed[doc['id']]:
            raise ValueError('Layout assignment disagrees with frozen split')
        for suffix in ('.jpeg', '-ocr.json', '-label.json'):
            source = data / 'layout' / doc['split'] / (doc['id'] + suffix)
            if not source.is_file():
                raise FileNotFoundError(source)
            (layout / source.name).symlink_to(source)
        count += 1
    rows = []
    with gzip.open(data / 'records.jsonl.gz', 'rt') as stream:
        next(stream)
        for line in stream:
            record = json.loads(line)
            if record['_id'] in allowed:
                _, clean, _ = prepare(record)
                rows.extend(dict(r, file=record['_id'], group=allowed[record['_id']]) for r in clean)
    assignments = dict.fromkeys(allowed.values(), 'train')
    counts = {}
    for kind in ('track', 'sku', 'recipient'):
        selected = select_clean(rows, assignments, kind, 'train')
        with (output / f'{kind}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['file', 'group', 'split', 'kind', 'raw', 'clean', 'ocr_indices'])
            writer.writeheader()
            writer.writerows(selected)
        counts[kind] = len(selected)
    return {'development_records': len(allowed), 'development_groups': len(assignments),
            'layout_documents': count, 'clean_rows': counts, 'held_out_groups_excluded': len(excluded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layout-python', type=Path, required=True)
    parser.add_argument('--clean-python', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    choices, epochs = selected_epochs(data / 'cross-validation-v1')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    audit = materialize(data, output)
    repo = Path(__file__).resolve().parents[1]
    layout_script = repo / 'packages/layout/ngn_fba_layout/train.py'
    clean_script = repo / 'packages/clean/ngn_fba_clean/train.py'
    audit.update(inner_best_epochs=choices, final_epochs=epochs, seed=42,
                 selection='Median completed epoch count selected by inner validation; outer scores unused',
                 source_sha256=hashlib.sha256((data / 'records.jsonl.gz').read_bytes()).hexdigest(),
                 code_sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), layout_script, clean_script)})
    (output / 'training-plan.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    for kind in ('track', 'sku', 'recipient'):
        run([args.clean_python, clean_script, '--data', output / f'{kind}.csv', '--train_only',
             '--file_col', 'group', '--outdir', output / f'models/clean/{kind}', '--epochs', epochs[kind],
             '--batch_size', 128, '--max_len', 512, '--accelerator', 'gpu', '--devices', 1,
             '--min_cov_clean', 1, '--seed', 42], repo, output / f'clean-{kind}.log')
    run([args.layout_python, layout_script, '--data_dir', output / 'layout', '--train_only',
         '--model_name', args.base_model, '--output_dir', output / 'models/layout',
         '--epochs', epochs['layout'], '--train_bs', 4, '--eval_bs', 4, '--seed', 42],
        repo, output / 'layout.log')
    (output / 'TRAINING_COMPLETE.json').write_text(json.dumps({'epochs': epochs}))


if __name__ == '__main__':
    main()

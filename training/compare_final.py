"""Compare the final fit and recovery release on the same frozen temporal holdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    args = parser.parse_args()
    data, fit = args.data.resolve(), args.fit.resolve()
    if not (fit / 'TRAINING_COMPLETE.json').is_file():
        raise RuntimeError('Final fit has not finished')
    root = Path(__file__).resolve().parents[1]
    evaluation = fit / 'comparison'
    evaluation.mkdir(exist_ok=False)
    candidates = {
        'recovery': (data / 'recovered-weights-v1/layoutlmv3', data / 'recovered-weights-v1/clean'),
        'final_fit': (fit / 'models/layout', fit / 'models/clean'),
    }
    cases = {}
    summaries = {}
    for name, (layout, clean) in candidates.items():
        command = [sys.executable, str(root / 'training/evaluate_pipeline.py'), '--data', str(data),
                   '--model', str(layout), '--clean-run', str(clean),
                   '--clean-code', str(root / 'packages/clean/ngn_fba_clean/runtime.py'),
                   '--postprocess', str(data / 'legacy_postprocess.py'),
                   '--output', str(evaluation / name), '--split', 'test']
        with (evaluation / f'{name}.log').open('w') as log:
            subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads((evaluation / name / 'cases.json').read_text())
        cases[name] = {row['id']: row for row in rows}
        if len(cases[name]) != len(rows):
            raise ValueError('Duplicate evaluation documents')
        summaries[name] = json.loads((evaluation / name / 'summary.json').read_text())
    if not cases['recovery'] or cases['recovery'].keys() != cases['final_fit'].keys():
        raise ValueError('Candidates were not evaluated on identical nonempty document sets')
    result = {'documents': len(cases['recovery']), 'split': 'original temporal test',
              'scope': 'Fully OCR-aligned business-review records; saved OCR reused, no OCR rerun. '
                       'This holdout has been inspected before; it is not a fresh prospective test.',
              'training_plan': json.loads((fit / 'training-plan.json').read_text()),
              'metrics': {name: summary['metrics']['new_pipeline'] for name, summary in summaries.items()},
              'historical_production': summaries['recovery']['metrics']['production']}
    pairs = [(all(cases['recovery'][key]['new_pipeline'].values()),
              all(cases['final_fit'][key]['new_pipeline'].values())) for key in cases['recovery']]
    result['paired_all_fields'] = {'both_correct': sum(old and new for old, new in pairs),
                                  'both_wrong': sum(not old and not new for old, new in pairs),
                                  'final_only_correct': sum(not old and new for old, new in pairs),
                                  'recovery_only_correct': sum(old and not new for old, new in pairs)}
    result['median_inference_seconds_without_ocr'] = {
        name: statistics.median(c['new_seconds_without_ocr'] for c in rows.values())
        for name, rows in cases.items()}
    manifest = {str(path.relative_to(fit)): hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                for path in sorted((fit / 'models').rglob('*')) if path.is_file()}
    (fit / 'weights-manifest.json').write_text(json.dumps(manifest, indent=2))
    (evaluation / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'training_plan'}, indent=2))


if __name__ == '__main__':
    main()

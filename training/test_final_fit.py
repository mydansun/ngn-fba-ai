"""Run with python training/test_final_fit.py; no model dependencies required."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from final_fit import materialize, selected_epochs


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    for fold, epoch in enumerate((4, 4, 2, 3, 5)):
        models = root / f'fold-{fold}/models'
        state = models / 'layout/checkpoint-500'
        state.mkdir(parents=True)
        (state / 'trainer_state.json').write_text(json.dumps({
            'global_step': 500, 'best_model_checkpoint': f'checkpoint-{epoch * 100}',
            'log_history': [{'epoch': epoch, 'step': epoch * 100, 'eval_loss': 0.1}],
        }))
        for kind in ('track', 'sku', 'recipient'):
            checkpoint = models / f'clean/{kind}/checkpoints'
            checkpoint.mkdir(parents=True)
            (checkpoint / f'best-{fold:02d}.ckpt').touch()
        # Outer evaluation is deliberately unreadable: it must not affect selection.
        (root / f'fold-{fold}/evaluation').mkdir()
        (root / f'fold-{fold}/evaluation/summary.json').write_text('not json')
    choices, epochs = selected_epochs(root)
    assert choices['layout'] == [4, 4, 2, 3, 5]
    assert epochs == {'layout': 4, 'track': 3, 'sku': 3, 'recipient': 3}
    (root / 'prepared').mkdir()
    (root / 'prepared/splits.json').write_text(json.dumps({
        'train': ['train', 'shared'], 'held-out': ['test', 'shared'],
    }))
    try:
        materialize(root, root / 'output')
    except ValueError as error:
        assert 'overlap' in str(error)
    else:
        raise AssertionError('Held-out group leakage must fail before materialization')
print('Final-fit selection and leakage checks passed')

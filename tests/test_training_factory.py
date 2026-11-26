import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.factory import ROOT, board, read_rows, require_completed, training_command


class TestTrainingFactory(unittest.TestCase):
    def test_training_command_is_bounded_and_keeps_early_checkpoint(self):
        cmd = training_command('python', Path('snapshot'), Path('run'), steps=100, seconds=900)
        for flag, value in {'--max-steps': '100', '--max-seconds': '860',
                            '--checkpoint-steps': '25', '--batch': '2',
                            '--grad-accum': '2'}.items():
            self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_default_data_keeps_the_v1_command_and_data_flag_is_passed(self):
        cmd = training_command('python', Path('snapshot'), Path('run'), steps=100, seconds=900)
        self.assertNotIn('--data', cmd)  # v1 pilot runs are reproduced unchanged
        cmd = training_command('python', Path('snapshot'), Path('run'), steps=100, seconds=900,
                               data=Path('training/train_verified_v1.jsonl'))
        self.assertEqual(cmd[cmd.index('--data') + 1], 'training/train_verified_v1.jsonl')
        self.assertNotIn('--seed', cmd)
        cmd = training_command('python', Path('snapshot'), Path('run'), steps=100, seconds=900, seed=3)
        self.assertEqual(cmd[cmd.index('--seed') + 1], '3')

    def test_fit_roster_samples_the_given_dataset(self):
        from training.evaluate_pilot import fit_roster
        v1 = fit_roster()
        verified = fit_roster(Path('training/train_verified_v1.jsonl'))
        self.assertEqual(len(v1), 36)
        self.assertNotEqual([r['prompt'] for r in v1], [r['prompt'] for r in verified])

    def test_incomplete_run_cannot_advance_to_export(self):
        with TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError, 'completed adapter'):
                require_completed(Path(path), 100)

    def test_board_rejects_a_missing_prediction(self):
        with TemporaryDirectory() as path:
            run = Path(path)
            adapter = run / 'adapter'
            adapter.mkdir()
            (adapter / 'run.json').write_text(json.dumps({
                'optimizer_steps': 100, 'max_steps': 100, 'stop_reason': 'completed', 'seed': 42,
            }))
            (adapter / 'adapter_model.safetensors').write_bytes(b'test artifact')
            (run / 'comparison.json').write_text('{}')
            cases = [r for p in (ROOT / 'eval').glob('eval_*.jsonl') for r in read_rows(p)]
            rows = [dict(id=r['id'], scorer='intent_v2', task_pass=True) for r in cases]
            for label, data in [('stock', rows), ('final', rows[:-1])]:
                (run / f'{label}.graded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in data))
            with self.assertRaisesRegex(RuntimeError, 'final: incomplete'):
                board(run)

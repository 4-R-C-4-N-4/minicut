"""Run a bounded training/export/development comparison, or verify its board entry."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DATA = ROOT / 'training/train_pilot_v1.jsonl'


def training_command(python, base, run, *, steps, seconds, data=None, seed=None):
    cmd = [str(python), '-m', 'training.train_lora', '--base', str(base),
           '--out', str(run / 'adapter'), '--max-steps', str(steps),
           '--max-seconds', str(seconds - 40), '--checkpoint-steps', str(steps // 4),
           '--batch', '2', '--grad-accum', '2']
    if data is not None:
        cmd += ['--data', str(data)]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    return cmd


def require_completed(run, steps):
    path = run / 'adapter'
    manifest = path / 'run.json'
    if not manifest.is_file() or not (path / 'adapter_model.safetensors').is_file():
        raise RuntimeError(f'{run}: no completed adapter')
    record = json.loads(manifest.read_text())
    if record.get('optimizer_steps') != steps or record.get('stop_reason') != 'completed':
        raise RuntimeError(f'{run}: optimizer step budget was not completed')
    return record


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def board(run):
    report = json.loads((run / 'comparison.json').read_text())
    train = json.loads((run / 'adapter' / 'run.json').read_text())
    require_completed(run, train['max_steps'])
    ids = {r['id'] for path in (ROOT / 'eval').glob('eval_*.jsonl') for r in read_rows(path)}
    if len(ids) != 200:
        raise RuntimeError('expected the 200-case development roster')
    arms = []
    for name in ('stock', 'final'):
        rows = read_rows(run / f'{name}.graded.jsonl')
        if len(rows) != len(ids) or {r['id'] for r in rows} != ids or any(
                r['scorer'] != 'intent_v2' or type(r['task_pass']) is not bool for r in rows):
            raise RuntimeError(f'{name}: incomplete or incompatible evaluation')
        arms.append({r['id']: r for r in rows})
    stock, final = arms
    gains = {i for i in ids if final[i]['task_pass'] and not stock[i]['task_pass']}
    losses = {i for i in ids if stock[i]['task_pass'] and not final[i]['task_pass']}
    summary = report['all']
    if (summary['n'] != len(ids) or set(summary['gains']) != gains or
            set(summary['regressions']) != losses or
            summary['stock_pass'] != sum(r['task_pass'] for r in stock.values()) or
            summary['final_pass'] != sum(r['task_pass'] for r in final.values())):
        raise RuntimeError('comparison disagrees with per-case results')
    keep = {i for i in ids if stock[i]['expected']['tool'] in {'trim', 'cut_out'}}
    inversions = [{i for i in keep if {rows[i]['expected']['tool'],
                   rows[i].get('predicted', {}).get('tool')} == {'trim', 'cut_out'}}
                  for rows in arms]
    keep_counts = [sum(rows[i]['task_pass'] for i in keep) for rows in arms]
    gate = len(gains) > len(losses) and keep_counts[1] > keep_counts[0] and not (
        inversions[1] - inversions[0])
    if report['gate_pass'] != gate:
        raise RuntimeError('comparison gate disagrees with per-case results')
    return {'run': run.name, 'steps': train['optimizer_steps'], 'seed': train['seed'],
            'n': len(ids), 'stock': summary['stock_pass'], 'tuned': summary['final_pass'],
            'gains': len(gains), 'regressions': len(losses), 'gate_pass': gate,
            'keep_remove': {'n': len(keep), 'stock': keep_counts[0], 'tuned': keep_counts[1]},
            'new_inversions': sorted(inversions[1] - inversions[0])}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run_stage(run, name, command, env):
    print(name, flush=True)
    with (run / f'{name}.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def pilot(args):
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', args.gpu) or args.steps < 4 or args.seconds <= 40:
        raise ValueError('provide a GPU UUID, at least four steps, and more than 40 seconds')
    server = args.llama / 'build/bin/llama-server'
    converter = args.llama / 'convert_lora_to_gguf.py'
    inputs = [server, converter, args.gguf, args.base / 'config.json', args.base / 'tokenizer.json',
              ROOT / 'training/train_lora.py', ROOT / 'training/evaluate_pilot.py',
              args.data, ROOT / 'eval/run_eval.py',
              ROOT / 'minicut/llm/prompts.py', ROOT / 'minicut/grammar/minicut.gbnf']
    inputs += sorted((ROOT / 'eval').glob('eval_*.jsonl'))
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    env = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': args.gpu,
           'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'}
    lock_dir = ROOT / 'training/output'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = subprocess.check_output([str(server), '--list-devices'], env=env, text=True)
        if devices.count('CUDA') != 1 or 'RTX 3090' not in devices:
            raise RuntimeError(f'expected one RTX 3090: {devices}')
        run = args.run
        run.mkdir(parents=True, exist_ok=False)
        hashes = {str(path): digest(path) for path in inputs}
        (run / 'factory-inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
        data = None if args.data == DEFAULT_DATA else args.data
        train = training_command(sys.executable, args.base, run, steps=args.steps, seconds=args.seconds,
                                 data=data, seed=args.seed)
        run_stage(run, 'train', ['timeout', str(args.seconds), *train], env)
        require_completed(run, args.steps)
        export_env = {**env, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(args.llama / 'gguf-py')}
        for label, path in [('early', run / f'adapter/checkpoint-{args.steps // 4}'),
                            ('final', run / 'adapter')]:
            run_stage(run, f'{label}.convert', [sys.executable, str(converter), '--base', str(args.base),
                      '--outfile', str(run / f'{label}.gguf'), str(path)], export_env)
        before = {**hashes, **{str(p): digest(p) for p in (run / 'early.gguf', run / 'final.gguf')}}
        run_stage(run, 'comparison', [sys.executable, '-m', 'training.evaluate_pilot', '--run', str(run),
                  '--gguf', str(args.gguf), '--server', str(server), '--data', str(args.data)], env)
        if any(digest(Path(p)) != h for p, h in before.items()):
            raise RuntimeError('factory inputs changed during the run')
        (run / 'factory-fingerprints.json').write_text(json.dumps(before, indent=2) + '\n')
        print(json.dumps(board(run), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('pilot')
    for flag in ('run', 'base', 'gguf', 'llama'):
        train.add_argument(f'--{flag}', type=lambda path: Path(path).resolve(), required=True)
    train.add_argument('--gpu', required=True)
    train.add_argument('--steps', type=int, default=100)
    train.add_argument('--seconds', type=int, default=900)
    train.add_argument('--data', type=lambda path: Path(path).resolve(), default=DEFAULT_DATA,
                       help='training rows (default: the v1 pilot set)')
    train.add_argument('--seed', type=int, help='training seed (default: train_lora\'s 42)')
    view = commands.add_parser('board')
    view.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pilot':
        pilot(args)
    else:
        print(json.dumps(board(args.run), indent=2))


if __name__ == '__main__':
    main()

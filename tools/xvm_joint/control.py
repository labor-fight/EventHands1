"""Freeze and launch the isolated XJ run; never modifies existing experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def freeze(config_path, destination):
    config_path, destination = Path(config_path).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads(config_path.read_text())
    protocol = Path(config['protocol']).resolve()
    run = config_path.parent
    files = {}
    # Existing modules are read-only. Include their actual executable versions,
    # not another experiment's historical source-freeze or result pointers.
    for folder in ('model', 'semkine', 'tools/s43_br', 'tools/s43_integration', 'tools/xvm_joint'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            files[str(path.relative_to(ROOT))] = sha(path)
    for path in sorted((ROOT / 'tests').glob('test_xvm_joint_*.py')):
        files[str(path.relative_to(ROOT))] = sha(path)
    inputs = {}
    for key in ('base_checkpoint', 'base_config'):
        path = Path(config[key]).resolve()
        inputs[key] = {'path': str(path), 'sha256': sha(path)}
    split = Path(config['data_root']) / 'splits_semkine.json'
    inputs['split'] = {'path': str(split), 'sha256': sha(split)}
    receipts = {}
    for path in sorted((run / 'engineering').glob('*cpu.json')):
        receipts[str(path)] = sha(path)
    dump(destination, {
        'schema': 'xvm_joint_actual_sourcefreeze_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {'path': str(protocol), 'sha256': sha(protocol)},
        'config': {'path': str(config_path), 'sha256': sha(config_path)},
        'files': files, 'inputs': inputs, 'engineering': receipts,
        'scope': 'Immutable code/config/protocol/checkpoint/split; actual event bytes are bound by input trace tokens.',
        'real_engineering_required_before_training': True,
    })
    return verify(destination, config_path)


def verify(freeze_path, config_path=None):
    path = Path(freeze_path).resolve()
    frozen = json.loads(path.read_text())
    if frozen['schema'] != 'xvm_joint_actual_sourcefreeze_v1':
        raise ValueError('Unexpected source-freeze schema')
    for key in ('protocol', 'config'):
        entry = frozen[key]
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Changed ' + key)
    if config_path is not None and Path(config_path).resolve() != Path(frozen['config']['path']):
        raise RuntimeError('Different training config')
    for relative, digest in frozen['files'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Changed executable: ' + relative)
    for key, entry in frozen['inputs'].items():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Changed frozen input: ' + key)
    for receipt, digest in frozen['engineering'].items():
        if sha(receipt) != digest:
            raise RuntimeError('Changed CPU engineering receipt')
    return {'status': 'PASS', 'sourcefreeze_sha256': sha(path),
            'files_verified': len(frozen['files']), 'inputs_verified': len(frozen['inputs'])}


def resources(index=2):
    index = int(index)
    if index not in range(2, 8):
        raise ValueError('Only physical GPU2-7 are authorized')
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used', '--format=csv,noheader,nounits']
    rows = subprocess.check_output(command, text=True).strip().splitlines()
    selected = None
    for row in rows:
        fields = [v.strip() for v in row.split(',')]
        if int(fields[0]) == index:
            selected = {'physical_gpu': index, 'uuid': fields[1],
                        'free_mib': int(fields[2]), 'used_mib': int(fields[3])}
    if selected is None:
        raise RuntimeError('Authorized physical device was not found')
    # This first run uses an unoccupied device, never an old-job latency window.
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    if any(line.split(',')[0].strip() == selected['uuid'] for line in apps):
        raise RuntimeError('Selected GPU has another compute process; queue instead of sharing')
    if selected['free_mib'] < (20 + 16) * 1024:
        raise RuntimeError('Insufficient free memory for the 16GiB process cap and 20GiB reserve')
    return selected


def launch(config_path, freeze_path, *, engineering=False, device='cuda:0'):
    config_path, freeze_path = Path(config_path).resolve(), Path(freeze_path).resolve()
    checked = verify(freeze_path, config_path)
    config = json.loads(config_path.read_text())
    run = config_path.parent
    resource = resources(config['physical_gpu'])
    if not engineering:
        receipt_path = run / 'engineering/real_gpu.json'
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') != 'PASS' or receipt.get('sourcefreeze_sha256') != checked['sourcefreeze_sha256']:
            raise RuntimeError('Actual GPU forward/backward engineering must pass for this exact freeze')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=resource['uuid'], OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
               XJ_GPU_MEMORY_LIMIT_GIB='16')
    mode = 'engineering' if engineering else 'train'
    command = [str(PYTHON), '-m', 'tools.xvm_joint.train', mode,
               '--config', str(config_path), '--source-freeze', str(freeze_path), '--device', device]
    log_path = run / 'logs' / (mode + '.log')
    log_path.parent.mkdir(exist_ok=True)
    if not engineering and (run / 'train/process.json').exists():
        raise RuntimeError('A training launch was already recorded; do not silently start another run')
    with log_path.open('xb') as log:
        process = subprocess.Popen(['nice', '-n', '10', *command], cwd=str(ROOT), env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(checked, command=command, pid=process.pid, log=str(log_path), resources=resource,
                  launched_utc=datetime.now(timezone.utc).isoformat(), mode=mode)
    dump(run / ('engineering/process.json' if engineering else 'train/process.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('verify'); p.add_argument('--source-freeze', required=True); p.add_argument('--config')
    p = sub.add_parser('launch'); p.add_argument('--config', required=True); p.add_argument('--source-freeze', required=True); p.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze': result = freeze(args.config, args.out)
    elif args.command == 'verify': result = verify(args.source_freeze, args.config)
    else: result = launch(args.config, args.source_freeze, engineering=args.engineering)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

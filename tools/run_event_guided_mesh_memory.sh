#!/usr/bin/env bash
# Train the user-confirmed 778-vertex local-memory tracker on seeds 3407 and 3408.
# Each seed uses two GPUs and the unchanged effective batch/schedule in its config.
# After successful training, select the fixed checkpoint grid by stateful recursive
# tracking on val_core. This launcher does not generate or report a performance row.
#
#   nohup bash tools/run_event_guided_mesh_memory.sh > logs/egm_memory_launcher.log 2>&1 &
#   EGM_PAIR_A=4,5 EGM_PAIR_B=6,7 bash tools/run_event_guided_mesh_memory.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EGM_PYTHON=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
EGM_CONFIG=configs/semkine/event_guided_mesh_memory_s3407.yaml
EGM_ARM=event_guided_mesh_memory
EGM_PAIR_A=${EGM_PAIR_A:-0,1}
EGM_PAIR_B=${EGM_PAIR_B:-2,3}
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p logs outputs/semkine

# Hold the lock through both seed processes and selection to prevent duplicate jobs
# during the interval before Lightning has written training_metadata.json.
exec 200>outputs/semkine/.event_guided_mesh_memory.launch.lock
if ! flock -n 200; then
  echo "An event-guided mesh memory launcher is already active." >&2
  exit 2
fi

for EGM_SEED in 3407 3408; do
  EGM_RUN_DIR=outputs/semkine/${EGM_ARM}_s${EGM_SEED}
  if [ -e "${EGM_RUN_DIR}/training_metadata.json" ] || compgen -G "${EGM_RUN_DIR}/*.ckpt" > /dev/null; then
    echo "Existing training in ${EGM_RUN_DIR}; use an explicit resume command." >&2
    exit 2
  fi
done

prepare_run() {
  local seed=$1
  local run_name=${EGM_ARM}_s${seed}
  local run_dir=outputs/semkine/${run_name}
  "$EGM_PYTHON" - "$EGM_CONFIG" "$seed" "$run_name" "$run_dir" <<'PY'
import hashlib
import pathlib
import sys
import tarfile
import yaml

source, seed, name, directory = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path(source).read_text())
assert cfg['MODEL']['EGM_MEMORY'] and cfg['TRACK']['UNROLL_PAIR']
assert int(cfg['TRAIN']['DEVICES']) == 2
cfg['SEED'] = int(seed)
cfg['TRAIN']['RUN_NAME'] = name
cfg['TRAIN']['OUTPUT_DIR'] = directory
cfg['EVAL']['OUTPUT_DIR'] = str(pathlib.Path(directory) / 'eval')
target = pathlib.Path(directory)
target.mkdir(parents=True, exist_ok=True)
frozen_config = target / 'config.yaml'
frozen_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

# Source, pre-registration and the frozen config; no datasets, weights or caches.
sources = sorted(set(pathlib.Path('model').rglob('*.py'))
                 | set(pathlib.Path('semkine').rglob('*.py'))
                 | {pathlib.Path('tools/select_checkpoint.py'),
                    pathlib.Path('tools/run_event_guided_mesh_memory.sh'),
                    pathlib.Path('docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md')})
with tarfile.open(target / 'source_snapshot.tar.gz', 'w:gz') as archive:
    for path in sources:
        archive.add(path, arcname=str(path), recursive=False)
    archive.add(frozen_config, arcname='config.yaml', recursive=False)
(target / 'source_sha256.txt').write_text(''.join(
    f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n'
    for path in sources + [frozen_config]))
PY
}

train_select() {
  local seed=$1 gpu_pair=$2 master_port=$3
  local run_name=${EGM_ARM}_s${seed}
  local run_dir=outputs/semkine/${run_name}
  local rc
  echo "[$(date --iso-8601=seconds)] training ${run_name} on GPUs ${gpu_pair}"
  if CUDA_VISIBLE_DEVICES=$gpu_pair MASTER_PORT=$master_port "$EGM_PYTHON" semkine/train.py \
      --config "${run_dir}/config.yaml" > "logs/${run_name}.log" 2>&1; then
    :
  else
    rc=$?
    echo "[$(date --iso-8601=seconds)] ${run_name}: training failed rc=${rc}; selection skipped" >&2
    return "$rc"
  fi

  # A zero exit code alone must not select an accidentally incomplete grid.
  if "$EGM_PYTHON" - "$run_dir" <<'PY'
import pathlib
import re
import sys
import yaml

run = pathlib.Path(sys.argv[1])
train = yaml.safe_load((run / 'config.yaml').read_text())['TRAIN']
every = int(train['SAVE_EVERY_N_STEPS'])
maximum = int(train['MAX_STEPS'])
expected = set(range(every, maximum + 1, every))
found = {int(m.group(1)) for path in run.glob('*step=*.ckpt')
         if (m := re.search(r'step=(\d+)', path.name))}
missing = sorted(expected - found)
if not expected or missing:
    raise SystemExit(f'Incomplete fixed checkpoint grid in {run}: missing {missing}')
PY
  then
    :
  else
    rc=$?
    echo "[$(date --iso-8601=seconds)] ${run_name}: grid check failed; selection skipped" >&2
    return "$rc"
  fi

  echo "[$(date --iso-8601=seconds)] ${run_name}: selecting stateful recursive fixed grid"
  if CUDA_VISIBLE_DEVICES=${gpu_pair%%,*} "$EGM_PYTHON" tools/select_checkpoint.py \
      --run-dir "$run_dir" > "logs/${run_name}_select.log" 2>&1; then
    echo "[$(date --iso-8601=seconds)] selection complete ${run_name}"
  else
    rc=$?
    echo "[$(date --iso-8601=seconds)] ${run_name}: selection failed rc=${rc}" >&2
    return "$rc"
  fi
}

prepare_run 3407
prepare_run 3408
train_select 3407 "$EGM_PAIR_A" 29637 &
EGM_PID_A=$!
train_select 3408 "$EGM_PAIR_B" 29638 &
EGM_PID_B=$!
echo "seed 3407 supervisor PID ${EGM_PID_A}; seed 3408 supervisor PID ${EGM_PID_B}"
EGM_RC_A=0
EGM_RC_B=0
wait "$EGM_PID_A" || EGM_RC_A=$?
wait "$EGM_PID_B" || EGM_RC_B=$?
echo "[$(date --iso-8601=seconds)] return codes: seed 3407=${EGM_RC_A}, seed 3408=${EGM_RC_B}"
if [ "$EGM_RC_A" -ne 0 ] || [ "$EGM_RC_B" -ne 0 ]; then
  exit 1
fi
echo "Both training runs and checkpoint selections completed. No performance row was generated."

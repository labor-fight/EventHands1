import os
import signal
import time
from pathlib import Path
R = Path('/data1/lyq/code/mesh/EventHands1')
for pid in [3599918, 3599934]:
    path = Path(f'/proc/{pid}/cmdline')
    if path.exists():
        cmd = path.read_bytes().decode(errors='replace')
        assert 'u1a_untied_debug_s3407.yaml' in cmd, cmd
        os.kill(pid, signal.SIGTERM)
time.sleep(12)
for mode in ['shared', 'untied']:
    name = f'u1a_{mode}_debug_s3407'
    for path in [R/'outputs/u1a'/f'{name}.log', R/'outputs/u1a'/f'{name}.time', R/'outputs/semkine'/name]:
        if path.exists():
            dst = path.with_name(path.name+'.attempt1')
            assert not dst.exists()
            path.rename(dst)
print('Preserved initial debug attempt; terminated only verified stuck U1a debug processes.')

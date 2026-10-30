"""Three FIT7 packets through the final zgz entry points, without supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.s43_rel.access_guard import InferenceAccessGuard
from .data import NativeInputReader, PUBLIC_INITIAL_STATE
from .train import configure_device, dump, sha, tensor_digest
from .zgz_eval import make_model, official_evalx, verify_identity


def run(protocol_path, freeze_path, output, device='cuda:0'):
    protocol, _, config = verify_identity(protocol_path, freeze_path)
    device, device_info = configure_device(device, config, engineering=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    official = official_evalx()
    started = time.perf_counter()
    rows = []
    with InferenceAccessGuard() as reader_guard:
        reader = NativeInputReader('ch_global', protocol['data_root'])
    for mode in ('w50', 's39'):
        for arm in ('primary', 'global'):
            model, _, _ = make_model(protocol, arm, mode, device)
            before = tensor_digest(model.state_dict())
            previous = torch.from_numpy(PUBLIC_INITIAL_STATE.copy()).to(device)[None]
            model.reset_state(previous)
            for ordinal in range(3):
                cutoff = 50*(ordinal+1)
                window = 50 if mode == 'w50' else min(200, cutoff)
                with InferenceAccessGuard() as guard:
                    native = reader.packet(cutoff-window, cutoff)['events']
                    ev5 = np.zeros((len(native), 5), np.float32)
                    ev5[:, 1:3] = native[:, :2]
                    ev5[:, 3] = (native[:, 2]-(cutoff-window)*1000).astype(np.float32)*np.float32(1e-6)
                    ev5[:, 4] = native[:, 3]
                    packet = official.ET.make_eval_packet(ev5, previous, previous.new_zeros((1, 10)),
                            torch.from_numpy(reader.K_source.copy()).to(device)[None], window, device)
                    result = model.forward_packet(packet)
                assert result.shape == (1, 51) and bool(torch.isfinite(result).all())
                audit = dict(model.last_audit)
                assert audit['new_native_events'] == len(reader.packet(cutoff-50, cutoff)['events'])
                assert audit['cutoff_ms'] == cutoff
                assert audit['absolute_camera_invalid'] == 0 and audit['local_camera_invalid'] == 0
                receipt = guard.receipt()
                assert receipt['blocked_pose_meta_attempts'] == 0 and receipt['blocked_aux_beta_attempts'] == 0
                rows.append(dict(mode=mode, arm=arm, ordinal=ordinal, window_ms=window,
                                 prediction_digest=tensor_digest({'prediction':result}),
                                 input_guard=receipt, packet_audit=audit))
                previous = result
            assert tensor_digest(model.state_dict()) == before
            del model
            torch.cuda.empty_cache()
    receipt = dict(status='PASS', schema='xj_zgz_fit7_entrypoint_engineering_v1',
                   eval_protocol_sha256=sha(protocol_path), eval_sourcefreeze_sha256=sha(freeze_path),
                   scope='ch_global first3 packets, both modes/arms, public initial, beta0, no label/GT/zgz sampling',
                   optimizer_steps=0, supervision_reads=0, zgz_reads=0, device=device_info,
                   reader_guard=reader_guard.receipt(), packets=rows,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   elapsed_s=time.perf_counter()-started)
    dump(output, receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='packets'}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.protocol, args.source_freeze, args.out, args.device)


if __name__ == '__main__':
    main()

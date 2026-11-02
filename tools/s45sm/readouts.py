"""S45-SM readouts: S45-TR's frozen readout tasks (tools/s45tr/readouts.py run_task, unchanged) for the S45-SM networks, one
task per process on one GPU among 2-7; outputs under outputs/s45sm/20261031/readouts/<net>/<task>.

Networks `<net>` (step 6000: tl_e1, tl_f1, tl_e3, zl_e3) and `<net>_5000` / `<net>_5500` (tl_e1); a network is an arm, a
zero-training cell or an oracle. Only the network loader is replaced: the checkpoint of the given step (the trunk's, for a
cell), the config, and the network's routing hook. A cell writes its swap receipt afterwards.
"""
from __future__ import annotations


from .paths import OUT, require, require_one_gpu, sha
from .protocol import verify_protocol
from .registry import oracles, registry
from .routing import cell_net, install_for_config, register_for_net, resolve, write_swap_receipt

TASKS_6000 = ("tl_e1", "tl_f1", "tl_e3", "zl_e3")


def parse(net):
    arm, _, step = net.rpartition("_") if net.rsplit("_", 1)[-1] in ("5000", "5500") else (net, "", "")
    return arm, int(step) if step else 6000


def readout(net, task):
    arm, step = parse(net)
    require(arm in registry()["arms"] or arm in oracles(), "unregistered S45-SM arm " + arm)
    require(task in (TASKS_6000 if step == 6000 else ("tl_e1",)), f"task {task} is not registered for {net}")
    require_one_gpu()
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    verify_protocol()
    register_for_net(arm)
    from tools.s45tr import readouts as R

    def load(_net, device):
        from config import load_config
        from model import MNISTModel
        checkpoint, config = resolve(arm, step)
        cfg = load_config(str(config))
        install_for_config(cfg)
        model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
        return model.to(device).eval(), cfg, checkpoint
    R.load = load
    R.run_task(net, task, out_root=OUT / "readouts")
    if cell_net(arm):
        write_swap_receipt(net, "readouts", task, step)
    print("COMPLETE", net, task, sha(OUT / "readouts" / net / task / "states.npz"), flush=True)

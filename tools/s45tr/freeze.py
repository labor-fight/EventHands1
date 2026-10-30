#!/usr/bin/env python3
"""Freeze the S45-TR protocol before any S45-TR reading, and later code manifests (refuses to overwrite).

  freeze.py protocol                          the master protocol: pre-registration section SHA, arms and constants,
                                              the first batch's sources, every config and the frozen modules they import
  freeze.py manifest <name> <reason> <files>  one more set of sources (an arm's code, the T0 tools, the readout tools),
                                              frozen before that code's first preflight / first reading
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45tr.common import (ARMS, CONFIGS, FORMAL_ARMS, MANIFESTS, OUT, PAIRS, PHYSICAL_GPUS, PROTOCOL,  # noqa: E402
                                ROOT, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG_SHA256, S45_RUN, T1A_GNN,
                                now, require, save, sha)

SECTION = ("<!-- EXPERIMENT_RECORD:S45TR-PREREG-20261030:BEGIN -->", "<!-- EXPERIMENT_RECORD:S45TR-PREREG-20261030:END -->")
FIRST_BATCH = (
    "tools/s45tr/__init__.py", "tools/s45tr/common.py", "tools/s45tr/make_configs.py", "tools/s45tr/train.py",
    "tools/s45tr/accept.py", "tools/s45tr/chain.py", "tools/s45tr/freeze.py",
    "semkine/s45tr_n1.py", "semkine/s45tr_prior.py", "tests/test_s45tr.py",
)
FROZEN_DEPENDENCIES = (
    "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml", "outputs/s45/baseline.json",
    "semkine/xyz_mesh.py", "model/model.py", "model/config.py", "model/mano_layer.py", "semkine/dataset.py",
    "semkine/domrand.py", "semkine/encoder.py", "semkine/events.py", "semkine/train.py", "semkine/eval_track.py",
    "semkine/xag.py", "semkine/xag_sources.py", "semkine/event_gnn.py", "tools/xyz_c1_repro/train.py",
    "tools/xcl/evaluate.py", "tools/xcl/interventions.py", "tools/make_s36_row.py", "tools/report_table.py",
    "tools/s45loc/common.py", "tools/s45loc/run.py", "tools/s45loc/analyze.py", "tools/s45loc/deep.py",
    "assets/mano_right.npz",
)
CONSTANTS = {
    "n2_lambda": 1.0,
    "n3_offset_ms": [100, 1000],
    "n3_rule": "replaces S45's large + directed branches (its own first draw u >= 0.5); sign +-1 w.p. 1/2, opposite "
               "sign if outside the valid run, then clipped into it; no noise on the offset prior",
    "n4_p": 0.5,
    "n4_rule": "independent coin p = 0.5; prior = S45 W50 closed-loop bank state at the latest grid endpoint e < "
               "window start (same recording); no noise; otherwise / no endpoint: S45's sample",
    "t1a_gnn": T1A_GNN,
    "t1b_tau": 0.5,
    "t1b_rule": "events the S45 ray association already matched (q > 0, valid) take T0's face + barycentric when T0's "
                "on-hand probability >= tau; otherwise S45's association; q, z and every other field unchanged",
    "t0_heldout_subject": "ylf",
    "t0_gate": {"median_template_error_mm_max": 8.0,
                "rule": "on the held-out subject's on-hand events (XAG valid), T0's median template-surface error "
                        "<= 8.0 mm (one mean MANO template edge, 7.9 mm) AND below the median error of S45's own ray "
                        "association on the same events under S45's mixed training prior contract; otherwise T1b "
                        "is not trained"},
}


def protocol():
    require(not PROTOCOL.exists(), "protocol already frozen: " + str(PROTOCOL))
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = SECTION
    require(text.count(begin) == 1 and text.count(end) == 1, "pre-registration section markers missing or duplicated")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prereg_section.md").write_text(text[text.index(begin):text.index(end) + len(end)] + "\n")
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    files = list(FIRST_BATCH) + list(FROZEN_DEPENDENCIES) + sorted(str(p.relative_to(ROOT)) for p in CONFIGS.glob("*.yaml"))
    p = {
        "study": "S45TR_20261030", "seed": 3407, "frozen_at_utc": now(),
        "prereg_anchor": "docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030",
        "prereg_section_sha256": sha(OUT / "prereg_section.md"),
        "s45": {"run": str(S45_RUN), "checkpoint_sha256": S45_CHECKPOINT_SHA256, "config_sha256": S45_CONFIG_SHA256},
        "arms": ARMS, "formal_arms": list(FORMAL_ARMS), "constants": CONSTANTS,
        "physical_gpus": list(PHYSICAL_GPUS), "pairs": [list(p) for p in PAIRS],
        "file_sha256": {f: sha(ROOT / f) for f in files},
        "readings_before_freeze": "none of S45-TR; S45-LOC / S45-DEEP readings are the published references",
    }
    save(PROTOCOL, p, exclusive=True)
    print(PROTOCOL, len(p["file_sha256"]), "files frozen")


def manifest(name, reason, files):
    require(PROTOCOL.exists(), "freeze the protocol first")
    target = MANIFESTS / f"{name}.json"
    require(not target.exists(), "manifest already frozen: " + str(target))
    rel = [str((ROOT / f).resolve().relative_to(ROOT)) for f in files]
    save(target, {"name": name, "reason": reason, "frozen_at_utc": now(), "protocol_sha256": sha(PROTOCOL),
                  "file_sha256": {f: sha(ROOT / f) for f in rel}}, exclusive=True)
    print(target, len(rel), "files frozen")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["protocol"]:
        protocol()
    elif len(argv) >= 4 and argv[0] == "manifest":
        manifest(argv[1], argv[2], argv[3:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze a T1A-UB round or a set of files (CPU only; never rewrites a manifest).

  freeze.py round --round K --tag T1AUB-R<K>-PREREG-YYYYMMDD
      writes the round's configs (refuses to overwrite), checks them against the generator, and writes
      outputs/t1aub/20261102/manifests/r<K>.json: SHA-256 of the registry, the configs, the training files, the tests, the
      dependency files and the pre-registration section between its ledger markers.
  freeze.py files --name NAME --round K --tag TAG FILE...
      writes manifests/<NAME>.json for further files (evaluation / reading tools, amendments), with the section SHA of TAG.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402

#: files the round's code depends on and does not change (frozen by SHA; tools.frozen.accepts applies)
DEPENDENCIES = (
    "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml", "configs/s45tr_20261030/s45tr_t1a_s3407.yaml",
    "model/model.py", "model/config.py", "model/mano_layer.py",
    "semkine/xyz_mesh.py", "semkine/xag.py", "semkine/xag_sources.py", "semkine/s45tr_t1a.py", "semkine/event_gnn.py",
    "semkine/encoder.py", "semkine/events.py", "semkine/dataset.py", "semkine/domrand.py", "semkine/train.py",
    "semkine/eval_track.py", "semkine/s45sm.py", "semkine/s45sm_fast.py", "semkine/s45sm_oracle.py",
    "semkine/s45sm_dw.py", "semkine/s45sm_srp.py", "semkine/s45sm_hier.py", "semkine/s45rt_router.py",
    "semkine/s45tr_t0.py", "tools/xyz_c1_repro/train.py", "tools/xcl/evaluate.py", "tools/s45tr/readouts.py",
    "tools/s45tr/common.py", "tools/make_s36_row.py", "tools/s45sm/routing.py", "tools/s45sm/registry.py",
    "tools/s45sm/summary.py", "tools/s45rt/chain_r2.py", "tools/frozen.py", "tools/frozen_amendments.json")


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def manifest(name, k, tag, files, kind):
    dst = C.MANIFESTS / f"{name}.json"
    C.require(not dst.exists(), "manifest exists: " + str(dst))
    digest, length = C.section(tag)
    body = {"study": C.STUDY, "round": k, "kind": kind, "seed": C.SEED, "prereg_tag": tag,
            "prereg_section_sha256": digest, "prereg_section_bytes": length, "budget": C.BUDGET,
            "file_sha256": {rel(f): C.sha(f) for f in sorted(set(files), key=str)}, "frozen_at_utc": C.now()}
    C.save(dst, body, exclusive=True)
    return dst


def freeze_round(k, tag):
    reg = C.registry()
    C.require(k in reg["rounds"], f"round {k} not registered")
    written = C.write_configs(k)
    bad = [p for p, ok in C.check_configs(k).items() if not ok]
    C.require(not bad, "configs differ from the generator: " + ", ".join(bad))
    files = [C.PACKAGE / f"registry_r{k}.json", *written, *(ROOT / f for f in C.TRAIN_FILES),
             ROOT / "tests/test_t1aub.py", *(ROOT / f for f in DEPENDENCIES)]
    return manifest(f"r{k}", k, tag, files, "ROUND")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("round")
    r.add_argument("--round", type=int, required=True)
    r.add_argument("--tag", required=True)
    f = sub.add_parser("files")
    f.add_argument("--name", required=True)
    f.add_argument("--round", type=int, required=True)
    f.add_argument("--tag", required=True)
    f.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    if a.what == "round":
        print(freeze_round(a.round, a.tag))
    else:
        import re
        C.require(a.name.isidentifier() and not re.fullmatch(r"r\d+", a.name), "a file manifest cannot take a round's name")
        print(manifest(a.name, a.round, a.tag, [ROOT / x for x in a.files], "FILES"))


if __name__ == "__main__":
    main()

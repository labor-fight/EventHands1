"""Freezing and verification of the S45-SM code and rounds (CPU only).

Manifests in outputs/s45sm/20261031/manifests/:
  r<k>.json, r<k>a.json, ...   the legacy per-round / amendment manifests (frozen before this package existed, and the
                               manifests of later rounds): a registry, configs, ledger-section SHA, and (legacy only)
                               the code the round ran. Never rewritten.
  p1_core.json                 the identity manifest of this package (kind CODE_IDENTITY): SHA-256 of every module of the
                               package, its tests and the dependency files; the SHA of the archive inventory of the code it
                               replaced and of the equivalence report; the code identity.

`verify_protocol()` = identity manifest (if present) + `verify_legacy()` + every registered round has a manifest.
`verify_legacy()` re-hashes every entry of every legacy manifest: a path that still exists in the tree must match; a
path whose code was archived at the migration (outputs/s45sm/20261031/legacy_code/inventory.json, named by the identity
manifest) must be in the inventory with the manifest's SHA and its archived copy must hash to it.
`code_identity()` = SHA-256 of the sorted (module file name, SHA-256) list of this package's `*.py` files; it does not
depend on the directory name, so it is the same before and after the package directory is renamed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import (BUDGET, IDENTITY_MANIFEST, MANIFESTS, PACKAGE, ROOT, STUDY, TARGET, now, read, require, save, sha)
from .registry import config_path, oracles, phases, registry
from tools.frozen import accepts   # core files changed by the 2026-11-01 cleanup (tools/frozen_amendments.json)

#: files the package depends on and that no round manifest freezes on its own (the r1 DEPENDENCIES list plus two inputs
#: the legacy manifests missed: the own-loop prior sampler and the oracle's base config)
DEPENDENCIES = ("configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml", "configs/s45at_20261031/s45at_dz_s3407.yaml",
                "configs/s45at_20261031/s45at_t1ddz_s3407.yaml",
                "model/model.py", "semkine/xyz_mesh.py", "semkine/xag.py", "semkine/xag_sources.py",
                "semkine/s45rt_router.py", "semkine/s45tr_t0.py", "semkine/s45tr_prior.py", "semkine/s45ct_route.py",
                "semkine/encoder.py", "semkine/events.py", "semkine/dataset.py", "semkine/train.py",
                "semkine/eval_track.py", "tools/xyz_c1_repro/train.py", "tools/xcl/evaluate.py",
                "tools/s45tr/readouts.py", "tools/s45tr/common.py", "tools/s45loc/analyze.py", "tools/make_s36_row.py",
                "tools/s45rt/chain_r2.py")


def section(tag):
    """(sha256, bytes) of a ledger section between its BEGIN / END markers in docs/EXPERIMENT_RECORDS.md."""
    text = (ROOT / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = f"<!-- EXPERIMENT_RECORD:{tag}:BEGIN -->", f"<!-- EXPERIMENT_RECORD:{tag}:END -->"
    require(text.count(begin) == 1 and text.count(end) == 1, "pre-registration markers missing: " + tag)
    body = text[text.index(begin):text.index(end) + len(end)]
    return hashlib.sha256(body.encode()).hexdigest(), len(body)


# --------------------------------------------------------------------------- code identity
def code_files():
    """The modules of this package (sorted file names)."""
    return sorted(p for p in PACKAGE.glob("*.py") if p.is_file())


def code_identity():
    listing = [[p.name, sha(p)] for p in code_files()]
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def _rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _portable(path):
    """The path relative to the repository root when it is inside it, else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def test_files():
    return sorted((ROOT / "tests").glob("test_s45sm_*.py"))


def identity_files():
    """{relative path: SHA-256} of what the identity manifest freezes: package modules, tests, dependencies."""
    files = [_rel(p) for p in code_files()] + [_rel(p) for p in test_files()] + list(DEPENDENCIES)
    return {f: sha(ROOT / f) for f in files}


def build_identity(archive, equivalence_report=None):
    """Write manifests/p1_core.json (exclusive). `archive` is the directory holding files/ and inventory.json of the
    code this package replaced; `equivalence_report` the verification report (its SHA is recorded)."""
    archive = Path(archive).resolve()
    inventory = archive / "inventory.json"
    require(inventory.is_file(), "no archive inventory: " + str(inventory))
    legacy = sorted(p.name for p in MANIFESTS.glob("r*.json"))
    target = MANIFESTS / IDENTITY_MANIFEST
    require(not target.exists(), "identity manifest exists: " + str(target))
    data = {"study": STUDY, "status": "FROZEN", "kind": "CODE_IDENTITY", "frozen_at_utc": now(),
            "package": _rel(PACKAGE), "code_identity_sha256": code_identity(),
            "legacy_archive": _portable(archive),
            "legacy_inventory_sha256": sha(inventory), "supersedes": legacy,
            "equivalence_report": None if equivalence_report is None else _portable(equivalence_report),
            "equivalence_report_sha256": None if equivalence_report is None else sha(equivalence_report),
            "budget": BUDGET, "target_t1ddz": TARGET, "file_sha256": identity_files()}
    save(target, data, exclusive=True)
    return target


# --------------------------------------------------------------------------- freezing rounds / amendments
def freeze_round(round_, tag):
    """Write manifests/r<k>.json (exclusive) for a registered round: its registry, its configs, the ledger section of
    its pre-registration and the code identity. Run before any reading of the round."""
    reg = registry()
    require(round_ in reg["rounds"], "round not registered")
    target = MANIFESTS / f"r{round_}.json"
    require(not target.exists(), "manifest exists: " + str(target))
    files = [reg["rounds"][round_]]
    for name in [a for a, s in reg["arms"].items() if s["round"] == round_] + \
            [o for o, s in oracles().items() if s["round"] == round_]:
        files += [_rel(config_path(name, ph)) for ph in phases(name)]
    digest, size = section(tag)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    save(target, {"study": STUDY, "status": "FROZEN", "round": round_, "frozen_at_utc": now(), "prereg_tag": tag,
                  "prereg_section_sha256": digest, "prereg_section_bytes": size, "budget": BUDGET,
                  "target_t1ddz": TARGET, "code_identity_sha256": code_identity(),
                  "file_sha256": {f: sha(ROOT / f) for f in files}}, exclusive=True)
    return target, len(files)


def freeze_files(name, round_, tag, files):
    """Write manifests/<name>.json (an amendment: its new files and ledger section); the round's earlier manifests
    stay. The name starts with `r`, since verify_protocol reads manifests/r*.json."""
    require(name.startswith("r"), "manifest names start with r (verify_protocol reads manifests/r*.json)")
    target = MANIFESTS / f"{name}.json"
    require(not target.exists(), "manifest exists: " + str(target))
    digest, size = section(tag)
    save(target, {"study": STUDY, "status": "FROZEN", "round": round_, "amendment": name, "frozen_at_utc": now(),
                  "prereg_tag": tag, "prereg_section_sha256": digest, "prereg_section_bytes": size,
                  "code_identity_sha256": code_identity(), "file_sha256": {f: sha(ROOT / f) for f in files}},
         exclusive=True)
    return target, len(files)


# --------------------------------------------------------------------------- verification
def legacy_manifests():
    return sorted(MANIFESTS.glob("r*.json"))


def identity_manifest():
    path = MANIFESTS / IDENTITY_MANIFEST
    return read(path) if path.is_file() else None


def _archive_of(identity):
    """(archive directory, inventory dict) named by the identity manifest, or (None, None)."""
    if identity is None:
        return None, None
    archive = Path(identity["legacy_archive"])
    archive = archive if archive.is_absolute() else ROOT / archive
    inventory = archive / "inventory.json"
    require(inventory.is_file(), "legacy archive inventory missing: " + str(inventory))
    require(sha(inventory) == identity["legacy_inventory_sha256"], "legacy archive inventory changed")
    return archive, read(inventory)["files"]


def verify_legacy(identity=None):
    """Every entry of every legacy manifest, against the tree or (for retired code) the archive. Returns the manifests."""
    manifests = legacy_manifests()
    require(manifests, "no frozen S45-SM manifest")
    archive, files = _archive_of(identity)
    memo = {}

    def digest(path):
        if path not in memo:
            memo[path] = sha(path)
        return memo[path]
    out = []
    for path in manifests:
        m = read(path)
        require(m["study"] == STUDY and m["status"] == "FROZEN", "S45-SM manifest identity: " + path.name)
        for name, want in m["file_sha256"].items():
            if files is not None and name in files:
                require(files[name]["sha256"] == want, f"archived S45-SM file differs from its manifest ({path.name}): {name}")
                require(digest(archive / "files" / name) == want, f"archived S45-SM copy changed ({path.name}): {name}")
            else:
                require(digest(ROOT / name) == want or accepts(ROOT / name, want),
                        f"frozen S45-SM file changed ({path.name}): {name}")
        out.append(m)
    return out


def verify_protocol():
    """The identity manifest (when the migration has happened), the legacy manifests, and that every registered
    round is frozen. Returns the manifests read."""
    identity = identity_manifest()
    if identity is not None:
        require(identity["study"] == STUDY and identity["status"] == "FROZEN" and identity["kind"] == "CODE_IDENTITY",
                "S45-SM identity manifest identity")
        for name, want in identity["file_sha256"].items():
            require(sha(ROOT / name) == want, f"frozen S45-SM file changed (identity manifest): {name}")
        require(identity["code_identity_sha256"] == code_identity(), "S45-SM code identity changed")
    manifests = verify_legacy(identity)
    frozen = {m["round"] for m in manifests}
    require(set(registry()["rounds"]) <= frozen, "a registered round has no frozen manifest")
    return manifests + ([identity] if identity is not None else [])


def config_frozen(config):
    """The config's SHA must be recorded under its relative path by some manifest."""
    rel = _rel(config)
    digest = sha(config)
    manifests = [read(p) for p in MANIFESTS.glob("r*.json")]
    require(any(m["file_sha256"].get(rel) == digest for m in manifests), "Config not frozen in any S45-SM manifest: " + rel)

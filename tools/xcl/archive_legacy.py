#!/usr/bin/env python3
"""Inventory and archive exactly the approved legacy XAG/XA deletion scope.

Default: archive small provenance files and inventory every removed byte; delete
nothing. Applying requires the exact inventory plus a PASS migration receipt tied
to its dependency scan. Preserved sources/C1/paired models are never traversed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = "outputs/deleted_20261019/xag_xa_legacy"
REVIEW_KEEP = ("c1_baseline_discrepancy_", "c1_deep_audit_", "c1_rng_rank_", "c1_training_semantics_")
XAG_FIXED = ("runs", "main_rows", "main_row_audits", "single_seed", "attempts", "diagnostics", "diagnostics_v2",
             "execution_logs", "logs", "preflight", "protocol.json", "runtime.json", "execution.json",
             "frozen_sources.json", "implementation_ready.json", "current_metrics.json")
PROTECTED = ("outputs/xag/sources", "outputs/xag/source_manifest.json", "outputs/xyz_c1_repro",
             "outputs/xag_paired", "tools/xyz_c1_repro", "tools/xag_paired_20261018",
             "configs/xyz_c1_repro_20261018", "configs/xag_paired_20261018")


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exists(filename):
    return filename.exists() or filename.is_symlink()


def scope(root=ROOT):
    root = Path(root)
    selected = [root / "outputs/xag" / name for name in XAG_FIXED]
    xag = root / "outputs/xag"
    if xag.is_dir():
        selected += [p for p in xag.iterdir() if p.name.startswith(("diagnostic_smoke", "source_benchmark"))]
    reviews = xag / "reviews"
    if reviews.is_dir():
        selected += [p for p in reviews.iterdir() if not p.name.startswith(REVIEW_KEEP)]
    selected += [root / name for name in ("outputs/xa", "configs/xag", "configs/xa", "tools/xag", "tools/xag_single_seed", "tools/xa")]
    semkine = root / "outputs/semkine"
    if semkine.is_dir():
        selected += [p for p in semkine.iterdir() if p.name.startswith("xa_") or
                     (p.is_symlink() and p.name.startswith(("xag_", "single_seed_xag_")))]
    result = sorted(set(p for p in selected if exists(p)))
    for selected_path in result:
        relative = selected_path.relative_to(root).as_posix()
        if any(relative == keep or relative.startswith(keep + "/") or keep.startswith(relative + "/") for keep in PROTECTED):
            raise ValueError(f"Deletion scope overlaps protected identity: {relative}")
    return result


def in_scope(filename, selected):
    return any(filename == parent or parent in filename.parents for parent in selected)


def dependency_scan(root, selected):
    """Conservative live-code references, including inactive historical entrypoints."""
    rows = []
    needles = ("tools.xag", "tools/xag/", "tools.xa.", "tools/xa/", "outputs/xag/protocol.json", "outputs/xag/frozen_sources.json")
    for area in ("semkine", "model", "tools"):
        folder = root / area
        if not folder.is_dir():
            continue
        for filename in sorted(folder.rglob("*.py")):
            if in_scope(filename, selected) or filename.name == "archive_legacy.py":
                continue
            for line, text in enumerate(filename.read_text(errors="replace").splitlines(), 1):
                if any(needle in text for needle in needles):
                    rows.append({"path": str(filename.relative_to(root)), "line": line,
                                 "text": text.strip(), "file_sha256": digest(filename)})
    return {"references": rows, "sha256": json_digest(rows),
            "meaning": "Conservative references may include inactive historical paths; migration receipt must explicitly assess every remaining reference."}


def walk_entry(filename):
    yield filename
    if filename.is_dir() and not filename.is_symlink():
        for directory, dirnames, filenames in os.walk(filename, followlinks=False):
            for name in sorted(dirnames + filenames):
                yield Path(directory) / name


def key_file(filename, root):
    relative = filename.relative_to(root).as_posix()
    if relative.startswith(("tools/", "configs/")):
        return filename.suffix in (".py", ".yaml", ".yml", ".json", ".sh", ".md")
    if relative.startswith("outputs/xag/reviews/reconfirm_metrics_20261018/"):
        return ((filename.suffix in (".json", ".md") and "main_row" in filename.name)
                or filename.name in ("strict_main_table.md", "strict_main_table_command.txt", "final_receipt.json"))
    if relative.startswith("outputs/semkine/xa_"):
        return filename.suffix == ".json" and "main_row" in filename.name
    return relative in {
        "outputs/xag/protocol.json", "outputs/xag/runtime.json", "outputs/xag/execution.json", "outputs/xag/frozen_sources.json",
        "outputs/xag/current_metrics.json", "outputs/xag/single_seed/matrix_mechanism_summary.json",
        "outputs/xag/single_seed/decision.json", "outputs/xa/step5/p3_gates.json"}


def inventory(root, selected):
    records = []
    for parent in selected:
        for filename in walk_entry(parent):
            info = filename.lstat()
            kind = "symlink" if stat.S_ISLNK(info.st_mode) else "directory" if stat.S_ISDIR(info.st_mode) else "file"
            record = {"path": filename.relative_to(root).as_posix(), "kind": kind,
                      "bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "mode": stat.S_IMODE(info.st_mode)}
            if kind == "symlink":
                record["target"] = os.readlink(filename)
            elif kind == "file" and key_file(filename, root):
                record["sha256"] = digest(filename)
                record["archive"] = "files/" + record["path"]
            records.append(record)
    return records


def archive(root, destination):
    selected = scope(root)
    records = inventory(root, selected)
    scan = dependency_scan(root, selected)
    destination.mkdir(parents=True, exist_ok=False)
    for record in records:
        if "archive" not in record:
            continue
        source, target = root / record["path"], destination / record["archive"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != record["sha256"]:
            raise RuntimeError(f"Archive digest differs: {source}")
    # Preserve source manifest identity; its live copy and 30GB cache remain intact.
    source_manifest = root / "outputs/xag/source_manifest.json"
    preserved = {}
    if source_manifest.is_file():
        target = destination / "files/outputs/xag/source_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_manifest, target)
        preserved[str(source_manifest.relative_to(root))] = digest(target)
    manifest = {"schema": 1, "status": "ARCHIVED_NOT_DELETED", "root": str(root),
                "selected_roots": [p.relative_to(root).as_posix() for p in selected],
                "protected": list(PROTECTED), "records": records,
                "n_entries": len(records), "regular_file_bytes": sum(r["bytes"] for r in records if r["kind"] == "file"),
                "key_file_sha256": {r["path"]: r["sha256"] for r in records if "sha256" in r},
                "preserved_file_sha256": preserved, "dependency_scan": scan,
                "no_deletion_performed": True}
    (destination / "inventory.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def apply_deletion(root, manifest_path, migration_path):
    manifest = json.loads(manifest_path.read_text())
    migration = json.loads(migration_path.read_text())
    if manifest.get("root") != str(root) or manifest.get("status") != "ARCHIVED_NOT_DELETED":
        raise ValueError("Inventory root/status mismatch")
    if migration.get("status") != "PASS" or migration.get("dependency_scan_sha256") != manifest["dependency_scan"]["sha256"]:
        raise ValueError("Deletion requires PASS migration receipt for this exact dependency scan")
    if migration.get("inventory_sha256") != digest(manifest_path):
        raise ValueError("Migration receipt must bind the exact inventory SHA256")
    if not migration.get("new_xcl_runtime_independently_verified"):
        raise ValueError("New XCL runtime needs independent verification before removal")
    selected = [root / value for value in manifest["selected_roots"]]
    if selected != scope(root):
        raise ValueError("Deletion scope changed; create a new inventory")
    current = inventory(root, selected)
    if current != manifest["records"]:
        raise ValueError("Legacy files changed since inventory; archive the new version first")
    for row in current:
        if "archive" in row and digest(manifest_path.parent / row["archive"]) != row["sha256"]:
            raise ValueError("Archived evidence missing or changed")
    # Require every surviving reference to be assessed by path and present hash.
    current_scan = dependency_scan(root, selected)
    assessed = migration.get("assessed_reference_files_sha256", {})
    for row in current_scan["references"]:
        if assessed.get(row["path"]) != row["file_sha256"]:
            raise ValueError(f"Unassessed live dependency: {row['path']}:{row['line']}")
    relative = [str(p.relative_to(root)) for p in selected]
    tracked = subprocess.run(["git", "ls-files", "-z", "--", *relative], cwd=root, check=True, capture_output=True).stdout.split(b"\0")
    tracked = [os.fsdecode(value) for value in tracked if value]
    if tracked:
        subprocess.run(["git", "rm", "-f", "--", *tracked], cwd=root, check=True)
    for filename in selected:
        if not exists(filename):
            continue
        if filename.is_symlink() or filename.is_file():
            filename.unlink()
        else:
            shutil.rmtree(filename)
    receipt = {"status": "DELETED", "inventory_sha256": digest(manifest_path),
               "migration_receipt_sha256": digest(migration_path), "deleted_roots": relative,
               "git_removed": tracked, "protected": list(PROTECTED)}
    target = manifest_path.parent / "deletion_receipt.json"
    with target.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--archive", default=DEFAULT_ARCHIVE, type=Path)
    p.add_argument("--apply", action="store_true"); p.add_argument("--inventory", type=Path)
    p.add_argument("--migration-receipt", type=Path)
    args = p.parse_args(argv)
    if args.apply:
        if not args.inventory or not args.migration_receipt:
            p.error("--apply requires --inventory and --migration-receipt")
        result = apply_deletion(ROOT, args.inventory.resolve(), args.migration_receipt.resolve())
    else:
        destination = args.archive if args.archive.is_absolute() else ROOT / args.archive
        if not destination.resolve().is_relative_to(ROOT / "outputs/deleted_20261019"):
            raise ValueError("Archive only inside the approved dated deletion archive")
        result = archive(ROOT, destination)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "key_file_sha256")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Package the completed opt-in implementation and evidence; no model changes."""
import hashlib
import json
from pathlib import Path
import tarfile

from .protocol import ROOT, OUT, write_json


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    document = ROOT / "docs/S43_REL_实现训练与流式结果_20261027.md"
    required = [OUT / "finalization_complete.json", OUT / "mean_shape_deployment_complete.json",
                OUT / "report/results.json", OUT / "mean_shape/results.json", document,
                OUT / "final_decision.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Do not package an unfinished delivery: " + repr(missing))
    protocol = json.loads((OUT / "protocol.json").read_text())
    unchanged = {path: sha(ROOT / path) == identity
                 for path, identity in protocol["frozen_existing_sources"].items()}
    if not all(unchanged.values()):
        raise RuntimeError("A frozen original dependency changed")
    if sha(Path(protocol["split_manifest"])) != protocol["split_sha256"]:
        raise RuntimeError("The original split manifest changed")
    files = set()
    for folder in (ROOT / "tools/s43_rel", ROOT / "configs/s43_rel"):
        files.update(path for path in folder.rglob("*")
                     if path.is_file() and path.suffix in (".py", ".cpp", ".yaml", ".md"))
    files.update((ROOT / "tests").glob("test_s43_rel_*.py"))
    files.update((document, ROOT / "docs/S43表征调研.md", ROOT / "docs/EXPERIMENT_RECORDS.md"))
    files.update((ROOT / "docs/S43_三种表征方案与主方案定义_第二轮重设计_20261027.md",
                  ROOT / "docs/S43_100单元交叉审查_第二轮重设计_20261027.md"))
    evidence = ("protocol.json", "correction_R1_prereg.json", "mean_shape_deployment_extension.json",
        "environment_snapshot.json", "full_design_lock.json", "postprocess_complete.json",
        "finalization_complete.json", "mean_shape_deployment_complete.json", "final_decision.json",
        "report/results.json", "report/main_table.md", "report/resources.json", "report/report.md",
        "report/historical_reference_rows.md", "report/historical_reference_rows.json",
        "mean_shape/results.json", "mean_shape/main_table.md", "mean_shape/report.md",
        "mechanisms/e12/summary.json", "mechanisms/e12/R1/summary.json", "mechanisms/e3/summary.json",
        "robustness/full_final/report.json", "audits/full_training.json", "audits/full_training_online.json",
        "audits/control_prefix_final.json", "audits/final_fixed50_purequery.json",
        "audits/final_common_cadence_grid.json", "audits/canonical_numeric_audit_summary.json",
        "audits/mesh_deployment_independent_audit.json", "audits/spatial_order_intervention.json",
        "engineering/mesh_service_final_sourcefreeze.json", "mean_shape/formal_sourcefreeze.json",
        "finalization/unit_tests_final.log")
    files.update(OUT / path for path in evidence if (OUT / path).exists())
    files.update((OUT / "mean_shape/executions").glob("*.json"))
    weights = {}
    for arm in ("s43", "enhanced", "no_order", "full"):
        checkpoint = OUT / "train/full_r1" / arm / "step=6000.ckpt"
        files.add(checkpoint)
        resume = checkpoint.with_name("last.ckpt")
        weights[arm] = dict(deployment=str(checkpoint.relative_to(ROOT)), deployment_sha256=sha(checkpoint),
                            resume=str(resume.relative_to(ROOT)), resume_sha256=sha(resume))
    records = {str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(files)}
    mechanism_caches = {str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in (OUT / "mechanisms/e12/existing_features.npz", OUT / "mechanisms/e12/synthetic_features.npz")}
    folder = OUT / "delivery"
    folder.mkdir(exist_ok=True)
    archive = folder / "s43_rel_source_evidence_weights.tar.gz"
    if archive.exists() or (folder / "manifest.json").exists():
        raise FileExistsError("Preserve the existing delivery")
    temporary = folder / "bundle.tmp.tar.gz"
    with tarfile.open(temporary, "w:gz") as target:
        for path in sorted(files):
            target.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    temporary.replace(archive)
    write_json(folder / "manifest.json", {
        "execution_complete": True, "archive":str(archive.relative_to(ROOT)),
        "archive_sha256":sha(archive), "archive_bytes":archive.stat().st_size,
        "files":records, "checkpoints":weights, "frozen_original_sources_unchanged":unchanged,
        "original_split_manifest_unchanged":True,
        "mechanism_caches_retained_in_workspace_not_duplicated":mechanism_caches,
        "scope":"Overlay bundle: new opt-in code/config/tests/docs, completed evidence, and four final deployment weights.",
        "dependencies":"Existing repository at frozen source identities, EventHandsTrain environment, original canonical S43 checkpoint, MANO asset and hand_data51 dataset are required and are not duplicated. Offline R1 prototype reproduction additionally uses the recorded E12 feature caches and baseline metrics retained in the workspace; extraction can be regenerated with the documented mechanism commands.",
        "resume":"Full last.ckpt optimizer/RNG/sequence states remain in workspace at recorded identities; archive contains weights-only step6000 for deployment.",
        "scientific_status":"See final_decision.json. Packaging completion is not evidence of a successful new mechanism."})
    print(json.dumps({"archive":str(archive),"files":len(records),"bytes":archive.stat().st_size}))


if __name__ == "__main__":
    main()

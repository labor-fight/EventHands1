"""Read-only S43-REL representation audit, with synthetic arithmetic witnesses.

This entry point deliberately has no task-fitting or task-score option.  It reads
old source/documents, NPY headers inside the frozen E12 cache, and metadata only.
The constructed nonnegative-Z examples are algebraic witnesses, not evidence of
reachable physical event streams or different MANO targets.  Old files are never
written.  New FIT/SELECT task experiments require a separate preregistration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/s43_revisit_20261028/rep_audit"
E12 = ROOT / "outputs/s43_rel_20261027/mechanisms/e12"
H, J = 24, 51
SOURCE_PATHS = (
    "docs/S43_REL_实现训练与流式结果_20261027.md",
    "docs/S43_三种表征方案与主方案定义_第二轮重设计_20261027.md",
    "tools/s43_rel/core.py", "tools/s43_rel/core_cpu.cpp",
    "tools/s43_rel/geometry.py", "tools/s43_rel/geometry_cpu.cpp",
    "tools/s43_rel/joint_reader.py", "tools/s43_rel/model.py",
    "tools/s43_rel/data.py", "tools/s43_rel/mechanisms.py",
    "tools/s43_rel/readouts.py", "tools/s43_rel/readouts_revision.py",
    "outputs/s43_rel_20261027/protocol.json",
    "outputs/s43_rel_20261027/correction_R1_prereg.json",
    "outputs/s43_rel_20261027/mechanisms/e12/implementation_manifest.json",
    "outputs/s43_rel_20261027/mechanisms/e12/existing_features.manifest.json",
    "outputs/s43_rel_20261027/mechanisms/e12/R1/implementation_manifest.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def cache_headers(path):
    """Only parse NPY headers. Never materialize features, targets or scores."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            result[info.filename[:-4]] = {
                "shape": list(shape), "dtype": str(dtype),
                "fortran_order": bool(fortran), "payload_materialized": False,
            }
    return result


def support_grams(a, lam):
    """Independent numpy definition, preserving the event-slot index."""
    gh = np.einsum("nh,nj,nk->hjk", lam, a, a)
    return gh, gh.sum(0), np.diagonal(gh, axis1=1, axis2=2).T.copy()


def binding_read(a, z, cosines, point_index):
    """Exercise the frozen production contraction on a fabricated binding."""
    from tools.s43_rel.geometry import GeometryBinding

    point_index = np.asarray(point_index, np.int64)
    n = len(a)
    ids = np.arange(len(z), dtype=np.int64)
    binding = GeometryBinding(
        uv=np.zeros((n, 2)), n=np.tile([1., 0.], (n, 1)), a=a,
        edge_ids=ids, edge_slot=ids, point_index=point_index,
        weights=np.ones(len(z)), normal_dot_delta=np.asarray(cosines),
        branch_regular=np.ones(n, bool), metadata={"synthetic_arithmetic": True},
    )
    return binding.read(SimpleNamespace(edge_ids=ids, Z=z.reshape(-1, 2, 2, 3, 2)))


def symmetric_point_z(lam, m=None):
    """Two unique +/-normal edges per point realize any |m| <= lambda.

    This proves feasibility in the nonnegative-Z cone of the contraction. It
    does not prove that the arrays are reachable by the event-memory recurrence.
    """
    m = np.zeros_like(lam) if m is None else np.asarray(m)
    if np.any(np.abs(m) > lam):
        raise ValueError("|m| <= lambda required")
    z = np.stack(((lam + m) / 2., (lam - m) / 2.), 1).reshape(-1, H)
    return z, np.tile([1., -1.], len(lam)), np.repeat(np.arange(len(lam)), 2)


def self_checks():
    """Small deterministic arithmetic checks, without data, labels or models."""
    from tools.s43_rel.core import RelConfig, RelationMemory
    from tools.s43_rel.joint_reader import joint_gram_features
    import torch

    result = {}
    # An actual pair of raw event streams, aliased by the 2x2 dictionary.
    ev = np.array([[0, 0, 1, 0], [2, 0, 3, 1],
                   [0, 0, 5, 0], [2, 0, 7, 1]], np.int64)
    shifted = ev.copy(); shifted[:, :2] += 1
    snapshots = []
    cfg = RelConfig(width=8, height=8, dtype="float64", backend="numpy")
    for raw in (ev, shifted):
        memory = RelationMemory(cfg)
        memory.ingest(raw, 10, 0, time_unit="ms", source_range=(0, len(raw)))
        snapshots.append(memory.readout(10))
        memory.release()
    assert not np.array_equal(ev, shifted)
    assert np.array_equal(snapshots[0].A, snapshots[1].A)
    assert np.array_equal(snapshots[0].Z, snapshots[1].Z)
    result["raw_to_cell_alias"] = dict(
        raw_streams_different=True, A_bitwise_equal=True, Z_bitwise_equal=True,
        scope="actual synthetic raw streams; within-cell position discarded; no MANO task labels")

    # Preserve zeroth, first and second direction moments; change the fourth.
    cosines = np.array([1., -1., 0., np.sqrt(.5), -np.sqrt(.5)])
    za = np.zeros((5, H)); zb = np.zeros_like(za)
    za[:, 0] = [.25, .25, .5, 0., 0.]
    zb[:, 0] = [0., 0., 0., .5, .5]
    a = np.zeros((1, J)); a[0, 0] = 1.
    pa = binding_read(a, za, cosines, np.zeros(5, int))
    pb = binding_read(a, zb, cosines, np.zeros(5, int))
    assert np.allclose(pa["f"], pb["f"], atol=1e-14, rtol=0)
    assert np.allclose(pa["lam"], pb["lam"], atol=1e-14, rtol=0)
    fourth = [(z[:, 0] * cosines**4).sum() for z in (za, zb)]
    assert abs(fourth[0] - fourth[1]) > .24
    result["Z_to_normal_moments_collision"] = dict(
        total_mass_equal=bool(za.sum() == zb.sum()), f_max_difference=float(np.abs(pa["f"]-pb["f"]).max()),
        lambda_max_difference=float(np.abs(pa["lam"]-pb["lam"]).max()),
        fourth_direction_moments=list(map(float, fourth)),
        scope="nonnegative-Z algebraic cone; raw reachability and MANO target relevance unproven")

    # m is distinguishable, yet current f/a/lambda and every support Gram agree.
    a = np.zeros((3, J)); a[0, 0] = a[1, 0] = a[2, 1] = 1.
    lam = np.zeros((3, H)); lam[:, 0] = 1.
    ma = np.zeros_like(lam); ma[:, 0] = [.5, -.5, 0.]
    mb = np.zeros_like(lam)
    packages = []
    for m in (ma, mb):
        z, nd, point = symmetric_point_z(lam, m)
        packages.append(binding_read(a, z, nd, point))
    for key in ("f", "a", "lam"):
        assert np.array_equal(packages[0][key], packages[1][key])
    assert not np.array_equal(packages[0]["m"], packages[1]["m"])
    result["point_direction_to_f_collision"] = dict(
        f_a_lambda_bitwise_equal=True, point_m_l2_difference=float(np.linalg.norm(ma-mb)),
        action_matrix_rank=int(np.linalg.matrix_rank(a)), points=len(a),
        scope="explicit cancellation in null(a.T); not a MANO task-risk result")

    # H-specific offdiagonal couplings are absent from C and diagG.
    a = np.zeros((2, J)); a[:, :2] = [[1., 1.], [1., -1.]]
    la = np.zeros((2, H)); la[:, :2] = [[3., 1.], [1., 3.]]
    lb = np.zeros_like(la); lb[:, :2] = [[1., 3.], [3., 1.]]
    ga, ca, da = support_grams(a, la)
    gb, cb, db = support_grams(a, lb)
    assert np.array_equal(ca, cb) and np.array_equal(da, db)
    assert not np.array_equal(ga, gb)
    pa, pb = [binding_read(a, *symmetric_point_z(ll)) for ll in (la, lb)]
    assert np.array_equal(pa["f"], pb["f"])
    for ll, c, d in ((la, ca, da), (lb, cb, db)):
        tc, td = joint_gram_features(torch.tensor(a)[None], torch.tensor(ll)[None])
        np.testing.assert_allclose(tc.numpy()[0], c, rtol=0, atol=1e-14)
        np.testing.assert_allclose(td.numpy()[0], d, rtol=0, atol=1e-14)
    result["R1_slot_coupling_collision"] = dict(
        f_C_diagG_bitwise_equal=True, lambda_different=True,
        per_slot_Gram_max_difference=float(np.abs(ga-gb).max()),
        G0_cross_before=float(ga[0, 0, 1]), G0_cross_after=float(gb[0, 0, 1]),
        actual_frozen_joint_gram_function_parity=True,
        scope="positive support factors, not certified physical MANO states/events")

    # Retaining m is not automatically necessary for a specific linear task.
    rng = np.random.default_rng(3407)
    a = rng.normal(size=(8, 3)); weights = rng.uniform(.5, 2., size=8)
    truth = np.array([.25, -.5, .1]); measurement = a @ truth
    gram = a.T @ (weights[:, None] * a)
    rhs = a.T @ (weights * measurement)
    recovered = np.linalg.solve(gram, rhs)
    np.testing.assert_allclose(recovered, truth, rtol=0, atol=1e-14)
    # Change the target measurement in null(A.T W): normal-equation solution
    # stays identical even though the pointwise residual distribution changes.
    q = rng.normal(size=8)
    null = q - a @ np.linalg.solve(gram, a.T @ (weights * q))
    second_rhs = a.T @ (weights * (measurement + null))
    np.testing.assert_allclose(second_rhs, rhs, rtol=0, atol=1e-13)
    result["linear_task_sufficient_normal_equations_control"] = dict(
        max_solution_error=float(np.abs(recovered-truth).max()),
        different_point_residuals_same_optimum=True,
        scope="known fixed linear Gaussian/quadratic task; REL f is NOT this calibrated RHS")

    # Local derivative identity concerns the chosen finite compatibility only.
    a = rng.normal(size=(5, 3)); cosine = rng.uniform(-1, 1, size=5)
    z = rng.uniform(.1, 2., size=5); analytic = a.T @ (cosine*z)
    eps = 1e-5; numerical = []
    for j in range(3):
        step = np.eye(3)[j]*eps
        plus = np.sum(z*np.tanh(cosine*(a@step)))
        minus = np.sum(z*np.tanh(cosine*(a@(-step))))
        numerical.append((plus-minus)/(2*eps))
    error = float(np.abs(analytic-numerical).max())
    assert error < 1e-8
    result["finite_compatibility_gradient"] = dict(
        max_error=error, scope="derivative at zero only; no physical likelihood or motion-speed calibration")

    # An output bias cannot invent a typed correction without observation.
    from tools.s43_rel.joint_reader import JointGramSupportAdapter
    reader = JointGramSupportAdapter().double()
    with torch.no_grad():
        reader.output.weight.fill_(.25)
        reader.output.bias.fill_(3.)
    f = torch.zeros((1,51,24),dtype=torch.float64)
    a = torch.ones((1,128,51),dtype=torch.float64)
    lam = torch.zeros((1,128,24),dtype=torch.float64)
    zero = reader(f,a,lam,torch.ones((1,3),dtype=torch.float64))
    assert torch.count_nonzero(zero)==0
    result["zero_evidence_hardgate"] = dict(
        arbitrary_geometry_and_nonzero_output_bias=True,output_exact_zero=True,
        scope="frozen joint_gram adapter; no model checkpoint or task target")
    result["storage_accounting"] = dict(
        scalar_type="float32; support factors exclude shared context and source history",
        existing_point_factor=11112,offline_R1=2736,
        production_R1_before_shared_MLP=288+1224+2601+1224,
        full_slot_upper_Gram=288+1224+24*(51*52//2),
        m_extra=128*24,selected_Z_extra="24*number_selected_edges; addresses and links additional",
        note="Dimensions are accounting, not evidence that a larger object helps the task")
    return result


def metadata_audit():
    path = E12 / "existing_features.npz"
    manifest = json.loads((E12/"existing_features.manifest.json").read_text())
    original = json.loads((E12/"implementation_manifest.json").read_text())
    revision = json.loads((E12/"R1/implementation_manifest.json").read_text())
    samples = manifest["samples"]
    fields = ("sample_id", "partition", "recording", "domain", "history_source", "source_sha256", "source_events")
    selected = [{k:s[k] for k in fields} for s in samples]
    assert not any(s["recording"].startswith("zgz") for s in selected)
    headers = cache_headers(path)
    return dict(
        path=str(path.relative_to(ROOT)), sha256=sha256(path), array_headers=headers,
        samples=selected, split_counts=dict(Counter(s["partition"] for s in selected)),
        recording_counts=dict(sorted(Counter(s["recording"] for s in selected).items())),
        original_feature_order=original["typed_order"],
        original_pool=original["feature_pool"], original_reset=manifest["reset_contract"],
        original_reader_budget={k:original[k] for k in ("MLP_JL", "MLP_hidden", "MLP_steps", "MLP_batch", "MLP_lr")},
        offline_R1_features=revision["features"], offline_R1_normalization=revision["normalization"],
        limitations=[
            "Only headers and metadata opened: no feature arrays, supervision arrays or prediction/score artifacts materialized.",
            "Original cache contains f/a/lambda after float32 analytic log transforms; inverse transform cannot recover pre-cache float64 exactly.",
            "Cache has no m, selected-edge Z, edge addresses, geometric links, UV/normals or branch_regular.",
            "Cache can compare support compression but cannot audit Z-to-m or m-to-f task losses without preregistered same-source re-extraction.",
            "Offline R1 is A288+f1224+diagG1224; deployed R1 additionally uses full summed C, then shared-token pooling.",
            "Both snippets reset at the identical 50ms boundary; this is not full continuous deployment history.",
        ], targets_materialized=False, new_task_scores_computed=False, zgz_opened=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--self-check", action="store_true", help="run only synthetic arithmetic; no task scores")
    parser.add_argument("--cache-metadata", action="store_true", help="also audit frozen E12 headers and metadata")
    args = parser.parse_args()
    if not args.self_check and not args.cache_metadata:
        parser.error("select --self-check and/or --cache-metadata; task-score execution is intentionally unavailable")
    if args.out.exists():
        raise FileExistsError(f"refuse to overwrite an audit: {args.out}")
    identities = {p:sha256(ROOT/p) for p in SOURCE_PATHS}
    identities[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    receipt = dict(
        schema="s43_revisit_representation_audit_v1", utc=datetime.now(timezone.utc).isoformat(),
        scope="read-only old source/docs/cache metadata plus synthetic arithmetic; no task training/inference",
        source_sha256=identities, task_preregistration_required=True,
        old_107_tests_reused_as_existing_evidence=True, old_tests_rerun=False,
        new_task_scores_computed=False, new_zgz_arrays_opened=False,
        checks=self_checks() if args.self_check else None,
        cache_metadata=metadata_audit() if args.cache_metadata else None,
    )
    receipt["source_files_unchanged"] = {p:sha256(ROOT/p)==digest for p,digest in identities.items()}
    if not all(receipt["source_files_unchanged"].values()):
        raise RuntimeError("a source changed while auditing; rerun with a new output path")
    args.out.mkdir(parents=True)
    write_json(args.out/"summary.json", receipt)
    print(json.dumps({"out":str(args.out), "synthetic_checks":len(receipt["checks"] or {}),
                      "new_task_scores_computed":False, "source_files_unchanged":True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

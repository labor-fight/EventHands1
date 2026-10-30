"""Small preregistered MANO/finite-sensor controls for the S43 B-R prototype.

The renderer's public geometry/texture and sensor law are known *synthetic
conditions*.  They are not claims of known texture or sensor phase on recorded
data.  True candidate/nuisance indices below are used only to generate events
and supply supervised evaluation targets; no representation receives them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from .geometry import (candidate_paths, fk_joints, provider_metadata,
                       render_paths)


DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs/s43_br_20261027/synthetic"
CONFIG = {
    "version": 1, "seed": 20261027, "bins": 50, "bin_us": 1000,
    "image_size": [24, 32], "source_size": [180, 240],
    "contexts": {"train": 12, "val": 6, "test": 12}, "samples_per_context": 8,
    "scales": [.8, 1., 1.2], "texture_seed": 7, "delta_log": .12,
    "ap": 2, "am": 2, "alpha": .9, "nu": .02, "cap": 8,
    "root_x_amplitudes_m": [-.015, 0., .015],
    "index_y_amplitudes_rad": [-.25, 0., .25],
    "K_cam_event_grid": [[300., 0., 120.], [0., 300., 90.], [0., 0., 1.]],
    "polarity": "physical model ON=0/OFF=1; not inferred from recorded polarity",
    "time": "50 complete half-open 1ms bins; finite brightness jumps assigned to bin",
    "independence_unit": "context; all eight sampled readouts stay in one split",
    "known_synthetic_conditions": "public MANO geometry/texture and finite sensor law",
    "forbidden_representation_inputs": ["true_k", "true_m", "target", "target_joints"],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_id() -> str:
    payload = {"config": CONFIG,
               "geometry_sha256": _sha(Path(__file__).with_name("geometry.py")),
               "synthetic_sha256": _sha(Path(__file__))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _split(index: int) -> str:
    if not 0 <= index < 30:
        raise ValueError("context index must be 0..29")
    return "train" if index < 12 else "val" if index < 18 else "test"


def sample_counts(paths: np.ndarray, true_k: int, true_m: int,
                  rng: np.random.Generator, *, ap: int = 2, am: int = 2,
                  alpha: float = .9, nu: float = .02, cap: int = 8,
                  return_details: bool = False):
    """Sample exactly the carrier's S=1 finite transition/readout model.

    One uniform phase is sampled per pixel at the epoch boundary and propagated
    through the complete epoch.  The caller selects one shared η for all pixels
    and bins; it is not independently resampled at each event or bin.
    """
    a = np.asarray(paths)
    if a.ndim != 4 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("integer paths[K,M,P,B+1] required")
    K, M, P, steps = a.shape
    if not 0 <= true_k < K or not 0 <= true_m < M or steps < 2:
        raise ValueError("candidate/nuisance index or path length invalid")
    if min(ap, am, cap) < 1 or not 0 <= alpha <= 1 or nu < 0:
        raise ValueError("invalid finite sensor law")
    phase = rng.integers(-am + 1, ap, size=P, dtype=np.int64)
    initial_phase = phase.copy()
    counts = np.empty((steps - 1, P, 2), dtype=np.uint8 if cap <= 255 else np.uint16)
    latent_total = np.zeros(2, dtype=np.int64)
    for b in range(steps - 1):
        h = phase + (a[true_k, true_m, :, b + 1] - a[true_k, true_m, :, b])
        plus = np.where(h >= ap, h // ap, 0)
        minus = np.where(h <= -am, (-h) // am, 0)
        phase = h - plus * ap + minus * am
        if np.any(phase < -am + 1) or np.any(phase >= ap):
            raise ArithmeticError("sampled phase escaped finite support")
        latent_total += [plus.sum(), minus.sum()]
        counts[b, :, 0] = np.minimum(rng.binomial(plus, alpha) + rng.poisson(nu, size=P), cap)
        counts[b, :, 1] = np.minimum(rng.binomial(minus, alpha) + rng.poisson(nu, size=P), cap)
    if return_details:
        return counts, {"initial_phase": initial_phase, "final_phase": phase,
                        "latent_total": latent_total}
    return counts


def fully_occluded_paths(paths: np.ndarray, background_level: int = -16) -> np.ndarray:
    """Declared full-occlusion negative control: every k/η predicts same radiance."""
    a = np.asarray(paths)
    if a.ndim != 4 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("integer paths[K,M,P,B+1] required")
    return np.full_like(a, int(background_level))


def _context_geometry(axis: str, index: int):
    if axis not in ("root_x", "index_y"):
        raise ValueError("axis must be root_x or index_y")
    axis_id = 101 if axis == "root_x" else 211
    rng = np.random.default_rng(np.random.SeedSequence([CONFIG["seed"], axis_id, index, 0]))
    # All nuisance/context draws are independent of the later supervised class.
    history = np.zeros(51, dtype=np.float64)
    history[:3] = [rng.uniform(-.018, .018), rng.uniform(-.03, .005), rng.uniform(.50, .64)]
    history[3:6] = rng.uniform(-.22, .22, 3)
    history[6:] = rng.normal(0., .025, 45)
    betas = rng.normal(0., .3, 10)
    K_cam = np.asarray(CONFIG["K_cam_event_grid"], dtype=np.float64)
    amplitudes = CONFIG["root_x_amplitudes_m"] if axis == "root_x" else CONFIG["index_y_amplitudes_rad"]
    states = candidate_paths(history, bins=CONFIG["bins"], amplitudes=amplitudes,
                             axis="tx" if axis == "root_x" else "index_y")
    paths = render_paths(states, betas, K_cam, image_size=tuple(CONFIG["image_size"]),
                         source_size=tuple(CONFIG["source_size"]), scales=CONFIG["scales"],
                         texture_seed=CONFIG["texture_seed"], delta=CONFIG["delta_log"])
    joints = fk_joints(states[:, -1], betas)
    return history, betas, K_cam, states, paths, joints


def generate_context(out: str | Path, axis: str, index: int) -> dict:
    """Generate/cache one independent geometry and eight noisy event readouts."""
    t0 = time.perf_counter()
    index = int(index)
    split = _split(index)
    out = Path(out)
    dest = out / axis / f"context_{index:02d}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cid = config_id()
    if dest.exists():
        with np.load(dest, allow_pickle=False) as data:
            if str(data["config_id"]) != cid:
                raise ValueError(f"cached context identity mismatch: {dest}")
        return {"axis": axis, "context": index, "split": split, "path": str(dest),
                "cached": True, "elapsed_seconds": time.perf_counter() - t0,
                "sha256": _sha(dest)}
    import torch
    torch.set_num_threads(1)
    history, betas, K_cam, states, paths, joints = _context_geometry(axis, index)
    axis_id = 101 if axis == "root_x" else 211
    rng = np.random.default_rng(np.random.SeedSequence([CONFIG["seed"], axis_id, index, 1]))
    n = CONFIG["samples_per_context"]
    true_k = rng.integers(0, paths.shape[0], size=n, dtype=np.int64)
    true_m = rng.integers(0, paths.shape[1], size=n, dtype=np.int64)
    observations = np.stack([sample_counts(paths, int(k), int(m), rng,
                                           ap=CONFIG["ap"], am=CONFIG["am"],
                                           alpha=CONFIG["alpha"], nu=CONFIG["nu"], cap=CONFIG["cap"])
                             for k, m in zip(true_k, true_m)])
    temp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(temp, paths=paths.astype(np.int16), counts=observations,
                        candidate_states=states.astype(np.float32),
                        candidate_joints=joints.astype(np.float32),
                        history=history.astype(np.float32), older_history=history.astype(np.float32),
                        betas=betas.astype(np.float32), K_cam=K_cam.astype(np.float32),
                        true_k=true_k, true_m=true_m,
                        target_state=states[true_k, -1].astype(np.float32),
                        target_joints=joints[true_k].astype(np.float32),
                        split=np.asarray(split), axis=np.asarray(axis), context=np.asarray(index),
                        config_id=np.asarray(cid))
    os.replace(temp, dest)
    return {"axis": axis, "context": index, "split": split, "path": str(dest),
            "cached": False, "elapsed_seconds": time.perf_counter() - t0,
            "sha256": _sha(dest), "paths_shape": list(paths.shape),
            "counts_shape": list(observations.shape)}


def _worker(task):
    return generate_context(*task)


def generate_dataset(out: str | Path = DEFAULT_OUT, workers: int = 6,
                     include_joint: bool = True) -> dict:
    """Generate 12/6/12 independent contexts, with eight samples in each.

    Source hashes are saved before generation.  The first context is synchronous
    so an engineering failure is visible before the remaining jobs are issued.
    """
    if workers < 1:
        raise ValueError("positive CPU worker count required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "generation_sources.json"
    receipt = {"config": CONFIG, "config_id": config_id(),
               "source_hashes_before_generation": {
                   "synthetic.py": _sha(Path(__file__)),
                   "geometry.py": _sha(Path(__file__).with_name("geometry.py")),
                   "carrier.py": _sha(Path(__file__).with_name("carrier.py"))},
               "workers": workers, "device": "cpu", "gt_input": False}
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["config_id"] != receipt["config_id"]:
            raise ValueError("generation directory belongs to another source/config identity")
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    start = time.perf_counter()
    axes = ["root_x", "index_y"] if include_joint else ["root_x"]
    first = generate_context(out, axes[0], 0)
    print(json.dumps({"engineering_first_context": first}, ensure_ascii=False), flush=True)
    (out / "first_context_receipt.json").write_text(json.dumps(first, indent=2) + "\n")
    tasks = [(str(out), axis, index) for axis in axes for index in range(30)
             if (axis, index) != (axes[0], 0)]
    records = [first]
    if workers == 1:
        for task in tasks:
            result = _worker(task)
            records.append(result)
            print(json.dumps({"context_done": result["axis"], "id": result["context"],
                              "seconds": result["elapsed_seconds"]}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                records.append(result)
                print(json.dumps({"context_done": result["axis"], "id": result["context"],
                                  "seconds": result["elapsed_seconds"]}), flush=True)
    records.sort(key=lambda x: (x["axis"], x["context"]))
    manifest = {"config": CONFIG, "config_id": receipt["config_id"], "axes": axes,
                "contexts": records, "elapsed_seconds": time.perf_counter() - start,
                "provider": provider_metadata(CONFIG["texture_seed"], CONFIG["delta_log"],
                                               tuple(CONFIG["image_size"]), CONFIG["scales"]),
                "context_split_counts_per_axis": CONFIG["contexts"],
                "sample_split_counts_per_axis": {k: v * CONFIG["samples_per_context"]
                                                  for k, v in CONFIG["contexts"].items()},
                "gt_usage": "true_k/m only generate observations; targets only supervision/evaluation"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_synthetic(out: str | Path = DEFAULT_OUT, axis: str = "root_x",
                   split: str | None = None) -> list[dict]:
    """Load contexts once; samples share immutable geometry/path arrays.

    Target/diagnostic keys are visibly named.  Feature constructors must only
    receive counts, paths, history, candidate states, and external conditions.
    """
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    if axis not in manifest["axes"] or split not in (None, "train", "val", "test"):
        raise ValueError("unknown synthetic axis/split")
    rows = []
    for context in manifest["contexts"]:
        if context["axis"] != axis or (split is not None and context["split"] != split):
            continue
        with np.load(context["path"], allow_pickle=False) as z:
            arrays = {k: np.array(z[k], copy=True) for k in z.files}
        if str(arrays["config_id"]) != manifest["config_id"]:
            raise ValueError("context config identity differs from manifest")
        group = f"{axis}/context_{context['context']:02d}"
        for i in range(len(arrays["counts"])):
            rows.append({"sample_id": f"{group}/sample_{i:02d}", "context_id": group,
                         "recording": group, "source": "synthetic_known_model",
                         "split": context["split"], "axis": axis,
                         "counts": arrays["counts"][i], "paths": arrays["paths"],
                         "candidate_states": arrays["candidate_states"],
                         "candidate_joints": arrays["candidate_joints"],
                         "history": arrays["history"], "older_history": arrays["older_history"],
                         "betas": arrays["betas"], "K_cam": arrays["K_cam"],
                         "target": arrays["target_state"][i],
                         "target_joints": arrays["target_joints"][i],
                         "target_class": int(arrays["true_k"][i]),
                         "diagnostic_true_m": int(arrays["true_m"][i]),
                         "event_cutoff_us": 50_000, "target_time_us": 50_000,
                         "polarity_contract": CONFIG["polarity"]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--root-only", action="store_true")
    args = p.parse_args()
    result = generate_dataset(args.out, args.workers, not args.root_only)
    print(json.dumps({"manifest": str(args.out / "manifest.json"),
                      "elapsed_seconds": result["elapsed_seconds"],
                      "contexts": len(result["contexts"])}), flush=True)


if __name__ == "__main__":
    main()

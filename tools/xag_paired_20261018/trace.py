#!/usr/bin/env python3
"""Full original zgz recursive trace with per-packet event/GT-source identities."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xag_paired_20261018 import verify_final as V
from tools.xag_paired_20261018.evaluate import configure_environment, set_policy, math_flags

COUNTS = {"zgz_global": 1386, "zgz_local": 1204}


def packet_sha(packet, names, *, canonical_sources=False):
    """C1 packet fingerprint encoding; optionally normalize only source extras."""
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        value = None if canonical_sources and name in V.SOURCE_FIELDS else getattr(packet, name, None)
        if value is None:
            h.update(b"None")
            continue
        array = value.detach().cpu().contiguous().numpy()
        h.update(str(array.dtype).encode()); h.update(str(array.shape).encode())
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


def source_cache_binding(protocol, cfg):
    manifest_path = V.path(protocol["source_manifest_path"])
    manifest = V.read(manifest_path)
    expected_root = ROOT / "outputs/xag/sources"
    V.require(Path(cfg["DATA"]["XAG_SOURCE_ROOT"]).resolve() == expected_root, "Different oracle source cache")
    records = {}
    for seq in COUNTS:
        for filename in ("metadata.json", "part.npy", "face.npy", "bary.npy", "offsets.npy", "done_ms.npy"):
            source = expected_root / seq / filename
            name = str(source.relative_to(ROOT))
            V.require(name in manifest["files"], "Evaluation GT source is not in original manifest")
            V.checked_file(source, manifest["files"][name]["sha256"])
            records[name] = manifest["files"][name]["sha256"]
        meta = V.read(expected_root / seq / "metadata.json")
        V.require(meta["seq"] == seq and meta["complete"] is True
                  and meta["version"] == "xag-source-v2-fg2px-depth10mm-face-majority-fk1", "Incomplete/different GT pseudo-source definition")
    return {"manifest": str(manifest_path), "manifest_sha256": V.sha(manifest_path), "files_sha256": records,
            "meaning": "Per-event-ms GT mesh geometric proxy cache; oracle diagnostics, not measured physical event origin"}


def grid(aux):
    import numpy as np
    ends, run_ids = [], []
    for rid, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)):
        values = np.arange(int(a) + 49, int(b), 50, dtype=np.int64)
        ends.extend(values); run_ids.extend([rid] * len(values))
    return np.asarray(ends, np.int64), np.asarray(run_ids, np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True); parser.add_argument("--run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_path = V.path(args.protocol).resolve(); protocol = V.verify_protocol(protocol_path)
    run = V.resolve_run(protocol, args.run); directory = V.path(run["output"])
    output = V.path(args.output_dir).resolve()
    V.require(output.is_relative_to(V.OUT) and not args.output_dir.is_symlink()
              and not output.exists(), "Choose a fresh trace directory within new paired study")
    configure_environment(); set_policy()
    contract = V.verify_run(protocol_path, protocol, run)
    checkpoint, _ = V.check_checkpoint(protocol_path, protocol, run)
    cfg = V.load_recipe(run)
    source_binding = source_cache_binding(protocol, cfg)
    import numpy as np
    import torch
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    device = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    sequences = sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg))
    V.require([seq for seq, _ in sequences] == list(COUNTS), "Both complete original zgz sequences required")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    arrays, results, packet_records = {}, [], []
    native_forward = model.forward_packet
    for sequence, legacy_dir in sequences:
        # Native eval packets encode window-relative times. Preserve the actual
        # absolute endpoint grid separately and verify source row/ms identities.
        _, offsets, aux, target = ET.load_sequence(Path(cfg["DATA"]["ROOT"]), legacy_dir, sequence)
        ends, run_ids = grid(aux)
        V.require(len(ends) == COUNTS[sequence], "Original W50 endpoint denominator changed")
        incoming, predictions = [], []
        def capture(packet):
            index = len(predictions)
            V.require(index < len(ends) and packet.prev_state.shape == (1, 51), "Trace packet order/batch changed")
            end = int(ends[index]); start = end - 49
            a0, a1 = int(offsets[start]), int(offsets[end + 1])
            source_rows = packet.xag_source_index.detach().cpu().numpy()
            source_ms = packet.xag_source_ms.detach().cpu().numpy()
            V.require(int(packet.events.shape[0]) == a1 - a0
                      and np.array_equal(source_rows, np.arange(a0, a1, dtype=np.int64))
                      and np.array_equal(source_ms, np.repeat(np.arange(start, end + 1, dtype=np.int64),
                                                              np.diff(offsets[start:end + 2]).astype(np.int64))),
                      "Actual events/source rows/milliseconds disagree with native endpoint clock")
            record = {"index": len(packet_records), "seq": sequence, "sequence_index": index,
                      "run_id": int(run_ids[index]), "end_ms": end, "window_ms": 50,
                      "n_events": int(packet.events.shape[0]), "events_sha256": packet_sha(packet, ("events",)),
                      "source_sha256": packet_sha(packet, V.SOURCE_FIELDS),
                      "source_field_sha256": {name: packet_sha(packet, (name,)) for name in V.SOURCE_FIELDS},
                      "canonical_common_packet_sha256": packet_sha(packet, V.RAW_FIELDS, canonical_sources=True),
                      "actual_full_packet_sha256": packet_sha(packet, V.RAW_FIELDS)}
            previous = packet.prev_state.detach().cpu().numpy()[0].copy()
            pred = native_forward(packet)
            prediction = pred.detach().cpu().numpy()[0].copy()
            V.require(previous.dtype == prediction.dtype == np.float32 and np.isfinite(previous).all()
                      and np.isfinite(prediction).all(), "Nonfinite trace keeps its full failure denominator")
            incoming.append(previous); predictions.append(prediction); packet_records.append(record)
            if len(packet_records) % 250 == 0:
                progress = output / "progress.json"
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps({"status": "TRACING", "n_endpoints": len(packet_records)}) + "\n")
                temporary.replace(progress)
            return pred
        model.forward_packet = capture
        with torch.no_grad():
            result = ET.track_sequence(model, mano, cfg, Path(cfg["DATA"]["ROOT"]), legacy_dir,
                                       sequence, 50, device, rng, 1.0, 1.0, None)
        V.require(result is not None and result["n_frames"] == len(predictions) == COUNTS[sequence], "Incomplete original recursive trace")
        arrays[sequence + "/incoming"] = np.stack(incoming)
        arrays[sequence + "/prediction"] = np.stack(predictions)
        arrays[sequence + "/target"] = np.asarray(target[ends], np.float32).copy()
        arrays[sequence + "/end_ms"] = ends; arrays[sequence + "/run_id"] = run_ids
        results.append(result)
    model.forward_packet = native_forward
    V.require(len(packet_records) == sum(value["n_frames"] for value in results) == 2590, "Full2590 denominator missing")
    np.savez_compressed(output / "states.npz", **arrays)
    with (output / "packets.jsonl").open("x") as stream:
        for record in packet_records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    V.verify_protocol(protocol_path)
    for name, expected in source_binding["files_sha256"].items(): V.checked_file(name, expected)
    V.require(math_flags() == V.MATH, "Strict math policy changed during trace")
    report = {"status": "COMPLETE", "name": run["name"], "cell": run["cell"], "seed": 3407, "step": 6000,
              "n_endpoints": 2590, "per_sequence": results, "checkpoint": str(checkpoint),
              "checkpoint_sha256": V.sha(checkpoint), "config_sha256": V.sha(V.path(run["config"])),
              "protocol_sha256": V.sha(protocol_path), "source_sha256": V.sha(__file__),
              "original_evaluator_sha256": V.sha(ROOT / "semkine/eval_track.py"),
              "states_sha256": V.sha(output / "states.npz"), "packet_journal_sha256": V.sha(output / "packets.jsonl"),
              "source_cache_binding": source_binding, "paired_training_binding_sha256": V.sha(directory / "paired_binding_rank0.json"),
              "old_source_binding_sha256": V.sha(directory / "xag_binding.json"), "math_flags": math_flags(),
              "contract_verification_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
              "oracle": run["cell"].startswith(("a1", "a2")),
              "runtime_policy": "XAG_PAIRED_C1_STRICT_20261018", "evaluation": "Unchanged ET.track_sequence, rng0, W50, all1386global+1204local; own predictions feed next step",
              "event_source_hashes": "Every packet has event-only, each source field, combined source, canonical-public and actual-full hashes in packets.jsonl",
              "clock_boundary": "end_ms/run_id reconstructed from the unchanged original valid-run loop and checked against every packet's exact absolute source row/ms; native packet times remain window-relative",
              "metric_definition": "MPJPE joint0 translation; MPVPE vertex0 translation; no Procrustes; original full-sequence endpoint means"}
    V.save_new(output / "report.json", report)
    print(json.dumps({"status": "COMPLETE", "run": run["name"], "n_endpoints": 2590,
                      "report": str((output / "report.json").relative_to(ROOT))}))


if __name__ == "__main__":
    main()

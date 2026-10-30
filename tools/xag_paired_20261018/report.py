#!/usr/bin/env python3
"""Validate completed paired results on CPU, then publish one original-format table.

No neural-network inference, training, score selection or baseline replacement.
All validation precedes creation of a fresh release and the final atomic switch
of outputs/xag/current_metrics.json. Existing products are never overwritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xag_paired_20261018 import verify_final as V
from tools.xag_paired_20261018 import evaluate as E
from tools.xag_paired_20261018 import trace as T

OUT = V.OUT
COUNTS = T.COUNTS
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
CELLS = ("a0f1", "a1f0", "a1f1", "a2f0", "a2f1")
POINTER = ROOT / "outputs/xag/current_metrics.json"
DECODER = V.C1 / "audit/metric_definition_20261018/recompute.py"
DECODER_SHA = "4b80314eaf4392ec0c44a4e989c4fbed4be040ffb07cc7e02416bb1b2096ed02"
S39 = ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"
CHUNK = 64
TOL_MM = .001


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(filename):
    filename = V.path(filename).resolve()
    return str(filename.relative_to(ROOT)) if filename.is_relative_to(ROOT) else str(filename)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked(filename, expected=None, records=None):
    filename = V.path(filename)
    V.require(filename.is_file() and not filename.is_symlink(), "Expected immutable regular file: " + str(filename))
    actual = V.sha(filename)
    if expected is not None:
        V.require(actual == expected, "Changed artifact: " + str(filename))
    if records is not None:
        records[relative(filename)] = actual
    return actual


def frozen(protocol, filename):
    key = relative(filename)
    V.require(protocol["file_sha256"].get(key) == V.sha(filename), "Required publication source is not frozen: " + key)


def write_new(filename, data):
    filename = Path(filename)
    V.require(filename.resolve().is_relative_to(OUT), "Publication writes only within the new paired study")
    with filename.open("xb") as stream:
        stream.write(data if isinstance(data, bytes) else json_bytes(data))
        stream.flush()
        os.fsync(stream.fileno())


def terminal_state(protocol_path, protocol):
    state = V.read(OUT / "status.json")
    V.require(state["status"] == "VERIFYING_ROWS_TRACES_AND_PUBLISHING" and state.get("results_ready") is False,
              "All five formal trajectories, rows and traces must complete before publication")
    V.require(state["protocol_sha256"] == V.sha(protocol_path)
              and set(state["runs"]) == {run["name"] for run in protocol["runs"]}, "Coordinator protocol/run identity changed")
    for run in protocol["runs"]:
        item = state["runs"][run["name"]]
        V.require(item["status"] == "TRAINED" and item["exit_code"] == 0
                  and item["paired_checkpoint_steps"] == list(V.GRID)
                  and all(item[key] == value for key, value in run.items())
                  and V.path(item["main_row"]).resolve() == OUT / "main_rows" / f"{run['prefix']}_main_row_step6000.json",
                  "Coordinator has not completed the registered run: " + run["name"])
    return state


def previous_pointer(protocol):
    V.require(V.path(protocol["previous_current_metrics_path"]).resolve() == POINTER
              and POINTER.is_file() and not POINTER.is_symlink(), "Explicit previous regular metrics pointer required")
    data = POINTER.read_bytes()
    V.require(hashlib.sha256(data).hexdigest() == protocol["previous_current_metrics_sha256"], "Current metrics pointer changed before publication")
    value = json.loads(data)
    V.require(value["status"] == "ACTIVE_STRICT_REEVALUATED_XAG_METRICS" and value["model_seed"] == 3407
              and value["step"] == 6000 and set(value["rows"]) == {"a0f0", *CELLS}, "Unexpected previous metrics identity")
    for cell, row in value["rows"].items():
        checked(row["main_row"], row["sha256"])
        V.require(row["oracle"] == cell.startswith(("a1", "a2")), "Previous oracle label changed")
    checked(value["completion_receipt"], value["completion_receipt_sha256"])
    return data


def completed_training(protocol_path, protocol, records):
    audit_path = OUT / "paired_training_audit.json"
    audit = V.read(audit_path)
    V.require(audit["status"] == "PASS" and audit["source_sha256"] == V.sha(ROOT / "tools/xag_paired_20261018/verify_final.py")
              and audit["protocol_sha256"] == V.sha(protocol_path)
              and audit["baseline_identity_sha256"] == protocol["baseline_identity_sha256"]
              and audit["source_manifest_sha256"] == protocol["source_manifest_sha256"], "Full training verification is missing or changed")
    V.require(len(audit["runs"]) == 5 and {run["name"] for run in audit["runs"]} == {run["name"] for run in protocol["runs"]},
              "Full verification must contain all five arms")
    saved = {item["name"]: item for item in audit["runs"]}
    current = {}
    for run in protocol["runs"]:
        # Read all real checkpoints, tapes, journals and rank snapshots again.
        result = V.verify_run(protocol_path, protocol, run)
        V.require(result == saved[run["name"]], "Saved full training audit differs from current physical evidence")
        current[run["name"]] = result
        records.update({relative(name): value for name, value in result["file_sha256"].items()})
    matrix = V.verify_matrix(protocol, protocol["runs"])
    V.require(audit["matrix"] == matrix and matrix["status"] == "PASS", "Five-arm source/initialization matrix audit changed")
    records.update({relative(name): value for name, value in matrix["file_sha256"].items()})
    checked(audit_path, records=records)
    return current, matrix


def load_rows(protocol_path, protocol, contracts, records):
    products = {}
    for run in protocol["runs"]:
        filename = OUT / "main_rows" / f"{run['prefix']}_main_row_step6000.json"
        receipt_path = filename.with_name(filename.stem + "_receipt.json")
        row, receipt = V.read(filename), V.read(receipt_path)
        checkpoint = Path(contracts[run["name"]]["checkpoint"])
        E.check_row(row, run, checkpoint)
        V.require(receipt["status"] == "COMPLETE" and receipt["runtime_policy"] == "XAG_PAIRED_C1_STRICT_20261018"
                  and receipt["seed"] == 3407 and receipt["step"] == 6000
                  and V.path(receipt["row"]).resolve() == filename
                  and receipt["row_sha256"] == V.sha(filename)
                  and receipt["wrapper_sha256"] == V.sha(ROOT / "tools/xag_paired_20261018/evaluate.py")
                  and receipt["original_evaluator_sha256"] == V.sha(ROOT / "tools/make_s36_row.py")
                  and receipt["original_reporter_sha256"] == V.sha(ROOT / "tools/report_table.py")
                  and receipt["protocol_sha256"] == V.sha(protocol_path)
                  and receipt["checkpoint_sha256"] == V.sha(checkpoint)
                  and receipt["config_sha256"] == V.sha(V.path(run["config"]))
                  and receipt["source_manifest_sha256"] == protocol["source_manifest_sha256"]
                  and receipt["math_flags"] == V.MATH and receipt["environment"] == E.POLICY_ENV
                  and receipt["output_redirection_count"] == 1
                  and receipt["contract_verification_sha256"] == digest(contracts[run["name"]])
                  and receipt["oracle"] == run["cell"].startswith(("a1", "a2")), "Original row producer/strict contract changed")
        directory = V.path(run["output"])
        V.require(receipt["paired_training_binding_sha256"] == V.sha(directory / "paired_binding_rank0.json")
                  and receipt["old_source_binding_sha256"] == V.sha(directory / "xag_binding.json")
                  and receipt["adaptation"]["scoring_cost_model_packet_functions_changed"] is False,
                  "New training/source provenance or bounded evaluator adaptation changed")
        seed = row["per_seed"]["3407"]
        V.require(seed["xag_binding"]["checkpoint_sha256"] == V.sha(checkpoint)
                  and seed["xag_binding"]["config_sha256"] == V.sha(V.path(run["config"])), "Row model/config SHA differs")
        for source in (filename, receipt_path): checked(source, records=records)
        products[run["cell"]] = {"run": run, "row": row, "path": filename, "receipt": receipt}
    return products


class Labels:
    """Independent raw51D original-meta reader; no dataset/model/FK import."""
    def __init__(self, filename):
        import numpy as np
        with filename.open("rb") as stream:
            V.require(struct.unpack("<i", stream.read(4))[0] == 51, "Original meta header differs")
        self.dtype = np.dtype([("pose", "<f8", (51,)), ("m0", "u1"), ("m1", "u1")])
        V.require((filename.stat().st_size - 4) % self.dtype.itemsize == 0, "Original meta record size differs")
        self.array = np.memmap(filename, mode="r", offset=4, dtype=self.dtype)

    def get(self, indices):
        import numpy as np
        values = self.array[np.asarray(indices, np.int64)]
        V.require(np.all(values["m0"] == 4) and np.all(values["m1"] == 13), "Original meta markers differ")
        result = values["pose"].astype(np.float32)
        V.require(np.isfinite(result).all(), "Nonfinite original GT51")
        return result


def independent_context(protocol, cfg, records):
    import numpy as np
    frozen(protocol, DECODER)
    checked(DECODER, DECODER_SHA, records)
    spec = importlib.util.spec_from_file_location("_xag_paired_independent_numpy_mano", DECODER)
    decoder = importlib.util.module_from_spec(spec); spec.loader.exec_module(decoder)
    root = Path(cfg["DATA"]["ROOT"])
    asset = Path(cfg["MANO"]["NPZ"])
    frozen(protocol, asset); checked(asset, records=records)
    sources = V.read(V.path(protocol["source_manifest_path"]))
    items = {}
    for seq, count in COUNTS.items():
        aux_path = root / "val" / f"{seq}_aux.npz"
        meta_path = root / "val" / f"{seq}.meta"
        for filename in (aux_path, meta_path):
            key = relative(filename)
            entry = sources["inputs"].get(key, sources["inputs"].get(str(filename)))
            V.require(entry is not None, "GT labels/context are not in original source-input manifest")
            checked(filename, entry["sha256"], records)
        with np.load(aux_path, allow_pickle=False) as values:
            aux = {key: values[key] for key in values.files}
        ends, ids = T.grid(aux)
        V.require(len(ends) == count, "Independent GT endpoint denominator changed")
        labels = Labels(meta_path)
        offsets_path = ROOT / "outputs/xag/sources" / seq / "offsets.npy"
        offsets = np.load(offsets_path, allow_pickle=False)
        items[seq] = {"aux": aux, "ends": ends, "ids": ids, "labels": labels, "target": labels.get(ends),
                      "offsets": offsets}
    return decoder, asset, items


def validate_arrays(arrays, items, cfg, *, rich):
    import numpy as np
    fields = ("incoming", "prediction", "target", "end_ms", "run_id") if rich else ("incoming", "prediction")
    V.require(set(arrays) == {seq + "/" + key for seq in COUNTS for key in fields}, "Full trace array inventory changed")
    rng = np.random.default_rng(0)
    initializations = feedbacks = 0
    for seq, count in COUNTS.items():
        item = items[seq]
        incoming, prediction = arrays[seq + "/incoming"], arrays[seq + "/prediction"]
        V.require(incoming.shape == prediction.shape == (count, 51) and incoming.dtype == prediction.dtype == np.dtype("float32")
                  and np.isfinite(incoming).all() and np.isfinite(prediction).all(), "Full finite float32 recursive arrays required")
        if rich:
            target = arrays[seq + "/target"]
            V.require(target.shape == (count, 51) and target.dtype == np.dtype("float32")
                      and target.tobytes() == item["target"].tobytes()
                      and arrays[seq + "/end_ms"].dtype == arrays[seq + "/run_id"].dtype == np.dtype("int64")
                      and np.array_equal(arrays[seq + "/end_ms"], item["ends"])
                      and np.array_equal(arrays[seq + "/run_id"], item["ids"]), "Trace GT/clock differs from independent original labels")
        offset = 0
        for a, b in item["aux"]["valid_runs_ms"]:
            n = len(np.arange(int(a) + 49, int(b), 50))
            if not n: continue
            noise = np.zeros(51, np.float32)
            for sl, name in ((slice(0, 3), "PREV_NOISE_T"), (slice(3, 6), "PREV_NOISE_R"), (slice(6, 51), "PREV_NOISE_POSE")):
                noise[sl] = rng.standard_normal(sl.stop - sl.start) * float(cfg.get("TRACK", {}).get(name, 0.))
            expected = item["labels"].get([int(a)])[0] + noise
            V.require(incoming[offset].tobytes() == expected.tobytes(), "Original noisy initialization differs")
            V.require(incoming[offset + 1:offset + n].tobytes() == prediction[offset:offset + n - 1].tobytes(), "Own-loop prediction feedback differs")
            initializations += 1; feedbacks += n - 1; offset += n
        V.require(offset == count, "Incomplete recursive valid-run census")
    V.require(initializations == 42 and feedbacks == 2548, "Original42 initialization/2548 feedback contract differs")
    return {"noisy_initializations": initializations, "ownloop_feedback_links": feedbacks, "full_endpoints": 2590}


def read_arrays(filename):
    import numpy as np
    with np.load(filename, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def validate_packets(rows, items):
    V.require(len(rows) == 2590, "Every packet must retain its source/event identity")
    comparable = []; offset = 0
    for seq, count in COUNTS.items():
        item = items[seq]
        for index in range(count):
            row = rows[offset + index]; end = int(item["ends"][index]); start = end - 49
            n = int(item["offsets"][end + 1]) - int(item["offsets"][start])
            V.require(row["index"] == offset + index and row["seq"] == seq and row["sequence_index"] == index
                      and row["run_id"] == int(item["ids"][index]) and row["end_ms"] == end
                      and row["window_ms"] == 50 and row["n_events"] == n, "Packet clock/event denominator changed")
            V.require(set(row["source_field_sha256"]) == set(V.SOURCE_FIELDS), "Six source-field hashes required")
            for value in [row[key] for key in ("events_sha256", "source_sha256", "canonical_common_packet_sha256", "actual_full_packet_sha256")] + list(row["source_field_sha256"].values()):
                V.require(isinstance(value, str) and bool(re.fullmatch("[a-f0-9]{64}", value)), "Missing actual event/source/packet fingerprint")
            comparable.append({key: row[key] for key in ("index", "seq", "sequence_index", "run_id", "end_ms", "window_ms", "n_events",
                                                         "events_sha256", "source_sha256", "source_field_sha256")})
        offset += count
    return comparable


def sequence_metrics_match(trace, row):
    sequence = trace["per_sequence"]
    V.require([value["seq"] for value in sequence] == list(COUNTS)
              and all(value["n_frames"] == COUNTS[value["seq"]] for value in sequence), "Trace sequence order/denominator changed")
    seed = row["per_seed"]["3407"]
    for value in sequence:
        V.require(all(value[key] == seed["per_seq"][value["seq"]][key] for key in METRICS), "Trace scores differ from original generated main row")
    for block, selected in (("overall", sequence), ("global", sequence[:1]), ("local", sequence[1:])):
        n = sum(value["n_frames"] for value in selected)
        V.require(all(sum(value[key] * value["n_frames"] for value in selected) / n == seed[block][key]
                      for key in METRICS), "Original endpoint-weighted aggregation changed")


def load_traces(protocol_path, protocol, products, contracts, items, cfg, records):
    reference_arrays = reference_packets = None
    decoded_inputs = {}; checks = []
    cache = T.source_cache_binding(protocol, cfg)
    records.update(cache["files_sha256"])
    for cell in CELLS:
        product = products[cell]; run = product["run"]; directory = V.path(run["output"])
        first_arrays = first_packets = None
        for repeat in (1, 2):
            saved = directory / f"strict_trace{repeat}"
            trace_path, arrays_path, packets_path = saved / "report.json", saved / "states.npz", saved / "packets.jsonl"
            report = V.read(trace_path)
            V.require(report["status"] == "COMPLETE" and report["name"] == run["name"] and report["cell"] == cell
                      and report["seed"] == 3407 and report["step"] == 6000 and report["n_endpoints"] == 2590
                      and report["checkpoint_sha256"] == contracts[run["name"]]["checkpoint_sha256"]
                      and report["config_sha256"] == V.sha(V.path(run["config"]))
                      and report["protocol_sha256"] == V.sha(protocol_path)
                      and report["source_sha256"] == V.sha(ROOT / "tools/xag_paired_20261018/trace.py")
                      and report["original_evaluator_sha256"] == V.sha(ROOT / "semkine/eval_track.py")
                      and report["states_sha256"] == V.sha(arrays_path) and report["packet_journal_sha256"] == V.sha(packets_path)
                      and report["math_flags"] == V.MATH and report["source_cache_binding"] == cache
                      and report["paired_training_binding_sha256"] == V.sha(directory / "paired_binding_rank0.json")
                      and report["old_source_binding_sha256"] == V.sha(directory / "xag_binding.json")
                      and report["contract_verification_sha256"] == digest(contracts[run["name"]])
                      and report["runtime_policy"] == "XAG_PAIRED_C1_STRICT_20261018"
                      and report["oracle"] == cell.startswith(("a1", "a2")), "Trace producer/source/checkpoint/strict binding changed")
            sequence_metrics_match(report, product["row"])
            arrays = read_arrays(arrays_path)
            loops = validate_arrays(arrays, items, cfg, rich=True)
            packets = [json.loads(line) for line in packets_path.read_text().splitlines()]
            comparable = validate_packets(packets, items)
            if repeat == 1:
                first_arrays, first_packets = arrays, packets
                decoded_inputs[cell] = arrays
                if reference_arrays is None:
                    reference_arrays, reference_packets = arrays, comparable
                else:
                    V.require(comparable == reference_packets, "Cross-arm events/source/clock differ")
                    for seq in COUNTS:
                        for key in ("target", "end_ms", "run_id"):
                            V.require(arrays[seq + "/" + key].tobytes() == reference_arrays[seq + "/" + key].tobytes(),
                                      "Cross-arm GT/clock differ")
            else:
                V.require(packets == first_packets and set(arrays) == set(first_arrays)
                          and all(arrays[key].dtype == first_arrays[key].dtype and arrays[key].shape == first_arrays[key].shape
                                  and arrays[key].tobytes() == first_arrays[key].tobytes() for key in arrays),
                          "Fresh-load strict traces do not repeat every input/prediction/source packet exactly")
            checks.append({"cell": cell, "repeat": repeat, **loops})
            for filename in (trace_path, arrays_path, packets_path): checked(filename, records=records)
    return decoded_inputs, checks


def completed_baseline(protocol, items, cfg, records):
    active = V.read(V.path(protocol["baseline_identity_path"]))
    row_path = V.path(active["main_row"]); row = V.read(row_path)
    receipt_path = row_path.with_name(row_path.stem + "_receipt.json")
    receipt = V.read(receipt_path)
    directory = V.path(protocol["reference_baseline_output"])
    trace_path, arrays_path = directory / "strict_trace/report.json", directory / "strict_trace/states.npz"
    for filename in (row_path, receipt_path, trace_path, arrays_path): frozen(protocol, filename); checked(filename, records=records)
    trace = V.read(trace_path)
    V.require(set(row["per_seed"]) == {"3407"} and row["per_seed"]["3407"]["step"] == 6000
              and row["per_seed"]["3407"]["n_frames"] == 2590 and row["per_seed"]["3407"]["sequences"] == list(COUNTS)
              and trace["status"] == "COMPLETE" and trace["n_endpoints"] == 2590
              and trace["checkpoint_sha256"] == active["checkpoint_sha256"] and trace["states_sha256"] == V.sha(arrays_path)
              and trace["config_sha256"] == V.sha(V.path(active["config"]))
              and trace["source_sha256"] == V.sha(ROOT / "tools/xyz_c1_repro/trace.py"), "Completed canonical C1 trace/row changed")
    V.require(receipt["status"] == "COMPLETE" and receipt["runtime_policy"] == "XYZ_C1_REPRO_20261018_STRICT"
              and receipt["row_sha256"] == V.sha(row_path) and V.path(receipt["row"]).resolve() == row_path.resolve()
              and receipt["seed"] == 3407 and receipt["step"] == 6000
              and receipt["wrapper_sha256"] == V.sha(ROOT / "tools/xyz_c1_repro/evaluate.py")
              and receipt["original_evaluator_sha256"] == V.sha(ROOT / "tools/make_s36_row.py")
              and receipt["original_reporter_sha256"] == V.sha(ROOT / "tools/report_table.py")
              and all(receipt[key] == value for key, value in V.MATH.items()), "Completed canonical C1 original strict row receipt changed")
    for block in ("overall", "global", "local"):
        V.require(row["two_seed_mean"][block] == row["per_seed"]["3407"][block]
                  and all(math.isfinite(row["two_seed_mean"][block][key]) for key in METRICS), "Canonical C1 single-seed metrics differ")
    sequence_metrics_match(trace, row)
    arrays = read_arrays(arrays_path)
    loops = validate_arrays(arrays, items, cfg, rich=False)
    return row_path, row, arrays, loops


def independent_metrics(decoder, asset, items, arrays_by_cell, rows_by_cell):
    import numpy as np
    mano = decoder.NumpyMano(asset)
    contexts = {seq: mano.context(item["aux"]["betas"]) for seq, item in items.items()}
    gt = {}
    for seq, item in items.items():
        values = [mano.decode(item["target"][start:start + CHUNK], contexts[seq]) for start in range(0, COUNTS[seq], CHUNK)]
        gt[seq] = tuple(np.concatenate([value[index] for value in values]) for index in (0, 1))
    comparisons = []
    for cell, arrays in arrays_by_cell.items():
        per_seq = {}; all_values = {key: [] for key in METRICS}
        for seq, count in COUNTS.items():
            values = {key: [] for key in METRICS}
            gj, gv = gt[seq]
            for start in range(0, count, CHUNK):
                stop = min(start + CHUNK, count)
                pj, pv = mano.decode(arrays[seq + "/prediction"][start:stop], contexts[seq])
                norm = lambda a: np.sqrt(np.sum(a * a, axis=-1)).mean(axis=-1) * 1000.
                frame = {"mpjpe_abs_mm": norm(pj - gj[start:stop]), "mpvpe_abs_mm": norm(pv - gv[start:stop]),
                         "mpjpe_ra_mm": norm((pj - pj[:, :1]) - (gj[start:stop] - gj[start:stop, :1])),
                         "mpvpe_ra_mm": norm((pv - pv[:, :1]) - (gv[start:stop] - gv[start:stop, :1]))}
                for key in METRICS: values[key].append(frame[key])
            values = {key: np.concatenate(value) for key, value in values.items()}
            V.require(all(np.isfinite(value).all() for value in values.values()), "Nonfinite independent MANO scores")
            per_seq[seq] = {key: float(value.mean()) for key, value in values.items()}
            for key, value in values.items(): all_values[key].append(value)
        aggregate = {"global": per_seq["zgz_global"], "local": per_seq["zgz_local"],
                     "overall": {key: float(np.concatenate(value).mean()) for key, value in all_values.items()}}
        for block in ("local", "global", "overall"):
            for metric in METRICS:
                actual, expected = aggregate[block][metric], rows_by_cell[cell]["two_seed_mean"][block][metric]
                difference = actual - expected
                V.require(abs(difference) <= TOL_MM and f"{actual:.2f}" == f"{expected:.2f}", "Independent NumPy metric differs: " + cell + "/" + block + "/" + metric)
                comparisons.append({"cell": cell, "block": block, "metric": metric, "numpy_mm": actual,
                                    "original_row_mm": expected, "difference_mm": difference})
    V.require(len(comparisons) == 72, "Exactly six models x twelve independent metric checks required")
    return {"status": "PASS", "decoder_sha256": DECODER_SHA, "models": 6, "prediction_endpoints": 6 * 2590,
            "metric_checks": len(comparisons), "tolerance_mm": TOL_MM, "all_two_decimal_strings_same": True,
            "max_absolute_difference_mm": max(abs(value["difference_mm"]) for value in comparisons), "comparisons": comparisons,
            "scope": "Independent CPU float64 NumPy MANO decoding of stored51D predictions and original-meta GT; no event-network inference or score selection",
            "definitions": "MPJPE subtracts joint0; MPVPE subtracts vertex0. local/global are sequence classes. RA is translation alignment, not Procrustes."}


def validate(protocol_path):
    protocol = V.verify_protocol(protocol_path)
    for filename in (Path(__file__), ROOT / "tools/report_table.py", S39, DECODER): frozen(protocol, filename)
    terminal_state(protocol_path, protocol)
    old_pointer = previous_pointer(protocol)
    records = {name: value for name, value in protocol["file_sha256"].items()}
    checked(protocol_path, records=records)
    contracts, matrix = completed_training(protocol_path, protocol, records)
    products = load_rows(protocol_path, protocol, contracts, records)
    cfg = V.load_recipe(protocol["runs"][0])
    decoder, asset, items = independent_context(protocol, cfg, records)
    arrays, trace_checks = load_traces(protocol_path, protocol, products, contracts, items, cfg, records)
    baseline_path, baseline_row, baseline_arrays, baseline_loops = completed_baseline(protocol, items, cfg, records)
    arrays = {"a0f0": baseline_arrays, **arrays}
    rows = {"a0f0": baseline_row, **{cell: product["row"] for cell, product in products.items()}}
    metric_audit = independent_metrics(decoder, asset, items, arrays, rows)
    V.verify_protocol(protocol_path)
    V.require(previous_pointer(protocol) == old_pointer, "Previous metrics pointer changed during validation")
    for filename, expected in records.items(): checked(filename, expected)
    audit = {"status": "PASS", "validated_at_utc": now(), "protocol_sha256": V.sha(protocol_path), "source_sha256": V.sha(__file__),
             "model_seed": 3407, "fresh_optimizer_steps": 6000, "new_training_arms": 5, "additional_training_replays": 0,
             "training_audit_sha256": V.sha(OUT / "paired_training_audit.json"), "matrix": matrix["checks"],
             "trace_checks": trace_checks, "baseline_loop_checks": baseline_loops, "numpy_metric_checks": 72,
             "cross_arm_comparison": "events, six source fields, source row/ms, GT targets and clocks only; each arm retains its own incoming states and predictions",
             "within_arm_repeat": "Every array byte and all2590 packet record fields match between two fresh strict evaluation loads",
             "source_files_sha256": records, "previous_current_metrics_sha256": hashlib.sha256(old_pointer).hexdigest(),
             "baseline_identity_sha256": protocol["baseline_identity_sha256"], "source_manifest_sha256": protocol["source_manifest_sha256"],
             "limits": "Single model seed3407; descriptive paired comparison, no cross-seed reliability or generalization claim. A1/A2 use GT geometric proxy sources and are not deployable."}
    return {"protocol": protocol, "products": products, "baseline_path": baseline_path, "old_pointer": old_pointer,
            "validation": audit, "metric_audit": metric_audit}


def table_sources(bundle):
    return [("S39", S39, "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"),
            ("C1_a0f0", bundle["baseline_path"], "XYZ C1：A0F0确定性配对基线（seed3407）"),
            *[(cell, bundle["products"][cell]["path"], label) for cell, label in (
                ("a0f1", "XAG：原关联＋增强GNN（配对重训，seed3407）"),
                ("a1f0", "XAG：部位GT对应＋原GNN（oracle，配对重训，seed3407）"),
                ("a1f1", "XAG：部位GT对应＋增强GNN（oracle，配对重训，seed3407）"),
                ("a2f0", "XAG：表面GT对应＋原GNN（oracle，配对重训，seed3407）"),
                ("a2f1", "XAG：表面GT对应＋增强GNN（oracle，配对重训，seed3407）"))]]


def generate_table(sources, release):
    aliases = release / "table_sources"; aliases.mkdir(exist_ok=False)
    command = [sys.executable, str(ROOT / "tools/report_table.py")]
    labels = []; manifest = {}
    for key, filename, label in sources:
        alias = aliases / f"{key}_main_row.json"
        alias.symlink_to(filename.resolve())
        V.require(alias.resolve() == filename.resolve(), "Table source alias differs")
        argument = str(aliases / key)
        command.append(argument); labels.extend(["--label", argument + "=" + label])
        manifest[key] = {"main_row": relative(filename), "sha256": V.sha(filename), "label": label}
    command.extend(labels)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    V.require(len(lines) == 10 and all(len(line.split("|")[1:-1]) == 9 for line in lines), "Original reporter must emit eight data rows and exactly nine columns")
    write_new(release / "main_table.md", result.stdout.encode())
    write_new(release / "table_command.json", command)
    write_new(release / "table_command.txt", (shlex.join(command) + "\n").encode())
    write_new(release / "table_sources.json", {"original_reporter_sha256": V.sha(ROOT / "tools/report_table.py"), "sources": manifest,
              "PCA6": "Original report_table.py published constant reference; not evaluated again in this study"})
    return command


def pointer_candidate(bundle, release, validation_sha):
    rows = {"a0f0": {"main_row": relative(bundle["baseline_path"]), "sha256": V.sha(bundle["baseline_path"]), "oracle": False}}
    for cell in CELLS:
        product = bundle["products"][cell]
        rows[cell] = {"main_row": relative(product["path"]), "sha256": V.sha(product["path"]),
                      "oracle": cell.startswith(("a1", "a2")), "run": product["run"]["name"]}
    protocol = bundle["protocol"]
    return {"status": "ACTIVE_PAIRED_STRICT_XAG_METRICS", "model_seed": 3407, "step": 6000, "rows": rows,
            "completion_receipt": relative(release / "validation.json"), "completion_receipt_sha256": validation_sha,
            "table": relative(release / "main_table.md"), "table_command": relative(release / "table_command.txt"),
            "release": relative(release / "release.json"), "protocol_sha256": bundle["validation"]["protocol_sha256"],
            "baseline_identity": protocol["baseline_identity_path"], "baseline_identity_sha256": protocol["baseline_identity_sha256"],
            "paired_reference_within_matrix": "a0f0", "a0f0_source": "Completed predesignated xyz_c1_stable_s3407 canonical baseline; no new A0F0 training or score selection",
            "training_identity": "Five fresh6000 single-seed3407 legacy-None strict-runtime arms, actual two-rank common inputs and shared initialization verified against canonical C1",
            "comparison_boundary": "Single-seed descriptive paired comparison; no cross-seed reliability or stable-superiority claim. A1/A2 use GT geometric proxy sources, not deployable measured event origins. PCA6 and S39 remain historical references.",
            "previous_metrics_archive": relative(release / "previous_current_metrics.json"),
            "previous_metrics_sha256": hashlib.sha256(bundle["old_pointer"]).hexdigest(), "preserved_old_main_rows": True}


def atomic_pointer_switch(filename, old_data, new_data):
    """Prepare same-directory bytes and compare old identity before os.replace."""
    V.require(filename == POINTER and filename.is_file() and not filename.is_symlink()
              and filename.read_bytes() == old_data, "Current metrics pointer changed before atomic switch")
    descriptor, name = tempfile.mkstemp(prefix=".paired_metrics_", suffix=".tmp", dir=filename.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, filename.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(new_data); stream.flush(); os.fsync(stream.fileno())
        V.require(filename.is_file() and not filename.is_symlink() and filename.read_bytes() == old_data,
                  "Current metrics pointer changed while preparing atomic switch")
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def publish(bundle, protocol_path):
    release = OUT / "release"
    V.require(not release.exists() and not release.is_symlink(), "Preserve existing release and all publication evidence")
    V.require(previous_pointer(bundle["protocol"]) == bundle["old_pointer"], "Previous metrics pointer changed after validation")
    release.mkdir(exist_ok=False)
    write_new(release / "previous_current_metrics.json", bundle["old_pointer"])
    write_new(release / "validation.json", bundle["validation"])
    write_new(release / "numpy_metric_audit.json", bundle["metric_audit"])
    command = generate_table(table_sources(bundle), release)
    notes = ("新五格均为同一seed3407、fresh6000、2×512、legacy-None采样与严格数值策略训练；"
             "实际全部两rank索引及共有数据字段与已完成C1基线一致，三F+新增39个参数张量初值一致。"
             "A0F0直接引用预先指定且完整重放通过的C1基线，没有另训、选点或挑seed。\n"
             "本表只作单seed配对描述，不据此宣称跨seed可靠性、稳定优势或已经唯一确定退化主因。"
             "A1/A2读取GT网格几何伪对应，属于oracle诊断；不是事件物理来源真值，不可部署。"
             "PCA6为原报告器常数，S39为保留的历史三seed参考，本轮未重新训练或评价。\n"
             "local/global为zgz两类序列；MPJPE减joint0腕点，MPVPE减vertex0；RA仅平移对齐，不做Procrustes。"
             "完整2590端点评分；成本由原生成器在严格运行时重新测量整个forward_packet。\n"
             "每臂两次独立加载全序列trace的incoming/pred/全部字段逐位重复；跨臂只比较事件、GT来源、target与时钟，保留各自闭环前态。"
             "独立NumPy MANO仅从已存51D数组复核72项，不新增事件网络推理。旧pointer原字节与旧主行均保留。\n")
    write_new(release / "main_table_notes.md", notes.encode())
    candidate = pointer_candidate(bundle, release, V.sha(release / "validation.json"))
    candidate_bytes = json_bytes(candidate)
    write_new(release / "current_metrics_candidate.json", candidate_bytes)
    files = {relative(filename): V.sha(filename) for filename in release.iterdir() if filename.is_file() and not filename.is_symlink()}
    report = {"status": "COMPLETE", "completed_at_utc": now(), "protocol_sha256": V.sha(protocol_path), "source_sha256": V.sha(__file__),
              "files_sha256": files, "table_command": command, "data_rows": 8, "columns": 9,
              "paired_cells": ["a0f0", *CELLS], "numpy_metric_checks": 72, "new_training_runs": 5, "additional_training_replays": 0,
              "active_pointer_transition": {"path": relative(POINTER), "old_sha256": hashlib.sha256(bundle["old_pointer"]).hexdigest(),
                                            "required_new_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                                            "activation_rule": "Release artifacts are complete; this release is active only when current_metrics.json has required_new_sha256. Pointer replacement is the final operation."},
              "scope": "All formal training evidence, original strict rows/costs, repeated full traces and independent NumPy scores verified before pointer replacement; canonical C1 baseline pointer unchanged"}
    write_new(release / "release.json", report)
    # No validation is deferred until after changing the active pointer.
    V.verify_protocol(protocol_path)
    terminal_state(protocol_path, bundle["protocol"])
    for filename, expected in bundle["validation"]["source_files_sha256"].items(): checked(filename, expected)
    for filename, expected in files.items(): checked(filename, expected)
    V.require(previous_pointer(bundle["protocol"]) == bundle["old_pointer"], "Previous metrics pointer changed while preparing release")
    atomic_pointer_switch(POINTER, bundle["old_pointer"], candidate_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate all physical final products on CPU; do not create release or change pointer")
    args = parser.parse_args(argv)
    # No CUDA or mathematical training policy is changed in parent processes.
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    sys.dont_write_bytecode = True
    protocol_path = V.path(args.protocol).resolve()
    bundle = validate(protocol_path)
    if args.check_only:
        print(json.dumps({"status": "PASS_CHECK_ONLY", "new_arms": 5, "numpy_metric_checks": 72, "published": False}))
        return
    publish(bundle, protocol_path)


if __name__ == "__main__":
    main()

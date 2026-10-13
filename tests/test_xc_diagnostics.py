"""Guard timestamps and failure retention in the research reporting pipeline."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.xc import diagnostics as D
from tools.xc import finalize as F


def test_only_descriptive_wrong_time_nonfinite_does_not_reject_true_time_effect():
    payload = {"numeric_valid": False, "invalid_numeric_paths": [
        "$.arms.g0t1.time_shift_2_control.rows[0].mpjpe_ra_mm[4]",
        "$.arms.g1t1.time_shift_2_control.summary.all.large.recovery_auc_ra_mm"]}
    assert F.mandatory_diagnostic_numeric(payload)
    payload["invalid_numeric_paths"].append("$.arms.g1t1.rows[0].mpjpe_ra_mm[4]")
    assert not F.mandatory_diagnostic_numeric(payload)
    assert not F.mandatory_diagnostic_numeric({"numeric_valid": False, "invalid_numeric_paths": []})


def test_recovery_rejects_one_bad_seed_despite_large_average_gain():
    payloads = {}
    for seed, candidate_auc in zip(F.SEEDS, (1.0, 1.0, 10.1)):
        payloads[seed] = {"arms": {arm: {"summary": {"all": {
            condition: {"recovery_auc_ra_mm": auc} for condition in ("natural", "large")}}}
            for arm, auc in (("g0t0", 10.0), ("g1t1", candidate_auc))}}
    assert F.recovery(payloads, "g0t0", "g1t1")["status"] == "fail"
    payloads[3409]["arms"]["g1t1"]["summary"]["all"]["natural"]["recovery_auc_ra_mm"] = None
    result = F.recovery(payloads, "g0t0", "g1t1")
    assert not result["checks"]["seed3409_natural_finite"]


class Tracker:
    def __init__(self, fail=False):
        self.fail = fail
        self.history = None
        self.calls = []
        self.route_stats = {}

    def reset_time_history(self):
        self.history = None

    def set_hand_context(self, *args):
        pass

    def set_time_history(self, past, interval, query_start_offset_s=0):
        self.history = (past.clone(), interval, query_start_offset_s)

    def forward_packet(self, packet):
        assert not packet.target.any()
        self.calls.append((packet.prev.clone(), self.history))
        if self.fail:
            raise RuntimeError("measured mock failure")
        prediction = packet.prev.clone()
        prediction[:, 0] += .001
        return prediction


def fixture(monkeypatch):
    gt = np.zeros((550, 51), np.float32)
    gt[:, 0] = np.arange(550) * .00001
    source = [{"previous_end_ms": 0 if i == 0 else 49 + 50 * (i - 1),
               "end_ms": 49 + 50 * i, "run_id": 0, "prev": np.zeros(51, np.float32)} for i in range(10)]
    data = {"gt": gt, "events": None, "offsets": None, "tsub": None}
    monkeypatch.setattr(D.D, "context", lambda *_: (torch.zeros(1, 10), torch.eye(3).unsqueeze(0)))
    monkeypatch.setattr(D.D.ET, "_window_events", lambda *_: np.zeros((3, 5), np.float32))
    monkeypatch.setattr(D.D.ET, "make_eval_packet", lambda _ev, prev, *_: SimpleNamespace(prev=prev, target=torch.zeros_like(prev)))

    def measure(_mano, predicted, truth, *_):
        pj = np.asarray(predicted, np.float32)[:, None, :3]
        gj = np.asarray(truth, np.float32)[:, None, :3]
        errors = np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000
        return errors, errors, pj, gj

    monkeypatch.setattr(D.D, "measure", measure)
    return {"zgz_local": data}, {"zgz_local": source}, {"zgz_local": [0]}


def test_replay_uses_49_then_50ms_own_history_and_resets_all_starts(monkeypatch):
    sequences, traces, selection = fixture(monkeypatch)
    model = Tracker()
    rows = D.replay(model, None, sequences, traces, selection, torch.device("cpu"))
    assert len(rows) == 4 and len(model.calls) == 40
    for begin in range(0, 40, 10):
        assert model.calls[begin][1] is None
        first_prior, first_history = model.calls[begin + 1]
        assert first_history[1:] == pytest.approx((.049, .001))
        assert torch.equal(first_history[0], model.calls[begin][0])
        assert first_prior[0, 0] == pytest.approx(float(first_history[0][0, 0]) + .001)
        for _, history in model.calls[begin + 2:begin + 10]:
            assert history[1:] == pytest.approx((.050, .001))
    assert model.history is None
    assert all(len(row["mpjpe_ra_mm"]) == 11 and row["trajectory_valid"] for row in rows)


def test_prediction_exception_retains_each_frame_and_each_start(monkeypatch, tmp_path):
    sequences, traces, selection = fixture(monkeypatch)
    rows = D.replay(Tracker(fail=True), None, sequences, traces, selection, torch.device("cpu"))
    assert len(rows) == 4
    assert all(len(row["mpjpe_ra_mm"]) == 11 and not row["trajectory_valid"] for row in rows)
    assert all(np.isnan(row["mpjpe_ra_mm"][1:]).all() for row in rows)
    target = tmp_path / "failure.json"
    D.save_payload(target, {"rows": rows, "summary": D.D.summarize(rows)})
    import json
    output = json.loads(target.read_text())
    assert output["numeric_valid"] is False
    assert output["invalid_numeric_count"] > 0
    assert len(output["rows"]) == 4 and len(output["rows"][0]["mpjpe_ra_mm"]) == 11

"""Fixed-final-checkpoint SELECT evaluation, independent four-arm rollouts.

The frozen training implementation is imported read-only.  This evaluator
does not select checkpoints, tune parameters, train, or read zgz.  Every arm
finishes the current forward before the separate supervision reader opens
labels.  Recorded confidence/entropy quantities are descriptive scores.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .data import PUBLIC_INITIAL_STATE, XJRunLanes, XJSupervisionReader, stack_inputs
from .geometry import GeometryConfig, JointGeometry
from .model import JointLocalConfig, JointLocalOperator, fixed_reference_joint_solve
from .train import (
    advance_chain, camera_valid, configure_device, dump, native_lnes_ms, sha,
    source_record_weights, supervision_digest, sync, tensor_digest,
)


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('primary', 'spatial', 'global', 'derived')
LOCAL_ARMS = ('primary', 'spatial', 'derived')
METRICS = ('joint_abs_mm', 'root_abs_mm', 'vertex_abs_mm',
           'joint_j0_mm', 'vertex_v0_mm', 'vertex_j0_mm')


def _select_rows(values, indices):
    """Slice a prepared-geometry/auxiliary tensor table without reusing states."""
    selected = torch.as_tensor(indices, dtype=torch.long, device=next(iter(values.values())).device)
    return {key: value.index_select(0, selected) for key, value in values.items()}


def _aux_row(aux, index):
    return {key: value[index:index + 1] for key, value in aux.items()}


def verify_evaluation_identity(protocol_path, freeze_path, *, person=None, backend=None):
    """Fail closed before checkpoint loading or any real input reader exists."""
    protocol_path, freeze_path = Path(protocol_path).resolve(), Path(freeze_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    frozen = json.loads(freeze_path.read_text())
    if protocol.get('schema') != 'xvm_joint_select_eval_protocol_v1':
        raise RuntimeError('Unexpected SELECT evaluation protocol')
    if frozen.get('schema') != 'xvm_joint_select_eval_sourcefreeze_v1':
        raise RuntimeError('Unexpected SELECT execution-freeze schema')
    if (Path(frozen['protocol']['path']).resolve() != protocol_path
            or sha(protocol_path) != frozen['protocol']['sha256']):
        raise RuntimeError('Evaluation protocol identity differs')
    for relative, digest in frozen['files'].items():
        path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        if sha(path) != digest:
            raise RuntimeError('Changed evaluation executable: ' + str(relative))
    for name, entry in frozen['identities'].items():
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Changed evaluation identity: ' + name)
    for name in ('parent_protocol', 'training_sourcefreeze', 'config', 'checkpoint'):
        entry = protocol[name]
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Changed fixed evaluation input: ' + name)
    from .control import verify
    verify(protocol['training_sourcefreeze']['path'], protocol['config']['path'])
    config = json.loads(Path(protocol['config']['path']).read_text())
    complete_path = Path(protocol['config']['path']).parent / 'train/complete.json'
    complete = json.loads(complete_path.read_text())
    if (complete.get('status') != 'COMPLETE'
            or complete.get('updates') != protocol['checkpoint']['updates']
            or complete['checkpoint']['sha256'] != protocol['checkpoint']['sha256']
            or Path(complete['checkpoint']['path']).resolve() != Path(protocol['checkpoint']['path']).resolve()):
        raise RuntimeError('Only the prespecified completed final checkpoint may be evaluated')
    if person is not None and (person not in protocol['subjects'] or person == 'zgz'):
        raise RuntimeError('Only the two SELECT persons are in scope')
    if tuple(protocol['arms']) != ARMS:
        raise RuntimeError('Frozen evaluation arm order differs')
    execution = frozen.get('execution', {})
    chosen = backend or execution.get('geometry_backend', 'frozen')
    if chosen not in ('frozen', 'vectorized'):
        raise RuntimeError('Unknown evaluation geometry backend')
    if chosen != execution.get('geometry_backend', 'frozen'):
        raise RuntimeError('Geometry backend differs from this execution freeze')
    if chosen == 'vectorized':
        parity = execution.get('parity_receipt')
        if not parity:
            raise RuntimeError('Vectorized execution requires a frozen parity receipt')
        parity_path = Path(parity['path'] if isinstance(parity, dict) else parity)
        if isinstance(parity, dict) and sha(parity_path) != parity['sha256']:
            raise RuntimeError('Changed geometry parity receipt')
        parity_result = json.loads(parity_path.read_text())
        if parity_result.get('status') != 'PASS':
            raise RuntimeError('Vectorized geometry parity has not passed')
        parity_sources = {'frozen_geometry_sha256': ROOT / 'tools/xvm_joint/geometry.py',
                          'fast_geometry_sha256': ROOT / 'tools/xvm_joint/evaluation_geometry.py',
                          'local_model_sha256': ROOT / 'tools/xvm_joint/model.py'}
        if any(parity_result.get(key) != sha(path) for key, path in parity_sources.items()):
            raise RuntimeError('Geometry parity was measured with different source files')
        if parity_result.get('checkpoint_sha256') != protocol['checkpoint']['sha256']:
            raise RuntimeError('Geometry parity used a different final checkpoint')
    return protocol, frozen, config, complete, chosen


@torch.no_grad()
def project_selected(base, states, K_event, vertex_ids):
    """Beta0 projection of saved absolute states, with no target information."""
    vertices, _ = base._fk(states.float(), states.new_zeros((len(states), 10)))
    if not bool(torch.isfinite(vertices).all()) or bool((vertices[..., 2] <= 1e-4).any()):
        raise ArithmeticError('Saved absolute control state is camera-invalid')
    z = vertices[..., 2]
    uv = torch.stack((K_event[:, None, 0, 0] * vertices[..., 0] / z + K_event[:, None, 0, 2],
                      K_event[:, None, 1, 1] * vertices[..., 1] / z + K_event[:, None, 1, 2]), -1)
    return uv.gather(1, vertex_ids[..., None].expand(-1, -1, 2))


def derived_normal_means(previous_absolute_uv, reference_uv, current_absolute_uv, normal):
    """Use previous *absolute proposal*, not previous emitted corrected state."""
    a = ((previous_absolute_uv - reference_uv) * normal).sum(-1)
    d = ((current_absolute_uv - previous_absolute_uv) * normal).sum(-1)
    return torch.stack((a, d), -1)


@torch.no_grad()
def endpoint_metrics(base, predicted_states, target_state, beta_target):
    """Absolute and explicitly named J0/V0 metrics; all results are mm."""
    beta0 = predicted_states.new_zeros((len(predicted_states), 10))
    pv, pj = base._fk(predicted_states.float(), beta0)
    gv, gj = base._fk(target_state.float().reshape(1, 51), beta_target.float().reshape(1, 10))
    norm_mean = lambda value: value.norm(dim=-1).mean(-1) * 1000.
    result = {
        'joint_abs_mm': norm_mean(pj - gj),
        'root_abs_mm': (pj[:, 0] - gj[:, 0]).norm(dim=-1) * 1000.,
        'vertex_abs_mm': norm_mean(pv - gv),
        'joint_j0_mm': norm_mean((pj - pj[:, :1]) - (gj - gj[:, :1])),
        'vertex_v0_mm': norm_mean((pv - pv[:, :1]) - (gv - gv[:, :1])),
        'vertex_j0_mm': norm_mean((pv - pj[:, :1]) - (gv - gj[:, :1])),
    }
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ArithmeticError('Nonfinite endpoint metric; do not drop this frame')
    return {key: value.cpu().numpy().astype(np.float64) for key, value in result.items()}


def association_diagnostics(aux, source_ids, candidate_mask):
    """Source-wise conditional candidate entropy, unknown, and actual coverage."""
    ids = source_ids.reshape(-1)
    active = aux['active'].reshape(-1).bool()
    alpha = aux['association_mass'].reshape(-1) * aux['source_count'].reshape(-1)[0]
    unknown = aux['unknown_score'].reshape(-1)
    declared_sources = torch.unique(ids[ids >= 0])
    entropy, full_entropy, unknown_values, source_rows = [], [], [], []
    for source in declared_sources:
        mask = active & (ids == source)
        if not bool(mask.any()):
            source_rows.append(dict(source_id=int(source), active=False, active_candidates=0,
                                    conditional_candidate_entropy=None, known_mass=0., unknown_score=1.))
            continue
        probability = alpha[mask]
        known = probability.sum()
        null = unknown[mask][0]
        conditional = probability / known.clamp_min(1e-30)
        entropy.append(float(-(conditional * conditional.clamp_min(1e-30).log()).sum()))
        full_entropy.append(float(-(probability * probability.clamp_min(1e-30).log()).sum()
                                  - null * null.clamp_min(1e-30).log()))
        unknown_values.append(float(null))
        source_rows.append(dict(source_id=int(source), active=True, active_candidates=int(mask.sum()),
                                conditional_candidate_entropy=entropy[-1],
                                assignment_plus_unknown_entropy=full_entropy[-1],
                                known_mass=float(known), unknown_score=float(null)))
    active_sources = len(entropy)
    return dict(source_supports=int(len(declared_sources)), active_sources=active_sources,
                candidate_records=int(candidate_mask.sum()), active_records=int(active.sum()),
                candidate_conditional_entropy_mean=float(np.mean(entropy)) if entropy else None,
                assignment_plus_unknown_entropy_mean=float(np.mean(full_entropy)) if full_entropy else None,
                unknown_score_mean=float(np.mean(unknown_values)) if unknown_values else None,
                validity_score_mean=float(aux['validity_score'][aux['active']].mean()) if bool(active.any()) else None,
                source_coverage=active_sources / len(declared_sources) if len(declared_sources) else 0.,
                per_source=source_rows)


class FourArmRuntime:
    """All legal forwards first; no supervision object or target argument exists."""
    def __init__(self, base, local, geometry, device):
        self.base, self.local, self.geometry = base, local, geometry
        self.device = torch.device(device)
        self.spatial = JointLocalOperator(replace(local.config, motion_conditioned_assignment=False)).to(device)
        self.spatial.load_state_dict(local.state_dict(), strict=True)
        self.spatial.eval()
        self.local.eval()
        self.previous = self.absolute_previous = None

    @torch.no_grad()
    def forward(self, batch):
        if len(batch) != 1 or not batch[0].get('valid', False):
            raise ValueError('One real chronological recording lane is required')
        item = batch[0]
        arrays = stack_inputs(batch)
        device = self.device
        field0 = torch.from_numpy(arrays['field0']).to(device)
        field1 = torch.from_numpy(arrays['field1']).to(device)
        K = torch.from_numpy(arrays['K_source']).to(device)
        beta0 = torch.zeros((4, 10), device=device)
        initial = torch.from_numpy(arrays['initial_state']).to(device).expand(4, -1)
        if self.previous is None or item['reset']:
            previous, absolute_previous = initial.clone(), initial.clone()
        else:
            previous, absolute_previous = self.previous.clone(), self.absolute_previous.clone()
        image = native_lnes_ms(item['events'], item['start_ms'], item['cutoff_ms'])
        images = torch.from_numpy(image).to(device)[None].expand(4, -1, -1, -1)
        stamp = time.perf_counter()
        absolute_raw = self.base(images, previous, betas=beta0, camera_K=K.expand(4, -1, -1)).float()
        absolute_valid = camera_valid(self.base, absolute_raw, beta0)
        absolute = torch.where(absolute_valid[:, None], absolute_raw, previous)
        sync(device)
        global_seconds = time.perf_counter() - stamp
        local_indices = torch.tensor([0, 1, 3], device=device)
        stamp = time.perf_counter()
        prepared = self.geometry.prepare(field0.expand(3, -1, -1, -1), field1.expand(3, -1, -1, -1),
            previous.index_select(0, local_indices), absolute.index_select(0, local_indices),
            K.expand(3, -1, -1), both_exposed=torch.full((3,), bool(item['both_snapshots_valid']), device=device))
        sync(device)
        geometry_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        _, motion_delta, motion_aux = self.local(**_select_rows(prepared.model_kwargs, [0, 2]))
        _, spatial_delta, spatial_aux = self.spatial(**_select_rows(prepared.model_kwargs, [1]))
        primary_aux, derived_aux = _aux_row(motion_aux, 0), _aux_row(motion_aux, 1)
        previous_uv = project_selected(self.base, absolute_previous[3:4], prepared.K_event[2:3], prepared.vertex_ids[2:3])
        means_derived = derived_normal_means(previous_uv, prepared.uv0[2:3], prepared.uv1[2:3], derived_aux['normal'])
        _, derived_delta, _ = fixed_reference_joint_solve(
            means_derived, derived_aux['covariance_px'], prepared.B0[2:3], prepared.B1[2:3],
            derived_aux['normal'], prepared.nominal_delta[2:3], derived_aux['weights'],
            self.local.state_scale, start_damping=self.local.config.start_damping,
            end_damping=self.local.config.end_damping)
        deltas = torch.zeros_like(absolute)
        deltas[0] = motion_delta[0]
        deltas[1] = spatial_delta[0]
        deltas[3] = derived_delta[0]
        proposed = absolute + deltas
        proposed_valid = camera_valid(self.base, proposed, beta0)
        empty = len(item['events']) == 0
        emit = proposed_valid & ~torch.full_like(proposed_valid, empty)
        predictions = torch.where(emit[:, None], proposed, previous)
        if not bool(torch.isfinite(predictions).all()):
            raise ArithmeticError('Nonfinite emitted SELECT state')
        sync(device)
        local_seconds = time.perf_counter() - stamp
        # This same-support contrast is determined without labels. Its error
        # against the GT geometric proxy will only be scored after all4forward.
        primary_previous_uv = project_selected(self.base, absolute_previous[0:1],
                                                prepared.K_event[0:1], prepared.vertex_ids[0:1])
        primary_aA = ((primary_previous_uv - prepared.uv0[0:1]) * prepared.normal[0:1]).sum(-1)
        aux_rows = {'primary': primary_aux, 'spatial': spatial_aux, 'derived': derived_aux}
        diagnostics = {}
        for geometry_row, arm in enumerate(LOCAL_ARMS):
            arm_index = ARMS.index(arm)
            diagnostics[arm] = association_diagnostics(aux_rows[arm], prepared.source_ids[geometry_row:geometry_row+1],
                                                       prepared.candidate_mask[geometry_row:geometry_row+1])
            diagnostics[arm].update(absolute_camera_invalid=int(not absolute_valid[arm_index]),
                                    local_camera_invalid=int(not proposed_valid[arm_index]),
                                    empty_fallback=int(empty),
                                    no_active_source_fallback=int(diagnostics[arm]['active_sources'] == 0),
                                    raw_update_translation_mm=float(deltas[arm_index, :3].norm()*1000.),
                                    raw_update_root_rad=float(deltas[arm_index, 3:6].norm()),
                                    raw_update_fingers_rad=float(deltas[arm_index, 6:].norm()),
                                    nonzero_raw_update=bool(torch.count_nonzero(deltas[arm_index])))
        diagnostics['global'] = dict(absolute_camera_invalid=int(not absolute_valid[2]),
                                     local_camera_invalid=0, empty_fallback=int(empty),
                                     no_active_source_fallback=0)
        return dict(inputs_complete=True, all_arms_forward_complete=True, predictions=predictions,
                    absolute=absolute, previous=previous, absolute_previous=absolute_previous,
                    prepared=prepared, aux=aux_rows, primary_aA=primary_aA,
                    derived_means_px=means_derived, deltas=deltas, diagnostics=diagnostics,
                    timings=dict(global_four_arms_s=global_seconds, geometry_three_arms_s=geometry_seconds,
                                 local_and_control_s=local_seconds))

    def commit(self, result):
        if not result.get('all_arms_forward_complete'):
            raise RuntimeError('Cannot advance partially completed arm state')
        self.previous = result['predictions'].detach().clone()
        self.absolute_previous = result['absolute'].detach().clone()


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def expected_scored_frames(runs, *, warmup=2):
    """Metadata-only full50ms packet count, removing warmup at every run."""
    total = 0
    for run in runs:
        start, stop = (run.start_ms, run.stop_ms) if hasattr(run, 'start_ms') else run
        total += max(0, (int(stop)-int(start))//50-int(warmup))
    return total


def validate_recording_inventory(records, expected):
    """Require every original recording exactly once, including shard merges."""
    names = [row['recording'] for row in records]
    wanted = sorted(name for values in expected.values() for name in values)
    if len(set(names)) != len(names) or sorted(names) != wanted:
        raise RuntimeError('Recording inventory duplicated, missing, or outside the original manifest')
    for row in records:
        person = row['recording'].split('_')[0]
        if person not in expected or row['recording'] not in expected[person] or row.get('person', person) != person:
            raise RuntimeError('Recording person identity differs from the original inventory')


def aggregate_recordings(records, *, person=None):
    """Equal recording weights; missing rows remain explicit and gate-blocking."""
    if not records:
        raise ValueError('No recording summary may be silently omitted')
    records = sorted(records, key=lambda row: row['recording'])
    missing_frames = [row['recording'] for row in records if row['scored_frames'] < 1]
    missing_position = [row['recording'] for row in records if row['position']['weight_total'] <= 0]
    arms = {}
    for arm in ARMS:
        arms[arm] = {metric: _mean_or_none([row['arms'][arm][metric] for row in records
                                          if row['arms'][arm][metric] is not None]) for metric in METRICS}
    return dict(person=person, recordings=len(records), recording_names=[row['recording'] for row in records],
                scored_frames=sum(row['scored_frames'] for row in records),
                missing_scored_recordings=missing_frames, missing_position_recordings=missing_position,
                a_mae_primary_px=_mean_or_none([row['position']['primary_a_mae_px'] for row in records
                                                if row['position']['primary_a_mae_px'] is not None]),
                a_mae_absolute_derived_px=_mean_or_none([row['position']['absolute_a_mae_px'] for row in records
                                                         if row['position']['absolute_a_mae_px'] is not None]),
                arms=arms, aggregation='equal recording means; missing records explicitly gate-blocking',
                scored_frame_coverage_complete=not missing_frames,
                position_recording_coverage_complete=not missing_position,
                independent_unit_note='recordings/variants correlated; descriptive only; no CI/p/population claim')


def gate_decision(persons, all_records):
    """The frozen0.9/1.05/strict-association rules, with explicit NA handling."""
    if set(persons) != {'ycy', 'ylf'}:
        raise ValueError('Both prespecified persons are required')
    position_complete = all(not row['missing_scored_recordings'] and not row['missing_position_recordings']
                            for row in persons.values())
    detail = {}
    for name, row in persons.items():
        detail[name] = dict(
            G1_position=(None if row['missing_position_recordings'] or row['missing_scored_recordings']
                         or row['a_mae_primary_px'] is None or row['a_mae_absolute_derived_px'] is None
                         else row['a_mae_primary_px'] <= .9 * row['a_mae_absolute_derived_px']),
            G2_joint=(None if row['arms']['primary']['joint_abs_mm'] is None or row['arms']['global']['joint_abs_mm'] is None
                      else row['arms']['primary']['joint_abs_mm'] <= 1.05*row['arms']['global']['joint_abs_mm']),
            G2_root=(None if row['arms']['primary']['root_abs_mm'] is None or row['arms']['global']['root_abs_mm'] is None
                     else row['arms']['primary']['root_abs_mm'] <= 1.05*row['arms']['global']['root_abs_mm']))
    endpoint_complete = all(row['scored_frames'] > 0 and
                            all(row['arms'][arm][metric] is not None
                                for arm in ('primary', 'global') for metric in ('joint_abs_mm', 'root_abs_mm'))
                            for row in all_records)
    association_complete = all(row['scored_frames'] > 0 and
                               all(row['arms'][arm]['joint_abs_mm'] is not None for arm in ('primary', 'spatial'))
                               for row in all_records)
    primary_j = _mean_or_none([row['arms']['primary']['joint_abs_mm'] for row in all_records
                              if row['arms']['primary']['joint_abs_mm'] is not None])
    spatial_j = _mean_or_none([row['arms']['spatial']['joint_abs_mm'] for row in all_records
                              if row['arms']['spatial']['joint_abs_mm'] is not None])
    g1 = None if not position_complete else all(row['G1_position'] is True for row in detail.values())
    g2 = None if not endpoint_complete else all(row['G2_joint'] is True and row['G2_root'] is True for row in detail.values())
    g3 = None if not association_complete or primary_j is None or spatial_j is None else primary_j < spatial_j
    passed = g1 is True and g2 is True and g3 is True
    return dict(G1_position=g1, G2_noninferiority=g2, G3_association=g3,
                per_person=detail, all_recording_macro_primary_joint_abs_mm=primary_j,
                all_recording_macro_spatial_joint_abs_mm=spatial_j,
                decision='CONTINUE_PERMITTED' if passed else 'STOP',
                reason=('all3frozen gates pass' if passed else 'failed or insufficient frozen gate; no threshold/length changes'),
                insufficient=any(value is None for value in (g1, g2, g3)),
                no_training_automatically_started=True)


def evaluate_recording(runtime, protocol, recording, output, *, device, start_time, max_seconds):
    from tools.s43_rel.access_guard import InferenceAccessGuard
    person = recording.split('_')[0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Recording evidence exists; no silent evaluation rerun')
    runtime.previous = runtime.absolute_previous = None
    with InferenceAccessGuard() as setup_guard:
        lanes = XJRunLanes(protocol['data_root'], subjects=(person,), recordings=(recording,),
                            batch_size=1, seed=int(protocol['seed']), warmup_steps=int(protocol['warmup']),
                            finite=True, memory_ms=int(protocol['history_ms']), log_counts=False)
    labels_reader = XJSupervisionReader(protocol['data_root'])
    expected_frames = expected_scored_frames(lanes.runs, warmup=int(protocol['warmup']))
    initial_local = tensor_digest(runtime.local.state_dict())
    arrays = {key: [] for key in ('predictions', 'absolute_proposals', 'previous', 'absolute_previous',
              'target0', 'target1', 'beta_target', 'target_ms', 'run_id', 'supervised', 'input_token',
              'supervision_token', 'primary_a_hat', 'primary_a_target', 'primary_aA', 'primary_a_weights',
              'source_ids', 'candidate_mask', 'observed', 'normal', 'vertex_ids', 'derived_means_px',
              'primary_derived_counterfactual_a_error', 'local_covariance_px', 'local_weights',
              'local_association_mass', 'local_unknown_score', 'local_validity_score', *METRICS)}
    sums = {arm: {key: 0. for key in METRICS} for arm in ARMS}
    counters = {arm: dict(absolute_camera_invalid=0, local_camera_invalid=0, empty_fallback=0,
                         no_active_source_fallback=0) for arm in ARMS}
    scored = frames = 0
    a_error = aA_error = a_weight = 0.
    source_chain = supervision_chain = ''
    try:
        while not lanes.exhausted:
            if time.perf_counter()-start_time > max_seconds:
                raise TimeoutError('Prespecified evaluation GPU budget exhausted; not mechanism failure')
            with InferenceAccessGuard() as input_guard:
                batch = lanes.peek_batch()
                if not batch[0].get('valid', False):
                    lanes.commit_batch(batch)
                    continue
                result = runtime.forward(batch)
            if not result.get('all_arms_forward_complete'):
                raise AssertionError('All four predictions must precede supervision')
            labels = labels_reader.read(batch, inputs_complete=True)
            label_token = supervision_digest(labels)
            source_chain = advance_chain(source_chain, batch.token)
            supervision_chain = advance_chain(supervision_chain, batch.token, label_token)
            selected = bool(labels.mask[0])
            item, prepared = batch[0], result['prepared']
            records = prepared.rho.shape[1]
            metrics = {key: np.full(4, np.nan, np.float64) for key in METRICS}
            primary_target = np.full(records, np.nan, np.float32)
            weights_np = np.zeros(records, np.float32)
            same_support_derived_error = np.full(records, np.nan, np.float32)
            if selected:
                target0 = torch.from_numpy(labels.target0.copy()).to(device).expand(3, -1)
                target1 = torch.from_numpy(labels.target1.copy()).to(device).expand(3, -1)
                targets = runtime.geometry.targets(prepared, target0, target1)
                fixed = source_record_weights(prepared.source_ids[0:1], targets['supervised'][0:1])
                primary = result['aux']['primary']['a_normal_px']
                actual = targets['target_a_px'][0:1]
                aA = result['primary_aA']
                ae, aAe = (primary-actual).abs(), (aA-actual).abs()
                a_error += float((ae*fixed).sum())
                aA_error += float((aAe*fixed).sum())
                a_weight += float(fixed.sum())
                primary_target = actual[0].cpu().numpy()
                weights_np = fixed[0].cpu().numpy()
                same_support_derived_error = aAe[0].cpu().numpy()
                metrics = endpoint_metrics(runtime.base, result['predictions'],
                                           torch.from_numpy(labels.target1[0].copy()).to(device),
                                           torch.from_numpy(labels.beta_target[0].copy()).to(device))
                for index, arm in enumerate(ARMS):
                    for key in METRICS:
                        sums[arm][key] += float(metrics[key][index])
                scored += 1
            for arm in ARMS:
                for key in counters[arm]:
                    counters[arm][key] += int(result['diagnostics'][arm].get(key, 0))
            row = dict(frame=frames, target_ms=item['target_ms'], run_id=item['run_id'],
                       reset=item['reset'], supervised=selected, input_token=batch.token,
                       supervision_token=label_token, source_hash_chain=source_chain,
                       supervision_hash_chain=supervision_chain,
                       diagnostics=result['diagnostics'], timings=result['timings'],
                       input_audit=input_guard.receipt())
            with (output/'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False)+'\n')
            for key, value in (
                ('predictions', result['predictions'].cpu().numpy()),
                ('absolute_proposals', result['absolute'].cpu().numpy()),
                ('previous', result['previous'].cpu().numpy()),
                ('absolute_previous', result['absolute_previous'].cpu().numpy()),
                ('target0', labels.target0[0]), ('target1', labels.target1[0]), ('beta_target', labels.beta_target[0]),
                ('target_ms', item['target_ms']), ('run_id', item['run_id']), ('supervised', selected),
                ('input_token', batch.token), ('supervision_token', label_token),
                ('primary_a_hat', result['aux']['primary']['a_normal_px'][0].cpu().numpy()),
                ('primary_a_target', primary_target), ('primary_aA', result['primary_aA'][0].cpu().numpy()),
                ('primary_a_weights', weights_np), ('source_ids', prepared.source_ids.cpu().numpy()),
                ('candidate_mask', prepared.candidate_mask.cpu().numpy()), ('observed', prepared.observed.cpu().numpy()),
                ('normal', prepared.normal.cpu().numpy()), ('vertex_ids', prepared.vertex_ids.cpu().numpy()),
                ('derived_means_px', result['derived_means_px'][0].cpu().numpy()),
                ('primary_derived_counterfactual_a_error', same_support_derived_error),
                ('local_covariance_px', torch.cat([result['aux'][arm]['covariance_px'] for arm in LOCAL_ARMS]).cpu().numpy()),
                ('local_weights', torch.cat([result['aux'][arm]['weights'] for arm in LOCAL_ARMS]).cpu().numpy()),
                ('local_association_mass', torch.cat([result['aux'][arm]['association_mass'] for arm in LOCAL_ARMS]).cpu().numpy()),
                ('local_unknown_score', torch.cat([result['aux'][arm]['unknown_score'] for arm in LOCAL_ARMS]).cpu().numpy()),
                ('local_validity_score', torch.cat([result['aux'][arm]['validity_score'] for arm in LOCAL_ARMS]).cpu().numpy()),
            ):
                arrays[key].append(value)
            for key in METRICS:
                arrays[key].append(metrics[key])
            runtime.commit(result)
            lanes.commit_batch(batch)
            frames += 1
            if frames % 100 == 0:
                dump(output/'progress.json', dict(status='RUNNING', recording=recording, frames=frames,
                                                  scored_frames=scored, elapsed_s=time.perf_counter()-start_time))
                print(json.dumps(dict(recording=recording, frames=frames, scored_frames=scored,
                                      elapsed_s=time.perf_counter()-start_time)), flush=True)
        if tensor_digest(runtime.local.state_dict()) != initial_local:
            raise AssertionError('Evaluation modified frozen local weights')
        if scored != expected_frames:
            raise RuntimeError(f'Incomplete recording coverage: expected{expected_frames} scored frames, got{scored}')
        artifact = {key: np.asarray(values) for key, values in arrays.items()}
        artifact['arms'] = np.asarray(ARMS)
        artifact['local_arms'] = np.asarray(LOCAL_ARMS)
        artifact['inference_beta'] = np.zeros(10, np.float32)
        temporary = output/'frames.tmp.npz'
        np.savez_compressed(temporary, **artifact)
        temporary.replace(output/'frames.npz')
        summary = dict(status='COMPLETE', recording=recording, person=person, frames=frames, scored_frames=scored,
                       expected_scored_frames=expected_frames,
                       arms={arm: {key: sums[arm][key]/scored if scored else None for key in METRICS} for arm in ARMS},
                       position=dict(primary_a_mae_px=a_error/a_weight if a_weight else None,
                                     absolute_a_mae_px=aA_error/a_weight if a_weight else None,
                                     primary_abs_error_weighted_sum=a_error,
                                     absolute_abs_error_weighted_sum=aA_error, weight_total=a_weight),
                       failures=counters, setup_input_audit=setup_guard.receipt(),
                       source_hash_chain=source_chain, supervision_hash_chain=supervision_chain,
                       frames_sha256=sha(output/'frames.npz'), trace_sha256=sha(output/'trace.jsonl'),
                       data_identity=lanes.manifest(), no_confidence_filtering=True,
                       beta_contract='predictions beta0, endpoint GT beta_target; local a proxy beta0',
                       scope='Fixed-checkpoint SELECT diagnosis; no zgz or population inference')
        dump(output/'metrics.json', summary)
        return summary
    except BaseException as exc:
        # A numerical/engineering failure does not become a silently excluded
        # recording or a fabricated valid endpoint metric.
        dump(output/'failure.json', dict(status='FAILED_PRESERVED', recording=recording,
                                        frames_completed=frames, scored_frames_completed=scored,
                                        error=repr(exc), traceback=traceback.format_exc(),
                                        source_hash_chain=source_chain, supervision_hash_chain=supervision_chain))
        if frames:
            np.savez_compressed(output/'partial_frames.npz', **{key: np.asarray(values) for key, values in arrays.items()})
        raise


def evaluate_person(protocol_path, freeze_path, *, person, device='cuda:0', out, backend=None, recording=None):
    protocol, frozen, config, complete, chosen = verify_evaluation_identity(
        protocol_path, freeze_path, person=person, backend=backend)
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() or (output/'summary.json').exists():
        raise FileExistsError('This person has an evaluation already; no hidden rerun')
    random.seed(int(protocol['seed']));np.random.seed(int(protocol['seed']));torch.manual_seed(int(protocol['seed']))
    torch.set_num_threads(int(frozen.get('execution', {}).get('cpu_threads', 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, device_record = configure_device(device, config, engineering=True)
    if device.type != 'cuda' or device_record['physical_gpu'] not in protocol['devices']:
        raise RuntimeError('Final SELECT replay uses only the declared physical GPU2/3')
    from tools.s43_integration.model import load_legacy
    base, _ = load_legacy(config['base_checkpoint'], config['base_config'], device)
    base = base.float().eval();base.render_fp32 = True
    if getattr(base, 'root_compose', 'add') != 'add':
        raise RuntimeError('Frozen physical51 additive chart required')
    for parameter in base.parameters():parameter.requires_grad_(False)
    checkpoint = torch.load(protocol['checkpoint']['path'], map_location='cpu', weights_only=False)
    if (checkpoint.get('updates') != protocol['checkpoint']['updates']
            or checkpoint.get('sourcefreeze_sha256') != protocol['training_sourcefreeze']['sha256']):
        raise RuntimeError('Checkpoint content does not match final training identity')
    local = JointLocalOperator().to(device).float().eval()
    local.load_state_dict(checkpoint['local'], strict=True)
    local_digest = tensor_digest(local.state_dict())
    if local_digest != complete['local_digest']:
        raise RuntimeError('Local tensor identity differs from completed training')
    gc = GeometryConfig(max_supports=int(config['num_supports']),
                        candidates_per_support=int(config['candidates_per_support']), radius_m=float(config['radius_m']))
    if chosen == 'vectorized':
        from .evaluation_geometry import FastJointGeometry
        geometry = FastJointGeometry(base, gc)
    else:
        geometry = JointGeometry(base, gc)
    runtime = FourArmRuntime(base, local, geometry, device)
    manifest = json.loads((Path(protocol['data_root'])/'splits_semkine.json').read_text())
    names = sorted(name for name in manifest['train']['trials'] if name.split('_')[0] == person)
    all_select = [name for name in manifest['train']['trials'] if name.split('_')[0] in protocol['subjects']]
    if len(all_select) != int(protocol['expected_recordings']) or not names:
        raise RuntimeError('Declared original SELECT recording inventory differs')
    full_person_inventory = list(names)
    if recording is not None:
        if recording not in names:
            raise RuntimeError('Recording shard is outside the prespecified person inventory')
        names = [recording]
    identity = dict(schema='xvm_joint_select_person_manifest_v1', person=person, recordings=names,
                    eval_protocol_sha256=sha(protocol_path), eval_sourcefreeze_sha256=sha(freeze_path),
                    final_checkpoint_sha256=protocol['checkpoint']['sha256'], local_digest=local_digest,
                    geometry_backend=chosen, device=device_record, inference_beta='zeros10',
                    input_guard='tools.s43_rel.access_guard.InferenceAccessGuard; one process dispatcher',
                    evaluation_scope='recording_shard' if recording is not None else 'complete_person',
                    full_person_recording_inventory=full_person_inventory,
                    GT_read='after all4arm forward', arms=list(ARMS), no_reference_augmentation=True,
                    created_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid())
    dump(output/'manifest.json', identity)
    started = time.perf_counter()
    records = []
    try:
        for name in names:
            records.append(evaluate_recording(runtime, protocol, name, output/'recordings'/name,
                device=device, start_time=started,
                max_seconds=float(protocol['max_gpu_hours'])*3600/len(protocol['subjects'])))
            dump(output/'progress.json', dict(status='RUNNING', person=person, completed_recordings=len(records),
                                              total_recordings=len(names), elapsed_s=time.perf_counter()-started))
        summary = dict(status='COMPLETE_RECORDING_SHARD' if recording is not None else 'COMPLETE',
                       identity=identity, records=records,
                       person_macro=aggregate_recordings(records, person=person),
                       elapsed_s=time.perf_counter()-started)
        dump(output/'summary.json', summary)
        return summary
    except BaseException as exc:
        dump(output/'failure.json', dict(status='FAILED_PRESERVED', person=person,
                                        error=repr(exc), traceback=traceback.format_exc(),
                                        completed_recordings=[row['recording'] for row in records],
                                        elapsed_s=time.perf_counter()-started))
        raise


def combine_persons(protocol_path, freeze_path, summaries, output):
    protocol, _, _, _, _ = verify_evaluation_identity(protocol_path, freeze_path)
    loaded = [(Path(path).resolve(), json.loads(Path(path).read_text())) for path in summaries]
    manifest = json.loads((Path(protocol['data_root'])/'splits_semkine.json').read_text())
    expected = {person: sorted(name for name in manifest['train']['trials'] if name.split('_')[0] == person)
                for person in protocol['subjects']}
    by_person = {}
    records = []
    for summary_path, row in loaded:
        identity = row['identity']
        if (row.get('status') not in ('COMPLETE', 'COMPLETE_RECORDING_SHARD')
                or identity['eval_protocol_sha256'] != sha(protocol_path)
                or identity['eval_sourcefreeze_sha256'] != sha(freeze_path)
                or identity['final_checkpoint_sha256'] != protocol['checkpoint']['sha256']):
            raise RuntimeError('Incomplete or mismatched person evaluation')
        person = identity['person']
        if person not in protocol['subjects']:
            raise RuntimeError('Unauthorized person summary')
        actual_names = [record['recording'] for record in row['records']]
        if (len(set(actual_names)) != len(actual_names) or not set(actual_names) <= set(expected[person])
                or sorted(identity['recordings']) != sorted(actual_names)
                or sorted(identity.get('full_person_recording_inventory', expected[person])) != expected[person]):
            raise RuntimeError('Person recording inventory duplicated, missing, or outside the original manifest')
        if row['status'] == 'COMPLETE' and sorted(actual_names) != expected[person]:
            raise RuntimeError('A complete-person summary may not hide missing recordings')
        for record in row['records']:
            folder = summary_path.parent/'recordings'/record['recording']
            if (json.loads((folder/'metrics.json').read_text()) != record
                    or sha(folder/'frames.npz') != record['frames_sha256']
                    or sha(folder/'trace.jsonl') != record['trace_sha256']):
                raise RuntimeError('Changed recording metrics/state/token evidence')
        recomputed = aggregate_recordings(row['records'], person=person)
        if recomputed != row['person_macro']:
            raise RuntimeError('Person macro does not match equally weighted recording summaries')
        records.extend(row['records'])
    validate_recording_inventory(records, expected)
    records = sorted(records, key=lambda row: row['recording'])
    if len(records) != int(protocol['expected_recordings']):
        raise RuntimeError('Every prespecified SELECT recording is required')
    if sum(row['scored_frames'] for row in records) != int(protocol['expected_supervised_frames_metadata_only']):
        raise RuntimeError('Total SELECT scored-frame coverage differs from frozen metadata inventory')
    by_person = {person: aggregate_recordings([row for row in records if row['person'] == person], person=person)
                 for person in protocol['subjects']}
    verdict = dict(status='COMPLETE', eval_protocol_sha256=sha(protocol_path),
                   eval_sourcefreeze_sha256=sha(freeze_path), persons=by_person,
                   gate_decision=gate_decision(by_person, records), records=records,
                   fixed_final_checkpoint_sha256=protocol['checkpoint']['sha256'],
                   scope='descriptive SELECT diagnostics on pretrained S43 training persons; no zgz/main-row/population claim')
    dump(output, verdict)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('person', 'combine'), nargs='?', default='person')
    parser.add_argument('--eval-protocol', required=True)
    parser.add_argument('--eval-source-freeze', required=True)
    parser.add_argument('--person', choices=('ycy', 'ylf'))
    parser.add_argument('--recording', help='One original recording shard; complete merge still requires all16')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--geometry-backend', choices=('frozen', 'vectorized'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--person-summaries', nargs='+')
    args = parser.parse_args()
    if args.command == 'person':
        if not args.person:parser.error('--person is required')
        result = evaluate_person(args.eval_protocol, args.eval_source_freeze, person=args.person,
                                 device=args.device, out=args.out, backend=args.geometry_backend, recording=args.recording)
        print(json.dumps(result['person_macro'], ensure_ascii=False, indent=2))
    else:
        if not args.person_summaries:parser.error('--person-summaries is required')
        result = combine_persons(args.eval_protocol, args.eval_source_freeze, args.person_summaries, args.out)
        print(json.dumps(result['gate_decision'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

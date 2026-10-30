"""Preregister the witness prototype's limited tests, before task outcomes."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/s43_revisit_20261028'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = OUT/'stage2_protocol.json'
    if path.exists():
        raise FileExistsError('Do not replace a scored prototype protocol.')
    p = dict(
        schema='s43_witness_prototype_v1',
        registered_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='Current user stages 1-3; root proceeds through evidence gates without another permission question.',
        prior='Stage1 all published attribution scores and 100 R1/R2 cross-reviews known. No witness task score read before this registration.',
        scope='Minimum B prototype and A/simple controls. Conditional information diagnostics are separate from LNES-free complete MANO training or deployment accuracy.',
        selection=dict(main='B native joint-counterexample episodes',backup='A finite full spatiotemporal field',
                       third='C ownership order, conditional research only',
                       no_novelty_certification=True, no_vote_selection=True),
        memory=dict(height=180,width=240,tile=8,epoch_ms=10,horizon_ms=50,slots=128,
                    count_dtype='uint32',coarse_cell=4,source_polarity='stored slots0/1',
                    same_timestamp='native integer-ms multiset, atomic complete watermark',
                    independent_reserve=32,joint_quotas=[20,19,19,19,19],
                    signed_quotas=[[10,10],[10,9],[10,9],[10,9],[10,9]],
                    update='old surviving selected episodes + newly sealed candidates; no resurrection of rejected content. This is not exact all-window topK or exact window reservoir.',
                    fill='unused quota filled by mode-independent source hash; duplicate cross-pool content stored/counted once',
                    transport_packet_event_cap=262144,
                    cap_note='Transport/workspace domain, not a clipping/rate cap on the whole stream. Chunk parity only where both packetizations satisfy domain.',
                    state_budget_MiB=6.0,
                    budget_note='Current full unfinished epoch + full coarse ring + all selected fine payloads + metadata + persistent geometry/cache. Temporary input/query/selection scratch and learned weights reported separately.'),
        geometry=dict(sigma_px=4.,MCP_MANO_indices=[13,1,4,10,7],
                      axes='Once chosen from public beta0 mean by +/-0.3rad maximal wrist-tip shortening, no task GT',
                      magnitude='Four wrist-relative finger joints mean5mm over50ms, actual achieved size logged, angle cap0.6rad',
                      trajectory='Exact FK projected endpoints0/50ms, image interpolation approximation; 3 MCP-tip segments, no visible-surface or radiance-likelihood claim',
                      kernel_grid_px=4,epoch_kernel_expected_bytes=540000,
                      score='Within-epoch time-centered +/-finger kernel difference dotted with true counts, divided by sqrt(1+count); owner argmax absolute signed response.',
                      nuisance='Global root movement is a controlled factor, not assumed eliminated by subtracting root flow.'),
        synthetic=dict(sequences=80,clean=40,moving_background=40,seed=3407,
                       factorization='2 public boundary root orientations(0,y0.3rad) x5fingers x2directions x2root x displacements(0,10mm/50ms)',
                       duration_ms=100,motion_ms=50,tail='hold terminal hand pose in last50ms',
                       sensor='Public binary hand mask/radiance threshold generator, integer-ms native time; threshold0.35 and common initialized phase seed, neither v2e provenance nor real sensor',
                       background='Static0.15 or physical moving rear radiance edge0.15/0.30 at x30->210px/100ms, hand0.70; foreground mask occludes rear plane',
                       initial_condition='Known synthetic public boundary only; target finger/sign never provided to selection or reader',
                       target='Different current FK terminal modes, not same-endpoint loops',
                       split='Four orientation x root contexts, leave one complete context per fold; clean/background same physical context stay together',
                       independence='Four controlled contexts, not80 independent people; no statistical generalization claim',
                       primary='10 legal finger/sign mode identification and active-finger FK terminal error; root/RA/absolute separately',
                       optional_cadence_queries_ms=[10,20,30,40,50,75,100],
                       cadence_note='Only predetermined observed query addresses; purequery and feedback effects separate. No post-score cherry-pick of onset/query.'),
        existing=dict(sample_cache='outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz',FIT=224,SELECT=64,
                      source='Exactly old288 sample IDs/native intervals. Reuse context/prior/target FK only, not old A/Z/LNES feature tensors.',
                      condition='Old shared-anchor, given-shape-history diagnostic: canonical anchor used aux beta. New reader is LNES-free but this is not beta0 deployment history or a fully independent front-end system.',
                      targets='FK21 joint delta63, as information proxy; not a51D end-to-end MANO method',
                      metrics=['FIT/SELECT absolute and J0 joint errors','four SELECT recordings','root displacement','local-motion strata if existing fixed masks applicable','capacity competition/retained coverage'],
                      no_new_zgz=True),
        controls=['uniform episodes','recency episodes','density episodes','anatomical quota without temporal disagreement',
                  'generic temporal score without MANO','full witness',
                  'same-byte spatial2x2 native-ms count field','uncompressed fine field16.48MiB information positive control',
                  'context only','coverage only'],
        control_limits='DEVO-style learned task-saliency is not assumed implemented. Absent that control, a win cannot certify novelty over task-aware coreset methods.',
        readers=dict(common='Single shared appropriate CNN mixed over sampler families; local3x3 nonlinear conv before4pxpool, context input same. FIT includes normal and selection-perturbed versions with injection removed from query context.',
                     cnn=dict(optimizer='Adam',lr=.001,steps=2000,batch=32,seed=3407,CPU_threads=8),
                     rbf=dict(ridge=.01,bandwidth='FIT median only',normalization='FIT only'),
                     report='Actual parameter/storage/steps plus FIT feasibility. Optional per-arm adapted heads require named source-frozen extension before task scores; none silently added.'),
        E1='Preserved task information under fixed current source/bytes and common reader; synthetic and existing separate. Full fine positive control must fit; bad positive control = unresolved tool/optimization, not raw information absence.',
        E2=dict(selection_error=['root x+10mm','root yaw+0.1rad'],
                schedule='Error introduced before selection; freeze retained dictionary; remove only injected anchor error at query, never supply current target finger state.',
                comparisons='B vs quota/generic/uniform and raw positive control with same query context; performance and actual counterexample retention separate',
                gates='Evidence of B value only if both readers improve existing SELECT J0 error over strongest simple selection by >=max(.25mm,2%), >=3/4 recordings same direction and none worsens>.5mm; synthetic beyond chance and raw positive control feasible, active-finger/onset gains rather than only absolute root. This is a conservative development gate, not significance or a universal theorem.',
                context_control='Coverage-only or context-only explaining gain blocks attribution to preserved event content'),
        E3=dict(contracts=['causal prefix','same-ms permutation','chunks1/7/50 where accepted','empty','reset','independent lanes','padding no update','source duplicate atomic rejection','overflow atomic rejection','state restore','purequery at fixedcommit50 plusquery25','training/online samecore','priorversion affects futurepolicy without rewriting old observation'],
                resources='Measure real native ch prefix and declared synthetic rates; all memory fields, input/selection workspace and policy refresh + ingest + seal/select + readout. Network and terminal MANO reported only if actually run, never inferred from old module timings.',
                interpretation='Local count write does not imply local encoder or free query. No all-system real-time claim from representation-only benchmark.'),
        decision=dict(proceed='Only if task evidence is not explained by controls and feasibility/contracts pass: independent40ch front end, own causal warmup, original MANO/loss, five controls, reasonable full training budget separately frozen.',
                      revise='At most one targeted preregistered correction based on FIT/implementation failure before reevaluation; preserve original negatives and disclose reused development SELECT.',
                      otherwise='Reject unsupported B value; move to A as an engineering front-end question, not automatically a Nature-grade original method. If data/positive control inconclusive, report that rather than false proof of impossibility.'),
        source_freeze='After engineering fixes, snapshot all actual implementation sources before first task fitting/scoring. Later changes use named amendments; no old-source edits.',
        design_sha256=sha(OUT/'selected_prototype_spec.md'),
        battle_sha256=sha(OUT/'battle/summary.json'),
        stage1_sha256=sha(OUT/'stage1_protocol.json'),
    )
    path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),registered_at_utc=p['registered_at_utc']),ensure_ascii=False))


if __name__ == '__main__':
    main()

# Audit A notes (full)

## 1 Protocol
- splits_semkine.json: train 72 seq / 9 subjects; val=val_core=test = [zgz_global,zgz_local] (_provenance notes same). NO sealed test distinct from selection set.
- dataset.py:475-476; select_checkpoint.py:7-12; make_s36_row.py:4-5,63-66
- category: name contains global/local (protocol.py:69-74; prepare_hand_data; aux["category"]). zgz_global 1 run; zgz_local 90 runs.
- eval: each valid_runs_ms segment inits prev=GT[a]+sample_init_noise (eval_track.py:149-159,118-124). TRACK noise scales from cfg.
- Alignment: root_align = subtract joint0 (127-128). Table MPJPE-local/global & MPVPE = mpjpe_ra_mm / mpvpe_ra_mm by seq name (make_s36_row.py:54-89). RA column = overall mpjpe_ra_mm recursive.

## 2 §9 mismatches only
- MATCH: root Linear 4624=512+16*257 (model.py:916-923); route no_grad (1460); out=delta+prev aa add (1592-1597); prev->root via a(prev)+prev_mlp.
- MISMATCH: diagram "逐维独立" — directed 20% uses block unit-direction (dataset.py:243-250), not per-dim iid.
- Note (outside diagram but related): docs L72 cites model.py:825/1133 — those lines are FK-graph / drifted; routed head is :923.

## 3 Confounds vs s37_routed
| arm | delta | univariate? | budget |
| s36_eventgnn | no ROUTED, PREV_RENDER | no | 6k, 2seed, bs512 |
| s37_fkgraph | encoder fk_graph | no; bs1024×1gpu; resumed_from last.ckpt (same-run) | 6k,2seed |
| s37_meshgraph | encoder mesh_graph | no | 6k,2seed |
| s37_rootinnov | ROOT_INNOV+PREV_MLP_ROOT=F + INIT_FROM matched s37 ckpt | no | 1.5k,2seed,hot |
| s37_rotw10 | LAMBDA_R 60→600 | yes | 6k,2seed |
| s37_edge6 | LAYERS 3→6 | ~yes; bs256×accum2 (eff 1024) | 6k,2seed |
| s37_rootfuse | ROOT_FUSION=per_joint | yes | 6k,2seed |
| s37_xyz_c0/c1 | xyz_mesh, no active/prev_mlp; geom F/T | no; 1.5k; **seed 3407 only** | |

## 4 Seeds
- 3407/3408: same yaml path, training_metadata seed field; CLI --seed. s37_routed resumed_from absent / INIT_FROM none → standard seed reps, no hot-start.

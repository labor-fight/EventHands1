# FastMETRO hand network (vendored)

Source: the official EvRGBHand repository, https://github.com/AlanJiang98/EvRGBHand, commit
`bc6cc3b40c172b89fb1b4a2223690ecc09fe65f6` (local copy `/data1/lyq/code/mesh/EvRGBHand`), MIT (`LICENSE`). Its
`src/modeling/fastmetro.py` and `transformer.py` are FastMETRO's official code (POSTECH AMI Lab, MIT; the transformer is
modified from DETR), with the event modality EvRGBHand uses for its event-only baseline (`src/configs/config/fastmetro_ev.yaml`).

Used by `semkine/cmp_fme.py` (CMP arm `fme`, docs/EXPERIMENT_RECORDS.md#cmp-amend8-20261110).

| file | origin | changes |
|---|---|---|
| `fastmetro.py` | `src/modeling/fastmetro.py` | the `is_main_process` import dropped, `pretrained` passed from `config['model']` (default true, as in a single process); the three adjacency files loaded from `data/` next to the module; the position encoding's integer ramps built on the CPU and moved to the device (`torch.cumsum` has no deterministic CUDA kernel in torch 2.1; same values) |
| `transformer.py` | `src/modeling/transformer.py` | the temporal-attention classes (`TemporalEncoder`, `TemporalAttentionLayer`, `DynamicPositionBias`, `build_temporal_encoder`) left out: FastMETRO_Hand_Network does not use them and they need `einops` |
| `data/mano_195_adjmat_*.pt` | `src/modeling/data/` | none (adjacency of the 195-vertex coarse MANO mesh, the decoder's attention mask) |
| `data/mano_195_downsampling.npy` | `src/modeling/data/mano_downsampling.npz` | `D[0]` (778 -> 195, the official `Mesh.downsample` of `_mano.py`) written as a dense float32 array, so no pickled sparse matrix is read at run time; every row has a single 1 |

Not vendored: `Loss.py` (needs pytorch3d for the silhouette renderer, which the FastMETRO arch does not use) and
`_mano.py` (smplx). The loss terms the FastMETRO arch uses (`Loss.get_loss_mano`) and the joint regressor of
`MANO.get_3d_joints` are written out in `semkine/cmp_fme.py` on this repository's `ManoLayer`.

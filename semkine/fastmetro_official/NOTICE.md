# Source and license notices

The vendored `fastmetro.py` and `transformer.py` come through the official
[EvRGBHand repository](https://github.com/AlanJiang98/EvRGBHand/tree/bc6cc3b40c172b89fb1b4a2223690ecc09fe65f6),
local checkout commit `bc6cc3b40c172b89fb1b4a2223690ecc09fe65f6`. The file mappings and existing
modifications are recorded in [README.md](README.md). Its original MIT text remains in `LICENSE`.

Both source files retain the FastMETRO / POSTECH copyright and MIT headers.
`transformer.py` additionally retains its notice that it was modified from
DETR, including the Facebook copyright and the link to DETR's license.
The following upstream license texts are included verbatim:

| Local file | Official upstream source | Retrieved revision | SHA-256 |
|---|---|---|---|
| `LICENSE.FastMETRO` | [postech-ami/FastMETRO/LICENSE](https://raw.githubusercontent.com/postech-ami/FastMETRO/c5ae2069168b438f613ce848b7443d5a90ae9cba/LICENSE) | `c5ae2069168b438f613ce848b7443d5a90ae9cba` | `2787e34d8a0a004a2960ac47d688bd205a5c38e343818f31035acc093205e8bf` |
| `LICENSE.DETR` | [facebookresearch/detr/LICENSE](https://raw.githubusercontent.com/facebookresearch/detr/29901c51d7fe8712168b8d0d64351170bc0f83e0/LICENSE) | `29901c51d7fe8712168b8d0d64351170bc0f83e0` | `b208c52b00a2df2c8f4c3298c34407bf2bfe409968213372251c91fcc737a1a5` |

These revisions identify the retrieved license texts; they do not assert that
either revision is the original ancestor revision used by EvRGBHand.

Current vendored source SHA-256 values (the source bytes were not changed when
adding these notices):

| Source file | SHA-256 |
|---|---|
| `fastmetro.py` | `90aba27eddf28feabf30f5c82d6323e6edf0930082ac15ae081daf564bad0b18` |
| `transformer.py` | `04f112936f94a44a085a681e64734a970157a72f4ab4d9d357304594ecb72ef4` |

The three `data/mano_195_adjmat_*.pt` files and the derived
`data/mano_195_downsampling.npy` retain the provenance recorded in `README.md`.

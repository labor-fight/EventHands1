# EvHandPose 来源与许可记录

本目录为 CMP 对比实验移植的网络源码。上游论文为 Jiang 等的 *EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision*（TPAMI 2024）；官方仓库为 [AlanJiang98/EvHandPose](https://github.com/AlanJiang98/EvHandPose)。核对日期：2026-11-10。

## 本机来源与现有改动

源码来自本机 `/data1/lyq/code/mesh/EvHandPose/` 副本。该副本没有 `.git` 元数据，无法取得其 checkout commit；不把当前远端 HEAD 当作本机副本的 commit。下表记录本次核对的原始文件与已移植文件 SHA256，文件路径相对于上述副本根目录。

| 本目录文件 | 本机原始来源 | 原始文件 SHA256 | 已移植文件 SHA256 |
|---|---|---|---|
| `backbone.py` | `core/model/backbone.py` | `5cdbb2fa4641743df61260eba3858f71d052a1edbb3996deee001e6b2b0338ce` | `b10e71659a9cedae55937ae5e3ae1212f7c5061d13a6fb87172d74d8bf901490` |
| `basic_models.py` | `core/model/utils/basic_models.py` | `e12d7ccc80a2430e4cc169c8107965957ffaee15015bc57e59436d74a917ec90` | `9344df8bc4c2e7b7cc3099d65e7844a307b66f86a81170e5e2c0e7f787835e9c` |
| `unet.py` | `core/model/utils/unet.py` | `6a1b5343b8354777e02d05894e29e029ec94f609e6a2add4c877d5fbf1f49743` | `dc11734742cf8964da0f95a2b5b50031ae09276a2a561b8fc400d029cdb2c3fa` |
| `utils.py` | `core/model/utils/utils.py` | `658c98f1621d06867e3b5df26d113058b863bfb2d47b083ba81c679bacac6948` | `4ab79c48f156c5f869bf20685f90b05b010bfb7fdfd674e6ee8fab103eca3c4b` |

`__init__.py` 是本工程添加的包说明文件，没有对应的上游源码文件。

将本目录与上述本机原始文件逐行比较，既有改动为：

- 四个源码文件顶部增加了两行移植来源注释。
- `backbone.py`：两处 `core.model.utils` 绝对导入改为包内相对导入；删除未使用的 `fvcore.nn.FlopCountAnalysis` 导入。
- `unet.py`：对 `utils` 与 `basic_models` 的两处导入改为包内相对导入。
- `basic_models.py` 与 `utils.py`：除顶部来源注释之外无改动。

本次只增加本说明和许可证文本，没有修改任何 `.py` 文件或网络实现。

## 上游已有的进一步来源

`basic_models.py`、`unet.py` 和 `utils.py` 都保留了上游的 “Adapted from UZH-RPG” 声明，指向 [uzh-rpg/rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid)。

`basic_models.py` 还保留了以下上游声明：

- ConvLSTM：[Atcold/pytorch-CortexNet 的 ConvLSTMCell.py](https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py)。
- ConvGRU：[jacobkimmel/pytorch_convgru 的 convgru.py](https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py)。

## 许可记录及未闭合事项

截至本次核对，本机 EvHandPose 副本未附许可文件；其官方仓库根目录及 README 也未见明确的 EvHandPose 源码许可证声明。公开仓库与论文引用不等于已取得完整的再分发许可说明。本说明记录这一资料缺口，不替 EvHandPose 作者授予权限，不为其新增或推定 MIT、Apache、GPL 等整项目许可，也不改变 EventHands1 的整体许可。

E2VID 官方仓库的 [LICENSE 原文](https://raw.githubusercontent.com/uzh-rpg/rpg_e2vid/master/LICENSE) 为 GNU GPL 第3版，现将该文件逐字节保存在本目录的 [LICENSE.rpg_e2vid](LICENSE.rpg_e2vid)，供上述已有 E2VID 来源声明一并查阅。附带该文本不代表 EvHandPose 自身的许可缺口已经解决，也不代表已独立核实所有进一步来源的完整许可链。

- 下载来源：`https://raw.githubusercontent.com/uzh-rpg/rpg_e2vid/master/LICENSE`
- 下载核对日期：2026-11-10
- 原文 SHA256：`3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`

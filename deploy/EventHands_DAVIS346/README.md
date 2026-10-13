# EventHands_DAVIS346：S39 事件相机手部 mesh 实时跟踪部署包

**S39** = `dt_dz_l3`（ResNet18 到 layer3 的跟踪网络，2.80 M 参数）+ W=200 ms 证据窗口（裁到开始跟踪时刻）+ DT2 自适应滤波 + 证据门 ramp1000。本包是它的最小推理实现，外加 DAVIS346 接入、标定、可视化和自检工具。只依赖 PyTorch、NumPy、OpenCV 和 dv-processing，不依赖训练仓库。

**一致性已验证**：在训练服务器上（L20、torch 2.1），本包对 zgz 测试受试者全部 2590 步的输出与官方 S39 评测**逐位相同**（三个种子，max |差| = 0）。事件由 dv-processing 直接读取原始 AEDAT4。验证范围还包括：主循环回放、模拟实时、CUDA graph、三种子集成、10 ms 步长、346×260 分辨率映射和 CPU 模式，详见文末。

| zgz（官方口径，三种子均值） | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global |
|---|---|---|---|---|
| S39（单模型，默认 `s39_s3408.pt`） | 10.27 mm | 7.36 mm | 8.94 mm | 6.25 mm |
| S39 三种子集成（`--weights` 给 3 个文件，计算量 ×3） | 9.84 mm | 7.11 mm | — | — |

> 以上是去根对齐（root-aligned）误差，测试条件是训练用的那台相机（见"已知限制"）。DAVIS346 上**没有带标注的数据，精度无法给出数字**。

---

## 1. 目录

```
run_davis346.py         实时跟踪主程序（DAVIS346 / AEDAT4 录制 / 模拟实时），显示、保存、录视频
calibrate_davis346.py   热像素屏蔽、事件计数尺度标定
verify_install.py       安装自检：与训练服务器的官方 S39 输出逐步对比
eh346/                  推理代码：net.py 网络与前态渲染、mano.py、filt.py 自适应滤波、events.py 事件映射与 LNES、
                        tracker.py 单步协议、sources.py 相机/文件、viz.py 显示
weights/                s39_s3407.pt、s39_s3408.pt（默认）、s39_s3409.pt：三个训练种子（各含网络与 MANO 缓冲区）
configs/s39.json        窗口、滤波增益表、训练相机的事件计数参考
configs/template_init.json  模板初始姿态（右手张开、掌心朝相机、手指向上）
examples/               calib_TEMPLATE.json、init_example.json、init_mano_example.json
tests/zgz_clip.npz      自检数据：zgz 原始事件（640×480）、协议初值、真值、服务器官方输出（7 MB）
dev_tools/              仅在训练服务器上用的导出与对比脚本（部署不需要）
```

## 2. 安装

需要 Python ≥ 3.9（dv-processing 2.x 的要求）。

```bash
# 1) PyTorch：GPU 主机按 https://pytorch.org 安装与驱动匹配的 CUDA 版本；只用 CPU 也能跑（见第 9 节）
# 2) 其余依赖
pip install -r requirements.txt
```

- **Linux**：DAVIS 是 USB 设备，普通用户需要 udev 规则。推荐装 iniVation 官方软件包（它会带上规则），或者手动添加 `/etc/udev/rules.d/65-inivation.rules`：
  `SUBSYSTEM=="usb", ATTR{idVendor}=="152a", ATTR{idProduct}=="84[0-1]?", MODE="0666"`
  然后 `sudo udevadm control --reload-rules && sudo udevadm trigger`，并重新插拔相机。以 iniVation 官方文档为准。
- **Windows**：dv-processing 有 win_amd64 轮子。相机驱动随 iniVation 的 DV 软件安装。
- 检查相机能否打开：`python -c "import dv_processing as dv; print(dv.io.camera.discover())"`。dv-processing 1.x 的接口是 `dv.io.CameraCapture`，本包两种都支持。

## 3. 先自检（不需要相机）

```bash
python verify_install.py                 # 默认 GPU；无 GPU 时自动用 CPU
python verify_install.py --weights weights/s39_s3407.pt
```

输出 `PASS` 即表示本机的事件映射、LNES、渲染、网络和滤波都与服务器一致。

- 同样的 GPU 型号加 torch 2.1 时会显示 `bit-identical`。
- 其他 GPU、CPU 或 torch 版本会有浮点差异，闭环会把它略微放大，所以判据是两边 RA-MPJPE 相差不超过 0.15 mm。服务器 CPU 实测差 0.004–0.04 mm。

## 4. 相机标定（内参，必须做）

网络要用相机内参，把上一步的手渲染到图像里再与事件比较，所以内参错了，跟踪就会系统性偏。

1. 用 DAVIS346 的 APS 灰度帧加棋盘格标定。可以用 iniVation DV 软件的相机标定模块，或者 OpenCV 的 `calibrateCamera`，得到 346×260 像素下的 `K` 和畸变 `dist`。
2. 复制 `examples/calib_TEMPLATE.json` 为 `calib.json`，填入 `K`、`dist`，**删除 `"_template"` 这一行**。
3. 可选项：
   - `"betas"`：你手的 10 维 MANO 形状。不填就用平均手；按实测，手型不准会让误差多约 0.3 mm，而且不会随时间消失。
   - `"rot180": true`：相机倒装时使用。

**几何换算**（代码已处理，供核对）：事件按 `x·240//346`、`y·180//260` 映射到网络的 240×180 输入，与训练数据 `x·240//640` 的做法相同。传给网络的内参是 `diag(640/346, 480/260, 1)·K`，即训练相机的 640×480 口径；网络内部再乘 0.375。在服务器上把 zgz 事件重采样到 346×260 做过测试：精度与原始数据一致（local 10.23 vs 10.27 mm），内参也精确还原。

## 5. 噪声与事件计数标定（强烈建议）

```bash
# 场景静止、画面里没有手，录 5 s，屏蔽热像素
python calibrate_davis346.py hotpix --seconds 5 --out hotpix.npy
# 在跟踪距离上像平时那样活动右手，录 20 s；推荐值会写进 calib.json
python calibrate_davis346.py counts --seconds 20 --hotpix hotpix.npy --calib calib.json
```

**为什么要标定计数**：S39 的滤波增益（每步吸收多少网络更新）随"最近 50 ms 的事件数"变化：事件少时几乎不更新，事件多时更新得多。这条曲线是按训练相机定的。DAVIS346 的像素数只有训练相机的约 1/3.4，灵敏度和噪声也不同，必须把原始计数换算到训练相机的量级。
- 计数换算偏小，手会跟不上、发粘；
- 计数换算偏大，输出会抖。

**标定方法**：

计数尺度 = (640·480)/(W·H) × 训练值 ÷ 本相机值

其中"值"是 50 ms 内每个有事件的像素平均触发的事件数。这个量基本只反映传感器灵敏度，与手动得快慢无关：训练集 9 条 local 序列都在 1.23–1.51 之间，而它们的事件数相差 10 倍以上。
- 默认用 local 参考（手在原地活动），整条手臂大幅移动的场景用 `--reference global`。
- 自测：在训练相机自己的数据上，用对应参考给出的推荐值为 1.09 和 1.08（真值 1.0）。
- 前提是手在 240×180 图里约 60–90 px 宽。查看器左上角会实时显示这个尺寸。
- 没有标定时，默认系数为像素数比 3.41。

## 6. 运行

```bash
python run_davis346.py --calib calib.json                     # 实时，默认权重 s39_s3408.pt
python run_davis346.py --calib calib.json --show-aps          # 在 APS 灰度帧上叠加，对准模板更容易
python run_davis346.py --calib calib.json --init my_init.json # 用你自己的初始姿态，事件一到立即开始跟踪
python run_davis346.py --calib calib.json --save run.npz --save-verts --video run.mp4
```

**操作流程**：
1. 窗口里会出现黄色模板手（右手张开、掌心朝相机、手指向上）。把右手放到模板上，尽量对齐位置、大小和朝向。用 `+`/`-` 调整模板大小（调整的是期望距离）。
2. 按 **SPACE** 开始跟踪，网格变成绿色。跟丢、手离开画面或被遮挡时，把手放回模板，再按 SPACE 重新开始。

| 按键 | 作用 |
|---|---|
| `SPACE` | 从模板（或 `--init` 文件）重新开始跟踪 |
| `T` | 停止跟踪 |
| `+` / `-` | 调整模板大小 |
| `S` | 立即保存 |
| `Q` / `ESC` | 退出 |

**测试录制**：先用 DV 软件录一段 AEDAT4，再：
- `python run_davis346.py --source rec.aedat4 --calib calib.json`：尽快处理完；
- `--source sim:rec.aedat4`：按真实时间节奏回放，走和真机完全相同的实时代码路径，可以在没有相机时测试实时逻辑。

**输出**（`--save`）：npz 包含以下字段，全部在相机坐标系（x 右、y 下、z 前，米）：
- `end_ms`：每步结束时刻，会话时钟，`t0_us` 为原点；
- `state51`：每步的 51 维状态；
- `joints21`：OpenPose 顺序的 21 个关节；
- `verts778`：MANO 顶点（加 `--save-verts` 时才有）；
- `n50`：最近 50 ms 的事件数；
- `compute_ms`：每步计算耗时；
- `K_model`、`betas`。

## 7. 初始状态

S39 是**跟踪器**：每一步都从上一步的状态出发，第一步需要一个初始状态。

- **`--init template`（默认）**：模板姿态放在画面中心，按内参算好深度，使手在 240×180 图里约 72 px 宽。
- **`--init file.json`**：用你自己的初始化方法（例如基于 APS 帧的 RGB 手部估计）得到的状态，格式见 `examples/`：
  - `state51`：`[transl(3), global_orient(3), hand_pose(45)]`。轴角表示，`hand_pose` 是相对 MANO `hands_mean` 的残差，与 smplx MANO（`flat_hand_mean=False`）一致；
  - 或者给 MANO 的各个键：`transl` / `global_orient` / `hand_pose` / `flat_hand_mean`。若 `hand_pose` 是完整姿态，令 `flat_hand_mean` 为 true；
  - 都在**本跟踪器的相机坐标系**下。`--rot180` 时是旋转后的坐标系。

**初始误差的影响**（zgz，三种子均值；只改初值，其余与官方相同）：

| 初值 | 初值 RA 误差 / 平移 / 根旋转 | local MPJPE | global MPJPE |
|---|---|---|---|
| 完全准确 | 0 / 0 / 0° | 10.18 | 7.34 |
| 官方协议噪声 | 6.4 mm / 6.5 mm / 4.1° | 10.27 | 7.36 |
| ×4 噪声 | 25 mm / 26 mm / 16° | 11.11 | 7.44 |
| 深度偏 3 cm | 6.4 mm / 31 mm / 4.1° | 10.27 | 7.36 |
| 平均手型（不用真值 β） | 6.6 mm / 6.5 mm / 4.1° | 10.59 | 7.67 |

初值误差在约 **1 秒**内被纠正：即使初值误差达到 16°，1 秒后的结果也与初值完全准确时相同。因此初始化不必很精确，但**手型 β 值得标定一次**。S39 **没有全局重定位**，跟丢后需要人工重新初始化（按 SPACE，或调用 `reset()`）。

## 8. 在自己的代码里调用

```python
import numpy as np
from eh346.events import SensorMap, EventBuffer
from eh346.tracker import S39Tracker

smap = SensorMap(346, 260, K, dist)                       # 你的标定
trk = S39Tracker(["weights/s39_s3408.pt"], device="cuda", count_scale=3.41)   # 用标定得到的 count_scale
trk.set_camera(smap.K_model()); trk.warmup()
buf = EventBuffer(t0_us=first_event_timestamp_us - first_event_timestamp_us % 1000)

# 每收到一批事件（时间有序；t 单位为 µs，x / y 是传感器像素，p 取 0/1）
x2, y2, p2, ok = smap.map(x, y, p); buf.push(t[ok], x2, y2, p2)

trk.reset(init_state51, start_ms=buf.latest_ms + 1, betas=None)   # 开始跟踪
end = trk.first_end()
# 主循环：buf.latest_ms > end 时，说明窗口内的事件已经到齐
out = trk.step(buf, end); end += trk.step_ms
verts, joints = trk.mesh()                                 # (778, 3)、(21, 3)，相机坐标系，单位米
```

单步协议：窗口 w = min(200, end − 开始时刻 + 1)，事件计数取最近 50 ms，滤波后的状态既是输出，也是下一步的前态。

## 9. 参数与性能

| 参数 | 说明 |
|---|---|
| `--weights a.pt,b.pt,c.pt` | 三种子集成：local 9.84 / global 7.11 mm（单模型为 10.27 / 7.36），计算量 ×3 |
| `--step-ms 10` | 100 Hz 输出（默认 50 ms，即 20 Hz）。增益已自动按步长折算，计数按 50 ms 速率换算。zgz 单种子实测：local 11.09 vs 10.68 mm，抖动不变，滤波延迟 local +5 ms、global +16 ms；但等待输出的时间从平均 25 ms 降到 5 ms，总延迟仍然更低 |
| `--cuda-graph` | CNN 用 CUDA graph 回放，输出逐位相同 |
| `--ba-us 2000` | 相机片上背景活动滤波（μs），`0` 为关闭；`--sw-ba-us` 是软件滤波 |
| `--count-scale` | 覆盖 calib.json 中的计数尺度 |
| `--rot180` | 相机倒装 |
| `--device cpu` | 无 GPU 时使用 |

**实测每步计算时间**（构建 W200 LNES + 渲染前态 + 网络 + 滤波，batch 1）：

| 环境 | p50 | p99 |
|---|---|---|
| L20 连续处理 | 5.0 ms | 6.3 ms |
| L20 按 20 Hz 实时节奏 | 12.1 ms | 18.1 ms |
| Xeon 8 线程 CPU | 8.5 ms | 11.3 ms |

都远小于 50 ms 的步长。按实时节奏变慢，是因为 GPU 在两步之间空闲、降频，`--cuda-graph` 在这种情况下帮助不大。需要更低延迟时可以锁定 GPU 频率：Linux 用 `sudo nvidia-smi -lgc <最低>,<最高>`，Windows 在 NVIDIA 控制面板选"最高性能优先"。

**端到端延迟** ≈ 等待下一次输出（平均 25 ms）+ 计算（5–12 ms）+ 滤波带来的约 11 ms 时间延迟。

## 10. 已知限制（请先读）

1. **训练数据是仿真事件，这是最大的风险。** 训练和评测的事件都是 **v2e 从 RealSense D435 RGB 视频仿真出来的**（640×480，60 fps 源视频）；标签是 WiLoR 加 RGB-D 优化得到的伪真值。模型**从未见过真实事件相机的数据**。DAVIS346 的噪声、阈值、延迟和极性特性都和仿真不同，精度预计会比 zgz 上的数字差，差多少只能实测。建议在光照充足、背景静止的场景下使用。
2. **只支持右手**（MANO 右手模型）。相机安装方向要使手的视角接近训练分布：手在相机前约 0.5–0.75 m，D435 水平视场约 56°。手的尺寸以查看器显示的 60–90 px 为准。
3. **local 小幅动作跟不满**：在 zgz 上，250 ms 内的手部位移只跟上约 65–70%；大幅动作约 97%。时间延迟约 11 ms，主要来自滤波。
4. **绝对深度有偏差**：zgz 上绝对平移误差约 4 cm，主要在深度方向；图像平面误差只有 2–4 px。画面叠加没问题，需要绝对 3D 位置的应用要注意。
5. **没有全局重定位**：跟丢后要重新初始化（第 7 节）。
6. **计数尺度是近似标定**：在训练相机上自测误差约 10%。若手明显"发粘"可以适当调大，抖动明显则调小。

## 11. 一致性验证记录（训练服务器，2026-10-06）

| 测试 | 结果 |
|---|---|
| `verify_install.py`，三个种子（1244 步，含 42 段协议初值） | 与官方输出逐位相同 |
| dv-processing 读原始 AEDAT4，完整 zgz 2590 步，三个种子 | 逐位相同：RA 8.9115 / 8.5271 / 8.6899，与官方主表一致 |
| `run_davis346.py` 主循环回放（zgz_global 69 s，1386 步） | 逐位相同 |
| 模拟实时 `sim:`，含 `--cuda-graph`（20 s） | 逐位相同，无跳步 |
| 三种子集成 | 与此前独立评测逐位相同：8.3777 / 7.1051 / 9.8427 |
| `--step-ms 10` | 与此前独立的 100 Hz 评测一致：local 11.093 / global 7.430 |
| 346×260 映射（zgz 重采样）+ 缩放内参 | local 10.59 / 10.16 / 9.95 vs 官方 10.68 / 10.19 / 9.94 |
| CPU（8 线程） | PASS：RA 差 ≤ 0.04 mm |

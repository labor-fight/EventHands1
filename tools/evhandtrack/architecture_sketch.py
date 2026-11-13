#!/usr/bin/env python3
r"""EvHandTrack 完整网络结构与信息传递（给画图 AI 用的说明文件）

====================================================================================================================
0. 用法与画图约定
====================================================================================================================
* 本文件只依赖 torch，可直接运行，做形状与参数量自检（随机数据、CPU、不读任何检查点）：
      python tools/evhandtrack/architecture_sketch.py
* 画图时按"方框 = 模块、箭头 = 张量（标形状）"来画。文件末尾的 GROUPS / FIGURE_BLOCKS / FIGURE_ARROWS / PANELS
  是结构化清单，可以直接照着画。
* ESMNet（Event Surface Mapping Network，事件→手表面映射网络）这次不展开，只画成一个"预训练权重（冻结）"方框，
  标出它的输入、输出即可。它在训练和部署中都是冻结的，不参与 EvHandTrack 的梯度更新。
* 方框分类与配色建议（见 GROUPS）：输入（灰）、冻结权重（蓝灰 + 锁）、无参数几何运算（白底虚线）、
  可学习模块（橙）、状态与闭环（绿）、只在训练时出现（红色虚线）。

====================================================================================================================
1. 一句话
====================================================================================================================
EvHandTrack 是事件驱动的 MANO 手部闭环跟踪器：每个更新间隔（50 ms），用冻结的 ESMNet 把事件映射到手表面，
在上一状态的 MANO 网格上算出"事件射线与前态表面的偏差"，按 16 个手部部位汇聚成部位节点，再在
"16 个部位节点 + 1 个手节点"的运动学树图上做 3 轮门控消息传递（门由前态关节几何算出），
最后每个关节局部读出自己的增量，加到上一状态上得到新状态，新状态又作为下一次更新的前态。

====================================================================================================================
2. 总流程（一次状态更新 k：x_{k-1} → x_k）
====================================================================================================================

 事件流 e_i = (x_i, y_i, t_i, p_i)                                  上一状态 x_{k-1} ∈ R^51
        │                                                                │
 [1 流式取样 + 7 维事件 token]（无参数）                         [3 前态 MANO 前向运动学]（无参数）
        │ 保留事件 N 个；token  N × 7                                    │ 网格 V 778×3、关节 J 16×3、
        │                                                                │ 部位坐标系 F 16×3×3
        ├──────────────►[2 ESMNet（预训练权重，冻结）]                    │
        │                 每个保留事件：在手上？哪个部位？模板表面点         │
        │                 → 表面路由：面 f_i、重心坐标 b_i                  │
        │                           │                                    │
        └─────────►[4 表面关联 Surface association]（无参数）◄───────────┘
                    P_i  = 路由点在前态网格上的位置
                    z_i  = P_i 的深度；d_i = 事件像素的相机射线
                    res_i = z_i d_i − P_i（事件射线与前态表面的偏差，"测量—预测"残差）
                    ω_ip = 事件对 16 个部位的软权重（LBS 蒙皮权重）
                            │
                    [5 事件编码 φ]（可学习，逐事件共享 MLP 12 → 64 → 128）
                            │
                    [6 节点输入构造]（无参数）：按部位加权平均 + 几何矩 + 前态切片
                            │ 部位节点 16 × 156、手节点 283
                    [7 节点编码]（可学习）→ h^0 ∈ R^{17×128}
                            │                     [8 状态相关边门控]（可学习，读前态关节几何）
                            │                       g_uv ∈ R^{62×128}，各轮共享
                    [9 运动学树传播 P，3 轮]（可学习；62 条有向边、4 种边型）
                            │ h^3 ∈ R^{17×128}
                    [10 局部读出]（可学习）：15 个关节头 → Δθ_j（各 3）；手节点头 → Δt（3）、Δθ_root（3）
                            │ Δ ∈ R^51
                    [11 加性状态更新]  x_k = x_{k-1} + Δ   ──────────► 作为下一次更新的前态（闭环）

====================================================================================================================
3. 与代码的对应（论文名 ↔ 代码标识；改名只是论文命名，不重训、不改模型）
====================================================================================================================
  EvHandTrack        PMR 设计 D1、variant "r"（semkine/pmr_trunk.py 的 PMRNet）+ rol5 微调（semkine/tpd_rol.py）
  ESMNet             SRNet 的论文名；部署权重是 r_s2d（semkine/pfo_router.py 的 SlimSR，stem "s2d"），
                     outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt（137,252 参数，约 74.5 M MACs）
  表面关联           PFA（semkine/s45sm.py 的 pfa_association）
  流式输入           semkine/tpd_str.py（论文主行用流式形式）
  外层模型           model/model.py 的 MNISTModel._xyz_mesh_forward / forward_packet
  检查点（三种子）   outputs/pmr/20261109/runs/pmr_d1r_rol5_s{3407,3408,3409}/pmr_d1r_rol5_s{种子}-step=1500.ckpt
  身份记录           outputs/evhandtrack/identity.json（python -m tools.evhandtrack.identity verify）
  成本（流式）       17.45 ms / 0.102 G MACs / 0.90 M 参数（主干 761,427 + ESMNet 137,252）

本文件第 5–8 节的主干代码与 semkine/pmr_trunk.py（variant "r"）逐模块同构：参数名、形状、计算顺序一致，
参数量 761,427 与真实模型相同（自检会断言）。为便于阅读，几何矩用直接公式而非代码里的数值稳定写法，
数学上相同。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn

# ==================================================================================================================
# 1. 常量（全部取自 EvHandTrack 的配置与代码）
# ==================================================================================================================
IMG_W, IMG_H = 240, 180          # 事件相机画布（hand_data51），像素
K_SCALE = 240.0 / 640.0          # 相机内参从 640×480 缩到 240×180 的比例（RENDER_SCALE 0.375）
UPDATE_MS = 50                   # 状态更新间隔 Δ（评测网格 50 ms；rol5 训练的包也是 50 ms）
WINDOW_MS = 50                   # 流式取样里估计事件率用的回看窗口 W
BUDGET = 2048                    # 每个更新间隔大约保留的事件数上限（流式规则按事件率自适应取样）
TOKEN_DIM = 7                    # 每个事件的 7 维 token
EPS_DT = 1e-6                    # token 里 log1p(间隔 / eps) 的 eps（秒）

N_VERTS, N_FACES = 778, 1538     # MANO 右手网格
N_PARTS = 16                     # MANO 16 个关节 / 蒙皮部位
HUB = 16                         # 手节点（hub）的下标
N_NODES = N_PARTS + 1            # 图上共 17 个节点
#: MANO 运动学树：部位 p 的父关节（0 = 手掌 / 腕，-1 = 无父）
PARENT = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
#: MANO 关节顺序（部位 p = 关节 p 驱动的那一段）
PART_NAMES = ("palm(wrist)", "index1", "index2", "index3", "middle1", "middle2", "middle3",
              "pinky1", "pinky2", "pinky3", "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3")

C = 128                          # 节点隐藏宽度（MODEL.ENCODER_HIDDEN）
N_ROUNDS = 3                     # 传播轮数（MODEL.ENCODER_LAYERS）
REL_HIDDEN = 64                  # 事件编码 φ 的隐藏宽度（MODEL.XYZ_REL_HIDDEN）
EDGE_DIM = 32                    # 边几何嵌入宽度
JOINT_HEAD = 64                  # 关节读出头隐藏宽度
HUB_HEAD = 128                   # 手节点读出头隐藏宽度
PHI_IN = 7 + 3 + 1 + 1           # φ 输入 = token 7 + 残差 3 + 深度 1 + 在手上标记 1 = 12
PART_STATS = 1 + 3 + 3 + 9 + 6 + 3   # 部位节点的固定统计量 = 25
HUB_STATS = PART_STATS - 3       # 手节点的固定统计量 = 22（没有视线方向）
EDGE_IN = 3 + 1 + 1 + 1 + 4      # 边几何特征 = 10
EDGE_TYPES = ("down: 父→子", "up: 子→父", "hub2part: 手→部位", "part2hub: 部位→手")

L_RES = 0.01                     # 残差的单位（米 / 单位）
L_LEV = 0.02                     # 力臂（事件点 − 关节）的单位
L_EDGE = 0.02                    # 边向量的单位
Z_REF, Z_SCALE = 0.45, 0.10      # 深度归一化 (z − 0.45) / 0.1
Z_MIN = 0.05                     # 表面关联的深度下限（米）
ONHAND_TAU = 0.5                 # ESMNet 在手上概率 ≥ 0.5 才路由到表面

#: 51 维状态 x 的布局（MANO 全轴角表示，PREDICT_DELTA：网络输出增量）
STATE_LAYOUT = {
    "transl": slice(0, 3),       # 根平移（相机系，米）
    "root": slice(3, 6),         # 根（腕）全局旋转，轴角
    "fingers": slice(6, 51),     # 15 个手指关节 × 3 轴角（相对 MANO 平均手型 hands_mean）
}
#: prev48 = x[3:51] = [根 3, 15 个手指关节 × 3]，按关节 0..15 排好，每个部位节点取自己那 3 维


# ==================================================================================================================
# 2. 输入
# ==================================================================================================================
#   * 事件流：每个事件 (x, y, t, p)，像素坐标、时间（微秒）、极性 0/1。
#   * 上一状态 x_{k-1} ∈ R^51（闭环：上一次更新的输出）。
#   * 手形参数 betas ∈ R^10（序列给定，不预测）与相机内参 K（3×3，按 K_SCALE 缩到 240×180 画布）。


# ==================================================================================================================
# 3. 第 1 步：流式取样 + 7 维事件 token（无参数；每个事件到达时只用它自己和之前的事件就能算出）
# ==================================================================================================================
class StreamFrontEnd:
    r"""逐事件到达的流式前端（semkine/tpd_str.py 的 Stream / stream_nodes / stream_tokens）。

    每个事件 i 到达时（t 为微秒，t_q 为上一次状态更新时刻，Δ = UPDATE_MS）：
      事件率   n_i = 流中在 i 之前到达、且时间 ≥ t_i − W 的事件数                 （W = 50 ms）
      步长     k_i = max(1, ceil(n_i / BUDGET))                                   （BUDGET = 2048）
      保留     c_i mod k_i == 0，c_i 为流的事件计数器                               （不截断）
      token    [ x/240, y/180, 2p−1, (t_i − t_q)/Δ, log1p(与流中前一事件的时间间隔 / eps),
                 min(t_i − 同像素同极性上次时间, Δ)/Δ, min(t_i − 同像素异极性上次时间, Δ)/Δ ]
               像素"上次时间"跨更新保持，不在更新时清零；从没触发过的像素记为 Δ（即 1.0）。
    update() 在每个更新时刻调用：返回本间隔保留的事件及其 token，作为这一次状态更新的输入。
    训练（两个阶段）用的是普通分包形式：包内均匀步长取 ≤ 2048 个事件、token 在包内算；
    流式形式只用于评测与部署（论文主行 = 训练好的权重零训练换成流式读数）。
    """

    def __init__(self, delta_ms: int = UPDATE_MS, window_ms: int = WINDOW_MS, budget: int = BUDGET):
        self.delta_us = delta_ms * 1000
        self.window_us = window_ms * 1000
        self.budget = budget
        self.counter = 0                         # c_i
        self.recent: List[int] = []              # 最近 W 内的到达时间（估事件率）
        self.last_t = None                       # 流中前一事件时间
        self.pix_last: Dict[Tuple[int, int, int], int] = {}   # (x, y, p) → 上次时间
        self.t_q = 0                             # 上一次状态更新时刻
        self.kept: List[Tuple[int, int, int, int]] = []
        self.tokens: List[List[float]] = []

    def ingest(self, x: int, y: int, t: int, p: int):
        self.recent = [s for s in self.recent if s >= t - self.window_us]
        n_i = len(self.recent)
        k_i = max(1, math.ceil(n_i / self.budget))
        if self.counter % k_i == 0:
            gap = 0 if self.last_t is None else t - self.last_t
            same = self.pix_last.get((x, y, p))
            opp = self.pix_last.get((x, y, 1 - p))
            age_s = self.delta_us if same is None else min(t - same, self.delta_us)
            age_o = self.delta_us if opp is None else min(t - opp, self.delta_us)
            self.kept.append((x, y, t, p))
            self.tokens.append([x / IMG_W, y / IMG_H, 2.0 * p - 1.0, (t - self.t_q) / self.delta_us,
                                math.log1p(gap * 1e-6 / EPS_DT), age_s / self.delta_us, age_o / self.delta_us])
        self.recent.append(t)
        self.last_t = t
        self.pix_last[(x, y, p)] = t
        self.counter += 1

    def update(self, t_now: int):
        """返回 (保留事件 N×4 [x, y, t, p], token N×7)，并开始下一个更新间隔。"""
        ev = torch.tensor(self.kept, dtype=torch.float64).view(-1, 4)
        tok = torch.tensor(self.tokens, dtype=torch.float32).view(-1, TOKEN_DIM)
        self.kept, self.tokens, self.t_q = [], [], t_now
        return ev, tok


def event_rays(ev_xy: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """事件像素的相机射线 d_i = K_r^{-1} [u, v, 1]（d_z = 1，所以射线参数就是深度）。ev_xy (B, N, 2) → (B, N, 3)。"""
    fx, fy = K[:, 0, 0] * K_SCALE, K[:, 1, 1] * K_SCALE
    cx, cy = K[:, 0, 2] * K_SCALE, K[:, 1, 2] * K_SCALE
    u, v = ev_xy[..., 0], ev_xy[..., 1]
    return torch.stack([(u - cx[:, None]) / fx[:, None], (v - cy[:, None]) / fy[:, None], torch.ones_like(u)], -1)


# ==================================================================================================================
# 4. 第 2 步：ESMNet（预训练权重，冻结；这次不展开画）
# ==================================================================================================================
class ESMNetFrozen(nn.Module):
    r"""ESMNet：只看当前更新间隔的事件（不读前态、不读 K、不读 betas），给每个保留事件一个手表面位置。

    画图：一个带锁的方框，标"ESMNet（预训练权重，冻结）"，只画输入与输出。
      输入   本更新间隔的事件图 4 × 180 × 240（每个极性：最后事件时间/Δ、log1p(事件数)）+ 保留事件的像素位置与 token
      输出   每个保留事件 i：
               在手上概率 o_i ∈ [0, 1]
               部位 logits ∈ R^16
               MANO 静息模板上的表面点 ∈ R^3（厘米）
      解码（无参数）
               routed_i = (o_i ≥ 0.5)
               部位 = argmax；面 f_i = 该部位（按面三个顶点的主蒙皮部位划分）里模板质心离预测点最近的三角面
               重心坐标 b_i = 预测点投影到该面上的重心坐标（截断到非负并归一化）
    权重：r_s2d_final.ckpt，137,252 参数；训练 EvHandTrack 的两个阶段和部署时都用同一份冻结权重。
    部署时它按更新间隔运行一次（豁免逐事件异步要求），在主干保留的那些事件上取输出。
    （内部结构仅供参考，不画：2×2 像素重排 → 1×1 卷积 → 4 级残差下采样 → FPN 回到 1/4 分辨率 →
      在每个事件像素处取特征并拼上 token → 逐事件 MLP → 三个线性头。）
    """

    N_PARAMS = 137_252

    def forward(self, event_image: torch.Tensor, node_xy: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("ESMNet 用预训练权重代替：r_s2d_final.ckpt（见本文件第 3 节）")


def esmnet_placeholder_routes(B: int, N: int, faces: torch.Tensor, gen: torch.Generator):
    """自检用：随机给出 ESMNet 解码后的路由 (routed (B,N) bool, face (B,N) long, bary (B,N,3))。"""
    routed = torch.rand(B, N, generator=gen) < 0.8
    face = torch.randint(0, faces.shape[0], (B, N), generator=gen)
    bary = torch.distributions.Dirichlet(torch.ones(3)).sample((B, N))
    return routed, face, bary


# ==================================================================================================================
# 5. 第 3 步：前态 MANO 前向运动学（无参数；训练阶段 2 里对前态可微）
# ==================================================================================================================
def rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) 轴角 → (..., 3, 3) 旋转矩阵。"""
    shape = aa.shape[:-1]
    a = aa.reshape(-1, 3)
    th = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = a / th
    Kx = torch.zeros(a.shape[0], 3, 3, dtype=a.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th.unsqueeze(-1)
    R = torch.eye(3, dtype=a.dtype) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)
    return R.view(*shape, 3, 3)


def node_frames(prev48: torch.Tensor, hands_mean: torch.Tensor) -> torch.Tensor:
    r"""(B, 16, 3, 3) 每个部位节点的前态坐标系 F_p。

    姿态 = [根 3, 手指 45 + hands_mean]；沿运动学树连乘得到每个关节的全局旋转 G_j = G_parent(j) R_j。
    部位 p（p ≥ 1）的坐标系 = 其父关节的全局旋转 G_parent(p)；手掌（p = 0）用单位阵（相机系），
    因为它的增量就是相机系下的根旋转。F 的列 = 该坐标系三个轴在相机系中的方向。
    """
    B = prev48.shape[0]
    pose = torch.cat([prev48[:, 0:3], prev48[:, 3:48] + hands_mean.view(1, -1)], 1).view(B, 16, 3)
    rot = rodrigues(pose)
    G = [rot[:, 0]]
    for j in range(1, N_PARTS):
        G.append(G[PARENT[j]] @ rot[:, j])
    eye = torch.eye(3, dtype=prev48.dtype).expand(B, 3, 3)
    return torch.stack([eye] + [G[PARENT[j]] for j in range(1, N_PARTS)], 1)


def prior_geometry(prev: torch.Tensor, betas: torch.Tensor, mano: Dict[str, torch.Tensor]):
    r"""前态几何：
      V = MANO(betas, x_{k-1})                 (B, 778, 3) 相机系网格（米）
      J = J_regressor · V                      (B, 16, 3)  前态关节（手掌 J_0 = 腕）
      F = node_frames(prev48)                  (B, 16, 3, 3)
    真实代码用 ManoLayer（assets/mano_right.npz）做 LBS；这里网格由 mano["verts_fn"] 给出（自检用随机网格）。
    """
    V = mano["verts_fn"](prev, betas)
    J = torch.einsum("jv,bvk->bjk", mano["J_regressor"], V)
    F = node_frames(prev[:, 3:51], mano["hands_mean"])
    return V, J, F


# ==================================================================================================================
# 6. 第 4 步：表面关联（Surface association，无参数；前态只进入几何量，不决定事件归属）
# ==================================================================================================================
def surface_association(routed, face, bary, live, verts, faces):
    r"""每个被 ESMNet 判为在手上的事件 → 它的路由面 f_i 与重心坐标 b_i，挂到前态网格上。

      vid_i  = f_i 的三个顶点（第 4 个槽位填重复顶点、权重 0）              (B, N, 4)
      A_i    = [b_i, 0]                                                    (B, N, 4)
      q_i    = 1（路由）/ 0（不在手上 → 整个事件进"残差槽"）               (B, N)
      z_i    = 路由点 P_i = Σ b_ir V[vid_ir] 在前态网格上的深度，≥ 0.05 m   (B, N)
    不做任何射线求交：事件归属只由 ESMNet 决定，前态只决定"这个表面点现在在哪里"。
    """
    B, N = routed.shape
    ok = routed & live
    fv = faces[face]                                                         # (B, N, 3)
    vid = torch.cat([fv, fv[..., :1]], -1) * ok[..., None]
    A = torch.cat([bary, torch.zeros_like(bary[..., :1])], -1) * ok[..., None]
    bidx = torch.arange(B).view(B, 1, 1)
    P = (bary[..., None] * verts[bidx, fv]).sum(-2)                          # (B, N, 3)
    z = P[..., 2].clamp_min(Z_MIN) * ok
    return {"vid": vid, "A": A, "q": ok.float(), "z": z}


# ==================================================================================================================
# 7. 第 5–6 步：事件几何量、部位软权重、几何矩（无参数，在 φ 之外算；阶段 2 训练对前态可微）
# ==================================================================================================================
def event_geometry(w, d, z, vid, verts):
    r"""每个事件：在手上质量 m_i = Σ_r w_ir，路由点 P_i = Σ_r w_ir V[vid_ir] / m_i，
    残差 res_i = z_i d_i − P_i（相机系）。表面关联下 z_i = P_i 的深度，所以残差在路由点深度处的像平面里：
    它就是"事件实际落在哪条射线"与"前态把这个表面点放在哪里"的偏差。不在手上的事件三项都为 0。"""
    m = w.sum(-1)
    B = verts.shape[0]
    bidx = torch.arange(B).view(B, 1, 1)
    P = (w.unsqueeze(-1) * verts[bidx, vid]).sum(2) / m.clamp_min(1e-12).unsqueeze(-1)
    on = (m > 0).unsqueeze(-1)
    res = torch.where(on, z.unsqueeze(-1) * d - P, torch.zeros_like(P))
    return m, torch.where(on, P, torch.zeros_like(P)), res


def part_weights(w, vid, lbs_w):
    """部位软权重 ω_ip = Σ_r w_ir · LBS[vid_ir, p]，(B, N, 16)；对 p 求和等于 m_i。"""
    return torch.einsum("bnr,bnrp->bnp", w, lbs_w[vid])


def node_moments(weight, P, res, origin, frame):
    r"""每个节点的加权几何矩（在节点坐标系里表达），返回质量 M (B, K) 与 22 维特征：
      [ log1p(M),
        平均残差 E[res] / L_RES                            (3)
        平均力臂 E[lever] / L_LEV，lever = P − 节点原点       (3)
        力臂×残差 E[lever res^T] / (L_LEV L_RES)            (9)
        力臂×力臂 E[lever lever^T] / L_LEV² 的上三角         (6) ]
    所有统计量都是对事件求和（可逐事件累加，没有 max 池化）；没有事件的节点特征全为 0。
    weight (B, N, K)，P / res (B, N, 3)，origin (B, K, 3)，frame (B, K, 3, 3)。"""
    M = weight.sum(1)
    Wn = weight / M.clamp_min(1e-12).unsqueeze(1)
    lever = P.unsqueeze(2) - origin.unsqueeze(1)                              # (B, N, K, 3)
    mr = torch.einsum("bnk,bnc->bkc", Wn, res)
    ml = torch.einsum("bnk,bnkc->bkc", Wn, lever)
    T = torch.einsum("bnk,bnkc,bnd->bkcd", Wn, lever, res)
    G = torch.einsum("bnk,bnkc,bnkd->bkcd", Wn, lever, lever)
    Ft = frame.transpose(-1, -2)
    mr = torch.einsum("bkij,bkj->bki", Ft, mr) / L_RES
    ml = torch.einsum("bkij,bkj->bki", Ft, ml) / L_LEV
    T = (Ft @ T @ frame) / (L_LEV * L_RES)
    G = (Ft @ G @ frame) / (L_LEV * L_LEV)
    iu = torch.triu_indices(3, 3)
    feats = torch.cat([torch.log1p(M).unsqueeze(-1), mr, ml, T.flatten(-2), G[..., iu[0], iu[1]]], -1)
    return M, feats * (M > 0).unsqueeze(-1).to(feats.dtype)


def kinematic_edges() -> List[Tuple[int, int, int]]:
    """62 条有向边 (源, 目标, 边型)：树的 15 条边两个方向（30 条）+ 手节点与 16 个部位双向（32 条）。"""
    out = []
    for c, p in enumerate(PARENT):
        if p >= 0:
            out.append((p, c, 0))          # down：父 → 子
            out.append((c, p, 1))          # up：子 → 父
    for p in range(N_PARTS):
        out.append((HUB, p, 2))            # hub2part：手 → 部位
        out.append((p, HUB, 3))            # part2hub：部位 → 手
    return out


def edge_features(J, src, dst, etype):
    r"""每条边的 10 维前态几何（门控的输入）：两端锚点之间的向量 / L_EDGE（3）、长度（1）、
    像平面占比 |v_xy| / |v|（透视缩短，1）、目标锚点深度 (z − 0.45)/0.1（1）、边型 one-hot（4）。
    锚点：部位节点 = 它的前态关节 J_p；手节点 = 腕关节 J_0。"""
    anchor = torch.cat([J, J[:, :1]], 1)                                      # (B, 17, 3)
    v = anchor[:, dst] - anchor[:, src]
    ln = v.norm(dim=-1, keepdim=True)
    fs = v[..., :2].norm(dim=-1, keepdim=True) / ln.clamp_min(1e-9)
    zd = (anchor[:, dst, 2:3] - Z_REF) / Z_SCALE
    oh = torch.nn.functional.one_hot(etype, 4).to(J.dtype).unsqueeze(0).expand(J.shape[0], -1, -1)
    return torch.cat([v / L_EDGE, ln / L_EDGE, fs, zd, oh], -1)


# ==================================================================================================================
# 8. 第 5–10 步：EvHandTrack 主干（可学习部分全在这里，761,427 参数）
# ==================================================================================================================
class NodeLinear(nn.Conv1d):
    """K 个互相独立的线性映射（每个节点一套权重），实现为分组 1×1 卷积。(B, K, in) → (B, K, out)。"""

    def __init__(self, k, n_in, n_out, zero=False):
        super().__init__(k * n_in, k * n_out, 1, groups=k)
        self.k, self.n_in, self.n_out = k, n_in, n_out
        if zero:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        B = x.shape[0]
        return super().forward(x.reshape(B, self.k * self.n_in, 1)).view(B, self.k, self.n_out)


class PropagationRound(nn.Module):
    """传播 P 的一轮：每种边型一个无偏置消息映射 W_t（4 个），一个所有节点共享的无偏置自映射 W_s。"""

    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in EDGE_TYPES])
        self.slf = nn.Linear(hidden, hidden, bias=False)


class EvHandTrackTrunk(nn.Module):
    r"""EvHandTrack 主干（= semkine/pmr_trunk.py 的 PMRNet，variant "r"）。

    可学习模块与参数量：
      phi        事件编码 φ：Linear(12→64) – ReLU – Linear(64→128)，所有事件共享             9,152
      enc_part   部位节点编码：16 个独立的 Linear(156→128)（无激活）                         321,536
      enc_hub    手节点编码：Linear(283→128)（无激活）                                        36,352
      edge_emb   边几何嵌入：Linear(10→32) – ReLU                                               352
      gate       门：Linear(32→128) – sigmoid，得到每条边、每个通道的门 g_uv ∈ (0,1)^128        4,224
      rounds     3 轮传播，每轮 4 个消息映射 W_t + 1 个自映射 W_s（128×128，无偏置）          245,760
      head_j1/2  15 个关节各自的读出头：Linear(128→64) – ReLU – Linear(64→3)（末层零初始化）  126,765
      head_h1/2  手节点读出头：Linear(128→128) – ReLU – Linear(128→6)（末层零初始化）          17,286
      合计                                                                                   761,427
    末层零初始化 → 训练开始时 Δ = 0（初始就是"保持前态"）。
    """

    def __init__(self, lbs_weights, J_regressor, hands_mean):
        super().__init__()
        self.register_buffer("lbs_w", lbs_weights.float().clone(), persistent=False)      # (778, 16)
        self.register_buffer("jreg", J_regressor.float().clone(), persistent=False)       # (16, 778)
        self.register_buffer("hands_mean", hands_mean.float().clone(), persistent=False)  # (45,)
        E = kinematic_edges()
        self.register_buffer("e_src", torch.tensor([e[0] for e in E]), persistent=False)
        self.register_buffer("e_dst", torch.tensor([e[1] for e in E]), persistent=False)
        self.register_buffer("e_type", torch.tensor([e[2] for e in E]), persistent=False)
        indeg = torch.zeros(len(EDGE_TYPES), N_NODES)
        for _, t, k in E:
            indeg[k, t] += 1
        # 每种边型：它的边下标，以及"按该边型的入边取平均"的散射矩阵 (E_t, 17)
        for k in range(len(EDGE_TYPES)):
            sel = torch.tensor([i for i, e in enumerate(E) if e[2] == k])
            S = torch.zeros(len(sel), N_NODES)
            S[torch.arange(len(sel)), self.e_dst[sel]] = 1.0 / indeg[k, self.e_dst[sel]]
            self.register_buffer(f"e_sel{k}", sel, persistent=False)
            self.register_buffer(f"e_scatter{k}", S, persistent=False)
        self.phi = nn.Sequential(nn.Linear(PHI_IN, REL_HIDDEN), nn.ReLU(inplace=True), nn.Linear(REL_HIDDEN, C))
        self.enc_part = NodeLinear(N_PARTS, C + PART_STATS + 3, C)
        self.enc_hub = nn.Linear(2 * C + HUB_STATS + 2 + 3, C)
        self.edge_emb = nn.Linear(EDGE_IN, EDGE_DIM)
        self.gate = nn.Linear(EDGE_DIM, C)
        self.rounds = nn.ModuleList([PropagationRound(C) for _ in range(N_ROUNDS)])
        self.head_j1 = NodeLinear(N_PARTS - 1, C, JOINT_HEAD)
        self.head_j2 = NodeLinear(N_PARTS - 1, JOINT_HEAD, 3, zero=True)
        self.head_h1 = nn.Linear(C, HUB_HEAD)
        self.head_h2 = nn.Linear(HUB_HEAD, 6)
        nn.init.zeros_(self.head_h2.weight)
        nn.init.zeros_(self.head_h2.bias)

    # ---------------------------------------------------------------- 第 5–6 步：节点输入
    def node_inputs(self, tok, d, mask, verts, assoc, prev48):
        r"""事件 → 17 个节点的输入。

        事件质量的分配（每个事件单位质量，不丢信息）：
          w_ir = q_i A_ir                 路由事件分给 3 个表面顶点（重心坐标）
          r_i  = 1 − q_i                  不在手上的事件整个进残差槽
        φ 的输入（每个事件 12 维）：
          在手上：[token 7, res_i / L_RES (3), (z_i − 0.45)/0.1, 1]
          残差槽：[token 7, 0, 0, 0, 0, 0]
        部位节点 p（16 个，各 156 维）：
          Φ_p = Σ_i ω_ip φ_i / Σ_i ω_ip                       学到的事件特征，按部位软权重加权平均   (128)
          s_p = 几何矩（原点 = 前态关节 J_p，坐标系 = F_p）+ 相机视线方向在 F_p 中的表达            (22 + 3)
          prev48 中关节 p 的 3 维                                                                 (3)
        手节点（hub，283 维）：
          Φ_h = Σ_i m_i φ_i / Σ_i m_i                         全手平均                             (128)
          s_h = 全手几何矩（原点 = 腕 J_0，相机系）                                                (22)
          Φ_r = Σ_i r_i φ_i(残差槽) / Σ_i r_i                 残差槽（ESMNet 判为不在手上的事件）   (128)
          残差槽占比 Σr / 事件数、是否有残差事件                                                   (2)
          prev48 中根旋转的 3 维                                                                   (3)
        """
        B, N, _ = tok.shape
        w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.float().unsqueeze(-1)       # (B, N, 4)
        r = (1.0 - assoc["q"]) * mask.float()                                         # (B, N)
        m, P, res = event_geometry(w, d, assoc["z"], assoc["vid"], verts)
        omega = part_weights(w, assoc["vid"], self.lbs_w)                              # (B, N, 16)
        J = torch.einsum("jv,bvk->bjk", self.jreg, verts)                              # (B, 16, 3)
        F = node_frames(prev48, self.hands_mean)                                       # (B, 16, 3, 3)
        Mp, sp = node_moments(omega, P, res, J, F)                                     # (B, 16), (B, 16, 22)
        view = F[..., 2, :]                                                            # 相机 z 轴在节点系中
        sp = torch.cat([sp, view], -1)                                                 # (B, 16, 25)
        eye = torch.eye(3).expand(B, 1, 3, 3)
        Mh, sh = node_moments(m.unsqueeze(-1), P, res, J[:, :1], eye)                  # (B, 1), (B, 1, 22)
        Mh, sh = Mh[:, 0], sh[:, 0]
        on = (m > 0).unsqueeze(-1).float()
        zn = ((assoc["z"] - Z_REF) / Z_SCALE).unsqueeze(-1)
        f_on = torch.cat([tok, res / L_RES, zn * on, on], -1)                         # (B, N, 12)
        f_off = torch.cat([tok, torch.zeros(B, N, 5)], -1)                             # (B, N, 12)
        phi_on, phi_off = self.phi(f_on), self.phi(f_off)                              # (B, N, 128)
        Phi_p = torch.einsum("bnk,bnc->bkc", omega, phi_on) / Mp.clamp_min(1e-12).unsqueeze(-1)
        Phi_h = torch.einsum("bn,bnc->bc", m, phi_on) / Mh.clamp_min(1e-12).unsqueeze(-1)
        rs = r.sum(1)
        Phi_r = torch.einsum("bn,bnc->bc", r, phi_off) / rs.clamp_min(1e-12).unsqueeze(-1)
        live = mask.float().sum(1).clamp_min(1.0)
        p3 = prev48.view(B, N_PARTS, 3)
        x_part = torch.cat([Phi_p, sp, p3], -1)                                        # (B, 16, 156)
        x_hub = torch.cat([Phi_h, sh, Phi_r, (rs / live).unsqueeze(-1), (rs > 0).float().unsqueeze(-1), p3[:, 0]], -1)
        return {"x_part": x_part, "x_hub": x_hub, "J": J, "F": F, "omega": omega, "res": res, "P": P}

    # ---------------------------------------------------------------- 第 7 步：节点编码
    def encode(self, x_part, x_hub):
        """h^0 = [16 个部位各自线性映射, 手节点线性映射] → (B, 17, 128)，没有激活。"""
        return torch.cat([self.enc_part(x_part), self.enc_hub(x_hub).unsqueeze(1)], 1)

    # ---------------------------------------------------------------- 第 8 步：状态相关边门控
    def gates(self, J):
        """g_uv = sigmoid(W_g · ReLU(W_e · edge_features(前态关节 J)))，(B, 62, 128)；每次前向算一次，3 轮共用。"""
        g = edge_features(J, self.e_src, self.e_dst, self.e_type)
        return torch.sigmoid(self.gate(torch.relu(self.edge_emb(g))))

    # ---------------------------------------------------------------- 第 9 步：运动学树传播 P
    def propagate(self, h, gate):
        r"""L = 3 轮，每轮对所有 17 个节点：
            h_v ← h_v + ReLU( W_s h_v + Σ_{t ∈ 4 种边型} mean_{u →_t v} g_uv ⊙ W_t h_u )
        "按边型分别取平均"：例如手掌节点从 5 个一级指节收 up 消息取平均；手节点从 16 个部位收 part2hub 消息取平均；
        每个部位只有一个父节点和一个手节点，down / hub2part 各一条。残差式更新（h + ReLU(…)）。"""
        for rd in self.rounds:
            upd = rd.slf(h)
            for k, lin in enumerate(rd.msg):
                sel = getattr(self, f"e_sel{k}")
                msg = gate[:, sel] * lin(h)[:, self.e_src[sel]]                         # (B, E_t, 128)
                upd = upd + torch.einsum("ev,bec->bvc", getattr(self, f"e_scatter{k}"), msg)
            h = h + torch.relu(upd)
        return h

    # ---------------------------------------------------------------- 第 10 步：局部读出
    def readout(self, h):
        r"""局部读出、无旁路：
            Δθ_j = head_j(h_j)，j = 1..15（每个手指关节只读自己的节点，各一套权重）       (15 × 3)
            [Δt, Δθ_root] = head_hub(h_hub)（平移与根旋转只读手节点）                       (6)
        手掌节点 h_0 不直接读出，它只通过消息影响别的节点。输出按状态布局排成 Δ = [Δt, Δθ_root, Δθ_1..15]。"""
        dj = self.head_j2(torch.relu(self.head_j1(h[:, 1:N_PARTS])))                  # (B, 15, 3)
        dh = self.head_h2(torch.relu(self.head_h1(h[:, HUB])))                        # (B, 6)
        return torch.cat([dh, dj.flatten(1)], -1)                                      # (B, 51)

    def forward(self, tok, d, mask, verts, assoc, prev48):
        n = self.node_inputs(tok, d, mask, verts, assoc, prev48)
        h0 = self.encode(n["x_part"], n["x_hub"])
        hL = self.propagate(h0, self.gates(n["J"]))
        return self.readout(hL), {"h0": h0, "hL": hL, **n}


# ==================================================================================================================
# 9. 第 11 步：一次完整的状态更新
# ==================================================================================================================
def evhandtrack_step(trunk, routes, tok, ev_xy, mask, prev, betas, K, mano, faces, n_events):
    r"""x_k = x_{k-1} + Δ。没有事件的更新间隔 Δ = 0（零事件门），状态原样保留。

    参数：routes = ESMNet 解码后的 (routed, face, bary)；tok (B, N, 7)；ev_xy (B, N, 2) 保留事件像素；
          mask (B, N) 有效事件；prev (B, 51)；n_events (B,) 本间隔事件总数。
    精度：几何与统计量在 autocast 之外算（训练 float32，评测 float64）；φ 和其余线性层在训练时走 bf16 autocast。
    """
    d = event_rays(ev_xy, K)                                                       # 射线 (B, N, 3)
    verts, J, F = prior_geometry(prev, betas, mano)                                # 前态网格
    routed, face, bary = routes
    assoc = surface_association(routed, face, bary, mask, verts, faces)            # 表面关联
    delta, aux = trunk(tok, d, mask, verts, assoc, prev[:, 3:51])                  # 主干 → Δ
    delta = torch.where((n_events <= 0).unsqueeze(-1), torch.zeros_like(delta), delta)
    return prev + delta, {"d": d, "verts": verts, "assoc": assoc, "delta": delta, **aux}


def run_stream(stream_events, x0, betas, K, trunk, esmnet, mano, faces):
    r"""部署（论文主行的流式形式）。事件逐个到达，每 50 ms 更新一次状态：

        front = StreamFrontEnd()
        x = x0
        for 每个事件 e：
            front.ingest(e)                                   # 到达时就定下它是否保留、它的 token
            if 到了更新时刻 t_q：
                ev, tok = front.update(t_q)                   # 本间隔保留的事件与 token
                routes = esmnet(本间隔的事件图, ev, tok)       # 冻结权重，每个更新间隔跑一次
                x = evhandtrack_step(trunk, routes, tok, ev[:, :2], mask, x, betas, K, mano, faces, n)
                输出 x                                         # MANO 姿态 + 平移 → 网格与关节
    """
    raise NotImplementedError("部署流程的伪代码，见文档字符串")


# ==================================================================================================================
# 10. 训练（两阶段；ESMNet 全程冻结，用同一份权重）
# ==================================================================================================================
def so3_trans_fk_loss(pred, gt, fk_joints):
    r"""损失 L = 1 · L_rot + 1 · L_trans + 2 · L_FK（三项都从同一个 MANO 解码读出），训练最小化 log10(L)。
      L_rot   = mean_j (1 − ⟨q_j^pred, q_j^gt⟩²)，16 个关节（含根）的四元数，对 q / −q 不变
      L_trans = SmoothL1(t^pred, t^gt)，米
      L_FK    = mean ‖(J^pred − J^pred_0) − (J^gt − J^gt_0)‖，21 个 OpenPose 关节、相对腕点，米
    fk_joints(x) → (B, 21, 3) 为 MANO 前向运动学（真实代码 model/model.py `_so3_fk_loss`）。"""
    raise NotImplementedError("见文档字符串")


TRAINING = {
    "阶段 1：从头训练（d1r）": [
        "初始化：随机初始化主干（读出末层为零 → 初始 Δ = 0）；ESMNet 载入 r_s2d 权重并冻结。",
        "样本：一个事件包，长度在 30–300 ms 间对数均匀随机；前态 = 包起点的真值 + 随机扰动，目标 = 包终点的真值。",
        "前态扰动（TRACK.PREV_NOISE_*，mixed 模式，三选一）：p=0.5 小高斯（平移 σ 5 mm、根旋转 σ 0.05 rad、手指每维 σ 0.05 rad）；"
        "p=0.3 大高斯（50 mm / 0.3 rad / 0.3 rad）；p=0.2 定向扰动（平移、根、手指三块各取随机方向，模长在 [0, 大扰动 σ] 均匀）。",
        "一步预测：x̂ = x_prior + Δ，对真值算 so3_trans_fk 损失。",
        "增强：域随机化（深度方向缩放 0.8–1.25、平移 ±14 px、绕光轴 ±15°、保留 25–100 % 事件）、极性翻转、逐像素极性互换、速度增强。",
        "优化：6000 步；2 卡 × 512（等效批量 1024）；Adam，学习率 4e-3，预热 500 步后余弦衰减；bf16。",
        "输入形式：分包（每包均匀步长取 ≤ 2048 个事件，token 在包内算）。",
    ],
    "阶段 2：闭环展开微调（rol5）": [
        "初始化：阶段 1 同种子第 6000 步的权重。",
        "样本：同一段运行里连续 K = 5 个 50 ms 事件包，共用一次增强；x_0 = 真值 + 扰动（只在第 0 包加）。",
        "展开：for k in 0..4: y_k = x_k + Δ(events_k, x_k)；L += so3_trans_fk(y_k, GT_k)；x_{k+1} = y_k（不截断梯度）。",
        "梯度穿过前态：前态 FK 网格、表面关联的深度、节点坐标系、边门控的关节几何都对 x_k 可微；"
        "ESMNet 的面与重心坐标不可微（也不依赖前态）。",
        "损失：log10(五步损失的平均)，五步权重相同。",
        "优化：1500 步；2 卡 × 512；Adam，学习率 1e-3，预热 100 步后余弦降到 0；bf16；逐包梯度检查点省显存。",
    ],
    "评测 / 部署": [
        "主行：流式形式（StreamFrontEnd），每 50 ms 更新一次，闭环递推；与训练的分包形式权重相同，零训练切换。",
        "三种子流式 MPJPE-L / MPJPE-G / MPVPE-L / MPVPE-G / RA：12.13 / 9.33 / 9.90 / 8.02 / 10.63 mm。",
    ],
}


# ==================================================================================================================
# 11. 画图清单（照着画即可）
# ==================================================================================================================
GROUPS = {
    "input": "输入（灰色）",
    "frozen": "冻结的预训练权重（蓝灰色 + 锁图标）：只有 ESMNet",
    "geometry": "无参数几何运算（白底、虚线边框）",
    "learned": "可学习模块（橙色）",
    "state": "状态与闭环（绿色）",
    "train": "只在训练时出现（红色虚线）",
}

FIGURE_BLOCKS = [
    {"id": "events", "group": "input", "label": "事件流 Event stream",
     "detail": "e_i = (x_i, y_i, t_i, p_i)，240 × 180"},
    {"id": "prior", "group": "state", "label": "上一状态 x_{k-1}",
     "detail": "51 维 = 平移 3 + 根旋转 3 + 15 个关节 × 3（轴角）；betas、K 为序列给定"},
    {"id": "frontend", "group": "geometry", "label": "流式取样 + 事件 token",
     "detail": "按最近 50 ms 事件率自适应取样（约 ≤ 2048 / 间隔）；token 7 维：位置、极性、时间、间隔、两种像素年龄"},
    {"id": "esmnet", "group": "frozen", "label": "ESMNet（预训练权重，冻结）",
     "detail": "输入：本间隔事件图 4×180×240 + 保留事件；输出：在手上概率、部位、模板表面点 → 面 f_i + 重心坐标 b_i"},
    {"id": "fk", "group": "geometry", "label": "前态 MANO FK",
     "detail": "网格 V 778×3、关节 J 16×3、部位坐标系 F 16×3×3"},
    {"id": "assoc", "group": "geometry", "label": "表面关联 Surface association",
     "detail": "P_i = 路由点在前态网格上的位置；res_i = z_i d_i − P_i；部位软权重 ω_ip（LBS）；不在手上的事件 → 残差槽"},
    {"id": "phi", "group": "learned", "label": "事件编码 φ",
     "detail": "逐事件共享 MLP 12 → 64 → 128；输入 [token 7, res 3, 深度 1, 在手上 1]"},
    {"id": "pool", "group": "geometry", "label": "部位汇聚 + 几何矩",
     "detail": "部位节点 16 × 156 = [Φ_p 128, 几何矩 22 + 视线 3, 前态关节 3]；手节点 283 = [Φ_h 128, 矩 22, 残差槽 Φ_r 128, 2, 根 3]"},
    {"id": "encode", "group": "learned", "label": "节点编码",
     "detail": "16 个部位各自线性 156 → 128；手节点线性 283 → 128；h^0 ∈ R^{17×128}"},
    {"id": "gate", "group": "learned", "label": "状态相关边门控",
     "detail": "每条边 10 维前态关节几何（向量、长度、透视缩短、深度、边型）→ 32 → 128，sigmoid；3 轮共用"},
    {"id": "prop", "group": "learned", "label": "运动学树传播 P（× 3 轮）",
     "detail": "17 个节点、62 条有向边、4 种边型；h_v ← h_v + ReLU(W_s h_v + Σ_t mean g_uv ⊙ W_t h_u)"},
    {"id": "readout", "group": "learned", "label": "局部读出",
     "detail": "15 个关节头 128 → 64 → 3 得 Δθ_j；手节点头 128 → 128 → 6 得 Δt、Δθ_root；Δ ∈ R^51"},
    {"id": "update", "group": "state", "label": "加性状态更新",
     "detail": "x_k = x_{k-1} + Δ（无事件时 Δ = 0）"},
    {"id": "output", "group": "state", "label": "输出：MANO 网格 / 关节", "detail": "MANO(betas, x_k)"},
    {"id": "loss", "group": "train", "label": "损失 L_rot + L_trans + 2 L_FK", "detail": "log10；rol5 时为 5 步平均"},
]

FIGURE_ARROWS = [
    ("events", "frontend", "事件逐个到达"),
    ("frontend", "esmnet", "本间隔事件 + 保留事件位置"),
    ("frontend", "assoc", "保留事件的射线 d_i（N × 3）"),
    ("frontend", "phi", "token（N × 7）"),
    ("esmnet", "assoc", "面 f_i、重心坐标 b_i、是否在手上"),
    ("prior", "fk", "x_{k-1}"),
    ("fk", "assoc", "前态网格 V"),
    ("assoc", "phi", "残差 res_i、深度 z_i、在手上标记"),
    ("assoc", "pool", "ω_ip、P_i、res_i"),
    ("fk", "pool", "关节 J、坐标系 F（节点原点与坐标系）"),
    ("prior", "pool", "prev48 切片（每个节点 3 维）"),
    ("phi", "pool", "φ_i（N × 128）"),
    ("pool", "encode", "x_part 16 × 156、x_hub 283"),
    ("encode", "prop", "h^0（17 × 128）"),
    ("fk", "gate", "前态关节几何（62 条边 × 10）"),
    ("gate", "prop", "g_uv（62 × 128）"),
    ("prop", "readout", "h^3（17 × 128）"),
    ("readout", "update", "Δ（51）"),
    ("prior", "update", "x_{k-1}"),
    ("update", "output", "x_k"),
    ("update", "prior", "闭环：x_k 作为下一次更新的前态（虚线回环）"),
    ("update", "loss", "训练时：与真值比较"),
]

PANELS = {
    "(a) 总体框架": "按 FIGURE_BLOCKS / FIGURE_ARROWS 画一次状态更新，右侧画闭环回环 x_k → x_{k-1}。",
    "(b) 运动学树图": (
        "17 个节点：手掌 palm 在中心，5 根手指各 3 个节点向外排（index1-3、middle1-3、ring1-3、pinky1-3、thumb1-3），"
        "手节点 hub 单独放在旁边并连到全部 16 个部位。边分四种颜色：down（父→子）15 条、up（子→父）15 条、"
        "hub2part 16 条、part2hub 16 条，共 62 条；每条边上标一个门 g_uv（由前态关节几何算出）。"
        "读出：15 个手指节点各自出 Δθ_j，hub 出 Δt、Δθ_root；palm 节点不直接读出。"),
    "(c) 训练": (
        "两阶段：① 单步从头训练 6000 步（前态 = 真值 + 扰动）；② rol5：把 (a) 展开 5 步，"
        "x_0 → y_0 → y_1 → … → y_4，每步一个 50 ms 事件包、各算损失，梯度沿前态链（红色虚线）回传。ESMNet 两阶段都冻结。"),
}


# ==================================================================================================================
# 12. 自检（随机数据、CPU）
# ==================================================================================================================
def _random_mano(gen):
    faces = torch.randint(0, N_VERTS, (N_FACES, 3), generator=gen)
    lbs = torch.softmax(torch.randn(N_VERTS, N_PARTS, generator=gen) * 3, -1)
    jreg = torch.softmax(torch.randn(N_PARTS, N_VERTS, generator=gen) * 3, -1)
    base = torch.randn(N_VERTS, 3, generator=gen) * 0.04

    def verts_fn(prev, betas):
        return base.unsqueeze(0) + prev[:, None, 0:3] + torch.tensor([0.0, 0.0, 0.45])

    return {"faces": faces, "lbs": lbs, "J_regressor": jreg, "hands_mean": torch.zeros(45), "verts_fn": verts_fn}


def _self_check():
    gen = torch.Generator().manual_seed(0)
    mano = _random_mano(gen)
    trunk = EvHandTrackTrunk(mano["lbs"], mano["J_regressor"], mano["hands_mean"])
    print("主干参数量（逐模块）：")
    total = 0
    for name, mod in trunk.named_children():
        c = sum(p.numel() for p in mod.parameters())
        total += c
        print(f"  {name:10s} {c:9,d}")
    print(f"  合计       {total:9,d}   （+ ESMNet {ESMNetFrozen.N_PARAMS:,d} = {total + ESMNetFrozen.N_PARAMS:,d}）")
    assert total == 761_427, total
    assert len(kinematic_edges()) == 62

    # 流式前端：随机事件
    front = StreamFrontEnd()
    t = 0
    for _ in range(400):
        t += int(torch.randint(10, 200, (1,), generator=gen))
        front.ingest(int(torch.randint(0, IMG_W, (1,), generator=gen)), int(torch.randint(0, IMG_H, (1,), generator=gen)),
                     t, int(torch.randint(0, 2, (1,), generator=gen)))
    ev, tok = front.update(t)
    B, N = 2, ev.shape[0]
    tok = tok.unsqueeze(0).expand(B, -1, -1).contiguous()
    ev_xy = ev[:, :2].float().unsqueeze(0).expand(B, -1, -1)
    mask = torch.ones(B, N, dtype=torch.bool)
    prev = torch.randn(B, 51, generator=gen) * 0.1
    K = torch.tensor([[603.45, 0.0, 325.09], [0.0, 602.96, 242.10], [0.0, 0.0, 1.0]]).expand(B, 3, 3)
    routes = esmnet_placeholder_routes(B, N, mano["faces"], gen)
    with torch.no_grad():
        for p in (trunk.head_j2.weight, trunk.head_h2.weight):   # 让输出非零，便于看形状
            p.normal_(0, 1e-3, generator=gen)
        x_new, aux = evhandtrack_step(trunk, routes, tok, ev_xy, mask, prev, None, K, mano, mano["faces"],
                                      torch.full((B,), N))
    print("\n一次状态更新的张量形状：")
    rows = [("保留事件 token", tok), ("射线 d", aux["d"]), ("前态网格 V", aux["verts"]),
            ("关联 vid", aux["assoc"]["vid"]), ("关联 A", aux["assoc"]["A"]), ("深度 z", aux["assoc"]["z"]),
            ("残差 res", aux["res"]), ("部位软权重 ω", aux["omega"]), ("前态关节 J", aux["J"]), ("坐标系 F", aux["F"]),
            ("部位节点输入", aux["x_part"]), ("手节点输入", aux["x_hub"]), ("h^0", aux["h0"]), ("h^3", aux["hL"]),
            ("Δ", aux["delta"]), ("新状态 x_k", x_new)]
    for name, t_ in rows:
        print(f"  {name:12s} {tuple(t_.shape)}")
    assert aux["x_part"].shape[-1] == C + PART_STATS + 3 and aux["x_hub"].shape[-1] == 2 * C + HUB_STATS + 5
    assert x_new.shape == (B, 51)
    print("\n自检通过。")


if __name__ == "__main__":
    _self_check()

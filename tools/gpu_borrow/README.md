# GPU 空闲借用包装器

`gpu-borrow` 包装一条训练命令：所选 GPU 连续空闲 600 秒后开跑，优先用户 `lmx` 使用其中任一卡时让出整个借用作业；之后重新等待连续 600 秒，再用恢复命令启动。它独立于本仓库的实验驱动，不需要修改老师的启动命令，也不需要他的账户密码。

每次启动都在系统级独立训练 cgroup 中执行；抢占时清理这个 cgroup，包括 `torchrun` 中另建 session 的 worker。训练仍使用自己的普通用户身份，不向 `lmx` 的进程发信号，也不接管已经运行的训练。`run` 必须由自己的普通用户执行，拒绝 root。

系统隔离后端须先由管理员安装一次。后端缺失时，`run` 拒绝启动，不回退到桌面所在的用户 scope。旧版 `systemd-run --user --scope` 会把训练放进 `user@1004.service`，与 Firefox、桌面和 ToDesk 共享 systemd-oomd 的压力处理范围；2026-11-09 的借卡冒烟已触发该风险。系统 slice 用于隔离桌面，工具不额外设置宿主内存准入、内存上限、交换限制或全局并发限制。

监控后端默认自动选择：优先使用 `pynvml`，未安装时回退到标准库解析 `nvidia-smi` XML；`probe` 会显示所用后端。可把 `nvidia-ml-py` 装进工具自己的独立环境，不必改训练环境，也不依赖 `psutil`。

## 管理员安装系统隔离后端

先查看安装计划：

```bash
/usr/bin/python3 -I /data1/lyq/code/mesh/EventHands1/tools/gpu_borrow/install_system.py \
  --user lyq --plan
```

管理员在本机执行一次安装，并在提示时输入 sudo 密码：

```bash
sudo /usr/bin/python3 -I /data1/lyq/code/mesh/EventHands1/tools/gpu_borrow/install_system.py \
  --user lyq
```

本机已经安装过旧版系统后端。撤回旧版工具添加的宿主内存策略，需要更新 root 持有的文件及已加载的 slice 属性：

```bash
sudo /usr/bin/python3 -I /data1/lyq/code/mesh/EventHands1/tools/gpu_borrow/install_system.py \
  --user lyq --update
```

更新源码不会自动修改服务器上已安装的策略；须完成这一步并验证，才能确认旧限制已撤回。

安装器先校验 slice 与 sudoers，再安装 root 持有的 helper、配置、`gpuborrow.slice` 和限定 helper 的免密 sudo 规则，并执行 `daemon-reload`；不会启动训练或重启用户服务。系统 helper 位于 `/usr/local/libexec/gpu-borrow-system`，配置为 `/etc/gpu-borrow-system.json`。包装器通过该 helper 管理自己的训练服务，训练命令仍以 `lyq` 身份执行。后续每个任务继续用普通用户运行 `gpu-borrow run`，不对整条训练命令使用 sudo。

安装后，用普通用户执行轻量验证：

```bash
gpu-borrow check-system
```

它通过系统 helper 运行一个极小的 CPU Python 命令，设置 `CUDA_VISIBLE_DEVICES=""`，不申请 GPU，也不启动训练；核对运行 UID、实际 cgroup 位于 `/gpuborrow.slice/`，以及工具原先添加的服务和 slice 内存 High、Max、交换上限均已恢复为不设限（systemd 为 `infinity`，cgroup 文件为 `max`）。成功时输出 `"verified": true`。此检查用于验证安装，不会启用 PMR 驱动的借卡对。

`run` 默认使用 `--isolation system`。兼容的 `user` 后端仅用于受控轻量测试，不用于本仓库的研究训练；它不提供训练与桌面的系统级隔离。

## 直接使用

从训练项目自己的目录运行，`--` 后面保留原来的训练命令：

```bash
python3 /data1/lyq/code/mesh/EventHands1/tools/gpu_borrow/cli.py \
  run --gpus 0 -- python train.py --config config.yaml
```

`train.py`、配置文件及其参数是示例，应替换为实际命令。相对路径按启动包装器时的工作目录解释。

本机短命令安装在 `~/.local/bin/gpu-borrow`，使用工具自己的监控环境。确认 `~/.local/bin` 在 `PATH` 中后即可使用；也可以直接写 `~/.local/bin/gpu-borrow`。后文均使用这个短命令，未安装时换成上面的 `python3 /绝对路径/cli.py` 即可。

## 单卡、多卡与卡号

显式选择物理 GPU 索引，或填写完整 GPU UUID：

```bash
gpu-borrow run --gpus 0 -- python train.py --config config.yaml

gpu-borrow run --gpus 0,1 -- \
  python -m torch.distributed.run --nproc_per_node=2 train.py --config config.yaml
```

未传 `--gpus` 时，工具读取启动时的 `CUDA_VISIBLE_DEVICES`：

```bash
CUDA_VISIBLE_DEVICES=0,1 gpu-borrow run -- python train.py --config config.yaml
```

所选索引会先解析成 GPU UUID，并按 UUID 锁定。子进程只看到借用的卡，CUDA 内部从逻辑卡 `0` 开始编号。例如选择物理卡 `2,3` 后，双卡训练使用逻辑卡 `0,1`。不要同时给子进程再设置另一组卡号。

多卡命令是一个整体。若借用 `0,1`，老师只使用卡 `0`，整个双卡作业也必须退出，不能只停掉其中一个 DDP rank。老师仅使用所选卡之外的 GPU 时，不会因此抢占这条借用命令。

借用卡可与直连卡组成同一个 DDP 作业（用户 2026-11-10 授权）。例如 GPU 2 和 GPU 5 空闲时：

```bash
gpu-borrow run --gpus 2,5 -- \
  python -m torch.distributed.run --nproc_per_node=2 train.py --config config.yaml
```

只要作业含借用卡，就把完整组合交给包装器；两张卡都连续空闲 600 秒后启动，任一卡需要让出时所有 rank 一起退出。GPU 4–7 上独立运行的其他作业各自启动。PMR 的新驱动 `tools/pmr/chain6.py` 在其候选池内支持全部二卡组合；`borrow_job2.py` 在原硬上限内提升训练进程的文件句柄软限，解决旧系统服务默认 1024 导致的 rol5 限制。

## 抢占、空闲计时与退出

默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--priority-user` | `lmx` | 按该用户实际出现的 GPU 进程判断优先使用 |
| `--idle-seconds` | `600` | 每次启动前需要的连续空闲时间 |
| `--poll-seconds` | `0.5` | 两次采样的目标间隔，单位秒 |
| `--grace-seconds` | `0` | 回收自己作业时的宽限时间，单位秒 |
| `--idle-memory-mib` | `256` | 启动前每张卡允许的最大已用显存，单位 MiB |
| `--idle-utilization` | `5` | 启动前每张卡允许的最大 GPU 利用率，单位 % |
| `--backend` | `auto` | 监控后端，可选 `auto`、`nvml`、`smi` |
| `--isolation` | `system` | 系统训练服务；`user` 仅用于受控轻量兼容测试 |

首次启动也要等 600 秒；被抢占后重新计时。`lmx` 在所选卡上的任意 GPU 上下文都会触发让卡；本作业之外的计算上下文（`C` 或 `C+G`，包括自己的另一条训练）也触发让卡。非优先用户的纯图形上下文（`G`，如小型 Xorg 基线）可保留；启动仍须满足下面的显存和利用率阈值。

启动前，每张所选卡还须同时满足已用显存 ≤ 256 MiB、GPU 利用率 ≤ 5%，并连续保持 600 秒。不满足条件的卡重置自己的计时；多卡作业须等全部所需卡都满足条件。GPU 利用率暂时为零，不代表持有上下文的训练已结束。自己的训练尚在运行时，也不会被当成空闲而另起一份。

NVML 优先读取内存信息 v2 的 `used`，不把驱动保留内存算作训练占用。旧驱动仅支持 v1 时保守使用原值，可能因基线高于阈值而一直等待；此时可用 `probe --backend smi` 对照，确认原因后选择后端或调整阈值。

默认 `--grace-seconds 0` 优先快速让卡，直接终止本次作业 cgroup 中的全部进程。这不会给训练临时保存检查点的机会；应让训练脚本定期保存。

设为正数时，先向自己的作业 cgroup 发送 `SIGTERM`，等待至作业结束或宽限期结束，再用 `SIGKILL` 清理残留进程。宽限期越长，让卡越慢；训练脚本是否响应 `SIGTERM` 保存状态，需要自行确认。`SIGSTOP` 会保留显存，不用于让卡。

发现外部使用，或监测失败、无法确认 GPU 状态时，工具让出自己的作业并重新等待。命令正常完成后包装器退出；训练自身报错或非零退出也结束，不无限重试失败任务。

这是轮询回收，不是老师启动前的显存预约。`0.5` 秒是采样间隔目标，还包含查询、进程退出和驱动清理时间，因此不是让卡完成时限，不能保证老师第一次申请显存就一定成功。

## 系统隔离与资源

借卡工具遵循按卡让出、连续空闲 600 秒后启动的 GPU 规则。系统 slice 只把训练与桌面所在的用户 cgroup 分开；工具不以宿主可用内存或 PSI 作为启动门槛，也不设置额外内存、交换或全局并发限制。不同 GPU 上的借卡作业可分别运行；同一张卡仍由按 UUID 的锁防止本工具重复启动。

各研究驱动原有的内存预算和调度规则由研究驱动负责，本工具不修改它们。系统本身的内存管理和 OOM 行为仍然适用。

## 断点恢复

工具不选择检查点，不改训练参数，也不保存模型状态。恢复能力由被包装的脚本提供。

如果同一条命令启动时就会自动找到最新检查点，直接使用该命令即可；抢占后工具仍用同一条命令启动。

如果首次启动与续训使用不同参数，可显式提供恢复命令：

```bash
gpu-borrow run --gpus 0,1 \
  --resume-command 'python train.py --config config.yaml --resume /path/to/last.ckpt' \
  -- python train.py --config config.yaml
```

`--resume` 只是示例训练脚本的参数，不是包装器的参数。需要保存哪些内容——模型、优化器、学习率、随机数和采样进度——仍由训练脚本决定；本工具不保证任意脚本都能逐位续训。

`--resume-command` 按 `shlex.split` 拆成参数直接执行，不经过 shell，不能在里面写管道、重定向、`&&`、通配符或变量展开。需要复杂启动步骤时，把它们写入自己的脚本，再把脚本作为命令。

## 只读查看

查看指定 GPU 的当前状态，不启动任务：

```bash
gpu-borrow probe --gpus 0,1,2,3
```

查看本工具记录的运行状态：

```bash
gpu-borrow status
```

默认状态目录为 `~/.local/state/gpu-borrow`；可用 `run --state-dir /path/to/private-dir` 和 `status --state-dir /path/to/private-dir` 指定，目录必须属于自己且权限为 `0700`。运行目录内有 `state.json`、`events.jsonl` 和 `command.log`，状态记录包含当前 attempt 的作业 unit（`scope` 字段）。运行状态属于包装器自己的记录；`probe` 用于查看当前 GPU 实际状态。

## 退出终端后继续运行

本机 `lyq` 的用户 systemd 已运行，且已启用 linger；可以用用户级临时服务保持包装器运行，无需 sudo：

```bash
systemd-run --user --unit=gpu-borrow-train --collect \
  --working-directory="$PWD" \
  -p KillMode=control-group \
  "$HOME/.local/bin/gpu-borrow" \
  run --gpus 0,1 -- /absolute/path/to/python train.py --config config.yaml
```

把 `/absolute/path/to/python` 换成训练环境的解释器。不同并行任务使用不同的 `--unit` 名称。该服务只包住这条命令，不接入已有的 TPD、PMR 或其他实验队列，也不修改它们的 STOP 文件。

查看日志或主动结束这条包装任务：

```bash
journalctl --user -u gpu-borrow-train.service -f
systemctl --user stop gpu-borrow-train.service
```

正常停止服务时，包装器收到 `SIGTERM` 后主动清理系统训练服务，再退出；不会再自动恢复。不要给服务设置 `Restart=always`，以免正常结束或真实训练失败后被外层重新启动。

外层包装器由用户 systemd 管理，内层训练由系统 systemd 管理，属于不同管理器；外层的 `KillMode=control-group` 不能单独覆盖内层训练。直接在终端运行时，用 Ctrl-C 或 `SIGTERM` 正常退出包装器。查看 `gpu-borrow status` 时，系统训练 unit 的名称为 `gpu-borrow-<uid>-<jobid>.service`，不要对它使用 `systemctl --user`。

系统 helper 在运行期间约每 0.5 秒核对包装器的 PID、进程启动时间和 UID；包装器死亡或身份不再匹配时，它结束这次作业自己的系统服务。该检查避免把重复使用的 PID 当作原包装器，也覆盖外层服务崩溃的通常情况；它依赖 helper 和系统服务管理器正常工作，不是即时退出保证。

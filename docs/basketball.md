# G1 篮球 Student 部署

入口：`scripts/run_basketball.py`。默认运行 MuJoCo，只有显式传入 `--real` 才会创建
`UnitreeCppEnv`。不使用 Python 原生 Unitree SDK。

## Sim2sim

在仓库根目录运行，使用 `-m` 确保加载当前 fork，而非 Conda 中其他目录的 editable 安装：

```bash
conda activate robojudo
python -m scripts.run_basketball --pre_hold 0 --hoop_pos 3.0 0.0 1.8
```

机器人从 JSON 的初始 root 位姿、关节姿态生成；篮球默认加载于
`initial_ball_position_m`，使用自由刚体与碗形手进行接触仿真，没有焊接或轨迹辅助。
仿真默认立即就绪，开始运行策略并推进 phase 投篮，无需按键。

Prepare 会在启动后自动完成，不需要按键启动。`pre_hold` 从 Prepare 完成时开始计时：

| 参数 | Prepare 完成后 | Space / B 的作用 |
| --- | --- | --- |
| `--pre_hold 0`（默认） | 立即运行策略并推进 phase，没有保持阶段 | 已自动开始，无须按键 |
| `--pre_hold -1` | 策略持续推理，phase=0，无限等待 | 立即推进 phase，开始投篮 |
| `--pre_hold T`，T > 0 | 策略持续推理，phase=0，T 秒后自动投篮 | 提前开始投篮，不再等待 T 秒 |
| 不传 `--pre_hold` | 等同于 `--pre_hold 0` | 已自动开始，无须按键 |

`--pre_hold 0` 的首个策略输入仍为 phase=0；该步下发完成后 frame 立即推进，
下一步为 phase=1/165，不会重复保持 phase=0。无限等待使用 `--pre_hold -1`。
这是对早期版本参数语义的调整；原先使用 `--pre_hold 0` 等待按键的命令需要改为 `-1`。
触发起跳不会再进入 Prepare，也不会清空历史。

**当前 student 没有训练 pre-hold，应使用 `--pre_hold 0`。** `-1` 和正数模式仅供
训练过保持阶段的模型使用；锁定 phase=0 并不会让投篮策略变成稳定站立策略。

```bash
# phase=0 维持 2 秒后自动开始，也可提前按 Space/B
python -m scripts.run_basketball --pre_hold 2

# 不使用 pre-hold：Prepare 完成后立即推理、投篮
python -m scripts.run_basketball

# 无限维持 phase=0，等待 Space/B
python -m scripts.run_basketball --pre_hold -1

# 不加载篮球 / 使用本地手柄
python -m scripts.run_basketball --pre_hold 0 --no_ball
python -m scripts.run_basketball --pre_hold -1 --joystick

# 无显示服务器的离线冒烟测试；无 pre-hold，直接投篮
python -m scripts.run_basketball --headless --pre_hold 0 --steps 200
```

仿真默认 Prepare 时间为 0，因为模型已经处于导出的初始状态；真机默认 3 秒。
`--prepare_seconds 3` 可以在仿真中检查原生插值。当前模型在初始姿态仅靠 PD 等待
3 秒会失稳，因此仿真默认不额外插值等待。所有模式在 Prepare 期间均不运行策略。

| 操作 | 键盘 | 手柄 |
| --- | --- | --- |
| 开始投篮 / 提前结束 pre-hold | Space | B |
| 重置并重新 Prepare | R | Y |
| Emergency Stop，进入 Damping | Esc | A |

急停锁存后不能通过重置恢复驱动，需要退出并重新启动程序。仿真急停后保留窗口并继续
阻尼仿真，关闭窗口或 Ctrl+C 退出。Prepare 中同样检查急停，且急停优先于投篮/重置。
重置在仿真中恢复机器人和篮球初始状态；真机只重新平滑进入初始关节姿态。

鼠标左键拖动旋转，右键拖动平移，滚轮缩放；Shift 改变拖动轴。
相机不强制跟随机体。Space 由 Controller 接管，不再触发 viewer 的暂停功能。
兼容层处理了 [MuJoCo 3.11 的相机 API 变更](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-11-0-july-27-2026)。

## 机载部署

以下是完成联调后的投篮命令，Prepare 后立即开始投篮，并非首次通信测试命令。
仅在 G1 机载工控机执行。远程开发服务器不执行该命令：

```bash
conda activate robojudo
python -m scripts.run_basketball --real --net_if eth0 --pre_hold 0 --hoop_pos 3.0 0.0 1.8
```

需要该 Conda 环境中安装 RoboJuDo 的 `unitree_cpp` 扩展（见
[Unitree 安装说明](unitree_setup.md)），以及项目现有依赖。键盘控制沿用原生 pynput，
需要可用的显示会话；无显示会话时加 `--no_keyboard`，仅使用 Unitree 手柄
（沿用 `UnitreeCtrl`）。`--net_if` 是机载机器人通信网卡。
Kp/Kd、nominal、安全位置范围及频率来自 JSON；C++ 后端 `control_dt` 同步设为 0.01 秒。
Prepare 使用原生关节插值且不运行 ONNX，完成后读取最新实测状态初始化历史。
退出/异常沿用 `UnitreeCppEnv.shutdown()`；急停键保持 Esc/A。

## 机载时序评估

Sim2sim 通过后可以进入有保护条件的机载地面联调，但不能据此确认起跳安全。
本仓库锁定的 `unitree_cpp` commit 为 `222028cdaa79cdd7c7cda6645ebd2b02d201be0c`：

- `DataBuffer` 覆盖最新值，不是不断积压的 FIFO；LowState 订阅初始化的队列长度参数为 1。
- `step()` 更新目标后直接调用 `LowCommandWriter()`，包含 1.0.3 的立即发送修复。
- 后台发送线程也会重复发送最后一个目标。在该封装中未见命令过期自动阻尼机制；
  Python 卡住不等于急停，日志也不是独立看门狗。

依据：[锁定版本的控制器源码](https://github.com/HansZ8/unitree_cpp/blob/222028cdaa79cdd7c7cda6645ebd2b02d201be0c/src/unitree_controller.cpp)、
[数据缓冲实现](https://github.com/HansZ8/unitree_cpp/blob/222028cdaa79cdd7c7cda6645ebd2b02d201be0c/src/unitree_controller.hpp)。
这说明封装设计消除了特定的软件排队等待，不代表已测得实际机载 DDS/执行器延迟。
机载 Conda 的实际安装版本和编译来源仍需对应确认，不能仅根据仓库 gitlink 判断。

当前模型没有训练 pre-hold，不能使用 `--pre_hold -1` 代替安全的地面联调。
先完成不连接机器人的离线计算评估；通信与急停验证应在有支撑条件下使用单独的
PD 测试流程，不运行该投篮策略。当前投篮入口没有单独的 PD 通信测试模式。
检查日志中的整步耗时是否持续有余量、周期是否有长尾、tick 是否长期重复、目标与实测关节
是否明显脱节，再决定是否进入投篮测试。`work_ms < 10` 只是必要的计算预算条件，
不是端到端执行延迟合格的证明。要覆盖 Python 卡死情形，需要在 C++/硬件侧验证
独立超时保护及其在跳跃中的处理策略。

## 纯文本日志

部署入口默认记录，每次运行独立创建 `logs/basketball/时间戳_PID/`，不生成二进制数据文件：

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 完整配置、CLI、模型 SHA-256、关节顺序、环境/依赖版本、C++ 模块路径 |
| `trace.jsonl` | 一行一条 JSON，包含逐步数据、计时和生命周期事件 |
| `summary.json` | 正常清理时生成，包含平均/最大耗时、超过 10 ms 的计数、tick 重复统计 |

```bash
# 仿真，默认就会保存日志
python -m scripts.run_basketball --pre_hold 0 --log_dir logs/basketball

# 离线计时对照时关闭详细日志（原有终端运行日志仍保留）
python -m scripts.run_basketball --headless --pre_hold 0 --steps 200 --no_log
```

`trace.jsonl` 的 `sample` 记录包括：

- Prepare 的逐步实测状态、插值比例、目标；Ready 时平铺的传感器/动作历史。
- 完整 469 维观测、29 维未裁剪网络输出 `raw_action`、裁剪后实际提交的 `target`。
- 实测关节角/速度、机体四元数（xyzw）、角速度；真机额外记录估计力矩、IMU 加速度和原始 tick。
- `mode`、推理使用的 `frame`/`phase`、下发后 `frame_after`、按键及控制器数据。
- 读取、控制器处理、观测构造、推理、`env.step()` 调用的分项耗时。
- `submit_call_ns` / `submit_return_ns` 标记 Python API 边界；`submitted` 表示 API 成功返回，
  **不表示已收到机器人执行确认**。MuJoCo 的 `submit_ms` 包括渲染和物理步进，不可当作 DDS 测量。
- `submit_interval_ms` 记录相邻成功提交的 API 调用开始时刻之差，用于检查实际命令提交节奏。

`cycle` 记录的 `work_ms` 包含大条数据日志的序列化/写入开销；`sample_log_ms` 单独计时。
`work_ms` 不含其自身那条小型 cycle 记录的写入及循环 sleep，
`start_interval_ms` 则包含上步全部写入、sleep 和调度抖动。
summary 的 `over_budget` 严格按大于 10 ms 计数，轻微 sleep 超时也会计入，需结合幅度判断。
所有纳秒时间均来自本机单调时钟；`metadata.json` 提供带时区的开始时间。

真机 `state_tick` 来自 C++ 返回的同一份 RobotState，`state_read_monotonic_ns` 是
Python 取到该副本的时间。`tick_delta` 与 `unchanged_tick_ms` 用于发现持续读取同一帧，
不能测量“消息在 DDS 中已经滞留了多久”；tick 单位、时钟同步和真实源时间不由本日志假设。
仿真没有伪造 tick，对应字段为空。
`policy_sample_monotonic_ns` 记录每次策略读取状态结束的本机时刻；仿真额外记录
`sim_time_s`，用于分别检查墙钟采样周期和物理仿真采样周期。

记录器不创建 Python 线程或异步队列，不缓存待发送动作。采用 1 MiB 文本缓冲、记录期间
每秒 flush，以及退出时关闭文件；没有每步 `fsync`。因此写入仍可能因磁盘阻塞影响周期，
需要在机载实际磁盘上比较开/关日志的耗时。强杀可能丢失最近缓冲，断电持久性不保证。
完整观测数据约 MB/s 量级，长时间 pre/post-hold 也持续记录，运行前留出相应磁盘空间。

## 观测与动作时序

固定 469 维 float32，按照导出 fields 的顺序构造，逐字段乘 JSON 中的 scale：

| 区间 | 内容 |
| --- | --- |
| `[0:1]` | phase |
| `[1:16]` | 5 帧重力投影 |
| `[16:31]` | 5 帧 pelvis 角速度 |
| `[31:176]` | 5 帧实测关节角减 nominal |
| `[176:321]` | 5 帧关节速度 |
| `[321:466]` | 5 帧已经提交的裁剪后 target 减 nominal |
| `[466:469]` | reset 时固化的机体坐标系篮筐向量 |

每步执行 `env.update → 最新观测 → ONNX → clip(nominal + output) → env.step`。
只有 `env.step` 成功返回后才滚动动作历史并推进 frame。没有动作平滑、额外观测归一化、
观测裁剪或软件延迟队列。控制循环的 sleep 仅用于维持 100 Hz，下一步重新获取最新状态。

Reset 将一次实测状态重复 5 次，将 `initial_pose - nominal` 重复 5 次。
pre-hold 期间正常推理并滚动历史，触发起跳只切换 phase 推进开关。
frame 每步增加 `reference_fps / policy_hz`，在 165 饱和；phase=1 持续推理维持，
不使用 JSON 的 `post_hold_s` 自动退出或回到站姿。

`--hoop_pos` 接收起跳前机体坐标系中的向量，默认 `(3, 0, 1.8)`，当前配置缩放后
为 `(0.75, 0, 0.45)`；它是模型输入，场景中没有添加实体篮筐。
MuJoCo 的关节位置、速度和驱动器映射均按机器人关节名称预先索引，篮球 freejoint 不参与。

每个 100 Hz 策略周期读取一次最新实测状态，传感器窗口是最近 5 次策略采样，
不是底层 DDS 连续到达的 5 个数据包。稳定运行时对应
`[s(t-40ms), s(t-30ms), s(t-20ms), s(t-10ms), s(t)]`，5 个点跨越 40 ms。
当前第 k 步推理的动作历史是 `[a(k-5), ..., a(k-1)]`；`a(k)` 提交后才进入下一步观测。
启动不足 5 步时，较早位置仍是 Reset 平铺的初始值。
这里没有按时间戳插值重采样；实际循环若超时，历史的真实采样间隔也会变长。
重复初始状态只消除窗口内的初始差分，不能把实测速度强行变成零。

## 离线 ONNX 耗时与历史检查

以下基准不创建 Environment、Controller 或 SDK 实例，也不连接机器人。
可在开发服务器和机载 Conda 环境中分别运行；决策应以机载 CPU 的结果为依据。

```bash
conda activate robojudo

# 不限速测计算开销：预热后统计 mean/p50/p95/p99/max、超过 10 ms 的次数
python -m scripts.benchmark_basketball --warmup 200 --steps 10000 --output logs/bench_compute.json

# 按 100 Hz 节拍测计算与文本日志开销，约 30 秒
python -m scripts.benchmark_basketball --warmup 200 --steps 3000 --paced --with_log --output logs/bench_100hz_log.json
```

两个 JSON 报告都包含模型 SHA-256、CPU provider、单线程 ORT 配置、会话加载与首次推理耗时。
`onnx_run` 只测 `session.run`；`policy_compute` 测观测拼装、推理、裁剪、虚拟动作历史更新，
使用 `--with_log` 时包括逐样本文本序列化/写入；不包括环境读取、Controller、关节适配或下发。
预热不计入统计。`compute_start_interval` 在 `--paced` 下表示相邻计算开始的实际墙钟间隔，
包括 sleep 与调度抖动；不限速运行时不能用它判断 100 Hz 节拍。
周期略高于 10 ms 也会计入 `over_budget`，需结合 P99/最大偏差判断幅度。

默认输入为初始姿态、零速度及变化的 phase；完整计算段使用虚拟动作历史，不是物理 rollout。
可以加 `--trace logs/basketball/<运行目录>/trace.jsonl`，让 **ONNX 单独计时段** 重放实测/仿真
观测；完整计算段仍使用合成状态。日志开关对照应在相同节拍下比较。
这些数字衡量计算耗时与主机周期，不能给出 DDS 排队、网络传输或电机执行延迟。

先运行带日志的 sim2sim，再验证终端打印的运行目录：

```bash
python -m scripts.run_basketball --headless --pre_hold 0 --steps 200
python -m scripts.verify_basketball_history logs/basketball/<运行目录> --output logs/history_check.json
```

验证器不调用 Policy 的观测函数：它从原始实测状态、已提交目标和 JSON 参数独立重建窗口，
逐字段对比全部 469 维输入，检查 Reset 平铺、oldest_to_newest、scale、篮筐常量及动作因果顺序。
`history_ok=true` 表示上述内容一致；不一致时返回非零退出码并列出行号/字段。
该结果验证日志到模型输入的一致性，不验证训练端的实现，也不证明底层状态没有延迟。

同时查看 `simulation_sample_interval`：当前 MuJoCo 配置应为 10 ms；
`policy_sample_interval` 是实际主机读取时刻间隔，可能因仿真计算/渲染而超过 10 ms。
机载日志没有仿真时间，查看实际采样周期和 `repeated_state_ticks`；后者非零表示相邻
策略周期读到了相同 tick，需要结合连续重复时长定位，不能仅据此推算端到端延迟。

## 离线验证

```bash
python -m unittest discover -s tests -p 'test_basketball*.py'

# 有显示会话时，验证 X11/GLFW 按键、双击选中/跟踪、拖动及缩放
ROBOJUDO_VIEWER_TESTS=1 python -m unittest discover -s tests -p test_mujoco_viewer_inputs.py
```

覆盖实测历史平铺、最新帧、所有 scale、动作裁剪与因果顺序、下发失败不推进、
pre-hold/触发/post-hold/重置、Prepare 禁止推理、急停优先级，以及带/不带篮球的
MuJoCo 和真实 ONNX rollout。硬件配置测试仅构造配置，不创建 SDK 实例或连接机器人。
这些检查验证部署接口与离线执行，不等同于实机投篮效果验证。

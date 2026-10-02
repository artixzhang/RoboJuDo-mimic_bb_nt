# G1 篮球 Student 部署

入口：`scripts/run_basketball.py`。默认运行 MuJoCo，只有显式传入 `--real` 才会创建
`UnitreeCppEnv`。真机路径不导入 MuJoCo / viewer，也不使用 Python 原生 Unitree SDK。

## Prepare、Ready 与 Start

默认流程（真机和交互仿真一致）：

1. 启动程序，在吊架支撑下 Prepare：从实测姿态平滑插值到 JSON 初始关节姿态，不运行策略。
2. Prepare 结束进入 Ready：持续发送初始姿态 PD 目标，不运行策略、不推进 phase、不计 pre-hold 时间。
   可以在这一阶段放下吊架、调整机体平衡、放球。PD 关节保持不等于主动平衡控制，
   不会自动将你调整后的关节角保存为新目标；初始深蹲姿态能否站稳需要支撑保护。
3. 按手柄 **Start** 或键盘 **Enter** 启动策略。Ready 中按 Space/B 也等同于 Start。
   此刻才开始执行下表的 pre-hold 语义。

| 参数 | 按 Start 后的行为 | 后续 Space/B |
| --- | --- | --- |
| `--pre_hold 0`（默认） | 首个输入 phase=0，此后每步推进，立即投篮 | 无需再次按键 |
| `--pre_hold -1` | 持续推理并下发策略动作，phase=0，无限等待 | 开始推进 phase |
| `--pre_hold T`，T > 0 | 持续推理并下发策略动作，phase=0，T 秒后推进 | 提前结束 pre-hold |

Start/Enter 只负责从 Ready 启动；策略已启动后再次按 Start/Enter 无效，结束 pre-hold 用 B/Space。
第一次按 B/Space 只启动策略，不会在同一步跳过 `-1` 或正数 pre-hold。
**当前 student 没有训练 pre-hold，实际投篮使用 `--pre_hold 0`。** `-1` 与正数仍运行策略，
不能用来代替放球阶段的 Ready。

Ready 以 100 Hz 正常滚动实测传感器历史，动作历史只记录实际提交的保持目标。
按 Start 或结束 pre-hold 都不清空历史；Prepare 结束时才平铺初始化。
Reset 会重新 Prepare，再回到 Ready 等待 Start。仿真显式 `--auto_start` 是唯一跳过 Ready 的模式；
真机拒绝该参数。相比旧版本，默认 `--pre_hold 0` 不再在 Prepare 后自动投篮。

| 操作 | 键盘 | 手柄 |
| --- | --- | --- |
| 从 Ready 启动策略 | Enter（或 Space） | Start（或 B） |
| 结束策略 pre-hold | Space | B |
| 重置并重新 Prepare、等待 Start | R | Y |
| Emergency Stop，进入 Damping | Esc | A |

急停优先于 Start/投篮/重置，锁存后不能通过 Reset 恢复驱动，需要退出并重新启动。
Prepare 中同样检查急停。真机急停后退出；仿真继续阻尼并保留窗口。

## 真机链路测试：直接运行

在机载 Conda 环境、仓库根目录执行。将 `eth0` 换成实际通信网卡。
**此命令会驱动真实电机进入初始姿态，整个测试保持吊架支撑；30 秒结束自动进入阻尼。**
无需篮球、MuJoCo 或显示会话。

```bash
conda activate robojudo_artix
python -m scripts.run_basketball --real --net_if eth0 --no_keyboard \
  --link_test --test_seconds 30 --pre_hold 0 --log_dir logs/basketball_link
```

程序启动后自动 Prepare（默认 3 秒），然后打印 `Ready: PD hold only, no inference`。
按手柄 Start 开始 30 秒计时。期间使用真实状态构造 469 维观测并运行 ONNX、真实 C++ 下发和日志；
**所有下发目标始终是初始姿态，网络输出只记录，任何 Start/B 都不能切换成投篮模式。**
推理 phase 正常推进到 1 并继续推理；这是计算/通信负载测试，不评估策略动作效果。
A 可随时提前阻尼退出，Ctrl+C/异常也执行 shutdown。

终端打印每次运行的日志目录，退出后自动生成：

- `metadata.json`：配置、模型 SHA-256、环境版本及 SDK 路径。
- `trace.jsonl`：Ready/推理、真实传感器、网络输出、实际 PD 目标及逐步计时。
- `summary.json`：运行结束原因、时序与 tick 统计。
- `link_report.json`：自动验证固定目标、全部观测历史、推理负载下的周期/提交耗时和状态重复。

报告 `status` 为 `software_checks_passed`、`issues_found` 或 `incomplete`。
`software_checks_passed` 只表示本次软件检查通过，不是投篮安全认证；按 A 或 Ctrl+C 提前退出为
`incomplete`，仍保留数据和报告。`findings` 会列出异常；工作耗时超过 10 ms、提交间隔或 tick
连续不变达到 20 ms 会报告问题。这些阈值用于定位明显异常，不是硬件延迟的验收标准。
`timing` 的统计只覆盖按 Start 后的推理阶段；Prepare/Ready 原始记录仍在 trace 中。

也可以对已有目录重新生成并打印报告：

```bash
python -m scripts.analyze_basketball_link "logs/basketball_link/实际运行目录"
```

测试覆盖真实状态读取、观测/推理、C++ 提交与持续状态反馈。`read_to_submit_return_ms` 是
Python 取得状态到提交调用返回的耗时，**不包含此前的传感器/DDS 接收延迟，也不是执行器确认**。
现有 SDK 接口没有提供可对齐的采集/接收时间戳和执行回执，报告明确列出这些未测项。
不需要为了这一步在工控机安装 MuJoCo。

## 真机投篮

完成链路检查后，退出测试程序，单独启动投篮模式：

```bash
python -m scripts.run_basketball --real --net_if eth0 --no_keyboard \
  --pre_hold 0 --hoop_pos 3.0 0.0 1.8
```

吊架支撑下 Prepare → Ready 中调整平衡、放球 → 人员离开运动范围后按 Start → 立即投篮。
Ready 维持 JSON 初始关节目标，不会主动控制浮动基座平衡；防坠支撑与人工放球要考虑此区别。
需要机载环境安装 RoboJuDo 的 `unitree_cpp` 扩展（见 [Unitree 安装说明](unitree_setup.md)）。
Kp/Kd、nominal、限位和频率来自 JSON；C++ `control_dt=0.01` 秒。
无显示会话时使用 `--no_keyboard` 和 Unitree 手柄；需要键盘时去掉此参数。

本仓库锁定的 C++ commit 是 `222028cdaa79cdd7c7cda6645ebd2b02d201be0c`：
`DataBuffer` 覆盖最新值，LowState 订阅队列长度参数为 1，`step()` 直接调用发送函数。
后台也会重复最后一个目标；该封装未见命令过期自动阻尼机制，Python 卡住不等于急停。
30 秒测试退出计时和 A 键处理依赖 Python 循环，并非独立看门狗。
依据：[控制器源码](https://github.com/HansZ8/unitree_cpp/blob/222028cdaa79cdd7c7cda6645ebd2b02d201be0c/src/unitree_controller.cpp)、
[数据缓冲实现](https://github.com/HansZ8/unitree_cpp/blob/222028cdaa79cdd7c7cda6645ebd2b02d201be0c/src/unitree_controller.hpp)。
机载实际安装版本仍需对应确认，不能仅根据仓库 gitlink 判断。

## Sim2sim

```bash
# 复现人工 Start 流程；Prepare 后等待 Enter/Space
python -m scripts.run_basketball --pre_hold 0

# 直接预览投篮：仿真专用，跳过 Ready 等待
python -m scripts.run_basketball --auto_start --pre_hold 0

# 无显示服务器的离线冒烟测试
python -m scripts.run_basketball --headless --auto_start --pre_hold 0 --steps 200
```

机器人和篮球使用导出初始位姿，篮球自由接触碗形手，无焊接/轨迹辅助。
`--no_ball` 不加载篮球；`--joystick` 使用本地手柄。仿真 Prepare 默认 0 秒，真机默认 3 秒；
可用 `--prepare_seconds` 覆盖。当前初始姿态在仿真中仅靠 PD 等待 3 秒会失稳，
因此无人支撑的投篮预览用 `--auto_start`。headless 不加它时仅测试 Ready，不会自动推理。
鼠标左键拖动旋转，右键平移，滚轮缩放；Shift 改变拖动轴，相机不强制跟随。
Space 由 Controller 接管，不会暂停 viewer；兼容层处理了
[MuJoCo 3.11 相机 API 变更](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-11-0-july-27-2026)。

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
python -m scripts.run_basketball --headless --auto_start --pre_hold 0 --steps 200 --no_log
```

`trace.jsonl` 的 `sample` 记录包括：

- Prepare 的逐步实测状态、插值比例、目标；Ready 时平铺的传感器/动作历史。
- 完整 469 维观测、网络输出 `raw_action`、策略目标 `policy_target`、实际提交的 `target`。
- `inference_ran`、`target_source`、`link_test` 区分 Ready、固定 PD 链路测试与策略控制；Ready 网络输出为空。
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

策略启动后每步执行 `env.update → 最新观测 → ONNX → clip(nominal + output) → env.step`。
Ready 跳过 ONNX，链路测试将策略目标替换为初始 PD 目标。
只有 `env.step` 成功返回后才滚动动作历史，且只有策略已启动时才推进 frame/pre-hold 计数。没有动作平滑、额外观测归一化、
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
python -m scripts.run_basketball --headless --auto_start --pre_hold 0 --steps 200
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

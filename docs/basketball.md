# G1 篮球 Student 部署

入口为 `scripts/run_basketball.py`，默认 MuJoCo；只有显式 `--real` 才创建
`UnitreeCppEnv`。真机路径不导入 MuJoCo/viewer，也不使用 Python Unitree SDK。
默认读取 `assets/models/g1/mimic_bb_nt/student_config.json` 和同目录 ONNX。

## 新配置与启动流程

部署已改为 `skill_command` one-hot 控制，删除旧 phase、frame、pre-hold/post-hold
及 `--pre_hold` 参数。不再依据动作长度、计时器或 `skills.definitions.end` 自动切换技能。
技能持续有效，直到下一次操作切换。

1. Prepare：从实测姿态插值到 JSON `initial_pose`，不运行策略。真机默认 3 秒，仿真默认 0 秒。
2. Ready：持续发送初始姿态 PD 目标，滚动实测历史，不运行策略。
3. Start / Enter：开始运行选中的技能，默认 `hold`。运行中再次 Start 不重启策略。
4. 按命令键切换技能；切换不清空传感器或动作历史。Ready 中也可预选技能，但仍需 Start。
5. `r` / 不带 R2 的 Y：重新 Prepare、清空历史并恢复默认技能，回到 Ready。
   最新篮筐坐标保留。仿真显式 `--auto_start` 会在每次 Prepare 后直接启动；真机拒绝该参数。

Prepare/Ready 的关节 PD 保持不等于主动平衡。真机使用吊架支撑完成初始化；
仿真无人支撑时可用 `--auto_start` 直接运行策略。

## 手柄与键盘

键盘和手柄可同时启用。真机默认 Unitree 手柄加图形键盘；仿真加 `--joystick`
启用本地手柄。`--no_keyboard` 关闭键盘，`--headless` 默认关闭 viewer 和键盘。
`--terminal_keyboard` 可在 SSH 或 headless 中启用终端键盘，无需 DISPLAY/pynput。
终端需要 TTY（SSH 可用 `ssh -t`），输入通过非阻塞轮询读取，退出时恢复终端设置。
图形键盘使用 pynput，需要可用显示会话；终端键盘只接收当前终端输入。

| 操作 / command ID | 键盘 | Unitree / 本地手柄 |
| --- | --- | --- |
| 启动策略 | Enter | Start |
| 0：hold | 1 | R2+A |
| 1：shoot | 2 | R2+B |
| 2：stand_to_hold | 3 | R2+X |
| 3 | 4 | R2+Y |
| 4 | 5 | R2+上 |
| 5 | 6 | R2+下 |
| 6 | — | R2+左 |
| 7 | — | R2+右 |
| 重置 | r | Y（不按 R2） |
| 急停并锁存阻尼 | Esc | Select / 本地 Back |

**当前 JSON 只有 ID 0–2**，其余按键保留映射，按下会提示无效 ID 并保持当前技能。
导出更多技能后，维度与标签从配置读取。键盘支持前 6 个，手柄支持前 8 个。
本地手柄的右扳机 RT 转换为 R2，采用按下 0.6、释放 0.4 的阈值。
先按住 R2，再按目标按钮；当前沿用项目的 SDL/Xbox 按键编号约定。

篮球入口的 A 不再急停，避免与 R2+A 冲突。Select/Back 和 Esc 优先于所有命令，
Prepare 中同样有效，锁存后不能用 Reset 恢复，需退出重启。
其他部署入口的手柄映射不变。一个控制周期收到多个不同技能命令时全部忽略并提示；
相同技能命令去重，不产生随机选择。

## 不停机修改篮筐

启动参数 `--hoop_pos R THETA Z` 和运行时输入都使用柱坐标 `[r, θ, z]`，
单位分别为米、弧度、米，直接传给导出的 `hoop_pos_cyl` 字段，**不是 XYZ**。
默认 `[3, 0, 1.8]`。程序不额外旋转坐标或将 θ 当作角度制。

按 `h` 后输入三个数并回车，例如：

```text
h3.5 0.2 1.8<Enter>
```

也接受 `h 3.5, 0.2, 1.8`。Backspace 编辑，Esc 始终急停。
输入期间的数字、r 和回车被文本编辑器接管，不触发技能、重置或启动。
输入不阻塞推理；回车后在控制周期边界整体更新三维坐标，并打印确认、记录日志。
负半径、非有限数或数量错误会被拒绝，旧目标保留。新目标跨技能切换和 Reset 保留。

下一次投篮的操作例子：先按 `1` 切换到 hold → 输入新的 `h…` 并回车 → 放球准备好后
按 `2` 切换 shoot。按照导出约定，应在两次投篮之间更新并在单次投篮中保持不变。
部署没有投篮结束检测，运行时输入会立即生效，不会自动等到某个动作阶段。
修改目标和技能均保留历史窗口，不需要重启进程。

## 启动示例

```bash
conda activate robojudo

# 仿真，图形键盘和本地手柄同时使用
python -m scripts.run_basketball --joystick

# 仿真，终端键盘；Prepare 后直接运行 hold，再按 2 投篮
python -m scripts.run_basketball --auto_start --terminal_keyboard

# headless 冒烟测试，仅运行默认 hold
python -m scripts.run_basketball --headless --auto_start --steps 200

# 真机，Unitree 手柄与终端键盘同时使用；替换通信网卡
python -m scripts.run_basketball --real --net_if eth0 --terminal_keyboard --hoop_pos 3 0 1.8

# 真机，仅手柄
python -m scripts.run_basketball --real --net_if eth0 --no_keyboard
```

真机需安装 `unitree_cpp` 扩展，见 [Unitree 安装说明](unitree_setup.md)。
Kp/Kd、nominal、限位和策略频率来自 JSON，当前频率 100 Hz。

新导出仅提供关节 `initial_pose`，没有根节点/篮球世界位姿。当前仿真场景默认根节点
位置 `[0, 0, 0.75]`、四元数 wxyz `[1, 0, 0, 0]`、篮球位置 `[0.3, 0, 0.85]`，
在 `make_config()` 中明确设置；这些是场景默认值，不是训练导出状态。
准确复现训练时需要对齐这些位姿。篮球通过自由接触运动，无焊接或轨迹辅助；
`--no_ball` 不加载篮球，目标篮筐坐标仅作为策略输入，场景没有实体篮筐。

## 观测与动作时序

当前为 471 维 float32、field-major、oldest-to-newest，逐字段乘 JSON scale，
没有额外归一化、输入裁剪、动作平滑或延迟队列。one-hot 维度变化时由配置调整偏移。

| 当前区间 | 内容 |
| --- | --- |
| `[0:3]` | 当前 skill_command one-hot |
| `[3:18]` | 5 帧重力投影 |
| `[18:33]` | 5 帧 pelvis 角速度 |
| `[33:178]` | 5 帧实测关节角减 nominal |
| `[178:323]` | 5 帧关节速度 |
| `[323:468]` | 5 帧实际已提交的裁剪后 target 减 nominal |
| `[468:471]` | 当前 hoop_pos_cyl，scale 为 `[0.25, 2, 0.25]` |

每步执行 `env.update → 输入事件 → 最新观测 → ONNX → clip(nominal + output) → env.step`。
只有提交成功后，目标才加入下一步动作历史。Ready 跳过 ONNX，但继续滚动传感器和保持目标历史。
Prepare 结束将实测传感器和实际初始目标平铺 5 次。历史记录最近 5 次策略采样，
稳定 100 Hz 时跨越 40 ms，不等于底层 DDS 连续 5 包，也不按时间戳重采样。
MuJoCo/SDK 的关节映射按名称处理，篮球 freejoint 不参与。

## 真机固定目标链路测试

以下操作会驱动真实电机进入初始姿态，应保持吊架支撑，无需篮球或显示会话。

```bash
python -m scripts.run_basketball --real --net_if eth0 --no_keyboard \
  --link_test --test_seconds 30 --log_dir logs/basketball_link
```

Prepare 后等待 Start，随后真实读取状态、运行 ONNX、记录日志；**每次提交始终是初始 PD
目标**，技能切换也不会将网络输出下发。30 秒后阻尼退出；Select 可提前停止。
报告自动写入 `link_report.json`，也可重新生成：

```bash
python -m scripts.analyze_basketball_link logs/basketball_link/实际运行目录
```

报告检查固定目标、历史一致性、周期/提交耗时和状态重复，状态为
`software_checks_passed`、`issues_found` 或 `incomplete`，只反映软件检查。
当前 SDK 不能测量传感器采集到执行器确认的端到端时延；Python 调用返回不是电机执行回执。
Python 循环也不是独立硬件看门狗，进程卡住不等于急停。

## 日志与离线检查

默认每次创建 `logs/basketball/时间戳_PID/`，包含：

- `metadata.json`：完整配置、CLI、模型 SHA-256、依赖版本和关节顺序。
- `trace.jsonl`：Prepare/Ready/active/damping 状态、完整观测、网络输出、实际目标及计时。
  控制样本记录 `command_id`、`hoop_pos_cyl`，切换记录 `skill_update` / `hoop_update` 事件。
- `summary.json`：退出原因、耗时和 tick 统计。

`--no_log` 可关闭详细记录（链路测试不允许）。日志采用同步文本缓冲，不创建动作队列。
每步序列化/写入可能影响周期，`cycle.work_ms` 包含样本日志开销；长时间运行注意磁盘用量。
`submit_call_ns` / `submit_return_ns` 是 Python API 边界；`submitted` 表示 API 成功返回。
`state_tick`、`unchanged_tick_ms` 可定位重复状态，不能据此推断完整 DDS/执行器延迟。

```bash
# 独立重建全部观测，检查历史、scale、命令和目标更新
python -m scripts.verify_basketball_history logs/basketball/实际运行目录

# 不连接环境或机器人：ONNX 单独计时 + 合成状态计算链路计时
python -m scripts.benchmark_basketball --warmup 200 --steps 10000 --output logs/bench_compute.json
python -m scripts.benchmark_basketball --warmup 200 --steps 3000 --paced --with_log --output logs/bench_log.json

# 离线测试；C++ 路径使用 fake SDK，不连接真实机器人
python -m unittest discover -s tests -p 'test_basketball*.py'
```

基准使用合成状态并遍历技能，不能作为投篮物理效果结论；`--trace` 可给 ONNX 单独计时段
重放当前格式观测。历史校验器针对新的 one-hot 导出格式，旧 phase 配置/日志需使用旧版工具。

# 23DoF G1 部署可行性评估实验报告

> 实验日期：2026-05-14
> 目的：评估 GR00T-WholeBodyControl 仓库的 release 29DoF 模型，能否以**最小代价**（不重训）部署到 23DoF G1 真机
> 结论：**不能**。release 模型把腰部 pitch 当作姿态控制器使用，硬件 23DoF 上不存在这个自由度，导致重心持续前移；最小代价路径不可行，必须走训练侧微调或推理时关节重定向

---

## 0. TL;DR

- 23DoF G1 真机比项目用的 29DoF 少 6 个关节：waist_roll、waist_pitch、左/右 wrist_pitch、左/右 wrist_yaw（**仍保留** wrist_roll）
- 设计了 sim 端虚拟拔关节方案（`--simulate-23dof`），把这 6 个关节硬钉死在 0、向策略反馈也写 0、PD 力矩清零
- 跑两组对照（A=原始参考 / B=清零 6 列参考），都出现持续小碎步前移、重心 X 单调增长
- 跑 C 对照（不加 flag）确认 penguin_walking 在原版不前移
- **关键发现**：release 模型默认站立姿态会调用腰部 pitch 后仰来抵消重心前倾；锁死腰 pitch = 拿走它的平衡工具
- 推荐下一步：**训练侧微调**（D1，高成本但干净），或低成本试 **关节重定向**（D2，把 waist_pitch 命令重新分配到 hip_pitch + ankle_pitch）

---

## 1. 背景与目标

### 1.1 现状
- 项目仓库：`/media/woan/.../lgy/GR00T-WholeBodyControl`，G1 全身控制，sim2sim 已跑通（`INSTALL_SIM2SIM.md`）
- 项目目标硬件：Unitree G1 **29DoF**
- 用户实际硬件：Unitree G1 **23DoF**
- 已有 release model：`gear_sonic_deploy/policy/release/model_*.onnx`，action_dim 硬编码 29

### 1.2 用户假设（待验证）
> "把输入和输出多余的 6 个关节置零，用最小代价看一下策略上机后的表现，再决定是否进行训练微调。"

### 1.3 验收准则（在 `INSTALL_SIM2SIM.md §13.6` 写过）
- 实验 A 能站立 ≥ 30 s 不摔
- 实验 A 至少能走 1 段下半身为主的动作
- 实验 B 全部参考动作能跑完不摔
- 关 flag 跑同一段动作没 regression

满足 → 真机谨慎试；任何一项不满足 → 转训练侧微调。

---

## 2. 硬件 vs 模型差异分析

### 2.1 23DoF 缺失的 6 个关节（来源：`gear_sonic_deploy/.../include/robot_parameters.hpp` 第 90-126 行 `G1JointIndex` 注释）

| MJCF idx | 关节名（MJCF） | 类型 | 23DoF 真机 |
|---------|---------------|------|------------|
| 13 | waist_roll_joint | 躯干侧倾 | ❌ 缺失 |
| 14 | waist_pitch_joint | 躯干俯仰 | ❌ 缺失 |
| 19 | left_wrist_roll_joint | 左腕滚转 | ✅ 保留 |
| 20 | left_wrist_pitch_joint | 左腕俯仰 | ❌ 缺失 |
| 21 | left_wrist_yaw_joint | 左腕偏航 | ❌ 缺失 |
| 26 | right_wrist_roll_joint | 右腕滚转 | ✅ 保留 |
| 27 | right_wrist_pitch_joint | 右腕俯仰 | ❌ 缺失 |
| 28 | right_wrist_yaw_joint | 右腕偏航 | ❌ 缺失 |

> ⚠️ **`gear_sonic/data_process/.../yaml` 里的 `anneal_23dof` mask** 是「0 waist + 6 wrist」（`g1_29dof_sonic_model12.yaml:382`），与硬件 23DoF 的「2 waist + 4 wrist」**不一致**，未来微调时不能直接复用。

### 2.2 IL ↔ MJCF 列序映射

参考动作 CSV（`joint_pos.csv` / `joint_vel.csv`）按 IL（IsaacLab）顺序存储，要清零必须先把 6 个 MJCF idx 转成 IL idx。来源：`gear_sonic_deploy/reference/gmr_to_deploy.py:65` 的 `MJ_TO_IL` 数组。

| MJCF idx | 关节 | IL CSV 列 |
|---------|------|----------|
| 13 | waist_roll | 5 |
| 14 | waist_pitch | 8 |
| 20 | left_wrist_pitch | 25 |
| 21 | left_wrist_yaw | 27 |
| 27 | right_wrist_pitch | 26 |
| 28 | right_wrist_yaw | 28 |

→ IL 列要清零的集合：`[5, 8, 25, 26, 27, 28]`

### 2.3 sim 端 MJCF 文件

`gear_sonic_deploy/g1/g1_23dof.xml` 的名字虽叫 23dof，但 `<jointpos>` 列表里**包含全部 29 个关节**——和 `g1_29dof.xml` 结构相同。换言之，**MJCF 层面没有真正的 23DoF 模型**，必须在仿真器层面虚拟拔关节。

---

## 3. 实施方案设计

### 3.1 候选方案对比（实验前权衡）

| 方案 | 改动量 | 真实度 | 风险 |
|------|--------|--------|------|
| 改 MJCF 把 6 个关节去掉 | 大（需重做 sensor 列表 / 名字索引） | 高 | sim2real 端各种索引会错位 |
| **sim 端虚拟拔关节** | 小（5 文件） | 中 | 选定方案 |
| 跳过 sim，直接上真机试 | 0 | — | 摔机风险 |
| 训练侧 fine-tune | 大（需训练 pipeline） | 高 | 留作 fallback |

### 3.2 选定方案：sim 端虚拟拔关节

**核心思路**：MuJoCo 仍然用 29 关节模型，但通过：
1. **Hard pin**：每步 `mj_step` 之后强制把这 6 个关节的 `qpos = qpos0`、`qvel = 0`（绝对刚度，重力打不开）
2. **High damping**：兜底，`dof_damping = 5000`
3. **Zero PD torque**：`compute_body_torques` 末尾把这 6 个 idx 的力矩清 0
4. **Zero LowState 反馈**：`PublishLowState` 把这 6 个 motor_state 的 q/dq/ddq/tau_est 写 0
5. **Zero CSV 列**（实验 B）：`zero_23dof_columns.py` 把参考动作 IL 列 [5, 8, 25, 26, 27, 28] 改成 0

C++ 部署侧（`G1_NUM_MOTOR=29`、ONNX action_dim=29）**不动**，验证策略输出 29 维但只让其中 23 维生效的退化行为。

---

## 4. 代码改动清单

### 4.1 修改的 4 个文件

| 文件 | 改动 |
|------|------|
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` | 模块级常量 `MISSING_23DOF_INDICES = [13,14,20,21,27,28]`；构造函数读 `config["SIMULATE_23DOF"]` 存入 `self.simulate_23dof`；`PublishLowState` 在 flag=True 时把 6 个 idx 的 motor_state 全置 0 |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | 模块级常量 `MISSING_23DOF_JOINT_NAMES`；`init_scene` 末尾（flag=True 时）调 `_lock_missing_23dof_joints` 设大 damping + 缓存 `(qpos_adr, dof_adr)`；新方法 `_pin_missing_23dof_joints` 硬复位 qpos/qvel；`sim_step` 在 `mj_step` 之后调用；`compute_body_torques` 末尾把 6 个 idx 的 PD 力矩清 0 |
| `gear_sonic/utils/mujoco_sim/configs.py` | `SimLoopConfig` 加 `simulate_23dof: bool = False` 和 `simulate_23dof_damping: float = 5000.0` |
| `gear_sonic/scripts/run_sim_loop.py` | 把上述两个字段塞进 `wbc_config["SIMULATE_23DOF*"]` |

### 4.2 新增 1 个文件

| 文件 | 作用 |
|------|------|
| `gear_sonic_deploy/reference/zero_23dof_columns.py` | 把 `example/` 14 段动作的 `joint_pos.csv` / `joint_vel.csv` 那 6 列清 0，写到 `example_23dof/`；其它文件原样复制 |

### 4.3 文档改动

| 文件 | 改动 |
|------|------|
| `INSTALL_SIM2SIM.md` | 新增 §12 完整键盘速查（基于官方 tutorials/keyboard.html）；新增 §13 23DoF 干跑验证流程，含表格、代码改动清单、实验 A/B 步骤、验收 checklist、风险提示 |

---

## 5. 实验设计

| 实验 | sim flag | 参考动作 | 目的 |
|------|---------|---------|------|
| **A** | `--simulate-23dof` | `reference/example/`（原始 29DoF 数据） | 测试「锁关节 + 策略仍按 29DoF 工作」时的表现 |
| **B** | `--simulate-23dof` | `reference/example_23dof/`（清零 6 列） | 测试「锁关节 + 参考动作也匹配 23DoF」是否更稳 |
| **C** | （不加） | `reference/example/` | 基准对照，确认 sim 改动没引入 regression、确认 penguin_walking 原始行为 |

启动方式：
- 终端 1：`python gear_sonic/scripts/run_sim_loop.py [--simulate-23dof]`
- 终端 2：`bash deploy.sh [--motion-data reference/example_23dof/] sim`
- 顺序：终端 1 先起 → 终端 2 → 终端 2 按 `]` → MuJoCo 窗口聚焦按 `9` → 终端 2 按 `T`

---

## 6. 实验过程与结果

### 6.1 实施过程中的迭代

#### 第一次：damping=200（太软）
- 启动后 MuJoCo 窗口里**腰部直接前倾**——重力是恒定力矩，damping 抗的是速度，所以会缓慢但不可逆地前倾
- **修正**：意识到 damping 不是位置约束，必须加位置约束才能"焊死"

#### 第二次：damping=5000 + 硬复位（绝对刚度）
- 改成在每个 `sim_step` 的 `mj_step` 之后 **强制写 `qpos = qpos0`、`qvel = 0`**
- damping 默认提到 5000 兜底
- 启动后腰部**完全静止**，外观像 23DoF 真机一样
- 副作用提醒：硬复位 qpos/qvel 不守恒能量（每步把转角抹平等于把重力做的功直接吃掉），但因为目标就是模拟"这个关节物理上不存在"，这是想要的行为

### 6.2 实验 A 结果（lock + 原始 CSV）
- 站立可以保持，但**有缓慢前倾趋势**（用户观察）
- 按 `T` 播 `penguin_walking`：**起步时持续向前小碎步走**，重心 X 单调前移
- **不可接受**——重心持续漂移意味着真机会撞墙 / 摔下平台

### 6.3 实验 B 结果（lock + 清零 CSV）
- 用 `python gear_sonic_deploy/reference/zero_23dof_columns.py` 生成 14 段清零参考
- `bash deploy.sh --motion-data reference/example_23dof/ sim` 切换数据源
- 现象与 A **基本一致**：同样小碎步前移、同样重心漂移
- 说明问题**不在参考动作侧**

### 6.4 实验 C 结果（无 lock + 原始 CSV）
- 关闭 `--simulate-23dof`，跑同一段 penguin_walking
- **完全正常**，不前移、姿态稳定
- 说明 sim 改动本身没有 regression，前移完全由"锁死 6 关节"引入

### 6.5 用户关键观察（原话）
> "29dof 版本的默认位置会调用腰部的 pitch 自由度 通过后仰让自己的姿态能保持中立"

这是整个实验最重要的产出——把"为什么前移"这个现象上升到了机理层面。

---

## 7. 关键发现：腰部 pitch 是 release 模型的姿态控制器

### 7.1 现象解释

29DoF release model 的标准站立姿态**不是关节全部为 0**，而是带有一定后仰的躯干：
- 腰部 pitch 微负（后仰几度）→ 躯干稍微后倾
- 这把上半身重量（手臂 + 头）的投影**往后推到脚跟附近**，与下半身的前倾力矩平衡
- 整体 COM 落在脚掌支撑面中心，无需迈步即可静止

锁死 waist_pitch = 0 之后：
- 躯干只能完全垂直
- 上半身重心自然落在**脚踝靠前**位置
- 重力产生持续前倾力矩，模型只能靠"迈步把脚送到 COM 下方"补偿
- 但每迈一步，COM 都会被惯性带着继续往前 → **正反馈**，于是出现持续小碎步前进

### 7.2 实验 B 为什么也不行？
- 实验 B 把**参考动作**的 6 维清零，但策略**自己的输出**仍然是 29 维的，而且策略内部状态（隐状态、观察）依然按 29DoF 训出来
- 策略想发命令到 waist_pitch（保持站立平衡），sim 把它压成 0 → 同样的失衡
- 清零参考只能解决"参考-反馈"那一段的 mismatch，解决不了"策略-世界"的 mismatch

### 7.3 这个发现的 OOD 性
- Release 模型在训练时显然见过腰部 pitch 自由活动的状态
- 我们强行把它压在 0，状态分布完全偏离训练分布
- 所以即使是看似无害的"站立"也会失败

---

## 8. 结论

### 8.1 主结论
> **「最小代价 zero-out」路径不可行。23DoF 真机不能直接跑 release 29DoF 模型，必须做训练侧微调或推理时关节重定向。**

### 8.2 验收准则对照

| 准则 | A 实测 | B 实测 |
|------|--------|--------|
| 站立 ≥ 30s 不摔 | ❌ 持续前移 | ❌ 持续前移 |
| 走 1 段下半身为主动作 | ❌ 重心漂移 | ❌ 重心漂移 |
| 全部参考动作跑完不摔 | — | ❌ |
| C 对照无 regression | ✅ | ✅ |

3 项关键准则全部不满足 → 走训练侧。

---

## 9. 后续路径选项

### D1：训练侧 fine-tune（推荐，正确答案）
- **做什么**：在训练数据里把 waist_roll、waist_pitch、L/R wrist_pitch、L/R wrist_yaw 这 6 维 mask 掉（observation 端置 0、action 端 mask）
- **代价**：需要训练 pipeline、GPU、数千~数万 step 微调
- **风险**：模型可能找不到不靠 waist_pitch 的平衡策略 → 需要 reward shaping 或 curriculum
- **关键坑**：yaml 里的 `anneal_23dof` mask 是 **0 waist + 6 wrist**，**与真机 23DoF 不匹配**（真机是 2 waist + 4 wrist）。直接复用会引入新的 mismatch，必须改 mask 配置

### D2：推理时关节重定向（备选，低成本试错）
- **做什么**：在 `unitree_sdk2py_bridge.LowCmdHandler` 里拦截策略输出
  - `motor_cmd[14].q`（waist_pitch）按系数 `k_hip ≈ -0.5` 加到 `motor_cmd[2/8].q`（hip_pitch_L/R）
  - 同时按 `k_ank ≈ +0.5` 加到 `motor_cmd[4/10].q`（ankle_pitch_L/R）
  - 把 motor[14].q 清 0
- **物理直觉**：通过下半身的"腿前倾 + 脚后仰"组合实现等效躯干后仰
- **关键风险**：策略观察的 hip / ankle joint_pos 会和它命令的不一致（因为 bridge 偷偷加了偏移），IMU 是真的会反映躯干姿态变化——**闭环可能进入新的 OOD**
- **代价**：30 分钟改完即可测试

### D3：假反馈欺骗策略（不推荐）
- **做什么**：把 LowState 里 motor_state[14].q 写成"等于策略命令"而不是 0
- **为什么不推荐**：策略的关节位置反馈被欺骗了，但 **IMU 是真的**——躯干没倾，IMU pitch 不变，模型仍会发现"我命令了但 IMU 没反应"，大概率出现和现在一样的症状

### 路径建议
1. 先花 30 分钟试 **D2**，如果观察重心漂移消失 → 真机可以小心试，并行准备 D1
2. D2 失败 → 直接走 **D1**

---

## 10. 风险与注意事项

### 10.1 已确认的风险
- **C++ 部署侧没改**：`G1_NUM_MOTOR=29`，ONNX 仍按 29 维 action 跑。真机上 motor_cmd[13/14/20/21/27/28] 不能被 SDK 错误下发到不存在的关节（Unitree SDK 默认这 6 个 motor index 是空的，理论 ignore，但**务必**真机端确认）
- **Release 模型对 OOD 的鲁棒性差**：实验已证明，把训练时见过的自由度强行压成 0 就足以让站立失败
- **anneal_23dof yaml mask 不匹配真机**：未来微调时不能直接套用，必须改 mask 配置

### 10.2 待验证的风险
- **关节重定向（D2）的 IMU 一致性**：理论上下半身实现的"等效后仰"会让 IMU 看到正常的躯干 pitch，与策略命令一致。但实际是否能稳定闭环未知
- **真机 SDK 对不存在关节的处理**：Unitree SDK2 在 23DoF 真机上对 motor_cmd[13]/[14]/[20]/[21]/[27]/[28] 的具体行为需现场验证

### 10.3 sim 端硬复位的副作用
- 硬复位 qpos/qvel 不守恒能量，会持续从系统抽走 / 注入能量
- 对仿真稳定性的影响：未观察到爆炸或异常
- 对策略观察的影响：观察值是基于 qpos 之前的状态读出，硬复位发生在 obs 之后下一个周期开始，所以策略看到的是"重置后"的状态，与"关节物理上不存在"的语义一致

---

## 11. 复现指引

### 11.1 一次性准备
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl

# 生成 23DoF 清零参考（实验 B 用）
source .venv_sim/bin/activate
python gear_sonic_deploy/reference/zero_23dof_columns.py
deactivate
```

### 11.2 实验 A：sim 锁关节 + 原始参考
```bash
# T1
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --simulate-23dof

# T2（另开终端）
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim
```

### 11.3 实验 B：sim 锁关节 + 清零参考
```bash
# T1 同上

# T2 多一个 --motion-data 参数
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --motion-data reference/example_23dof/ sim
```

### 11.4 实验 C：基准对照
```bash
# T1（不加 flag）
python gear_sonic/scripts/run_sim_loop.py

# T2 同实验 A
```

### 11.5 启动序列（任何实验）
1. 终端 1 起来后 MuJoCo 窗口出现，机器人悬停
2. 终端 2 起来后等待键盘
3. 终端 2 按 `]` → 启动 policy
4. **MuJoCo 窗口**鼠标点一下让它聚焦，按 `9` → 落地
5. 终端 2 按 `T` → 播放第 0 段动作
6. `N` / `P` 切换动作；`O` 退出

### 11.6 启动日志关键标志
开了 `--simulate-23dof` 时应能看到：
```
[run_sim_loop] simulate_23dof=True (damping=5000.0) — 6 missing joints will be zeroed in LowState and locked in MuJoCo
[simulate_23dof] locked 6 missing joints with damping=5000.0 + per-step qpos/qvel hard pin
```
没看到这两行 → flag 没透传到位，检查 `configs.py` / `run_sim_loop.py` / `base_sim.py`。

---

## 12. 附录

### 12.1 文件索引

| 文件 | 状态 | 作用 |
|------|------|------|
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` | 改 | 23DoF flag + LowState 6 维清零 |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | 改 | hard pin + 高 damping + PD 力矩清零 |
| `gear_sonic/utils/mujoco_sim/configs.py` | 改 | `simulate_23dof` / `simulate_23dof_damping` 配置项 |
| `gear_sonic/scripts/run_sim_loop.py` | 改 | CLI flag 透传到 wbc_config |
| `gear_sonic_deploy/reference/zero_23dof_columns.py` | 新 | 清零 IL 列 [5,8,25,26,27,28] |
| `gear_sonic_deploy/reference/example_23dof/` | 新 | 14 段清零参考动作 |
| `INSTALL_SIM2SIM.md §13` | 新 | 23DoF 干跑验证流程文档 |
| `INSTALL_SIM2SIM.md §12` | 新 | 完整键盘速查表 |
| `23DOF_EVALUATION_2026-05-14.md` | 本文件 | 实验报告 |

### 12.2 23DoF 关节缺失清单（速查）

```
缺失（6 个）：
  MJCF idx | 关节
       13   waist_roll_joint
       14   waist_pitch_joint   ← 关键：postural balance
       20   left_wrist_pitch_joint
       21   left_wrist_yaw_joint
       27   right_wrist_pitch_joint
       28   right_wrist_yaw_joint

保留（部分）：
       19   left_wrist_roll_joint   ← 不要误删
       26   right_wrist_roll_joint  ← 不要误删
```

IL CSV 列：清零 `[5, 8, 25, 26, 27, 28]`。

### 12.3 相关项目内文档
- `INSTALL_SIM2SIM.md` — sim2sim 安装与运行手册（含 §12 键盘速查、§13 23DoF 流程）
- `gear_sonic_deploy/.../include/robot_parameters.hpp:90-126` — `G1JointIndex` 枚举（硬件真值来源）
- `gear_sonic_deploy/reference/gmr_to_deploy.py` — GMR pkl → deploy pkl 转换器（含 `MJ_TO_IL` 映射定义）
- `gear_sonic/data/training_recipes/g1_29dof_sonic_model12.yaml:382` — `anneal_23dof` mask 定义（**注意**与真机 23DoF 不一致）

### 12.4 后续 TODO（如选 D1）
- [ ] 在 yaml 里新建 `anneal_hardware_23dof` mask（2 waist + 4 wrist 而非 0 waist + 6 wrist）
- [ ] 修改 observation pipeline，把这 6 维输入 mask 到 0
- [ ] 训练 / fine-tune 至少 10k step
- [ ] 重新运行本文件 §11 的 A/B/C 实验，验证 fine-tuned 模型在 sim 锁关节下能稳定站立 ≥ 30s
- [ ] 真机吊起来空载试站立 ≥ 1 分钟
- [ ] 真机落地试站立 + 简单步态

### 12.5 后续 TODO（如选 D2）
- [ ] 在 `unitree_sdk2py_bridge.py` 加 `RemapWaistToHipAnkle(low_cmd)` 函数
- [ ] 在 `LowCmdHandler` 里调用
- [ ] 在 `PublishLowState` 里反向修正 hip/ankle 反馈，让策略观察对齐它的命令
- [ ] 用 sim 验证重心漂移是否消失
- [ ] 调系数 `k_hip` / `k_ank`（初值 ±0.5），看哪个组合最稳
- [ ] 失败 → 回到 D1

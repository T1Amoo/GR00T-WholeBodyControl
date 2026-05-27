# Sonic 库架构学习笔记

> 项目位置：`/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic/`
> 学习路线：从大到小（目录结构 → MDP 组件 → 训练循环 → 数据流 → 模型）。
> 本文档为不断追加结构：先建骨架，每次探讨完一个话题就把结论补在末尾。

---

## 0. 一句话定位

`gear_sonic/` 是个把 **IsaacLab manager-env**、**HuggingFace TRL（PPO）** 和 **自家 motion-library** 拼起来的 humanoid motion-imitation 训练工程。三层架构：

```
仿真环境 (envs/)  →  RL 训练循环 (trl/)  →  配置驱动入口 (train/eval scripts + config/)
                              ↑
                  共享工具链 (utils/, data/, isaac_utils/)
```

整个系统的运行形状由 `config/` 树决定（Hydra `defaults: list` 组合 + `+exp=...` 选择实验）。

---

## 1. Repo 顶层

```
gear_sonic/
├── train_agent_trl.py         训练入口；Hydra 加载 config，构造 env+model，调 PPO trainer 跑
├── eval_agent_trl.py          单机 eval：加载 ckpt 跑仿真出 metrics
├── eval_exp.py                批量 eval：扫多 motion 文件 / 多 ckpt
├── pyproject.toml             包定义（Hatch 构建）
├── version.py / __init__.py   常规
└── __pycache__/               忽略
```

---

## 2. `config/` — Hydra 配置树（架构骨架，决定一切组合）

整个项目用 Hydra `defaults: list` + `+key=value` 组合配置，所以这个树的形状 = 系统的逻辑结构。

```
config/
├── base.yaml                  顶层默认：定义 num_envs / device / seed / 加载哪几个组件
├── base/
│   ├── hydra.yaml             Hydra 自身设置（输出目录、override 行为）
│   └── structure.yaml         output 目录命名 schema
├── base_eval.yaml             eval 任务的顶层默认
├── eval_exp.yaml              批量 eval 的顶层默认
│
├── algo/                      RL 算法配置（PPO 超参）
│   ├── ppo_im_phc.yaml        ★ 实际用的：PPO 主超参（actor_lr / entropy_coef / etc.）
│   └── trl/                   把 HF TRL PPOConfig 的字段单独成 group
│
├── trainer/                   Trainer 类选择
│   ├── trl.yaml               基础 PPO trainer
│   └── trl_ppo_aux.yaml       带 auxiliary loss 的 PPO trainer
│
├── actor_critic/              Actor-Critic 网络结构
│   ├── mlp.yaml               最朴素的 MLP A-C
│   ├── encoders/              观测编码器（如 token encoder）
│   ├── decoders/              动作头
│   ├── critics/               critic 头
│   ├── quantizers/            VQ/code-book 类（可能为 universal token 设计）
│   └── universal_token/       Universal-Token 风格的 A-C 结构
│
├── manager_env/               ★ IsaacLab Manager-based env 的全部 MDP 组件
│   ├── base_env.yaml          顶层 env：装载下面所有 manager
│   ├── observations/          obs manager
│   │   ├── policy/global.yaml + local_dir_hist.yaml   policy obs
│   │   ├── critic/             critic obs（可独立于 policy obs，常带特权信息）
│   │   ├── tokenizer/          离散 token obs
│   │   └── terms/              单个 obs 项的实现配置
│   ├── actions/                action manager（关节 PD 目标等）
│   │   ├── tracking/           跟踪类 action
│   │   └── terms/              单个 action term
│   ├── commands/               command manager（motion clip 抽样、anchor、smpl 通路）
│   │   ├── tracking/
│   │   └── terms/
│   ├── rewards/                reward manager（每个奖励项一份）
│   │   ├── tracking/
│   │   └── terms/
│   ├── terminations/           termination manager（time_out / ee / foot）
│   │   ├── tracking/
│   │   └── terms/
│   ├── events/                 event manager（reset / push / domain rand）
│   │   ├── tracking/
│   │   └── terms/
│   ├── curriculum/             curriculum manager
│   └── recorders/              episode recorder（用于 dataset 导出）
│
├── exp/                        ★ 实验组合层（每个 yaml = 一次 paper experiment）
│   └── manager/universal_token/all_modes/
│       ├── sonic_release.yaml          原 release ckpt 的训练配置
│       ├── sonic_release_23dof.yaml    我们 23DoF baseline
│       ├── sonic_release_23dof_h5d.yaml / h5e_c / h5e_e   死曲线调试分支
│       ├── sonic_h2.yaml / sonic_bones_seed.yaml           其他实验
│
├── callbacks/                  HF TRL callbacks
│   ├── model_save.yaml         save_interval / 输出目录
│   ├── wandb.yaml              W&B 上报
│   ├── im_eval.yaml            im(itation) eval
│   ├── im_resample.yaml        adp_samp（自适应失败率重采样）
│   └── read_eval.yaml          读取离线 eval 结果
│
├── aux_losses/                 auxiliary loss 注册表（行为克隆类）
│   ├── terms/                  单项 aux loss
│   └── universal_token/        组合后的 aux loss bundle
│
├── missing_dofs/               硬件缺失关节配置
│   └── 23dof_hardware.yaml     ★ D1 23DoF 配置（mjcf / il 双索引集合）
│
├── opt/                        优化器/scheduler 默认（wandb 这里其实是命名 quirk）
│   └── wandb.yaml
```

---

## 3. `envs/` — IsaacLab 环境层

```
envs/
├── manager_env/
│   ├── modular_tracking_env_cfg.py    ★ 顶层 env class，组合 obs/action/reward managers
│   └── __init__.py
├── wrapper/
│   ├── manager_env_wrapper.py         ★ 在 IsaacLab env 外裹一层：tensor reshaping、
│   │                                     missing-dof action mask、与 TRL trainer 接口对齐
│   └── missing_dofs_env.py            硬件缺失关节专用 wrapper（apply mask、调整 obs/action）
└── env_utils/
    └── joint_utils.py                 关节索引/名字互转
```

---

## 4. `trl/` — RL 训练层（基于 HuggingFace `trl` 改造）

```
trl/
├── trainer/
│   ├── ppo_trainer.py             ★ 主训练循环（经过定制的 PPO，sync_advantage_normalization 等）
│   └── ppo_trainer_aux_loss.py    带 auxiliary loss 的版本
├── modules/                       模型组件
│   ├── actor_critic_modules.py    A-C 基础组件
│   ├── universal_token_modules.py ★ Universal-Token 版本的 actor 头/编码器
│   ├── base_module.py             基类
│   └── data_utils.py              tensor 转换工具
├── losses/
│   └── token_losses.py            token 类损失函数
├── callbacks/                     训练流程钩子
│   ├── hv_callback_handler.py     callback handler 主控
│   ├── im_eval_callback.py        训练中 imitation eval
│   ├── im_resample_callback.py    ★ adp_samp，按失败率重采样 motion
│   ├── model_save_callback.py     ckpt 存盘
│   ├── wandb_callback.py          wandb 推送
│   └── read_eval_callback.py      读外部 eval 文件
└── utils/                         共享小工具
    ├── common.py / data.py        基础 utils
    ├── math.py / rotation_conversion.py / kornia_transform.py / torch_transform.py
    ├── order_converter.py         joint order 重排
    ├── rl.py                      RL helper（GAE 等）
    └── scheduler.py               学习率 scheduler
```

---

## 5. `utils/` — 共享工具（运行时业务逻辑）

```
utils/
├── motion_lib/                ★ motion 数据库
│   ├── motion_lib_base.py     基类，含 fk_batch、加载流程；pose_aa zero-out 在这层
│   ├── motion_lib_robot.py    机器人特化版
│   ├── skeleton.py            skeleton 描述
│   └── torch_humanoid_batch.py 批处理 FK
├── mujoco_sim/                 mujoco 旁路仿真（sim2sim eval 用）
│   ├── base_sim.py / robot.py
│   ├── configs.py / metric_utils.py
│   └── image_publish_utils.py
├── data_collection/            data collection runtime
│   ├── episode_state.py / telemetry.py
│   ├── keyboard_subscriber.py / zmq_state_subscriber.py
│   ├── transforms.py / text_to_speech.py
├── inference/                  推理时辅助
│   ├── initial_poses.py        启动姿态
│   └── vla_utils.py            VLA 接口
├── joint_constants.py          关节顺序/名字常量表
├── joint_mask.py               ★ apply_missing_dof_mask
├── batch_normalizer.py         batch normalizer（用于 obs 标准化）
├── average_meters.py           滑动平均统计
├── config_utils.py             config 解析工具
├── common.py / logging.py / inference_helpers.py
```

---

## 6. `data/` + `data_process/` — 数据管线

```
data/                           训练时的 dataset 接口（HF datasets / lerobot 风格）
├── exporter.py                  episode → dataset 导出
├── features_sonic_vla.py        feature schema for VLA training
├── video_writer.py              视频导出
└── robot_model/robot_model.py   机器人模型类描述

data_process/                    一次性数据预处理脚本（不进训练循环）
├── convert_soma_csv_to_motion_lib.py
├── extract_soma_joints_from_bvh.py
├── filter_and_copy_bones_data.py
├── split_pkl_files.py
└── zero_23dof_motion_lib.py    ★ 已确认是 no-op
```

---

## 7. 周边

```
camera/                          真机相机驱动（仅推理阶段用）
├── composed_camera.py / sensor.py / sensor_server.py
└── drivers/{oak,realsense,usb_camera,dummy}.py

scripts/                         独立可执行入口
├── launch_inference.py          推理 daemon
├── launch_data_collection.py    人采遥操数据
├── pico_manager_thread_server.py  Pico 头显 server
├── process_dataset.py / run_data_exporter.py
├── run_camera_viewer.py         相机调试
├── run_sim_loop.py              纯仿真 loop（无 RL）
└── run_vla_inference.py         VLA 推理

isaac_utils/                     纯数学
├── maths.py / rotations.py
```

---

## 8. 关键架构线索（后续讨论伏笔）

1. **配置驱动**：`config/exp/.../*.yaml` 的 `defaults: list` 把 `manager_env/{obs,action,reward,...}/`、`algo/`、`actor_critic/` 各组件像积木拼起来，再用 `+exp=...` 选择某次实验。所有运行时配置都从这层 hydra-compose 出来。
2. **MDP 全 manager 化**：obs/action/reward/termination/event/curriculum/command 七大 manager 都是 IsaacLab pattern——每个 yaml 一项，相加成完整 MDP。
3. **动作流双层**：`envs/manager_env/` 给 IsaacLab 用，`envs/wrapper/manager_env_wrapper.py` 给 TRL trainer 用（apply_missing_dof_mask 就在这层）。
4. **模型可插拔**：`actor_critic/` 下分 encoder / decoder / critic / quantizer / universal_token 五类，`algo/ppo_im_phc.yaml` 选其中一种组合。
5. **callbacks 解耦训练循环**：im_eval / model_save / wandb / im_resample 都是 callback，主 trainer 不需要知道。

---

## 推荐学习顺序

1. **config 组合规则**：搞懂 `base.yaml` → `exp/.../sonic_release_23dof.yaml` → 如何被 compose 成最终 cfg。
2. **manager_env MDP**：每个 manager 的职责 + 一个具体 reward/observation term 的实现路径。
3. **trainer 主循环**：`ppo_trainer.py` 的 rollout → advantage → update 流程，与标准 PPO 的差异点。
4. **motion_lib 数据流**：motion 文件 → fk_batch → command/observation/reward 的链路。
5. **模型架构**：universal_token 是什么，actor_critic 怎么组装。

---

## 探讨记录（按时间追加）

<!-- 后续每次深入一个话题，就在这下面追加一节 -->

---

## A. Hydra compose 实战:从 cmdline 到运行时 cfg(2026-05-26)

### A.1 入口:`@hydra.main` 与 search path

`gear_sonic/train_agent_trl.py:493`:

```python
@hydra.main(version_base="1.2", config_path="config", config_name="base")
def main(config):
    ...
```

`config_path="config"` → 相对脚本目录的 `config/`(也就是 `gear_sonic/config/`)。`config_name="base"` → 起点是 `config/base.yaml`。

### A.2 三种 yaml 文件的 package 行为

每个 yaml 顶部可能有也可能没有 `# @package` 行,这一行决定它的内容**塞到 cfg 树的哪个分支**:

| 顶部行 | 行为 | 例 |
|--------|------|----|
| `# @package _global_` | 内容直接合并到根 | `config/exp/.../sonic_release.yaml` |
| `# @package _group_`(等价于无,默认) | 内容挂到 group 路径下 | `config/callbacks/im_resample.yaml` 内容挂到 `callbacks.im_resample.*` |
| 无 + 在 defaults list 里被引用 | 与 _group_ 同 | 多数 manager_env terms |

**确认方法**:
```bash
head -3 config/callbacks/*.yaml config/exp/manager/universal_token/all_modes/*.yaml | head -40
```

### A.3 `defaults: list` 是怎么 compose 的

看一个实验入口 `config/exp/manager/universal_token/all_modes/sonic_release_23dof.yaml` 的 `defaults:` 段(它本身是 `_global_` 包):

```yaml
# @package _global_
defaults:
  - /base@_here_
  - override /algo: ppo_im_phc
  - override /trainer: trl
  - override /actor_critic: universal_token/...
  - override /manager_env: base_env
  - override /missing_dofs: 23dof_hardware
  - _self_
```

执行顺序:
1. Hydra 读 `config_name=base` → `config/base.yaml`(顶层)
2. 用户 cmdline `+exp=manager/universal_token/all_modes/sonic_release_23dof` → 把这个文件以 `_global_` 合并进根
3. 这个文件 `defaults:` 里的 `override /algo: ppo_im_phc` → 用 `config/algo/ppo_im_phc.yaml` **替换** `algo` 这个 group(原 base 里的 algo 被丢)
4. `_self_` 表示"本文件内的字段在最后再合并一次"——所以本文件后面写的 `manager_env.commands.motion.static_reset_prob: 0.05` 会覆盖 group 默认值

**关键点**:`override /xxx: yyy` 不是 import,是**整个 group 替换**。如果你想"在 base 的 algo 上加一个字段",别用 override,直接在 `_self_` 段写 `algo.entropy_coef: 0.02`。

### A.4 cmdline override 六种语法

(这次 H5i 踩过 `callbacks.im_resample.motion_resample_frequency=50` 的具体坑。)

| 写法 | 语义 | 何时用 |
|------|------|--------|
| `key=value` | **覆盖**已有 key,不存在则抛 "not in struct" | 改已有超参 |
| `+key=value` | **追加**新 key(就算 schema 没有也写入) | 加自定义字段 |
| `++key=value` | **覆盖或追加**(不抛) | 不确定 key 在不在用这个 |
| `~key` | 删除 key | 极少用 |
| `+exp=path/to/exp` | 加 `_global_` 包(常用入口) | 选实验 |
| `key=null` | 显式置空 | 关掉某 callback |

**踩坑诊断**:报 `Key 'X' is not in struct, object_type=dict`,**先别盲目换 `++`**。先确认 `X` 实际挂在 cfg 树的哪条路径(看 yaml 顶部的 `# @package` 决定 package),再选合适语法。

### A.5 看运行时 cfg 长什么样

每次 run 都会在 `logs_rl/.../<run_dir>/.hydra/.hydra/` 留下三个文件:

- `config.yaml` — compose 后的最终 cfg(给 task_function 的那个 OmegaConf object)
- `overrides.yaml` — cmdline 上敲的所有 `+key=val` 列表(**审计 H6 vs 云端 4096 配置差异时第一时间看这个**)
- `hydra.yaml` — hydra 自身配置

实际用法:

```bash
# 想知道 H6 用的 motion_resample_frequency 是啥
cat logs_rl/.../sonic_release_23dof_h6_idle_test-20260525_174242/.hydra/.hydra/overrides.yaml

# 看完整运行 cfg 的某分支
yq '.manager_env.commands.motion' logs_rl/.../config.yaml
```

### A.6 一个值得记的细节:`_self_` 位置

```yaml
defaults:
  - override /algo: ppo_im_phc
  - _self_                     # _self_ 在最后 → 本文件的字段覆盖 group 默认
```

vs

```yaml
defaults:
  - _self_                     # _self_ 在前 → group 默认覆盖本文件字段
  - override /algo: ppo_im_phc
```

两者语义**反向**。Sonic 的 exp yaml 都是 `_self_` 在最后,所以"在 exp yaml 里直接写 `algo.entropy_coef: 0.02` 会赢"。

---

## B. MotionLibBase 数据流:从 PKL 到 reward(2026-05-26)

> 关键文件:`gear_sonic/utils/motion_lib/motion_lib_base.py`(2000 行),`torch_humanoid_batch.py`,`gear_sonic/envs/manager_env/mdp/commands.py`(TrackingCommand)
>
> 记录这条主要因为我们 H1 修复时踩过"全局 vs 本地索引"的坑(参见 `D1_23DOF_DEBUG_LOG.md` 2026-05-25 晚那条)。

### B.1 三层索引空间(必须分清)

```
全局空间                          本地空间                      env-level
─────────                          ─────────                      ──────────
_motion_data_keys                 _curr_motion_ids              env.motion_ids[env_id]
  numpy array, str                 1D long tensor                1D long tensor
  size=_num_unique_motions         size=max_num_load_motions     size=num_envs
  ~129785 (filter 过的全集)        ~min(num_envs, 1024)          == num_envs

  "所有发现 motion 的名字"          "当前 GPU 上加载哪几个"        "每个 env 当前在跑哪个"
```

**注意**:`env.motion_ids` 存的是**本地** idx(0..len(_curr_motion_ids)-1)。下游 `get_motion_state(motion_ids,...)` 用本地 idx 去 `_motion_lengths` 等已加载缓冲区做 gather。

### B.2 数据加载流程

`MotionLibBase.load_data()`(`motion_lib_base.py:472`)被 `__init__` 调用:

```
__init__()
  ├── _load_data_keys()        扫 PKL 目录,填 _motion_data_keys (全局名字)
  ├── load_data()              首次加载默认子集
  │   ├── sample_motions(N)    从全局按 _sampling_batch_prob 采 N 个 → global_ids
  │   ├── load_motions(global_ids)
  │   │   ├── _load_motion_files()          读 PKL,拿 pose_aa / trans
  │   │   ├── (★) zero out missing pose_aa  ← 23DoF 修复在这里
  │   │   ├── mesh_parsers.fk_batch()       SMPL+H FK,出 dof_pos / body_pos / body_quat
  │   │   ├── 把 body_pos_w/dof_pos/... .to(self._device)   全压 GPU
  │   │   └── 填 _motion_lengths / _motion_dt / 各种 *_w buffer
  │   └── _curr_motion_ids = global_ids  ← 本地空间建立
  └── 后续 step 里的 resample 走 ImResampleCallback
```

23DoF 修复改的就是 `(★)` 那行——`pose_aa[:, missing_pose_aa_idx, :] = 0`。如果不改,下游 `fk_batch` 用 29DoF axis-angle 算 reference body_pos,跟物理 23DoF 锁的 default ≈ 0 完全脱节。详见 `D1_23DOF_DEBUG_LOG.md` 2026-05-20 那条。

### B.3 motion_resample 与 ImResampleCallback

callback 路径:`gear_sonic/trl/callbacks/im_resample_callback.py`,在 `on_step_end` 里:

```python
if (global_step + 1) % motion_resample_frequency == 0:
    motion_lib.resample_motions(...)
```

`resample_motions` 干两件事:

1. 调用 `sample_motions(max_num_load_motions)` 从全局采新 batch → 新 global_ids
2. `_curr_motion_ids = new_global_ids`,**清掉**旧的 GPU buffer,加载新的

意义:每隔 `motion_resample_frequency` 步,GPU 上的 motion 子集就洗一遍牌。frequency 越小,actor 在固定计算预算下见到的 unique motion 越多(本地 H5i 实验从 250→50,coverage ×5)。**但 frequency 太小,IO 占总时间比就显著上升**(每次 reload ~30s)。

### B.4 sampling 概率:adp_samp(自适应失败率)

`_sampling_batch_prob` 不是均匀分布。`gear_sonic/utils/motion_lib/motion_lib_base.py` 维护每个 motion 的 `failure_rate`(从 reward callback / termination 反馈),失败率高的 motion **被采样概率更高**(curriculum)。

但是 `motion.yaml` 默认有个 `adaptive_sampling.uniform_sampling_rate: 0.1` —— 即使 adp_samp 完全把某 motion 概率压到 0,uniform 也会以 10% 概率随机 mix 进来。这是 H1 修复"static_reset 跳过当前 batch 没问题"的兜底:`neutral_idle_loop_001` 早晚被 uniform 抽中。

### B.5 TrackingCommand 怎么用 motion_lib

`gear_sonic/envs/manager_env/mdp/commands.py` 的 `TrackingCommand.__init__` / `_resample_command`:

```python
# __init__ 末段(motion_lib 已加载):
self.motion_ids = self.motion_lib.sample_motions(num_envs)         # 本地 idx
self.motion_start_time_steps = ...                                  # int tensor

# _resample_command(到时间或 reset 触发):
new_local_ids = self.motion_lib.sample_motions(len(env_ids))
self.motion_ids[env_ids] = new_local_ids
self.motion_start_time_steps[env_ids] = ...

# H1 修复在这之后插入:
if self._static_motion_id is not None and self.cfg.static_reset_prob > 0:
    static_mask = torch.rand(len(env_ids), device=...) < self.cfg.static_reset_prob
    if static_mask.any():
        static_env_ids = env_ids[static_mask]
        self.motion_ids[static_env_ids] = self._static_motion_id    # ← 必须本地 idx
        self.motion_start_time_steps[static_env_ids] = 0
```

`_static_motion_id` 在 `__init__` 末段算:

```python
keys = self.motion_lib.curr_motion_keys                             # str list of LOADED names
matches = [i for i, k in enumerate(keys) if "neutral_idle_loop_001" in k]
self._static_motion_id = matches[0] if matches else None
```

**关键点**:不能用 `_motion_data_keys`(全局) 做 enumerate,要用 `curr_motion_keys`(本地)。如果用了全局,会出现"打印 motion_id=3105,看着像成功,sim 一步就 PhysX CUDA error"(`max_num_load_motions=1024` 越界)。

### B.6 reward / termination 怎么读到 reference

`gear_sonic/envs/manager_env/mdp/rewards.py:59-78`(tracking_anchor_pos)、`:138-159`(tracking_relative_body_pos):

```python
def tracking_anchor_pos(env, command_name, std):
    cmd: TrackingCommand = env.command_manager.get_term(command_name)
    error = (cmd.anchor_pos_w - env.scene["robot"].data.root_pos_w).norm(dim=-1)
    return torch.exp(-error / std)
```

`cmd.anchor_pos_w` 是 property,从 `motion_lib.get_motion_state(self.motion_ids, current_time)` 现算的——不是 reset 时缓存。所以 reset 时只需要把 `motion_ids` / `motion_start_time_steps` 写对,reward 自动跟着对。

### B.7 一张图总结数据流

```
PKL files (data/motion_lib_bones_seed/...)
    │
    │ _load_data_keys()
    ▼
_motion_data_keys (全局 str array, ~129785)
    │
    │ sample_motions(N) + adp_samp prob
    ▼
global_ids (1D long, N=min(num_envs,1024))
    │
    │ _load_motion_files() → pose_aa[:, missing_pose_aa_idx]=0 → fk_batch()
    ▼
GPU buffers: _motion_dof_pos, _body_pos_w, _body_quat_w, ...   (本地空间 0..N-1)
    │
    │ get_motion_state(local_ids, time) ← 每步调用
    ▼
TrackingCommand.anchor_pos_w / dof_pos / body_pos_w (per env)
    │
    │ rewards.py / terminations.py 读
    ▼
PPO reward / done signal
```

每隔 `motion_resample_frequency` 步,GPU buffers 整体重建一次(走 ImResampleCallback)。

---

## C. 待补充的下一批主题(自己挑顺序)

按对训练 / 部署理解收益降序:

1. **PPO trainer 主循环**(`trl/trainer/ppo_trainer.py`):collection vs learning 时间(从 log 看 4096 env 是 3.2s + 1.5s),`num_steps_per_env=24` 决定 rollout 长度,`num_mini_batches=4` × `num_learning_epochs=5` = 20 次 grad update / iter。GAE λ、ratio clip、`sync_advantage_normalization` 与原版 TRL 的差异。
2. **Actor-critic 模型组合**(`trl/modules/universal_token_modules.py` + `config/actor_critic/universal_token/`):encoder + quantizer + decoder 的形状,obs 1751 维怎么走到 64 维 token,token 怎么进 decoder 出 action 29 维。这一块直接对应 deploy 的 `encoder.onnx` + `decoder.onnx`。
3. **Wrapper 层 missing_dof mask**(`envs/wrapper/missing_dofs_env.py`):action mask 怎么乘上,obs 哪些维度被 zero,跟 motion_lib 那层 `pose_aa[:, missing, :]=0` 配合的语义。
4. **Episode_Reward / term_step / Env/Metrics 这些指标怎么算**:`Env/term_step/foot_pos_xyz/mean` 是"被 foot termination 终止的 episode 的平均长度",`Env/Episode_Termination/foot_pos_xyz: 0.6336` 是"占总 termination 的 63%"。看 termination manager 的实现 + ppo_trainer 的 episode log 聚合逻辑。
5. **Adp_samp 全套指标**:`Env/adp_samp/{prob_max_over_uniform, effective_num_bins, num_concentrated_bins}` 的具体计算,以及"什么时候 adp_samp 就过强了反而要调低 weight"。
6. **Deploy split pipeline**:`gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp` 里 obs concat 顺序,`encoder.onnx`(1751→64)+ `decoder.onnx`(994=64 token+930 dec_extra → action 29),与训练时 actor 的对应关系。`g1_deploy_onnx_ref.cpp:3125` 的强制 zero-out。
7. **Motion 文件 PKL schema**:`data_process/convert_*` 各脚本输入输出,`pose_aa` (3+J,3) axis-angle,`trans` (T,3) global pos,SMPL+H 关节顺序,与 `MISSING_23DOF_INDICES_MJCF` 的对应。

每挑一个,我都可以接着这个文档继续按 A/B 的密度写一段(行号 + 代码 + 经验,200 行内)。

---

## D. PPO trainer 主循环:从 rollout 到 grad step(2026-05-26)

> 关键文件:`gear_sonic/trl/trainer/ppo_trainer.py`(2297 行,继承 HF `trl.PPOTrainer`),配套配置 `gear_sonic/config/algo/ppo_im_phc.yaml`。
>
> 这一节追的就是 wandb log 里那行 `collection_time / learn_time / fps` 是怎么算出来的,以及 ratio clip / GAE / adaptive LR 在哪儿。

### D.1 入口与一次 iteration 的形状

`TRLPPOTrainer.train()`(`ppo_trainer.py:1634`)是整个训练的核心 for-loop:

```python
# ppo_trainer.py:1697
for batch_idx in range(1, args.num_total_batches + 1):
    # 1. 收集 num_steps_per_env 步 rollout(no_grad)
    obs_dict = self._rollout_step(model, obs_dict)        # 884
    # 2. 整理 rollout buffer
    rollout_data = self._get_rollout_data(...)            # 1122
    # 3. num_ppo_epochs × num_minibatch × micro_batch 三层 grad update
    for ppo_epoch_idx in range(args.num_ppo_epochs):
        for mini_batch_start in range(0, local_batch_size, local_mini_batch_size):
            for micro_batch_start in range(0, local_mini_batch_size, per_device_train_batch_size):
                forward_results = self._forward_model(...)     # 1255
                loss_dict = self._compute_loss(...)            # 1305 → _compute_ppo_loss 1333
                accelerator.backward(loss_dict["loss"])
                grad_norm = self._gradient_clipping()
                if grad_norm is not None: optimizer.step()     # 1788 NaN-skip
    # 4. 跨 GPU 同步 normalizer / adp_samp
    self.sync_running_mean_std()                          # 1933
    self.sync_adaptive_sampling()                         # 1958
    # 5. 算 metrics + wandb log + lr_scheduler.step + callbacks
```

`ppo_im_phc.yaml` 实际配置:`num_steps_per_env=32`、`num_learning_epochs=5`、`num_mini_batches=4`。所以**每 iter 做 5×4=20 次 grad update**。

`local_batch_size = num_envs × num_steps_per_env`(单 GPU,云端 4096 envs × 32 步 = 131072 transitions);`local_mini_batch_size = local_batch_size / num_mini_batches = 32768`。如果 `per_device_train_batch_size < local_mini_batch_size` 还会再切 micro batch(梯度累积)。

### D.2 `_rollout_step`:32 步收集 + GAE

`ppo_trainer.py:884` 全程在 `with torch.no_grad():` 里:

```python
# 909
for i in range(self.num_steps_per_env):
    policy_state_dict = self.policy_step(policy_model, obs_dict, cur_dones=dones)  # 807
    # storage 缓存当前 obs / action / mu / sigma / log_prob / done / rewards
    self.storage.update_key(...)
    obs_dict, rewards, dones, infos = self.env.step(policy_state_dict)
```

收集完 32 步后:

```python
# 977-989: 用 critic 一次性算所有时刻的 V(s)
all_values = self._chunked_value_evaluate(value_model, all_obs_dict, episode_attnmask)  # 851
values, last_values = all_values[:-1], all_values[-1]
# 993-997: time_outs(踩到 max_episode_length 而不是 termination)的 reward 加上 bootstrap
new_rewards = rewards + gamma * time_outs * values
# 999-1009: 算 returns / advantages 并写回 storage
returns, advantages = self._compute_returns(values, last_values, ...)  # 2084
```

`_chunked_value_evaluate`(`851`)用 chunk_size=1024 切片防止 critic 一次吃太多显存——critic 输入是带 episode 维的 (B, T+1, obs_dim),T+1=33,B=4096,直接前向 OOM。

### D.3 GAE 算法

`_compute_returns`(`ppo_trainer.py:2084`)是教科书 GAE,反向遍历:

```python
# 2114-2122
for step in reversed(range(num_steps)):
    next_values = last_values if step==T-1 else values[step+1]
    next_is_not_terminal = 1.0 - dones[step].float()
    delta = rewards[step] + (1-d) * γ * next_values - values[step]
    advantage = delta + (1-d) * γ * λ * advantage      # GAE 累加
    returns[step] = advantage + values[step]
```

`gamma=0.99 / lam=0.95`(`ppo_im_phc.yaml:13-14`)。注意 `dones[step]` 在 timeout 那一帧也是 1,但前面 `new_rewards += γ * time_outs * V(s_T)` 把 bootstrap 项补回来了——这是 IsaacLab 的标准处理(timeout 不应该截断 V 估计)。

### D.4 `sync_advantage_normalization`(与原版 TRL 不同的点)

`ppo_im_phc.yaml:29` 默认开。`_compute_returns:2126-2135`:

```python
if self.sync_advantage_normalization:
    advantages = self.accelerator.gather(advantages)        # 跨 GPU 拼接
    advantages = (advantages - global_mean) / (global_std + 1e-8)
    advantages = advantages.reshape(num_processes, ...)[process_index]   # 切回本地
else:
    advantages = (advantages - local_mean) / (local_std + 1e-8)          # 仅本地标准化
```

**意义**:多卡训练时,每张卡各自标准化会让"在某张卡上恰好都是难轨迹"的 batch advantage 全部接近 0;全局标准化保证 advantage 量级跨卡一致。本地 4060 单卡两者等价;云端 L20 单卡也等价(因为 `num_processes=1`)。**只在多卡训练时这个 flag 才有差异**。

### D.5 `_compute_ppo_loss`:三件套 + adaptive LR

`ppo_trainer.py:1333` 是 PPO 的本体。先算 KL 用于自适应 LR:

```python
# 1372-1382
kl = sum( log(σ_new/σ_old) + (σ_old² + (μ_old-μ_new)²) / (2σ_new²) - 0.5, axis=-1 )
local_kl_mean = kl.mean()
kl_mean = self.accelerator.gather(local_kl_mean).mean()
self._adjust_learning_rate_based_on_kl(kl_mean, optimizer)   # 2142
```

**这是 KL 对正态分布的解析式,不是经验估计**。然后 `_adjust_learning_rate_based_on_kl`:

```python
# 2154-2166: desired_kl=0.01
if kl_mean > 0.02:           new_lr = max(1e-5, lr/1.5)    # KL 太大,缩
elif 0 < kl_mean < 0.005:    new_lr = min(2e-4, lr*1.5)    # KL 太小,放
```

注意 `args.learning_rate` **被原地改**——下次 `_compute_ppo_loss` 也会基于这个新 lr 调整。这是 LegEd Lab 风格的 schedule,跟 HF 默认的 LR scheduler 完全不一样。本地 H6 训练时 lr 经常在 1e-5 ~ 5e-5 之间漂移,就是这个机制在动。

接下来三个 loss 项(`1391-1411`):

```python
# value loss(clipped)
vpredclipped = clamp(vpred, V_old ± cliprange_value)
vf_loss = max( (vpred - return)², (vpredclipped - return)² ).mean()
# policy loss(clipped surrogate)
ratio = exp(new_logp - old_logp)
pg_loss = max( -A*ratio, -A*clamp(ratio, 1-cliprange, 1+cliprange) ).mean()
# entropy bonus
entropy_loss = -entropy.mean()
loss = pg_loss + vf_coef * vf_loss + entropy_coef * entropy_loss
```

`clip_param=0.2` / `value_loss_coef=1.0` / `entropy_coef=0.01`(`ppo_im_phc.yaml:12,15,16`)。 `entropy_coef` 跟 wandb 那行 `entropy=-37` 直接相关:entropy 越负,这一项的 loss 越大,鼓励 actor 张开 σ。entropy 一直单调下降说明 actor 在收紧策略——后期看 `init_noise_std=0.05` + 学到的 σ。

### D.6 NaN-skip 与 `_gradient_clipping`

`ppo_trainer.py:1786-1790`:

```python
grad_norm = self._gradient_clipping()
if grad_norm is not None:
    optimizer.step()
else:
    print("NaN in gradient! Skipped!!!!")
```

`_gradient_clipping` 内部用 `max_grad_norm=1.0`(`ppo_im_phc.yaml:19`)做 clip,如果发现任何 param 的 grad 含 NaN 直接返回 None,这一 micro batch 整步丢弃。**这就是 H6 训练里偶尔会看到 "NaN in gradient! Skipped!!!!" 但训练继续不崩的原因**——单步丢弃,不是整个 iter 失败。

### D.7 wandb 那一行的来源

```
iter 1198/100000  reward=1.81  entropy=-42.3  collection_time=3.18s  learn_time=1.58s  fps=27345
```

`ppo_trainer.py:1843-1888` 全员:

```python
# 1841: eps = 累计 episode 数 / 累计 wall time
eps = int(self.state.episode / (time.time() - start_time))
# 1847-1854: 当前 iter 所有 done env 的 reward 平均
metrics["objective/rewards"] = mean(state.rewbuffer)
# 1856-1862: episode length
metrics["objective/length"] = mean(state.lenbuffer)
# 1880-1885: fps = 步数 × env 数 × 卡数 / (collection + learn)
"fps": num_steps_per_env * num_envs * num_processes / (collection_time + learn_time),
"collection_time": collection_time,    # rollout 32 步耗时
"learn_time": learn_time,              # 20 次 grad update 耗时
```

**实测**:本地 4060 1024 envs `collection ~3s + learn ~1.5s = 4.5s`,fps≈7300;云端 L20 4096 envs `collection ~3.2s + learn ~1.5s = 4.7s`,fps≈27000。**collection 几乎不增长是因为 IsaacLab GPU sim 是 batch 并行的**;learn 只动 5×4=20 次梯度,跟 batch_size 弱相关。

### D.8 callbacks 怎么挂进来

`ppo_trainer.py:1685` 训练开始时 `self.callback_handler.on_train_begin(...)`,`1916` 每个 iter 末尾 `on_step_end` —— im_resample / model_save / wandb / im_eval 都在 `on_step_end` 触发。看 `gear_sonic/trl/callbacks/im_resample_callback.py`:

```python
def on_step_end(self, args, state, control, **kwargs):
    if (state.global_step + 1) % motion_resample_frequency == 0:
        motion_lib.resample_motions(...)
```

`global_step` 在 `1873` 自增,所以 `motion_resample_frequency=50` 意味着每 50 iter 重采一次 motion。**注意不是每 50 个 env step**,是每 50 个 outer iter——本地 H5i 50 iter ≈ 50×4.5s = 225s,云端 ≈ 50×4.7s = 235s。

### D.9 与原版 HF `trl.PPOTrainer` 的差异点

| 项 | HF 原版 | Sonic 改造 |
|----|---------|-----------|
| 数据源 | language model rollout | IsaacLab `env.step()` 实时仿真 |
| reward | reward model + KL penalty | env reward(tracking 加权) |
| storage | `RolloutBuffer`,token level | `RolloutStorage`,(env, step, dim) |
| GAE | 可选 | **强制 GAE+bootstrap**(`_rollout_step:993`) |
| LR schedule | HF scheduler | **adaptive KL-based**(`_adjust_learning_rate_based_on_kl`) |
| advantage norm | local | **可选 cross-GPU global norm**(`sync_advantage_normalization`) |
| NaN handling | 没有 | **micro-batch skip**(`1790`) |
| advantage 维度 | scalar | **multi-critic weighted**(`1399-1403`) |

最后一项很有意思:`multi_critic_advantage_weights` 让 critic 输出多维(每维代表一类 reward 比如 anchor / dof / bodyvel),按权重加权进 ratio loss。Sonic 配置默认 None(单 critic),但留了这个口子。

### D.10 一个值得记的"陷阱":`empty_cache_every_n_ppo_epoch`

`ppo_im_phc.yaml:30` 配 `empty_cache_every_n_ppo_epoch: 3`,但代码 `ppo_trainer.py:1813-1816` 把这块 **comment 掉了**:

```python
# if self.empty_cache_every_n_ppo_epoch > 0 and (ppo_epoch_idx + 1) % ... == 0:
#     gc.collect()
#     torch.cuda.empty_cache()
```

config 字段还在,但 PPO epoch 内不再清缓存——只在 iter 末尾 `1913-1914` 清一次。如果你看到 OOM 日志却觉得"不是已经设了 empty_cache 吗",就是这个原因。要恢复就把那段反注释。

---


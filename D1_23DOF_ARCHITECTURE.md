# D1 23DoF 改动总览:从 29DoF release 到 23DoF 部署

> 写于 2026-05-26。归集所有为支持 D1 G1 23DoF 真机部署而做的代码/配置改动。
> 配套文档:
> - `D1_23DOF_DEBUG_LOG.md` — 历次 hypothesis 调试时间线(H1/H2/H4/H5*/H6)
> - `D1_23DOF_DESIGN_2026-05-15.md` — 早期设计构想
> - `D1_23DOF_USAGE_2026-05-18.md` — 训练/部署如何启用
> - `SONIC_ARCH_NOTES.md` § A/B/D — Hydra compose / motion_lib / PPO trainer 一般机制

---

## 第一部分:原理(Why)

### 1.1 一句话定义问题

Sonic 官方 release 训练在 **G1 29DoF 仿真模型**上,真机是 **23DoF G1 D1 变种**,缺 6 个电机:

| 关节名 | MJCF idx | IL idx | 物理意义 |
|--------|---------|--------|---------|
| `waist_roll_joint` | 13 | 5 | 腰部侧倾 |
| `waist_pitch_joint` | 14 | 8 | 腰部俯仰(承重姿态) |
| `left_wrist_pitch_joint` | 20 | 25 | 左腕俯仰 |
| `left_wrist_yaw_joint` | 21 | 27 | 左腕偏航 |
| `right_wrist_pitch_joint` | 27 | 26 | 右腕俯仰 |
| `right_wrist_yaw_joint` | 28 | 28 | 右腕偏航 |

注:**MJCF 顺序和 IsaacLab 顺序对相同关节给出不同 idx**。两套要分别维护。

### 1.2 为什么不能直接把 ckpt 装上

actor 网络结构是 **29 维 action 输出**,obs 里也带 29 维 `last_action / joint_pos / joint_vel`。直接装上机会有四类问题(每类对应一种修复策略):

1. **Action 越权**:actor 给 6 个不存在的电机发 ±330° 目标 → 真机 PD 把这些目标转成扭矩 → 关节飞出 → 跌倒。
2. **Obs 错位**:真机 23DoF SDK 在缺失关节槽位上塞 0,但仿真是真物理仿出来的非 0 值 → 训练分布与部署分布不一致 → actor OOD。
3. **Reference 错位**:motion_lib 喂的 reference body_pos / dof_pos 还在按 29DoF 算(包括缺失关节的 axis-angle 旋转贡献) → reward 信号错位 → actor 学到的"对齐 reference"在真机上根本不存在的对应关系。
4. **Reset 分布零覆盖**:训练里 robot 永远从 motion 起始姿态 reset(都不是 idle),actor 从来没在"机器人静止 + reference 静止"区域被监督 → 部署刚开机平地零关节那一帧落在 39σ OOD 尾部 → μ 飞出。

### 1.3 修复总策略 = 五道防线

```
                     [训练侧]                                 [部署侧]
                        │                                        │
   ┌────────────────────┼─────────────┬───────────────┐         │
   │                    │             │               │         │
 ① Reference         ② Sim          ③ Actor        ④ Obs/Reward    ⑤ Deploy guard
   pose_aa zero       lock 6 dofs    σ 锁紧         mask 6 dim     强制写 0
   (motion_lib)       (env wrapper)  (a-c module)   (mdp terms)    (cpp:3125)
        │                  │              │              │               │
        ▼                  ▼              ▼              ▼               ▼
   reference 与         物理状态与    actor 在 6 dim   obs 输入与    最后兜底,
   真机对齐             真机对齐      不学/不漂移      真机一致      防 actor 没
                                                                   收敛全场也安全
```

**核心思想**:让"训练时 actor 看到的世界"和"部署时 actor 看到的世界"在 6 个缺失关节上完全一致 —— 全部归 0,且 actor 在那 6 维不输出意义。

---

## 第二部分:技术(How)—— 五道防线展开

### 2.1 防线 ① — Reference 抹零(motion_lib 层)

**问题**:motion PKL 里 `pose_aa` 是 SMPL 24+5 关节的 axis-angle,29DoF 部分都填了真值。直接走 `mesh_parsers.fk_batch()` → 出来的 reference body_pos / dof_pos 包含 6 个缺失关节的旋转贡献 → 跟物理上锁住的 23DoF 不一致。

**修复**:在 axis-angle 进 FK **之前** 把缺失关节那 6 行轴向归零。

文件 `gear_sonic/utils/motion_lib/motion_lib_base.py:1871-1877`:
```python
# Hardware-absent joints: zero the corresponding axis-angles so FK produces
# body_pos / dof_pos consistent with the physics where these joints are locked
# at default qpos (assumed 0). Must run AFTER all augmentations.
missing_dofs_mjcf = self.m_cfg.get("missing_dofs_mjcf", None)
if missing_dofs_mjcf:
    missing_pose_aa_idx = [d + 1 for d in missing_dofs_mjcf]
    pose_aa[:, missing_pose_aa_idx, :] = 0
```

`+1` 是因为 `pose_aa` 第 0 行是 root,后面才是 J 个关节,所以 mjcf joint idx d 对应 pose_aa 行 d+1。

**为什么不是改 dof_pos / body_pos**:FK 后那些是导出量,改它们要保证拓扑一致非常脆。`pose_aa` 是 FK 的输入,改最干净。

**踩过的坑**:之前 `data_process/zero_23dof_motion_lib.py` 想离线把 PKL 的 `dof_pos` 抹零,但这个脚本是 **no-op** —— 因为 `motion_lib_base` 加载流程根本不读 `dof_pos`,而是从 `pose_aa` 重新跑 FK。详见 `D1_23DOF_DEBUG_LOG.md` 2026-05-20 那条。

### 2.2 防线 ② — Sim 锁住缺失关节

**问题**:即使 actor 在 6 dim 输出 0,如果 IsaacLab / Mujoco 把这 6 个关节当成正常自由度物理仿真,重力还是会让它们漂走(尤其 `waist_pitch` 受躯干重量)。obs 里读到的 `joint_pos` 就不是 0,还是跟真机偏。

**两套仿真(训练/部署测试)各自锁:**

#### IsaacLab(训练用)— `gear_sonic/envs/wrapper/missing_dofs_env.py`

子类化 `ManagerBasedRLEnv`,在每次 `sim.step` 之后立刻把 6 关节硬钉到 default qpos:

```python
# missing_dofs_env.py:51-58
orig_sim_step = self.sim.step
def _sim_step_with_lock(*args, **kwargs):
    ret = orig_sim_step(*args, **kwargs)
    self._pin_missing_joints()      # write_joint_state_to_sim,把 qpos/qvel 强制写回
    return ret
self.sim.step = _sim_step_with_lock
```

放置时机:`sim.step → pin → record_post_physics_decimation_step → render → scene.update`。`scene.update` 把仿真状态读进 buffer,pin 在它之前,obs/reward 都看到锁住后的状态。**没用 EventTermCfg(interval) 因为 interval 在 reward 之后,会先用 unpinned 状态算一帧 reward。**

#### Mujoco(sim2sim 测试用)— `gear_sonic/utils/mujoco_sim/base_sim.py:261-282`

双保险:
```python
self.mj_model.dof_damping[dof_adr] = 5000.0          # 大阻尼
# 每次 mj_step 之后:
for qpos_adr, dof_adr in self._missing_23dof_addrs:
    self.mj_data.qpos[qpos_adr] = default_qpos       # 硬钉
    self.mj_data.qvel[dof_adr] = 0.0
```

启用方式:命令行 `python ... run_sim_loop.py +simulate_23dof=true` → `run_sim_loop.py:40-42` 把 flag 注入 `wbc_config["SIMULATE_23DOF"]`。**没加这个 flag 就退回 29DoF 物理仿真,机器人秒倒**(参见 [Memory: groot_deploy_d1_23dof_sim_flag])。

### 2.3 防线 ③ — Actor 输出锁紧(分布层)

**问题**:即使 sim 锁了,reference 抹了,actor 网络的 mu/sigma 头还是会从 obs 推出 6 个非零 μ —— 这些 μ 不影响仿真状态(被 sim ② 兜了),但它们会:
- 进 PPO log_prob 计算 → 反向梯度污染 trunk
- 进 obs 的 `last_action`(被防线 ④ 抹零兜住,但 actor 内部还在算)

**修复**:在 actor 构造分布时,对 6 个 locked dim 强制 μ=0、σ=const(1e-4)。

文件 `gear_sonic/trl/modules/actor_critic_modules.py:139-150, 313-316`:
```python
# __init__:
locked_il = algo_config.get("missing_dofs_action_il", None)
if locked_il is not None and len(locked_il) > 0:
    mask = torch.zeros(self.num_actions, dtype=torch.bool)
    for idx in locked_il:
        mask[int(idx)] = True
    self.register_buffer("locked_action_mask", mask, persistent=False)
    self.locked_action_std_const = float(algo_config.get("missing_dofs_action_std", 1e-4))

# build_distribution:
if self.locked_action_mask is not None:
    mean = mean.masked_fill(self.locked_action_mask, 0.0)
    std = std.masked_fill(self.locked_action_mask, self.locked_action_std_const)
```

**意义**:这 6 dim 上 KL = log(σ_new/σ_old) + ... 永远是常数 → 反传梯度恒为 0 → trunk 不会因 6 个不存在的 dim 学坏。

启用方式:exp yaml 里 `algo.config.missing_dofs_action_il: ${missing_dofs.indices_il}`(`sonic_release_23dof_h5h.yaml:34`)。

### 2.4 防线 ④ — Obs / Reward 抹零(MDP 层)

**问题**:即使 sim、actor、reference 都对齐了,obs manager 里某些 term 是从 stock IsaacLab 函数算的(`joint_pos_rel`、`joint_vel_rel`、`last_action`),它们对 6 dim 没有特殊处理 —— 训练时这 6 dim 偶尔有 numerical 残留,部署时真机 SDK 报 0。这一致性差距还是会让 actor OOD。

**修复**:替换 obs term 的 func 为 masked 版本。

文件 `gear_sonic/envs/manager_env/mdp/observations.py:35-69`:
```python
def joint_pos_rel_masked(env, asset_cfg=SceneEntityCfg("robot")):
    out = _stock_joint_pos_rel(env, asset_cfg).clone()
    out[..., MISSING_23DOF_INDICES_IL] = 0.0
    return out
# 同理 joint_vel_rel_masked / last_action_masked
```

`last_action_masked` 还有第二层意义:即使 ③ 把分布锁了,actor 输出的 mean **可能在 deploy 端没被 ③ 兜住**(老 ckpt) → `env.action_manager.action` 存的是 raw 29-d → 下一步 `last_action` 就被污染。masked 版本斩断这个反馈环。

`gear_sonic/envs/manager_env/mdp/observations.py:72-82` 还有个 `joint_pos_multi_future_select_joints_masked` 用于 tokenizer obs(选择性关节子集)。

#### Reward 也要避开 6 dim

`joint_limit` reward 算所有关节的 limit penalty。改成只 regex match 23 个真存在的关节(`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof.yaml:67-93`):
```yaml
joint_limit:
  params:
    asset_cfg:
      joint_names:
        - left_hip_pitch_joint
        - ...                  # 23 个,显式列出
```

**意义**:6 个 missing joint 被 sim ② 物理锁在 default qpos(总在 limit 内),所以即使不排除也是 0 贡献 —— 但显式排除让"reward 不依赖物理 lock"的语义更干净。

### 2.5 防线 ⑤ — Deploy 端兜底

**问题**:就算训练完美,网络上线时 actor 收敛精度 + onnx 量化都可能让 6 个 locked dim 输出 ±0.05 的小残差 —— 真机 PD 还是会把它当 target → 漂走。

**修复**:cpp 部署代码在每次推理后强制把 6 个 idx 写 0。

文件 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:3124-3125`:
```cpp
static const int D1_MISSING_IL_IDX[6] = {5, 8, 25, 26, 27, 28};
for (int idx : D1_MISSING_IL_IDX) floatarr[idx] = 0.0f;
```

注意这里用的是 **IL idx**(行 25/26 顺序也排序了,跟 `joint_constants.py` 的 `[5,8,25,27,26,28]` 不完全同序但是同集合)。这是 actor 输出后、PD 计算前的最后一道闸。

### 2.6 五道防线的冗余设计

任何一道单独失效,系统不一定崩 —— 但训练分布与部署分布的距离会变大。比如:
- 只有 ⑤,没有 ①②③④:训练 reference 还在 29DoF,actor 学的是错位 reference 上的策略,部署兜底零关节但 actor 在 OOD obs 区飞掉(H1 失败模式)。
- 只有 ①②③④,没有 ⑤:训练完美,但 onnx 量化误差让 6 dim 输出 ±0.05,真机 PD 把它当 target 累积漂移。

5 层一起上才稳。**这是显式工程化的"训练-部署对称"约束**,不是过度防御。

---

## 第三部分:具体改动点清单(What)

按照"配置 → 数据 → 模型 → 仿真 → 部署"分组列出。所有改动都对 `enabled: true` 的 `missing_dofs` 配置生效;`enabled: false` 时全部 no-op,29DoF release 训练完全不受影响。

### 3.1 单一真相源(Single Source of Truth)

| 文件 | 内容 | 用途 |
|------|------|------|
| `gear_sonic/utils/joint_constants.py` | `MISSING_23DOF_INDICES_MJCF=[13,14,20,21,27,28]`<br>`MISSING_23DOF_INDICES_IL=[5,8,25,27,26,28]`<br>`MISSING_23DOF_JOINT_NAMES=[6 names]` | python 全局唯一来源 |
| `gear_sonic/config/missing_dofs/23dof_hardware.yaml` | `enabled / variant / indices_mjcf / indices_il / joint_names` | hydra 配置层来源 |
| `gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp:3124` | `D1_MISSING_IL_IDX[6]` | cpp 部署侧来源 |

**约定**:python 改 → yaml 改 → cpp 改,三处必须保持同步。`joint_mask.py` / `motion_lib_base.py` / `actor_critic_modules.py` / `observations.py` / `manager_env_wrapper.py` / `missing_dofs_env.py` / `mujoco_sim/base_sim.py` / `mujoco_sim/unitree_sdk2py_bridge.py` 全部 import 自 `joint_constants.py`,不写 magic number。

### 3.2 工具层

| 文件 | 函数 | 改动 |
|------|------|------|
| `gear_sonic/utils/joint_mask.py` | `apply_missing_dof_mask(actions, missing_idx)` | 新文件:把 actions 在 missing_idx 列就地置 0 |
| `gear_sonic/utils/motion_lib/motion_lib_base.py:1874-1877` | `load_motions` 内的 augmentation 末尾 | 新增 `if missing_dofs_mjcf: pose_aa[:, [d+1 for d in missing_dofs_mjcf], :] = 0` |

### 3.3 仿真环境层

| 文件 | 改动 |
|------|------|
| `gear_sonic/envs/wrapper/missing_dofs_env.py` | **新建** `MissingDofsLockEnv(ManagerBasedRLEnv)`,wrap `self.sim.step` 每次步进后 pin 6 关节到 default qpos/qvel |
| `gear_sonic/envs/wrapper/manager_env_wrapper.py:16-17, 848-849` | import `MISSING_23DOF_INDICES_IL` + `apply_missing_dof_mask`,在 actions 进 IsaacLab env 之前抹零 |
| `gear_sonic/train_agent_trl.py:149-156` | 读 `config.missing_dofs.enabled`,True 时把 env class 换成 `MissingDofsLockEnv` |
| `gear_sonic/utils/mujoco_sim/base_sim.py:27-28, 261-282, 329-331` | mujoco 仿真侧锁 6 关节(damping=5000 + per-step hard-pin),并在 torque 计算时把 6 个 missing 的 torque 强制 0 |
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:25, 65, 186-191` | SDK bridge 在 simulate_23dof 模式下把 6 关节的 motor_state 全置 0(模仿真机 SDK 行为) |
| `gear_sonic/scripts/run_sim_loop.py:40-44` | 把 cmdline `+simulate_23dof=true` 翻译成 `wbc_config["SIMULATE_23DOF"]` |

### 3.4 MDP terms

| 文件 | 改动 |
|------|------|
| `gear_sonic/envs/manager_env/mdp/observations.py:22, 35-82` | 新增 `joint_pos_rel_masked` / `joint_vel_rel_masked` / `last_action_masked` / `joint_pos_multi_future_select_joints_masked` |
| `gear_sonic/envs/manager_env/mdp/actions.py:7` | re-export `apply_missing_dof_mask`(给 action term 用) |

### 3.5 Actor / Critic

| 文件 | 改动 |
|------|------|
| `gear_sonic/trl/modules/actor_critic_modules.py:139-150` | `__init__` 读 `missing_dofs_action_il` / `missing_dofs_action_std`,注册 `locked_action_mask` buffer |
| `gear_sonic/trl/modules/actor_critic_modules.py:313-316` | `build_distribution`:对 mask 位置 `mean.masked_fill(0.0)`、`std.masked_fill(1e-4)` |

### 3.6 Hydra 配置

| 文件 | 改动 |
|------|------|
| `gear_sonic/config/missing_dofs/23dof_hardware.yaml` | **新建** missing_dofs config group(enabled/indices_mjcf/indices_il/joint_names) |
| `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof.yaml` | **新建** 23DoF baseline exp。引入 `/missing_dofs@missing_dofs: 23dof_hardware`,把 `motion.motion_lib_cfg.missing_dofs_mjcf` 接到 SSOT,override 4 个 obs term 为 masked 版本,override `joint_limit` reward 用显式 23 关节列表 |
| `sonic_release_23dof_h5d/h5e_c/h5e_e/h5f/h5g/h5h.yaml` | 调试分支(各种 hypothesis) |
| `sonic_release_23dof_h6_idle.yaml` | **当前云端在跑的实验**:在 H5h 基础上加 `static_reset_prob: 0.05` + `static_motion_name: "neutral_idle_loop_001"` 修复 H1 |
| `gear_sonic/config/manager_env/commands/terms/motion.yaml` | 新增 `static_reset_prob: 0.0` / `static_motion_name: null` 字段(默认 no-op) |

### 3.7 Command(reset 分布修复 — H1)

| 文件 | 改动 |
|------|------|
| `gear_sonic/envs/manager_env/mdp/commands.py:248-264` | `TrackingCommand.__init__` 末段:从 `motion_lib.curr_motion_keys`(本地 idx,**不能用全局**)找 static motion,缓存 `_static_motion_id` |
| `gear_sonic/envs/manager_env/mdp/commands.py:2920-2940` | `_resample_command`:5% env 在 reset 时把 `motion_ids` 改成 `_static_motion_id`、`motion_start_time_steps` 钉到 0 |
| `gear_sonic/envs/manager_env/mdp/commands.py:4194-4196` | `TrackingCommandCfg` dataclass 新增字段 `static_reset_prob: float = 0.0` / `static_motion_name: str | None = None` |

修复原理:见 `D1_23DOF_DEBUG_LOG.md` 2026-05-25 H1 一节。简言之,训练 reset 分布从未覆盖"机器人静止 + reference 静止",actor 没在那个区域被监督 → 部署平地零关节落在 39σ OOD 尾部 → μ 飞出 ±330°。让 5% env 进 idle motion 区域被 tracking reward 自然约束 → actor 学到"近似静止 + 缓慢漂移",collapse 该 OOD 区域。

### 3.8 Deploy 部署侧

| 文件 | 改动 |
|------|------|
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:3124-3125` | `static const int D1_MISSING_IL_IDX[6] = {5, 8, 25, 26, 27, 28}`;每次 actor 输出后强制写 0 |
| `gear_sonic_deploy/policy/release/observation_config.yaml` | 替换为官方 issue #122 提供的 `g1_wrist_joints_10_clean.yaml`(修复 1762 vs 1751 维度匹配)。详见 [Memory: groot_deploy_obs_yaml_1762_vs_1751] |

### 3.9 测试

| 文件 | 覆盖 |
|------|------|
| `tests/test_joint_constants.py` | mjcf vs il idx 一致性、joint_names 与 idx 长度对齐 |
| `tests/test_missing_dofs_config.py` | yaml 字段与 python 常量同步 |
| `tests/test_action_mask.py` | `apply_missing_dof_mask` 正确性 |
| `tests/test_obs_mask.py` | 三个 masked obs term 行为 |
| `tests/test_reward_mask.py` | joint_limit reward 排除 6 dim |
| `tests/test_zero_23dof_motion_lib.py` | motion_lib `pose_aa` 抹零路径 |
| `tests/test_sonic_release_23dof_yaml.py` | exp yaml compose 后 missing_dofs 字段贯通 |

---

## 第四部分:启用一行总结

启用 23DoF 训练 + 部署只需:
```bash
# 训练
python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release_23dof_h6_idle +checkpoint=sonic_release/last.pt

# Mujoco sim2sim 测试
python gear_sonic/scripts/run_sim_loop.py +simulate_23dof=true ...

# 真机部署:cpp 编译时无 flag 切换 —— 5 个 idx 是常量,行 3124-3125 永远生效
```

`+exp=*sonic_release_23dof*` 会触发 `defaults: - /missing_dofs@missing_dofs: 23dof_hardware`,**SSOT 联动**:
- `motion_lib_cfg.missing_dofs_mjcf` ← `${missing_dofs.indices_mjcf}` → 防线 ①
- `manager_env_wrapper` 读 `config.missing_dofs.enabled` → 启 `MissingDofsLockEnv` → 防线 ②
- `algo.config.missing_dofs_action_il` ← `${missing_dofs.indices_il}` → 防线 ③
- obs/reward override 直接写在 exp yaml → 防线 ④

29DoF release 训练:`enabled: false` 全链路 no-op。

---

## 附录:历史迭代节点

| Hypothesis | 修复点 | 结论 |
|-----------|-------|------|
| H1(reset 分布)| commands.py + motion.yaml + h6_idle.yaml | **当前云端验证中** |
| H2(权重/locked dim)| 排除 — 权重健康,locked w_norm 仅大 23%,deploy line 3125 兜住 |
| H4(SMPL encoder)| `encoder_sample_probs.smpl: 0.0` | 已合入 sonic_release_23dof.yaml |
| H5d/e_c/e_e(死曲线)| 各种 mask 调整尝试 | 大部分回退 |
| H5h(actor σ 锁紧)| `missing_dofs_action_il/std` | 合入 |
| H6(= H5h + H1) | 当前实验 | 跑中,等 5k iter ckpt |

完整时间线见 `D1_23DOF_DEBUG_LOG.md`。

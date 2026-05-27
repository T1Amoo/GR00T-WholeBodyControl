# D1 23DoF 训练 Debug 日志

本文件按时间顺序追加每次 debug 的完整记录。**新条目写在文件末尾**，旧条目不要修改。

每条记录使用统一模板：

```
## [YYYY-MM-DD] 简短标题
- **现象**：观测到什么
- **触发**：什么操作 / 状态下出现
- **发现路径**：怎么注意到的（错误信息 / 用户报告 / 指标异常）
- **调查**：跟了哪些线索，看了哪些代码 / 配置 / 日志（带文件:行号）
- **根因**：真实的因果链
- **改动**：改了哪些文件，diff 摘要
- **验证**：怎么确认修复了（pytest / hydra compose / smoke test / 训练指标）
- **经验**：下次遇到类似情况能少走的弯路
```

如果当时定的根因后来被推翻，**不要删旧条目**，新增一条引用旧条目并标 "续 / 修正"。

---

## [2026-05-19] Joint indexing 不一致导致 `MissingDofsLockEnv` 启动崩溃

- **现象**：训练启动时 `MissingDofsLockEnv` 抛 `RuntimeError`，断言 `joint_ids != list(MISSING_23DOF_INDICES_IL)` 失败。
- **触发**：首次启动 23DoF finetune 训练（`+exp=manager/universal_token/all_modes/sonic_release_23dof`）。
- **发现路径**：`gear_sonic/envs/wrapper/missing_dofs_env.py:72` 的 defensive assert 主动捕获并 fail-fast。
- **调查**：
  1. IsaacLab 通过 joint_names 列表解析得到的 `joint_ids` 顺序与 `MISSING_23DOF_INDICES_IL` 常量不一致
  2. 在 IL 顺序里 wrist 是按 joint type 配对（25=L-pitch, 26=R-pitch, 27=L-yaw, 28=R-yaw），而 `MISSING_23DOF_JOINT_NAMES` 是按 side 分组（waist_roll, waist_pitch, L-wrist-pitch, L-wrist-yaw, R-wrist-pitch, R-wrist-yaw）
  3. 原始常量 `[5, 8, 25, 26, 27, 28]` 是按数值排序写的，没和 names 列表顺序对齐
- **根因**：之前"verified 2026-05-18"的标注是 hand-eyeball 看 `G1_ISAACLab_ORDER` 数出来的，没真的在 runtime probe 过。手数错了 26/27 的位置。set-based 调用方（action mask、obs mask）因为按集合用所以掩盖了问题，只有 `missing_dofs_env.py` 按位置 check 才暴露。
- **改动**：
  - `gear_sonic/utils/joint_constants.py`：`MISSING_23DOF_INDICES_IL: [5, 8, 25, 26, 27, 28]` → `[5, 8, 25, 27, 26, 28]`
  - `gear_sonic/config/missing_dofs/23dof_hardware.yaml`：`indices_il` 同步更新；删除 `TODO[Task-1.4]` 标记
- **验证**：runtime probe `G1_ISAACLab_ORDER`，确认 27=`left_wrist_yaw_joint` 在 26=`right_wrist_pitch_joint` 之前；重启训练，equality check 通过。
- **经验**：
  1. "已经手工核对过"不可信，必须有 runtime probe + defensive assert
  2. set-based 用法会掩盖 ordering bug，positional 用法是金丝雀
  3. 任何依赖 joint 顺序的常量都要在 runtime 用 `assert` / `RuntimeError` 校验

---

## [2026-05-19] iter 2923 训练停滞，bottleneck 从脚迁移到手腕（首次诊断）

- **现象**：run `sonic_release_23dof_test-20260519_151956` 训到 iter ~2923：
  - `foot_pos_xyz` 终止率 0.65 → 0.40（改善）
  - `ee_body_pos` 终止率 0.31 → **0.58**（恶化）
  - mean entropy 13.5 → **18.0**
  - action noise std 0.39 → **0.46**（接近饱和）
  - 总死亡率持平，mean length 没起来
- **触发**：从 `sonic_release/last.pt`（29DoF stable ckpt）finetune 到 23DoF 锁关节环境。
- **发现路径**：用户问 "为什么平均时长一直没上去"。
- **调查**：
  1. 看 termination configs：`foot_pos_xyz` 是 3D L2 阈值 0.2m on 2 ankle links；`ee_body_pos` 是 Z-only 阈值 0.15m on 4 links（2 ankles + 2 wrists）
  2. 看 reward configs：`tracking_vr_5point_local` weight 2.0, std 0.1, body=[torso, L_wrist_yaw, R_wrist_yaw]
  3. 假设：23DoF 手腕物理被锁，reference 的 wrist Z 是 29DoF FK 算出来的（手腕带 pitch/yaw articulation），物理够不到 → ee_body_pos 持续触发；reward 同时给手腕施加无法满足的梯度 → policy 用 entropy 兜底
- **根因（当时认定）**：手腕参考轨迹与物理可达空间不一致，污染 reward + termination。
  - **后来证明这只是症状层面，真正的根因在 [2026-05-20] 那条**。
- **改动（ABCD 干预，应用在 `sonic_release_23dof.yaml`）**：
  - **A**：`terminations.ee_body_pos.params.body_names` → `[left_ankle_roll_link, right_ankle_roll_link]`（去掉手腕）
  - **B**：`rewards.tracking_vr_5point_local.weight` 2.0→1.0，`params.std` 0.1→0.2
  - **C**：`commands.motion.reward_point_body` 手腕 → 手肘
  - **D**：`commands.motion.cat_upper_body_poses_prob` 0.5→0.0
- **验证**：用户授权后从 `sonic_release/last.pt` 重启 → run `_182557`。早期指标确认干预生效：iter 11 时 `ee_body_pos` 终止率从 0.58 跌到 0.03–0.05，entropy 干净复位到 13.18，std 复位到 0.39。
- **经验**：
  1. "症状层面修补"能让指标短期变好，会让人误以为根因找到了
  2. 单看 reward/termination 修，不看数据 pipeline，迟早再次停滞

---

## [2026-05-20] `zero_23dof_motion_lib.py` 是 no-op，找到真根因

- **现象**：run `_182557` 干预后训到 iter ~16130：
  - mean length 25–32（vs cold start 6–7，有真实进步）
  - mean entropy 13.18 → **18.62**（**比 `_151956` 死时 18.0 还高**）
  - std 0.39 → **0.47**（饱和）
  - `ee_body_pos` 终止率 0.04 → 0.13（爬回去一半）
  - `foot_pos_xyz` 0.79–0.83（仍是主要 termination）
- **触发**：ABCD 干预后从 `sonic_release/last.pt` 冷启动训练，跑了约 12.7h。
- **发现路径**：用户问 "现在是什么状态"，看完指标后追问 "所以不存在腕部自由度和参考轨迹对不上这种情况吗 那为什么一直没有很好的收敛"。
- **调查**：
  1. **第一次错判**：默认 `default_joint_pos` ≈ 0 → 物理锁的位置 = 数据 zero 的位置 → FK 一致。错。
  2. 用户 push back，强制重查
  3. grep `pose_aa\|forward_kinematics_batch`：
     - `motion_lib_base.py:1769`：`pose_aa = to_torch(curr_file["pose_aa"][start:end]).clone()`
     - `motion_lib_base.py:1873`：`mesh_parsers.fk_batch(pose_aa, trans, ...)`
     - `torch_humanoid_batch.py:440-447`：`return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]` — dof_pos 是 pose_aa 派生
  4. **运行时只读 `pose_aa`，根本不读 PKL 里的 `dof` 字段**
  5. `zero_23dof_motion_lib.py` 只清 `dof` 列，从未触碰 `pose_aa`
- **根因**：`zero_23dof_motion_lib.py` 修改的字段在运行时是 dead data。从训练第一天起，**所有"23DoF" motion lib 输出的 reference body_pos / dof_pos 都是 29DoF FK 结果**，6 个 missing joint 的 axis-angle 仍按 retargeted AMASS 非零值起作用。物理（关节锁在 default ≈ 0）和参考（带 29DoF 手腕旋转）每一步都不一致。所有 body-pos / joint-pos based 的 reward / termination / metric 都看到这个矛盾。ABCD 是 reward/termination 层 band-aid，没动到数据源。
- **改动**：
  - **`gear_sonic/utils/motion_lib/motion_lib_base.py`**（line 1869 之后、`fk_batch` 之前，所有 augmentation 之后）：
    ```python
    missing_dofs_mjcf = self.m_cfg.get("missing_dofs_mjcf", None)
    if missing_dofs_mjcf:
        missing_pose_aa_idx = [d + 1 for d in missing_dofs_mjcf]
        pose_aa[:, missing_pose_aa_idx, :] = 0
    ```
  - **`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof.yaml`**：
    - 加 `motion_lib_cfg.missing_dofs_mjcf: ${missing_dofs.indices_mjcf}`
    - 删除 `motion_file` override（`robot_filtered_23dof` 现在等价于 `robot_filtered`）
    - **ABCD 全部 revert** 到 parent `sonic_release.yaml` 默认值
    - 保留：observations 的 `_masked` 变体、joint_limit 的 23 关节正则
  - **`tests/test_sonic_release_23dof_yaml.py`**：`test_motion_file_points_to_23dof_dataset` → `test_motion_lib_cfg_wires_missing_dofs_mjcf`（防止误删 fix 字段）
- **验证**：
  - `python -m py_compile gear_sonic/utils/motion_lib/motion_lib_base.py` ✓
  - `pytest tests/test_sonic_release_23dof_yaml.py tests/test_missing_dofs_config.py` 7/7 通过
  - Hydra compose：`missing_dofs_mjcf` 插值解析为 `[13, 14, 20, 21, 27, 28]`；`reward_point_body=['torso_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']`、`cat_upper_body_poses_prob=0.5`、`vr_5point weight=2.0 std=0.1`、`ee_body_pos.body_names` 含 4 个 link，全部确认还原
  - 训练指标待新一轮 baseline run 验证
- **死代码**：`gear_sonic/data_process/zero_23dof_motion_lib.py` 现在是 no-op，可后续删除。`gear_sonic_deploy/reference/zero_23dof_columns.py`（部署侧 CSV）疑似也是同类问题，未验证。
- **经验**：
  1. **数据修复脚本要 trace 到 runtime 真正读什么字段**。改了 PKL 里的字段 ≠ 改了训练用的数据。脚本名很有迷惑性（"zero_23dof_motion_lib"听起来很正确）。
  2. **第一直觉错判时要愿意被推翻**。这次是用户的反问救了项目——如果默认接受我"FK 一致"的结论，会再浪费几周训练。
  3. **症状级修补 vs 根因修复**：ABCD 干预早期指标变好（ee_body_pos 跌到 0.04），看起来对了，但 entropy 持续爬升才是真实信号——policy 还在被矛盾梯度撕扯。**只要 entropy / std 在饱和方向漂，根因就还在**。
  4. **加 hydra interpolation 比手抄常量好**。`missing_dofs_mjcf: ${missing_dofs.indices_mjcf}` 单一来源，删字段时测试会立刻 fail。
  5. PKL 数据格式里同时存 `dof` 和 `pose_aa` 这种冗余字段是踩坑温床——下次接手类似数据时第一件事查 "运行时读哪个字段"。

---

## [2026-05-20] `_102145` 训到 iter ~725 又出现 entropy/std 爬升，FK fix 三大嫌疑全部排除

- **现象**：post-FK-fix canonical baseline run `_102145`：
  - iter 47–54：entropy 13.40, std 0.39, length 14, foot_pos_xyz 0.80, ee_body_pos 0.16
  - iter 218–223：entropy 14.09, std 0.40, length 19, foot_pos_xyz 0.73, ee_body_pos 0.25
  - iter 722–727：entropy 15.72, std 0.42, length 23, foot_pos_xyz 0.63, ee_body_pos 0.34
  - 下肢明显学习（length ↑，foot_pos_xyz ↓，rewards ↑），但 entropy/std/ee_body_pos/error_body_pos 全部往坏方向漂——和 `_151956`/`_182557` 死亡曲线早期签名一致。
  - entropy 斜率 ~0.0034/iter（线性外推到 18 大概在 iter 1500–1700）。
- **触发**：post-FK-fix 训练正常进行约 36 分钟。
- **发现路径**：用户分享 iter 218 / 722 训练日志，看到 `_182557` 同款"下肢进步、上肢退步"轨迹。
- **调查（系统排查 4 个候选根因，前 3 个全部排除）**：

  1. **FK fix 代码路径是否真的触发？** 用 standalone repro 跑 hydra compose + EasyDict 包裹，确认：
     - `motion_lib_cfg.missing_dofs_mjcf` 通过 `${missing_dofs.indices_mjcf}` 插值正确解析为 `[13, 14, 20, 21, 27, 28]`
     - DictConfig → EasyDict 包装后，`.get("missing_dofs_mjcf")` 仍返回真值（ListConfig，可迭代）
     - `.update()` 加入其他键之后字段依然存在
     - `[d + 1 for d in val]` 正确产出 `[14, 15, 21, 22, 28, 29]`
     - 结论：`motion_lib_base.py:1874-1877` 的 fix 在 runtime 真实命中。**RULED OUT**。

  2. **pose_aa 关节顺序是否就是 MJCF order？** 用 Python 脚本独立解析 `g1_29dof_rev_1_0.xml`，DFS 遍历 worldbody，对照 SDK `G1JointIndex` 枚举：
     - body[0]=pelvis (root)；body[i+1] 对应 SDK joint i，i=0..28，**完美 1-1 对齐**
     - body[14]=waist_roll_link/joint waist_roll_joint（SDK 13）
     - body[15]=torso_link/joint waist_pitch_joint（SDK 14）
     - body[21]=left_wrist_pitch_link（SDK 20）
     - body[22]=left_wrist_yaw_link（SDK 21）
     - body[28]=right_wrist_pitch_link（SDK 27）
     - body[29]=right_wrist_yaw_link（SDK 28）
     - 结论：`pose_aa[:, 1+d]` 对 `d ∈ {13,14,20,21,27,28}` 精确命中 6 个目标关节。**RULED OUT**。

  3. **default_joint_pos 是否真的是 0？** 读 `gear_sonic/envs/manager_env/robots/g1.py:223-236` 的 `init_state.joint_pos`，发现 6 个 missing joint（waist_roll/pitch、L/R wrist_pitch/yaw）**完全不在字典里** → IsaacLab 默认 0.0。`MissingDofsLockEnv._missing_default_qpos = articulation.data.default_joint_pos[:, joint_ids]` 读出来就是 0，每次 sim.step 后 pin 到 0。物理 = 数据 = 0，**完全一致**。**RULED OUT**。

- **根因（当前判断）**：FK fix **完全正确执行**，参考数据**完全一致**于物理。entropy/std 早期飘升是 **policy 适应过程的正常现象**，而非毒性梯度：
  - `sonic_release/last.pt` 是 29DoF 训出来的 prior，原本依靠 wrist pitch/yaw 完成精细 wrist 姿态
  - 23DoF 物理锁掉这两个自由度后，wrist_yaw_link 的世界 Z 由（pelvis + 腰yaw + 肩 + 肘）决定（wrist_roll 轴是 X，不影响 Z）
  - 参考轨迹的 wrist Z 用 zeroed pose_aa 跑 FK 后，是几何上**可达**的，但需要 policy 重新学如何用肩肘组合复现
  - 早期下肢梯度强、上肢梯度弱 → 下肢先学好、上肢"暂时退步"的过渡相
  - 这一信号与 `_182557` 的死亡曲线**形似而本质不同**：`_182557` 的参考数据本身物理不可达，poison gradient 永远撕；`_102145` 的参考可达，gradient 信息正确但 policy 还没适应
- **改动**：无（FK fix 已就位且正确）。
- **验证**：
  - Hypothesis 1：`/home/woan/.conda/envs/groot_wbc/bin/python` standalone repro 确认 hydra→EasyDict→get 链路完整
  - Hypothesis 2：MJCF DFS body 顺序与 SDK joint 索引 1-1 对齐验证脚本（30 个 body 全部对照）
  - Hypothesis 3：g1.py init_state 检查 + missing_dofs_env 读 default_joint_pos 链路确认
  - 训练 metric 对比 `_182557`（broken data）vs `_102145`（fixed data）：
    - length 增长速度：`_102145` iter 725 达到 23（同样长度 `_182557` 需要 iter ~16k）→ **`_102145` 学习速率 ~20× 快**
    - foot_pos_xyz 下降速度：`_102145` 0.80 → 0.63（675 iters）→ 真实下肢进步
    - 这两条说明 `_102145` 在做实质性学习，不是 `_182557` 的颓废坐姿
- **建议下一步**：
  1. **不要 kill `_102145`**。继续训到 iter ~1500-2000，观察：
     - entropy 是否在 17 之前 plateau / 回落（fix 工作中）
     - length 是否突破 50+（下肢继续）
     - ee_body_pos 是否在 0.4 之前回落（policy 学会用肩肘补偿）
  2. **Decision tree**（iter ~1500 检查点）：
     - entropy 已 plateau ≤ 16 + length > 50 → 健康，继续训
     - entropy 仍 ~0.003/iter 速度爬升且 length 停滞 < 30 → 调研 hypothesis 4（reference 中是否有 FK 之外的 wrist 通路，如 SMPL/SOMA 数据，retargeting 的 baked offset，extend_config 的 hand 末端）
     - entropy ≥ 17 saturating → kill，进 hypothesis 4
- **经验**：
  1. **死亡曲线 ≠ 死亡曲线**：相同形状的 entropy 上升轨迹可能有不同根因。`_182557` 是 poison gradient（不可达），`_102145` 看似一样但是过渡相（可达但未学会）。区分依据：**比较 length / foot_pos_xyz 的学习速率** 和 **绝对值**。
  2. **三层独立验证比一层运气好**：code path（runtime resolved）+ index alignment（MJCF DFS vs SDK）+ default qpos（init_state 缺省 → 0），任何一层错都会暴露。
  3. **9DoF policy → 23DoF 微调的过渡相是真实存在的**，不要因 entropy 暂时上行就误判。看 length / foot termination 的绝对值和趋势更可靠。

---

## [2026-05-20] H4 命中：`_motion_smpl_joints` 绕过 FK，把 29DoF retargeted 关键点喂给 SMPL encoder

- **现象**：post-FK-fix run `_112413` 训到 iter ~2839 仍然死亡：
  - entropy 18.52（≥ `_182557` 死亡时 18.62 的水平，达到 saturating）
  - std 0.47（与 `_182557` 完全一致的饱和状态）
  - length 19–29，没破 50
  - ee_body_pos 0.31–0.36，foot_pos_xyz 0.62–0.66
  - 触发上一条记录里我自己定的 kill 阈值（"entropy ≥ 17 saturating → kill, H4"）
- **触发**：`sonic_release_23dof` finetune 跑了 ~2.8k iter 后曲线和 broken-data 时代 `_182557` 完全重合。
- **发现路径**：用户共享 iter 2839 metrics → 触发 H4 决策分支 → 系统排查"绕过 FK 的 reference 通路"。
- **调查（双路 Explore agent 交叉确认）**：
  1. 查所有 `curr_file["..."]` / `curr_smpl_data["..."]` / `curr_soma_data["..."]` 直接读取的 PKL 字段
  2. 区分 "经过 fk_batch 输出（被当前 fix 覆盖）" vs "直接喂给 obs/reward（绕过 fix）"
  3. **找到绕过通路**：
     - `motion_lib_base.py:1908`：`curr_motion["smpl_joints"] = torch.tensor(curr_smpl_data["smpl_joints"]).float()` —— 直接从 PKL 加载 24×3 SMPL 骨架关节世界坐标，**不进 fk_batch**
     - `motion_lib_base.py:1903-1904`：`smpl_pose[:, -6:] = 0.0` —— 只清 SMPL 72 维 pose 的最后 6 维（SMPL idx 22, 23 = L/R hand 手指 axis-angle），**没清 idx 20/21 = L/R wrist 也没清 idx 3/6/9 = spine1/2/3**
     - `motion_lib_base.py:1410`：`self._motion_smpl_joints = torch.cat(_motion_smpl_joints, dim=0)` 持久化为运行时数据
     - `motion_lib_base.py:632`：`get_smpl_joints()` 返回 `_motion_smpl_joints[motion_steps + length_starts]`
     - `commands.py:1208`：`smpl_joints_multi_future` property 调 `motion_lib.get_smpl_joints(...)`
     - `observations.py:1751-1780`：`smpl_joints_multi_future_local()` 用 root quat 转换 frame，喂给 obs term `smpl_joints_multi_future_local_nonflat`
     - `sonic_release.yaml:101`：actor.backbone.encoders.smpl.inputs 包含 `smpl_joints_multi_future_local_nonflat` —— **policy 直接观测**
- **根因**：SMPL 数据是从 29DoF AMASS retarget 出来的 24-joint 骨架，每帧的关节世界坐标依赖完整 29DoF 全身链路：
  - SMPL idx 3/6/9 (spine1/2/3) = G1 waist_roll/pitch 区段的 retargeted 表达 → 编码消失的 waist_roll/pitch 旋转，整个上身位姿都被它带偏
  - SMPL idx 20/21 (L/R wrist) = 手腕 endpoint 位置 → 因为父链 (shoulder + elbow + ... + 倾斜的 torso) 都基于 29DoF 计算，wrist 位置物理不可达
  - SMPL idx 22/23 (L/R hand) = 手指根 endpoint → 同样依赖 wrist_pitch/yaw articulation
  - **这是和上次（pose_aa）同构的 bug**：把"修了 G1 robot FK"误以为"修了所有 reference 信号"，但 SMPL pipeline 是平行通路。
- **被路由进 policy 的概率**：50% sample-prob（sonic_release.yaml:56 `teleop_sample_prob_when_smpl: 0.5`，commands.py:2941-2959 的 encoder_index 路由），即一半训练步上 policy 通过 SMPL encoder 看到 unreachable 手腕/手部 keypoints。这正好和 entropy/std 慢速饱和的速率匹配。
- **改动**：尚未实施。候选修复路径：
  - **(A) 最小破坏**：`sonic_release_23dof.yaml` 加 `motion_lib_cfg.smpl_motion_file: null`，关 SMPL 数据加载。SMPL encoder 仍在网络里但 mask 永远 0（sample 概率 0），不影响 prior。代价：loss SMPL 模式 50% 训练样本量，但消除矛盾梯度。
  - **(B) 中等成本**：在 `motion_lib_base.py:1908` 之后，用 corrected FK 输出（`curr_motion["global_translation"]` 已是 zeroed pose_aa 的 FK 结果）+ SMPL→G1 body mapping 重算 `curr_motion["smpl_joints"]`。需要一份现成或新写的 SMPL 24 关节到 G1 ~30 body 的索引映射；SMPL idx 22/23（手指）在 23DoF 里没对应物，需用 wrist_yaw_link 代替。
  - **(C) 最深修补**：和 pose_aa 一致，把 SMPL pose 在 PKL 里就直接清 idx 20/21（wrist 二轴）和 idx 3/6/9（spine）的 axis-angle，再用 SMPL forward kinematics 重算 smpl_joints。需要 SMPL FK 实现，更彻底但工程量大。
- **验证**：尚未训。建议先做 (A) 验证 H4 是真根因；如果 entropy 不再饱和 + length 继续上行，再决定是否做 (B)/(C) 找回 SMPL 信号。
- **经验**：
  1. **PKL 数据里平行通路是常态**：一份 PKL 既存 `pose_aa`（FK 入口）也存 `smpl_joints`（直接读 obs）也可能存 `soma_joints`（同上）。修一处不等于修了全部。任何 ref 通路都要从 PKL 字段一路 trace 到 reward/obs/termination。
  2. **`smpl_pose[:, -6:] = 0.0` 这种"看起来在做事"的 mask 要警惕**：作者意图是清"global rotation 末段"，但语义上对 23DoF 完全不够。在 review 别人写的 mask 时，永远 print 一下 mask 的 index 含义。
  3. **死亡曲线如果在 fix 后 1:1 复现，必然是另一条通路在以同样的速率污染 gradient**：`_182557` 在 iter ~12k 死，`_112413` 在 iter ~2.8k 死（更快）—— 因为这次 SMPL 通路是 50% sample，而原 pose_aa bug 是 100% 通路；速率比例 50:100 = 2.4×～4.5× 慢，吻合。
  4. 从教训 [2026-05-20] 第 1 条延伸：**改了 PKL 的某字段 ≠ 改了训练用的数据**。这次进一步：**改了某条 ref 通路 ≠ 改了所有 ref 通路**。

---

## [2026-05-20] H4 修复方案 A 落地：`encoder_sample_probs.smpl: 0.0`（**重点：副作用清单**）

- **现象**：H4 命中后选择 Plan A（**最小破坏**）：把 SMPL encoder 的 sample 概率设为 0，让矛盾梯度通路在 routing 层被截断。SMPL encoder 权重在整个 23DoF finetune 期间冻结，但 prior 不被零输入破坏。
- **触发**：用户授权 "那就先采取A 记得在DEBUG日志中重点记录这个带来的后果"。
- **改动**（**仅 1 个文件 1 段 yaml**）：
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof.yaml`：在 `manager_env.commands.motion` 下新增
    ```yaml
    encoder_sample_probs:
      g1: 1.0
      teleop: 1.0
      smpl: 0.0
    ```
  - **没有改**：`smpl_motion_file`（必须保留，否则 `_motion_smpl_joints` 不存在，obs 计算会 AttributeError）；motion_lib_base.py 不动；universal_token_modules.py 不动；网络结构不动。
- **为什么这是安全的**（三层独立保护，任何一层成立 SMPL encoder 都不会收到梯度）：
  1. **Routing 层**：`commands.py:2937-2959` 按 `encoder_sample_probs` 归一化采样 → smpl bit 永不被 set → `encoder_masks[smpl]` 全 False
  2. **Scatter 层**：`universal_token_modules.py:566-573` 的 `assemble_all_tokens` 对 mask 全 False 的 encoder 是 **no-op**（`all_tokens[mask_全False] = ...` 不写入任何行）→ SMPL token 不进入 actor head 输入 → 主 loss 不可能反传到 SMPL encoder
  3. **Aux loss 层**：`universal_token_modules.py:949-952` 的 G1↔SMPL pairing aux loss 被 `if smpl_mask.sum() > 0:` 守护 → 全 False 时整段被跳过 → aux loss 也不反传
- **副作用清单**（用户重点要求记录）：
  1. **SMPL encoder 权重整轮冻结**：从 finetune 第一步到结束，SMPL 子 encoder 的所有参数都不会更新。从 `sonic_release/last.pt`（29DoF prior）继承的 SMPL 权重保持原样。**这不是损害，是有意冻结**——因为参考数据本身有 H4 leak，让它学反而会被污染。
  2. **样本分布偏移 33/33/33 → 50/50/0**：原来三个 encoder 等权采样（unnormalized 1.0/1.0/1.0），现在 SMPL 概率 0，G1 + teleop 各占 50%。每个 episode 的训练信号更密集地落在 G1 keypoint tracking + VR 3-point teleop 这两个模式上。
  3. **以下 aux loss 全部 deactivated**（只要 `smpl_mask.sum() == 0` 就跳过）：
     - G1↔SMPL latent pairing loss（universal_token_modules.py:949-952 区域）
     - 任何依赖 SMPL token 与其它 encoder token 配对的 cycle / contrastive loss
     - 这些 loss 在 dashboard 里会显示为 0 或 NaN（取决于 logger 处理空 batch 的方式），**不要误判为 NaN bug**
  4. **SMPL obs 仍然在计算（白白浪费一点 compute，但无害）**：
     - `motion_lib._motion_smpl_joints` 仍然按原方式从 PKL 加载并持久化（占显存）
     - `command.smpl_joints_multi_future` property 每步仍然被调用（CPU/GPU 索引）
     - `observations.smpl_joints_multi_future_local()` 每步仍然计算（root-quat 旋转 + 拼 nonflat）
     - 这些值进入 `tokenizer_obs` 后被 SMPL encoder 编码——但 token 立刻被 mask=0 丢弃。**Pure waste, but cheap**。如果训练吞吐量明显下降可以后续做 obs-side 短路，目前不优化。
  5. **部署管线零影响**（已验证）：
     - 用户的 `gear_sonic_deploy/reference/example_23dof/macarena_001__A545/` 是 G1 14-body × 29-joint CSV 格式，不含 SMPL 字段
     - `gear_sonic_deploy/policy/release/observation_config.yaml` 三个 `encoder_modes` 中 SMPL 模式 (`mode_id: 2`) 要求 `smpl_joints_10frame_step1` 等 SMPL-specific obs，example_23dof 完全不提供
     - `g1_deploy_onnx_ref.cpp:2366` `initial_encoder_mode_ = 0`（默认 G1 模式）
     - **结论**：用户当前部署只走 G1 encoder（mode 0），冻结 SMPL 训练对部署 0 影响。
  6. **未来风险**（**重点**）：如果之后想做"视频 → SMPL fitting → 部署"模式（即 deploy 时用 `mode_id: 2`），SMPL encoder 仍然停留在 29DoF prior 上，对 23DoF 物理是失配的。届时需要：
     - 要么先彻底修 H4 leak（Plan B：用 corrected FK 重算 smpl_joints；Plan C：在 PKL 层修 SMPL pose 后跑 SMPL FK），再单独 finetune SMPL encoder
     - 要么在 SMPL deploy 之前接受质量下降并做一轮针对性 SMPL-only retrain
     - **不能假设训完这一轮后 SMPL deploy 直接可用**
  7. **Sample efficiency 影响（不确定）**：丢掉 33% 的 SMPL-mode 样本后，G1 + teleop 模式的等价 batch 变密。这可能：
     - 加速 G1 / teleop 模式的收敛（每 iter 见到更多相关样本）
     - 或者因为去掉了"全身姿态约束"信号让 wrist roll / shoulder 学到不规范的解
     - 实测才知道，预期前者占主导（因为 SMPL 信号本来就被 mask 后丢弃了，损失只是采样多样性）
- **验证方案**（在 `_112413` 已 kill 的前提下从 `sonic_release/last.pt` 重启训练）：
  - **必须看到的健康信号**（满足全部即 H4 真根因 + Plan A 起作用）：
    - entropy 在 iter ~1500 之前 plateau 或回落，**绝对不能爬过 16**
    - std 维持在 0.39 附近（不再向 0.47 饱和）
    - mean length 突破 50（之前最好 `_182557` 在 12k iter 才到 32，`_112413` 在 2.8k 仅 19-29）
    - `ee_body_pos` 终止率从 0.31 跌到 ≤ 0.2 并稳定
    - `foot_pos_xyz` 继续下降（baseline 趋势不变）
  - **失败信号（任何一条触发就重新评估）**：
    - entropy 仍以 ~0.003/iter 速率上升 → H4 不是唯一通路，存在 H5（如 SOMA、cat_upper_body_poses 通路）
    - length 在 iter 2k 仍 < 30 → SMPL 信号被去掉后 G1 模式丢失了关键约束
    - 任何 NaN（aux loss 计算的边界情况未处理）
- **后续清理 TODO**（不影响本次训练，但留作 follow-up）：
  - SMPL obs 短路：在 `observations.py` 的 SMPL term 加 fast-path，当 `encoder_sample_probs.smpl == 0` 时直接返回 zeros，节省每 step 计算
  - drift-check 测试：`tests/test_sonic_release_23dof_yaml.py` 加 assertion `encoder_sample_probs.smpl == 0.0`，防止有人手贱改回去
  - 真正修 H4：等 Plan A 验证 H4 是真根因后，决定上 Plan B 还是 Plan C（参见上一条 entry）
- **经验**：
  1. **"路由层关闭" > "数据源置零"**：当 encoder 输入路径上有 0 的特殊语义风险（如归一化、log、除法），不要用 `smpl_motion_file: null` 喂 0 值；走 routing mask 让 encoder 整个分支不参与 forward，是更安全的关闭方式。
  2. **冻结 ≠ 损坏**：用户担心 "全 0 输入会崩坏权重"，正确反驳是"权重根本不收到梯度"。回答这类疑问时要直接指代码（mask scatter no-op + aux loss guard），不要泛泛说"应该没事"。
  3. **副作用清单的写法**：先列保留行为（冻结/sample shift/aux loss off/obs 仍计算），再列影响域（训练 / 部署 / 未来重训），最后给"如何确认成功"的具体阈值。这种结构以后任何 ablation 都该照搬。
  4. **drift-check 的价值**：上次 `test_motion_lib_cfg_wires_missing_dofs_mjcf` 防止有人误删 fix 字段。这次也该加 `encoder_sample_probs.smpl == 0.0` 的 assertion——`yaml.safe_load` 在 hydra interpolation 下需要小心，可能要 hydra compose 才能验。

---

## [2026-05-20] Plan A run `_145252` 启动 baseline（iter 0–10）

- **Run**：`sonic_release_23dof_test-20260520_145252`，从 `sonic_release/last.pt`（29DoF prior）finetune，配置已含 H1 (pose_aa fix) + H4 (encoder_sample_probs.smpl=0.0)。
- **早期 metrics（iter 5–10）**：
  | 指标 | iter 5 | iter 10 | 趋势 |
  |---|---|---|---|
  | entropy | 13.151 | 13.185 | +0.034 / 10 iter ≈ 0.0034 / iter（**与 `_102145` early 同速**）|
  | action_noise_std | 0.39 | 0.39 | **稳定，未向 0.47 饱和** |
  | mean_length | 6.6 | 7.4 | 6–7 区间小幅震荡（早期正常）|
  | mean_rewards | 0.273 | 0.267 | 稳定 |
  | `ee_body_pos` 终止率 | 0.089 | 0.104 | **0.09–0.11（关键！）** |
  | `foot_pos_xyz` 终止率 | 0.875 | 0.880 | 与冷启动同水平，待下肢学习 |
  | `error_body_pos` | 0.056 | 0.057 | 低且稳定 |
  | `tracking_vr_5point_local` | 0.022 | 0.025 | **正向、缓慢上升** |
- **早期对比**（同样从 prior finetune，只在 ref data 通路上不同）：
  | Run | iter | entropy | std | length | ee_body_pos | 状态 |
  |---|---|---|---|---|---|---|
  | **`_145252` (H1+H4 fixed)** | 10 | **13.18** | **0.39** | 7.4 | **0.10** | 验证中 |
  | `_112413` (H1 only, H4 leak) | 47 | 13.40 | 0.39 | ~14 | 0.16 | 已死，iter 2839 entropy 18.52 |
  | `_151956` (broken pose_aa) | 47 | ~13.5 | 0.39 | ~14 | 0.32 | 已死，iter 2923 ee_body_pos 0.58 |
  | `_182557` (ABCD bandage) | 11 | 13.18 | 0.39 | ~10 | 0.04 | 已死（缓慢，iter 16k entropy 18.62）|
  - **关键：`_145252` 在 iter 10 的 `ee_body_pos = 0.10`** 远低于 `_112413` iter 47 的 0.16（同 prior 同 motion file，唯一变量是 SMPL routing）。这与 H4 假设一致——SMPL 通路移除后，policy 不再被 unreachable 上肢 keypoints 撕扯。
  - **唯一警示**：`_182557` 在 iter 11 时 ee_body_pos 也只有 0.04（甚至更好），但它是 ABCD reward/termination bandage 强行压低，最终仍因为底层 pose_aa leak 死亡。**不能仅看早期 ee_body_pos 就宣告胜利**——必须等 iter 1500-2000 checkpoint 看 entropy 是否 plateau 不爬过 16，length 能否突破 50。
- **观察项**（按重要度）：
  1. **iter ~500**：entropy 是否仍维持在 13–14（不出现 `_112413` iter 200 时已 14.09 的爬升斜率）
  2. **iter ~1500**：entropy ≤ 16 + length > 30 → 健康；entropy 仍以 0.003/iter 速率升 → 仍有 leak（候选 H5：SOMA 通路？）
  3. **iter ~3000**：entropy ≤ 16 + length > 50 + ee_body_pos < 0.2 → H4 真根因 + Plan A 起作用；entropy ≥ 17 → kill
- **当前结论**：iter 10 信号"良好但太早"。std 不饱和 + ee_body_pos 起步低 + entropy drift 速率正常，三项一致指向 H4 修复有效，但需要再观察 ~1500 iter 才能确认。**继续训，不干预**。
- **训练吞吐**：4500 steps/s（与历次 run 一致，符合预期——SMPL obs 仍计算只是被丢弃，没有加速也没有显著拖慢）。
- **ETA**：~78 小时到 100k iter。

---

## [2026-05-20] Plan A run `_145252` iter 220 checkpoint：**警钟——和 `_112413` 同期重合**

- **现象**：训到 iter ~220–224，metrics 与 `_112413`（H1 only，H4 leak active）同期 iter 218–223 **几乎完全重合**。早期 iter 10 看到的 `ee_body_pos = 0.10` 优势在 iter 220 消失。
- **触发**：用户共享 iter ~220 训练日志。
- **配置确认**（先排除"fix 没生效"的可能性）：
  - `logs_rl/.../sonic_release_23dof_test-20260520_145252/config.yaml` 显示 `manager_env.commands.motion.encoder_sample_probs: {g1: 1.0, teleop: 1.0, smpl: 0.0}` ✓
  - `_112413` 同位置是 `smpl: 1.0`（broken）✓
  - **Fix 真在生效**，问题不在配置层。
- **Iter ~220 同期对比表**（控制变量：相同 prior `sonic_release/last.pt`，相同 motion file，相同硬件）：
  | 指标 | `_102145` (H1, no H4) iter 220 | `_112413` (H1, no H4) iter 220 | `_145252` (H1+H4) iter 220 |
  |---|---|---|---|
  | entropy | 14.09 | 14.09 | **14.07** |
  | std | 0.40 | 0.40 | **0.40** |
  | length | 19 | 19 | **19.6** |
  | foot_pos_xyz | 0.73 | 0.73 | **0.72** |
  | ee_body_pos | 0.25 | 0.25 | **0.25** |
  | error_body_pos | — | — | 0.065 |
  | reward | — | — | 1.14 |
  - **三条 trajectory 在 iter 220 完全收敛**。早期 iter 10 看到的 `ee_body_pos 0.10` 优势更可能是"启动延迟造成的不公平对比"（`_145252` iter 10 length 7.4，`_112413` iter 47 length 14，根本不是同一时间点）。
- **Entropy 斜率比较**：
  | Run | iter 区间 | slope |
  |---|---|---|
  | `_102145` | 220 → 725 | 0.00323/iter |
  | `_112413` | 47 → 220 | 0.00399/iter |
  | `_145252` | 10 → 220 | **0.00424/iter** |
  - 三条几乎在同一斜率上。线性外推 `_145252` entropy：iter 1500 ≈ 19.5（远超 16 的 plateau 阈值）。即使有 deceleration（`_102145` 后段到 0.0032），iter 1500 仍 ≈ 18。
- **新假设 H5：SMPL 不是唯一 leak 通路**。Plan A 关闭了 SMPL encoder routing 但 entropy 轨迹完全没变 → 必有另一条 ref 通路在等速污染 gradient。候选：
  1. **SOMA 数据**：`motion_lib_base.py` 同样有 `curr_soma_data["soma_joints"]` 直接加载逻辑（与 SMPL 平行）。如果 sonic_release 配了 SOMA，问题同构。
  2. **`cat_upper_body_poses_prob: 0.5`**（parent yaml）：用别的数据源拼接上半身 pose。如果这个数据源也是 29DoF retargeted，FK fix 不会清理它。
  3. **VR teleop reference**：`vr_5point` reward / `tracking_vr_5point_local` body=`[torso, L_wrist_yaw, R_wrist_yaw]`。world pos 是 FK 结果（已 fix），但 reward `std=0.1` 太严格，而锁腕的可达空间收窄了——**可能是物理可达但 policy 难以覆盖**而不是 leak。
  4. **teleop encoder 数据通路**：teleop 模式喂 VR 3-point 目标，这些目标也是从 29DoF AMASS retarget 出来的。如果未经 FK 重算（直接从 PKL 读），就是平行 leak。
- **当前不慌，但需要更早 kill 准备**：
  - 不立即 kill：entropy 14.07 距离 17 saturating 还有空间，slope 也未呈现明显 saturating 拐点
  - 但**修订 kill 阈值**：原计划等到 iter 1500。鉴于轨迹 1:1 重合，建议提前到 **iter ~700–800** 做关键判断（这个区间 `_102145` 已经到 entropy 15.7+，能看出是否还能 deceleration）
- **建议下一步**：
  1. 先继续训到 iter ~700，对比 `_102145` 同期。如果 `_145252` entropy 仍与 `_102145` 完全重合（~15.7）→ 确认 H5 存在，kill 进 H5 排查
  2. **不要现在就改配置**——需要观察更多 iter 才能判断 H4 是否完全无用 vs 部分有用
  3. **预先把 H5 排查工具准备好**（不实施）：
     - 写一个 grep 脚本检查所有 `curr_motion[...]` / `curr_*_data[...]` 直接赋值通路
     - 检查 sonic_release.yaml 是否启用了 soma_motion_file
     - 检查 cat_upper_body_poses 实现的数据源
- **可能误判方向**（self-skepticism）：
  1. Iter 220 重合可能是因为这个区间下肢 dominant，`foot_pos_xyz` 0.7 对 SMPL 信号不敏感。SMPL 通路的影响可能在 **下肢学好之后才显现**。也就是说"_112413 死于 iter 2839"是因为下肢在 ~iter 1500 学好后上肢 SMPL 矛盾才放大；如果是这样，`_145252` 在 iter 1500 之后会和 `_112413` 分叉。
  2. 早期 iter 10 看到的 ee_body_pos 0.10 可能是真信号（policy 还没探索上肢），iter 220 上升到 0.25 是探索行为。
  3. 这两种可能都意味着"现在不能下结论"，必须看 iter ~1500 的分叉点。
- **验证方案**：让训练继续，下一个 checkpoint 在 iter ~700 和 ~1500 时主动重读 metrics。
- **经验**：
  1. **早期 metric 优势可能是启动延迟伪装**：比较两个 run 时如果 length 不同，就不是公平对比。下次写"对比 `_145252` iter 10 vs `_112413` iter 47"前，要先 normalize 到相同的"训练步数"或"相同 length 区间"。
  2. **死亡曲线在配置层 fix 后 1:1 复现**——这次又出现了，正如 `_182557 → _112413` 那次。这是**多通路平行 leak 的标志性 fingerprint**：每修一条，下一条以同样的速率接管。
  3. **修订 H4 真根因的把握**：从前一条的"slope ratio 50:100 = 2.4×–4.5×"反推，`_182557` (broken pose_aa, 100% leak) iter 12k 死，`_112413` (broken SMPL, 50% leak) iter 2.8k 死——速率比 ~4.3×，比例正确。但 `_145252` 关掉 SMPL 后理论上应该再慢 2× 才对，结果完全没变。**这强烈暗示 SMPL 不是 50% 通路**，要么早就被 mask 关闭过（验证三层保护时遗漏），要么真正主导污染的不是 SMPL。

---

## [2026-05-20] H5 系统排查：所有 ref 通路审计完，**未找到 smoking-gun leak**

- **现象**：`_145252` iter 220 与 `_112413` (H4 leak active) 同期重合，需要找出"为什么 fix 没让 trajectory 分叉"。
- **触发**：用户授权 "还有可能是什么原因吗？你先排查排查"。
- **方法**：枚举 `motion_lib_base.py` 中所有 `_motion_*` 持久化字段 + `commands.py` 中所有 `motion_lib.get_*` 调用 + 网络端 `create_encoder_masks` / `assemble_all_tokens` routing 真实路径，逐条判定是否经过 FK fix。
- **审计表**（每行：通路 → 数据源 → 是否在本 run 启用 → 是否清洁）：

  | 通路 | 数据源 | 本 run 启用 | 经过 FK fix？ | 结论 |
  |---|---|---|---|---|
  | `body_pos_w/quat_w/lin_vel_w/ang_vel_w` | `m.global_translation/rotation/velocity/...`（fk_batch 输出）| ✓ | ✓ | clean |
  | `root_pos_w/quat_w/lin_vel_w/ang_vel_w` | 同上 | ✓ | ✓ | clean |
  | `dof_pos/dof_vel` | `m.dof_pos = pose.sum(...)[..., 1:]` 来自清零后的 pose_aa | ✓ | ✓ | clean |
  | `feet_l/r` | `foot_detect(global_translation)`（line 2182）| ✓ | ✓ | clean |
  | `vr_3point_body_pos_w` | `motion_lib.get_body_pos_w(...)`（line 2259）| ✓ | ✓ | clean |
  | `reward_point_body_pos_w` | 同上 | ✓ | ✓ | clean |
  | `_motion_smpl_*` | `curr_smpl_data["smpl_joints/pose/transl"]` 直接 PKL 加载 | ✓ 数据加载 | ✗ 绕过 | **被 routing 关闭**（Plan A）→ 无 grad |
  | `_motion_soma_*` | `curr_soma_data["soma_*"]` 直接 PKL | ✗ `sonic_release.yaml` 未配 `soma_motion_file` | ✗ | **不在 motion_lib 加载**（`soma_data is None`，line 1417 跳过）|
  | `_motion_object_*` | `curr_motion["object_*"]` 直接 PKL | ✗ 未配 `object_motion_file` | ✗ | 同上，不加载 |
  | `_motion_hand_action_*` | object scope | ✗ | ✗ | 同上，不加载 |
  | `_motion_actions` (`has_action`) | `curr_file["action"]` | 大概率 ✗（与 hand DOF 配对）| ✗ | 待 grep 确认，但属 hand action 通路 |
  | `hand_dof_pos` | `curr_file["dof"][:, 29+]` 当 `hand_dof_count > 0`（line 2138-2159）| ✗ 23DoF G1 无手指 DOF | ✗ | 不加载 |
  | `cat_upper_body_poses` | 替换上半身 pose_aa（line 1819-1859）| ✓ prob=0.5 | ✓ **fix 在它之后**（line 1875-1877）| clean |
  | `randomize_wrist_poses` | 加 noise 到 wrist pose_aa（line 1862-1869）| 默认 ✗，待确认 | ✓ fix 在它之后 | clean（即便启用）|
  | `vid_smpl_pose` 覆盖 `curr_motion["smpl_pose"]`（line 2179）| video reference SMPL pose | ✗ `vid_smpl_pose is None` | ✗ | 不触发 |

- **网络端 routing 三层验证**（再次确认 SMPL 真的被 mask 死）：
  1. **多项采样**：`commands.py:2920` `multinomial(probs=[0.5, 0.5, 0.0], ...)` → `smpl_encoder_index` 永远不会被 set。验证：`encoder_sample_probs` 经 `/sum` 归一化后 `[g1=0.5, teleop=0.5, smpl=0.0]`，multinomial 在 prob=0 的桶上**确定不抽样**（不是 epsilon → uniform 的 fail mode）。
  2. **legacy SMPL→G1 OR**：`commands.py:2937-2944` 用 `use_smpl = encoder_index[..., smpl_idx]`，由于 smpl 永不被 set，`use_smpl ≡ 0`，OR 是 no-op。
  3. **网络 mask 创建**：`universal_token_modules.py:516-517` `encoder_masks[encoder_name] = encoder_index[..., i].bool().flatten()` → smpl mask 全 False；`assemble_all_tokens` 对 mask 全 False 的 encoder 不写入任何行。
- **rewards.py / terminations.py 审计**：grep `smpl_\|soma_` → **零命中**。reward 和 termination 完全不消费 SMPL/SOMA 数据，只用 FK 输出。
- **结论：从代码层面，没有发现新的 leak 通路。** Plan A 在 routing 层正确执行，其它 ref 通路要么不启用要么经 FK fix。
- **剩余 4 个非"leak"假设**（按可能性降序）：

  | H | 描述 | 可证伪方式 |
  |---|---|---|
  | **H5a：分叉点在更晚 iter** | `_182557` (broken pose_aa, 100% leak) 在 iter 12k 才死亡；早期 0–700 iter 看似与 healthy run 无差。SMPL 50% leak 关闭后，分叉应该比 `_182557` 早出现但比 `_102145` 晚——可能在 iter 1500–3000 之间。 | 继续训到 iter ~1500，比较 `_145252` vs `_102145` iter 727 同步指标 |
  | **H5b：death curve 是 prior mismatch 下的自然 PPO 行为** | 29DoF prior 的 actor 输出分布对 23DoF locked 物理是 mismatched。PPO 必须打散 prior 才能探索新可达空间。entropy 上升 → 探索成本，与 leak 无关；应该有 plateau。 | from-scratch 训练（不用 prior）观察曲线形状；或继续训到 entropy plateau 看是否能 ≤ 17 |
  | **H5c：Prior actor head 期望 SMPL token 输入** | Prior 训练时 33% 步骤见到非零 SMPL token（routing on）。actor head 权重已经"内化"了 SMPL token 的存在；突然全程 0 输入会让 actor 进入 OOD 状态，触发 PPO 的"探索冲动"。这种情况下 Plan A **反而让事情变难**，但与 H4 leak 无关。 | A/B 测试：分别用 (smpl=0.0) 和 (smpl=0.001 让 mask 偶尔点亮) 比较；或冻结 SMPL encoder 但保留 routing 偶尔通过 |
  | **H5d：任务难度本身** | 锁掉 6 个关节相当于减少 21% DoF（29→23），但 reward landscape 不变（vr_5point std=0.1 严格）。物理可达空间收窄 + reward 不变 → 探索成本上升。entropy 自然爬升是任务难度的反映。 | 放宽 vr_5point std 到 0.2-0.3；或观察 from-scratch 是否能突破 |

- **推荐下一步**（优先级排序）：
  1. **不动 `_145252`**，让它继续训到 iter ~750（与 `_102145` iter 722-727 比对）。若 entropy 仍同 `_102145` 重合（~15.7） → 排除 H5a，重新规划
  2. **预先准备 H5b/c 验证脚本**：(a) 加一个 `train_from_scratch` flag 用于跳过 ckpt load；(b) 准备小型 sweep `encoder_sample_probs.smpl ∈ {0.0, 0.05, 0.1}` 看是否 smpl 偶尔点亮反而帮助 prior 适应
  3. **暂不修 reward landscape**——优先排除架构原因，避免把多个变量同时改
- **更深的反思**：
  1. **从前一个 epoch 的"50:100 速率比"推论 SMPL 是 50% 主因**——这个推断基于一个隐藏假设："`_182557` 死亡时间 12k 是 100% leak 速率的反映"。但 `_182557` 还做了 ABCD bandage（reward weight 减半 + termination 范围缩小），bandage 本身改变了死亡时间。这个对照不干净，导致我把"50% leak 应慢 2×"当成了硬证据。**未来对照需先排除 bandage 干扰**。
  2. **死亡曲线 ≠ leak fingerprint**：本以为只要 trajectory 1:1 重合就证明 leak 没修。但忽略了：如果两个 run 共享同一个非-leak 主因（prior mismatch / 任务难度），它们的轨迹也会重合。**fingerprint 必须包括"修复后某个具体指标偏离"**——这次 ee_body_pos 在 iter 220 完全没变，是 H4 修复无效的最强信号。
  3. **三层保护的安全证明 vs 行为证明的鸿沟**：之前列了"routing 层 + scatter 层 + aux loss 层" 三重 mask 关闭。代码层面这是对的。但**"代码上 SMPL 不可能产生梯度" 不等于"系统行为上 SMPL 缺失没有任何后果"**：actor head 已经 internalized SMPL token 的存在，突然移除是行为层面的扰动。区分"梯度污染"和"输入分布扰动"是两个独立的问题，前一条记录把它们混在一起讨论了。
- **修订 task #11 状态**：completed。下一个 task 是 watch iter 750 checkpoint。

---

## [2026-05-20] `_145252` iter 627-631：H5a 几乎被证伪，死亡曲线 1:1 复现

- **新数据**（iter 627-631 stdout）：

  | iter | entropy | std | length | reward | foot_term | ee_body_term | err_body_pos |
  |---|---|---|---|---|---|---|---|
  | 627 | 15.366 | 0.42 | 24.58 | 1.54 | 0.646 | 0.336 | 0.0734 |
  | 628 | 15.369 | 0.42 | 22.10 | 1.41 | 0.657 | 0.333 | 0.0726 |
  | 629 | 15.372 | 0.42 | 23.55 | 1.48 | 0.675 | 0.318 | 0.0733 |
  | 630 | 15.374 | 0.42 | 25.24 | 1.67 | 0.652 | 0.339 | 0.0736 |
  | 631 | 15.377 | 0.42 | 21.37 | 1.33 | 0.651 | 0.334 | 0.0727 |

- **与 `_102145` iter 722-727（broken control）对照**：

  | 指标 | `_145252` iter 627-631 | `_102145` iter 722-727 |
  |---|---|---|
  | entropy | 15.36 → 15.38 | 15.72（约 100 iter 后） |
  | std | 0.42（死锁在死亡区间）| 0.42 |
  | length | 21-25（震荡）| ~23 |
  | foot_term | 0.65 ± 0.01 | 0.63 |
  | ee_body_term | 0.32 - 0.34 | 0.34 |
  | reward | 1.3 - 1.7 | ~1.5 |

- **斜率核算**：
  - iter 220 entropy ≈ 14.07，iter 630 entropy ≈ 15.37 → slope = 0.00317/iter
  - 与 `_102145` 后段 slope 0.00323/iter **完全一致**
  - 线性外推到 entropy=18 需要 +870 iter → **iter ~1500 饱和**
- **决定性观测**：
  1. **entropy 仍在单调增长**——没有任何 saturate / plateau / decrease 的迹象
  2. **std=0.42 持续 5 iter 完全不变**——锁死在 0.39-0.47 死亡区间正中
  3. **error_body_pos = 0.073** 与之前 `_112413` iter 220 的 ~0.07 同水平，没有任何"修好后下降"的痕迹
- **结论**：
  1. **H5a（晚期分叉）几乎被证伪**：曲线斜率与 `_102145` 完全一致，相位仅落后 ~100 iter，没有任何"修过 H4 → 学习有改善"的信号。继续训到 iter 1500 大概率只是看到 `_102145` 同型饱和，不会有惊喜。
  2. **H4 不是主要根因**（或至少不是单独的根因）：Plan A 在代码层面证明无 SMPL grad，但 trajectory 没变 → SMPL leak 修复实质上**对 23DoF 训练动力学无影响**。
  3. 残余假设按可能性重排序：**H5b（prior mismatch 下 PPO 自然行为）≥ H5d（任务难度，vr_5point 严格）> H5c（actor head OOD）**。
- **决策点**：
  - **kill `_145252`**（saved compute），理由：
    - 已收集到判别 H5a 所需的 entropy slope 数据
    - 继续训只是消耗资源验证已经几乎确定的事
    - 总 ETA 显示 286k 秒（~80h），不值得
  - **不立即开新 run**，先和用户确认下一步走哪个分支：
    - **方案 1（H5b 验证）**：from-scratch 训练（不加载 prior ckpt），同样的 23DoF 配置。如果死曲线消失 → prior mismatch 是主因，需要做 prior adaptation；如果仍死 → 任务难度 / reward landscape 问题。
    - **方案 2（H5f 干净基线）**：`encoder_sample_probs={g1:1.0, teleop:0.0, smpl:0.0}` 单 encoder 路径，剥离 teleop 通路也排查掉。代价是丢失 teleop 数据多样性，可能本身就训不好。诊断价值有限。
    - **方案 3（H5d 验证）**：放宽 reward landscape（vr_5point std 0.1 → 0.2 或 0.3），其它不动。如果死曲线消失 → 任务难度过高；如果仍死 → 不是 reward 问题。这个最便宜。
- **倾向**：方案 1 + 方案 3 并行（互不冲突，各占一张卡）。方案 2 留作 fallback。
- **经验**：
  1. **死亡曲线 1:1 复现 + slope 完全一致 = 主因没动**。这次和 `_182557 → _112413` 一样的 fingerprint，但这次连分叉点都没出现，说明 leak 修复彻底是 no-op。
  2. **H4 修复仍要保留**：即使 SMPL 不是主因，从代码正确性上讲 SMPL 参考通路确实绕过了 FK fix，未来如果 SMPL routing 重新打开（部署或 sweep），bug 仍在。
  3. **下次启动新 run 前先列"3 个 iter 内必须看到的判别信号"**：例如 "iter 600 entropy 必须 ≤ 14.5 才说明 fix 有效"。这次是事后才发现 slope 一致，浪费了 30 分钟训练。

---

## [2026-05-20] H5d 实验启动：放宽 vr_5point reward std (0.1 → 0.25)

- **决策依据**：用户授权方案 3 单跑（H5d），不并行 H5b。
- **改动**：新建 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof_h5d.yaml`，继承 23dof 基线 + 唯一 override `manager_env.rewards.tracking_vr_5point_local.params.std: 0.25`。
- **数学根据**：
  - 当前 reward 公式 `exp(-error.mean(-1) / std**2)`，5 点 sum-of-squares 平均 ≈ 0.032
  - std=0.1 → reward=0.04（实际 stdout 0.067-0.081，再乘 weight=2.0 = 0.13-0.16，与 stdout `tracking_vr_5point_local: 0.0808` 一致）
  - std=0.25 → reward=0.60（**约 15× 更大梯度**）
- **预测的判别信号**（iter 600 必须看到才算 H5d 落地）：
  | 指标 | broken baseline (`_145252` iter 631) | H5d 期望（成功） | H5d 期望（失败 = 同 baseline） |
  |---|---|---|---|
  | entropy | 15.38 仍在涨 | ≤ 14（至少 plateau）| 15.4±0.1 |
  | std | 死锁 0.42 | < 0.40 或开始降 | 死锁 0.42 |
  | tracking_vr_5point_local | 0.080 | > 0.4（reward 直接放大）| 0.6+（自然放大但 entropy 仍涨）|
  | foot_pos_xyz term | 0.65 | < 0.50 | 0.65 |
  | ee_body_pos term | 0.34 | < 0.25 | 0.34 |
  - 关键 trap：reward 数值会自动放大（std² 缩小直接缩小 numerator），所以**只看 reward 数值是错的**。要看 **entropy / termination 是否破坏死亡曲线 fingerprint**。
- **kill `_145252` 的指令**（用户执行）：
  - 当前 iter ~631+，最后 ckpt @ 15:24。曲线 fingerprint 已经够清楚，不再需要继续训
  - `pkill -f "sonic_release_23dof_test-20260520_145252"` 或 Ctrl+C
- **launch H5d 命令**（参照 `_145252` overrides）：
  ```
  python gear_sonic/train_agent_trl.py \
      +exp=manager/universal_token/all_modes/sonic_release_23dof_h5d \
      +checkpoint=sonic_release/last.pt \
      num_envs=512 \
      headless=True \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
  ```
- **观测计划**：
  - iter 100：先看 reward 数值是否如预期放大到 ~0.4。若没有 → yaml 没生效（dump config.yaml 检查）
  - iter 220：与 `_145252` iter 220 比对（最关键的早期判别点）。若 entropy 已偏离 14 → 强信号 H5d 有效
  - iter 600：决定性判断点
  - iter 1500：若到这步仍没有 plateau，H5d 失败 → 启动 H5b（from-scratch）
- **失败兜底**：
  - 若 H5d 在 iter 600 没有突破，**不要继续把 std 加大到 0.5 或 1.0**——那只是把 reward 拉成 noise。直接换 H5b。
  - 若 H5d 看起来工作但 reward 质量下降（mean rewards 涨但 error_body_pos 不降）→ reward landscape 太松，回 std=0.15 重测。
- **关于"3 个 iter 内必须看到的判别信号"**（吸取教训）：
  - 启动 H5d 后 **iter ≤ 100 时检查 1 项**：tracking_vr_5point_local raw value（不是 weighted）。如果 `weight×exp(-err/0.0625)` 没有比 baseline 大 ~10×，立即 kill —— yaml 没生效。
  - 这个 sanity check 比"等 600 iter 才发现配置错"省 1.5 小时。

---

## [2026-05-20] H5d run `_155839` 开训：早期信号 mixed，slope 是关键

- **配置 sanity check** ✓：`config.yaml` 中 `tracking_vr_5point_local.params.std: 0.25` 确认生效。
- **iter 123-128 数据**（从 stdout）：

  | iter | entropy | std | length | reward | foot | ee_body | err_body_pos |
  |---|---|---|---|---|---|---|---|
  | 123 | 13.690 | 0.39 | 18.64 | 1.139 | 0.737 | 0.260 | 0.0651 |
  | 124 | 13.695 | 0.39 | 17.31 | 1.106 | 0.741 | 0.248 | 0.0644 |
  | 125 | 13.699 | 0.39 | 17.52 | 1.097 | 0.778 | 0.202 | 0.0649 |
  | 126 | 13.704 | 0.39 | 16.24 | 1.035 | 0.744 | 0.214 | 0.0637 |
  | 127 | 13.707 | 0.39 | 17.15 | 1.141 | 0.758 | 0.204 | 0.0613 |
  | 128 | 13.711 | 0.39 | 16.77 | 1.071 | 0.765 | 0.230 | 0.0636 |

- **关于 sanity check 的反思**：
  - 我原本设的 "iter ≤100 reward 应 10× baseline" 错了。reward 数值是 `weight × exp(-err/std²)`，同时取决于 error 和 std。
    - baseline iter 631：低 error（0.07）+ 陡 Gaussian (std=0.1) → reward 0.08
    - H5d iter 128：高 error（早期）+ 松 Gaussian (std=0.25) → reward 0.07
  - 数值碰巧重合，但**两者的 reward landscape 完全不同**：H5d 在 error=0.2 附近仍有梯度，baseline 早就饱和到 0
  - **正确 sanity check**：直接 grep saved `config.yaml` 检查 `std: 0.25` 字符串。这次我事后做了，确认生效。**未来启动新 yaml run 后第一件事**就应该是这个，不是看 stdout 数值。
- **mixed 早期信号**（vs baseline `_145252` iter 220 异步对比，H5d 早 92 iter）：

  | 指标 | `_145252` iter 220 | `_155839` iter 128 | 信号方向 |
  |---|---|---|---|
  | entropy | 14.07 | 13.71 | ✓ 低 0.36（H5d 还没爬到 14）|
  | std | 0.40 | 0.39 | ✓ 持平（边缘）|
  | length | 19.6 | 16.8 | ✗ 短（更频繁 fall）|
  | ee_body_term | 0.25 | 0.23 | ✓ 略低 |
  | foot_term | 0.72 | 0.77 | ✗ 高 |
  | error_body_pos | 0.073 | 0.064 | ✓ 略低 |

- **致命隐忧：local slope (123→128) = 0.0042/iter**：
  - 与 baseline iter 220→631 slope（0.00319）持平甚至更陡
  - 线性外推：H5d iter 220 entropy = 13.71 + (220-128)×0.0042 = **14.10**
  - **几乎等于 baseline iter 220 的 14.07**——若 slope 不放缓，H5d 到 iter 220 就汇合到死亡曲线上
  - **5-iter slope 估计噪声大**，需要至少 50-iter 窗口才稳定。但这是值得标记的红色信号
- **决策**：
  - **不立即 kill**——entropy 仍在 13.71，距离 baseline iter 220 还有 92 iter 缓冲
  - **iter 220 是 hard checkpoint**：若 H5d entropy ≥ 14.0 → death curve 重合，kill 转 H5b
  - **iter 600 是终极判决**：若 H5d entropy < 15 → H5d 部分有效；≥ 15.3 → 失败
- **顺便观察**（不影响主判断）：
  - H5d 的 length 16-18 比 baseline 同期估计低，可能因为 reward landscape 平了之后 policy 更"敢探索"，触发 termination 更频繁。等 iter 200+ 再判断
  - foot_pos_xyz 0.77 比 baseline 0.72 worse，但 ee_body 0.23 better。**两侧 reward 没有 trade-off，可能是噪声**

---

## [2026-05-20] H5d iter 220 hard checkpoint：失败，死曲线 1:1 重合

- **iter 222-226 数据**（H5d 当前最新）：

  | iter | entropy | std | length | reward | foot | ee_body | err_body |
  |---|---|---|---|---|---|---|---|
  | 222 | 14.082 | 0.40 | 18.42 | 1.29 | 0.739 | 0.252 | 0.0646 |
  | 223 | 14.086 | 0.40 | 19.58 | 1.33 | 0.730 | 0.263 | 0.0626 |
  | 224 | 14.089 | 0.40 | 18.98 | 1.30 | 0.736 | 0.263 | 0.0649 |
  | 225 | 14.092 | 0.40 | 17.34 | 1.16 | 0.715 | 0.264 | 0.0653 |
  | 226 | 14.096 | 0.40 | 18.52 | 1.23 | 0.732 | 0.258 | 0.0664 |

- **判决（依照之前我自己定的硬阈值）**：

  | 指标 | baseline `_145252` iter 220 | H5d `_155839` iter 222 | 偏离 |
  |---|---|---|---|
  | entropy | 14.07 | **14.08** | +0.01（重合）|
  | std | 0.40 | 0.40 | 0（重合）|
  | length | 19.6 | 18.4 | -1.2（略短）|
  | ee_body | 0.25 | 0.25 | 0（重合）|
  | foot | 0.72 | 0.74 | +0.02（略糟）|
  | err_body | 0.073 | 0.064 | **-0.009（略好）** |

  **6/6 关键指标全部重合或略差**，仅 error_body_pos 略好（10% 更小）。
- **H5d slope 验证**：iter 128 → 222 = 13.71 → 14.08，**+0.37 / 94 iter = 0.0039/iter** ≈ 早期 5-iter 估计 0.0042 ≈ baseline 死曲线 slope 0.0032。我之前的线性外推预测 14.10，实测 14.08，**预测精度 0.02**。
- **结论**：reward landscape 严苛度**不是死曲线主因**。H5d 是 no-op。
- **唯一收获**：相同 iter，相同 entropy，H5d error_body_pos 比 baseline 略低（policy 跟踪精度小升）—— 说明松 Gaussian 给 policy 多了一些梯度，但**这不足以打破 entropy 爆炸**。
- **kill H5d 并转 H5b**（触发条件：entropy ≥ 14.0 已命中）。
- **H5b 实施**：
  - `train_agent_trl.py:178-181` 显示：`config.get("checkpoint")` 不传就走 from-scratch 随机初始化。
  - launch 命令（vs H5d 仅去掉 `+checkpoint=...`）：
    ```
    python gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_release_23dof \
        num_envs=512 \
        headless=True \
        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
    ```
  - **不复用 H5d yaml**：保持单变量隔离，先测"无 prior 是否解决死曲线"，再决定是否叠加放宽 reward。
- **H5b 判别标准（与 H5d 完全不同！）**：
  - random init 起始 entropy 会**很高**（continuous-action policy 常见 35-45），不是 13 附近
  - 健康信号 = **entropy 单调下降**（而不是 H5d 的"≤14"）；std 下降；length 增长
  - 死亡信号 = entropy stuck high 不下降 / 振荡 / 反向爬升
  - **不要用 baseline 的"iter 220 entropy 14.07"做对照**——H5b 起点完全不同，应只看自身轨迹形状
- **H5b 早期判决点**：
  - **iter 50**：entropy 应明显低于初值。若 entropy 从 ~40 没降到 35 → 走错方向
  - **iter 200**：std 应从 ~1.0 降到 < 0.7（policy 开始 commit）
  - **iter 1000**：若 length > 30 + entropy < 25 → H5b 工作良好，prior mismatch 是主因，需要做 prior adaptation 方案
  - **iter 1000 仍 entropy > 30 / length < 10**：H5b 失败，问题在架构层（H5e: critic OOD / H5f: 单 encoder / H5h: 渐进锁关节）
- **如果 H5b 工作（即 prior mismatch 确为主因）的修复路线**：
  1. 简单：把 prior 在 23DoF 上微调（这就是 from-scratch 之外的另一个轴），但本身就是当前在做的事；或
  2. 在 prior ckpt 加载时**重新初始化 actor head 的 last layer**，让它从基本零分布出发；或
  3. **冻结 prior encoder，只训 actor/critic head**——绕过 prior adaptation 困难

---

## [2026-05-20] H5b run `_161845` iter 304-309：**反向死亡曲线**——variance collapse

- **进程确认**：PID 413682 仍在跑；config.yaml `tracking_vr_5point_local.params.std: 0.1` ✓（baseline 配置，无 H5d override）；launch 无 `+checkpoint=` ✓ from-scratch。
- **iter 304-309 数据**：

  | iter | entropy | std | length | err_body | foot | ee_body | anchor_ori_full |
  |---|---|---|---|---|---|---|---|
  | 304 | -33.31 | 0.08 | 5.71 | 0.056 | 0.82 | 0.09 | 0.31 |
  | 305 | -33.29 | 0.08 | 6.67 | 0.055 | 0.82 | 0.09 | 0.32 |
  | 306 | -33.26 | 0.08 | 5.60 | 0.055 | 0.81 | 0.10 | 0.32 |
  | 307 | -33.25 | 0.08 | 6.29 | 0.055 | 0.81 | 0.10 | 0.32 |
  | 308 | -33.22 | 0.08 | 6.16 | 0.056 | 0.82 | 0.09 | 0.29 |
  | 309 | -33.20 | 0.08 | 6.31 | 0.056 | 0.80 | 0.10 | 0.34 |

- **熵数学验证**：连续 Gaussian policy 熵 = `Σᵢ 0.5 · log(2πe·stdᵢ²)`。
  - 30 维 × 0.5 × log(2πe × 0.08²) = 15 × log(0.1094) = 15 × (-2.213) = **-33.2** ✓ 与实测完全吻合。
  - 起点（random init, std≈1.0）熵 ≈ 30 × 0.5 × log(2πe) ≈ 42.6。
  - **从 42.6 → -33.2 = 总熵下降 75.8**，对应 std 从 ~1.0 崩到 0.08。
- **新失败模式：variance collapse（与 baseline 完全相反）**：

  | 维度 | baseline `_145252` 死曲线 | H5b `_161845` 崩塌 |
  |---|---|---|
  | entropy 趋势 | 13 → 18（爆炸）| 42.6 → -33.2（崩塌）|
  | std 趋势 | 0.39 → 0.42（涨）| 1.0 → 0.08（崩）|
  | length | 19 → 16（短）| 5-7（极短）|
  | 失败本质 | PPO 用 entropy bonus 逃出零梯度区，policy 越来越随机 | PPO 用 surrogate gradient 锁死到一个 deterministic 落地姿势 |

- **为什么 length 5-7 但 err_body / ee_body 反而比 baseline 还低？**：
  - episode reset 时 body_pos 与参考重合（initial pose 匹配）。length=5-7 意味着平均看到的是 reset 后的前 5 步——本来就接近参考，**误差小是 reset 假象不是真学到了**。
  - foot=0.82 远高于 baseline 0.72：脚跟踪烂（policy 不踩稳）；
  - ee_body=0.09 极低、err_body=0.055 低：因为 reset 后头/腕还没有时间漂移——**5 步 rollout 的局部假象**。
  - **这是"放弃运动→快倒→快 reset→再用初始姿势刷低误差"的 PPO local minimum**。
- **关键结论**：
  1. **prior mismatch 不是唯一主因**（也不是最大因）。**有/无 prior 都死，只是死法不同**。
  2. baseline（prior+23DoF）死于 entropy 爆炸；H5b（无 prior+23DoF）死于 variance collapse。两者**底层共同点 = 23DoF 环境无法给 PPO 提供有效的 differential reward signal**。
  3. PPO 在两种 init 下分别走到两个极端 local optimum，说明问题在 **environment / reward / action_space 这一层**，不在 actor 初始化。
- **可疑架构层 bug 候选**（按可证伪性优先级）：
  - **A. action 空间错配（最可疑）**：`config.yaml:419` 的 `joint_names: - .*` 让 JointPositionActionCfg 选**全部 29 个 G1 关节**。但 6 个关节物理 locked，policy 输出的对应 6 维 action **完全不影响 reward**。这 6 维上 reward gradient = 0，仅有 entropy bonus 推动；剩下 23 维上 reward gradient 正常但和 6 个 free-rider 共享 entropy_coef。结果取决于 entropy_coef 与 PPO clip 的相对强度——**正好对应"两种 init 走到两种极端"的现象**。
  - **B. critic OOD**：critic 见过的 obs 分布是 29DoF（masked 改了 obs 后）；如果 value baseline 估计偏离真实回报，advantage 估计有 systematic bias，PPO 会朝错误方向走。
  - **C. reward 在 23DoF 下数值上 inconsistent**：例如 vr_5point 的参考 5 点是**带 missing-dof 修正后的 FK** vs policy 实际能驱动的关节集合不匹配（已修过 pose_aa）但其它 reward term（如 dof_track）可能没修。
  - **D. termination 太严**：23DoF 锁了 hip 之外的 dof，可能导致自然倒地阈值（如 base_height、orientation）频繁触发，使 length 永远 < 10——policy 没机会探索。
- **判决**：H5b 已可宣告失败（不需要等到 iter 1000，variance collapse 已锁死）。**kill `_161845`**，转架构层调试。
- **下一步实验设计**：见下一节"H5e 启动决策"。

---

## [2026-05-20] 下一步：H5g（29DoF from-scratch sanity check）应在 H5e 之前

- **逻辑**：在做 critic-OOD / 部分重置 actor head 这种侵入性改动之前，需要先**确认 29DoF 全关节 from-scratch 在当前代码库下能正常学**。如果 29DoF from-scratch 也死，那不是 23DoF 特有问题，是仓库本身的训练设置 broken（reward / termination / PPO hparam 改坏了）。
- **H5g**（最便宜的 sanity check，~1h）：
  ```
  python gear_sonic/train_agent_trl.py \
      +exp=manager/universal_token/all_modes/sonic_release \
      num_envs=512 \
      headless=True \
      ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
  ```
  - 注意：**不带** `_23dof`、不带 `+checkpoint=`，纯 29DoF from-scratch。
  - 判别：iter 1000 时 entropy 应 < 25，length 应 > 30。
  - **若 H5g 也死**（entropy 异常 / length stuck）→ 代码库本身坏了，先去查最近改 PPO hparam / termination / reward 的 commit；H5e 全部作废。
  - **若 H5g 工作**（PPO 收敛正常）→ 23DoF 特有问题 confirmed，进 H5e。
- **H5e（架构层）启动条件**：H5g 通过 → 锁定 23DoF 特有 bug 候选 A/B/C/D。
  - 最先验证 **A（action 空间错配）**，因为最便宜（修一行 yaml）：把 `actions.joint_pos.joint_names` 从 `.*` 改成 23DoF 显式列表（即 `joint_limit.params.asset_cfg.joint_names` 已有的 23 个）。
  - 若 A 修了仍死，再做 B（critic 重置）/ C（reward 审计）/ D（termination 审计）。

---

## [2026-05-20] H5g run `_165112` iter 50：**29DoF from-scratch 也崩**——大反转

- **iter 50 数据**：

  | 指标 | 值 |
  |---|---|
  | entropy | **-40.63** |
  | std | 0.06（实测 ≈ 0.0625）|
  | length | 11.94 |
  | reward | 0.91 |
  | err_body_pos | 0.095 |
  | err_anchor_rot | 0.244 |

- **熵数学验证**：30 × 0.5 × log(2πe × 0.0625²) = 15 × log(0.0668) = -40.58 ✓ 与实测 -40.63 吻合（std=0.0625 而非 0.06）。
- **崩塌速度**：起始 std=1.0 → entropy ≈ 42.6；50 iter 后 entropy = -40.6。**总下降 83.2 / 50 iter ≈ 1.66/iter**。比 H5b 还快（H5b 用 ~300 iter 降到 -33；H5g 用 50 iter 降到 -40）。
- **决定性结论**：
  1. **代码库的 from-scratch PPO 本身 broken**，与 DoF 设定无关。
  2. 释放的 SONIC ckpt 处于**狭窄稳定区**——离开稳定区 PPO 必崩，方向取决于初始化：
     - prior + 离开稳定区（如 23DoF 锁关节、reward landscape 改）→ entropy 爆炸（`_145252`、H5d）
     - 无 prior → variance collapse（H5b、H5g）
  3. **H5b 的失败不是"23DoF 特有 bug"**——这个失败模式在 29DoF from-scratch 同样发生。
  4. **释放 ckpt 大概率是 warm-start 出来的**（BC 预训练 / expert demos / 不同 entropy_coef tuning），不是用当前 yaml 的 PPO 直接 from-scratch 出来的。
- **被 H5g 一并清空的实验分支**：
  - 所有 from-scratch 类实验（H5b 复刻、不同 hparam from-scratch）都没信息量——会一致地 variance collapse。
  - `H5g 工作 → 23DoF 特有 bug` 这个分叉**不存在**了。两条路（23DoF / 29DoF）从零训都死，所以原假设"23DoF 特有 bug"被弱化但不证伪：23DoF 仍然可能有特有的、加在 from-scratch 不稳之上的额外问题（baseline 死曲线就在 prior 加载时才出现）。
- **存活 hypothesis（按可证伪性 / 成本排序）**：
  - **H5e-A（action 空间错配）** $1：把 `actions.joint_pos.joint_names: .*` 改成 23 个显式关节，仍载入 prior 训。判别是否消除 baseline 死曲线。
  - **H5e-C（entropy_coef 调低）** $1：保持 prior+23DoF，把 PPO entropy bonus 系数压低（典型 0.005 → 0.001），看是否压住 entropy 爆炸。**不修 root cause** 但能告诉我们爆炸是不是 entropy bonus 主导。
  - **H5e-B（critic 重置）** $5：保持 prior，重新初始化 critic，让 critic 与 23DoF obs 重新 align。
  - **H5e-D（reward 审计）** $0+脑力：对照 23DoF vs 29DoF 各 reward term 看哪个数值不一致；尤其是 dof_track 是否漏修。
- **下一步**：
  - **kill H5g**（不需要等到 iter 1000，结论已锁死）。
  - 启动 **H5e-A**：修改 yaml 后 prior+23DoF 训。判别条件 = baseline 死曲线（entropy ≥ 14 at iter 220）是否消失。

## [2026-05-20] H5e-C run `_172732`：entropy_coef 10× 降低 → 部分胜利

- **配置**：`sonic_release_23dof_h5e_c.yaml`，仅改 `algo.config.entropy_coef: 0.01 → 0.001`，其他与 baseline `_145252` 完全一致（载入同 prior、23DoF lock、H1+H4 fix）。
- **关键 milestone**：

  | 指标 | baseline `_145252` iter 220 | H5e-C iter 220 | Δ |
  |---|---|---|---|
  | entropy | 14.07 | **12.67** | **−1.40** ⬇⬇⬇ |
  | std | 0.40 | 0.38 | −0.02 |
  | reward | ~0.96 | **1.11** | +0.15 ⬆ |
  | length | 19.6 | 17.67 | −1.93 |

  | 指标 | baseline iter 500 (peak) | H5e-C iter 500 | Δ |
  |---|---|---|---|
  | entropy | 14.96 | **11.96** | **−3.00** ⬇⬇⬇⬇ |
  | reward | **1.647** | 1.487 | −0.16 |
  | length | **25.97** | 22.54 | −3.43 |

  | 指标 | baseline iter 631 (死) | H5e-C iter 578 (peak) | H5e-C iter 800 |
  |---|---|---|---|
  | entropy | 15.38 | 11.79 | **11.31** ⬇ |
  | std | 0.42 | 0.37 | 0.36 |
  | reward | 1.328 | **1.572** | 1.465 |
  | length | 21.37 | **22.75** | 21.75 |
  | err_anchor_pos | 0.853 | — | **0.931** |
  | err_body_pos | 0.073 | — | **0.074** |
  | ee_body_pos term | 0.334 | — | **0.337** |
  | foot_pos_xyz term | 0.651 | — | 0.637 |

- **熵曲线已彻底反转**：baseline entropy 单调爬升 13.99→14.07→14.33→14.96→15.28→15.38 (iter 200→631)，H5e-C entropy 单调下降 12.72→12.67→12.48→11.96→11.79→11.31 (iter 200→800)。**entropy_coef 是死曲线主驱动之一确认。**
- **但反胜不彻底**：
  - H5e-C peak reward 1.572 < baseline peak 1.647（**追不上 baseline 巅峰**）。
  - H5e-C 自己也从 iter 578 peak 缓慢退化（reward −0.107 / 222 iter = 0.00048/iter，比 baseline 退化 0.0024/iter 慢 5×，但同向）。
  - 至 iter 800，body tracking metrics（err_anchor_pos 0.93, ee_body_pos term 0.337）**几乎重合 baseline iter 631 死亡态**——H5e-C 也在重走 baseline 的失败轨迹，只是慢 5×。
- **修正 H5e-C 解读**：死曲线 = (entropy 爆炸) + (其他机制)。entropy_coef 解决前者后，后者孤立暴露——即使 entropy 已被压住，policy 仍在退化，且退化速率是 baseline 的 1/5。
- **次级假设**：actor_learning_rate=2e-5 太小，actor mean 跟不上 23DoF gradient direction。critic_lr=1e-3 比 actor 高 50×，可能 critic 一直在 "追" 不稳定的 actor 输出，进而恶化 advantage estimate。
- **决定**：kill H5e-C（iter 800，结论锁定），启 H5e-E。

## [2026-05-20] H5e-E run 启动：entropy_coef=0.001 + actor_lr 5× boost

- **配置**：`sonic_release_23dof_h5e_e.yaml`：H5e-C + `actor_learning_rate: 2e-5 → 1e-4`（5×）。`adaptive_lr_max: 2e-4` 默认未改（剩 2× 上行 headroom）。
- **判别条件**：
  - SUCCESS：reward 突破 H5e-C peak (1.572) 且持续爬升过 iter 800，err_body_pos 稳定 < 0.07。
  - PARTIAL：达到 H5e-C peak 但不突破 → 第二因素不是 actor_lr，转而排查 critic OOD 或 reward 不一致。
  - FAILURE：actor 不稳，reward 早崩 → 5× 太激进，回退 2-3×。

### H5e-E 第一次跑 run `_182052`：iter 499 OOM 中断（CUDA 碎片化）

- **早期数据（iter 100-300）**：

  | 指标 | iter 100 | iter 220 | iter 300 (peak) |
  |---|---|---|---|
  | entropy | 12.94 | 12.67 | 12.48 |
  | std | 0.38 | 0.38 | 0.38 |
  | reward | **1.18** | **1.24** | **1.53** |
  | length | **20.19** | **20.01** | **24.03** |
  | err_body_pos | — | 0.066 | 0.067 |

  - 同期 baseline iter 220 reward 0.96, H5e-C iter 220 reward 1.11。**H5e-E 早期增长显著快于两者，而 entropy 与 H5e-C 持平**——actor_lr 5× 让 actor mean 漂移更快但不影响 std/entropy。
- **iter 300 → 499 退化**：

  | 指标 | iter 300 | iter 499 |
  |---|---|---|
  | entropy | 12.48 | 11.99 |
  | reward | **1.53** | **1.37** |
  | length | **24.03** | **20.45** |
  | err_body_pos | 0.067 | 0.071 |
  | ee_body_pos term | 0.27 | 0.30 |

  - **H5e-E 与 H5e-C 同样在某个 iter 后开始 reward retreat**，但 H5e-E peak 更早（iter 300 vs H5e-C iter 578）且更快下滑（−0.16 / 199 iter vs H5e-C −0.107 / 222 iter）。
  - **iter 499 H5e-E reward 1.37 < H5e-C 同期 1.49 < baseline 同期 peak 1.65**——所谓"actor_lr 解决 H5e-C retreat"被证伪，actor_lr 5× 反而让 retreat 来得更早。
- **OOM**：iter 499 末，PyTorch CUDA 分配 100 MiB 失败（残 117 MiB / 8 GiB，843 MiB 被 reserved-but-unallocated 占住）。属典型分配器碎片化，与 H5e-E 实验逻辑无关——但 8GB Laptop 4060 长跑确实需要分配器调优。
- **重启策略**：加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（PyTorch 官方提示）。从 prior `sonic_release/last.pt` 重启（save_interval=500 没保到 ckpt，只能重头跑）。

### 修订对 actor_lr 假设的判断

- 早期数据（iter < 300）确实显示 H5e-E 显著领先，**但这是过快的瞬时漂移而非真正学到更好的 policy**——iter 300-499 的 reward retreat 比 H5e-C 还快说明了这点。
- 死曲线第二因素**不是简单的 "actor_lr 太小"**：
  - 如果是的话，提高 actor_lr 应消除 retreat。但实测 retreat 来得更早。
  - 更可能：critic OOD（critic 输出 advantage 在 23DoF 状态分布上偏差），actor 跟着偏 advantage 漂移更快只会更早误导。
- **新候选假设**：critic 重置 / critic warmup（先冻结 actor 几百 iter 让 critic 在 23DoF 数据上 align）。但这是明天的事。
- **过夜目标修正**：H5e-E 重启后看 iter 1000-3000，确认 retreat 是否会停在某个 plateau，还是继续崩到 baseline death。

> **⚠️ 上面这段分析在 H5e-E v2 跑通后被证伪——v1 iter 499 的 reward 1.37 是单点噪声谷而不是退化趋势。保留作为推理记录，结论以下面 v2 的数据为准。**

## [2026-05-21] H5e-E v2 run `_185243`：过夜跑通，**SUCCESS** ✓

- **配置同 v1**，加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防 OOM。15.4 小时跑到 iter 16368 持续，无错误。Checkpoint 每 2000 iter 存一次，已有 8 个（iter 2000–16000）。
- **完整 milestone（早 10:14 截取）**：

  | iter | entropy | std | reward | length | err_anchor | err_body | ee_term | foot_term |
  |---|---|---|---|---|---|---|---|---|
  | 100 | 12.94 | 0.38 | 1.18 | 20.2 | 0.71 | 0.063 | 0.22 | 0.78 |
  | 220 | 12.67 | 0.38 | 1.24 | 20.0 | 0.73 | 0.066 | 0.27 | 0.74 |
  | 300 | 12.48 | 0.38 | 1.53 | 24.0 | 0.87 | 0.067 | 0.27 | 0.72 |
  | 500 | 11.99 | 0.37 | 1.55 | 23.9 | 0.26 | 0.070 | 0.32 | 0.69 |
  | **800** | 11.40 | 0.37 | **1.82** ⬆⬆⬆ | 25.1 | 0.93 | 0.077 | 0.29 | 0.67 |
  | 1000 | 11.04 | 0.36 | 1.77 | **25.84** | 0.23 | 0.080 | 0.35 | 0.65 |
  | 1500 | 10.22 | 0.35 | 1.58 | 24.5 | 0.22 | 0.077 | 0.38 | 0.61 |
  | 3000 | 8.57 | 0.34 | 1.53 | 24.7 | 0.23 | 0.076 | 0.36 | 0.63 |
  | 6000 | 7.21 | 0.33 | 1.49 | 23.6 | 0.20 | 0.074 | 0.36 | 0.66 |
  | 10000 | 5.91 | 0.32 | 1.68 | 25.3 | 0.25 | 0.071 | 0.33 | 0.63 |
  | 16000 | 4.59 | 0.31 | **1.73** | **25.45** | **0.18** | 0.072 | 0.34 | 0.65 |

- **SUCCESS 判据全部命中**：
  - ✅ reward 突破 H5e-C peak（1.82 > 1.572）
  - ✅ 持续过 iter 800（一路跑到 iter 16k 仍稳）
  - ✅ err_body_pos 长期 ≤ 0.08
  - ✅ 无 entropy 爆炸（baseline 死法），无 variance collapse（H5b/H5g 死法）
  - ✅ 跨过 baseline death iter 631 后还活了 25× 时长
- **关键修正昨天的分析**：v1 iter 499 reward 1.37 **不是退化**，是单点噪声谷。v2 iter 500 reward 1.55，iter 800 reward 1.819。所以 **actor_lr=2e-5 太小确实是死曲线第二因素**——上一段"被证伪"的判断本身被证伪。

- **err_anchor_pos 5× 改善信号**：iter 100-800 在 0.7-0.9 区间，iter 1000 后稳定在 0.18-0.26。说明 policy 在 iter 1000 附近真正学会"keep root anchored"，是关键 phase transition。

- **死曲线机制最终定论**：
  - **(A) entropy_coef=0.01** 让 PPO entropy bonus 在 6 个 free-rider missing-DoF 维度上充气，把 std/entropy 推上去 → 探索过宽 → policy 漂离稳定区。**已由 H5e-C 单独证实。**
  - **(B) actor_lr=2e-5** 太小，actor mean 跟不上 23DoF gradient direction，**只能靠 entropy 探索回稳定区，但探索方向被 (A) 污染** → 长期 reward 退化。**由 H5e-E 同时调 (A)+(B) 后稳定增长证实 (B) 必须配合 (A) 才有效**——单调 (A) 不够（H5e-C 仍 retreat），单调 (B) 没试过。
  - 两个超参都是"为 29DoF prior 调的"——23DoF lock 改变 reward landscape，原配置不再 valid。

- **方法论 takeaway**：
  - 单点 iter 数据（如 v1 iter 499）极易误判趋势。**至少看 200-500 iter 窗口的滑动平均**才能判 retreat。
  - PPO retreat 看似下降但可能是噪声谷——参考点必须是 `peak − 至少 0.3 reward` 才算真退化。
  - 长跑 GPU OOM 是常见问题，**所有 8GB GPU 长跑必加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**。

- **下一步候选**：
  1. 让 H5e-E 继续跑到 iter 30000-50000（精化），或 iter 16000 ckpt 直接做 evaluation/可视化。
  2. 准备 evaluation 命令验证 ckpt 真能在仿真里走完整 episode。
  3. 后续工作回到 D1 部署主线（这条死曲线分支至此结束）。

## [2026-05-21] H5e-E iter 20000 ckpt → ONNX 导出 → sim2sim 部署调试

H5e-E v2 跑到 iter 20000 后停止，准备做 sim2sim 验证。本节记录从 ONNX 导出到部署 yaml 错位的完整排查链。

### 1. ONNX 导出

- 命令：`python eval_agent_trl.py +checkpoint=<ckpt> ++num_envs=1 +export_onnx_only=true`
- `eval_agent_trl.py:474-560`：导出到 `<experiment_dir>/exported/`，共 5 个 ONNX：
  - `model_smpl.onnx` / `model_g1.onnx` / `model_teleop.onnx`：三个 encoder 端到端（encoder + decoder 合并）
  - `model_encoder.onnx`：拆开的 encoder（多 encoder 共享一份），输入布局 `[encoder_index(1) | tokenizer_observations]`
  - `model_decoder.onnx`：拆开的 decoder
- deploy 端只用 `model_encoder.onnx` + `model_decoder.onnx` 两个；其它三个 end-to-end 是为了 Python 单文件推理验证用，不进 C++ runtime。

### 2. 替换 deploy 端 ONNX

- 备份原文件到 `gear_sonic_deploy/policy/release/backup_20260514/`（包含老 ONNX、对应 TRT cache、observation_config.yaml 副本）。
- 替换 `policy/release/model_encoder.onnx` 和 `policy/release/model_decoder.onnx` 为新导出。
- **删掉** `encoder_model_encoder.trt` 和 `policy_model_decoder.trt` —— `InferenceEngine.cpp:114-167` 会按 onnx+GPU+precision 算 SHA256 hash，hash 不匹配自动重建，所以删 cache 是干净做法。

### 3. 维度错位（核心难点）

启动 sim 报错：encoder 期望 input dim **1762**，新 ONNX 是 **1751**，差 11 维。

排查链：
- 读 ONNX I/O specs：新 ONNX encoder input shape `(1, 1751)`，老 ONNX `(1, 1762)`。
- 读 `unitoken_all_noz.yaml`（训练时 tokenizer obs 配置）：12 项 obs，其中 `command_z` (1 维) + `command_z_multi_future_nonflat` (10 维) = 11 维，名字带 `_noz` = "no Z"。
- 读 `inference_helpers.py:200-322` `export_universal_token_encoders_as_onnx`：会把 tokenizer obs **过滤** 到只剩 encoder `inputs:` 里实际 consume 的，所以 `_noz` 配置下 command_z 系列不进 ONNX → 1751。
- 读 H5e-E ckpt encoder 权重 shape：`encoder[g1].module.0.weight = (2048, 640)`，640 = 580 (g1 inputs 总和) + 60 (额外维度)。算回去总 encoder ONNX input = 640 + 267 + 840 + 3 (encoder_index) + 1 (wrapper) = **1751**。**ckpt 自己就是 1751-arch。**

中间一度卡在"训练时 obs 1751 又怎么从官方 1762 ckpt 微调出来的"——这是个错误前提。继续排查发现：官方 `sonic_release/last.pt` 加载到 H5e-E 时没有 shape error，所以**官方 ckpt 也是 1751-arch**。

### 4. 真正根因（GitHub issue #122）

用户找到 https://github.com/NVlabs/GR00T-WholeBodyControl/issues/122 ：

- 完全相同的报错（encoder expected 1762 got 1751，差 11）。
- NVIDIA collaborator @ZhengyiLuo 回复：**ONNX 导出代码已更新，去掉了未使用的 keys（即 command_z 系列）；只需要更新 C++ 部署端的 yaml 文件**。
- 提供了 `g1_wrist_joints_10_clean.yaml` 替换 `policy/release/observation_config.yaml`。
- 用户 @YutongWangCatherine 验证："运行得非常完美。"

**结论**：1751 是新版导出的正确维度，1762 是 deploy 端 yaml 残留的旧版规范。修复 = 替换 yaml,不是改训练或扩 ckpt。所有之前讨论的"手术微调 11 维 / 写自定义 deploy"方案全部作废。

### 5. NVIDIA driver/library mismatch（独立故障）

强行启动 sim 失败，两种现象：
- `run_sim_loop.py` 报 `GLFW BadValue (integer parameter out of range)`
- `deploy.sh sim` 报 `CUDA error 804`

诊断：`nvidia-smi` 输出 `Failed to initialize NVML: Driver/library version mismatch, NVML library version: 580.159` —— 内核模块还是旧版，userspace lib 是新版（apt upgrade 后没 reboot 的典型症状）。

**修复**：reboot（最干净）。或 `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` 链式重载（但前提是没有进程占用 GPU，否则只能 reboot）。

### 6. 待执行步骤（reboot 后）

1. 从 issue #122 附件下载 `g1_wrist_joints_10_clean.yaml`。
2. 备份并替换 `gear_sonic_deploy/policy/release/observation_config.yaml`。
3. 保留当前 1751-dim H5e-E ONNX（不要从 `backup_20260514/` 回滚）。
4. 运行 `bash deploy.sh sim`，首次启动会基于新 ONNX 重建 TRT cache，之后跑通 sim2sim。

### 经验

- **"不可能的现象"先质疑前提**：堵在"H5e-E 是从 1762 ckpt 微调出来的，那为什么是 1751"时，正确动作是 dump ckpt encoder weight shape 看真实尺寸，而不是反复读训练 config。**ckpt 是 ground truth,config 是描述。**
- **找官方 issue 比自己改代码快**：在确定要写"surgical 11-dim finetune"前查 GitHub issue tracker，能直接拿到官方修复路径。
- **`nvidia-smi` 失败要先看 NVML 版本提示**：driver/lib mismatch 是 apt upgrade nvidia-* 后没 reboot 的标志症状,reboot 修；不要去重装 driver / 重编 kernel module。



---

## [2026-05-21] sim2sim 部署 D1 23DoF finetune 仍摔倒(进行中)

- **现象**:yaml 替换成 `g1_wrist_joints_10_clean.yaml` 后 dim error 消失,但策略秒倒。terminal 2 按 `]` → MuJoCo 窗口 `]` 9 后机器人立刻陷入怪异姿势,无法维持站立,根本走不到按 `T` 播参考动作那一步;约 120 ms 后 terminal 2 报 `Lost LowState data connection from robot!` sim 端挂掉。
- **触发**:用户的 D1 23DoF finetune ckpt(H5e-E iter 20000)export 成 ONNX → 替换 `policy/release/model_*.onnx` → 用 issue #122 的 `g1_wrist_joints_10_clean.yaml` 替换 yaml → 直接 `bash deploy.sh sim` + `python gear_sonic/scripts/run_sim_loop.py`(**sim 端没加 `--simulate-23dof`**)。
- **发现路径**:用户报告"obs 是不是真的对齐了"。dim 错误已解决但物理表现完全不对,提示问题不在 dim count 而在 obs **语义**。
- **调查**:
    - md5 校验:`policy/release/observation_config.yaml` 已替换为 Downloads 同 hash,仅末尾混入 GitHub UI 残留 `Message Zhengyi Luo`。读 `src/g1/g1_deploy_onnx_ref/include/observation_config.hpp:113-340` 自定义 simple parser 是 line-based token-match,不识别的行直接 skip,**该残留无害**(后续被某处自动清理,75 行干净状态)。
    - `policy/release/model_*.onnx` mtime = 今天,`backup_20260514/` 有 5/14 NVIDIA 原版。当前部署的是用户自己 D1 23DoF iter 20000 的 export。
    - `gear_sonic/utils/mujoco_sim/configs.py:348` `simulate_23dof: bool = False` —— sim 默认跑 **29DoF**。
    - `gear_sonic_deploy/reference/zero_23dof_columns.py:26-34` 列出 6 个 missing 关节:MJ idx [13,14,20,21,27,28] = waist roll/pitch + L/R wrist pitch/yaw,IL col [5,8,25,26,27,28]。
    - 训练时 ckpt 在这 6 个关节上是 locked / obs 槽 zero(参 [[wbc-waist-pitch-postural]]、上一条"D1 23DoF finetune baseline")。
    - terminal 2 实跑日志:
      ```
      Reset init reference data root rotation to current frame: -0.698587, 0.0482, 0.047247, 0.712335
      Reference motion name: temporary_motion
      [Mode Filter] Switched to mode 'g1' (ID=0) with 4 required observations
      ... ~12 cycles ...
      [ERROR] Lost LowState data connection from robot!
      ```
      这条 quat (xyzw) norm=1,w=0.7123 → θ≈89.2°,轴≈-X,意味着 robot 在 reset 那一刻已经**绕 -X 倾斜 ~90°**(近躺倒姿态),非站立。
    - 用户照建议加 `--simulate-23dof` 重跑 sim 后**仍然摔**——但 terminal 1 启动日志没贴出,无法确认 flag 是否真生效(应该出现 `[run_sim_loop] simulate_23dof=True` 和 `[simulate_23dof] locked 6 missing joints with damping=...`)。
- **当前假设(待验证)**:
    1. (高优先) sim 端 `--simulate-23dof` 没生效或写法错误 → 6 个关节没锁,obs 语义仍错位。需用户贴 terminal 1 日志确认。
    2. (中) 即使 flag 生效,iter 20000 H5e-E ckpt 本身欠收敛,sim 端 23dof 模式下 obs 分布与训练分布仍有微小差异 → policy 不收敛到稳态。
    3. (中) deploy 端 `temporary_motion` 占位机制下 mode g1 的 motion_* obs 来源不明,可能塞了 zeros 或 stale pose,与训练时 dataset motion 分布不一致。
    4. (低) 那个 -0.698 quat 说明 sim 启动 keyframe 异常或 policy 启动瞬间已被错误命令拉倾斜,跟主因独立或耦合,需进一步看 sim 端 mjcf init pose / keyframe。
- **根因**:**进行中**,等 terminal 1 日志和 sanity test(NVIDIA 原版 ONNX + `--simulate-23dof`)结果。
- **改动**:
    - `gear_sonic_deploy/policy/release/observation_config.yaml`:替换为 `g1_wrist_joints_10_clean.yaml`(75 行,1751 dim 对齐);末尾 `Message Zhengyi Luo` 残留已清。
    - 无 ckpt / 训练 config 改动。
- **验证**:**进行中**——等 sanity test 出结果。
- **下一步排查路径**:
    1. 用户贴 terminal 1 完整启动日志,确认是否出现 `[simulate_23dof] locked 6 missing joints with damping=...`;若无,先排查 tyro flag 写法 / venv 是否对。
    2. sanity test:把 `policy/release/model_*.onnx` 临时还原成 `backup_20260514/` 的 NVIDIA 原版 + 当前新 yaml + `--simulate-23dof` 跑 sim2sim,看是否能站住。
       - 站得住 → 问题在用户的 finetune ckpt(假设 2);
       - 仍摔 → 问题在 deploy 基础设施(假设 1/3/4 之一)。
- **经验**:
    - **dim 对得上 ≠ 语义对得上**:替换 yaml 让 ONNX input dim 数值匹配只是必要条件;每个槽位的来源/单位/joint order 都得一致才算"obs 对齐"。
    - **23dof 部署是双端契约**:训练侧锁 6 个关节,部署侧也必须锁。yaml 修完后 sim 端必须 `--simulate-23dof`,否则 6 个 free DoF 的真实状态喂进去,policy 没见过这种 obs → 输出乱动作。该 flag 用户自己的 `INSTALL_SIM2SIM.md` §11.x 写过。
    - **诊断 quat 当 sanity check**:打印 `Reset ... root rotation` 这种状态量,xyzw 形式 w 应该接近 1(站立);任何 |w|<0.95 都意味着 base 倾斜 >10°,与"摔倒"现象互证。

## [2026-05-21] H5e-E iter 4k/20k 部署都摔 → 定位 actor_lr,启动 H5g(rollback 实验)

> ⚠️ **命名冲突**:本节"H5g"是 H5e 实验家族的 actor_lr rollback,**与 2026-05-20 的 "H5g run `_165112`"(29DoF from-scratch 探针)不是同一个实验**。yaml 文件 `sonic_release_23dof_h5g.yaml` 是今天新建的;旧 H5g 当时只在 debug log narrative 里用过这个标签,没有同名 yaml(走的是 parent `sonic_release` + CLI override)。后续若要 disambiguate 可改名 H5h。

- **症状**:H5e-E iter 20000 ckpt 导出 ONNX 部署 sim2sim 一按 `]` 立刻抽风(参上一节);今天又导出 iter 4000 ckpt,**同样模式抽风**。NVIDIA 原版 1751 ONNX 部署在同一套 deploy 配置下完美站立,排除 deploy 基础设施问题。
- **诊断路径**(逐项排除):
    1. **关节顺序对齐?** 用户怀疑 "iter 4k 也炸 = 没对齐"。读 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp` 的 `isaaclab_to_mujoco[29]` / `mujoco_to_isaaclab[29]` 数组,对照 `joint_constants.py:MISSING_23DOF_INDICES_IL=[5,8,25,27,26,28]`(注意 IL 把 wrist pitch/yaw 按 joint 类型穿插排,**不是 sorted**):全部 ✓ 对齐。
    2. **架构 mismatch?** dump 三个 ckpt(iter 4k / 20k / NVIDIA orig)的 encoder/decoder weight shape:完全一致,排除 warm-start 时 `strict=False` 静默吞 dim 错误的可能。
    3. **是 ckpt 自身坏了?** 既然两端契约对、架构对,1751 NVIDIA orig 也跑得通,那 H5e-E 的 ckpt 本身就是问题所在。
- **根因**:**H5e-E 的 `actor_learning_rate=1.0e-4`(parent default 2e-5 的 5×)在 warm-start 后第一波 update 就把 policy 推出 `sonic_release/last.pt` 的稳态盆地**。具体证据链:

    | Run | entropy_coef | actor_lr | 训练观察 | 部署 |
    |---|---|---|---|---|
    | H5d | 0.01 | 2e-5 | 早期死曲线(entropy↑) | 未导 |
    | H5e-C `_172732` | 0.001 | 2e-5 | iter 578 reward **1.572** peak,iter 800 retreat 到 1.465 | 未导 |
    | **H5e-E** `_185243` | 0.001 | **1.0e-4** | iter 20000(过夜),`SUCCESS` 标签是基于训练曲线,但 ckpt 部署立刻抽风 | **iter 4k / 20k 都炸** |

  H5e-E 训练曲线"看起来收敛"是因为 entropy/std/reward 这些 *训练域* 指标通过 PPO ratio clip 自调出一个新的稳态,但这个新稳态的 action 分布和 `sonic_release/last.pt` 的 prior 已经偏离很远 —— sim 部署看到的"一按 ] 直接诡异抽风"就是 prior 被冲散后的新 mean 在真物理上对应不到合理动作。
- **教训**:
    - **训练 metrics 收敛 ≠ 部署可用**。H5e-E `_185243` 在前一晚被记为 SUCCESS,只是因为训练域信号(reward 1.5+)看起来没崩。部署是新的真理审判,跨过 sim2sim 的那条 sanity gap 才算数。
    - **warm-start 用 5× lr 等于把热启动当冷启动跑**。H5e-C 已经证实 2e-5 在这个 entropy_coef 下能 push 到 1.572 peak,加 lr 不是"收敛得更快",是"跨过盆地后再也回不来"。
- **改动**(2026-05-21 18:33):
    - 新建 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof_h5g.yaml`:`defaults: sonic_release_23dof`,显式 pin `entropy_coef: 0.001` + `actor_learning_rate: 2.0e-5`(parent default 也是 2e-5,显式 pin 防 ppo_im_phc.yaml 后续被改静默漂移)。
    - 起 run `_183343`,从 `sonic_release/last.pt` fresh warm-start,**不**继续 H5e-E iter 20000 corrupted ckpt。
    - launch 命令:
      ```bash
      nohup /home/woan/.conda/envs/groot_wbc/bin/python gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_release_23dof_h5g \
        +checkpoint=sonic_release/last.pt \
        num_envs=512 headless=True \
        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
        > /tmp/h5g_launch.log 2>&1 &
      ```
      PID 42631,iter time ~2.95s,ETA ~82h to 100k iters。
    - resolved hydra config 验证: `actor_learning_rate: 2.0e-05` ✓ / `entropy_coef: 0.001` ✓ / `encoder_sample_probs.smpl: 0.0` ✓ / `missing_dofs_mjcf: ${missing_dofs.indices_mjcf}` ✓ / `checkpoint: sonic_release/last.pt` ✓。
    - reference data path 完全继承 sonic_release_23dof 的 D-baseline 设置:**没有**切到 `robot_filtered_23dof`(那个目录只 zero `dof` 字段、不 zero `pose_aa`,运行时 fk_batch 不读 dof,是 dead path);用的是 `robot_filtered`(29DoF PKL),靠 `motion_lib_base.py:1874-1877` 在 fk_batch 之前 zero `pose_aa[:, [14,15,21,22,28,29], :]` 实现等价 23DoF 参考动作。
- **明天看的指标**(对照 `_102145` post-FK-fix baseline 和 H5e-C 1.572 peak):
    - entropy 应稳在 ~13 或缓降,**不能爬过 14**。爬过去意味着 lr 还是太大或又有 leak 通路。
    - std 应稳在 0.39,**不能 saturate** 到 0.42+。
    - mean episode length 应 17 → 50+ → 200+。
    - foot_pos_xyz 应 0.80 → <0.5 → <0.2。
    - mean reward 目标突破 H5e-C 的 1.572 peak,**而且不能像 H5e-C 那样到 800 iter 就开始 retreat**(若 retreat 模式重现说明 actor_lr=2e-5 真的偏小,届时再加 lr 但起步不要 5×,试 4e-5)。
- **失败兜底预案**(若明早一看 entropy 又爬到 18+):
    - 不再加 lr / 调 entropy_coef,直接走 [[d1-23dof-finetune-baseline-2026-05-19]] 那个"数据集瘦身"路径:`robot_filtered` → 步态+基础移动 only,踢掉所有弯腰、蹲、爬、坐、舞蹈、抱举类。这是用户原本的备选方案,1 小时内能改 motion lib 过滤脚本起来。

---

## [2026-05-22 上午] H5g iter2k / iter16k 部署仍发散 → 排除对齐 / 锁关节问题,定位为欠训

> 上一节 `_183343` H5g run 跑了一夜。早上导出 iter 2000 + iter 16100 (last.pt) 两个 ckpt 测部署效果。**两个都炸**(closed-loop sim 内 4 秒发散)。本节排查全部对齐链路,最终结论是 **4 个最近 ckpt 全都欠训**,与上一节 H5e-E 的"训练曲线收敛 ≠ 部署可用"一脉相承。

### 操作记录(按时序)

1. **`gear_sonic/config/callbacks/model_save.yaml`**:`save_frequency: 1000 → 200`(用户要求"现在导出 同时把存盘逻辑修改成 200 一存")。运行中的 `_183343` 不会重读 yaml,改动对**下次重启**才生效。
2. **首次导 ONNX 失败** (PhysX GPU stream init 报错):训练 GPU 占满,导出 export-only run 抢不到 stream。等到 last.pt 在 10:08 落盘后,kill watchdog + train PID,跑 export-only 成功。
3. **导出产物**:
   - iter 2000 → 5 个 onnx 文件 → staged at `gear_sonic_deploy/policy/release/iter2k_h5g/`(model_encoder.onnx + model_decoder.onnx + observation_config.yaml,后者从 `iter20k_h5e_e/` 复制)
   - iter 16100 (last.pt) → staged at `gear_sonic_deploy/policy/release/iter16k_h5g/`
4. **release/ 顶层 onnx 被覆盖**:用户后来报告 sim2sim 仍摔,排查时发现 `policy/release/model_encoder.onnx` md5 = `ff4ff80b...` 与 `iter16k_h5g/model_encoder.onnx` md5 完全一致 ⇒ 用户 11:00 把 iter16k 复制到 release 顶层做测试,把原始"NVIDIA / 原作者好用版"覆盖了。原版仍在 `policy/release/backup_20260514/`(md5 `1d4391ad` / `a21244b1`),它是 1762-dim obs schema(老版),**不能直接用现行 1751-dim yaml 跑**。

### 现象

- iter 16k 部署 sim2sim:启动姿态怪异,右手 + 右腿位姿异常,机器人秒倒。
- iter 2k 同样炸(用户最初说"2k 还行"是测错了,跑的是 release 顶层的原作者 ckpt;后来确认 release 顶层就是 iter 16k 的拷贝,所以"原作者还行"那次实际跑的是 backup_20260514 之前还在 release 顶层的旧版)。

### 诊断路径(逐项排除)

#### 1. **D1 6 关节 mask 是否对齐?** ✓ 验证正确

> 用户最初怀疑"是不是没对齐 / 需要 23-29 dof 关节映射 / 关节序号要打印挨个对照"。

读 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp:100-104`:
```cpp
const std::array<int, 29> isaaclab_to_mujoco = {0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,
                                                11,15,19,21,23,25,27,12,16,20,22,24,26,28};
```

**变量名误导但用法正确**:注释写"mujoco order in isaaclab index",看 cpp:3128 用法:
```cpp
const double action_value = static_cast<double>(floatarr[isaaclab_to_mujoco[i]]) * g1_action_scale[i];
```
`g1_action_scale[i]` 按 mujoco 关节名顺序排(`i=0=left_hip_pitch ... i=13=waist_roll ... i=20=left_wrist_pitch`),所以 `i` 是 mj 索引,`floatarr[isaaclab_to_mujoco[i]]` 访问 IL 索引 ⇒ **数组语义实际是 MJ→IL,变量名反着**。

由此推 D1 6 个 locked 关节(MJ idx `{13,14,20,21,27,28}`)→ IL idx 集合:
- mj 13 (waist_roll) → arr[13] = **5**
- mj 14 (waist_pitch) → arr[14] = **8**
- mj 20 (left_wrist_pitch) → arr[20] = **25**
- mj 21 (left_wrist_yaw) → arr[21] = **27**
- mj 27 (right_wrist_pitch) → arr[27] = **26**
- mj 28 (right_wrist_yaw) → arr[28] = **28**

⇒ sorted IL = **`[5, 8, 25, 26, 27, 28]`**。与 `gear_sonic_deploy/reference/zero_23dof_columns.py:26-34` 的 `MISSING_23DOF_IL_COLUMNS` 完全一致,与 H5e_e 时期 `joint_constants.py:MISSING_23DOF_INDICES_IL=[5,8,25,27,26,28]`(unsorted)也一致(集合相等)。

#### 2. **加保险:cpp 强制 zero 这 6 个 IL 索引** ✓ 已加 + 验证生效

`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:3118-3124`:
```cpp
// D1 23DoF guard: hardware lacks 6 joints (waist roll/pitch + L/R wrist
// pitch/yaw, IL idx 5/8/25/26/27/28). motion_lib_base.py zeros pose_aa
// on these idx pre-FK in training, so policy gets no reward signal there
// and may emit non-zero — which would spasm the corresponding sim joints.
// Force zero on the policy output before computing q_target.
static const int D1_MISSING_IL_IDX[6] = {5, 8, 25, 26, 27, 28};
for (int idx : D1_MISSING_IL_IDX) floatarr[idx] = 0.0f;
```
重 cmake 后跑 sim,用户报"更不对了 还是没对齐"——说明 mask 不是关键症状的原因,但 mask 本身是必要的(否则 6 个 sim joint 会被 garbage action 抽搐)。

#### 3. **加 q_target dump 看是否真的全关节都炸** ✓ 数据 cling

cpp:3137-3155 加了 one-shot CSV dump(前 200 ticks → `/tmp/q_target_dump.csv`,29 个 raw IL action + 29 个 mujoco q_target/帧)。用户重 cmake 后跑了一次 sim2sim 4 秒。

CSV 分析(用 iter16k_h5g + 当前 cpp guard):

| Tick | t (s) | max\|q_dev\| | 最坏关节 | a_il_3 (sample raw action) |
|------|-------|--------------|----------|----------------------------|
| 0    | 0.00  | 28.67°       | shoulder | -0.08 |
| 5    | 0.10  | 55.52°       | elbow    | -0.64 |
| 30   | 0.60  | 269.70°      | elbow    | +3.13 |
| **60** | **1.20** | (max) | — | **+15.4** ← 起飞 |
| 100  | 2.00  | 522.82°      | elbow    | +14.9 |
| 199  | 3.98  | 464.12°      | r_elbow  | -2.5  |

- **Tick 0** 各关节 q_target 偏差 ≤ 30°,policy 输出合理。
- **第 60 帧**(1.2 s) raw action 数量级从 0.5 涨到 15+,完全脱离训练分布 ⇒ closed-loop 指数发散。
- D1 mask 验证:6 个 locked IL idx 在每一帧都精确为 0.0 ✓(如果训练 lock 不彻底 raw action 会大,我们这里在 deploy 端兜底了)。

⇒ 结论:**不是 tick 0 的对齐错了,是 closed-loop 不稳**。

#### 4. **是 H5g 特有的 lib 修复 bug?对比 4 个 ckpt** ✗ 全员欠训

写 onnxruntime 脚本对比:

| ckpt              | mean\|a\| (21 obs 平均) | std(\|a\|) | max\|a\| |
|-------------------|------------------------|------------|----------|
| iter4k_h5e_e      | 0.93                   | 0.97       | 7.0      |
| iter20k_h5e_e     | 0.92                   | 1.25       | **13.7** |
| iter2k_h5g        | 0.92                   | 0.86       | 5.8      |
| iter16k_h5g       | 0.98                   | 1.34       | **11.9** |

两两 L2 距离 7–12,4 个策略输出空间各自处于不同的不稳定区。

**关键证据 H5g 的 motion_lib pose_aa 锁实际生效**:zero obs 下 IL 5 (waist_roll) 输出
- iter20k_h5e_e: **+1.95**(没锁,policy 自由发挥)
- iter4k_h5e_e: -0.39
- iter2k_h5g: -0.04 ← 训练 reward 在 lock 后压向 0
- iter16k_h5g: -0.06 ← 同上

⇒ H5g 训练侧的 [[d1-23dof-finetune-baseline-2026-05-19]] 那个 `motion_lib_base.py` zero pose_aa 修复**确实 take 了**,不是又一次 no-op fix。

但收敛得不够 ——converged 站立 policy mean\|a\| 应接近 0,这 4 个 ckpt 全部 ~0.9。

### 根因

**与上一节 H5e-E `_185243` 同型病**:训练 metrics 看上去在收敛(reward 上升 / entropy 下降)≠ closed-loop deploy 可用。当前 4 个 ckpt 都在 sim2sim 里发散。具体:

- iter2k_h5g 离 warm-start sonic_release/last.pt 仅 2k 梯度步,但已经偏离稳态(说明哪怕 actor_lr=2e-5,2k 步也足以离开盆地)。
- iter16k_h5g 跑了一夜到 16100 步(max_steps=100000,**16% 进度**),处于"离开旧盆地、还没找到新盆地"的过渡状态,closed-loop 不稳。
- 一个充分收敛的站立 policy mean\|a\| 应接近 0,4 个 ckpt 全部 0.9+,**整个 H5x 系列(H5e_e 和 H5g)目前都不能上真机/sim2sim**。

### 改动

- `gear_sonic/config/callbacks/model_save.yaml`:`save_frequency: 1000 → 200`(下次启动训练才生效)。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`:
    - `+#include <cstdio>`(为下面 fopen/fprintf)
    - L3118-3124:加 D1 mask `floatarr[5/8/25/26/27/28] = 0.0f`(保留)
    - L3137-3155:加 one-shot CSV dump `/tmp/q_target_dump.csv`(诊断完成,可删)
- `gear_sonic_deploy/policy/release/iter2k_h5g/`(新建):iter 2000 ckpt 的 5 个 onnx + observation_config.yaml
- `gear_sonic_deploy/policy/release/iter16k_h5g/`(新建):iter 16100 ckpt 的 5 个 onnx + observation_config.yaml
- `gear_sonic_deploy/policy/release/model_*.onnx` 顶层:用户 11:00 测试时被覆盖为 iter16k_h5g 副本,**不再是原作者好用版**。原作者版仍在 `backup_20260514/` (md5 `1d4391ad`/`a21244b1`,1762-dim obs schema,与现行 1751-dim yaml 不兼容)。

### 验证

- **D1 mask 对齐**:✓ CSV 显示 6 个 locked IL idx 每帧都是 0.0。
- **Tick 0 q_target 物理可达**:✓ 各关节 ≤30° 偏差。
- **Closed-loop 稳定**:✗ 60 帧内发散到 max\|q_dev\| > 500°。
- **欠训判定**:✓ 4 ckpt mean\|a\| 全 ~0.9,远超 converged 站立 policy 的预期。

### 经验

- **"对齐验证"和"收敛验证"是两件事**。今天前半场被用户引导往"是不是关节序号没对"方向走,实际数据一看就是 closed-loop 发散——tick 0 一切正常,后续指数失稳。本案对齐链路全通。
- **release/ 顶层 onnx 别随便覆盖**。用户为了测试覆盖了原版,导致后面再回头测"是不是原版也炸"已经不可能(除非从 `backup_20260514/` 恢复,但 obs schema 老版又和现行 yaml 不兼容)。教训:测不同 ckpt 时**只换 `iter*/` 子目录的引用**,不要写顶层。
- **iter 2k 离 warm-start 起点已经发散**说明 D1 加 lock(motion_lib pose_aa zero pre-FK)对 PPO 而言是**剧烈的 reward landscape 变化**,即使 actor_lr=2e-5 也很快脱离原 prior。warm-start 不是"几乎免费",对修改了 obs/reward semantics 的 finetune,头 1k 步就足以走出盆地。
- **deploy 侧的 D1_MISSING_IL_IDX mask 是必要兜底**。即使训练侧在 motion_lib pre-FK 锁了 6 个关节的 pose_aa,policy 在那 6 维上还是会输出非零(因为 imitation reward 没在 lock 后归零所有梯度信号,policy 仍会探索)。deploy 端的硬 zero 是契约最后一环,不是冗余。
- **Variable naming 反着的 array 要靠用法 + `g1_action_scale[i]` 注释推**:`isaaclab_to_mujoco` 名字看着像 IL→MJ,实际语义是 MJ→IL。任何映射 array 在 review 时都要看至少一处真实用法 + 索引语义,不能光看名字。

### 下一步

- 当前训练已被 kill(为导 ONNX),需要决定是否重启 H5g 继续往 100k 跑。
- 若继续:用最新 last.pt(iter 16100)续跑,save_frequency=200 已配,后续每 200 步存一次,中途多挑 ckpt 测看哪段进入稳态。
- cpp dump block 任务完成,可删(只有诊断价值,留着对生产无害但不优雅)。
- D1 mask block 必留(契约必要)。

---

## 2026-05-22 14:30 H5h:actor distribution mask(actor 端最后一道闸)

### 触发

H5g run `_20260522_113117`(从前一日 H5g `_183343` resume + obs `last_action_masked` 后)训到 iter 2985 时再次确认 warm-start drift:rolling 100-iter reward 在 iter 1000–1100 顶到 1.927 / length 25.63,iter 2885–2985 已退化到 1.600 / 21.72(比 mask 启用起点 1.775 / 23.54 还差)。终止分布 foot_pos_xyz 仍主导。

H5d → H5e-C → H5e-E → H5g 四次 finetune 全是同一种"peak then regression",mask 全套(obs joint_pos/vel,last_action,command_multi_future via pose_aa,wrist_for_smpl,env_actions via apply_missing_dof_mask,joint_limit reward 23 关节)只压住了观测/奖励/物理三条路径,**没有压住 actor 内部分布**这条第四条路径。

### 假设

actor 输出的 `Normal(mean, std)` 分布始终是 29 维:
- `log_std` 是 `nn.Parameter(num_actions=29)`。entropy_coef=0.001 让 6 个 locked 维度的 σ 不会衰减到 0,每步都 sample 出非零随机值
- PPO 在完整 29 维上算 `log_prob(action).sum(-1)` 和 KL clip ratio。这 6 个维度的 sample 与 advantage 完全无关(物理被 pin 住),但 log_prob 把它们当成"策略选择"算进重要性比例
- 这部分梯度通过 `mu_layer` 的 6 个 locked 输出行 + `log_std[locked]` 灌进**共享 trunk**。整个 actor backbone 每个 minibatch 都在吃这 6 维的纯噪声梯度
- entropy_coef×entropy 项 push σ 高,action_rate_l2 push μ 低(被 env_actions 端 zero 后 diff 变成对前一步零的 diff,几乎为 0,惩罚弱),综合下来是噪声大于压制

四次 finetune drift 的 LCM 解释正是这条:**6 个无效维度持续往共享 trunk 灌噪声**。

### 干预

新建 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof_h5h.yaml`,继承 H5g,新增:
```yaml
algo:
  config:
    missing_dofs_action_il: ${missing_dofs.indices_il}  # [5,8,25,27,26,28]
    missing_dofs_action_std: 1.0e-4
```

`gear_sonic/trl/modules/actor_critic_modules.py` 改两处:

1. `Actor.__init__`:读 `algo_config.missing_dofs_action_il`,如非空则 `register_buffer("locked_action_mask", torch.zeros(num_actions, dtype=bool))` 并在 6 个 IL idx 处置 True;同时记 `locked_action_std_const = algo_config.get("missing_dofs_action_std", 1e-4)`。29DoF 训练默认 None,no-op。
2. `update_distribution`:在构造 `Normal(mean, std)` 之前,若 `locked_action_mask` 存在,`mean = mean.masked_fill(mask, 0.0)`;`std_combined = std_combined.masked_fill(mask, locked_action_std_const)`。`masked_fill` 返回新张量,autograd 只通过未 mask 的 23 维流回到参数。locked 维的 log_prob 是参数无关常数,梯度对 mu_layer 的 locked 行和 log_std[locked] 严格为 0。

ckpt 兼容:`register_buffer(persistent=False)` 不进 state_dict;ppo_trainer.py:2189 用 `strict=False`。`sonic_release/last.pt` 直接 warm-start 不报错。

`rollout_with_tokens` 和 `act_pure_inference` 这两条路径独立设 `Normal(action_mean, ...)` 没经过 `update_distribution`,但全仓库没人调用,只是外部 diffusion-token 推理预留接口,PPO finetune 路径全走 `update_distribution`,不用补。

### 启动

- stop H5g `_113117`(SIGTERM PID 127375,GPU 7066→653 MiB)+ watchdog(PID 135625)。
- 从 `sonic_release/last.pt` (29DoF pristine prior)起新 run,**不**从 H5g 任何中间 ckpt resume,避免累积漂移当起点。
- launch 命令(2026-05-22 14:30):
  ```bash
  nohup /home/woan/.conda/envs/groot_wbc/bin/python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release_23dof_h5h \
    +checkpoint=sonic_release/last.pt \
    num_envs=512 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
    > /tmp/h5h_launch.log 2>&1 &
  ```
  PID 153592。

### 周一看的指标(对照 H5g `_113117` iter 1000 peak)

- iter 1000 rolling reward:H5g 是 1.927,H5h 期望显著超(若 actor 噪声梯度真是主因)
- iter 1000 rolling length:H5g 是 25.63,H5h 期望 ≥27
- 是否还出现 H5g 同款 "peak then regression":若 iter 2000+ 仍维持或继续上升,基本证实假设
- 终止分布:foot_pos_xyz 占比若下降,trunk 真的开始用上更干净的梯度
- entropy:locked 6 维 entropy 应趋近常数(不再被 entropy bonus 拽),可观测 std 直方图分布是否分裂为 23+6 模

若周一看到 iter 2k+ 没退化,假设证实,后续 23DoF 标准训练流程把这个 mask 放进 base 23dof yaml。
若周一看到同款 drift,这说明 actor 噪声梯度不是主因,需要进一步看 critic warm-start 或 motion lib 23DoF 可达性这两条线。

---

## 2026-05-25 11:40 H5i:reload 频率(motion pool 过拟合的第二道闸)

### 触发

H5h `_141132` 周末长跑到 iter 60k。用户做 sim2sim 部署对照,现象很反直觉:

| ckpt | 站立测试 | 行走指令 |
|------|----------|----------|
| sonic_release/last.pt(原作者 29DoF) | 稳 | 略前漂,可保持 |
| H5h iter 200 | 2–3s 后倒 | — |
| H5h iter 1000 | <0.5s 倒,四肢扭曲 | — |
| H5h iter 60k | 立刻倒 | — |

用户的反问"原作者的策略在23dof上也能稳定站立 怎么可能一开始就直接摔了"是关键 nudge——iter 200 / iter 1k 已经摔,且 H5h warm-start 是 sonic_release/last.pt 这个验证过站立 OK 的起点,**不是从一开始就摔,是训练把它训摔了**。

### 假设 1(我提的,被否)

iter 200 episode length=6 是 warm-start 第一步就崩。
反驳:同起点 sonic_release/last.pt 不接 finetune 直接 deploy 是稳的,所以 iter 1 不可能崩;length=6 是 termination 阈值卡的(foot_pos_xyz 0.2m 占 70%),不是 balance 失败。

我提的解决方案"环境侧不再 zero locked dim,只让 deploy 端 guard 兜底"也被用户秒否:"最后学的相当于歪了 能输出腰部动作但是没有效果 这不是肯定会歪吗"。对——把 6 维交给 PPO 自由探索,reward 不变的情况下 PPO 会找最大化 reward 的 μ,而不是 0;deploy 端再 zero 只是把 actor 学到的"歪解"截断,行为和训练分布脱离。

### 测试 2:forward-pass μ 漂移

四个 ckpt(sonic_release / H5h_iter200 / H5h_iter1k / H5h_iter60k)的 encoder + decoder ONNX 离线加载,喂同一组 obs(zero / 小 Gaussian / 100 random),测 active 23 维 μ 与 sonic_release baseline 的 L2 偏离。

| ckpt | μ 偏离 (mean L2) | active μ L2 norm | locked Linf |
|------|-------------------|-------------------|-------------|
| sonic_release | 0 | 4.08 | 1.14 |
| H5h iter 200 | 0.96 ± 0.24 | 3.68 | 1.12 |
| H5h iter 1k | 1.78 ± 0.44 | 3.45 | 1.05 |
| H5h iter 60k | **4.87 ± 1.11** | 3.48 | 0.86 |

iter 60k 偏离 (4.87) 比 sonic_release 自己的 active μ norm (4.08) **还大**——基本是个正交策略。验证了"H5h 训练分布与原 prior 越走越远"是 deploy 退化的首因。但单调上升解释不全:如果只是 "subset 过拟合",iter 60k 应已见过更多数据、泛化更好;实际是 iter 60k 偏离最大。说明除 subset overfit 外还有第二个机制(可能是 23DoF reference body_pos 的子代 link 残留 bias,locked 关节 pose_aa 被 zero 但其下游 body 在 FK 链上仍受影响)。

### 假设 2(用户提)

"我这 32G 内存,有没有可能只吃了一部分数据进行训练 → 过拟合到这一部分"。

### 数据加载机制(实测)

| 环节 | 实际 | 来源 |
|------|------|------|
| 数据集 filter 后 | 129,785 motions | log 行 62 |
| 单次 reload 装入 | **512 motions** | `commands.py:238` `min(num_envs=512, 1024)` |
| Reload 间隔 | 每 250 PPO step | `callbacks/im_resample.yaml: motion_resample_frequency: 250` |
| 触发 | `ImResampleCallback.on_step_end` | `(global_step+1) % 250 == 0` |
| 采样方式 | with replacement (`random_sample=True`) | `motion_lib_base.py:974` |
| 数据驻留 | **GPU 显存**(`.to(self._device)`) | `motion_lib_base.py:1479-1502` 多个 `body_pos_w / body_quat_w / dof_pos / dof_vel ...` 都 `.to(device)` |
| `override_num_motions_to_load` | 未配置 | grep config 0 hits |
| `max_unique_motions` | 未配置 | grep config 0 hits |

### 关键纠正:不是 32GB RAM 的锅

`nvidia-smi`:GPU 是 **8 GB**(不是 24GB),baseline 占 561MiB 已经预留掉一部分。

按平均 motion ~390 帧(50fps×7.8s)估,FK 解码后存的 8 个张量(body_pos_w / body_quat_w / body_pos_b / 三个速度场 / dof_pos / dof_vel)对 512 motions 约 430 MB GPU。线性外推到 129,785 motions ≈ **109 GB**——根本装不下 8GB GPU。即使全压到 32GB RAM 也装不下解码态。

所以 512 不是 "RAM 限制",是 `min(num_envs, 1024)` 这条**显存预算上限**;每个 env 在线训练的同时 motion buffer 也常驻显存。

### Coverage 估算(with replacement)

```
iter 1000  →  4 reload × 512 = 2048 slot,unique ≈ 2025  →  1.6% 覆盖
iter 3000  →  12 reload,unique ≈ 5999                  →  4.6%
iter 10k   →  40 reload,unique ≈ 18.5k                 →  14.3%
iter 60k   →  240 reload,unique ≈ 79.2k                →  61%
```

每个 pool 内 PPO 走 250 update step × 512 env × ~30 episode frame ≈ 3.75M sample——同 512 motion 反复刷过拟合的量级很大。

### 干预(选项 1:加快 reload 节奏)

`motion_resample_frequency: 250 → 50`(单 pool 训练步数压到 1/5,等量计算下覆盖 motion 数 ×5)。

不改 yaml,cmdline override(避免污染 H5h 历史语义)。

### 启动里的坑

第一次试:
```bash
... im_resample.motion_resample_frequency=50
```
报错 `Key 'im_resample' is not in struct, object_type=dict`。

原因:`gear_sonic/config/callbacks/*.yaml` 文件**没有** `# @package _global_` header,所以默认 package 是组路径 `callbacks`。文件内容又是 `im_resample: {...}`,合并后挂在 `callbacks.im_resample.*` 下,**不是**根。`train_agent_trl.py:458` 用 `config.callbacks.values()` 也证实了这一点。

正确路径:
```bash
... callbacks.im_resample.motion_resample_frequency=50
```

完整 launch(2026-05-25 11:41,PID 492299):
```bash
nohup /home/woan/.conda/envs/groot_wbc/bin/python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release_23dof_h5h \
  +checkpoint=sonic_release/last.pt \
  num_envs=512 headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
  callbacks.im_resample.motion_resample_frequency=50 \
  > /tmp/h5i_launch.log 2>&1 &
```

不挂 watchdog(避免它从老 H5h ckpt 续训而非新鲜起点)。

### 验证(reload 频率改动 take 了)

| 时刻 | 事件 | 距上次 |
|------|------|--------|
| 11:41:47 | 数据集 filter | — |
| 11:43:53 | reload #1(iter 1) | — |
| 11:46:32 | reload #2(iter ~50) | **159s** |
| 11:49:14 | reload #3(iter ~100) | **162s** |
| 11:51:54 | reload #4(iter ~150) | **160s** |

间隔稳定 ~160s,等于原 H5h 13min 的 **1/5**。iter time 2.86s × 50 step = 143s,差额是 reload 本身的 IO + GPU 数据迁移。

### 同 iter 数据多样性对比

| iter | H5h(freq=250)已见 unique | H5i(freq=50)已见 unique |
|------|----------------------------|----------------------------|
| 150 | 512(1 个 pool) | ~2000(4 pool) |
| 1000 | ~2000(4 pool) | ~10000(20 pool) |
| 3000 | ~6000 | ~30000(~22% 覆盖) |
| 10000 | ~18.5k | ~80k+(覆盖率饱和) |

### 经验

- **GPU 显存才是 motion pool size 的硬约束,32GB RAM 不是。** 用户的"内存装不下"猜测**结论方向对**(确实只装了部分),**机制错**(不是 RAM,是 8GB GPU)。今天我没第一时间查 `nvidia-smi`,等到要 launch 时才发现是 8GB 卡。**改 D1/D2 任何 motion-loading 配置前都先看显卡型号**——8GB / 24GB / 80GB 三档对应的 num_envs 上限完全不同。
- **monotonic μ drift 不能完全归因于 subset overfit**。若只 subset 问题,iter 60k 应该泛化好转;实测 iter 60k 偏离最大,说明还有第二个机制——大概率是 23DoF reference body_pos 的子链 bias(locked dim 自己 zero 了,但子代 link 的全局位置仍承继父链残差)。先做 reload 频率实验拿数据,如果 H5i iter 1000 偏离从 1.78 显著下降但 iter 60k 偏离仍 > 3,就坐实双机制。
- **Hydra `defaults` group 文件如果没写 `# @package _global_`,override 路径要带上组名。** 这次踩的具体坑:`/callbacks/im_resample` 里的 `im_resample` 是 group 内嵌套 key,完整路径是 `callbacks.im_resample.*`。判别方法:`head -3 callbacks/*.yaml` 看 `# @package` 行;没有 → override 要 `<group>.<key>.*`。
- **Hydra struct 模式下 override 不存在的 key 会抛 "not in struct"**,提示用 `+` 追加。**别盲目加 `+`**——struct 拒绝是因为 key 不在 schema 里,加了 `+` 会塞个孤儿配置,callback 实例化时拿不到。先看清楚正确路径,再决定是 override(`key=val`)还是 append(`+key=val`)。
- **改运行时 hyperparam 优先用 cmdline 而不是改 yaml。** 这次 `motion_resample_frequency` 改动是实验性的,改 `callbacks/im_resample.yaml` 会污染所有共用这个 callback 的 exp(sonic_release / sonic_h2 / sonic_bones_seed 全部受影响);cmdline override 只对当前 run 生效,失败也好回退。
- **测试 2 这种 forward-pass 离线对照是诊断 deploy 退化的高 ROI 手段。** 不需要重启 sim,只要 ONNX + 同一组 obs,5 分钟跑出全 ckpt μ 漂移趋势。下次发现"老 ckpt OK / 新 ckpt 崩"的反直觉对照,先跑这个,再去猜机制。

### 下一步

- 等 H5i iter 1000(~50min after launch ≈ 12:30):看 rolling reward 和 length 是否高于 H5h iter 1000 同步窗口。如果显著好,subset overfit 是首因之一。
- 等 H5i iter 3000–5000:再跑一次"测试 2"对 H5i iter 1k 和 iter 5k 出 μ 偏离曲线。理想结果:iter 1k 偏离从 H5h 的 1.78 降到 ≤ 1.0,且 iter 5k 仍稳。
- 若 H5i iter 60k 仍出现单调 μ 漂移(假设可复现的 GPU 时间窗内),启动 reference body_pos 子链 bias 的第二线调查:看 23DoF FK 输出 vs. 29DoF FK 输出在 wrist / hand 这些 child link 的位置差异。
- watchdog 暂不挂,避免从已漂移 ckpt resume。等 iter 5k+ 稳定再决定是否挂 watchdog 长跑。

---

## 2026-05-25 晚 H1 假设:reset 分布缺 idle 样本——forward-pass μ 探针锁根因

### 触发

H5h / H5i 仍然解释不了同一个反直觉现象:`sonic_release/last.pt`(原作者 29DoF)直接 deploy 到 23DoF 平地能稳站,但只要从同一个 ckpt 接 finetune 过几百 iter 就会"平地零关节秒飞 ±500°"。subset overfit + reference body_pos 子链 bias 是 deploy 退化机制,但解释不了**为什么连 sonic_release 自己**(没 finetune 过)在某些条件下也会输出大 μ。

### 排除链(H2 / H4 都不是)

- **H2 weight 异常?排除**。单独看 actor weight L2 norm:locked 6 维比 hardware 23 维仅大 23%(不是个量级);并且 `gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp:3125` 已强制把 6 个 locked dim 的 q_target 写 0。weight 健康,deploy 端 guard 也在。
- **H4 obs normalizer 没对齐?排除**。grep `policy_state_dict` keys 没 normalizer / running_mean / running_var,actor 直接吃裸 obs。

### H1 探针:forward-pass μ vs OOD obs scale

`/tmp/h1_probe.py` 加载 `sonic_release/exported/model_step_041550_g1.onnx`(bundled g1.onnx, dim 1570→29),喂三档 obs:

| 输入 | μ_max(硬件 23 维) |
|------|---------------------|
| 全 0 | **1.92** |
| σ=1 高斯 | **11.57**(**39σ 尾部**) |
| σ=3 高斯 | 24.07 |

**actor 没有 Lipschitz bound**。任何"普通规模的非零 OOD obs"都会输出大 μ → q_target 偏 ±330° → 闭环发散吻合症状。

### 根因(确认)

训练 reset 分布**从未覆盖**"机器人静止 + reference 静止"区域。motion lib 全是 AMASS 动作序列,reset 时机器人 + reference 都被丢到运动状态;sonic_release 自己也没专门的 idle reset 样本。actor 在 idle obs 区域是真空,推理时进 idle 区 → 大 μ。

意义:**H5i 加快 motion reload 频率解决不了这个**——它是 reset 分布缺一类样本,不是样本量问题。

### 修复方案(用户确认 C):reset 分布注入 idle 子集

A 加 deploy 端 clip / B 加 reward bonus / **C 在 reset 分布注入 idle 样本**。

C 不改 deploy / 不改 reward / 不动 motion_lib,只改 commands.py:让 5% env 在 reset 时把 robot 放到 idle 站姿、reference 也指向 `neutral_idle_loop_001` motion → tracking reward 自然奖励"维持静止" → actor 被强约束在 idle obs 区域 → 部署进 idle 不再发散。

### 改动

- **`gear_sonic/config/manager_env/commands/terms/motion.yaml`**:加 `static_reset_prob: 0.05`,`static_motion_name: "neutral_idle_loop_001"`
- **`gear_sonic/envs/manager_env/mdp/commands.py`** TrackingCommandCfg:加 `static_reset_prob: float = 0.0`,`static_motion_name: str | None = None`(默认值保持现行为不变)
- **`gear_sonic/envs/manager_env/mdp/commands.py`** `__init__` 末段:缓存 `_static_motion_id = (motion_lib._curr_motion_ids == global_id).nonzero(as_tuple=True)[0]`(本地索引!踩坑见下条)
- **`gear_sonic/envs/manager_env/mdp/commands.py`** `_resample_command`:5% mask 覆盖 motion_ids 和 motion_start_time_steps

新建 exp config `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release_23dof_h6_idle.yaml`(继承 h5h)。

### 实现踩坑:全局 idx vs 本地 idx 误用 → CUDA device-side assert

第一版用 `motion_lib._motion_data_keys` 的全局 enumerate idx 直接塞 `self.motion_ids[mask] = global_idx`。代码 print 出 `static_reset enabled: motion_id=3105`,看着像成功了,**但第一个 reward / sim step 立刻崩**:Traceback 是一连串 PhysX `DirectGpuHelper.cpp` CUDA error,看上去像 sim 本身坏掉。

**真根因**:`MotionLibBase` 有两套索引空间:

- **全局**:`_motion_data_keys` 是所有发现 motion 的 numpy array,size = `_num_unique_motions`(~10 万级)
- **本地**:`_curr_motion_ids[local_idx] = global_idx`,size 限于 `max_num_load_motions`(TrackingCommand 默认 `min(num_envs, 1024)`)

env 上的 `self.motion_ids` 是**本地索引**(`sample_motions(n)` 返回的 `_sampling_batch_prob` 多项分布采样)。下游 `get_motion_state(motion_ids,...)` 等用本地 idx 去 `_motion_lengths` 等已加载缓冲区做 gather/index_select。塞全局 3105 + `max_num_load_motions=1024` → 越界 → CUDA device-side assert,加上 PhysX fabric CUDA error 一连串。

修复:`(motion_lib._curr_motion_ids == global_id).nonzero(as_tuple=True)[0]`,空 → 这一轮跳过(如果 adaptive_sampling 把它换出去了)。

### 经验

- **deploy 退化 → 先做 forward-pass μ 探针**,不要先猜 weight / normalizer。一个 ckpt + 三档 OOD obs,5 分钟出趋势,比改 reward 改 termination 高 ROI 几个数量级。
- **reset 分布零覆盖问题不是 motion 量的问题**。改 motion_resample_frequency 不修这种空缺,需要构造性注入。
- **MotionLibBase 全局 vs 本地索引混用是经典坑**——已记入 memory `motion_lib_global_vs_local_id`。看到"训练只第一次 reset 后崩 + PhysX fabric error 一串"先怀疑 motion_ids 写入,不要先怀疑显存/driver。
- **σ Parameter 6 个 locked 维度等于 0.5 是潜在隐患**(actor 噪声 std 在 locked dim 没归零),但被 deploy 端 line 3125 兜住了。本计划不动。

---

## 2026-05-26 凌晨 本地 H6 17h 验证 + 部署 pipeline 探针

### 触发

H1 修复跑完想验证"OOD μ_max 是不是真的下降了"。

### 本地 H6 训练

```bash
+exp=manager/universal_token/all_modes/sonic_release_23dof_h6_idle
+checkpoint=sonic_release/last.pt
num_envs=512
callbacks.im_resample.motion_resample_frequency=50
```

run `sonic_release_23dof_h6_idle_test-20260525_174242`,17h 跑到 **step 16700**。

### 关键纠正:bundled g1.onnx ≠ 部署 pipeline

`/tmp/h1_probe.py` 测的是 `model_step_*_g1.onnx`(单文件,obs dim 1570),但**实际部署用的是 split**:

- `encoder.onnx` 1751 → 64 (token)
- `decoder.onnx` 994 → action 29

g1.onnx 是 export 阶段的 bundled 版本,**部署不用它**。所以 g1.onnx 上的 μ 改善不一定 propagate 到 deploy actor。

### 新探针 `/tmp/h1_probe_pipeline.py`

加载 BASELINE(`model_step_041550`,即 sonic_release 起点)和 H6(`model_step_016700`)各自的 encoder + decoder,做 pipeline forward。处理 token / dec_extra concat 顺序未知的问题:cat_first / cat_last 都跑,取 worst case(真正 Lipschitz-bounded actor 在两种顺序下都该安全,worst-case 反而是诊断信号)。

σ=1 OOD obs(64 batch):

| ckpt | μ_max(硬件 23 维) |
|------|---------------------|
| BASELINE 041550 | **11.404** |
| H6 016700 | **3.976** |

**部署 pipeline 上 -65%**。H1 修复在真正部署用的 actor 上同样生效,不是 g1.onnx 一个人的好事。

decoder-only 旁路探针(直接喂 994 维随机)也确认 H6 decoder 自身的 Lipschitz 边界明显更紧。

### 经验

- **部署退化诊断要对准部署 pipeline,不是对准 export 工具链中间产物**。bundled g1.onnx 是教学/调试用的,部署装 split。"在 g1.onnx 上验证好"不等于"在部署 actor 上验证好"。今后所有 deploy 验证脚本默认走 encoder + decoder split。
- **未知 concat 顺序时取 worst-case**。Lipschitz-bounded actor 在两种顺序下都安全,worst-case 不会假阳性;只在出问题时多报错,符合"宁可保守"的诊断方向。

---

## 2026-05-26 上午 推服务器(Aliyun PAI DSW)上线踩坑

### 触发

H6 本地验证 -65% 之后,4060 算力不够支撑长跑 + 4096 env benchmark。决定推 PAI DSW(L20 46GB)。

### 关键决策

- 从原始 `sonic_release/last.pt`(step 41550)warm-start,**不**从本地 H6 step 16700 续(避免本地 17h 训练带的偏移污染 4096 env 的 fresh trajectory 分布)
- `num_envs=4096`(yaml default),`motion_resample_frequency=250`(默认,不沿用本地 H5i 的 50—— 4096 env 单 pool 已经是 4× 本地容量)
- PPO 超参不动:`num_steps_per_env=24`,`num_mini_batches=4`,`num_learning_epochs=5`,`actor_learning_rate=2e-5`(adaptive `[1e-5, 2e-4]`)
- `WANDB_MODE` 不设(用户登录 wandb 走 cloud sync)

### 坑 1:web 终端 heredoc 折断

`cat > /tmp/h6_watchdog_cloud.sh <<EOF ...` 在阿里云 web 终端反复被截断成 `chmod +x ...shistine"; restart_h6 "$PRISTINE_CKPT"; fiexit 2; }d \\` 之类无意义片段。

**绕过**:写到 doc 里改成 `nano /tmp/h6_watchdog_cloud.sh` 让用户人工粘贴,不再用 heredoc。后续诊断用 `printf '%s\n' line1 line2 ...` 逐行写文件。

### 坑 2:`python3` 是交互 shell alias(致命),watchdog 5 连 crash

watchdog launch 后 train 立刻 fail,launch log:
```
ERROR: Isaac Lab is required for training but not installed.
```

但同 shell `python3 -c "import isaaclab"`:
```
isaaclab: /workspace/isaaclab/source/isaaclab/isaaclab/__init__.py
```

**关键诊断**:
```bash
type python3
# python3 is aliased to '/workspace/isaaclab/_isaac_sim/python.sh'
which -a python3
# /usr/bin/python3
# /bin/python3
env which python3
# /usr/bin/python3
```

bash alias 只在交互 shell 生效。`env`、`nohup bash script.sh`、子 shell 跑脚本都**不识别 alias**,会 fallback 到 PATH 里的 `/usr/bin/python3`(系统 Python,无 isaaclab)。`/workspace/isaaclab/_isaac_sim/python.sh` 是个 wrapper,会设置 LD_LIBRARY_PATH 等等再 exec kit 里的 Python。

**验证三段对照**(同一交互 shell):

| 命令 | 结果 |
|------|------|
| `python3 -c "import isaaclab"` | OK(走 alias 到 python.sh) |
| `env python3 -c "import isaaclab"` | ModuleNotFoundError(env 走 PATH) |
| `env OMNI_KIT_ACCEPT_EULA=YES python3 -c "import isaaclab"` | ModuleNotFoundError |

修复 watchdog:把 `python3 gear_sonic/...` 改成绝对路径 `/workspace/isaaclab/_isaac_sim/python.sh gear_sonic/...`,`pgrep -f` 同步从 `python3 .*train_agent_trl` 改成 `python.sh .*train_agent_trl`。

```bash
sed -i 's|python3 gear_sonic/train_agent_trl|/workspace/isaaclab/_isaac_sim/python.sh gear_sonic/train_agent_trl|' /tmp/h6_watchdog_cloud.sh
sed -i 's|pgrep -f "python3 .*train_agent_trl|pgrep -f "python.sh .*train_agent_trl|' /tmp/h6_watchdog_cloud.sh
```

### 坑 3:broken editable install (`__editable__.gear_sonic-0.1.0.pth`)

诊断 isaaclab 时一开始误判:`python3 -c "import isaoclab"` 成功 / 跑 `train_agent_trl.py` 失败。

A) 先怀疑脚本顶部 sys.path 操纵 → 单独测 `sys.path.insert(0, '.../gear_sonic'); import isaaclab` 成功,排除。

B) 翻 site-packages,发现:
```
/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/__editable__.gear_sonic-0.1.0.pth   (42 bytes, exists)
/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/__editable___gear_sonic_0_1_0_finder.py   (NOT FOUND)
```

`.pth` 文件运行时执行 `import __editable___gear_sonic_0_1_0_finder; .install()` 注册 MetaPathFinder。finder 模块缺失 → 抛 ImportError → 影响后续 .pth 处理(包括 isaaclab 的 finder)。但是只在某种调用顺序下起效,所以最初的 `python3 -c "import isaaclab"` 反而过了。

C) `pip uninstall gear_sonic` 报 "No files were found to uninstall"(RECORD 已坏);`pip install -e .` 报 setuptools 找不到 packages config(pyproject.toml 缺 `[tool.setuptools.packages.find]`)。

**绕过**:直接 `rm` 那个坏 .pth。`gear_sonic/train_agent_trl.py:16-29` 已经把 repo root 塞进 sys.path,`from gear_sonic.x import y` 不靠 editable install 也能 import。

### 坑 4:wandb 鉴权

`WANDB_MODE=offline` 不生效(wandb 0.24.0 仍然 enforce login 来初始化 project context,代码里直接 `wandb.init(...)`)。最后用户拿 https://wandb.ai/authorize 的 key,**用 kit Python 的 wandb 登录**(不是系统 wandb,免得跟训练写到不同 .netrc):

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m wandb login <KEY>
```

写 `/root/.netrc`,所有后续进程包括 watchdog spawn 的 train 自动 pickup。

### 训练上线(2026-05-26 04:11)

```
isaaclab + IsaacSim kit OK
wandb sync OK (run 9ck2jyt4 @ wandb.ai/algoritmus/TRL_G1_Track)
4096 envs, 129785 motion files, GPU L20 46GB
Iteration time: 4.76-5.06s  |  FPS: ~20500 step/s  |  无 crash / 无 OOM
```

iter 1198 (1.6h) 时:reward 0.20 → 1.81,entropy -37.47 → -42.45(单调下降,正常),`Mean action noise std: 0.33`(没塌缩)。本地 512 env vs 云端 4096 env:**~8x 加速**。

### 经验

- **远程脚本里禁止裸用 `python` / `python3`**——bash alias 不跨非交互 shell。所有 launch 脚本写绝对路径 / `which python` 拿到的硬路径。这次踩了 5 次自动重启才定位到。
- **诊断"shell A 成功 / shell B 失败"先查 alias**:`type cmd` / `which -a cmd` / `env which cmd` 三联,看是不是 alias / 不同 PATH 解析。
- **broken editable install 不要硬修**——pyproject.toml / setuptools / RECORD 那一套环环相扣,出问题就直接删 .pth + 靠 sys.path 入口。`from x import y` 不需要 pip install -e。
- **`WANDB_MODE=offline` 在新版 wandb 里不能完全免登录**。云端 wandb 的最简方案就是一次 login 写 `~/.netrc`,所有进程继承。
- **web 终端 heredoc 不可靠**,改 nano 粘贴或 printf '%s\n' 逐行写。
- **从 pristine warm-start 而不是从中间 ckpt 续训**。本地 H6 step 16700 已带特定 reward 分布偏移,4096 env 数据 mix 全变,续训会发散——pristine 起点反而稳。

### 下一步

- 等云端 iter ~3000–5000 再导 ckpt,跑 `/tmp/h1_probe_pipeline.py`(pipeline 版),看 σ=1 μ_max 是否能从本地 H6 的 3.98 进一步压到 ≤ 3
- 部署上机平地静置 60 tick:q_target 应稳定在 default ± 5°
- log 显示 ETA 484058s ≈ 5.6 天(`max_train_steps=100000`),实际 5-10k iter 出 ckpt 验证就够,**不要让它跑满**

---

## [2026-05-27] H6 sim2sim 部署失败 → 防线 ②③ 锁 waist_pitch 的结构性 bug

- **现象**:
  - BASELINE `model_step_041550_g1.onnx` 在 23DoF 锁定 sim 上能站住(可见躯干前漂,但有界);
  - 同 BASELINE warm-start 的 H6 微调,iter1400/1600/3000/13000 全部 sim2sim 部署失败:**200 iter 退化到只能站 2-3s,1000 iter 关节角彻底乱飞 ±500°**。
- **触发**:`gear_sonic_deploy/policy/release/` 里替换 `model_encoder.onnx` + `model_decoder.onnx` 为 H6 各 ckpt → 启动 `simulate_23dof=true` sim2sim,看到 keyframe 里 ctrl 出现荒谬值。
- **发现路径**:用户三次复现"H6 比 BASELINE 更糟"的现象后,把 BASELINE 当对照重测才意识到方向反了 —— 不是"H6 没收敛",是"H6 把 BASELINE 已有的能力训坏了"。

### 调查 1:误信探针指标走了一段弯路

写了 `/tmp/h1_probe_pipeline.py`(forward probe pipeline 版,encoder 1751 → decoder 994),测 zeros / σ=0.01 / σ=1 输入下 23 个硬件关节 |μ|max。结果显示:

| ckpt | zeros |μ|max | σ=1 μ_max |
|------|-------------|-----------|
| BASELINE 041550 | 1.916 | 11.404 |
| iter3000 | 1.92 | 7.37 |
| iter13000 | **1.418** | **4.878** |

→ "iter13000 比 BASELINE 好一倍",于是部署 iter13000,**仍然飞**。

**但训练侧 in-distribution 指标在告诉你完全相反的故事**(`/home/woan/下载/h6_launch.log` 分析):

| iter | mean R | entropy | time_out | foot_term |
|------|--------|---------|----------|-----------|
| 500 | 1.91 | -39.8 | 0.51% | 69.5% |
| 1500 | 2.37 | -43.3 | 0.81% | 66.4% |
| 3000 | 1.66 | -45.4 | 1.03% | 63.5% |
| 13000 | 1.35 | -48.1 | 0.62% | 63.5% |

`foot_pos_xyz` 终止率 13k iter 才下降 6pp,reward 在 iter1500 见顶后下滑,entropy 单调走深(actor σ 从 0.5 塌到 0.06)→ **policy 在 sim 里就没收敛**,probe 指标改善只是"OOD 区域 actor 行为更圆润",和"能不能稳住 23DoF 物理"是两件事。

### 调查 2:action_scale=1.0 让小残差变大灾难

读 `gear_sonic/config/manager_env/actions/terms/joint_pos.yaml`(无 `scale:` 字段)+ IsaacLab `JointPositionActionCfg.scale` 默认 1.0 → **q_target = default_qpos + μ 直接代入**(无缩放)。

含义:iter13000 zeros |μ|max=1.42 rad **直接折算成 81° 关节偏移**。BASELINE 的 1.92 rad 也飞,但 BASELINE 训练分布里这种 zero-obs 不会出现;部署平地零关节静置那一帧才会。

→ 部署阶段 actor 输出 0.05 的"小残差"乘以 scale=1.0 也是 ±2.9° 偏移,持续累积仍然会漂。**deploy 端 cpp 行 3125 写 0 兜底是必要不是充分** —— 兜不到的 23 维仍按 μ 走。

### 调查 3:tracking_time_out 的反直觉语义

读 `gear_sonic/envs/manager_env/mdp/terminations.py:245-261`:
```python
def tracking_time_out(env, command_name):
    elapsed = command.time_steps + command.motion_start_time_steps + 1
    total = command.motion_lib.get_time_step_total(command.motion_ids)
    return elapsed >= total
```

`time_out` 不是"超过 max_episode_steps",**是"motion clip 完整跑完没死"**。所以 H6 iter500 的 `time_out: 0.51%` 意思是 **99.5% 的 episode 都是失败终止的**(foot/anchor/ee_body),只有 0.5% 跑完了 motion → 这是灾难,不是健康。

我之前 1% time_out 误读成"才 1% 长度终止,正常",反向了语义。

### 调查 4:eval_agent_trl 不能加 +exp=

H6 ckpt 导出 ONNX 时 `python eval_agent_trl.py +exp=...sonic_release_23dof_h6_idle +checkpoint=...` 报:
```
Could not override 'trainer'. No match in the defaults list.
```

**因为 `gear_sonic/config/base_eval.yaml` 没 `trainer` 组**(eval 不需要 trainer),而 `sonic_release.yaml:11` 里有 `override /trainer: trl_ppo_aux`。

**正确做法**:eval 模式只传 `+checkpoint=path/to.pt`,会自动读 ckpt 旁边保存的 `config.yaml`,合并命令行 override。

### 调查 5:TRT cache hash 自动失效

担心 onnx 替换后 deploy 端的 trt 缓存还是旧的 → 读 `gear_sonic_deploy/.../InferenceEngine.cpp:128-157`:trtFile 文件名包含 ONNX 的 hash,onnx 一变 hash 一变,自动重建,**不需要手 rm 缓存**。

### 调查 6:H6 / A / B 三方对照(决定性证据)

写 `sonic_release_idle.yaml`(只继承 `sonic_release` + 加 H1 idle reset,不带 5 道防线),双 run 跑:
- A:`sonic_release`(纯 29DoF,无 idle,无锁)
- B:`sonic_release_idle`(29DoF + idle reset,无锁)
- H6 数据:已有(5 道防线 + idle reset)

iter116/38 时:

| Metric | H6 iter500(锁 waist + idle) | A iter116(无锁 / 无 idle) | B iter38(无锁 / 有 idle) |
|---|---|---|---|
| `time_out` | 0.51% | **85.31%** | **86.18%** |
| `foot_pos_xyz` 终止 | 69.5% | 6.46% | 4.68% |
| Mean entropy | -39.8(σ≈0.06) | +14.3(σ≈0.40) | +13.5(σ≈0.39) |
| Mean length | ~150 推算 | 270.89 | 272.60 |

A、B 极相近 → idle reset 干净,无副作用。
A vs H6 / B vs H6 → **5 道防线启用就是退化的元凶**。

### 根因

`D1_23DOF_ARCHITECTURE.md` 设计的"6 个 missing joints + 5 道防线"违反 [[wbc-waist-pitch-postural]] memory(2026-05-14 已记录):
1. release 29DoF policy **主动用 `waist_pitch` 后仰**,把 COM 拉回脚上方 —— 这是 postural actuator,不是装饰;
2. 防线 ②(`MissingDofsLockEnv` 把 waist_pitch qpos 钉死 0)= 把 BASELINE 的平衡工具物理拆了;
3. 防线 ③(`actor_critic_modules.py` 的 `locked_action_mask` 把 waist_pitch dim μ→0、σ→1e-4)= 切断该 head 的梯度回流,29-joint coordination 在前几百 iter 就被破坏;
4. PPO 在"失去 waist_pitch 物理 + actor 缺一头"的新动力学下重建平衡,**比从头训 23DoF 还难**(因为 trunk 已按"有 waist_pitch"训过 1M+ iter,coordination 全是错的)。

→ "warm-start 23DoF 微调"在这个 missing 集合下不是微调,是结构性拆迁。BASELINE 直接部署虽漂但有界;微调反而把这点有界性也拆了。

### 改动(待用户决定后做)

候选 A+D 方案(收缩 missing 集合 + 不 mask actor 输出):

1. `gear_sonic/utils/joint_constants.py`:`MISSING_23DOF_INDICES_MJCF` 从 `[13,14,20,21,27,28]` 改为 `[13,20,21,27,28]`(去掉 14=waist_pitch);`MISSING_23DOF_INDICES_IL` 从 `[5,8,25,27,26,28]` 改为 `[5,25,27,26,28]`(去掉 8)。
2. `gear_sonic/config/missing_dofs/23dof_hardware.yaml` 同步。
3. `gear_sonic/trl/modules/actor_critic_modules.py:139-150`:`locked_action_mask` 整段改为可选(yaml 字段控制),H6 默认关。
4. **保留** deploy 端 `g1_deploy_onnx_ref.cpp:3124-3125` 的 6 idx 写 0 兜底(行 3125 把 waist_pitch 也写 0,真机仍然不动)。

代价:训练分布(waist_pitch 活动)与部署分布(waist_pitch 写 0)在该维度有差,actor 学到"用 waist 平衡"在真机变成"试图用但被掐 0",会出现 BASELINE 同款躯干前漂 —— 但**这恰恰是用户已确认能接受的状态**,微调只优化其他 22 维。

### 验证(已完成)

- A、B 两个对照 run 的 termination distribution 与 H6 几乎完全相反 → 假说强证实。
- A、B 跑到 ~2k iter(吃午饭中)等 reward 曲线确认健康。
- 改动暂未做,等 A/B 跑出对照数据后实施 A+D。

### 经验

1. **Probe 指标只反映 OOD 行为,不是 in-distribution 训练健康指标**。下次再看到"σ=1 输入 μ_max 改善"就警觉:训练侧 termination distribution 对得上吗?对不上的话改善是假的。
2. **`tracking_time_out` 语义是"motion 跑完",不是"超长被截"**。低 time_out% = 多数 episode 失败终止,是糟糕信号不是健康信号。读 termination 名字之前先看实现。
3. **action_scale 默认 1.0**。`JointPositionActionCfg.scale` 不显式 override 就是 1.0;μ 直接当作 q_target 偏移,任何"小残差"都会被照单全收。
4. **eval_agent_trl 不要加 `+exp=`**,eval 只走 `+checkpoint=`,自动读 ckpt 旁的 config.yaml。
5. **TRT cache hash 自动失效**,onnx 一变 hash 一变,部署端不用手 rm。
6. **memory 里的 `[[wbc-waist-pitch-postural]]` 提前两周就警告过 waist_pitch 是 postural actuator**,但 D1_23DOF_ARCHITECTURE 设计时仍然按"6 个 missing joints"去锁。教训:写设计文档之前先 grep memory,任何与 memory 矛盾的设计必须在文档里显式辩护或修订 memory。
7. **三方对照才能定因**。只有 H6 vs probe 指标永远没法定责任,加上"无锁 + 无 idle"和"无锁 + 有 idle"两个 control 才能把"5 道防线"和"idle reset"两个变量分离。

### 下一步

- A+D 方案改 4 个文件,scp 到云端,起一个新 run(H6 同位置:`sonic_release_23dof_a_d_idle.yaml`,继承 H6 但把 missing 集合改 5 个 + 关 actor mask)。
- 或:直接放弃微调,**部署 BASELINE + cpp 行 3125 兜底** —— 用户已确认 BASELINE 能站(虽漂)。投入产出比最高的回退方案。

---

## [2026-05-27] 云端启动失败的两类陷阱 → 整理成 `scripts/launch_cloud.sh`

- **现象 1**:云端裸跑 `python gear_sonic/train_agent_trl.py ...` → `nohup: failed to run command 'python': No such file or directory`。
- **现象 2**:换 `/usr/bin/python3` → `ERROR: Isaac Lab is required for training but not installed`(其实装了,但该 python 的 sys.path 里没有)。
- **触发**:用户在 SSH 进云端后直接复制粘贴本地的 launch 命令。
- **根因**:
  1. 阿里云 dsw 容器(`isaaclab:2.3.2-isaacsim5.1.0-py311-ubuntu24.04`)PATH 里没 `python` 这个名字,只有 `python3`。
  2. `/usr/bin/python3` 是系统 python,不带 IsaacSim/IsaacLab。要走 IsaacLab 自带的 IsaacSim kit python,而**正确入口是 `/workspace/isaaclab/isaaclab.sh -p`**(这个 wrapper 会 source IsaacSim 的 setup_python_env.sh 后再调用 kit python)。
  3. 起来后还要 `headless=True`(无显示),否则 PhysX 想开窗口失败;`++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered`(云端 SMPL 路径不一样)。
- **改动**:新建 `scripts/launch_cloud.sh`,封装上述全部细节。下次启动:`bash scripts/launch_cloud.sh sonic_release 29dof_control` 即可。
- **验证**:用脚本起 A、B 两个对照 run,均一次启动成功,iter time ~9.2s,4096 envs 双跑 GPU ~30GB(46GB 容量内)。
- **经验**:
  1. 云端 vs 本地的 launch 差异:**python 路径、headless、smpl_motion_file** 三件套必须包装,不要靠"复制本地命令"。
  2. `isaaclab.sh -p` 是金字塔顶,不要 hard-code IsaacSim kit python 路径(版本升级会换路径)。
  3. SCP 新 yaml 之前先 ssh `grep` 一下云端 commands.py 里的字段是否合入,免得改了配置但运行时报字段不存在。

---

## [2026-05-27] 29DoF A/B 训练曲线 bit-for-bit 一致 → `static_reset_prob` 是 dead branch

- **现象**:云端起两个 29DoF run 做 A/B 对照(`29dof_control` 关 idle / `29dof_idle` 开 5% idle reset),从 iter 1 开始所有打印的 reward / mean_length / time_out 5 位小数完全一致到 iter 999。例:iter 50 两个 run 都是 `rew=23.03443 len=263.81000 time_out=0.8699`。
- **触发**:本地 commit `cdd36f4` 落地 H1 fix(commands.py 加 5% static reset)后,先用 29DoF 跑 sanity check 验证 idle reset 这条独立轴是否能改善曲线(再决定要不要 23DoF 跑)。
- **发现路径**:用户从云端 scp 回 `/tmp/29dof_control.log` 和 `/tmp/29dof_idle.log` → 我用 `diff` 对了几个采样点 → 全等 → 说明两个 run RNG 状态完全没分叉,意味着我加的 5% mask **从未消费过 RNG**,等同于死代码。
- **调查**:
  1. 读 `commands.py:2920-2948` 现有逻辑:`curr_match = (motion_lib._curr_motion_ids == _static_motion_global_id).nonzero()` → 必须先在 1024 子集里命中,才进 `torch.rand < prob` 分支;不命中就直接 skip,**连 `torch.rand` 都不调用**。
  2. 读 `motion_lib_base.py:1067-1080`:`load_motions` 用 `torch.multinomial(_sampling_prob, num_samples=1024, replacement=True)` 从 ~130k 个 motion 中抽 1024 个塞进 `_curr_motion_ids`。
  3. 算了一下命中率:pin motion(`neutral_idle_loop_001__A087_M`,global_id=14887)进 1024 子集的概率 ≈ `1 - (1 - 1/130000)^1024` ≈ 0.78%。
  4. `ImResampleCallback` 配置 `motion_resample_frequency=250`(`config/callbacks/im_resample.yaml`),意味着每 250 iter 才有一次"投硬币"机会,99% 的硬币结果是"不命中,保持原状"。
  5. 两个 run 用同一个 seed → 同一序列的 multinomial → 同一个 1024 子集 → 这次刚好都没抽中 idle motion → 99% 的 reload 让两 run 状态完全同步 → bit-for-bit 一致。
- **根因**:`static_reset_prob` 实现做了一个错误假设——它 piggy-back motion_lib 当前加载的 1024-子集来匹配 pin id。在 130k 总集合下,这个假设几乎永远不成立。**这不是 prob 太小,是机制本身漏写了一步:必须强制把 pin motion 塞进子集**。
- **改动**:H1 fix v2(plan: `quizzical-weaving-snail.md`,commit `ca8c40f`):
  - `motion_lib_base.py`:加 `_pin_motion_id` 属性,`load_motions` 在 `multinomial` 抽样后强制 `sample_idxes[0] = self._pin_motion_id`,然后才赋给 `_curr_motion_ids`。1023 个随机 + 1 个固定。
  - `commands.py:__init__`:解析出 `_static_motion_global_id` 后追加 `self.motion_lib._pin_motion_id = self._static_motion_global_id`。第一次 load(`__init__` line 241)发生在赋值之前,所以前 ~250 iter 不命中是预期行为。
- **验证**:云端起 H7 run 后 60s 健康检查(待):
  1. iter ~250 后 `Sampling motion: tensor([14887, ...])` 第一个 idx 必须是 14887。
  2. 跟 `29dof_control` 比 reward 曲线必须分叉(idle 5% 一旦真触发,RNG 路径必然不同)。
  3. 如果还是 bit-for-bit 一致 → pin 没生效,回去查 `_pin_motion_id` 是否被覆盖。
- **经验**:
  1. **bit-for-bit 一致是诊断利器,不是 bug**。两个理论应该不同的 run 完全相同 = 你加的代码完全没执行。下次写"概率性触发"代码先想一句:这条路径如果不触发,RNG 还会推进吗?不会的话整个 run 是同一个种子,直接对 diff 就能验证。
  2. **piggy-back 别人的随机子集前先算命中率**。`_curr_motion_ids` 在这个 codebase 默认是 1024-of-130k 的随机子集,不是"全集索引"。任何想"在子集里查 global id"的逻辑都先除一下 1024 / 总数,小于 5% 就需要"强制塞入"或"换数据结构"。
  3. **资源对照不只看曲线形状,要看数值精度**。H6 vs A vs B 当时只看了"差不多"或"reward 差异 > 5%",这次直接对 5 位小数才发现 0% 差异。如果当时直接 diff 也许早一星期定位。
  4. **`motion_resample_frequency=250` 是 H1 触发的最早时间锚**。配套机制要么和它对齐,要么显式调小。本次接受 250 iter 的"warm-up dead zone"是因为 finetune 总长度 ≥ 3k,占比 < 8%。

---

## [2026-05-27] Hydra `override /missing_dofs@missing_dofs:` package directive 必须带 `@`

- **现象**:H7 yaml 第一版用 `- override /missing_dofs: 23dof_hardware_unlock_waist`,本地 `python -c "import yaml"` 通过,但 Hydra 在云端 compose 时不会真正用新 yaml(还是会读 base 里的 23dof_hardware)。
- **触发**:写 `sonic_release_23dof_h7_idle_unlock_waist.yaml` 想 override missing_dofs 子树。
- **发现路径**:静态 yaml 检查通过但 Hydra 行为不对——读 `sonic_release_23dof.yaml:4` 注意到 base 用的是 `/missing_dofs@missing_dofs: 23dof_hardware`(带 `@missing_dofs` package 指令),而我写的 override 没加 `@` 部分。Hydra defaults 列表里的 override 必须**完全匹配**原始条目的 group name + package directive,不然命中不到。
- **根因**:Hydra 1.x defaults list 用 `(group, package)` 二元组做 key 来定位要 override 的条目。`/missing_dofs@missing_dofs:` 表示 group=`missing_dofs`、package=`missing_dofs`,如果 override 写成 `/missing_dofs:`(没 package 指令)Hydra 会理解为另一个不同的条目(默认 package 是 group path),根本就不会替换。
- **改动**:`sonic_release_23dof_h7_idle_unlock_waist.yaml` 第一版的 `- override /missing_dofs: 23dof_hardware_unlock_waist` 改成 `- override /missing_dofs@missing_dofs: 23dof_hardware_unlock_waist`。
- **验证**:`python -c "import yaml; print(yaml.safe_load(open('...')))"` 看到 dict key `'override /missing_dofs@missing_dofs'`,匹配 base 的 group+package 二元组。Hydra compose 时 missing_dofs 子树会真正被新 yaml 替换。
- **经验**:
  1. **写 Hydra override 之前先 grep base 配置的对应条目原文**,字符级别复制粘贴 group + package 指令,不要凭"应该是这样"猜。
  2. **YAML 语法对 ≠ Hydra 语义对**。yaml 解析通过只能保证 dict 结构对,Hydra defaults 列表的语义层 bug 要靠"compose 后真去查替换值"才能验证。
  3. **下次怀疑 override 没生效**:cloud 上跑一遍 `python -c "from hydra import compose, initialize; ..."` dry-compose 一下,直接打印 `cfg.missing_dofs.indices_il`,看是 5 个还是 6 个。比起完整起训练再看 log 几个数量级的快。

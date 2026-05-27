# GR00T-WholeBodyControl Sim2Sim 安装与运行手册

> 目标：在 MuJoCo 中跑通 sim2sim（用 SONIC 的 C++ 推理栈驱动一个虚拟 G1 机器人）。
> 官方参考：
> - <https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html>
> - <https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html>

---

## 0. 总览（架构）

Sim2sim **必须同时跑两个进程**，分别在两个终端：

| 终端 | 进程 | 作用 | 环境 |
|------|------|------|------|
| **T1** | `python gear_sonic/scripts/run_sim_loop.py` | MuJoCo 物理仿真 + 与 WBC 桥接 | Python venv `.venv_sim`（Python 3.10） |
| **T2** | `bash deploy.sh sim` | C++ SONIC 推理（TensorRT + ONNXRuntime） | 编译产物，需 TensorRT |

仅安装 `install_mujoco_sim.sh` 只能启动 T1，**T2 需要单独装 C++ 部署栈**。

---

## 1. 当前进度（✅ 全部完成，sim2sim 已跑通）

- [x] `git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git`
- [x] `git lfs pull`
- [x] `bash install_scripts/install_mujoco_sim.sh` —— 已创建 `.venv_sim`（Python 3.10）
- [x] 安装 TensorRT 10.13.0.35 + cuda11.8（deb 方式 + 软链到 `~/TensorRT`）
- [x] 下载模型 checkpoint（policy 88 MB + planner 739 MB）
- [x] 安装 C++ 部署栈系统依赖（just 1.43.0、ONNX Runtime 1.16.3）
- [x] `just build` 通过 → `target/release/g1_deploy_onnx_ref`
- [x] 双终端 sim2sim 跑通

## 1.1 实测版本（本机 2026-05-14）

| 组件 | 版本 / 路径 |
|------|------------|
| OS | Ubuntu 22.04 (Linux x86_64) |
| GPU CUDA toolkit | 11.8 (`/usr/local/cuda-11.8`) |
| TensorRT | 10.13.0.35+cuda11.8（deb 装于 `/usr/lib/x86_64-linux-gnu/`，软链到 `~/TensorRT/{lib,include}`） |
| ONNX Runtime | 1.16.3 (`/opt/onnxruntime`) |
| just | 1.43.0 (`/usr/local/bin/just`) |
| Python (sim) | 3.10.20，uv 管理，于 `.venv_sim/` |
| 关键 ldd 链接 | libnvinfer.so.10 → `~/TensorRT/lib/`<br>libcudart.so.11.0 → `/usr/local/cuda-11.8/lib64/`<br>libonnxruntime.so.1.16.3 → `/opt/onnxruntime/lib/` |

> ⚠️ **TensorRT 是 deb 安装，不是 tar.gz 解压**。本仓库脚本（`setup_env.sh` 和 `FindTensorRT.cmake`）期望 `$TensorRT_ROOT/lib`、`$TensorRT_ROOT/include` 结构，所以 deb 装好后做了软链兼容（见 §3）。如果以后 `apt upgrade` 把 TensorRT 升级了，需要重跑软链命令。

---

## 2. 前置条件 / 系统要求

| 项目 | 要求 |
|------|------|
| OS | Ubuntu 20.04 / 22.04 / 24.04（你是 Linux x86_64，OK） |
| GPU | NVIDIA GPU + 已装驱动（`nvidia-smi` 能看到） |
| CUDA | 本机已装 **CUDA 11.8**（`nvcc -V` 已确认），TensorRT 要下匹配 11.8 的包 |
| Python | sim 端用 `.venv_sim` 自带 Python 3.10，不需要改系统 Python |
| 磁盘 | 至少 30 GB（TensorRT + ONNXRuntime + 编译产物） |
| sudo | 需要（装 apt 包、写 `/opt/onnxruntime`） |

---

## 3. 安装 TensorRT 10.13（手动）

> NVIDIA 强制要求登录账号下载，**这一步只能你自己做**。
> 本机实际用的是 **deb 本地 repo 包**：`nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-11.8_1.0-1_amd64.deb`

### 3.1 下载（需 NVIDIA Developer 账号）

浏览器打开 <https://developer.nvidia.com/tensorrt> → 登录 → TensorRT 10.13 GA → **CUDA 11.8** → Ubuntu 22.04 amd64 deb。

### 3.2 安装 deb（sudo）

```bash
sudo dpkg -i ~/下载/nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.13.0-cuda-11.8/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

### 3.3 做软链接到 ~/TensorRT（兼容项目脚本）

```bash
mkdir -p ~/TensorRT/lib ~/TensorRT/include
cd ~/TensorRT/lib && for f in /usr/lib/x86_64-linux-gnu/libnvinfer*.so* /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so*; do [ -e "$f" ] && ln -sf "$f" .; done
cd ~/TensorRT/include && for h in /usr/include/x86_64-linux-gnu/NvInfer*.h /usr/include/x86_64-linux-gnu/NvOnnxParser*.h; do [ -e "$h" ] && ln -sf "$h" .; done
```

### 3.4 写 ~/.bashrc

```bash
echo 'export TensorRT_ROOT=$HOME/TensorRT' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$TensorRT_ROOT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3.5 验证

```bash
dpkg -l | grep -E "^ii.*libnvinfer[^-]"        # 应有 libnvinfer10  10.13.0.35-1+cuda11.8
ls $TensorRT_ROOT/lib/libnvinfer.so            # 软链接存在
ldd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref 2>&1 | grep -E "nvinfer|cudart"
# 期望: libcudart.so.11.0（不是 .so.12）+ libnvinfer.so.10
```

---

## 4. 下载模型 checkpoint

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl

# 用项目里 .venv_sim 自带的 Python 即可（已经装了 huggingface_hub）
source .venv_sim/bin/activate
python download_from_hf.py            # 默认下载 SONIC deploy 用的 checkpoint
deactivate
```

> 如需训练 checkpoint：`python download_from_hf.py --training`（sim2sim 不需要）。
> 如果走 huggingface 慢，可以提前 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 5. 安装 C++ 部署栈系统依赖

> ⚠️ 这步会用 sudo apt 装很多包（cmake/clang/eigen/zmq/json/libgtest/...），并把 ONNXRuntime 装到 `/opt/onnxruntime`，还可能尝试装 CUDA dev 包。
> 先确认你能联网到 NVIDIA developer 镜像和 GitHub。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy

chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
```

脚本会：
- `sudo apt install` 一堆 dev 包（build-essential、cmake、cppzmq、nlohmann-json、libgtest 等）
- 装 `just` 命令运行器（`/usr/local/bin/just`）
- 下载 ONNX Runtime 1.16.3 到 `/opt/onnxruntime`
- 检查/补装 CUDA runtime + headers（如缺失）
- `git lfs install --force`

**安装结束后建议重开一个终端**，让 PATH/`/usr/local/lib` 等生效。

---

## 6. 配置环境变量（每次新终端要 source 一次）

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
```

这会设置：
- `CMAKE_PREFIX_PATH`、`onnxruntime_DIR`
- `CUDAToolkit_ROOT`、`CUDA_HOME`、`LD_LIBRARY_PATH`
- `TensorRT_ROOT`（从 `~/.bashrc` 读出）
- ROS2 环境（如果装了；没装会 warning，不影响 sim2sim）

> 可选持久化：
> ```bash
> echo "source $(pwd)/scripts/setup_env.sh" >> ~/.bashrc
> ```

---

## 7. 编译 C++ 推理栈

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
just build
```

构建时长大约 5 ~ 15 分钟，看 CPU。
查看可用命令：
```bash
just --list
```

---

## 8. 🚀 运行 sim2sim — 每次启动按这个流程

> ⚠️ **务必两个终端同时跑，且终端 1 先启动**，否则终端 2 会 ZMQ 连接失败。

### 8.1 终端 1：启动 MuJoCo 仿真

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```
**预期**：弹出 MuJoCo 窗口，G1 机器人静止悬停在原地，等待终端 2。

### 8.2 终端 2：启动 C++ 策略

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh        # 如果没写进 .bashrc，每次都要 source
bash deploy.sh sim
```
**预期**：终端 2 开始打日志，等待键盘命令。

### 8.3 键盘操作（按这个顺序按！）

> **重点**：起步必须 `]` → MuJoCo 窗口 `9` → 终端 2 `T`，三步缺一不可。

| 步骤 | 键 | 在哪儿按 | 作用 |
|------|----|---------|------|
| 1️⃣ | `]` | 终端 2 | **启动 policy**（推理开始） |
| 2️⃣ | `9` | **MuJoCo 窗口**（鼠标点一下让它聚焦） | 让机器人从悬停状态落地 |
| 3️⃣ | `T` | 终端 2 | **执行参考动作**（开始播第 0 段） |
| ▶ | `N` / `P` | 终端 2 | 下一段 / 上一段动作 |
| 🔄 | `R` | 终端 2 | 重置当前动作 |
| ⏹ | `O` | 终端 2 | 停止 |

参考动作存在 `gear_sonic_deploy/reference/example/`，每个子文件夹是一段；`N`/`P` 按字典序在它们之间切换。想加自己的动作见 §11。

---

## 9. 故障排查

| 现象 | 检查点 |
|------|--------|
| `TensorRT_ROOT is not set` | 确认 `~/.bashrc` 写入了 `export TensorRT_ROOT=$HOME/TensorRT`，并已 `source` |
| `libnvinfer.so` 找不到 | `LD_LIBRARY_PATH` 是否包含 `$TensorRT_ROOT/lib` |
| `cuda_runtime.h` 缺失 | 重跑 `install_deps.sh`，或手动 `sudo apt install cuda-cudart-dev-12-x cuda-compiler-12-x` |
| `just: command not found` | `install_deps.sh` 没跑完；或 PATH 不含 `/usr/local/bin` |
| MuJoCo 窗口闪退 | 确认显卡驱动 + OpenGL；试 `MUJOCO_GL=egl python gear_sonic/scripts/run_sim_loop.py` |
| HuggingFace 下载慢 | `export HF_ENDPOINT=https://hf-mirror.com` 后重跑 `download_from_hf.py` |
| 终端 2 报 ZMQ 连接失败 | 确认终端 1 已经在跑 `run_sim_loop.py`；防火墙没拦本机环回 |

---

## 10. 速查命令表

```bash
# 进入项目
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl

# 一次性：装 sim 端 venv
bash install_scripts/install_mujoco_sim.sh                      # ✅ 已完成

# 一次性：装 C++ deploy 系统依赖
chmod +x gear_sonic_deploy/scripts/install_deps.sh
./gear_sonic_deploy/scripts/install_deps.sh

# 一次性：下载 checkpoint
source .venv_sim/bin/activate && python download_from_hf.py && deactivate

# 一次性:编译
cd gear_sonic_deploy && source scripts/setup_env.sh && just build

# 每次跑 sim2sim — 终端 1
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# 每次跑 sim2sim — 终端 2
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh sim

# 添加自定义参考动作（GMR pkl → deploy，详见 §11）
source .venv_sim/bin/activate
python gear_sonic_deploy/reference/gmr_to_deploy.py --input <gmr.pkl> --motion_name <name>
cd gear_sonic_deploy/reference && python convert_motions.py ../../<gmr>_deploy.pkl example/
```

---

## 11. 添加自定义参考动作（GMR pkl → deploy）

> 目标：把 [lgy/GMR](../GMR) 重定向出来的 G1 29DoF pkl 变成 sim2sim 终端 2 能识别的参考动作。
> 实测 case：`penguin_29dof.pkl` (T=380, 30fps) → `example/penguin_walking/`，sim2sim 里按 `N` 即可切到。

### 11.1 输入要求（GMR pkl 格式）

`lgy/GMR/scripts/{bvh,smplx,fbx}_to_robot.py` 输出的 pkl 应包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `fps` | int | 帧率 |
| `root_pos` | (T, 3) | 世界系 base 位置 |
| `root_rot` | (T, 4) | base 四元数，**xyzw** (scipy 约定) |
| `dof_pos` | (T, 29) | 29 个关节角，**MJCF 顺序** |

> ⚠️ GMR 默认就是这个格式。手搓 pkl 时注意 `root_rot` 是 xyzw（`q[3:7][[1,2,3,0]]`），不是 MuJoCo 原生的 wxyz。

### 11.2 两步转换流程

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate

# Step 1: GMR pkl → deploy pkl
#   做的事：MJ→IL DOF 重排、Pinocchio FK 算 14 个 tracked link 的 world 位姿、
#          数值微分得 lin_vel/ang_vel、xyzw→wxyz、补 _body_indexes / time_step_total
python gear_sonic_deploy/reference/gmr_to_deploy.py \
    --input penguin_29dof.pkl \
    --motion_name penguin_walking
# 产物: penguin_29dof_deploy.pkl

# Step 2: deploy pkl → CSV 目录（C++ 端实际读这个）
cd gear_sonic_deploy/reference
python convert_motions.py ../../penguin_29dof_deploy.pkl example/
# 产物: example/penguin_walking/{joint_pos,joint_vel,body_pos,body_quat,body_lin_vel,body_ang_vel}.csv
#       + metadata.txt + info.txt

deactivate
```

### 11.3 在 sim2sim 中使用

参考动作目录就是 `gear_sonic_deploy/reference/example/`，每个子文件夹一段，按字典序排列。
重启终端 2（`bash deploy.sh sim`）后：
- 按 `T` 播第 0 段
- `N` / `P` 切换到 `penguin_walking`
- 屏幕上会看到机器人按重定向的轨迹运动

新动作不需要改任何代码或配置——`convert_motions.py` 把它放进 `example/` 就会被自动发现。

### 11.4 关键约定速查（写自己脚本时对照）

| 项 | 值 / 说明 |
|----|----------|
| Tracked links（14 个） | pelvis, {L,R}\_{hip\_roll, knee, ankle\_roll}, torso, {L,R}\_{shoulder\_roll, elbow, wrist\_yaw} |
| `_body_indexes` | `[0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]` |
| 关节顺序（MJCF, 29 个） | 见 `gmr_to_deploy.py:50` `MJCF_BODY_JOINTS` |
| MJ → IL 重排 | 见 `gmr_to_deploy.py:65` `MJ_TO_IL`（来自 `convert_soma_csv_to_motion_lib.py`） |
| `body_quat_w` 约定 | **wxyz**（与 GMR pkl 的 xyzw 相反，`gmr_to_deploy.py` 已自动转换） |
| Pinocchio q 维度 | 43（29 body + 14 hand），手部 DoF 用 `default_body_pose` 自动补 0 |
| 角速度计算 | 中心差分 `rotvec(q_{t+1} ∘ q_{t-1}^{-1}) / (2dt)` |

---

## 12. 完整键盘速查表

> 来源：<https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html>
> §8.3 是「起步必按」的精简版，本节是完整参考。所有按键都在 **终端 2** 按，除非特别注明。

### 12.1 系统控制（两种模式都生效）

| 键 | 作用 |
|----|------|
| `]` | **启动控制系统**（必须先按这个，policy 才开始推理） |
| `O` | 停止控制并退出（紧急停止） |
| `Enter` | 在 **Normal mode** ↔ **Planner mode** 之间切换 |
| `I` | 重新初始化 base 四元数 + 复位 heading |
| `Z` | 切换 encoder mode（mode 0 / mode 1，需要加载了 encoder） |
| `F` | 报告电机温度（语音 TTS） |

### 12.2 Normal Mode — 参考动作回放

| 键 | 作用 |
|----|------|
| `T` | 播放当前动作直到结束 |
| `R` | **重放** 当前动作（回到第 0 帧暂停）/ 紧急停止 |
| `N` / `P` | 下一段 / 上一段动作 |
| `Q` / `E` | delta heading 左 / 右调整（±π/12） |

### 12.3 Planner Mode — 实时位移

| 键 | 作用 |
|----|------|
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 朝左前 / 右前微调 heading 后前进 |
| `,` / `.` | 横移左 / 右（strafe） |
| `Q` / `E` | 调整朝向（±π/6） |
| `J` / `L` | 调整 delta heading（±π/12） |
| `N` / `P` | 下一组 / 上一组 motion set |
| `1`–`8` | 在当前 motion set 内选模式 |

### 12.4 速度 / 高度

| 键 | 作用 |
|----|------|
| `9` / `0` | 减速 / 加速 |
| `-` / `=` | 降低 / 升高身高（非站立 motion set，0.2–0.8 m） |

### 12.5 紧急停止

| 键 | 作用 |
|----|------|
| `O` | 退出 |
| `R` / `` ` `` / `~` | 立即重置动量（紧急刹车） |

### 12.6 MuJoCo 窗口专用

| 键 | 作用 | 在哪儿按 |
|----|------|---------|
| `9` | 让悬停的机器人落地（启动 sim2sim 时必按） | **MuJoCo 窗口**（先点一下让它聚焦） |

> ⚠️ `9` 在 MuJoCo 窗口里是「落地」，在终端 2 是「减速」。两个窗口互不影响——按之前先看清焦点在哪。
> ⚠️ `R` 在 Normal mode 里是「重放当前动作」，但官方文档同时把 `R` 列为紧急停止键，说明它会强制中断当前推理状态。慎按。

---

## 13. 23DoF 干跑验证流程（手头是 23DoF G1 时）

> 目标：在不动模型 / 不重训练的前提下，**用 sim 模拟 23DoF 硬件**，验证 29DoF 策略在「6 个缺失关节被钉住 + 输入端那 6 维永远是 0」的条件下能否稳住、能否走，再决定是否值得做微调。
>
> 思路：
> - **Sim 端虚拟拔关节**：sim 内部把 6 个不存在的关节用大 damping 锁死（动不了），同时把 publish 给策略的 LowState 里那 6 维强制置 0、PD 力矩也 forced 0。
> - **输入参考动作两套对照**：A) 原始 29DoF 参考 CSV（看策略对「想动但动不了」的反应）；B) 提前清零 6 列后的参考 CSV（看策略对「干净的 23DoF 期望」的反应）。

### 13.1 缺失的 6 个关节（与 23DoF 真机一致）

| MJCF idx | 关节 | IL CSV 列 |
|---------|------|-----------|
| 13 | waist_roll_joint | 5 |
| 14 | waist_pitch_joint | 8 |
| 20 | left_wrist_pitch_joint | 25 |
| 21 | left_wrist_yaw_joint | 27 |
| 27 | right_wrist_pitch_joint | 26 |
| 28 | right_wrist_yaw_joint | 28 |

> 注：23DoF 仍**保留** WristRoll（MJCF 19, 26）。来源：`gear_sonic_deploy/.../robot_parameters.hpp` 的 `G1JointIndex` 注释。
> IL 列号通过 `gmr_to_deploy.py:65` 的 `MJ_TO_IL` 映射得到。

### 13.2 改了哪些代码

| 文件 | 改动 |
|------|------|
| `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` | 加 `MISSING_23DOF_INDICES = [13,14,20,21,27,28]`；构造函数读 `config["SIMULATE_23DOF"]`；`PublishLowState` 在 flag=True 时把 6 个 idx 的 q/dq/ddq/tau_est 强制写 0 |
| `gear_sonic/utils/mujoco_sim/base_sim.py` | `init_scene` 末尾若 flag=True 调 `_lock_missing_23dof_joints`：① 给 6 个关节的 `dof_damping` 设大值（默认 5000）兜底；② 缓存 6 个关节的 (qpos_adr, dof_adr)。`sim_step` 在 `mj_step` 之后调 `_pin_missing_23dof_joints` **硬复位 qpos/qvel = 0**（绝对刚度，重力打不开）。`compute_body_torques` 末尾把 6 个 idx 的 PD 力矩清 0 |
| `gear_sonic/utils/mujoco_sim/configs.py` | `SimLoopConfig` 加 `simulate_23dof: bool = False` 和 `simulate_23dof_damping: float = 200.0` |
| `gear_sonic/scripts/run_sim_loop.py` | 把上述 2 个字段塞进 wbc_config 字典（key 大写） |
| `gear_sonic_deploy/reference/zero_23dof_columns.py`（新增） | 把 `example/` 每段动作的 `joint_pos.csv` / `joint_vel.csv` 那 6 列清 0，写到 `example_23dof/` |

### 13.3 准备 23DoF 参考动作（实验 B）

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic_deploy/reference/zero_23dof_columns.py
deactivate
```
产物：`gear_sonic_deploy/reference/example_23dof/<motion>/...`，结构与 `example/` 完全相同，只是 `joint_pos.csv` / `joint_vel.csv` 那 6 列全是 `0.000000`。

### 13.4 实验 A：sim 锁关节 + 原始 29DoF 参考动作

终端 1：
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --simulate-23dof
```
启动日志里应能看到：
```
[run_sim_loop] simulate_23dof=True (damping=200.0) — ...
[simulate_23dof] locked 6 missing joints with damping=200.0
```

终端 2：照常 `bash deploy.sh sim`（用默认 `reference/example/`）。

观察：
- 站立时：腰 / 腕能否稳住？
- 按 `T` 走第 0 段：策略会发命令，6 个关节会被钉住，看 PD 误差是否拖崩。
- 切到带腰部 / 腕部动作的段（例如 dance），策略输出会和真实不一致，看是否会摔。

### 13.5 实验 B：sim 锁关节 + 23DoF 清零参考动作

`deploy.sh` 已有 `--motion-data PATH` 参数，默认 `reference/example/`。直接传 `example_23dof/`：

```bash
# 终端 1（同实验 A）
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --simulate-23dof

# 终端 2
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
bash deploy.sh --motion-data reference/example_23dof/ sim
```

切回实验 A：把 `--motion-data` 删掉或换回 `reference/example/` 即可，无副作用。
对比 A 和 B 的稳定度——B 应当显著更稳，因为参考动作里那 6 维与 sim 反馈一致都是 0。

### 13.6 验收 checklist

- [ ] 实验 A 能站立 ≥ 30 s 不摔
- [ ] 实验 A 至少能走 1 段下半身为主的动作（如 walking_quip_360）
- [ ] 实验 B 全部参考动作能跑完不摔
- [ ] 关闭 `--simulate-23dof` 跑同一段动作，确认 sim 端没 regression（兜底回归）

3 项满足 → 真机可以谨慎试，建议先吊起来空载试站立。
全没满足 → 需要至少做最小微调：把训练数据里那 6 维 mask 掉再 fine-tune 几千步。

### 13.7 已知风险（先看再做）

- **C++ 部署侧没改**：`G1_NUM_MOTOR=29`，ONNX 仍按 29 维 action 跑。这是有意的——sim 验证的就是「策略输出 29 维但只让其中 23 维生效」的退化行为。**真机也是这个用法**：上机前需要保证 motor_cmd[13/14/20/21/27/28] 不会被 SDK 错误下发到不存在的关节（Unitree SDK 默认这 6 个 motor index 是空的，ignore 即可，但务必 `nvidia-smi`/真机端确认一遍）。
- **release 模型是 29DoF 训出来的**：这个验证流程的 OOD 性可能较强；如果 A/B 都摔，下一步是改训练侧 mask。
- **不要直接复用 yaml 里的 anneal_23dof**：那个 mask 是 *3 wrist + 0 waist*，与硬件 23DoF *2 waist + 4 wrist* 不一致，用了反而引入新的不匹配。


  本地改 → git add/commit → git push mine 23dof
  云端: git fetch mine 23dof && git reset --hard mine/23dof 
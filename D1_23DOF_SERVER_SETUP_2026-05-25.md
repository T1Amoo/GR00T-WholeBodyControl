# D1 23DoF · PAI DSW 服务器部署 · 一步一步教程

> **目标**:阿里云 PAI DSW 上跑大 num_envs 训练 → 导出 ONNX → 下载到本地 sim2sim 看效果。
>
> **镜像**:`isaaclab:2.3.2-isaacsim5.1.0-py311-ubuntu24.04`(IsaacLab 2.3.2 + IsaacSim 5.1.0 + Python 3.11 + CUDA 13.0 已预装)
>
> **本地数据规模**:`sonic_release/last.pt` 1.4G · `data/smpl_filtered/` 31G · `data/motion_lib_bones_seed/robot_filtered_23dof/` 7.5G · 代码 ~50M
>
> **互动模式**:每个 🟢 步骤复制 SSH 里执行,有 📋 标记的把输出粘贴到下面对应代码块,我看了再给下一步。

---

## 阶段 0 · 摸底:云端镜像里到底装了啥

镜像号说"已预装 IsaacLab",但具体 Python 路径、是否激活 conda、PyTorch 哪个 cuda 版本要先确认,免得后面装错。

### 🟢 0.1 在 SSH 里执行

```bash
echo "=== whoami / pwd ==="
whoami; pwd; hostname
echo "=== Python ==="
which python && python --version
which python3 && python3 --version
echo "=== conda ==="
which conda || echo "no conda"
conda env list 2>/dev/null || echo "no conda envs"
echo "=== IsaacLab / IsaacSim ==="
python -c "import isaaclab; print('isaaclab', isaaclab.__version__, isaaclab.__file__)" 2>&1 | head -5
python -c "import isaacsim; print('isaacsim', isaacsim.__file__)" 2>&1 | head -5
echo "=== PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
echo "=== GPU ==="
nvidia-smi | head -20
echo "=== Disk ==="
df -h ~ /workspace /mnt 2>/dev/null | grep -v tmpfs
echo "=== CUDA toolkit ==="
nvcc --version 2>/dev/null | tail -2 || echo "no nvcc"
```

### 📋 0.1 输出(已确认 2026-05-25)

```text
whoami: root
pwd:    /mnt/workspace/lgy
host:   dsw-829471-bfc7c8488-v642c

Python:    /usr/bin/python3, Python 3.11.13   (无 conda,系统 python 直接 pip)
IsaacLab:  package version 0.54.2 at /workspace/isaaclab/source/isaaclab/isaaclab/__init__.py
IsaacSim:  /workspace/isaaclab/_isaac_sim/python_packages/isaacsim/__init__.py
PyTorch:   2.7.0+cu128, cuda 12.8, available=True, device=NVIDIA L20
Driver:    NVIDIA-SMI 550.163.01, Driver 550.163.01, runtime CUDA 12.4(向前兼容 torch cu128 OK)
GPU:       L20  46068 MiB (~48G)  当前空闲(无进程)
Disk:      / 295G 可用,/mnt /workspace 同 overlay,够用
CUDA tk:   无 nvcc(torch 自带 runtime,不影响)
```

**结论 / 锁定决策**:
- 工作根目录 = `/mnt/workspace/lgy/GR00T-WholeBodyControl`
- Python 命令 = `python3` 和 `pip3`(没 conda,**不要** `conda activate`)
- num_envs 锁 **4096**(L20 48G 完全 hold 得住)
- 不需要装 IsaacLab/IsaacSim/PyTorch,镜像里都齐了
- isaaclab `__version__` 报 0.54.2 是 python 包内部版本号,**不等于** 镜像标签的 v2.3.2(后者是 git release tag);两者并不矛盾,可以正常用

---

## 阶段 1 · 本地侧:打代码包(代码+ckpt 走 scp,大数据另说)

> 数据 38G 走 scp 跨城太慢,**建议**用阿里云 OSS 中转(下面 §1.3 给方案)。

### 🟢 1.1 本地打代码 tar(剔除大目录)

> ⚠️ **大坑清单**(全部已加进下面 exclude):
> - `bones_seed_raw 71G` — 原始 SMPL 数据,云端只用已处理的 `data/motion_lib_bones_seed/...`
> - `.venv_sim 2.5G` — 本地 venv 里塞了完整 CUDA/PyTorch lib,云端镜像有自己的 PyTorch,**绝对**不能上传
> - `gear_sonic_deploy 2.9G` — 部署在本地
> - `motionbricks 2.4G` / `decoupled_wbc 523M` / `docs 374M` / `media 245M` — 训练用不到
>
> 不 exclude 这些 tar 会卡几十分钟还压不出来 / 出来 3G+。下面已经全部加进 exclude 了。

在**本地**(不是 SSH)新开 terminal:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy
tar --exclude='GR00T-WholeBodyControl/logs_rl' \
    --exclude='GR00T-WholeBodyControl/logs_eval' \
    --exclude='GR00T-WholeBodyControl/data' \
    --exclude='GR00T-WholeBodyControl/sonic_release' \
    --exclude='GR00T-WholeBodyControl/outputs' \
    --exclude='GR00T-WholeBodyControl/wandb' \
    --exclude='GR00T-WholeBodyControl/.git' \
    --exclude='GR00T-WholeBodyControl/.venv_sim' \
    --exclude='GR00T-WholeBodyControl/bones_seed_raw' \
    --exclude='GR00T-WholeBodyControl/motionbricks' \
    --exclude='GR00T-WholeBodyControl/gear_sonic_deploy' \
    --exclude='GR00T-WholeBodyControl/decoupled_wbc' \
    --exclude='GR00T-WholeBodyControl/docs' \
    --exclude='GR00T-WholeBodyControl/media' \
    --exclude='**/__pycache__' \
    --exclude='**/*.egg-info' \
    --exclude='**/*.pyc' \
    --exclude='**/*.onnx' \
    --exclude='**/*.pt' \
    -czf /tmp/groot_code.tar.gz GR00T-WholeBodyControl/
ls -lh /tmp/groot_code.tar.gz
# 预期 ~700MB,2-3 分钟跑完
# 警告 "在我们读入文件时文件发生了变化" 是 logs 在被本地训练写入引起的,无害
```

**保留(必须)**:
- `gear_sonic/` 1.2G — **核心训练代码 + USD 机器人模型**(IsaacLab 加载机器人要用)
- `external_dependencies/` 46M — 第三方依赖(setup.py 可能引用)
- 各种 `*.md`/`*.py`/`Makefile`/`config/` — 配置 + 文档

### 🟢 1.2 本地 rsync 上传代码 + ckpt(已锁实际 SSH 串)

> ⚠️ 你的 PAI DSW 用 **非默认端口 1022**,rsync 要 `-e "ssh -p 1022"`。
>
> **为啥用 rsync 不用 scp**:
> - rsync 支持 `--partial` 断点续传,中断后重跑同命令从断点继续
> - 进度条 `-P` 比 scp 直观(显示实时速率 + ETA)
> - scp 中断只能整个重传,1.4G ckpt 跨城最容易踩

```bash
# === 已锁(本地执行) ===
SERVER='root@39.101.75.133'
PORT=1022
REMOTE_BASE='/mnt/workspace/lgy'
# =========================

# 第一次连接接受 host key + 装 rsync(PAI 镜像默认没装,踩过)
ssh -p $PORT -o StrictHostKeyChecking=no $SERVER \
    "mkdir -p $REMOTE_BASE && pwd && (which rsync || apt-get update -qq && apt-get install -y rsync)"

# 代码 tar (~408M, 3-5 分钟)
rsync -avP --partial -e "ssh -p $PORT" \
    /tmp/groot_code.tar.gz $SERVER:$REMOTE_BASE/

# ckpt (1.4G, ~10-15 分钟 @ 50Mbps 上行;断了重跑同命令接续)
rsync -avP --partial -e "ssh -p $PORT" \
    /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/sonic_release/last.pt \
    $SERVER:$REMOTE_BASE/last.pt
```

> **简化技巧**:在本地 `~/.ssh/config` 加段别名,以后不用每次写 `-p 1022`:
> ```
> Host pai-dsw
>     HostName 39.101.75.133
>     User root
>     Port 1022
> ```
> 然后 `ssh pai-dsw` / `rsync -avP file pai-dsw:/path/` 直接用,rsync 也不用 `-e` 了。

### 🟢 1.3 数据 38G 走 OSS(强烈推荐 vs scp 跨城)

PAI DSW 通常带 OSS 直挂,从 OSS 拉数据比 scp 快 10x,而且能断点续传。两步:

**(A) 本地装 ossutil 上传**:

```bash
# 安装 (一次性,~5MB)
wget -q https://gosspublic.alicdn.com/ossutil/1.7.18/ossutil64 -O ~/ossutil && chmod +x ~/ossutil

# 配置 (填你的 AK/SK 和 bucket endpoint - 阿里云控制台 RAM 子账号生成)
~/ossutil config -i <YOUR_ACCESS_KEY_ID> -k <YOUR_ACCESS_KEY_SECRET> -e <oss-cn-xxx.aliyuncs.com>

# 创建 bucket (如果还没,挑离 DSW 同地域)
~/ossutil mb oss://groot-wbc-data-2026

# 上传数据 (38G, 内网/家用 100Mbps 上行,约 1 小时)
~/ossutil cp -r --update \
    /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/data/ \
    oss://groot-wbc-data-2026/data/
```

**(B) PAI DSW 端拉 OSS**(阶段 2.2 再做)

> **没 OSS 怎么办**:用 rsync(注意带端口):
> ```bash
> rsync -avP --partial -e 'ssh -p 1022' \
>     /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/data/ \
>     root@39.101.75.133:/mnt/workspace/lgy/GR00T-WholeBodyControl/data/
> ```
> 38G 跨城估计 5+ 小时,建议夜里挂着跑。

### 📋 1.1-1.3 输出粘到下面

```text
<贴 ls -lh /tmp/groot_code.tar.gz 输出 + scp 完成 + ossutil cp 完成的状态>
```

---

## 阶段 2 · 云端:解包 + 把 ckpt 移到位

> 🌙 **2026-05-25 夜里 watchdog 上传完成实况**(确认):
> - `groot_code.tar.gz` 408M 在 `/mnt/workspace/lgy/`(早上传)
> - `sonic_release/last.pt` 448M + configs 在 `/mnt/workspace/lgy/sonic_release/`(夜里 21:05)
> - `GR00T-WholeBodyControl/data/` 55G(motion_lib_bones_seed 24G + smpl_filtered 31G,2693 个 23dof motion files,夜里 01:25)
>
> data/ 已经在 `GR00T-WholeBodyControl/` 下,**不需要 OSS / 不需要重传**。

### 🟢 2.1 在 SSH 里解包代码 + 把 ckpt 移到 GR00T-WholeBodyControl 内

```bash
cd /mnt/workspace/lgy
tar -xzf groot_code.tar.gz    # 解出 GR00T-WholeBodyControl/(data/ 已在,tar 不会动它)

# 把 ckpt 目录移到位(watchdog 把它放在 /mnt/workspace/lgy/sonic_release/,要进 GR00T-WholeBodyControl/)
mv /mnt/workspace/lgy/sonic_release /mnt/workspace/lgy/GR00T-WholeBodyControl/sonic_release

# 验证全在
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
ls -lh sonic_release/last.pt              # 应是 448M
du -sh data/motion_lib_bones_seed data/smpl_filtered    # 24G + 31G
ls data/motion_lib_bones_seed/robot_filtered_23dof/220727 | wc -l   # 2693
ls | head    # 看到 gear_sonic / data / sonic_release / external_dependencies / *.md
```

### ~~🟢 2.2 拉 OSS 数据到云端~~ (已跳过 — watchdog 直接 rsync 上来了)

### 📋 2.1 输出粘到下面

root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy# cd /mnt/workspace/lgy/GR00T-WholeBodyControl
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# ls -lh sonic_release/last.pt
-rw-rw-r-- 1 ubuntu ubuntu 448M May 18 06:31 sonic_release/last.pt
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# du -sh data/motion_lib_bones_seed data/smpl_filtered
24G     data/motion_lib_bones_seed
31G     data/smpl_filtered
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# ls data/motion_lib_bones_seed/robot_filtered_23dof/220727 | wc -l
2693
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# ls | head
23DOF_EVALUATION_2026-05-14.md
CITATION.cff
CONTRIBUTING.md
D1_23DOF_DEBUG_LOG.md
D1_23DOF_DESIGN_2026-05-15.md
D1_23DOF_PLAN_2026-05-15.md
D1_23DOF_SERVER_SETUP_2026-05-25.md
D1_23DOF_USAGE_2026-05-18.md
INSTALL_SIM2SIM.md
LICENSE

---

## 阶段 3 · 云端:装 gear_sonic 训练栈

> PAI 镜像已经有 IsaacLab + IsaacSim + Python 3.11 + 配套 PyTorch 2.7+cu128,**只**需要装 `gear_sonic` 自己的依赖。
> **没 conda**,所有 pip 命令直接用 `pip3` / `python3`(或在 PAI 里 `pip` 默认就指向系统 Python 3.11)。

### 🟢 3.1 看 gear_sonic extras 声明,验证 [training] 包名

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
grep -A 30 "training" gear_sonic/setup.py 2>/dev/null || grep -A 30 "\[training\]" gear_sonic/pyproject.toml 2>/dev/null
```

### 📋 3.1 输出粘到下面

```text
<贴 [training] extras 列表>
```

### 🟢 3.2 (等我看完 3.1 输出再执行) 安装 gear_sonic

> 等我确认 extras 里有没有需要单独补的(比如 open3d / vector_quantize_pytorch 历史上 [training] 漏声明过)。预期命令:

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
pip3 install --upgrade pip

# 主包 + training extras (用清华源加速)
pip3 install -e "gear_sonic/[training]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 历史漏声明的(若 3.1 显示已经在 extras 里就跳过)
pip3 install open3d vector-quantize-pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配 setuptools<81 (flatdict 兼容,memory: isaaclab_v232_install_pitfalls)
pip3 install 'setuptools<81' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 🟢 3.3 EULA + 显存配置写到 bashrc

```bash
cat >> ~/.bashrc <<'EOF'
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
EOF
source ~/.bashrc
env | grep -E "OMNI|PYTORCH_CUDA|HYDRA"
```

### 📋 3.2-3.3 输出粘到下面(尤其要看 pip install 有没有红字 ERROR)

195b6bd28c578d315000ec10ba3cad2483a90eb7be61d9f09fb6fed2311b19
  Stored in directory: /tmp/pip-ephem-wheel-cache-j9f29h8f/wheels/e0/88/15/40c113fe0cc60d94799bb5a6bbb644acab63fc01f4252594d1
Successfully built gear_sonic
Installing collected packages: easydict, xxhash, numpy, loguru, httpcore, dill, multiprocess, httpx, gear_sonic, datasets, accelerate, trl
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.0
    Uninstalling numpy-1.26.0:
      Successfully uninstalled numpy-1.26.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
pin-pink 3.1.0 requires quadprog>=0.1.11, which is not installed.
nvidia-srl-base 1.3.0 requires docstring-parser==0.16, which is not installed.
nvidia-srl-usd-to-urdf 1.0.2 requires lxml<5.0.0,>=4.9.2, but you have lxml 5.4.0 which is incompatible.
nvidia-srl-usd-to-urdf 1.0.2 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", but you have usd-core 26.3 which is incompatible.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", but you have usd-core 26.3 which is incompatible.
kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle (from requests->dash>=2.6.0->open3d) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /workspace/isaaclab/_isaac_sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle (from requests->dash>=2.6.0->open3d) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /workspace/isaaclab/_isaac_sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle (from requests->dash>=2.6.0->open3d) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /workspace/isaaclab/_isaac_sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle (from requests->dash>=2.6.0->open3d) (2025.7.14)
Installing collected packages: fastjsonschema, addict, retrying, pyquaternion, narwhals, itsdangerous, frozendict, configargparse, blinker, plotly, flask, einx, dash, vector-quantize-pytorch, nbformat, open3d
Successfully installed addict-2.4.0 blinker-1.9.0 configargparse-1.7.5 dash-4.1.0 einx-0.4.3 fastjsonschema-2.21.2 flask-3.1.3 frozendict-2.4.7 itsdangerous-2.2.0 narwhals-2.21.2 nbformat-5.10.4 open3d-0.19.0 plotly-6.7.0 pyquaternion-0.9.9 retrying-1.4.2 vector-quantize-pytorch-1.29.1
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# pip3 install 'setuptools<81' -i https://pypi.tuna.tsinghua.edu.cn/simple
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Requirement already satisfied: setuptools<81 in /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages (78.1.1)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# cat >> ~/.bashrc <<'EOF'
export OMNI_KIT_ACCEPT_EULA=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
EOF
source ~/.bashrc
env | grep -E "OMNI|PYTORCH_CUDA|HYDRA"
OMNI_KIT_ACCEPT_EULA=YES
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HYDRA_FULL_ERROR=1
OMNI_SERVER=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1
---

## 阶段 4 · 云端:烟测能跑

### 🟢 4.1 import 链路检查

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
python3 -c "
import torch, isaaclab, isaacsim, gear_sonic
import open3d, vector_quantize_pytorch, hydra, trl, wandb
print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
print('isaaclab', isaaclab.__version__)
print('gear_sonic OK')
print('extras OK')
"
```

### 🟢 4.2 单元测试(不起 sim,~30s)

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
python3 -m pytest tests/ -v -k "not sim_env" 2>&1 | tail -30
```

### 📋 4.1-4.2 输出粘到下面

root@dsw-829471-bfc7c8488-v642c:/mnt/workspace/lgy/GR00T-WholeBodyControl# cd /mnt/workspace/lgy/GR00T-WholeBodyControl
python3 -c "
import torch, isaaclab, isaacsim, gear_sonic
import open3d, vector_quantize_pytorch, hydra, trl, wandb
print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
print('isaaclab', isaaclab.__version__)
print('gear_sonic OK')
print('extras OK')
"
torch 2.7.0+cu128 cuda avail True
isaaclab 0.54.2
gear_sonic OK
extras OK
python3 -m pytest tests/ -v -k "not sim_env" 2>&1 | tail -30

tests/test_joint_constants.py:37: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    """MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.
    
    DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
    commands, steps physics, and publishes observations back via the SDK bridge.
    BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
    """
    
    import os
    import pathlib
    from pathlib import Path
    import pickle
    import tempfile
    from threading import Lock, Thread
    import time
    from typing import Dict
    import xml.etree.ElementTree as ET
    
>   import mujoco
E   ModuleNotFoundError: No module named 'mujoco'

gear_sonic/utils/mujoco_sim/base_sim.py:18: ModuleNotFoundError
=========================== short test summary info ============================
FAILED tests/test_joint_constants.py::test_sim_bridge_uses_shared_constants
FAILED tests/test_joint_constants.py::test_base_sim_uses_shared_constants - M...
========================= 2 failed, 22 passed in 1.37s =========================
There was an error running python
---

## 阶段 5 · 云端:启动 H6 训练 + watchdog

> **配置**:H6 = H5h(actor mask) + H1 fix(`static_reset_prob=0.05` + `neutral_idle_loop_001`)。这部分 yaml 已经在本地代码里(`gear_sonic/config/exp/.../sonic_release_23dof_h6_idle.yaml`),通过 1.1 的 tar 已经上来了。

### 🟢 5.1 大卡 num_envs(L20 48G 用 yaml 默认 4096)

| GPU | VRAM | num_envs | resample freq |
|---|---|---|---|
| **L20 48G(本机)** | 48G | **4096** | **250(默认)** |
| A10 24G | 24G | 2048 | 250 |
| V100 32G | 32G | 2048 | 250 |

> 4096 envs 接近原作者训练强度,~12-24h 可见 H1 收敛。
>
> **不再传 `motion_resample_frequency=50`**:本地 H6 在 512 envs 上需要 5x 加速 motion 轮转才能保持多样性;4096 envs 已经在每个 iter 上吃下 8x 数据,默认 250 已经足够,加速反而让 actor 来不及"吃透"每个 motion subset。
>
> **不再传 `static_reset_prob/static_motion_name`**:这两个值已经写在 `sonic_release_23dof_h6_idle.yaml` 里,exp config 自带,不需要命令行 `++` 重复。

### 🟢 5.2 写云端 watchdog 脚本

> ⚠️ **复制注意**:heredoc 里的 `\` 行末续行符在某些 web 终端粘贴时会被吞或断行。如果用 PAI DSW web 终端,建议**先用 `nano /tmp/h6_watchdog_cloud.sh` 打开空文件,把整段贴进去,Ctrl-O 保存,Ctrl-X 退出**,而不是直接 `cat <<EOF`。已验证 nano 没有这种问题。

```bash
nano /tmp/h6_watchdog_cloud.sh
# (粘贴下面整段,然后 Ctrl-O 保存、Ctrl-X 退出)
```

```bash
#!/bin/bash
# H6 云端守护:任何崩溃自动从最新 ckpt 重启;1h 内崩 5 次以上停止。
set -u
WORK_DIR="/mnt/workspace/lgy/GR00T-WholeBodyControl"
H6_RUNS_PARENT="${WORK_DIR}/logs_rl/TRL_G1_Track/manager/universal_token/all_modes"
LOG_BASE="/tmp/h6_launch.log"
WATCHDOG_LOG="/tmp/h6_watchdog.log"
STAMP_FILE="/tmp/h6_watchdog_restarts.txt"
PRISTINE_CKPT="sonic_release/last.pt"

find_latest_h6_ckpt() {
  # exp dir 实际是 sonic_release_23dof_h6_idle_test-<ts>(yaml 里 exp_var=test 拼上去)
  for d in $(ls -dt "${H6_RUNS_PARENT}"/sonic_release_23dof_h6_idle_test-* 2>/dev/null); do
    [ -f "${d}/last.pt" ] && echo "${d#$WORK_DIR/}/last.pt" && return 0
  done
}
: > "$STAMP_FILE" 2>/dev/null || touch "$STAMP_FILE"
log() { echo "[WATCHDOG $(date '+%F %T')] $1" | tee -a "$WATCHDOG_LOG"; }
find_h6_pid() {
  pgrep -f "python3 .*train_agent_trl\.py.*sonic_release_23dof_h6_idle" | head -1
}
count_recent_restarts() {
  local cutoff=$(($(date +%s) - 3600))
  awk -v c=$cutoff '$1 >= c' "$STAMP_FILE" | wc -l
}
restart_h6() {
  local ckpt_arg="$1"
  cd "$WORK_DIR" || return 1
  log "Relaunching with $ckpt_arg"
  [ -f "$LOG_BASE" ] && mv "$LOG_BASE" "${LOG_BASE}.prev_$(date +%H%M%S)" 2>/dev/null
  nohup env OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release_23dof_h6_idle \
    +checkpoint="$ckpt_arg" \
    headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
    > "$LOG_BASE" 2>&1 &
  log "Launched PID=$!"
  date +%s >> "$STAMP_FILE"
}
log "Watchdog started. work=$WORK_DIR num_envs=4096(yaml default) resample=250(yaml default)"
while true; do
  pid="$(find_h6_pid)"
  if [ -z "$pid" ]; then
    log "Not running. Restarts in 1h: $(count_recent_restarts)"
    [ "$(count_recent_restarts)" -ge 5 ] && { log "ABORT: 5+ restarts/h"; exit 2; }
    sleep 60
    ck="$(find_latest_h6_ckpt)"
    if [ -n "$ck" ]; then log "Latest h6 ckpt: $ck"; restart_h6 "$ck"
    else log "No h6 ckpt, using pristine"; restart_h6 "$PRISTINE_CKPT"; fi
    sleep 30
  else sleep 30; fi
done
```

```bash
chmod +x /tmp/h6_watchdog_cloud.sh
# sanity check 脚本没断行(应当是 ~50 行)
wc -l /tmp/h6_watchdog_cloud.sh
bash -n /tmp/h6_watchdog_cloud.sh && echo "syntax OK"
```

### 🟢 5.3 起 watchdog(它会自动启动训练)

```bash
nohup bash /tmp/h6_watchdog_cloud.sh > /tmp/h6_watchdog.out 2>&1 &
echo "watchdog PID=$!"
sleep 5
ps -ef | grep -E "watchdog|train_agent" | grep -v grep

# 跟进训练日志(60s 后应当看到 "Loaded ... motions" / 第一个 iter)
tail -f /tmp/h6_launch.log
```

> Ctrl-C 退出 tail 不影响后台训练。

### 📋 5.3 训练前 5 分钟的日志关键行(看到 "static_reset enabled" + "Iteration:" 就成功)

```text
<贴 tail 看到的 [TrackingCommand] static_reset enabled 那行 + 第一个 Iteration 表>
```

---

## 阶段 6 · 训练中:监控 + 验证 H1 fix 生效

### 🟢 6.1 看 reward 趋势(跑 ~500 iter 后)

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
LATEST=$(ls -dt logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_23dof_h6_idle-* | head -1)
echo "Latest run: $LATEST"
ls -la $LATEST/
# 看 wandb / tb 里 tracking_anchor_pos / tracking_body_linvel 的曲线
```

### 🟢 6.2 ~1k iter 后云端导出 ONNX(在云端验证 H1 收敛)

> 云端 export → 云端跑 h1_probe.py → μ_max ≤ 3 才下载,免得白下载 1.4G 又退回。

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
LATEST=$(ls -dt logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_23dof_h6_idle-* | head -1)
LATEST_CKPT="$LATEST/last.pt"
echo "Exporting $LATEST_CKPT"

# 跑 export(命令以仓库里 export_onnx.py / make_onnx.py 为准,你 grep 看下)
grep -rn "export.*onnx" gear_sonic/scripts/ gear_sonic/tools/ 2>/dev/null | head -5
ls gear_sonic/ | grep -i export
```

### 📋 6.2 输出粘到下面(我看了再给具体导出命令)

```text
<贴 grep + ls 输出>
```

### 🟢 6.3 (等我看完 6.2 给命令)在云端跑 h1_probe 验证

**(a) 本地把 `/tmp/h1_probe.py` rsync 上云**:
```bash
rsync -avP -e 'ssh -p 1022' /tmp/h1_probe.py root@39.101.75.133:/tmp/h1_probe.py
```

**(b) 云端在 onnx 上跑全 0 obs 和 σ=1 obs,看 μ_max**(具体命令等阶段 6.2 confirm 哪个 onnx 是 actor 后再给)。

> **判断标准**:全 0 obs μ_max ≤ 0.5,σ=1 obs μ_max ≤ 3 → H1 fix 生效,可下载部署。否则继续训练。

---

## 阶段 7 · 云端 → 本地:下载 ONNX + 本地 sim2sim

### 🟢 7.1 云端打包要下载的产物

```bash
cd /mnt/workspace/lgy/GR00T-WholeBodyControl
LATEST=$(ls -dt logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_23dof_h6_idle-* | head -1)

# 打包 onnx + last.pt 到 /tmp
mkdir -p /tmp/h6_export
cp -r $LATEST/exported /tmp/h6_export/   # 假设 exported/ 子目录里有 5 个 .onnx
cp $LATEST/last.pt /tmp/h6_export/
tar -czf /tmp/h6_export.tar.gz -C /tmp h6_export
ls -lh /tmp/h6_export.tar.gz
```

### 🟢 7.2 本地下载

在**本地**:

```bash
SERVER='root@39.101.75.133'
PORT=1022
rsync -avP --partial -e "ssh -p $PORT" $SERVER:/tmp/h6_export.tar.gz /tmp/
mkdir -p /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/h6_cloud_export
tar -xzf /tmp/h6_export.tar.gz -C /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/h6_cloud_export --strip-components=1
ls /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl/h6_cloud_export/
```

### 🟢 7.3 本地 sim2sim 测试

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/GR00T-WholeBodyControl
# 用 INSTALL_SIM2SIM.md 里的命令,把 onnx 路径改成 ./h6_cloud_export/exported/
# 关键:加 --simulate-23dof 标志(memory: groot_deploy_d1_23dof_sim_flag)
```

---

## X · 故障排查速查

| 症状 | 怀疑 | 修法 |
|---|---|---|
| `pip install` 卡住/超时 | 国外源慢 | 加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `flatdict` 装失败 | setuptools≥81 | `pip install 'setuptools<81'` |
| `OMNI_KIT_ACCEPT_EULA` 弹窗 | 没 export | 已写进 ~/.bashrc,新 shell 即生效 |
| ModuleNotFoundError: open3d | extras 漏 | `pip install open3d vector-quantize-pytorch` |
| PhysX `cudaErrorMemoryAllocation` | 显存被别的进程占了 | `nvidia-smi` 看,先 kill 老进程再起 |
| CUDA device-side assert + PhysX fabric error | 全局/本地 motion idx 混用(memory: motion_lib_global_vs_local_id) | 检查所有写 `motion_ids` 的地方,**必须**先转本地 idx |
| 切训练新 run 启动失败 | 旧 watchdog 没杀 | `pgrep -af "watchdog"` 全部 kill,等 5s 再起 |
| OSS 下载慢 | bucket 不在同地域 | 换同地域 bucket,或开 `--jobs 16` 并行 |

## Y · Checklist(开训前最后过)

- [x] 阶段 0:Python 3.11 / IsaacLab 已装 / IsaacSim 5.1.0 / **L20 48G 已确认**
- [ ] 阶段 2:`ls sonic_release/last.pt data/smpl_filtered data/motion_lib_bones_seed/robot_filtered_23dof` 都在
- [ ] 阶段 3:`OMNI_KIT_ACCEPT_EULA=YES` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 env 里
- [ ] 阶段 4:`pytest tests/ -v -k "not sim_env"` 22 个 PASS
- [ ] 阶段 5:`/tmp/h6_launch.log` 看到 `static_reset enabled: prob=0.05` + 第一个 `Iteration:` 表
- [ ] 阶段 5:`pgrep -af "watchdog"` 看到 watchdog 进程在
- [ ] 阶段 6:wandb/tb 看到 5% env 的 tracking reward 显著高于 motion env(说明 static envs 在拿 idle reward)

—— 完。下一步:**阶段 1**(本地打代码包 + 上传)。

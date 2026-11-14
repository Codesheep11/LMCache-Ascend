# 在新 Ascend 910B 服务器上部署 xDS 后端

本文记录 Codesheep11 fork 的 **vLLM 0.26.0 + LMCache 0.5.6** 部署路线。适用于单机 910B、专用 NVMe SSD，以及 Qwen 普通 K/V 或 DeepSeek-V2-Lite MLA。它使用 `LMCacheMPConnector` 的 `xds_worker` 模式，不是旧版 0.4.3 的 in-process xDS 后端。

部署组成：

```text
vLLM worker / LMCache-Ascend → 注册该 worker 自己的 HBM pages
                           → xDS readv/writev → SPDK → 专用 NVMe
LMCache MP server          → 缓存键、命中查询和租约（不转发 KV payload）
MLA leader worker          → 一份 SSD 读取 → HCCL → 同 TP 组其他 workers
```

KV 直接读写最终的 vLLM HBM pages，不配置应用层 KV staging slots。HCCL 等通信库仍可能使用内部 workspace。驱动不允许为 IPC 导入的 NPU 内存做 P2P 注册，因此注册和实际 I/O 必须在拥有 HBM 的 worker 中执行。

## 1. 版本与验证范围

以下是本次升级的可复现源码基线；分支以后会继续移动，复现实验应固定 commit。

| 仓库 | 地址 | 分支 / tag | 基线 commit |
| --- | --- | --- | --- |
| xDS | https://github.com/Codesheep11/xDS | `fix/ascend-hal-large-pages` | `e4b4150a5699056791e13124caa4f97ca2ae2705` |
| vLLM | https://github.com/vllm-project/vllm | tag `v0.26.0` | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | https://github.com/Codesheep11/vllm-ascend | `v0.26.0-xds` | `e4e7b63affd434682d6e306f95de25f1e97b7563` |
| LMCache | https://github.com/Codesheep11/LMCache | `v0.5.6-xds` | `d78857dbada4958e77e50e0ac9dbe5e4e2d97b10` |
| LMCache-Ascend | https://github.com/Codesheep11/LMCache-Ascend | `v0.5.6-xds` | `7307ce0cc744a2bb539edefa642a37acfdb12be0` |

已验证环境：aarch64、openEuler 22.03 LTS SP4、内核 `5.10.0-303.0.0.206.oe2203sp4.aarch64`、Ascend 驱动 25.5.1、910B3 64 GiB。容器使用 CANN 9.1.0、Python 3.12、torch 2.10.0、torch-npu 2.10.0.post4。其他内核/驱动组合需要重新编译模块并验收，不能直接复制 `.ko`。

镜像：`quay.io/ascend/vllm-ascend:v0.26.0rc2`，固定 digest：

```text
quay.io/ascend/vllm-ascend@sha256:0e3fb0746832158d66ab2a6bc7263a077a8359ae25c9d5be1f52cf2348c52f56
```

现有机器验证过 Qwen3-30B-A3B TP4 和 DeepSeek-V2-Lite TP4 的恢复正确性，包括 MLA 150K 缓存场景。下面将该环境整理为新机安装步骤；尚未在一台全新服务器上从零重放全部步骤。NVIDIA 实机验证、混合 KV groups、Mamba/GDN、跨机 MLA reader 分组不在本文验收范围。

## 2. 宿主机准备与源码目录

先安装与服务器匹配的 Ascend 驱动/固件、Docker、Git、C/C++ 编译器、make、当前内核的 headers/devel、pciutils、numactl 和 NVMe 工具。确认 `npu-smi info` 正常，`/lib/modules/$(uname -r)/build` 存在。模型文件放在另一块磁盘，不能放到即将交给 SPDK 的 SSD。

下文宿主机统一使用 `/opt/xds-stack`；容器内对应 `/upgrade`。在同一个宿主机 shell 中执行：

```bash
export STACK=/opt/xds-stack
sudo mkdir -p "$STACK"
sudo chown "$(id -u):$(id -g)" "$STACK"
cd "$STACK"
git clone -b fix/ascend-hal-large-pages https://github.com/Codesheep11/xDS.git xDS
git clone -b v0.26.0 https://github.com/vllm-project/vllm.git vLLM
git clone -b v0.26.0-xds https://github.com/Codesheep11/vllm-ascend.git vLLM-Ascend
git clone -b v0.5.6-xds https://github.com/Codesheep11/LMCache.git LMCache
git clone -b v0.5.6-xds https://github.com/Codesheep11/LMCache-Ascend.git LMCache-Ascend
git -C xDS checkout e4b4150a5699056791e13124caa4f97ca2ae2705
git -C vLLM checkout 568afb3a13806beb53bb2e6bd518269357b237c0
git -C vLLM-Ascend checkout e4e7b63affd434682d6e306f95de25f1e97b7563
git -C LMCache checkout d78857dbada4958e77e50e0ac9dbe5e4e2d97b10
git -C LMCache-Ascend checkout 7307ce0cc744a2bb539edefa642a37acfdb12be0
git -C xDS submodule update --init --recursive
git -C LMCache-Ascend submodule update --init --recursive
git -C LMCache-Ascend/third_party/hcomm fetch --tags
git -C LMCache-Ascend/third_party/hcomm checkout 4ca1db912ce02a5f5075849a3fa3faaf731bdd60
mkdir -p run/logs
```

hcomm 此处固定 CANN 9.1.0 对应版本。不要随意更新其 HEAD，也不要混装官方 dev 分支与本文 fork。xDS 自带 SPDK 与内核模块；不需要额外的 direct-p2p、spdk_blob 仓库。

## 3. 宿主机 P2P 模块

`ASCEND_SYMVERS` 必须来自**当前已安装的厂商驱动 build**，包含 `hal_kernel_p2p_get_pages` / `hal_kernel_p2p_put_pages` 的匹配符号版本。仓库附带的符号表只适用于它对应的驱动组合。

```bash
cd "$STACK/xDS"
export ASCEND_SYMVERS=/path/to/matching/ascend_hal.symvers  # 改成真实路径
make kernels-ascend ASCEND_SYMVERS="$ASCEND_SYMVERS"
sudo install -D -m 0644 kernels/modules/ascend/xds_ascend.ko   /lib/modules/"$(uname -r)"/extra/xds_ascend.ko
sudo depmod -a
sudo modprobe xds_ascend
modinfo xds_ascend
ls -l /dev/xds-ascend
```

当前 xDS 分支包含 HAL 大页兼容修复。新机安装正式 `xds_ascend.ko`，不需要本机升级实验曾用的 `xds_ascend_probe.ko`、`probe_open.so` 或 `LD_PRELOAD` 重定向。大页兼容不等于支持 expandable/VMM 内存；vLLM 仍需设置 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`。

## 4. 专用 SSD、ACS 与 hugepages

**xDS Blobstore 初始化会覆盖所选设备的数据。以下步骤只能针对已确认可清空的专用 SSD。** 先检查序列号、挂载、swap、LVM/RAID/Ceph 和使用进程，再选 PCI BDF。不能沿用其他机器的 BDF。

```bash
lsblk -o NAME,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS
sudo nvme list
findmnt
swapon --show
lspci -D -nn
lspci -t
```

检查 SSD 与 NPU 间 PCIe 路径及 NUMA 拓扑。实验要求 PCIe 交换机 ACS 关闭：先逐个列出交换机 bridge 的 BDF，读取并保存 ACS Control，再写入 0 并回读。只对实际存在 ACS capability 的 bridge 执行：

```bash
export BRIDGE_BDF=0000:00:00.0  # 必须替换为实际交换机 bridge
sudo setpci -s "$BRIDGE_BDF" ECAP_ACS+6.w
# 保存上述原值后，对每个相关交换机 bridge 重复：
sudo setpci -s "$BRIDGE_BDF" ECAP_ACS+6.w=0000
sudo setpci -s "$BRIDGE_BDF" ECAP_ACS+6.w
```

仅关闭一层交换机不够；检查所有相关层级。该设置重启后可能丢失，也会改变 PCIe 隔离行为，专用实验服务器应记录其原值和恢复方式。

将确认空闲的 SSD 从 Linux NVMe 驱动绑定到 UIO。下例要求人工填入 BDF，**不会代替你卸载文件系统或停止数据服务**：

```bash
export SSD_BDF=0000:00:00.0  # 替换为确认可清空的 NVMe 控制器
sudo modprobe uio_pci_generic
printf '%s' "$SSD_BDF" | sudo tee "/sys/bus/pci/devices/$SSD_BDF/driver/unbind"
printf uio_pci_generic | sudo tee "/sys/bus/pci/devices/$SSD_BDF/driver_override"
printf '%s' "$SSD_BDF" | sudo tee /sys/bus/pci/drivers_probe
readlink "/sys/bus/pci/devices/$SSD_BDF/driver"
sudo HUGEPAGES_2M=1024 HUGEPAGES_NODE=node0 "$STACK/xDS/scripts/env/hugepages.sh"
sudo mkdir -p /dev/hugepages
mountpoint -q /dev/hugepages || sudo mount -t hugetlbfs none /dev/hugepages
```

根据 SSD/CPU 所在 NUMA 节点调整 `HUGEPAGES_NODE`。脚本设置该节点的 2 MiB 大页总数为 1024，不是追加 1024 页。不要在其他 SPDK 服务使用大页时直接降低配置。

## 5. 镜像、源码绑定与 Python 环境

此路线复用官方镜像中与 vLLM-Ascend 基线匹配的原生算子，安装 fork 的 Python 修改。当前 fork 的原生基线是 `155f68974c802633c19650ece430d41ee6802fa6`。若以后改动原生算子源码，必须重新编译，不能继续使用此复用方法。

先从**未绑定源码的镜像容器**提取原生文件，避免 bind mount 隐藏它们：

```bash
export IMAGE=quay.io/ascend/vllm-ascend@sha256:0e3fb0746832158d66ab2a6bc7263a077a8359ae25c9d5be1f52cf2348c52f56
docker pull "$IMAGE"
docker create --name xds-native-source "$IMAGE"
docker cp xds-native-source:/vllm-workspace/vllm-ascend/vllm_ascend "$STACK/run/image-native"
docker rm xds-native-source
for item in _cann_ops_custom lib include _build_info.py   vllm_ascend_C.cpython-312-aarch64-linux-gnu.so libvllm_ascend_kernels.so; do
  cp -a "$STACK/run/image-native/$item" "$STACK/vLLM-Ascend/vllm_ascend/"
done
```

启动容器。`MODEL_ROOT` 应指向已有模型目录，按机器型号修改 `SOC_VERSION`；选择空闲卡，不能把其他用户任务当作可停止的实验进程。

```bash
export MODEL_ROOT=/path/to/models
export SOC_VERSION=Ascend910B3
docker run -d --name xds-lmcache-026 --privileged --ipc=host --net=host   --ulimit memlock=-1:-1   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro   -v /usr/local/dcmi:/usr/local/dcmi:ro   -v "$STACK":/upgrade   -v "$STACK/vLLM":/vllm-workspace/vllm   -v "$STACK/vLLM-Ascend":/vllm-workspace/vllm-ascend   -v "$STACK/LMCache":/workspace/LMCache   -v "$STACK/LMCache-Ascend":/workspace/LMCache-Ascend   -v "$STACK/xDS":/workspace/xDS   -v "$MODEL_ROOT":/models:ro   -v /tmp:/tmp -v /dev/hugepages:/dev/hugepages   -e SOC_VERSION="$SOC_VERSION"   -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:False   -e XDS_P2P_BACKEND=ascend   -w /upgrade --entrypoint /bin/bash "$IMAGE" -c 'exec sleep infinity'
docker exec -it xds-lmcache-026 bash
```

以下安装和启动命令均在**容器内**执行。不要把宿主机 `/opt/xds-stack` 与容器 `/upgrade` 混用。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv --system-site-packages /upgrade/.venv
source /upgrade/.venv/bin/activate
python -m pip install uv setuptools setuptools-scm wheel packaging ninja cmake pybind11
cat > /upgrade/run/constraints.txt <<'EOF'
torch==2.10.0
torch-npu==2.10.0.post4
transformers==5.14.1
triton-ascend==3.2.2
fastapi>=0.133.0,<0.137.0
starlette>=1.0.1,<2
numpy==2.2.6
attrs==24.2.0
psutil==6.0.0
grpcio==1.78.0
grpcio-observability==1.78.0
protobuf>=6.31.1,<7
opentelemetry-api==1.40.0
opentelemetry-sdk==1.40.0
prometheus-client<=0.24.1,>=0.18.0
EOF
printf 'numpy==2.2.6\n' > /upgrade/run/overrides.txt
VLLM_TARGET_DEVICE=empty COMPILE_CUSTOM_KERNELS=0 NO_GPU_EXT=1   uv pip install --python /upgrade/.venv/bin/python --no-build-isolation   --constraint /upgrade/run/constraints.txt --override /upgrade/run/overrides.txt   -e /vllm-workspace/vllm -e /vllm-workspace/vllm-ascend   -e /workspace/LMCache -e /workspace/LMCache-Ascend
```

NumPy override 是当前组合中 triton-ascend 的旧 pin 与 vLLM/OpenCV 依赖之间的兼容处理；仅加 pip constraint 无法解决互斥声明。当前模型和基础算子验证通过，不代表所有 Triton 算子已覆盖。安装需要 C++/Rust 构建工具及各项目声明的 build dependencies；按实际构建错误补齐，不要用 `--no-deps` 跳过依赖求解。LMCache-Ascend 的 HCCL 绑定需要以这里的 CANN/hcomm 源码构建。

源码通过 bind mount + editable install 同步。Python 修改需要重启服务/worker 才能保证生效；C++/内核修改需要重新编译，绑定目录不会自动更新二进制。

## 6. 编译 xDS 用户态组件

```bash
cd /workspace/xDS
# 在专用容器内安装 SPDK 的系统构建依赖：
./spdk4xds/scripts/pkgdep.sh
make BUILD=release PYTHON=/upgrade/.venv/bin/python JOBS=16 sdk service
export PYTHONPATH=/workspace/xDS/sdk/src:${PYTHONPATH:-}
python -c 'import torch, torch_npu, vllm, lmcache, lmcache_ascend, xds; print(torch.__version__, torch_npu.__version__); print(lmcache.__file__); print(lmcache_ascend.__file__)'
python -m pip freeze > /upgrade/run/pip-freeze.txt
```

确认导入路径指向绑定的源码，而不是镜像里另一份旧代码。若动态链接器找不到 SPDK/DPDK 库，检查构建输出和 `ldd /workspace/xDS/src/service/xds_service`，将实际 build/lib 路径加入 `LD_LIBRARY_PATH`；不要复制其他版本的 `.so` 来补齐。

## 7. 启动 xDS 服务

在容器内重新填入宿主机确认的 `SSD_BDF`，按允许的 CPU affinity 选择四个空闲 reactor cores。下面 mask `0xf` / main core `0` 只是 cores 0–3 的示例，必须按本机拓扑调整。

```bash
export SSD_BDF=0000:00:00.0  # 替换为实际专用 SSD BDF
python - <<'PYCONFIG'
import json, os
from pathlib import Path
p = Path('/upgrade/run')
bdf = os.environ['SSD_BDF']
(p / 'bdev.json').write_text(json.dumps({'subsystems': [{
    'subsystem': 'bdev', 'config': [{'method': 'bdev_nvme_attach_controller',
    'params': {'name': 'XDSNvme0', 'trtype': 'PCIe', 'traddr': bdf}}]}]}, indent=2))
(p / 'setup.json').write_text(json.dumps({
    'bdev_name': 'XDSNvme0n1', 'json_config_file': str(p / 'bdev.json'),
    'reactor_mask': '0xf', 'main_core': 0, 'rpc_addr': '/tmp/xds-npu-rpc.sock',
    'xds_peer_bdfs': [bdf], 'xds_cluster_size_kb': 2048,
    'xds_metadata_page_count': 4096, 'xds_blob_size_bytes': 67108864,
    'xds_blob_count': 2048, 'xds_queue_count': 4, 'xds_worker_poll': True,
    'xds_enable_direct': True, 'xds_gpu_buffer_size_bytes': 17179869184,
    'xds_enable_blob_fast_map': True, 'xds_device_backend': 'ascend'
}, indent=2))
PYCONFIG
/workspace/xDS/src/service/xds_service   --config /upgrade/run/setup.json --check-config
XDS_SERVICE_ALLOWED_UID=0 nohup /workspace/xDS/src/service/xds_service   --config /upgrade/run/setup.json --service-name xds-npu   > /upgrade/run/logs/xds.log 2>&1 &
echo $! > /upgrade/run/xds.pid
```

`--check-config` 只验证配置，不能验证 SSD 可安全清空。首次正式启动才会操作设备。此示例预设 2048 个 64 MiB blob，SSD 需至少容纳 **128 GiB 加元数据**；并非按请求长度动态创建无限空间。服务端总池与各队列份额必须覆盖所有 worker 的 `LMCACHE_XDS_BLOB_COUNT`，不能只增加客户端 blob 数。启动后检查日志中设备、队列初始化及无错误，再启动客户端。

## 8. 启动 MP metadata server 和 Qwen TP4

每个新容器 shell 先激活环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /upgrade/.venv/bin/activate
export PYTHONPATH=/workspace/xDS/sdk/src:${PYTHONPATH:-}
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export XDS_P2P_BACKEND=ascend
```

MP server 通过扩展模块管理 xDS 元数据。其 `--supported-transfer-mode` 使用框架的 `engine_driven` 值；worker/connector 才指定 `xds_worker`。`--l1-size-gb 0.1` 是框架所需的小型 L1 配置，不是 KV payload 的 CPU 中转池。下面的入口先初始化可见 NPU 0，保持与已验证的 Ascend 启动顺序一致。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 nohup python -m lmcache.v1.multiprocess.server   --host 127.0.0.1 --port 18127 --prometheus-port 18128   --chunk-size 128 --l1-size-gb 0.1 --eviction-policy LRU   --supported-transfer-mode engine_driven   --server-module '{"module_path":"lmcache.v1.multiprocess.modules.xds_metadata"}'   > /upgrade/run/logs/lmcache-mp.log 2>&1 &
echo $! > /upgrade/run/lmcache-mp.pid
```

启动 Qwen3-30B-A3B（替换真实模型路径）。先用单请求验收，测试性能时确保四张卡空闲。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30
export LMCACHE_MP_TRANSFER_MODE=xds_worker
export LMCACHE_XDS_SERVICE=xds-npu
export LMCACHE_XDS_BLOB_COUNT=24
export LMCACHE_XDS_QUEUE_DEPTH=64
export LMCACHE_XDS_MLA_READ_MODE=leader
export LMCACHE_XDS_MLA_BATCH_BYTES=536870912
nohup python -m vllm.entrypoints.openai.api_server   --model /models/Qwen3-30B-A3B --served-model-name qwen   --host 127.0.0.1 --port 18126 --tensor-parallel-size 4   --max-model-len 33920 --max-num-seqs 1 --max-num-batched-tokens 4096   --kv-cache-memory-bytes 2147483648 --gpu-memory-utilization 0.45   --additional-config '{"enable_cpu_binding":false}'   --enforce-eager --enable-expert-parallel --no-enable-prefix-caching   --shutdown-timeout 30   --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":18127,"lmcache.mp.mp_transfer_mode":"xds_worker"}}'   > /upgrade/run/logs/vllm.log 2>&1 &
echo $! > /upgrade/run/vllm.pid
```

这里的 QD=64、MLA batch=512 MiB 是实验参数，不是默认值（默认分别为 4、64 MiB）。Qwen 的各 rank K/V 不同，不能广播一份代替；MLA 参数只影响支持的 MLA 模型。DeepSeek-V2-Lite 使用 `leader` 时同一 TP 组由一个 reader 读 SSD，再通过 HCCL 分发，支持读取与通信重叠；`direct` 则各 rank 自己读取。当前要求 TP device group 与 Gloo group 对应同一逻辑 KV shard，不能直接推广到任意 DCP/跨机分组。

150K MLA 要另外提高 `--max-model-len`、HBM KV budget 和 blob 数；已有实验使用每客户端 128 blobs。上述 Qwen 的 2 GiB HBM/24 blobs 配置不是 150K 通用配置。只支持已适配的单 KV group 普通 K/V 和 DeepSeek-V2 MLA，不能套用其他文档中 `lmcache_driven` 对 DeepSeek-V4/DSA 的支持范围。

## 9. 请求与验收

先确认 `/health` 成功。下面依次发送一个用于落盘的前缀请求和一个带新后缀的恢复请求，禁用 vLLM prefix caching，避免把本地 HBM 命中误认为 SSD 恢复。

```bash
curl --fail http://127.0.0.1:18126/health
python - <<'PYCLIENT'
import requests
from transformers import AutoTokenizer
model = '/models/Qwen3-30B-A3B'
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
prefix = tok.encode('请简要介绍计算机存储系统。' * 500, add_special_tokens=False)[:1024]
assert len(prefix) == 1024
suffix = tok.encode('请回答：什么是缓存？', add_special_tokens=False)
for name, ids in [('store', prefix + suffix[:1]), ('restore', prefix + suffix)]:
    r = requests.post('http://127.0.0.1:18126/v1/completions', json={
        'model': 'qwen', 'prompt': ids, 'max_tokens': 16,
        'temperature': 0, 'seed': 1}, timeout=600)
    r.raise_for_status()
    print(name, r.json())
PYCLIENT
# 同时检查 SPDK I/O 计数（在两次请求前后分别保存快照）：
python /workspace/xDS/spdk4xds/scripts/rpc.py -s /tmp/xds-npu-rpc.sock bdev_get_iostat
```

该脚本是启动/命中 smoke test，不等于逐字节正确性证明。验收至少包括：

1. 无命中请求有 SSD 写入；恢复请求日志确认命中，SPDK 读字节增加，worker 无注册/I/O/collective 错误。MP metadata 命中本身不证明 payload 已成功恢复；首次恢复可能需要等待异步 store 完成后再请求。
2. 对同一 token 输入，比较关闭 connector 的完整 prefill 与开启恢复时的输出 token、必要时 logits；使用相同采样参数与模型精度。单次文本相同不足以排除部分 KV 损坏。
3. 使用独立测试进程执行 HBM 写入前/恢复后的逐层校验，覆盖 latent/rope 两个 plane、所有 TP ranks、跨 chunk 边界和非连续 block mapping。现有验证覆盖不替代新机验收。
4. 性能测试分开报告 TTFT、KV 恢复窗口、首次 I/O 前开销、SSD I/O 窗口和通信等待。带宽分母写清窗口；MLA SSD 字节是一份，不能乘 TP 数冒充 SSD 带宽。

可先执行不需要真实模型的回归测试（在容器内，安装 pytest 后）：

```bash
cd /workspace/LMCache
python -m pytest -q --confcutdir=tests/v1/multiprocess/xds tests/v1/multiprocess/xds
cd /workspace/LMCache-Ascend
python -m pytest -q --confcutdir=tests/v1/xds tests/v1/xds
```

这些回归测试包含模拟设备路径，不能替代真实 NPU/SSD 的端到端验证。

## 10. 停止、重启与排障

按 vLLM → MP server → xDS 的顺序正常停止。先停止接收请求，向本部署记录的 PID 发 SIGTERM，确认 workers 退出、I/O 排空和 HBM 注册释放后再停止 xDS；不要用全局 `pkill` 停其他用户任务。正常关闭完成前不要卸载模块或切回 NVMe 驱动。发生不确定的 I/O/collective 失败时，保护仍可能被 DMA 使用的 HBM 和租约，排查后重启该部署，不要强行提前释放。

| 现象 | 检查与处理 |
| --- | --- |
| HAL 注册失败 | 确认正式大页修复模块已加载；设置 expandable_segments=False；必须注册 worker 原始 HBM，不能注册 IPC 导入地址 |
| `/dev/xds-ascend` 不存在 / symbol mismatch | 检查宿主机模块加载、内核 headers 与厂商符号版本；不要强制忽略符号版本 |
| SPDK 找不到设备 / 无 P2P | 检查 BDF、UIO 绑定、PCIe 路由、各级 ACS、hugepages 与内核日志 |
| 启动成功但不命中 | 检查 server module、两端 transfer mode、service name、chunk size、模型/token 完全相同及 store 是否完成 |
| blob 耗尽 | 按实际层数、KV dtype、TP/MLA 布局计算缓存字节，检查每队列份额；客户端与服务端容量一起调整 |
| MLA hang / collective 错误 | 检查各 rank 控制流和 TP 通信组一致；调试可使用 direct 模式隔离 HCCL 分发问题 |
| 模块从错误目录导入 | 检查 venv、editable 安装与 PYTHONPATH；避免旧 0.4.3 包覆盖本次环境 |
| 延迟抖动 / TTFT 高 | 检查共享 NPU/CPU/SSD 负载、NUMA、GC、元数据调度和通信等待；不能仅凭 TTFT 推断 SSD 慢 |

旧的 `xds_read_slot_count` / `xds_write_slot_count` 和旧 YAML `xds_backend` 配置不用于本文路线。升级源码前保存 commit manifest、依赖清单和启动配置；先在独立目录/容器验收，再切换现有服务。

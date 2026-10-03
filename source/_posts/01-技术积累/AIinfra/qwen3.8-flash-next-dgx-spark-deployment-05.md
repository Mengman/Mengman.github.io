---
title: 【Qwen3.8-Flash-Next 端侧部署】05-DGX Spark 统一内存管理与推理执行流程
date: 2026-10-02T08:41:48.724Z
tags: [qwen, vllm, aiinfra, dgx-spark]
categories: aiinfra
typora-root-url: ../../../
---

前几篇分别讨论了 PLE 卸载、NVFP4 权重量化、FP8 KV Cache，以及 MTP 和运行时配置。把这些优化组合起来之后，模型已经具备在单台 DGX Spark 上运行的条件。不过，能够加载权重，还没有回答另一个问题：当长 prompt、多个生成请求和同机进程一起运行时，剩余内存是否足够？

这个问题在统一内存机器上尤其具体。GPU 扩大 KV Cache，会压缩 CPU 和文件缓存可以使用的空间；PLE 虽然通过 mmap 从 NVMe 按需读取，访问过的数据页仍然要占内存；一个请求结束后，CUDA allocator 和 Linux page cache 也不一定立即把空间还成空闲页。因此，我们需要把启动预算和请求执行放在一起看，才能理解配置为什么能启动，又为什么可能在运行一段时间后出现压力。

本文沿用 MiaAI-Lab 的 [Qwen3.8-Flash-Next-Single-DGX-Spark 项目](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)，先解释启动脚本怎样分配整机内存，再让一个长请求经过 CPU、GPU 和 NVMe，最后回到运行期间的监控。



## 一、DGX Spark 的内存分配

DGX Spark 的 CPU 和 GPU 通过统一内存架构访问同一组 128GB LPDDR5x 内存。在 UMA 的内存架构下，这 128GB 的内存既充当系统内存提供给 CPU 程序使用，也充当显存给 GPU 程序使用。

NVIDIA 在[系统说明](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)中标称其容量为 128 GB，但是实际 Linux 系统显示为 121.69GB。 这是因为硬件厂商按照十进制来计算的内存容量，而操作系统是按照二进制来统计内存大小，所以大概有 7GB 的差额。

确定总容量后，我们再看哪些数据需要放进去。在离散 GPU 服务器上，主机内存和显存可以分别做预算；在这台机器上，两条执行路径最终会使用同一组物理内存。各类数据的区别主要在于谁分配、怎样驻留，以及什么时候可以回收。

| 数据                         | 驻留与使用方式                                          | 影响容量的主要因素                     |
| :--------------------------- | :------------------------------------------------------ | :------------------------------------- |
| 主模型权重、MTP 参数         | 加载后由 GPU 执行路径反复读取                           | checkpoint、量化实现、MTP 配置         |
| KV Cache 与 GDN 循环状态     | 由运行时建立缓存池或状态缓冲区                          | 缓存格式、块布局、请求长度与并发       |
| CUDA Graph、激活和算子工作区 | 捕获、profiling 和 forward 期间分配，部分空间会保留复用 | 批次形状、prefill chunk、后端实现      |
| PLE 文件页                   | packed 文件通过 mmap 映射，实际访问的页进入 page cache  | 查询分布、缓存冷热、内核回收           |
| CPU 对象与传输缓冲区         | tokenizer、调度器、Python 进程及 CPU worker 使用        | 请求数、输入类型、进程与缓冲区生命周期 |
| 系统和同机服务               | 内核、驱动及其他进程使用                                | 系统状态与共存负载                     |



系统实际上的内存分配与回收机制更加复杂，分配的内存使用完后，并不一定会释放回系统。以 KV Cache 为例，vLLM 通常会在启动 profiling 后确定 KV pool 大小，并预分配相应空间。请求越来越长，首先消耗的是池中的空闲块；从整机观察，内存不一定每生成一个 token 就增长一次。

请求结束后，这些块可以释放回到 KV pool；也可能保留为可复用的前缀缓存；无论哪种情况，它们都属于已经分配好的池。因此，“请求释放了 KV 块”并不代表系统可用内存增加了。与之相比，PLE 的热页集合、首次出现的执行形状（会生成对应的 CUDA Graph）和临时缓冲区可能继续改变整机内存用量。

理解这两类变化后，我们就知道如何在项目启动的时候如何规划内存：首先估算已知内存成本并限制可分配的空间大小，其次在实际运行中定时观测内存使用的变化。



## 2. 推导 GPU 预算

项目的 [`start.sh`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/start.sh) 默认根据权重、缓存需求和主机预留推导 `gpu_memory_utilization`。这个参数简称 GMU，表示 vLLM 用于规划模型执行内存的比例。

计算从权重开始。脚本统计本地 checkpoint 快照目录的字节数，再减去通过 CPU mmap 路径访问的 PLE 表，得到 GPU 常驻权重的估计值。随后加上运行时和 MTP 预留，最后为 KV Cache 提出一个容量目标。

下面按源码顺序整理计算主体。变量均表示配置或测量输入，容量单位为 GiB；

```
#计算除开 PLE 模型权重大小
w = WEIGHT_BYTES / 2**30 - PLE_GIB

#若关闭 MTP，扣除 checkpoint 中已计入、实际却不加载的草稿权重。
w -= min(MTP_OFF_CREDIT, max(w, 0))

# 汇总本次配置下近似固定的成本：权重、运行时预留、MTP 预留。
fixed = w + OVERHEAD_GIB + MTP_GIB

# 估算一个最大长度请求的 KV 需求，并从 bytes 换算为 GiB。
# KV_BYTES_PER_TOKEN 是 BF16 基准；KV_MULT 按缓存格式修正，
# 本项目 BF16 取 1.0，FP8 取 0.58。这里没有乘以并发请求数。
kv_need = MAX_MODEL_LEN * KV_BYTES_PER_TOKEN * KV_MULT / 2**30

# 在单请求需求与配置的 KV 池目标中取较大值，再加上固定成本。
# wish 表示希望获得的总预算，最终还要服从整机容量上限。
wish = fixed + max(kv_need, KV_TARGET_GIB)

# 从 Linux 报告的总内存中扣除主机预留，得到 GPU 预算上限。
# 预留供系统、CPU 进程、PLE page cache、驱动空闲页等共同使用。
cap = MEM_TOTAL_GIB - HOST_RESERVE_GIB

# 需求超过上限时按上限分配；需求较小时只使用所需预算。
budget = min(wish, cap)

# 将容量预算换成传给 vLLM 的比例，并向下取到三位小数，
# 避免四舍五入后分配的预算超过前面算出的上限。
gmu = math.floor(budget / MEM_TOTAL_GIB * 1000) / 1000

# 用取整后的实际比例重新计算容量，保证后续估算与启动参数一致。
budget = gmu * MEM_TOTAL_GIB

# 扣除固定成本，得到预计可留给 KV 的空间。
# 这是启动前估算；最终 KV pool 大小仍由 vLLM profiling 确定。
kv_exp = budget - fixed
```

其中，`fixed` 汇总这次配置下近似固定的开销，`kv_need` 估算一个最大长度请求所需的缓存容量。`KV_TARGET_GIB` 则允许为并发和缓存复用准备更大的池，所以脚本取两者的较大值。这里没有直接乘以 `MAX_NUM_SEQS`，也就不能据此推断所有并发请求都能同时达到最大上下文。

提出需求之后，还要检查整机是否能够承担。`cap` 从总内存中扣除 `HOST_RESERVE_GIB`，剩下的作为给 GPU 预算设置的上限；`wish` 是配置固定开销加上 KV Cache 最大需求，它是 vLLM 在推理过程中实际需求的量；  如果需求超过上限，最终预算以主机预留为准。

最后使用 `budget` 和 `MEM_TOTAL_GIB`  计算出 vLLM 的参数 `gpu_memory_utilization`.

**带入真实的配置走一遍计算流程**

项目文件 [`.env.sample`](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/.env.sample)  中与内存占用相关的配置有：

```text
# 模型上下文长度 256K
MAX_MODEL_LEN=262144
# MTP 草稿 token 数量
MTP_NUM_SPECULATIVE_TOKENS=3
KV_CACHE_DTYPE=fp8
KV_TARGET_GIB=20
HOST_RESERVE_GIB=26
# 最大并行请求数
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
```

在  [03-NVFP4 权重与 FP8 KV Cache](https://mengman.github.io/2026/09/15/01-%E6%8A%80%E6%9C%AF%E7%A7%AF%E7%B4%AF/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-03/#BF16-%E6%A0%BC%E5%BC%8F%E5%86%85%E5%AD%98%E5%8D%A0%E7%94%A8) 中已经计算了以 FP16 格式保存 KV Cache 的内存占用为 29,482 bytes/token， 改用 FP8 内存占用减小为 17,194 bytes/token，比例约为 0.58。 带入上下文长度 256K。 
$$
\text{单请求估算 KV} = 262,144 × 29,482 × 0.58 / 2^{30} \approx 4.17GiB
$$
这个值小于配置为 KV Cache 预留的 20GiB，所以按照 20GiB 的预留量来计算。

```text
fixed = 71.75 + 5.60 + 1.49 = 78.84 GiB
wish  = 78.84 + 20         = 98.84 GiB
cap   = 121.69 - 26        = 95.69 GiB
```

`fixed` 固定占用部分，模型权重 71.75GiB，5.60 GiB 为运行时预留，1.49 GiB 为 [MTP 模块参数大小](https://mengman.github.io/2026/09/28/01-%E6%8A%80%E6%9C%AF%E7%A7%AF%E7%B4%AF/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-04/#3-2-MTP-%E8%BF%90%E8%A1%8C%E6%88%90%E6%9C%AC)。

`wish` 期望分配的大小为 `98.84 GiB` 超过了预留的可分配的上限，所以设置 95.69 GiB 为最终的分配内存大小。

最后计算出实际分配的比例为 0.786，内存大小为 `95.65 GiB`。

```
gmu    = floor(95.69 / 121.69 × 1000) / 1000 = 0.786

budget = 0.786 × 121.69 ≈ 95.65 GiB

kv_exp = cap - fixed = 95.65 - 78.84  ≈ 16.81 GiB
```

三位小数看起来很细，但在 121.69 GiB 的机器上，每 0.001 就约为 124.6 MiB。脚本向下取整后重新计算预算，正是为了让显示结果与传给 vLLM 的比例一致，避免把无法使用的零头也算进 KV 余量。

这里我们还可以继续计算一下，这种情况下理论上最大能并行多少个请求：

```
max_num_seqs = kv_exp / 单请求 KV = 16.81 / 4.17 ≈ 4
```



## 3. 如何把预算实际分配下去

上一节我们已经详细讨论了预算的计算流程。现在时候将内存分配预算上限通过 vLLM 的启动参数和 Docker 的资源限制参数，将预算分配给 vLLM 的推理服务。 具体的参数如下：

| 启动参数                   | 值            | 接收方 | 本项目中的用途                                    |
| :------------------------- | ------------- | :----- | :------------------------------------------------ |
| `--gpu-memory-utilization` | 0.786         | vLLM   | 传入推导出的 GPU 内存规划比例                     |
| `--memory`                 | 95.65+5 ≈ 100 | Docker | 设置容器 memory cgroup 上限                       |
| `--memory-swap`            | 95.65+5 ≈ 100 | Docker | 与 memory 上限配合，限制容器的交换空间使用        |
| `--ipc`                    | `host`        | Docker | 使用宿主机 IPC 命名空间，涉及退出时的共享内存清理 |

Docker 的 `--memory` 参数在计算出来的 vLLM 分配上限额外增加了 5GiB，确保 Docker 进程的稳定性； `--memory-swap` 与 `--memory` 设成相同值，用于禁止容器额外使用 swap。



## 5. 使用 watchdog 监控实际运行状态

启动公式使用的是配置和经验估计。实际执行还会遇到新形状、长请求、视觉输入或同机负载，部分分配也可能超出 vLLM 的规划范围。所以需要监控实际的运行状态，在服务内存消耗超过分配阈值之前，实现主动关闭/重启服务。

### 5.1 监控项目

项目文件 `files/memwatch.sh` 同时观察 `MemAvailable` 和 `MemFree`。按照 [Linux 对 `/proc/meminfo` 的定义](https://docs.kernel.org/filesystems/proc.html)，前者估计不发生交换时还能分配多少内存，会考虑部分可回收缓存；后者表示当前未使用的内存。

有了 PLE mmap，这两个数就需要结合解释。假设 `MemFree` 很低，但 `MemAvailable` 仍有几十 GiB，可能只是文件缓存占据了大量可回收页。反过来，如果 `MemAvailable` 也在下降，低 `MemFree` 就说明即时可用页与回收余量都不足。

监控程序根据 `MemAvailable` 和 `MemFree` 的状态设置了停止容器的规则。

| 条件                         | 默认判定                                     |
| :--------------------------- | :------------------------------------------- |
| 整体可用容量不足             | `MemAvailable < 6 GiB`                       |
| 可回收余量已低，且空闲页不足 | `MemAvailable < 10 GiB` 且 `MemFree < 2 GiB` |

每个条件各有一个计数器，连续命中 5 个采样才停止容器；该条件恢复时，对应计数器清零。采样循环通常间隔约 1 秒，加上命令执行时间，所以应理解为连续低水位，而非精确的五秒定时器。两条规则不需要同时成立，任何一条持续命中都可以触发保护。

watchdog 约每 10 个采样轮次检查新增的 `NV_ERR_NO_MEMORY` 日志，并记录缓存、匿名页、slab、页表等信息。驱动错误计数提供诊断线索，当前停止逻辑仍由上述低内存条件触发。

### 5.2 停机流程

真正触发停止时，watchdog 先归档容器日志和自身日志，再调用 `docker stop`，默认给服务 30 秒退出时间。超时后 Docker 才会强制终止，命令失败时脚本另有 kill 回退。

保留退出窗口与这套部署的 `--ipc host` 有关。进程若来不及清理，POSIX shared memory 可能留在宿主机 `/dev/shm` 中，使容器消失后仍有占用。优雅退出给运行时机会释放共享资源，但最终是否清理完成仍要通过系统状态确认。

watchdog 至此补上了运行期间的保护。它按间隔采样，无法保证赶在所有快速分配失败之前动作，所以预算、启动检查和持续测试仍然是必要的前置工作。



## 6. 验证流程 

经过前面的请求链路，再看配置变化就容易判断了。增大上下文会消耗更多缓存块；增加并发会增加同时保留的状态，也可能扩大图和缓冲区；增大 prefill chunk 会提高单轮计算规模，却增加峰值激活。PLE 热页、主机进程和这些 GPU 路径继续竞争同一份容量。

因此修改了一项参数之后，需要经过实际的验证，验证标准是：在实际需要的长度、并发和请求类型下，服务能持续执行，缓存容量足够，整机内存用量也能在请求之间恢复到稳定范围。

可以按下面的顺序检查：

1. **先核对生效配置与预算。** 在目标机器运行 `--no-launch`，记录 checkpoint、PLE 大小、KV dtype、MTP、上下文、并发和 chunk，确认主机上限是否压缩了 KV 目标。
2. **再看启动阶段的真实分配。** 从加载开始观察内存，记录 profiling 后的 KV block 数、图捕获峰值和 `/health` 就绪时间。将启动估算与实际缓存池分别保存。
3. **让目标请求形态真正出现。** 先测目标长度，再加入所需并发和 prefill/decode 混合流量；有图像或视频输入时，还要覆盖相应编码开销。短请求的吞吐结果不能证明多个最长请求能够同时驻留。
4. **最后观察多轮请求后的稳定性。** 同时查看延迟、`MemAvailable`、`MemFree`、缓存与匿名页、driver 趋势和错误日志，区分冷启动、PLE 预热以及后续热服务。若出现持续增长，应定位来源并调整配置后重测。

每轮尽量只修改一类参数，这样才容易把变化对应到具体机制。如果增大 chunk 后 TTFT 改善，但 KV pool 缩小，就能明确记录这次交换；如果同时改了并发、MTP 和主机预留，则很难从一个 tokens/s 数字判断是哪项起作用。



## 7. 小结

单机部署的内存问题贯穿了服务的整个生命周期。启动时，脚本从实际总量出发，扣除权重和运行时成本，在 KV 目标与主机预留之间取较小的预算；请求执行时，GPU 使用模型和历史状态，CPU 准备 PLE 数据，NVMe 与 page cache 共同提供按需访问的文件页；请求结束后，许多空间继续留在缓存池和 allocator 中等待复用。

这些驻留方式使大模型能够在有限容量里运行，也让“还剩多少内存”无法由单个数字完整回答。GMU、cgroup、主机预留与 watchdog 各自覆盖一部分问题。把预算计算与真实请求链路对照起来，再用目标负载验证峰值和持续水位，才能判断这组配置是否适合长期运行。



## 参考资料

- MiaAI-Lab：[部署项目](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)、本地 [start.sh](https://file+.vscode-resource.vscode-cdn.net/d%3A/source/EdgeInfer/Qwen3.8-Flash-Next-Single-DGX-Spark/start.sh)、[示例配置](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/.env.sample)与 [CHANGELOG](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/CHANGELOG.md)
- MiaAI-Lab：[PLE offload 补丁](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_offload.py)与 [memwatch.sh](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/memwatch.sh)
- NVIDIA：[DGX Spark System Overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)
- Linux kernel documentation：[The /proc Filesystem](https://docs.kernel.org/filesystems/proc.html)

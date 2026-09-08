---
title: 【Qwen3.8-Flash-Next 端侧部署】01-模型架构与整体方案
date: 2026-09-08T09:23:47.791Z
tags: [qwen, vllm, aiinfra, dgx-spark]
categories: aiinfra
typora-root-url: ../../../
---

Qwen3.8-Flash-Next 的主模型有 125B 参数，另外还有 51B n-gram embedding 参数。把这样一个模型放到只有 128GB 统一内存的 DGX Spark 上运行，问题不只是选择一种更低精度的权重量化格式。模型权重、PLE 查找表、KV Cache、CUDA 工作区和 Linux 文件页缓存最终会竞争同一个内存池，任何一项估算得过于乐观，都可能让服务在加载或长请求阶段耗尽系统内存。

`Qwen3.8-Flash-Next-Single-DGX-Spark` 项目给出了一套可以实际运行的单机部署方案：主模型使用 NVFP4 权重；PLE 查找表不常驻 GPU，而是放在本地 NVMe 上，由 CPU 按需读取；全局注意力层的 KV Cache 使用 FP8；启动脚本再根据系统容量为 vLLM、KV Cache 和主机进程划定边界。

这篇文章先介绍模型结构、DGX Spark 的硬件条件和部署整体方案。PLE 的文件布局与 CPU/GPU 协作、NVFP4 与 FP8 KV Cache 的实现、MTP 和 CUDA Graph 的配置，以及统一内存的详细预算，会在后续文章中分别展开。

## 资料来源与致谢

本文的模型架构部分主要依据 Qwen Team 发布的 [Qwen3.8-Flash-Next 架构介绍](https://qwen.ai/blog?id=qwen3.8-flash-next)和配套的[技术报告](https://arxiv.org/abs/2608.30320)。部署部分基于 MiaAI-Lab 开源的 [Qwen3.8-Flash-Next-Single-DGX-Spark 项目](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)，并结合仓库中的启动脚本、vLLM 补丁和实测记录进行分析。

感谢 Qwen 团队公开模型权重、架构说明和实验结果，也感谢 MiaAI-Lab 将单台 DGX Spark 上的部署过程、踩过的内存问题和相应补丁整理成可复现的开源项目。本文是对这些工作的独立解读；项目实现和测试数据的原始出处归相应团队所有。

## 先明确部署对象

Qwen3.8-Flash-Next 是一个多模态 MoE 模型。按照 Qwen 官方给出的规模，它包含 125B 主模型参数和额外的 51B n-gram embedding 参数，每个 token 实际激活约 6B 参数。它原生支持 262,144(256K) token 上下文，并可通过 YaRN 继续扩展。

这里需要把四个容易混在一起的数字分开：

| 数字       | 表示什么                               | 对部署的影响                                 |
| ---------- | -------------------------------------- | -------------------------------------------- |
| 125B       | Transformer 主干及相关模块的参数规模   | 决定主模型权重的大致存储量                   |
| 51B        | 额外的 n-gram embedding 参数           | 参数很多，但每个 token 只查少量表项          |
| 6B         | 每个 token 激活的参数量                | 更接近一次前向计算的主要矩阵计算规模         |
| 约 98.6GiB | 本项目使用的 NVFP4 checkpoint 文件规模 | 是磁盘上的模型文件大小，不是完整的运行时内存 |

MoE 的作用是让每个 token 只经过一部分专家，因此 125B 主模型不需要在每次前向中把所有参数都用于矩阵计算。但没有被激活的专家权重仍然需要保存在某处。参数稀疏降低了计算量，不会自动让权重从内存中消失。

同样，51B n-gram embedding 参数也不能简单加到“每 token 激活参数”中。这部分参数的使用方式不是大规模矩阵乘法，而是根据局部 token 上下文定位少量表项。这个访问特征使它可以采用不同于主模型权重的存储方案，也成为单机部署能够成立的重要条件。

## 模型结构如何影响推理

Qwen3.8-Flash-Next 不是在标准 Transformer 上简单扩大 MoE。它对 token mixing、残差连接和 embedding 都做了调整。理解这些结构后，才能解释为什么它的 KV Cache、PLE 和运行时状态需要分别处理。

![Qwen3.8-Flash-Next 模型架构图](/image/01-技术积累/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-01/model-architecture.png)

### GDN 与 QSA 交替出现

模型采用 Gated DeltaNet（GDN）和全局注意力组成的混合结构。官方说明是每四层中有三层使用 GDN，剩下一层使用全局注意力；在 Qwen3.8-Flash-Next 中，全局注意力进一步采用 Qwen Sparse Attention（QSA）。按项目所对应的模型配置，可以把它理解为 36 个 GDN 层和 12 个全局注意力/QSA 层交替组成主干。

GDN 和标准自注意力处理历史信息的方式不同。

标准全注意力为了让当前 token 直接访问过去的 token，需要保存历史位置的 K 和 V。上下文越长，KV Cache 越大，注意力计算和内存访问也随之增长。GDN 则把历史信息持续压缩到固定大小的状态中，不需要为每一个历史 token 保存一份标准 K/V。

但只依靠固定状态不容易精确取回很久以前的某个细节，因此模型每隔几层仍然安排一个全局注意力层。QSA 先用轻量索引器把上下文压缩成 micro-block，并从中选出重要区域，再执行注意力计算。它减少了长序列上的注意力计算量和索引开销，同时保留了从全局上下文中精确检索信息的路径。

![QSA 模块结构](/image/01-技术积累/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-01/qsa_arch.png)

这个混合结构对部署有两个直接影响。

第一，只有一部分层需要标准的全局注意力 KV Cache，因此缓存规模小于同层数、全部采用全注意力的模型。第二，KV Cache 并没有消失：12 个全局注意力/QSA 层仍然需要保存和读取 K/V；GDN 层也有自己的递归状态和辅助缓存。后面讨论 FP8 KV Cache 时，必须把这几类状态分开计算。

### Gated Residual 将残差流扩展为四个分支

普通 Transformer 通常围绕一条 residual stream 工作，每一层从中读取 hidden states，完成计算后再写回同一条残差路径。Qwen3.8-Flash-Next 的 Gated Residual（GR）把残差状态扩展为四个并行分支，并通过与当前内容相关的 gate 控制各分支的读取和写入。

从模型设计角度看，多分支残差让不同信息沿不同路径跨层传播，减少早期特征在深层网络中被反复混合后逐渐稀释的问题。Qwen 的实验还观察到，其中一个分支会形成从早期注意力层通向中后层的长程路径。动态 gate 同时有助于抑制激活异常值和改善训练稳定性。

从推理角度看，层间状态不再只是一个简单的二维 hidden-state 张量。运行时需要维护多分支残差状态，并让相关算子正确处理这些读写关系。GR 本身不是本项目节省几十 GiB 内存的主要手段，但它会影响模型执行、激活存储和 CUDA Graph 捕获，不能把该模型完全当成普通 Transformer 来估算。

### n-gram embedding 是一条附加的条件记忆路径

标准 token embedding 根据单个 token ID 查询一个向量。n-gram embedding 使用当前 token 和前面若干 token 组成的局部上下文生成查找键，因此可以为常见短语、代码片段和局部模式提供更有针对性的表示。

Qwen3.8-Flash-Next 为此增加了 51B 参数，但最终只在网络靠前位置使用一个 n-gram embedding 层。这一点很重要：这里不是每个 Transformer layer 都配置一套 51B 查找表，也不是把每一层的 hidden states 送到 CPU 做 embedding。

推理时，模型层之间仍然传递 hidden states。例如长度为 512 的输入经过 token embedding 后，可以表示为一个 `[512, hidden_size]` 的张量；后续层继续在这类连续向量上计算。与此同时，推理引擎保留原始 token ID、位置和序列边界，根据这些离散信息计算 n-gram ID，再从大表中读取相应向量。查找结果经过反量化、投影、门控和短卷积后，与 GPU 上的 hidden states 融合。

因此，CPU 可以独立完成的是 n-gram ID 计算和大表查询，而不是整个 PLE 模块。后续融合仍然依赖当前隐藏状态和 GPU 上的模型参数。

这条路径的价值在于，它用大量、稀疏访问的存储容量补充主干模型，而每个 token 只触发少量查找，不需要让全部 51B 参数参与矩阵计算。Qwen 官方把 n-gram embedding 描述为可以放在 host memory 并与模型计算异步预取的参数。MiaAI-Lab 的项目进一步结合 DGX Spark 的内存条件，把查找表做成紧凑文件，放到本地 NVMe，通过 mmap 由 CPU worker 按需访问。

项目和 vLLM 代码中使用了 `PLE` 这个名称。它受到 Per-Layer Embedding 等工作的启发，但在当前 Qwen 模型中，实际需要关注的是靠前位置的单个 n-gram embedding 模块。后续文章会继续沿用项目中的 PLE 命名，同时明确它和严格意义上的 Per-Layer Embedding 不是同一个结构。

### MTP 用额外计算换取更少的解码步骤

Multi-Token Prediction（MTP）模块尝试在一次模型执行中提出多个后续 token，再由主模型验证这些草稿。草稿中连续通过验证的 token 可以一次加入输出，从而减少生成同样长度文本所需的 engine step 数量。

MTP 并不会保证固定加速。可预测文本通常有更高的草稿接受率，随机性更强或多模态输入下的收益可能较低。它也不是免费的：草稿模块有自己的权重和输出头，验证多个位置会增加临时计算，CUDA Graph 需要覆盖相应的执行宽度。

本项目的示例配置启用 3 个 speculative tokens，并为 MTP 估算约 1.49GiB 的额外常驻开销。MTP 是否值得启用，不能只看 tokens/s，还要放到统一内存预算和实际请求类型中判断。

## DGX Spark 的 128GB UMA 可以怎样使用

DGX Spark 使用 GB10 Grace Blackwell Superchip，包含 Arm CPU、Blackwell GPU 和 128GB LPDDR5x coherent unified system memory。NVIDIA 的[硬件说明](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)强调 CPU 和 GPU 共享这套统一内存，这与常见服务器上“系统内存加独立 GPU 显存”的结构不同。

统一内存给这次部署提供了可能性。模型权重不必受限于一块 48GB 或 80GB 独立显存，CPU 和 GPU 也可以围绕同一个物理内存池交换数据。对于 PLE 这种 CPU 查表、GPU 融合的工作流，这种结构减少了传统离散设备之间的一部分数据管理复杂度。

但 128GB UMA 并不等于应用可以获得 128GB 的空白 GPU 显存。这个内存池还要容纳：

- Linux 内核和基础服务；
- vLLM 的 Python 进程、CPU worker 和共享内存；
- GPU 驱动、CUDA context、allocator 和算子工作区；
- 模型权重和 MTP 权重；
- CUDA Graph 捕获后保留的内存；
- KV Cache 和 GDN 相关状态；
- PLE mmap 文件实际访问后进入内存的文件页；
- 容器内其他主机侧分配；
- 同一台机器上运行的其他进程。

因此，UMA 同时带来了容量共享和资源竞争。GPU 多占用一部分，主机可用内存就会减少；PLE 文件页进入 page cache，也会挤压模型运行空间。模型可以成功加载，并不代表它能稳定处理长上下文或连续请求。

项目记录的 Linux `MemTotal` 约为 121.69GiB，而不是按十进制标称容量直接得到的 128GiB。当前示例配置进一步设置 `HOST_RESERVE_GIB=26`，把 GPU 侧预算限制在大约 95GiB以内，为系统、vLLM 主机进程、PLE page cache、驱动所需空闲页和运行期增长留下空间。

## 运行时内存占用

项目使用的 `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` checkpoint 约为 98.6GiB。这个数字容易造成一个直觉：模型文件小于 128GB，似乎直接加载即可。问题在于 98.6GiB 只是 checkpoint 文件的总体积，其中既包含适合在 GPU 侧执行的权重，也包含约 26.82GiB 的 PLE 表；服务运行时还会产生文件中没有的 KV Cache、工作区和进程内存。

项目启动脚本给出的主要内存项如下。数字采用当前仓库中的测量值和估算值，后续版本或不同 checkpoint 可能变化。

| 内存项目                   | 约占用   | 处理方式                              |
| -------------------------- | -------- | ------------------------------------- |
| 完整 NVFP4 checkpoint      | 98.6GiB  | 存在本地 Hugging Face cache 中        |
| 其中 PLE 表                | 26.82GiB | 重新打包后放在 NVMe，由 CPU mmap 查询 |
| GPU 侧模型权重             | 71.75GiB | 保留在 GPU 可访问的统一内存中         |
| vLLM/CUDA 运行时开销       | 5.6GiB   | 启动预算中的测量估计                  |
| MTP（启用 3 个草稿 token） | 1.49GiB  | 计入固定预算                          |
| FP8 KV Cache 目标          | 16GiB    | 受主机预留上限约束                    |
| 主机预留空间               | 26GiB    | 留给系统、主机进程、页缓存和增长余量  |

其中有两个地方需要注意。

第一，PLE 表从 GPU 权重预算中减掉，并不表示这 26.82GiB 完全不占内存。packed 文件存放在 NVMe，只有实际访问的页进入 page cache；如果访问提示或回收策略不合理，文件缓存仍然会消耗 UMA。这里节省的是“整张表长期常驻 GPU 可用空间”，不是凭空消除数据。

第二，`KV_TARGET_GIB=16` 是希望留给 KV Cache 的目标，不是一个脱离其他配置的固定结果。vLLM 能建立多大的缓存池，还受到模型权重、运行时 profiling、CUDA Graph、MTP、上下文长度和主机预算上限影响。项目启动脚本会根据实时 `MemTotal` 和配置重新计算，而不是假定所有机器都能得到完全相同的结果。

按照项目脚本的计算方法，GPU 侧预算大致可以写成：

```text
需要的预算 = GPU 侧权重
           + vLLM/CUDA 运行时开销
           + MTP 开销
           + max(上下文所需 KV，KV Cache 目标)

最终预算 = min(需要的预算，MemTotal - 主机预留空间)
```

脚本再把最终预算换算成 vLLM 的 `--gpu-memory-utilization`。在项目记录的机器上，26GiB 主机预留使预算上限约为 94.87GiB，对应约 0.780 的利用率。这里的 0.780 不是保守地只用某块独立显存的 78%，而是在整个统一内存池上为主机侧明确留出空间。

## 部署整体方案

到这里，可以把项目的做法归纳成一条比较清楚的原则：先按访问方式给数据分类，再决定存储位置和精度，而不是把整个 checkpoint 当成一种同质数据。

### 主模型权重留在 GPU 可高效访问的位置

MoE、注意力、GDN 和视觉塔中的矩阵权重会在推理过程中反复使用，适合由 GPU 执行。项目使用 NVFP4 checkpoint 降低权重容量，并通过 ModelOpt 相关计算路径让大部分量化权重直接参与 GPU 运算。

这部分不能像 PLE 一样简单放到 NVMe 上逐层现取。矩阵权重的访问量和复用方式与离散查表完全不同，如果每层都等待 CPU 或存储设备提供大块权重，推理速度会受到严重影响。项目最终让约 71.75GiB 非 PLE 权重留在 GPU 可访问内存中。

少数形状不满足现有 MXFP8 内核要求的线性层采用局部 BF16 模拟，而不是把整个模型反量化为 BF16。这样可以处理兼容性问题，又不会让全部权重膨胀到单机无法承受的规模。

### PLE 表放到 NVMe，由 CPU 按需查询

PLE 访问由 token ID 和局部上下文确定，每次只读取大表中的少量行。项目在首次启动时把相应参数整理成约 27GB 的 packed 文件，之后通过 mmap 访问。CPU worker 计算 n-gram ID、找到文件偏移并收集表项，GPU 收到数据后完成后续反量化与融合。

为了避免随机查询触发过量顺序预读，项目对映射区域使用 `MADV_RANDOM`；prefill 阶段又根据整段已知输入，通过批量 `posix_fadvise(..., WILLNEED)` 提前请求需要的页面。这两个策略分别限制无效读取和利用可预知访问，目标都是减少 NVMe 等待和 page cache 浪费。

GB10 不支持原 vLLM PLE offload 路径所依赖的一类 CUDA stream memory operation，因此项目还替换了 CPU/GPU 同步机制。GPU 发布查表请求，CPU worker 完成 gather 和复制后更新共享完成标记，GPU 再继续执行。这会牺牲一部分原本期望的异步重叠，但避免了在该硬件上死锁。

这些实现将在 PLE 专题中结合数据布局和 prefill/decode 时序详细说明。

### 全局注意力层使用 FP8 KV Cache

KV Cache 与模型权重不同，它会随已处理 token 数和并发序列数增长。在长上下文部署中，即使量化权重已经能装入内存，KV Cache 仍可能成为服务容量的主要限制。

项目让 QSA kernel 读取 FP8 E4M3 格式的 K/V，在计算时将所需 tile 转成 BF16，并应用相应的 K/V scale。这样避免长期用 BF16 保存全部主 KV Cache。由于 GDN side cache 和其他辅助状态仍可能使用 BF16，整体缓存收益会低于只比较单个元素字节数得到的理论两倍，但仍能明显增加可容纳的 token 数。

当前示例配置以 16GiB 作为 FP8 KV Cache 目标，服务原生 262,144 token 上下文。更长的 YaRN 上下文可以配置，但会缩小并发余量，并需要重新验证长请求下的内存稳定性。

### vLLM 推理加外挂优化

项目没有重新实现一套推理引擎，而是在 vLLM 上提供模型适配和针对 GB10 的补丁。服务采用单 GPU、TP=1，对外暴露 OpenAI 兼容 API；Docker 使用 host network 和 host IPC，并挂载本地 checkpoint、packed PLE 文件及生成后的补丁文件。

启动前，`start.sh` 会检查 checkpoint shard 是否完整、PLE 表能否生成或找到，以及当前机器是否有明显的竞争进程。它拒绝在单台 Spark 上关闭 PLE offload，因为完整 PLE 表与其余权重一起进入统一内存会破坏现有预算。

随后脚本根据模型权重、运行时估计、MTP、KV Cache 需求和 `HOST_RESERVE_GIB` 计算 GPU budget，再传给 vLLM。容器的 cgroup 内存限制用于约束主机侧进程，但项目明确指出，它不能代替 vLLM 的 GPU 预算，也不能完整限制 GB10 上的 GPU 分配。

[`start.sh`](./start.sh) 最终构造的 vLLM 参数把这些决策落实为命令行选项：

```bash
VLLM_ARGS+=("--tensor-parallel-size" "1")
VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
VLLM_ARGS+=("--enable-chunked-prefill")
VLLM_ARGS+=("--distributed-executor-backend" "mp")
```

其中 `mp` 不是普通的部署偏好：项目注释说明，只有 multiprocess executor 才会创建 PLE offload worker。

服务运行后，独立 watchdog 持续观察 `MemAvailable` 和 `MemFree`。只有连续多次低于阈值才触发停止，避免把一次短暂波动误判为持续内存压力。这个机制负责在异常增长时保住系统响应能力，但它只是最后的保护；真正决定服务能否稳定运行的仍然是启动前的预算和运行参数。

## 推理流程

下面这张简图省略了模型内部的大部分算子，只保留影响部署的几条数据路径：

![组件调用流程](/image/01-技术积累/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-01/call_flow.png)

在 prefill 阶段，输入 token 已经确定，CPU 可以批量计算 n-gram ID 并预取 PLE 表项，GPU 则按分块执行主模型并建立缓存。在 decode 阶段，新 n-gram 依赖刚刚生成或确认的 token，CPU 查询和 GPU 前向形成逐步依赖，卸载带来的延迟更难完全隐藏。MTP 尝试让一次验证接受多个 token，以减少 engine step 数量。

这也说明了各项优化之间的关系：PLE offload 释放固定容量，FP8 KV Cache控制动态增长，MTP 和 CUDA Graph 改善 decode 执行，Chunked Prefill 控制长输入阶段的峰值，而统一内存预算决定这些功能能够使用多少空间。

## 这套方案解决了什么问题

单台 DGX Spark 能运行这个模型，不意味着 176B 级模型可以无条件装入 128GB 内存。这里成立的是一个更具体的部署条件：

- 使用约 98.6GiB 的 NVFP4 checkpoint，而不是 BF16 原始权重；
- 从 GPU 侧权重中移出约 26.82GiB、访问稀疏的 PLE 表；
- 让约 71.75GiB 主模型权重保留在适合 GPU 计算的位置；
- 使用 FP8 保存主要 KV Cache，为原生长上下文和有限并发提供容量；
- 对不兼容的量化算子做局部回退，不进行全模型 BF16 展开；
- 对 vLLM 的预算设置主机侧上限，并为系统、CPU worker 和 page cache 预留空间；
- 接受 PLE 随机读取和主机握手引入的部分延迟，以换取单机可部署性。

它更适合作为本地研发、模型验证和低并发长上下文服务方案，而不是多租户、高并发数据中心推理的替代品。NVMe 性能、系统中其他进程、请求上下文长度和 MTP 接受率都会影响实际结果。

## 小结

Qwen3.8-Flash-Next 的架构本身为这种部署提供了空间：MoE 控制每个 token 的矩阵计算量，GDN 与 QSA 的混合减少全量 KV Cache 和长序列注意力成本，单个 n-gram embedding 模块则允许把大量条件记忆放在主干计算之外。但这些特性不会自动适配一台 128GB UMA 机器。

MiaAI-Lab 项目的主要工作，是把模型结构提供的可能性落实为一套具体的数据放置和运行策略。它没有把所有东西都塞进所谓的“共享显存”，而是让高复用权重、稀疏查找表、动态缓存和主机运行空间采用不同的管理方式。

下一篇将专门分析 PLE：packed 表中的 NVFP4 code 和 FP8 scale 如何组织，mmap 与 Linux page cache 如何工作，CPU 能提前完成哪些计算，以及 prefill 和 decode 阶段为什么具有不同的 CPU/GPU 协作方式。

## 参考资料

- Qwen Team：[Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency](https://qwen.ai/blog?id=qwen3.8-flash-next)
- Qwen Team：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320)
- MiaAI-Lab：[Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
- NVIDIA：[DGX Spark System Overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)

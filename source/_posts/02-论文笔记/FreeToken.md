---
title: 【FreeToken 解读】边缘侧推理框架 FreeToken 论文笔记
date: 2026-08-30T08:00:07.380Z
tags: [LLM, EdgeInfer]
categories: paper
typora-root-url: ../../image/02-论文笔记/FreeToken
---

## 五分钟全景图 

### **一句话概括**

FreeToken 是一个**边缘原生的 MoE（混合专家）模型推理系统**，它不再把个人电脑（如游戏本、工作站）当作"小 GPU"，而是将其视为**由 GPU、CPU、内存、PCIe 组成的异构推理平台**，通过**带宽自适应执行**、**语义感知缓存**和**弹性内存管理**，让消费级硬件能够交互式运行原本只在数据中心才可行的前沿规模 MoE 模型（35B~753B 参数）。

### **三大创新点**

1. **带宽自适应执行（Bandwidth-Adaptive Execution）**：不再"全量 offload 到 GPU"或"全量在 CPU 上算"，而是在每个 decode 步骤中，根据实时测量的 **PCIe 带宽** 和 **CPU 专家处理带宽** ，动态计算最优的 miss 专家分割比例 ，将部分 miss 专家通过 PCIe 填入 GPU 缓存，其余直接在 CPU 上就地计算，两者并发执行，最大化利用两种带宽。
2. **语义感知状态缓存（Semantic-Aware State Caching）**：针对 Agent 工作负载频繁编辑上下文（删除 tool output、thinking segment）的问题，FreeToken 在**特殊 token 边界**（如 `<tool_call>`、`<thinking>`）锚定**循环状态（recurrent state）检查点**。当上下文被编辑时，只需从最近的存活锚点重新计算新后缀，避免了整段长上下文的重复预填充。
3. **弹性边缘资源管理（Elastic Edge Resource Management）**：消费级 GPU 的 VRAM 被桌面合成器、浏览器等共享，可用预算动态变化。FreeToken 允许在运行时**无需重启引擎**即可调整 GPU 专家缓存和 KV 缓存的大小比例，并实现**快速冷启动**（直接加载 FTW 格式权重，跳过 GPU warm-up）。

### **启发**

- **MoE 推理不应只关注"预测命中率"**：现有工作（MoE-Infinity、ProMoE）大多优化"如何预测下一个专家"，但预测再准，miss 的专家最终还是要走 PCIe。FreeToken 的 insight 是：**与其消除 miss，不如让 miss 被服务得更高效**——把 CPU 当作计算资源而非存储仓库，是解锁边缘推理性能的关键。
- **Agent 工作负载是"系统性"挑战，而非"单次推理"挑战**：Agent 的上下文编辑、工具调用、多轮对话使得 KV cache 复用率骤降。FreeToken 的语义锚点方案很务实：不追求通用 prefix tree 的最大复用，而是利用 Agent 框架本身的**结构化编辑模式**（只删除整块语义单元）来设计缓存策略。
- **消费级硬件推理的工程天花板在于软件栈**：FreeToken 将调度决策（路由去重、缓存命中、miss 分割、victim 选择）全部以 **设备端数据（device-side data）** 的形式固化到 CUDA Graph 中，避免了 host-device 同步开销。这提示我们：在边缘场景，**kernel 融合和 graph capture 的工程深度**往往比算法 novelty 更能决定最终性能。



## 1 Introduction

**开源 MoE 模型的能力在快速逼近闭源 SOTA，但运行它们的硬件门槛（数据中心级 GPU 集群）并未降低**。FreeToken 的突破点在于：它不把"100M 台消费级 GPU 机器"视为硬件问题，而视为**系统软件缺失问题**。它提出将个人机器视为"统一弹性推理平台"，通过软件栈的端到端协同设计（模型布局、专家驻留、CPU-GPU 执行、状态复用、运行时内存管理）来填补这一鸿沟。

论文从三个技术方面来讨论的当前端侧推理系统(llama.cpp，kTransformers，Ollama 等)对于 MoE 部署的存在的技术问题：

1. **Prefill 阶段稠密化**。虽然每个 token 在只激活几个专家网络，但是因为超长的输入序列，还是会激活每个 MoE 层中的所有专家网络，使得 MoE 的计算不再具有稀疏性。让 prefill 阶段同时具有**计算密集性**和**内存-显存参数搬运**的挑战性。
2. **Decode 阶段专家网络的动态性**。现有的推理框架的专家调度策略无法很好的处理 miss 的专家网络，它们的策略要么是(llama.cpp)模型参数加载时，固定加载一部分到显存中；要么是动态加载 miss 的专家。 在 CPU、PCIe 带宽、GPU 三者之间没有做到高效的利用。
3. **边缘侧资源的多样性**：由于消费级设备的多样性（不同的 CPU、PICe带宽、内存大小、GPU算力等），动态性（同时和其他程序一起调度资源），放大了解决前两个问题的技术难度。


FreeToken 提出的解决方案：

1. **Bandwidth-adaptive execution**: 将边缘设备有限的带宽瓶颈转化为运行时的调度信号；在 Prefill 阶段通过双缓冲区机制，交替加载当前专家网络参数和下一层专家网络参数，通过计算来掩盖参数加载的延迟。Decoding 阶段通过 PCIe 的带宽，来调整专家网络加载到 GPU 中运算与停留在内存中交给 CPU 运算的比例。
2. **Semantic-aware caching**: 在 Prefill 阶段通过语义上的标签位置来缓存 KV cache 解决 Agent 重写导致的 fix-Windows-size attention 模块的缓存失效；在 Decoding 阶段，基于**相邻token专家路由的局部性**，使用 LRU 策略来调度专家网络。
3. **Elastic edge resource management:**  为了适配消费级设备的使用场景，在不重启的推理框架的情况下，动态的调整推理所需要的显存资源。



## 2 Challenges in Edge MoE Serving

### 2.1 Challenges in the Prefill Stage: Transfer and Recomputation Costs

Prefill 阶段的瓶颈来自两个方向：

1. **专家传输（Expert Transfer）**：Prefill 阶段涉及数千个 token，每个 token 路由的专家合集几乎覆盖全部专家。因此，每一层都必须从 CPU 内存通过 PCIe 传输**几乎整个专家池**。对 DeepSeek-V4-Flash（FP4, ~140GB），这一传输在 RTX 5090（PCIe 5.0 x16, ~60 GB/s）上耗时 ~2s，在 RTX 4060 笔记本（PCIe 4.0 x8, ~11.8 GB/s）上超过 10s。
2. **上下文重计算（Context Recomputation）**：Agent 每轮 tool call 都会编辑上下文（删除旧 thinking、压缩 tool output）。混合注意力模型（含循环层或滑动窗口）依赖稀疏的 state checkpoint，一旦编辑点落在 checkpoint 之后，就必须从最近的有效 checkpoint 重新计算数千个 token。消费级 GPU 的 BF16 算力仅为 H100 的 1/5~1/10，重复计算开销巨大。

> 在包含循环层(Recurrent Layer) 或者固定窗口大小的 Attention 层网络中，**模型将之前的所有状态压缩成一个固定大小的状态向量**，而不是像标准 Attention 中按照 token 逐个保持 KV cache，这导致 Agent 在上下文重写之后，循环层无法复用之前的状态，需要从头开始计算。



### 2.2 Challenges in the Decode Stage: Cache Misses and Limited CPU Bandwidth

Decoding 阶段的瓶颈归因于三个递进问题：

1. **静态专家放置几乎总是 miss**：llama.cpp 在 load 时固定 tensor 位置，KTransformers 在 prefill 时决定"hot expert"，但路由随每个 token、每个工作负载变化。冻结的 placement 只能覆盖很小一部分 routed traffic，多数专家最终落到 CPU 上执行，GPU 和 PCIe 闲置。
2. **消费级 CPU 带宽无法独立支撑 decode**：双通道 DDR4 (~50 GB/s) 或 DDR5 (~80-90 GB/s) 的带宽远低于 GPU 显存带宽（RTX 4090/5090 为 1-1.8 TB/s）。CPU-only 路径的 decode 速度被 DRAM 带宽封顶，与核心数无关。
3. **最优工作划分是硬件特定的**：miss 专家可以通过 PCIe 传到 GPU 执行，也可以在 CPU 就地执行。两种方式没有绝对优劣：PCIe-only 方案在 host 带宽充裕时浪费 CPU 算力，CPU-only 方案在 PCIe 带宽充裕时浪费未来 cache hit 的机会。最优比例依赖于具体硬件的 和 实测值。



### 2.3 Challenges in Resource Management: Nothing on the Edge Is Dedicated

边缘设备的资源状态具有高度动态性：

- **VRAM 预算及分配随时变化**：GPU 与桌面合成器、浏览器、游戏共享，可用 VRAM 在不同启动时刻不同，甚至在同一会话中也可能突然减少或增加。KV cache 随 Agent 会话增长，专家 cache 相对固定，所以"在首个 turn 选择的 split"在多个 turn 后已经不再最优。
- **引擎冷启动慢且频繁**：DeepSeek-V4-Flash 的 ~140GB 专家池从 NVMe（~7 GB/s）加载就需要 ~20s，且边缘用户经常按需启动/关闭引擎或切换模型，启动延迟成为用户体验瓶颈。



## 3 FreeToken Design

### 3.1 Prefill Codesign: Pipelined Loading and Semantic-Aware State Caching

针对 §2.1 的两个预填充瓶颈，FreeToken 提出两个机制：

1. **全层双缓冲（Full-Layer Double Buffering）**：由于 prefill 阶段几乎激活每层的全量专家，FreeToken 放弃"按需加载"，而是分配两个 full-layer buffer。GPU 在计算第 $l$ 层路由专家的同时，DMA 流并行加载第$l+1$层的**全量专家**到另一个 buffer。因为传输与计算重叠，prefill 总时间近似等于单次全量专家池的传输时间，专家计算被完全隐藏。
2. **语义锚点状态缓存（Semantic-Aware State Caching）**：对混合注意力模型中的循环层（recurrent layer），FreeToken 在**特殊 token 边界**（`<thinking>`、`<tool_call>`、`<tool_output>`、conversation turn）锚定循环状态 checkpoint。Agent 框架（如 OpenClaw、SWE-agent）总是以这些语义块为单位进行截断/删除，因此锚点 checkpoint 大概率在编辑后仍然有效。当新请求到达时，FreeToken 从最深的存活 checkpoint 恢复，只对新的 suffix 重新预填充。

![image-20260905180918717](/fig2.png)



### 3.2 Decode Codesign: Semantic-Aware Expert Caching and q* Policy

这是论文最核心的技术贡献，解决 §2.2 的 decode 瓶颈：

1. **语义感知专家缓存（Semantic-Aware Expert Caching）**：FreeToken 维护一个**跨所有 MoE 层的全局 LRU 专家缓存**，缓存内容随路由动态变化。每个 cache hit 刷新专家的 recency，cache miss 时将新选中的专家填入（驱逐 LRU 专家）。这与 KTransformers 的 “prefill 时 pin hot expert” 形成本质区别：缓存**持续跟随 token 级别的路由变化**，而非依赖于 load-time 或 prefill 阶段的静态预测。

2. **带宽自适应执行（Bandwidth-Adaptive Execution）**：设第 t 步有 m 个 miss 专家。FreeToken 将 miss 集合分为两部分：

   - $\mathcal{F}$：通过 PCIe 填入 GPU cache，然后在 GPU 上执行（大小 q）
   - $\mathcal{C}$：直接在 CPU 上就地执行（大小 m - q）

   两个分支**并发执行**。设 $B_P$ 为实测 PCIe 专家传输带宽，$B_H$ PCIe 的总带宽，则留给 CPU 处理 miss experts 的 PCIe 带宽为：
   $$
   B_R = \max(B_H - B_P, 0)
   $$
   

   两分支完成时间分别为：
   $$
   T_{\text{fill}}(q) \approx \frac{q \cdot S}{B_P}, \quad T_{\text{cpu}}(m-q) \approx \frac{(m-q) \cdot S}{B_H - B_P}
   $$
   因为计算时长由最慢的那个决定，所以应该平衡两分支，可得最优分割比：（平衡的意思就是 $\frac{T_{fill}}{T_{cpu}} = 1$）
   $$
   q^* \approx m \cdot \frac{B_P}{B_H}
   $$
   

   当 $B_H \approx B_P$ 时，$q^* \approx m$，系统退化为纯 PCIe 填充模式。

因为 Decoding 阶段主要是 memory bound ，所以处理时间约等于数据传输时间。

* 即使 CPU 能处理大部分 miss 专家，仍保留一个专家的  PCIe 传输，确保缓存持续预热，避免冷启动陷阱。

* PCIe 传输和 CPU 计算**共享同一 host 内存带宽**。GPU DMA 从内存读取权重占用的 host PCIe 带宽为  $B_P$ ，剩余的 $B_H - B_P$ 带宽才能被 CPU 核心用于读取权重并计算。因此，如果 $B_H = 80\text{ GB/s}，B_P = 50\text{ GB/s}$，则 PCIe 饱和时 CPU 仍可用剩余的 30 GB/s 处理 miss 专家，这 30 GB/s 的"残差带宽"如果不被利用就浪费了。
* **$B_P$ 和 $B_H$ 的实测差异**：从 Table 1 看，RTX 4060 笔记本的 $B_P = 11.8$，$B_H = 47.5$，故 $q^* \approx 0.25m$（约 1/4 miss 走 PCIe，3/4 走 CPU）；而 RTX 5090 桌面的 $B_P = 49.0$，$B_H = 53.8$，故 $q^* \approx 0.91m$（几乎全部走 PCIe）。这解释了为何不同硬件的最优策略差异巨大。
* **并发执行 vs 串行执行**：公式假设两分支完全并发，且 CPU 分支不会因 PCIe 传输而 stall（因为 DMA 控制器独立于 CPU 核心）。如果 host 是统一内存架构（UMA）且 DMA 与 CPU 争用内存控制器，实际并发度会略低，但 FreeToken 通过实测带宽已经隐含了这一因素。



### 3.3 Elastic Memory Management for Edge-Native Runtimes

针对 §2.3 的资源动态性问题，FreeToken 提供两种机制：

1. **运行时缓存重配置（Runtime Cache Reconfiguration）**：在加载完成非 MoE 层的权重之后，FreeToken 将剩下的显存划分为 KV cache page 和专家缓存（complete-expert slot）, 它们的大小是动态调整的。在任意调度安全点（scheduler safe point），FreeToken 可以在不重启引擎、不重载 CPU 专家池的前提下，重新构建 GPU 专家缓存的 size 以及其与 KV cache 的 split。重新配置后，系统重新建立 captured execution path。
2. **快速引擎启动（Fast Engine Bootstrap）**：
   - 加载优化：FreeToken 提供 **FTW（FreeToken Weight）格式**，将专家权重预先合并为 runtime bank layout。加载时用并行 direct I/O 直接读到最终 host 布局，加载完成后再 pin 内存（避免先 pin 空 buffer 导致的零页故障开销）。
   - 无需 warm-up：第一个请求以 cold cache 状态服务，miss 由 §3.2 的普通 decode 路径处理，cache 在正常服务过程中逐渐加热。



## 4 Implementation

### 4.1 CUDA-Graph-Compatible LRU Cache

FreeToken 的核心工程挑战：**专家缓存是高度动态的**（miss 专家集合、fetch 数量、移除的专家每步都在变化），如果由 host 控制，每一步都需 host-device 同步，破坏 CUDA Graph 的 zero-overhead replay 优势。FreeToken 的解决方案：**将所有路由依赖的控制逻辑表示为 device-resident data，固化在 statically captured graph 内**。

- 对每个 MoE 层，单个 GPU kernel 完成：**去重 routed experts → 查 residency table → 计算 → 选择 eviction victims → 将 logical expert ID 重写为物理 slot ID（或 CPU-assignment flag）**。
- Victim selection 用**单趟扫描**识别 K 个 LRU 候选，miss 路径消耗前 个，避免了"逐 slot 扫描"的 LRU 陷阱。
- 结果 copy work list 驱动一个**融合的 PCIe 传输**（所有 expert banks 共享相同的 logical expert-to-slot mapping）。
- **CPU 分支也被 capture 进同一个 graph**：pinned I/O buffers、host-function submit node、同步节点、结果 copy 全部预录，replay 时无需 Python 调度。



### 4.2 Expert Storage and Platform Adaptation

FreeToken 通过归一化的专家参数存储设计来实现参数的高效加载；在模型加载时候，选择与硬件适配的 kernel 来实现平台的适配。

1. **专家存储归一化**：将各模型的 checkpoint 布局统一为 **Expert Banks**，每个 bank 以 `layer_id * num_experts + expert_id` 为 leading dimension。所有 bank 中同一 (layer, expert) 的行组成一个完整专家。这使得 GPU kernel 和 CPU executor 共享同一 logical expert identity，无需关心底层物理格式差异。
2. **FTW 格式**：离线将专家权重合并为 runtime bank layout，启动时跳过 tensor discovery 和 repacking，用并行 direct I/O 直接读到精确大小的 host banks（读完后才 pin）。
3. **平台适配**：load 时选择与 expert 表示、GPU arch、CUDA 环境匹配的 kernel。若无法建立 pinned/DMA 路径（某些 OS/driver 限制），fallback 到纯 CPU MoE backend（专家权重 pageable，所有 routed 专家在 CPU 执行，只有 activations 和 outputs 穿过 CPU-GPU 边界）。



## 5 Evaluation

### 5.1 Experimental Setup

实验设计围绕三个维度展开：

1. **硬件多样性**：6 台机器，覆盖从 8GB RTX 4060 笔记本（PCIe 4.0 x8, 11.8 GB/s）到 96GB RTX PRO 6000 工作站（PCIe 5.0 x16, 51.5 GB/s），以及中间的 3090/4090/5090 服务器和 5090 桌面。特别地，**服务器 CPU 线程被 cap 到 6-8 核**以模拟真实边缘系统的 CPU 能力（而非数据中心级的数百核）。
2. **模型覆盖**：Qwen3.6-35B-A3B（BF16，~35B 总参，~3B 活跃），DeepSeek-V4-Flash（284B 总参，~13B 活跃，MXFP4 量化），GLM-5.2（753B 总参，~40B 活跃，NVFP4，433GB checkpoint）。
3. **工作负载多样性**：
   - W1: AIME 数学推理（单轮，decode-dominated）
   - W2: OpenCode + SWE-bench（coding agent，3 个 scripted turns）
   - W3: Claude Code + SWE-bench（native protocol，并发 subagents，56-65k tokens）
   - W4: OpenClaw + Email/Calendar（13 fixed turns，~24.5k token system context）
4. **基线对比**：llama.cpp、Ollama、KTransformers、MoE-Infinity（后两者仅支持 subset）。



![image-20260905185955190](/table_1.png)

### 5.2 Main Results

![image-20260905190358617](/fig3.png)

Figure 3 展示了端到端结果：

1. **Decode throughput**：

   - FreeToken 在 Qwen3.6 上 77-83 tok/s，在 DSV4-Flash 上 22-25 tok/s，分别是 strongest baseline 的 1.8-2.3× 和 1.5-1.9×。
   - **稳定性**：FreeToken 在 Agent 工作负载下的 decode 吞吐量相比单轮 W1 仅下降 ≤12%，而 KTransformers 在 DSV4-Flash 上从 W1 到 W2 已下降 31%。这证明 FreeToken 的语义感知缓存对长上下文多轮对话的鲁棒性。
   - MoE-Infinity 仅能跑 W1（8.8 tok/s），其 per-expert prefill staging cap 导致长 prompt 工作负载 abort，且 server 不保留跨请求的 KV cache。

2. **Time-to-First-Token (TTFT)**：

   - FreeToken 在 6 个 multi-turn 单元中有 5 个取得最低 mean TTFT（Qwen3.6×W3 中 KTransformers 的 GPU-prefill 路径稍快）。
   - **Tail latency 是关键区别**：FreeToken 的最差 turn TTFT 在所有 workload 下 <44s，而每个基线都在某些 workload 下超过 150s（llama.cpp 232s，KTransformers 946s）。这些长尾延迟超过了真实 Agent 客户端的超时阈值（OpenClaw 120s idle watchdog，Claude Code ~10min timeout），意味着 baseline 在真实部署中可能被客户端提前终止。

   

### 5.3 Breakdown and Cross-Hardware Analysis

从三方面分析验证 FreeToken 机制的有效性和通用性：

1. **Pipelined Prefill（Figure 4a）**：
   - 双缓冲开启时，prefill 是 transfer-bound，每 8192-token chunk 完成时间 ≈ 1.19-1.22s，几乎等于 64.4GB expert pool 在 52.7 GB/s PCIe 上的传输时间。吞吐量在 16k tokens 时达到 6.7k tok/s。
   - 关闭双缓冲（串行化 transfer vs compute）时，吞吐量在 4k、8k、16k 时分别下降 19%、25%、26%，惩罚随 prompt 长度增长而增大（因为可隐藏的计算比例增加）。
2. **Expert Locality（Figure 4b）**：
   - FreeToken 的 global LRU 在相同 cache capacity 下，decode-time expert miss rate 在 Qwen3.6 上为 16%（vs KTransformers 41%，llama.cpp 62%），在 DSV4-Flash 上为 39%（vs 59%，89%）。
   - 这意味着：**LRU 跟随 token-level 路由变化的效果远优于 prefill 时的静态 hot expert 选择**。即使在 agent 工作负载（路由变化更剧烈）下，这一结论仍然成立（bands 表示 W1-W4 的 min-max range）。
3. **Cross-Hardware Serving（Figure 5）**：
   - FreeToken 在所有 5 个 consumer 系统上都是最强引擎：领先 baseline 1.3×（3090/4090）到 2.1×（5090 desktop）。
   - 两个 5090 系统（服务器 vs 桌面）的对比：相同 GPU，但 host 从多通道服务器切换到双通道消费级桌面，FreeToken 仅损失 4% decode 吞吐量，而 llama.cpp 损失 20%（因为其 CPU-resident 专家在双通道 DDR5 上带宽受限）。
   - RTX 4060 笔记本（8GB, PCIe x8）：FreeToken (NVFP4) 跑出 39.3 tok/s，是 RTX 4090 的 92%，且超过 Codex 生产 median（33 tok/s）。
   - GLM-5.2（753B）在 RTX PRO 6000 上：FreeToken 14.9 tok/s vs llama.cpp 7.3 tok/s（2.0×），KTransformers 因 host memory 不足（需要 753GB-1.5TB 但只有 512GB）无法运行。

![image-20260905190511005](/fig4.png)



![image-20260905190539921](/fig5.png)



## 6 Related Work

论文将相关工作分为三条线：

1. **专家 offloading 与缓存**（EdgeMoE, Mixtral-offloading, MoE-Infinity, ProMoE, ExpertFlow, FineMoE, HOBBIT, SiDA, SMoE, Pre-gated MoE）：
   - 共同假设：miss 专家只能通过 PCIe 传输到 GPU。
   - FreeToken 的突破：承认 miss 不可避免，而是改变"miss 如何被服务"——将部分 miss 交给 CPU 就地执行，利用 host 残差带宽。
2. **混合 CPU-GPU 执行**（FlexGen, DeepSpeed-Inference, PowerInfer, llama.cpp, Ollama, Fiddler, KTransformers, HybriMoE, SMoE）：
   - 已有工作将 CPU 作为计算资源，但划分策略要么是静态的（KTransformers 在启动时固定），要么需要 host-side 调度（每步重新计算），无法纳入 CUDA Graph。
   - FreeToken 的突破：用闭合形式的 公式，决策成本极低，可驻留 device，且被 capture 进 graph。
3. **服务基础设施与分层内存**（vLLM, SGLang, FlashInfer, SGLang HiCache, WiSP, eLLM, FluxMoE）：
   - 这些系统移动"被动字节"（page 或 expert absent 时只能 fetch）。
   - FreeToken 的突破：expert weights 提供了"额外自由度"——缺失的专家也可以就地计算。FreeToken 将两种 lever（缓存 + 就地计算）结合在同一运行时中。



## 7 Conclusion

结论重申核心论点：一旦 MoE 的稀疏激活使得计算变得可行，本地推理的关键就不再是"模型能否塞进 GPU"，而是"系统能否有效编排整台机器"。FreeToken 将 GPU、CPU、host memory、interconnect 视为**统一推理平台**，通过三机制（bandwidth-adaptive execution + semantic-aware caching + elastic memory management）将开源 MoE 模型从"数据中心独占"变为"个人机器可部署"。

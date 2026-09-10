---
title: DeepSeek-V4.1-Flash 技术解读：持久化 KV Cache 压到约 1/8，百万上下文 Agent 的成本拐点
date: 2026-09-10T06:59:47.207Z
tags: [llm, deepseek, aiinfra]
categories: aiinfra
typora-root-url: ../../../
---

## 0. 摘要 / TL;DR

DeepSeek-V4.1-Flash 是一个 552B backbone、196B Engram 参数的多模态 MoE 模型，prefill 每 token 激活 **8B**，decode 每 token 激活 **16B**，支持 **100 万 token** 上下文。它的核心目标不是继续堆参数，而是把长上下文 Agent 的部署成本打到新低。

它通过 **CED、CSA2、FP4 KV Cache、SWA Bounded Replay** 等联合优化，把全局 KV Cache 压到 **890 bytes/token**，约为 DeepSeek-V4-Flash 的 **1/4**；持久 KV Cache 约为 DeepSeek-V4-Flash 的 **1/8**。与此同时，模型能力不降反升，在多个 Agentic benchmark 上达到或超过闭源前沿模型。

![fig-1](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-1.png)

> (a) DeepSeek-V4.1-Flash 与 Kimi-K3、GLM-5.3、Opus5、GPT5.6 Sol 在 Terminal-Bench 3.0、DeepSWE v1.1、CyberGym、Automation-Bench 上的表现对比；(b) DeepSeek 各代模型每 token 全局 KV cache 大小，V4.1-Flash 降至 890 bytes/token。



### 模型参数

| 指标 | DeepSeek-V4.1-Flash |
|---|---|
| Backbone 参数 | 552B |
| Engram 参数 | 196B |
| 激活参数 | Prefill 8B / Decode 16B |
| 上下文长度 | 1M token |
| 全局 KV Cache | 890 bytes/token，V4-Flash 的 1/4 |
| 持久 KV Cache | V4-Flash 的 1/8 |
| Decode FLOPs | 4K→1M 仅增加约 1/4 |
| Reuse Mode 推理 kernel | Prefill 15 / Decode 11 |



### V4.1-Flash 能力横向对比

以下数据取自报告 Table 3，均为 Max reasoning effort 下的关键对比。

| Benchmark | V4.1-Flash | V4-Flash | V4-Pro | 最强对比模型 | 结论 |
|---|---:|---:|---:|---:|---|
| Codeforces Rating | **3471** | 3289 | 3348 | — | 表中最高 |
| MathArena Apex | **65.6** | 58.6 | 65.3 | Kimi-K3 65.6 | 持平开源最佳 |
| GPQA Diamond | 90.9 | 89.9 | 92.4 | GPT-5.6 Sol 94.1 | 强，但非最高 |
| DeepSWE v1.1 | **74.2** | 54.4 | 62.7 | Opus-5 74.0 / GPT-5.6 Sol 73.0 | 超过闭源前沿 |
| Terminal-Bench 2.1 | **90.6** | 82.7 | 87.9 | Opus-5 89.1 | 表中最高 |
| Terminal-Bench 3.0 | 30.0 | 7.6 | 11.8 | Opus-5 43.3 | 优于前代，落后最强闭源 |
| Automation-Bench | **54.8** | 37.7 | 43.2 | Opus-5 50.3 | 表中最高 |
| Agents’ Last Exam | **31.8** | 25.2 | 25.7 | Opus-5 28.6 | 表中最高 |
| CyberGym | **88.1** | 76.7 | 83.3 | GPT-5.6 Sol 84.5 | 表中最高 |
| HLE w/ tools | **63.9** | 51.5 | 60.0 | Opus-5 63.6 | 表中最高 |
| Chartography w/ tools | 78.9 | — | — | Opus-5 84.0 | 优于 Kimi-K3，落后 Opus |
| BabyVision w/ tools | 89.6 | — | — | Opus-5 94.1 | 优于 Kimi-K3 |
| ZeroBench-main w/ tools | 49.0 | — | — | GPT-5.6 Sol 53.0 | 优于 Kimi-K3，落后闭源 |



### 能力画像

- **Agent 能力是最大亮点**：DeepSWE v1.1、Terminal-Bench 2.1、Automation-Bench、Agents’ Last Exam、CyberGym、HLE w/ tools 等核心 Agentic benchmark 上，V4.1-Flash 取得第一或超过 Opus-5、GPT-5.6 Sol。
- **推理与编码达到顶级开源水平**：Codeforces 3471 为表中最高；MathArena Apex 65.6 与 Kimi-K3 持平；GPQA Diamond 90.9 高于 V4-Flash。
- **多模态视觉 Agent 强于开源，弱于最强闭源**：Chartography、BabyVision、ZeroBench 上优于 Kimi-K3，但仍落后 Opus-5 / GPT-5.6 Sol。
- **参数效率极高**：V4.1-Flash-Base 与 V4-Pro-Base 在世界知识、推理、编码能力相当，但是总参数量仅为后者的 1/3， prefill 和 decoding 阶段激活参数约为 1/6 和 1/3，内部 held-out 评估提升 5%–10%。

---



## 1. 引言：长上下文 Agent 的真正瓶颈

Agent 工作负载正在变得越来越 input-heavy。工具调用、长文档、多轮交互、代码仓库、截图与日志，都会产生超长上下文。过去几年，稀疏 attention 显著降低了长序列的计算成本，但新的瓶颈逐渐转移到 **KV Cache 的存储、迁移和带宽**上。

DeepSeek-V4 的架构是 global attention + local SWA 的混合：global branch 维护全局 KV，SWA 维护局部 KV。对于足够长的序列，global KV 会主导运行时 KV footprint，受 HBM 容量限制；持久 KV 则受 SSD 和 host memory 限制。I/O 与互联带宽进一步限制 cache migration 和 loading。这些约束直接限制 serving throughput，抬高部署成本。

DeepSeek-V4.1-Flash 的目标很明确：**进一步压缩 KV Cache footprint**，同时不牺牲能力。它采用 Causal Encoder-Decoder 架构，decoder global KV 从 encoder 最后一层 hidden state 投影得到，使 prefill 只激活 8B、decode 激活 16B。更重要的是，它在架构、缓存精度、部署策略三个层面联合优化，把全局 KV 压到 V4-Flash 的约 1/4，把持久 KV 压到约 1/8。

![fig-2](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-2.png)

>长上下文 decode 计算成本几乎被压平：单 token Decode FLOPs 随上下文长度的变化。DeepSeek-V4.1-Flash 在 4K 到 1M 上下文扩展 256 倍时，Decode FLOPs 仅增加约 1/4，显著低于 DeepSeek-V4-Flash。

---



## 2. 核心亮点

1. **KV Cache 极限压缩** 
   全局 KV 降至 890 bytes/token，约为 V4-Flash 的 1/4；持久 KV 约为 V4-Flash 的 1/8。

2. **CED：Causal Encoder-Decoder，prefill 计算接近减半** 
   decoder global KV 从 encoder 最后一层投影得到；prefill 每 token 激活 8B，decode 激活 16B。

3. **CSA2：跨层复用 main KV、indexer K、Top-K 索引** 
   三种模式 Full / Reindex / Reuse，同时压缩 entry size、sequence dimension、layer dimension。

4. **FP4 Main KV Cache**  

   V4.1-Flash 扩展 QAT 到 main KV，main KV 存储接近减半，SWA KV 保留 FP8。

5. **SWA Bounded Replay** 
   不再持久化 SWA KV，缺失时只回放最近 $n_{win}$ 个 token，把持久 KV 压到 V4-Flash 的 1/8。

6. **模型能力不降反升，Agent 能力突出** 
   DeepSWE v1.1 74.2%、Terminal-Bench 2.1 90.6%、Automation-Bench 54.8%、CyberGym 88.1%，多个 Agentic benchmark 超过闭源前沿。

7. **推理系统大幅简化** 
   Reuse Mode 层 prefill 仅 15 个 kernel，decode 仅 11 个；Decode FLOPs 从 4K 到 1M 仅增约 1/4。

8. **训练与优化器改进** 
   45T 多模态 token，64K 起步，34T 扩到 1M；Muon + AdamW + Sinkhorn-balanced update；head-wise Muon。

9. **AI Infra 改动** 
   DSec 百万级沙箱，单节点容器密度 2500+；异步 RL/OPD；样本级 dispatch、token 级中断、状态持久化。

10. **可控推理 effort 与多 Agent** 
    effort $b \in [1,100]$，API 三档 low=50、high=75、max=100；多 Agent 在 ProgramBench、FrontierSWE v2 上全面优于单 Agent。

---



## 3. 模型架构

### 3.1 总览与多模态架构

DeepSeek-V4.1-Flash 是一个多模态 MoE Transformer，输入图像和文本，自回归生成文本。语言 backbone 有 40 层，分成 20 层 causal encoder 和 20 层 decoder。每层同时包含 global attention 和 SWA，但前两层只用 SWA。

视觉侧由 DeepSeek-ViT 和 MLP projector 组成。DeepSeek-ViT 从零训练，使用 2D-RoPE、RMSNorm、SwiGLU，并用 3×3 pixel-unshuffle 把视觉 token 数减少 9 倍，支持最高约 1344×1344 输入。视觉 embedding 与文本 embedding 从语言模型预训练一开始就联合处理。

MoE 侧保留 DeepSeekMoE 的 shared expert 和 fine-grained routed experts，并引入 **模态特定负载均衡**：文本和图像 token 分别维护 expert-wise correction bias，避免模态间路由偏好互相干扰。

![模型架构图](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-3.png)

>  DeepSeek-V4.1-Flash 总体架构。40 层网络分为 20 层 causal encoder 和 20 层 decoder；前两层使用 SWA，其余使用 CSA2；同时包含 Single-Pass mHC、Engram、DSpark 和 Hierarchical Sparse Indexer。



### 3.2 CED：Causal Encoder-Decoder

CED 是 V4.1-Flash 降低 prefill 计算的关键。它把 Transformer 底部 $L/2$ 层作为 causal encoder，上半部分作为 decoder。对于 decoder 的 global attention，KV 不再由各自层的 hidden state 生成，而是直接从第 $L/2$ 层 hidden state 投影：

$$
C_l = H_{L/2}W_l^{KV},\quad Z_l = H_{L/2}W_l^{Z},\quad l > L/2
$$

这意味着对于完整长序列的 global-attention 主路径，prefill 主要计算前一半的 causal encoder 层，就能获得 decoder 的 global KV。对于 SWA，CED 仍保持逐层计算，但通过 Decoder SWA Bounded Replay 限制回放长度。严格地说，decoder 仍需对最近 $n_{win}$ 个 token 做逐层 SWA replay，因此报告给出的复杂度是 $O(NL/2+n_{win}L/2)$，而不是完全不执行 decoder。

整体上，对于 $N \gg n_{win}$，prefill 复杂度从 $O(NL)$ 降到约 $O(NL/2)$，接近减半。对输入密集、工具调用频繁的 Agent 工作负载，这个收益非常直接。



### 3.3 CSA2：Compressed Sparse Attention 2

CSA2 是 KV Cache 压缩的架构核心。它从三个维度同时压缩 attention 成本：

1. **entry size**：GQA、MLA 减少 KV 头数或共享 latent；
2. **sequence dimension**：每 $m$ 个 token 压缩成一个 entry；
3. **layer dimension**：跨层复用 main KV、indexer K 和 Top-K indices。

CSA2 有三种静态模式：

| 模式 | 行为 |
|---|---|
| Full | 自己生成 main KV、indexer K，并计算 Top-K 索引 |
| Reindex | 复用前层 main KV 和 indexer K，但用自己的 indexer Q 重新打分并选 Top-K |
| Reuse | 直接复用前层 main KV 和 Top-K 索引，不做新索引计算 |

所有模式中，每层仍计算自己的 query 和 SWA KV。CSA2 相比 CSA 去掉了压缩重叠和绝对位置 embedding，并让 indexer K 从 main KV 投影得到，简化实现、提升训练效率。

具体配置上：

- Encoder 18 层使用 CSA2，压缩率 $m=2$，分三组，每组第一层 Full，其余五层 Reuse。
- Decoder 20 层使用 CSA2，压缩率 $m=1$，分五组；第一组第一层 Full，其余三层 Reuse；后四组第一层 Reindex，其余三层 Reuse。
- Top-K 选择 512 个 KV entry。

![CSA2](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-4.png)

> CSA2 的三种工作模式：Full、Reindex、Reuse。绿色表示当前层计算，黄色表示复用最近 Full 层的 main KV 和 indexer K，红色表示复用最近索引层的 Top-K 索引。



### 3.4 Hierarchical Sparse Indexer

跨层索引复用减少了 indexer 计算次数，但剩余 indexer 仍可能扫描完整上下文。V4.1-Flash 在 decoder 中引入 **Hierarchical Sparse Indexer**：

- 第一个 Full 层扫描全部 causally visible 的 main KV，选出自己的 Top-512；
- 同时做 blockwise 候选选择，每个 block 取最大 index score，选出最高分 block；
- 例如选 2048 个 block、每个 8 个位置，得到 16384 个候选位置；
- 后续 Reindex 层只在这个候选池中打分并选 Top-K；
- Reuse 层不做新索引，直接复用最新 Top-K。

这样，对于固定候选池大小，后续 indexer 每 query 的打分成本与上下文长度解耦，从线性变为常数。第一个 Full 层仍需全范围扫描，但后续深层 indexer 的成本被显著压低。

![Hierarchical Sparse Indexer](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-5.png)

> Hierarchical Sparse Indexer。第一个 Full 层构建共享候选池，后续 Reindex 层只在候选池中选择 Top-512。



### 3.5 FP4 Main KV Cache

长上下文 Agent 需要大 per-request KV Cache，存储成本很高。DeepSeek-V4 已经对 FP4 indexer query/key 做 QAT。V4.1-Flash 进一步把 QAT 扩展到 main KV Cache，用 FP4 降低存储。

具体选择：

- 主 KV 使用 NVFP4 风格的 E2M1 数值，并为每 16 个通道配置一个 E4M3 scale；
- 省略 NVFP4 的第二级 global scale，因为 main KV 的动态范围足够；
- 在 RoPE 之后量化；
- SWA KV 仍保留 FP8，因为它对量化更敏感。

相比 FP8 main KV，FP4 几乎把存储 footprint 减半，无论在 HBM 还是 offload 到 SSD。



### 3.6 SWA Bounded Replay

持久 KV Cache 过去包含 global KV 和 SWA KV。SWA KV 的特点是：只在会话内分钟级窗口内复用，会话结束或下一轮开始后就“死掉”，长期持久化既贵又低效。

V4.1-Flash 的策略是：

1. **SWA KV 不再持久化到 SSD**，改放在每台机器 10% host DRAM 组成的分布式内存池，短 TTL、快速回收；
2. **Global KV 仍保留在持久 KV Cache**，至少 72 小时；
3. 缺失 SWA KV 时，用 **SWA Bounded Replay** 近似恢复：只回放最近 $n_{win}$ 个 token，而不是 $L \times n_{win}$ 个 token。

Encoder 和 Decoder 都应用 Bounded Replay。Decoder 侧，global KV 从 encoder 最后一层投影，唯一阻碍 prefill 在 encoder 结束的是 decoder SWA KV；Bounded Replay 让它只回放最近 $n_{win}$ 个 prompt token，并把重建的 decoder SWA KV 只用于 decoding，不用于 prefix caching。

报告显示，这种近似重建对响应质量影响可忽略。它把“灾难性 miss”变成了“低成本近似恢复”，是持久 KV 降到 1/8 的关键。



### 3.7 其他高效扩展

- **Single-Pass mHC**：把 input mixing 系数依赖前移一个 block，使 residual update、input mixing、coefficient prediction 可以融合。部署时用 Mega-mHC kernel，激活内存流量减半。
- **Engram**：条件记忆模块，196B 参数，N-gram 阶数 {2,3,4}，8 个 hash head，FP8，放在第 1 和 14 层 **（报告采用 zero-indexed，即通常所说的第 2、15 个 Transformer block）**。推理时可通过 RDMA 预取。
- **DSpark**：半自回归 speculative decoding，3 个 Transformer block，sliding window 128，confidence-scheduled verification，动态选择验证长度。
- **优化器**：线性变换用 Muon，Query/Key 用 head-wise Muon；Engram、embedding、预测头用 momentum + Sinkhorn balancing，降低优化器状态内存。

---



## 4. 通用基础设施

### 4.1 训练基础设施

多模态训练方面，V4.1-Flash 支持：

- 对比学习阶段通信与计算重叠：视觉特征在文本 forward 时 all-gather，文本特征在文本 backward 时 all-gather；
- 视觉编码器与 LLM 解耦执行，避免相互干扰；
- 长序列图像分片：每张图只加载一次，按 CP rank 做负载均衡；
- 增量图像传输与 CPU 侧预处理缓存，服务 RL rollout。

CSA2 训练方面，由于共享层可能跨 pipeline stage，需要额外协调：

- **Shadow indexers**：每个 stage 放轻量可执行副本，参数逻辑 owner 唯一；
- **Pipeline payload extensions**：跨 pipeline 边界传递中间表示和稀疏路由信息；
- **Micro-batch 级共享状态管理**：跟踪共享状态生命周期，最后一个 consumer 完成后释放。

Engram 训练方面，embedding 表按行分片，优化器状态再分片，FP8 存储与读取，预取与视觉编码器计算重叠，梯度缓冲后返回 owning rank。



### 4.2 推理系统

推理系统以效率为第一目标。虽然架构复杂，但 kernel flow 很简洁：

- FlashMLA 的 fused-RoPE-attention-RoPE-cast；
- DeepGEMM 的 Mega-Gate、Mega-mHC、Mega-MoE；
- TileKernels、DeepSelect TopK kernel。

结果：绝大多数 Transformer 层，也就是 CSA2 Reuse Mode 层，prefill 只需 **15 个 kernel**，decode 只需 **11 个 kernel**。

部署上采用 **Encoder-Prefill-Decode（EPD）分离**，视觉编码、prefill、decoding 可独立扩展并重叠执行。

持久 KV 管理上：

- Global KV 长期保留在持久 KV Cache；
- SWA KV 放 host DRAM 分布式池，短 TTL；
- 缺失时用 Bounded Replay 兜底；
- 不再把 SWA KV 写入 SSD。

---



## 5. 预训练

V4.1-Flash 在 45T token 多模态语料上预训练，文本:多模态约 7:1。文本数据强调信息增益，过滤低质量模型生成内容和低质机器翻译；多模态数据包括 image-text pairs、interleaved image-text、domain-specific data，并用 SmolVLM 做质量打分。

模型设置：

- 40 层，hidden 5120；
- 20 encoder + 20 decoder；
- MoE：1 shared expert + 384 routed experts，每 token 激活 6 个；
- 视觉编码器：32 层，hidden 1024，16 heads，patch 14。

训练设置：

- 从零开始训练稀疏 attention，序列长度 64K，无 dense warmup；
- 34T token 时扩展到 1M；
- batch size 100.6M token；
- 学习率 2.6e-4，28T 后 cosine 衰减到 2.6e-5；
- 优化器：Muon + AdamW + Sinkhorn-balanced update。

Base 评估中，V4.1-Flash-Base 与 DeepSeek-V4-Pro-Base 总体能力相当，V4.1-Flash-Base 在内部 held-out 评估上提升 5%–10%；总参数 552B 相对 1.6T 约为 1/3；激活参数为 prefill 8B、decode 16B，相对 V4-Pro 的 49B 约为 1/6 和 1/3。

![](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-6.png)

> base model 的参数效率和数据质量提升: DeepSeek-V4-Flash-Base、DeepSeek-V4-Pro-Base、DeepSeek-V4.1-Flash-Base 在 held-out 评估集上的 Bits-per-byte 对比。V4.1-Flash-Base 在所有任务上取得最低 BPB。



---

## 6. 后训练

V4.1-Flash 的后训练没有算法层面的新创新，仍是 **SFT + RL + OPD**。所有实质变化都在数据管线：大规模自动任务合成、环境构建、质量过滤、难度校准。

任务被形式化为 `(problem, environment, verification system)` 三元组，从难度和正确性两个维度评估。通用 Agent 环境来自内部员工和外部伙伴的真实工作流反馈，构建 mocked tools 和失败回放环境。Coding Agent 环境来自内部复杂任务和 GitHub 高星仓库，由多个专门 agent 协作构建、测试、审查、修复。

RL 在合成任务上大规模异步执行。Rollout 与训练解耦到 agent sandbox 和 worker container，运行在 DSec 上。模型合并用于跨 run 重初始化，聚合不同 scaffold 的改进。

DSec 是生产级沙箱平台，支持百万级并发实例。它通过 sharding + relaxed consistency 调度，单节点容器密度从约 1000 提升到 **2500+**。安全上使用 AppArmor、eBPF 网络策略，防止 reward hacking 和环境破坏。

可控推理努力方面，V4.1-Flash 引入标量 effort $b \in [1,100]$，API 暴露三档：low=50、high=75、max=100。长度惩罚为：

$$
r^{len}_{b,j} = -\min\left(C_{max}, k(b)\frac{\ell_{b,j}}{L_{norm}}\right)
$$

其中 $k(b)=k_0\exp(-(b-b_{min})/\tau)$。effort 越高，token 惩罚越弱，允许更长推理。

异步后训练方面：

- 样本级 dispatch 维持 rollout 并发；
- token 级中断支持快速切换 checkpoint；
- KV cache 和 expert routing 按 token 粒度持久化，恢复后无需重新 prefill；
- OPD 使用 40+ teacher 模型，支持异构 teacher 和动态配置切换。

多 Agent 方面，Agent Team 模式下 lead agent 可创建 teammate，共享 repo、mailbox、task board。RL 奖励包含任务性能、协作 bonus 和 derived-latency penalty。

![fig-7](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-7.png)

> 随着 RL 训练规模扩大，DeepSeek Harness Minimal 模式下多个 code agent benchmark 的性能提升；扩展到 1M 上下文后，Terminal-Bench v3.0 等长时任务继续提升。



![fig-8](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-8.png)

> 跨多个 Claude Code 版本和异构 scaffold 联合训练时，性能随累计 RL steps 提升。评估基准为 DeepSWE v1.1。



---

## 7. 评测

### 7.1 推理与 Agent

后训练评测聚焦推理和 Agent 能力。核心结果如下：

| Benchmark | V4.1-Flash | V4-Flash | V4-Pro | 对比模型 | 结论 |
|---|---:|---:|---:|---:|---|
| Codeforces Rating | **3471** | 3289 | 3348 | — | 表中最高 |
| MathArena Apex | **65.6** | 58.6 | 65.3 | Kimi-K3 65.6 | 持平开源最佳 |
| GPQA Diamond | 90.9 | 89.9 | 92.4 | GPT-5.6 Sol 94.1 | 强，但非最高 |
| DeepSWE v1.1 | **74.2** | 54.4 | 62.7 | Opus-5 74.0 | 超过闭源前沿 |
| Terminal-Bench 2.1 | **90.6** | 82.7 | 87.9 | Opus-5 89.1 | 表中最高 |
| Terminal-Bench 3.0 | 30.0 | 7.6 | 11.8 | Opus-5 43.3 | 优于前代，落后最强闭源 |
| Automation-Bench | **54.8** | 37.7 | 43.2 | Opus-5 50.3 | 表中最高 |
| Agents’ Last Exam | **31.8** | 25.2 | 25.7 | Opus-5 28.6 | 表中最高 |
| CyberGym | **88.1** | 76.7 | 83.3 | GPT-5.6 Sol 84.5 | 表中最高 |
| HLE w/ tools | **63.9** | 51.5 | 60.0 | Opus-5 63.6 | 表中最高 |

在核心推理任务上，V4.1-Flash 的 Codeforces 评分 3471，超过 V4-Flash 和 V4-Pro；MathArena Apex 65.6，与 Kimi-K3 持平；GPQA Diamond 90.9，高于 V4-Flash。

Agentic 任务提升更明显。DeepSWE v1.1 达到 74.2%，高于 Opus-5（74.0%）和 GPT-5.6 Sol（73.0%）的成绩；Terminal-Bench 2.1 达到 90.6%；Automation-Bench 54.8%；Agents’ Last Exam 31.8%；CyberGym 88.1%。需要注意，报告的不同模型列可能来自各自或指定的 scaffold，不能据此推出在完全相同 harness 下仍保持相同排序。![table-3](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/table-3.png)



### 7.2 多模态与视觉 Agent

多模态方面，V4.1-Flash 在视觉推理和专业图表理解上超过 Kimi-K3，但落后最强闭源模型：

- Chartography w/ tools：78.9%；
- BabyVision w/ tools：89.6%；
- ZeroBench-main w/ tools：49.0%。

报告也承认，与 giant closed-source systems 相比，整体多模态性能仍有差距。



### 7.3 Reasoning Effort 控制

V4.1-Flash 暴露 reasoning-effort 设置。effort 从 25 提升到 100：

- 八个推理基准平均 Pass@1：67.1% → 76.3%；
- DeepSWE v1.1：66.0% → 74.2%；
- Terminal-Bench 2.1：82.4% → 90.6%；
- 输出 token 约增加 2.5 倍。

收益前倾：60–80 区间已能恢复大部分准确率，effort 100 更适合极难任务。

![fig-9](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-9.png)

> 性能与输出长度随 reasoning effort 的变化。每个面板绘制 Pass@1 和平均输出 token 数，effort 从 25 到 100。



### 7.4 Scaffold 与多 Agent

跨 scaffold 评测覆盖 Claude Code、Codex、OpenCode、Pi、mini-SWE、DeepSeek Harness Minimal/Standard/PTC。V4.1-Flash 表现稳健，不依赖单一 harness。

多 Agent 方面，在 ProgramBench 和 FrontierSWE v2 上，多 Agent 配置在所有 deadline 下优于单 Agent。

- ProgramBench Almost@1：多 Agent 峰值 30.04%，单 Agent 20.39%；
- FrontierSWE v2 Mean@5：多 Agent 20 小时 32.90%，单 Agent 28.20%。

![fig-10](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-10.png)

> 单 Agent 与多 Agent 配置在 ProgramBench 和 FrontierSWE v2 上的 test-time compute scaling，横轴为 per-rollout wall-clock deadline。



---

## 8. 结论、局限与未来方向

DeepSeek-V4.1-Flash 的核心贡献是：**以 KV Cache 压缩为中心，通过 CED、CSA2、FP4 KV、SWA Bounded Replay 和推理/训练 infra 协同，把百万上下文 Agent 的部署成本显著压低，同时能力不降反升。**

它把全局 KV 压到 890 bytes/token，约为 V4-Flash 的 1/4；持久 KV 压到约 1/8；Decode FLOPs 在 4K→1M 下仅增加约 1/4。与此同时，它在 DeepSWE v1.1、Terminal-Bench 2.1、Automation-Bench、Agents’ Last Exam、CyberGym、HLE w/ tools 等核心 Agentic benchmark 上，部分达到或超过闭源前沿模型。

局限也很明确：

- CSA2 稀疏选择错误、SWA Bounded Replay 的近似状态重建，可能在未测试边界条件下造成能力退化；
- 新架构的鲁棒性边界尚未完全刻画；
- 标准 benchmark 逐渐饱和，与最强闭源模型在极难推理和边缘案例上仍有差距。
- **评测证据主要来自模型作者报告，部分结果使用内部框架或不同 Agent scaffold；横向排名不应脱离模型版本、上下文长度、reasoning effort、采样数与工具配置单独引用。**
- **890 bytes/token、1/4 和 1/8 是缓存 footprint 指标，不代表端到端延迟、吞吐或总服务成本会同比例改善；报告没有给出一套完整的生产吞吐、首 token 延迟和单位请求成本对照表。**

未来方向包括：扩展压力测试和评估协议，重点关注长上下文稀疏检索和 SWA 状态重建边界；联合 scaling 数据、模型容量和 RL；推进模型-harness 协同设计。

![fig-11](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-11.png)

> 不同 scaffold 下 reasoning effort 对轨迹长度和 Pass@1 的影响。提高 effort 会单调增加输出 token，但 Pass@1 并非严格单调，中间存在平台和回落。

![](/image/01-技术积累/AIinfra/deepseek-v4.1-flash-tech-report-interpretation/fig-12.png)

> 八个推理密集型 benchmark 上，Pass@1 和平均输出 token 随 reasoning effort 的变化。effort 从 25 到 100 提供平滑、可预测的成本—准确率控制。

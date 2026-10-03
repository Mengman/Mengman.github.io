---
title: 【Qwen3.8-Flash-Next 端侧部署】04-运行时优化 MTP、CUDA Graph 与 Chunked Prefill
date: 2026-09-28T14:38:55.464Z
tags: [qwen, vllm, aiinfra, dgx-spark]
categories: aiinfra
typora-root-url: ../../../
---

上一篇讲述了如何通过量化操作减少模型对于显存/内存的需求，这是模型能够在有限显存/内存环境启动的基础条件。 真正开始处理请求后，vLLM 还要决定每轮执行多少 token、同时调度多少序列、哪些形状使用 CUDA Graph，以及长 prompt 以多大的 chunk 进入模型。

这些配置相互影响。MTP 增加一次 decode 中验证的 token 数，CUDA Graph 必须覆盖相应的 batch width；提高并发会增加同一步中的专家和缓存访问；扩大 prefill chunk 可以提高矩阵计算效率，却会增加激活峰值，增加 decode 请求的 token 间延迟。单独把每个参数调到最大，并不能得到稳定的单机配置。

## 一、Prefill 和 Decode 阶段

LLM 的一次生成请求可以分为两个阶段， prefill 和 Decode。

Prefill 阶段读取完成的输入 prompt token，执行并行矩阵计算，并逐步建立 KV cache 和 GDN 状态。这个阶段由于是按照 token 并行计算，所以是**计算密集型**，同时长输入会产生较大的中间激活和工作区。

Decode 阶段每轮通常只为每条序列增加一个新 token。它的矩阵规模较小，但是要重复读取大量的模型权重和 KV cache、GDN 状态，因此这个阶段是**访存密集**的。

如何平衡 LLM 请求的 Prefill  与 Decode 阶段是推理服务框架的重点优化方向，MTP， Chunked Prefill，CUDA Graph 正是对于这两个阶段的优化技术。

* **MTP**: 优化 decode 阶段，通过一个小的草稿模型，同时降低内存访问与矩阵计算量。
* **Chunked Prefill**: 通过优化长序列 Prefill ，实现 prefill 阶段与 Decode 阶段更好的调度，降低 token 间延迟。
* **CUDA Graph** 和动态批处理则横跨调度与执行。



## 二、 Multi-Token Prediction（MTP）

在一个标准的 Decode 阶段，每一轮计算迭代，会生成一个 token，然后再将这个新生成的 token 作为下一轮迭代的输入，继续向后生成 token，直到出现 `<EOS>` token 或者达到最大生成长度。

```TEXT
step 1: prompt         → token A
step 2: prompt + A     → token B
step 3: prompt + A + B → token C
```

MTP 简单来说，就是使用一个比原模型，参数量小很多的小模型来代替原模型执行 decode，然后将多轮的 decode 结果一次性给原模型检验。 原模型称为目标模型(target model), 小模型称为草稿模型(draft model)。

草稿模型根据当前状态依次提出 K 个候选 token，目标模型进行一次推理检查这K个草稿 token。若前几个草稿与目标模型接受的结果一致，就接收为正式结果；如果第 i 个 token 没有通过验证，那么就用目标模型在 i 位置的输出替换草稿模型，后续的 token 都丢弃掉。

 以 K=3 为例，目标模型一轮验证：

```
3 个草稿 Token + 1 目标 Token = 4 个 Token
```

也就是说在最好的情况下，目标模型一次推理可以得到 4 个token。

在实际的情况中，可预测文本通常有更高接受率，随机性较强、专业术语密集或多模态请求的接受率可能不同。因此，MTP 不能用一个固定加速倍数描述。



> 对于 MTP 技术细节感兴趣可以看我之前**投机解码**相关的论文解读：
> [投机解码专题 核心论文-1： Speculative-Decode-Fast-Inference-from-Transformers-via-Speculative-Decode](https://mengman.github.io/2026/07/02/02-%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/Speculative-Decode-Fast-Inference-from-Transformers-via-Speculative-Decode/)
>
> [投机解码专题 核心论文-2： Accelerating-Large-Language-Model-Decode-with-Speculative-Sampling](https://mengman.github.io/2026/07/03/02-%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/Accelerating-Large-Language-Model-Decode-with-Speculative-Sampling/)
>
> [投机解码专题 自主多头预测架构-1：MEDUSA](https://mengman.github.io/2026/07/09/02-%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/Medusa/)



## 三、Qwen3.8-Flash-Next 的 MTP 配置

### 3.1 MTP 模块架构

![image-20260928154020658](/image/01-技术积累/AIinfra/qwen3.8-flash-next-dgx-spark-deployment-04/qwen3.8-flash-next-arch-with-mtp-detail.png)

Qwen3.8-Flash-Next 的 MTP module 是一层 Qwen Sparse Attention（QSA） 加上一个 MoE 模型。 它的输入包含两个部分：

隐含残差状态：它是上一轮迭代的**四分支隐含状态**，如果是第一轮 MTP ，那么它就是目标模型最后一层的残差状态，输入 shape `[4, 2560]` 。

上一个 token：将刚刚生成的 token 作为输入， 进行 embedding（与目标模型共享）， 输入 shape `[2560]` 。

两个输入进行融合，token embedding 通过广播的方式加到**隐含残差状态**的4个维度上，作为一个完整的输入进入草稿模型。

草稿模型输入下一轮的 token 和**隐含残差状态**，然后进行下一次 MTP 迭代，直到达到配置的 K 个草稿 token。

除此之外 Qwen3.8-Flash-Next 在训练阶段还专门针对 MTP 进行多步训练优化，目的是提高草稿模型的接受率。



### 3.2 MTP 运行成本

1. **模块参数开销**：量化后的 MTP 模块的参数量约为 1.49GB，需要常驻内存，Embedding layer 和 Prediction Head 则是和目标模型共享，不占用额外的显存开销。
2. **Prediction Head 带宽开销**：Prediction Head 的参数量与 MTP 模块相当（后面做详细分析），做 MTP 的过程中会频繁的读取 prediction head 的参数，这会大量占用内存带宽。



### 3.3 裁剪 Prediction Head，优化带宽

上面说 prediction head 的参数量与 MTP 模块的参数量大致相当，现在我们就具体计算一下。 Qwen3.8-flash-next 的 prediction head 是一个覆盖 248,320 token 词表的 BF16 `lm_head`。hidden size 为 2,560，因此权重大小约为：

```
248,320 × 2,560 × 2 bytes
= 1,271,398,400 bytes
≈ 1.18GiB
```

当 K=3 时，这 1.18GiB 权重被内存系统重复读取三次，由于草稿模型本身是很轻量级的，读取这些参数会大大降低 MTP 过程的速度。如果可以对 prediction head 进行剪裁那么就可以减小这部分的带宽消耗。

[FastMTP](https://arxiv.org/html/2509.18362v1)、[FR-Spec](https://arxiv.org/html/2502.14856v1)、[VocabTrim](https://arxiv.org/html/2506.22694v1) 三篇论文都研究了基于高频 token 的 MTP 推理优化，它们有类似的结论：

* 约 25% 的词表覆盖约 95% 的 token 出现次数
* 草稿平均接受长度下降不大：**3.89 → 3.63** (FR-Spec),  **2.690 → 2.622** (FastMTP)
* 速度提升 ： **12%** (FR-Spec)， **12.6%**  (FastMTP)



基于以上研究结论，项目的 [patch_mtp_draft_vocab.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_mtp_draft_vocab.py) 允许从完整输出头抽取常用 token 行。例如 65,536 行的 BF16 slice 大小为：

```text
65,536 × 2,560 × 2 bytes
= 0.3125GiB
```

每次少读约 0.87GiB，MTP-3 一轮少读约：

```text
(1.18 - 0.3125) × 3 ≈ 2.61GiB
```

但是由于要额外保存一份 MTP `lm_head` 参数，常驻内存反而多了一份约 0.31GB。



### 3.4 缩小草稿词表不影响输出结果

论文 [投机解码专题 核心论文-1： Speculative-Decode-Fast-Inference-from-Transformers-via-Speculative-Decode](https://mengman.github.io/2026/07/02/02-%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/Speculative-Decode-Fast-Inference-from-Transformers-via-Speculative-Decode/) 证明了投机解码方式能够确保模型最终的输出与目标模型的结果是保持一致的。 在草稿模型预测阶段，若目标 token 不在缩小后的词表里，草稿模型会提出其他候选；target model 随后会拒绝不匹配的草稿，并按照正常验证流程给出结果，这只会降低草稿模型输出 token 的接受率。



## 四、CUDA Graph 优化 kernel launch 与调度开销

在 decode 阶段每轮有大量形状相对稳定的小 CUDA kernel 被调用，如果每次都由 CPU 逐个发起，kernel 的启动和 Python/C++ 调度开销会相比 prefill 阶段会占更高比例。

**CUDA Graph**  是 NVIDIA 在 CUDA 10 中引入的一种任务图（Task Graph）执行模型，旨在降低 GPU 任务提交和调度的 CPU 开销（CPU Overhead），并提升大规模异步任务序列的执行效率。它会先捕获一组 GPU 操作及其依赖，后续使用固定地址的输入输出 buffer 重放整张图，减少重复提交。

然而实际的 decode 情况要比上面的描述更加复杂，在 vLLM 中按照 batch 中 query token 长度是否一致可以分为 **Uniform Decode** 与 **Non-Uniform Batching** 。 



> **query length**
>
> **query length 是本次 forward 为某个请求处理的 token 数**。



### 4.1 vLLM CUDA Graph 的五种模式

vLLM 目前支持五种 CUDA Graph 模式： **None**、**piecewise**、**full**、**full_decode_only**, **full_and_piecewise**,  下面分别介绍这五种模式。

在介绍这五种模式之前，我们快速回顾一下一个标准的 attention 计算的 GPU 操作：

```
输入 → 矩阵乘法 A → Attention → 矩阵乘法 B → 激活函数 → 输出
```

普通执行时，CPU 需要依次向 GPU 提交这些操作。模型有很多层时，一次 forward 可能需要提交大量 GPU kernel，每次提交都有开销。

CUDA Graph 提供了两个步骤：

1. **捕获（capture）**：提前把这些 GPU 操作、执行顺序和依赖关系记录下来。
2. **重放（replay）**：后续用一次 graph launch 提交已记录的工作，让 GPU 按图执行。

每次重放前，可以向固定的输入缓冲区写入新 token、新位置和新元数据。因此，**重放的是执行流程，输入数据仍然可以变化，输出也会重新计算**。图内使用的内存地址和执行结构则需要满足固定条件。[PyTorch 的 CUDA Graph 说明](https://docs.pytorch.org/docs/stable/notes/cuda#cuda-graphs)



#### 4.1.1  `NONE`：不使用 CUDA Graph

```text
CPU 提交矩阵乘法 A
CPU 提交 Attention
CPU 提交矩阵乘法 B
CPU 提交激活函数
```

没有捕获，也没有重放。

所有计算照常在 GPU 上进行，区别只是每次都重新提交操作。



####  4.1.2 `PIECEWISE`：把 forward 拆成几个片段，分别捕获

假设 Attention 留在图外，那么执行结构可能是：

```
[图 1：矩阵乘法 A]
          ↓
[正常提交：Attention]
          ↓
[图 2：矩阵乘法 B → 激活函数]
```

运行时，CPU 的工作变成：

```
重放图 1 → 提交 Attention → 重放图 2
```

这里的 `piece` 指的是**计算流程的片段**。具体切在哪里由编译配置决定，上面只是简化示例。

这样做的原因是：某些操作难以安全地捕获，或者需要更灵活的执行路径。把它们放在图外，其他片段依然能使用 CUDA Graph。

所以 `PIECEWISE` 的特点是：

- 捕获一部分计算，图外部分正常执行。
- 兼容性比较好。
- 仍然存在多次图重放和图外操作的提交开销。



#### 4.1.3 `FULL`：把整个模型 forward 捕获为完整图

```text
[完整图：矩阵乘法 A → Attention → 矩阵乘法 B → 激活函数]
```

运行时，CPU 可以通过一次 graph launch 提交这段完整工作。

这里的“完整”是指**被捕获的模型 forward 范围**。一个请求生成几百个 token，仍然需要很多轮 forward；`FULL` 并不表示把整个回答的生成过程一次性记录下来。

它要求 Attention 等操作也支持图捕获，否则就无法把这段 forward 完整包进去。



#### 4.1.4 `FULL_DECODE_ONLY` decode 时用`FULL`，其他情况不用图

这种模式根据请求的实际状态来选择 CUDA graph 的执行模式。当请求批次是 decode 阶段时，采用  `FULL` 模式，其他情况就用 `NONE`。

例如，处理一个新请求的 prompt 时正常执行；进入符合条件的 decode 阶段后，重放完整图。

这个模式省去了为其他批次准备分段图的开销。它适合主要负责 decode 的实例。



#### 4.1.5 `FULL_AND_PIECEWISE`：decode 时用`FULL`，其他批次用`PIECEWISE`

它的选择规则是：

```
符合条件的 decode 批次 → FULL
其他批次             → PIECEWISE
```

相比上一项，变化只在“其他批次”：

```
FULL_DECODE_ONLY：其他批次不使用图
FULL_AND_PIECEWISE：其他批次使用分段图
```

因此它需要准备 decode 完整图，以及其他批次的分段图。通常会增加捕获时间和内存占用。

它也不会在同一次 forward 中先执行完整图，再执行分段图；每次 forward 根据批次类型选择一种执行方式。



#### 4.1.6 总结

| 模式                 | 捕获范围                                       | 一次 forward 的执行方式                          |
| -------------------- | ---------------------------------------------- | ------------------------------------------------ |
| `NONE`               | 不捕获                                         | 正常提交 GPU 操作                                |
| `PIECEWISE`          | 捕获多个片段                                   | 重放片段，片段之间正常执行                       |
| `FULL`               | 捕获完整 forward                               | 重放完整图                                       |
| `FULL_DECODE_ONLY`   | 捕获 decode 的完整 forward                     | decode 请求重放完整图；其他情况正常提交 GPU 操作 |
| `FULL_AND_PIECEWISE` | decode 请求捕获完整 forward； 其他情况捕获片段 | decode 请求重放完整图；其他情况重放片段          |



### 4.2 vLLM 的两种请求模式 Uniform Decode 与 Non-Uniform Batching

在 vLLM 的请求调度中，并没有明确去用 Prefill 和 Decode 这两个概念去区分请求的计算阶段，而是通过 `num_new_tokens`  表示本轮实际参与计算的 token 数量，来对请求做区分。 这种方式既兼容了 Prefill 和 Decode 这两个概念，同时也可以扩展到 chunked prefill 与 **投机采样**流程中。 

举例来说：对于 Prefill 阶段的请求，它参与计算的 token 数可能等于输入 prompt token 数量，而对于 Decode 阶段的请求，参与计算的 token 数量就为 1， 而 MTP 为 1 + K （K 是草稿模型输出的 token 数量）。

根据一个请求批次中 query length 是否一致，分为了 Uniform Decode 和 Non-Uniform Batching 两种类型。



#### 4.2.1 Uniform Decode

uniform decode 表示在一个批次中所有请求的 query length 都是相同的。

**普通自回归解码**

如果一个批次中的请求都处于 decode 阶段，那么它们的 query length 都为 1。

```text
请求 A： [a]
请求 B： [b]
请求 C： [c]

query lengths = [1, 1, 1]
```



**推测解码的主模型验证**

如果每个请求都有 \(K=3\) 个草稿 token，常见的验证输入包含：

```
请求 A： [已确定 token, 草稿 1, 草稿 2, 草稿 3]
请求 B： [已确定 token, 草稿 1, 草稿 2, 草稿 3]
请求 C： [已确定 token, 草稿 1, 草稿 2, 草稿 3]

query lengths = [4, 4, 4]
```

这也属于 Uniform Decode。因此，**Uniform Decode 并不局限于每请求一个 token**。vLLM 的 dispatcher 会结合配置的 `uniform_decode_query_len` 判断是否适合对应的执行路径。[dispatcher 定义](https://docs.vllm.ai/en/latest/api/vllm/v1/cudagraph_dispatcher/)



#### 4.2.2 Non-Uniform Batching

Non-Uniform Batching 代表在一个请求批次中 query length 不相同，这通常是 prefill 阶段，或者是请求批次中混合 prefill 与 decode 阶段的请求。



最常见的是 prefill 和 decode 混合：

```text
请求 A：decode，处理 1 个 token
请求 B：decode，处理 1 个 token
请求 C：prefill，处理 128 个 prompt token

query lengths = [1, 1, 128]
```

也可以是不同长度的 prefill：

```
query lengths = [128, 256, 64]
```

或者草稿数量不同的验证批次：

```
query lengths = [4, 2, 3]
```



#### 4.2.3 两种请求的优化

Uniform Decode 的 query 形状更规整，一些 attention backend 能针对它使用专门的执行路径，可以使用`FULL` 模式将整个 forward 过程都捕获下来，后续的 forward 流程只需要执行 CUDA Graph 的重放就行。

对于 Non-Uniform Batching 如果采用 `FULL_DECODE_ONLY` 那么就不使用 CUDA Graph 进行正常提交，或者使用  `FULL_AND_PIECEWISE` 采用分段方式进行捕获和重放。

| CUDA Graph 配置      | Uniform Decode  | Prefill / 一般混合批次 |
| -------------------- | --------------- | ---------------------- |
| `FULL_DECODE_ONLY`   | 完整 CUDA Graph | 不使用 CUDA Graph      |
| `FULL_AND_PIECEWISE` | 完整 CUDA Graph | 分段 CUDA Graph        |



#### 4.2.4 MTP 的 CUDA Graph 优化

我们再将 MTP 的情况考虑进来， MTP 的计算分为两个阶段：

```
MTP 模块产生草稿 token
            ↓
目标模型验证这些草稿 token
```

以每个请求产生三个草稿 token 为例。对于目标模型，验证时通常要一起处理：

```
当前待处理 token + 草稿 1 + 草稿 2 + 草稿 3
```

因此，每个请求本轮的 query length 是 `4`。

如果三个请求都提交三个草稿：

```
请求 A：q = 4
请求 B：q = 4
请求 C：q = 4
```

**这仍然是 Uniform Decode，只是统一长度从普通 decode 的 1 变成了 4。**

于是，目标模型验证阶段的执行方式如下：

| 模式                 | 验证三个请求，每个请求三个草稿           |
| -------------------- | ---------------------------------------- |
| `NONE`               | 正常执行验证 forward                     |
| `PIECEWISE`          | 用分段图执行验证 forward                 |
| `FULL`               | 用兼容的通用完整图                       |
| `FULL_DECODE_ONLY`   | 用支持 `query_length=4` 的 decode 完整图 |
| `FULL_AND_PIECEWISE` | 用支持 `query_length=4` 的 decode 完整图 |



### 4.3 具体优化内容

有了前面的关于 CUDA Graph 和**请求类型**介绍，我们就可以继续来聊，如何使用 CUDA Graph 对请求进行优化。

首先，项目设置 `CUDAGRAPH_MODE=FULL_DECODE_ONLY` 这样确保 **Uniform Decode** 请求使用 CUDA Graph 来加速推理。

CUDA Graph 捕获时，会记录 GPU 操作、执行依赖、kernel 参数和使用的内存地址。后续重放这张图，需要保持兼容的执行结构和缓冲区尺寸；**缓冲区里的数据可以在每次重放前更新**。因此，运行时会提前准备几种常用尺寸的图，供后续批次选择。

具体影响输入大小的有两个运行时候的参数，一个是 batch 中的请求数（记作 S），另外一个就是 MTP 草稿 token 数量（记作 K）， 目标模型在验证的时候还会额外再生成一个 token 所以最终的 query_length = 4。

项目以 `MAX_NUM_SEQS` 来设置最大的请求数，`S=1..MAX_NUM_SEQS` ，一个 batch 的 token 数量就为 `(K+1)×S`, `MAX_NUM_SEQS=4` 会得到 `{4,8,12,16}`。如果图列表只有 4、8、16，宽度 12 的批次可能补齐到 16 后重放，不能补齐或没有兼容图时则正常执行。



## 五、使用 Chunked Prefill 控制长输入峰值

### 5.1 Chunked Prefill 的意义

如果一个 128k prompt 一次进入模型，GPU 会同时处理大量 token，这可能导致对于显存需求超过启动时的预设上限。另外，在处理多请求的情况下，一个长 prompt 请求会阻塞其他处于 decode 阶段的请求，导致它们的 token 延迟大幅度的增加。

Chunked Prefill 通过把长 prompt 拆成多个小的 chunk，分成多个调度计算批次，来解决这个问题。

```text
128k prompt
  ├── chunk 1: 2048 tokens
  ├── chunk 2: 2048 tokens
  ├── ...
  └── chunk 64: 2048 tokens
```

每个 chunk 完成后，必要的 K/V 和 recurrent state 被写入 cache，临时数据可以释放或复用。这样把峰值从“完整 prompt 的激活”限制为“一个 chunk 的激活”。

分块也允许调度器在长 prefill 之间插入 decode 工作，避免一个超长 prompt 长时间独占整个 engine。不过是否真正改善在线延迟，还取决于 chunk width 和调度策略。



### 5.2 token chunk 大小的取舍

较大的 chunk 具有更好的矩阵规模，可以减少 chunk 之间的固定开销。项目对相同服务做过 2,048 和 8,192 的对照，在一组 32k prompt 测量中：

| chunk width |       prefill |   TTFT |
| ----------: | ------------: | -----: |
|       2,048 | 2,133 token/s | 15.38s |
|       8,192 | 2,366 token/s | 13.87s |

8,192 token chunk 的 prefill 吞吐提高约 10.9%，TTFT 降低约 9.8%。代价是 peak activation 增加；vLLM 在 profile 后从同一 GPU budget 中缩小 KV pool。项目另一组 matched measurement 中，KV token capacity 从 1,282,724 降到 1,249,637，约减少 2.6%。

因此当前默认仍是 2,048，8,192 作为 opt-in。对于主要处理离线长文档、并发很低的场景，较大 chunk 可能更合适；如果更看重 KV 容量、在线混合请求和内存余量，默认值更稳妥。



### 5.3 chunk 对 decode 的影响

当已有请求正在 decode，又到来一个长 prompt 时，prefill chunk 会占据一个较长的 engine step。项目测量过两个 decode stream 与一个 64k prompt 并存的情况：

- 2,048-token chunk 下，prefill 期间 streamed chunk 间隔 p95 约为 1,111ms；
- 改为 1,024-token chunk 后，p95 降到约 666ms；
- 代价是该 64k prompt 的 prefill throughput 下降约 5.5%。

这个实验仍然说明：更大的 prefill chunk 有利于总吞吐，却会让同时进行的 decode 等待更久。

所以，Chunked Prefill 没有对所有服务都适用的“最佳值”。离线批处理通常偏向大 chunk；交互式服务需要控制长 prefill 对 decode tail latency 的影响。



### 5.4 PLE 与 Chunked Prefill 也有联动

默认 2,048-token chunk 会触发约 32,768 个 PLE row lookup。较大的 chunk 提供更大的预取集合，有利于 NVMe queue depth 和批量 page fault，但也会增加 gather、传输和 PLE short convolution 的临时激活。

PLE 位于网络前部，CPU lookup 和 GPU 第一层计算的重叠窗口有限。chunk 增大后，CPU 可以一次准备更多行，GPU 也有更长的前部计算；哪一边成为瓶颈需要通过 CPU wait time、page fault 和 GPU trace判断，不能只根据理论并行性推断。



## 六、小结

本文讨论的优化参数可以总结在一张表中：

| 参数                   | 增大后的主要收益                     | 主要成本                                    |
| :--------------------- | :----------------------------------- | :------------------------------------------ |
| MTP K                  | 一轮可能接受更多 token               | draft 计算、输出头读取、verify width 增大   |
| CUDA Graph sizes       | 更多 batch width 可以直接重放        | 启动更慢，常驻图更多                        |
| MAX_NUM_SEQS           | 短请求 aggregate throughput 可能提高 | KV、专家、PLE 和调度压力增加                |
| MAX_NUM_BATCHED_TOKENS | prefill GEMM 效率提高、TTFT 下降     | 激活峰值增大，KV pool 缩小，decode 等待变长 |

当前示例配置的目标比较明确：

```text
原生 context:              262,144
MTP speculative tokens:   3
MAX_NUM_SEQS:             4
MAX_NUM_BATCHED_TOKENS:   2,048
CUDAGRAPH_CAPTURE_SIZES:  auto
CUDAGRAPH_MODE:           FULL_DECODE_ONLY
COMPILATION_MODE:         0
```

它优先保证单机长上下文和可复现性，没有追求短 context 下的最大并发峰值。



**调参时的测试顺序**

同时修改 K、并发、chunk 和 graph mode 后，即使吞吐变化，也很难知道原因。更可靠的顺序是：

1. 固定 KV dtype、context 和 `MAX_NUM_SEQS`，扫描 MTP K，记录 step time、accepted tokens/step 和 aggregate tokens/s。
2. 对选定 K 检查每个 `(1+K)×S`是否命中 full graph。
3. 保持 decode 配置不变，扫描 prefill chunk，记录 TTFT、prefill throughput、peak activation 和最终 KV pool。
4. 增加混合流量测试，观察长 prefill 到来时的 decode p95/p99 step gap。
5. 最后再调整 `MAX_NUM_SEQS`，并检查最大长请求是否仍能放入 cache。

每次测试应记录是冷启动还是热服务、是否启用 reduced draft vocab、实际 graph mode 和 PLE page-cache 状态。缺少这些条件时，不同轮的 tokens/s 很难比较。



**MTP、CUDA Graph 和 Chunked Prefill 分别处理不同的运行时成本**：

- MTP 用较小的 draft 计算换取 target model 一次验证多个位置；
- CUDA Graph 减少稳定 decode shape 上的 kernel launch 和调度开销；
- 动态批处理把多条序列组合成 GPU 更容易利用的执行宽度；
- Chunked Prefill 限制长 prompt 的单轮激活，并在吞吐、KV 容量和在线延迟之间取舍；
- reduced draft vocab 可降低 MTP 每步读取流量，但不是默认配置，但是会增加内存占用。

这些参数最终受同一个 128GB UMA 预算约束。下一篇将把权重、PLE page cache、KV Cache、容器进程和 driver 分配放回统一内存池，说明启动脚本怎样计算预算，以及一次请求实际如何经过 CPU、GPU 和 NVMe。



## 参考资料

- Qwen Team：[Qwen3.8-Flash-Next Architecture](https://qwen.ai/blog?id=qwen3.8-flash-next)
- vLLM：[MTP (Multi-Token Prediction)](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
- vLLM：[CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)
- vLLM：[Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)
- MiaAI-Lab：[Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)

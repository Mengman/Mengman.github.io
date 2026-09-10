---
title: 【Qwen3.8-Flash-Next 端侧部署】02-PLE 如何从 NVMe 送到 GPU
date: 2026-09-10T09:23:47.791Z
tags: [qwen, vllm, aiinfra, dgx-spark]
categories: aiinfra
typora-root-url: ../../../
---

上一篇介绍整体部署方案时，我们把 Qwen3.8-Flash-Next 的参数分成了两类。一类是注意力、GDN、MoE 和视觉塔中的矩阵权重，它们会在推理过程中被 GPU 反复读取；另一类是 n-gram embedding 查找表，它的容量很大，但每个 token 只访问少量离散表项。

这两类参数不应该采用相同的部署方式。前者需要留在 GPU 能够高效访问的位置，后者则可以放到容量更大、速度更慢的存储层级中。MiaAI-Lab 的单机 DGX Spark 项目正是利用了这种差异：约 26.82GiB 的量化 PLE 表存放在本地 NVMe，通过 mmap 映射到 CPU worker；GPU 只接收当前批次实际查到的 packed rows，再完成反量化和后续计算。

这种做法常被概括为“PLE CPU offload”，但这个名称容易让人误以为整个 PLE 模块都在 CPU 上执行。实际上，CPU 只负责从 token 序列得到查找位置，并从大表中收集对应行。PLE 的投影、门控、短卷积以及与 hidden states 的融合仍在 GPU 上完成。

本文基于 Qwen Team 的 [Qwen3.8-Flash-Next 架构介绍](https://qwen.ai/blog?id=qwen3.8-flash-next)和[技术报告](https://arxiv.org/abs/2608.30320)，并结合 MiaAI-Lab 的 [Qwen3.8-Flash-Next-Single-DGX-Spark 项目](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)进行分析。感谢 Qwen 团队公开模型架构和实验结果，也感谢 MiaAI-Lab 将 PLE 打包、mmap 访问、预取以及 GB10 同步补丁完整开源。

## 先把 PLE 这个名称说清楚

PLE 通常是 Per-Layer Embedding 的缩写。Google 在 [Gemma 3n 文档](https://ai.google.dev/gemma/docs/gemma-3n)中使用这个名称，指可以在模型运行内存之外生成或缓存、再随各层执行注入模型的 embedding 参数。

Qwen3.8-Flash-Next 的设计受到 Gemma 3n PLE 和 Engram 等工作的启发，但 Qwen 官方对自己的结构使用的是 N-gram Embedding。它并没有给每一层配置一张大型 embedding 表，而是在网络靠前位置放置一个 n-gram embedding 模块。技术报告中的消融实验显示，在固定参数预算下，把表分散到多个层没有稳定收益；最终模型使用单个 n-gram embedding 层，并把它放在第二个网络层附近，使查表可以与第一层计算重叠。

项目和 vLLM 的代码沿用了 PLE 这个模块名。为了与代码一致，本文也使用 PLE，但它在这里具体指 Qwen3.8-Flash-Next 的单个 n-gram embedding 子层，而不是泛指所有 Per-Layer Embedding 方案。

这个区别不仅是术语问题，它直接影响部署设计。如果模型每一层都依赖一张不同的大表，那么 CPU 必须在前向过程中反复等待新的 hidden states 或层级信息，预取和流水化会复杂得多。Qwen 当前的 n-gram lookup 只依赖输入 token、位置和局部上下文，而且只在网络前部消费一次，这才形成了相对独立的 CPU 查找链路。

## PLE 在模型中做什么

### token embedding 与 n-gram embedding

假设输入有 512 个 token。tokenizer 先把文本转换为长度为 512 的 token ID 序列：

~~~text
[t0, t1, t2, ..., t511]
~~~

普通 token embedding 根据每一个 token ID 查询基础词表。若模型 hidden size 为 H，查询后得到的张量形状为：

~~~text
[512, H]
~~~

从这里开始，Transformer 主干处理的是连续的 hidden-state 向量，不是一个个 token ID。每一层输入和输出的主干形状仍然近似为 <code>[sequence_length, hidden_size]</code>；Qwen3.8-Flash-Next 使用四分支 Gated Residual 后，运行时还会维护扩展的残差状态。

不过，推理引擎不会在完成初始 embedding 后丢掉 token ID。调度器仍需要保存每条序列的 token、位置、长度和边界。PLE 正是从这组离散信息建立另一条支路。

对于位置 i，unigram embedding 的查找键只有当前 token：

~~~text
key1(i) = ti
~~~

n-gram embedding 则把以当前位置结束的局部上下文作为键。例如：

~~~text
bigram:  (t(i-1), ti)
trigram: (t(i-2), t(i-1), ti)
~~~

实际实现不会为所有可能的 token 组合建立一个无碰撞的巨大字典，而是通过确定性的 hash 把不同 n-gram 映射到有限大小的表中。多个 hash head 可以为同一局部上下文生成多个表索引，降低单次哈希碰撞对表示能力的影响。当前 checkpoint 每个 token 共查询 16 个 head，每个 head 返回一个 160 维向量，拼接后的 PLE embedding 宽度为 2,560。

这里的 n-gram embedding 不是 token embedding 的替代品。token embedding 提供当前 token 的基础表示，n-gram embedding 提供与局部上下文相关的附加记忆。两条路径最终在模型前部汇合。

### 查表之后还有哪些计算

从 vLLM 的 Qwen3.8-Flash-Next 实现看，PLE 的完整路径可以概括为：

~~~text
token IDs、positions、sequence boundaries
              │
              ▼
       计算多组 n-gram ID
              │
              ▼
        embedding table lookup
              │
              ▼
        反量化 PLE vectors
              │
              ▼
          K/V projection
              │
              ├──────────────┐
              ▼              │
       normalization + gate  │ 当前层 hidden states
              │              │
              ▼              │
      dilated short convolution
              │
              ▼
       residual accumulation
~~~

前两步只需要 token ID、位置、序列边界和固定的哈希参数。大表查询也只需要 n-gram ID。CPU 可以在不知道当前层 hidden states 数值的情况下完成这些工作。

反量化之后的投影、归一化和 gate 则使用模型参数，其中门控和残差写入还依赖当前 hidden states。short convolution 也维护与序列相关的状态。因此，CPU 可以独立准备 PLE lookup result，但不能独立算出最终注入模型的 PLE 输出。

换一种说法，CPU 链路提供的是“当前 token 需要的条件记忆材料”，GPU 决定这些材料如何根据当前网络状态进入残差流。

## n-gram embedding 的模型作用

大语言模型需要同时处理两类问题。一类是局部而稳定的模式，例如常见短语、词法组合、固定代码片段和局部语法；另一类是依赖当前问题、长上下文和多步推理的动态关系。

标准 Transformer 只能通过矩阵计算和注意力共同学习这两类能力。即使某个局部模式在训练数据中反复出现，模型也要在每次前向中通过若干层重新构造相应表示。

[Engram 论文](https://arxiv.org/abs/2601.07372)把 n-gram lookup 称为 **conditional memory**，并将它与 MoE 的 **conditional computation** 区分开来：

- MoE 根据 token 选择少量专家，把更多参数容量放到条件计算中。
- n-gram memory 根据离散上下文直接寻址，把更多参数容量放到条件存储中。

查找表变大时，每个 token 仍然只读取固定数量的表项，所以查找计算量不会随表容量等比例增长。模型可以用较少的额外 FLOPs保存大量局部模式，再把主干网络的计算留给更动态的组合与推理。

Engram 的机制分析还给出了两个有用的解释。第一，条件记忆承担一部分静态模式重建后，主干网络的早期层不必反复完成同样的工作，相当于给后续推理留下了更多有效深度。第二，局部依赖由 lookup 处理后，注意力有更多容量用于全局上下文。

这些结论可以解释 Qwen 采用 n-gram embedding 的方向，但不能把 Engram 的所有实验直接当作 Qwen3.8-Flash-Next 的结论。Qwen 自己的技术报告提供了更直接的消融结果：

- 加入 n-gram embedding 后，多个知识、数学、推理和多语言评测相对无 n-gram baseline 有提升。
- 增大 n-gram vocabulary 时，训练 loss 持续下降，但下游评测会逐渐饱和或波动。
- 在固定参数预算下，n-gram memory 与 MoE expert 之间存在分配取舍，并不是表越大越好。
- 把同样的 n-gram 参数分到多个网络层，没有稳定优于单层方案。
- 最终选择单层并放在网络靠前位置，既保留效果，也便于 host-memory prefetch。

所以，**PLE 的意义不是提供一个外部知识库，也不是在推理时做语义向量检索。它是模型训练出来的、以局部 token 模式为键的参数化条件记忆。**哈希索引和表中向量都是模型架构的一部分，部署时不能更换成普通向量数据库。

## 适合卸载的是查找表，不是整个子层

判断一个模块是否适合 CPU offload，不能只看它有多少参数，还要看参数怎样被访问。

PLE 查找表具有几个适合卸载的特征：

- 每个 token 访问的行数固定且很少。
- 查找地址可以由 token ID 和位置确定，不必等待深层 hidden states。
- gather 属于内存访问操作，没有足够大的矩阵计算让 GPU 算力发挥优势。
- 表容量很大，但绝大部分行在一次请求中不会使用。
- PLE 位于网络很靠前的位置，prefill 时可以在第一层执行期间准备结果。

与之相对，MoE、注意力和 GDN 权重会在每个请求中大范围、反复地参与矩阵乘法。把这些权重放在 NVMe 上按层读取，会让推理不断等待大块数据传输，访问模式和 PLE 完全不同。

项目把约 26.82GiB PLE 表移出 GPU 常驻权重后，GPU 侧模型权重从完整 checkpoint 的约 98.6GiB 降到约 71.75GiB。节省出的空间用于 vLLM/CUDA 运行时、MTP 和 KV Cache。

但 offload 并不等于这 26.82GiB 不再使用系统内存。mmap 文件的已访问页面会进入 Linux page cache，而 DGX Spark 的 page cache、CPU 进程和 GPU 分配共享同一个 128GB UMA 池。项目真正避免的是把整张表复制为不可轻易回收的匿名内存或 GPU 常驻参数。

## packed 文件打包 PLE 权重 

项目没有让 CPU worker 在 206 个 safetensors shard 中逐次解析和查找 PLE 权重，而是在第一次启动时运行 [build_ple_packed_table.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/build_ple_packed_table.py)，生成一个适合按行索引的连续文件。

### 原始数据由 128 个 row shard 组成

构建脚本首先读取 <code>model.safetensors.index.json</code>，找出名称匹配 <code>ngram_embedding.shard_0.weight</code> 的 PLE 参数前缀。当前 checkpoint 把表按行拆成 128 个 shard，每个 shard 又包含两部分：

- <code>weight</code>：U8 张量，每个字节存两个 NVFP4 code；
- <code>weight_scale</code>：FP8 E4M3 张量，每 16 个原始数值对应一个 block scale。

此外还有一个 FP32 global scale，也就是代码中的 <code>weight_scale_2</code>。它对所有 shard 保持一致，不需要复制到每一行 packed data 中。

构建器逐个 mmap safetensors 中对应的数据区间，把一个 shard 的 code 和 block scale 在列方向拼接，再按 shard 顺序写入输出文件：

[`files/build_ple_packed_table.py`](./files/build_ple_packed_table.py) 中真正完成布局转换的是下面几行：

```python
w0, dt = view(f"{prefix}.shard_0.weight"); assert dt == "U8", dt
s0, dt = view(f"{prefix}.shard_0.weight_scale"); assert dt == "F8_E4M3", dt
rows, cw = w0.shape
sw = s0.shape[1]
width = cw + sw

with open(tmp, "wb") as out:
    for i in shards:
        w, _ = view(f"{prefix}.shard_{i}.weight")
        s, _ = view(f"{prefix}.shard_{i}.weight_scale")
        for c in range(0, rows, CH):
            np.concatenate([w[c:c + CH], s[c:c + CH]], axis=1).tofile(out)
```

`np.concatenate(..., axis=1)`说明 codes 与 scales 是同一行内的两个连续区段；外层按 shard 顺序遍历，因此全局 row ID 可以直接映射为文件中的固定行偏移。

~~~text
shard 0 rows
shard 1 rows
...
shard 127 rows
~~~

输出文件旁边还有一份 JSON metadata，记录 <code>rows_per_shard</code>、<code>num_shards</code>、<code>row_width</code>、code 和 scale 的宽度以及总行数。worker 加载 mmap 时会验证文件大小、行数和宽度，避免 checkpoint 与 packed artifact 不匹配。

构建过程按块流式执行，不需要先在内存中展开整张表。脚本注释给出的峰值 RAM 低于 1GiB；写入临时文件并完成大小校验后再重命名，避免中断时把半成品当作有效缓存。

### 一个 160 维向量在 packed 文件中占 90 字节

当前模型每个 PLE head 的维度是 160。NVFP4 用 4 bit 保存一个 code，因此 160 个 code 占：

~~~text
160 × 4 bit = 640 bit = 80 byte
~~~

量化以 16 个值为一个 block，所以一行有：

~~~text
160 / 16 = 10 个 block
~~~

每个 block scale 使用一个 FP8 E4M3 数值，共 10 字节。于是 packed row 宽度为：

~~~text
80 byte codes + 10 byte block scales = 90 byte
~~~

一个 token 查询 16 个 head，CPU worker 需要向 GPU 输出：

~~~text
16 × 90 byte = 1,440 byte/token
~~~

如果 CPU 先把全部 2,560 个值展开成 BF16，再交给 GPU，则需要：

~~~text
2,560 × 2 byte = 5,120 byte/token
~~~

保持 packed 表示可以把这部分传输量降到约 28%，同时避免 CPU 为反量化结果分配更大的中间缓冲区。项目因此让 CPU 返回 code 和 scale，GPU 只对当前批次查到的行执行反量化。

### NVFP4 code 和 scale 如何反量化

一个字节的低 4 bit 和高 4 bit 分别是两个 code。每个 code 中，低 3 bit 选择幅值：

~~~text
0, 0.5, 1, 1.5, 2, 3, 4, 6
~~~

最高 bit 表示正负号。得到 FP4 value 后，每 16 个值乘以对应的 FP8 block scale，最后再乘整个表共享的 FP32 global scale：

~~~text
dequantized value
  = FP4 code value
  × FP8 block scale
  × FP32 global scale
~~~

FP8 block scale 的任务是为每个小块恢复局部动态范围，global scale 再调整整体尺度。这些 FP8 scale 属于 NVFP4 PLE 行的量化元数据，与后续文章中的 FP8 KV Cache 不是同一类数据。

## 使用 mmap 加载 PLE 参数

packed 文件生成后，CPU worker 使用只读 <code>numpy.memmap</code> 建立二维 U8 视图，再通过 <code>torch.from_numpy</code> 得到零拷贝、file-backed 的 tensor。原本为 embedding weight 和二维 scale 参数预留的匿名内存随即被替换为空 tensor。

下面是 [`files/patch_ple_offload.py`](./files/patch_ple_offload.py) 写入 vLLM worker 的关键逻辑。为了可读性，这里移除了 patch generator 外层的字符串引号，代码语义未改：

```python
mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(rows, width))
mm._mmap.madvise(mmap.MADV_RANDOM)
table = torch.from_numpy(mm)  # zero-copy, file-backed, evictable

emb._packed_table = table
emb._packed_table_mmap = mm
emb._packed_table_fd = os.open(path, os.O_RDONLY)

# Release the never-touched anonymous allocations.
emb.weight.data = torch.empty(0, dtype=emb.weight.dtype)
if emb.weight_scale is not None and emb.weight_scale.dim() == 2:
    emb.weight_scale.data = torch.empty(0, dtype=emb.weight_scale.dtype)
```

这段代码同时说明了三件事：数据来自只读文件映射，`torch` tensor 与映射共享存储，原来的匿名 weight/scale allocation 会被释放。PLE offload 的内存收益正是由这三步共同产生的。

这样做与普通文件读取有明显区别。

如果启动时调用 read 把 26.82GiB 文件全部装入进程内存，数据会以匿名 RSS 或普通进程 buffer 的形式长期存在，和 GPU 权重一起挤压 UMA。mmap 只建立虚拟地址映射；CPU 第一次访问某一行时，内核才把包含该行的页面从 NVMe 读入 page cache。

暂时不用的 file-backed page 可以被 Linux 回收，之后再次访问时重新从 NVMe 读取。这种可回收性让 PLE 表能够在有限内存中运行。

不过，mmap 不是“磁盘上的无限内存”。已经 fault 进来的页面仍然占物理内存；访问模式不合理时，内核可能为每次小查询读入大量不需要的邻近数据。DGX Spark 上的优化重点因此从“如何加载整个表”变成“每次 page fault 到底带入多少数据”。

## 随机查表与 Linux readahead

### 默认 fault-around 带来的额外读取

Linux 通常假定文件映射存在一定的局部性。当一个页面发生 fault 时，内核可能把周围一段文件一起读入，后续顺序访问就不必逐页等待磁盘。

这对模型权重的连续读取很有效，但 PLE 的 hash lookup 接近随机访问。当前 token 查询的一行只有 90 字节，下一个 n-gram ID 很可能落在文件的远处。项目在未设置访问提示时观察到，内核可能为了一个 90 字节行读取约 64KiB 的 fault-around 窗口，其中绝大部分不会再次使用。

这里浪费的不只是 NVMe 带宽。无效读入的页面会停留在 page cache，而 page cache 与 GPU 分配共享 UMA。如果每个 decode token 都把许多无关页面带入内存，最终会挤压 KV Cache 和驱动所需的空闲页。

### MADV_RANDOM 的作用

[patch_ple_offload.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_offload.py)在建立 memmap 后调用：

~~~text
madvise(MADV_RANDOM)
~~~

这个提示告诉内核：应用将随机访问这段映射，不要按照顺序流式读取来做大范围预取。一次缺页主要读入实际包含目标行的 4KiB 页面。如果一个 90 字节行跨越页面边界，则需要两个页面。

项目在同一台 DGX Spark 上记录了以下变化：

| 指标 | 默认 mmap | MADV_RANDOM |
| --- | ---: | ---: |
| 每个 decode token 的磁盘读取量 | 约 1,366KiB | 约 57KiB |
| Host MemAvailable | 约 10.9GiB | 约 12.95GiB |

磁盘读取量下降约 24 倍，空闲可用内存增加约 2GiB。项目同时说明 decode 速度没有可测量的变化，因为原来的读取量仍未达到 NVMe 吞吐瓶颈；主要收益是减少无效 page cache，而不是提高磁盘顺序带宽。

这些数字是该项目主机上的测量结果，不应直接当作所有 SSD 和内核版本的固定比例。可复用的结论是：对小行、随机 hash lookup，默认 fault-around 可能比真正需要的数据大几个数量级，访问提示必须与工作负载匹配。

## prefill 需要另一种预取

<code>MADV_RANDOM</code>阻止内核无依据地猜测相邻页面，但 prefill 阶段又存在一批可以提前知道的访问位置。项目没有放弃这部分信息，而是增加了显式批量预取。

### prefill 阶段可以提前确定表项

处理 prompt 时，整段 token ID 已经由 tokenizer 给出。即使 GPU 还在执行第一层，CPU 也可以为当前 prefill chunk 计算全部 n-gram ID，并得到将要访问的 packed row offset。

以默认 2,048-token chunk 为例，每个 token 查询 16 行，一个 chunk 可能产生：

~~~text
2,048 × 16 = 32,768 次 row lookup
~~~

这些行在文件中分散，但地址集合是已知的。CPU 可以先把每一行覆盖的页面算出来，去重并排序，再通过：

~~~text
posix_fadvise(fd, offset, 4096, POSIX_FADV_WILLNEED)
~~~

通知内核这些页面很快会被使用。NVMe 可以同时看到一批请求，避免单线程 <code>index_select</code>在每一行上串行等待一次缺页。

项目实现还考虑了 row 跨页的情况：为每个 offset 同时计算起始页和结束页；页面集合通过 unique 去重，排序后的文件偏移也更适合块设备调度。预取只是 advisory，失败时会退回正常 page-fault 路径，不影响结果正确性。

### MADV_RANDOM 与 WILLNEED 并不冲突

两者解决的是不同问题：

- <code>MADV_RANDOM</code> 表示不要因为访问某一页，就自动把大片相邻区域也读入。
- <code>POSIX_FADV_WILLNEED</code> 明确列出应用已经知道将要访问的页面。

前者减少错误猜测，后者利用确定信息。对 PLE 这种随机但可批量预知的访问，两者配合比单纯依赖通用 readahead 更合适。

项目在一次冷的 280-row 独立 gather 测量中观察到约 13 倍加速，而 decode 上只有约 3% 变化，因为 decode 每个 engine step 的 page fault 数量较少。README 将 prefill 吞吐提升主要归因于这项预取，但也明确说明没有做完全隔离的 A/B 测试。因此，更准确的表述是：实现机制和单项微基准支持这一解释，端到端提升的归因仍包含推断。

## CPU 与 GPU 的职责边界

把 PLE 链路画成两个并行部分，会比只说“offload 到 CPU”更准确：

~~~text
CPU / NVMe                                  GPU

token IDs、positions
        │
        ▼
计算 n-gram IDs                     token embedding
        │                                  │
        ▼                                  ▼
计算 packed row offsets              第一层主干计算
        │                                  │
        ▼                                  │
预取页面并 gather rows                      │
        │                                  │
        └──────── packed rows ──────────────┤
                                           ▼
                                  NVFP4 PLE 反量化
                                           │
                                           ▼
                                  projection / gate
                                           │
                                           ▼
                                  short conv / 融合
                                           │
                                           ▼
                                      后续模型层
~~~

CPU 侧工作包括：

- 按序列边界和位置计算 n-gram ID；
- 把不同 hash head 的 ID 转换为全局表行号；
- 对即将访问的页面发出预取提示；
- 从 mmap tensor 中 gather 90 字节 packed rows；
- 把当前批次结果组织到共享传输缓冲区。

GPU 侧工作包括：

- 接收 packed code 和 block scale；
- 使用 GPU 上保留的 global scale 反量化；
- 把各 head 的 160 维结果还原并拼成 2,560 维 PLE embedding；
- 执行 K/V projection、归一化、gate 和 short convolution；
- 根据当前 hidden states 更新和写回残差状态；
- 继续执行后续 GDN、QSA 和 MoE 层。

把反量化放在 GPU 还有一个实际好处：CPU 到 GPU 只传 1,440 字节/token packed data。如果在 CPU 展开为 BF16，传输量会变成 5,120 字节/token，并增加 CPU 中间内存。

## prefill 和 decode 的协作方式不同

### prefill 阶段

prefill 的输入 token 全部已知，CPU 可以为当前 chunk 一次计算大量 n-gram ID。PLE 位于网络靠前位置，因此理想执行顺序是：

~~~text
CPU：n-gram hash → 页面预取 → gather
GPU：token embedding → 第一层计算
                          │
两条路径在 PLE 子层前汇合 ┘
~~~

只要 CPU 查询在 GPU 到达 PLE 位置前完成，查表延迟就能被前一段 GPU 计算部分隐藏。实际重叠程度取决于 chunk 大小、page cache 热度、NVMe 延迟和同步实现。

默认 2,048-token chunk 会产生较多 row lookup，批量预取更容易发挥作用；同时，chunk 不能无限增大，因为 PLE short convolution 和其他 prefill 激活也会占用内存。chunk 大小的整体取舍会在运行时专题中讨论。

### decode 阶段

decode 每一步的新输入来自上一轮刚刚生成或验证通过的 token。在 token 确定之前，CPU 无法知道它参与构成的下一组 n-gram。依赖关系是：

~~~text
GPU 生成或验证 token
        │
        ▼
CPU 计算新 n-gram 并查表
        │
        ▼
GPU 使用 PLE 结果完成下一次前向
~~~

因此，CPU 的查表链路虽然不依赖深层 hidden states，却仍然依赖上一轮 GPU 输出的 token ID。它不能在自回归生成开始前把未来所有 PLE 结果一次算完。

MTP 会让一次 engine step 处理多个待验证位置，CPU 可以为这一批已调度 token 统一 gather，但被拒绝的草稿会改变后续序列，仍然不能突破自回归的数据依赖。decode 阶段通常每步查询的行数少于 prefill，显式预取的收益也相对较小。

## GB10 上的同步适配

vLLM 原有 PLE CPU offload 设计使用 CUDA stream memory operations 在 CPU worker 与 GPU stream 之间建立 semaphore。大致逻辑是：

1. GPU 提供输入和输出缓冲区。
2. CPU worker 读取输入、完成 lookup，并异步复制结果到 GPU output buffer。
3. CPU/GPU 通过 stream wait/write value 操作表示缓冲区是否可用。
4. GPU stream 在结果到达后继续执行 PLE 和后续 kernel。

这种机制可以减少主线程阻塞，并允许 CPU 查表、数据复制和 GPU 前部计算重叠。

但项目在 GB10 上检测到 <code>CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0</code>。仓库中的问题记录和补丁注释说明，相关 wait 调用虽然可能返回成功，但后续 kernel 或 CUDA Graph replay 却不能按原设计完成同步，最终表现为模型在 warmup 后无法继续。

[patch_ple_offload.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_offload.py) 因此把 stream semaphore 替换为 host-side handshake。

### host-side handshake 的执行过程

GPU worker 为每个请求分配一个递增的 sequence number，并在共享 CPU 内存中维护 <code>int64[1]</code>完成标记：

~~~text
GPU worker                                      CPU worker

seq = seq + 1
发送 num_tokens、num_reqs、seq  ───────────────→ 接收请求
        │                                        │
        │                                  读取 token IDs
        │                                  计算 n-gram IDs
        │                                  mmap gather
        │                                        │
        │                                  H2D copy 到 GPU buffer
        │                                  copy stream synchronize
        │                                        │
等待 done_flag >= seq  ←────────────────── done_flag = seq
        │
        ▼
开始当前 forward / graph replay
~~~

CPU worker 会等所有 PLE 层目标缓冲区的 copy stream 完成，再写入对应 sequence number。GPU worker 先短暂 busy-spin，之后以很短的 sleep 间隔轮询；如果在默认 300 秒超时内仍未完成，则抛出异常，而不是无限等待。

CPU 侧发布完成状态的代码同样来自 [`files/patch_ple_offload.py`](./files/patch_ple_offload.py)：

```python
for dp_rank, request in requests_by_dp.items():
    flags = []
    for layer_name in self._layers:
        for target in self._worker_targets[dp_rank][layer_name]:
            target.copy_stream.synchronize()
            if target.done_flag is not None:
                flags.append(target.done_flag)
    for flag in flags:
        flag[0] = request.seq
```

先 `synchronize()`、后写 `request.seq` 的顺序非常关键。完成标记表达的是“当前序号的数据已经落入 GPU buffer”，而不只是“CPU 已经发起 copy”。

这样可以保证 GPU 开始执行当前 forward 时，PLE output buffer 已经完整写入。原来的 stream-memory semaphore 变成 no-op，eager execution 和 CUDA Graph 都遵循同一套主机完成条件。

### 稳定性与重叠之间的取舍

host-side handshake 的代价很直接：模型线程在 forward 之前等待 CPU worker，无法像原设计那样充分把 PLE 查表隐藏在前层 GPU 计算后面。

项目认为这项损失可接受，原因是 PLE 位于网络很靠前，原本可以用于重叠的 GPU 计算窗口只有前面的少量操作；而在 GB10 上继续使用不受支持的 stream memory operation 会导致服务无法启动。

这里也需要区分两种判断：

- Qwen 架构允许 host-memory prefetch 与模型前部计算重叠，这是模型层面的设计能力。
- 当前单 DGX Spark 补丁为了适配 GB10，采用更保守的同步方式，没有完全实现这种理想重叠。

所以，CPU lookup 可以形成独立数据准备链路，并不意味着当前实现中的 CPU 与 GPU 在所有阶段都完全异步。

## 启动时如何接入 vLLM

项目没有维护一份完整的 vLLM fork，而是在每次启动时从指定容器镜像提取原始文件，再通过 patch generator 生成替换文件并 bind mount 回容器。与 PLE 相关的入口包括：

- [build_ple_packed_table.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/build_ple_packed_table.py)：把 checkpoint 中的 PLE shard 流式整理为 packed mmap 文件；
- [patch_ple_layer.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_layer.py)：增加 NVFP4 PLE quant dispatch、packed row 反量化和页面预取；
- [patch_ple_offload.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_offload.py)：挂载 mmap table，并把 GB10 不支持的 stream semaphore 改为 host handshake；
- [start.sh](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/start.sh)：检查 artifact，设置环境变量并启动容器。

启动脚本通过以下环境变量启用这条路径：

```bash
# start.sh 中的 docker run 参数
-e VLLM_PLE_CPU_OFFLOAD=1 \
-e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR \
-e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \
-v $PATCHED_PLE:$PLE_PKG:ro \
-v $OFFLOAD_DIR/ple_offload_layer.py:$VLLM_PKG/model_executor/layers/ple_offload_layer.py:ro \
-v $OFFLOAD_DIR/connector.py:$VLLM_PKG/v1/ple_offload/connector.py:ro \
-v $OFFLOAD_DIR/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro
```

这段原始启动参数也解释了项目为什么不需要维护完整的 vLLM fork：生成后的补丁文件以只读 bind mount 覆盖容器中的对应模块，其他 vLLM 代码仍来自指定镜像。

如果 packed table 不存在，启动脚本会用一个受限的临时容器生成它。运行时 CPU worker 识别到 packed 文件后，不再把 checkpoint 中的二维 weight 和 block-scale shard 加载到匿名 RAM，而是直接附着只读 mmap table。global scale 和反量化查找表则保留在 GPU worker 中。

项目拒绝在单台 Spark 上设置 <code>PLE_OFFLOAD=false</code>。这不是一般性的 vLLM 限制，而是当前 98.6GiB checkpoint、128GB UMA 和其余运行时开销共同决定的部署边界。

## PLE CPU offload 的性能局限

PLE CPU offload 让模型能够装入单台 DGX Spark，但会引入新的性能变量。

### NVMe 随机访问延迟

热 page cache 命中时，gather 主要消耗 CPU 内存访问；冷页面则需要等待 NVMe。SSD 的随机读取延迟和队列处理能力比标称顺序带宽更重要。把 packed 文件放到较慢的外接存储，可能显著增加首轮 prefill 或 decode 延迟。

### page cache 容量

更大的 page cache 可以提高重复 n-gram 的命中率，但会与 KV Cache、驱动和主机进程争夺 UMA。项目使用 <code>MADV_RANDOM</code>减少无效页面，却不能保证所有实际读入的页面长期保留。

### token 内容和访问局部性

重复文本、常见短语或具有相似局部模式的请求可能复用页面；变化较大的 token 序列会访问更多分散位置。即使 token 数相同，PLE I/O 也可能不同。

### 并发与批量

prefill 的大批量 lookup 更适合显式预取，但也会增加瞬时工作量。decode 并发增加后，每个 engine step 可以合并更多行，NVMe queue depth 可能改善；与此同时，GPU 必须等待整个批次的 PLE output buffer 准备完成，慢请求会影响这一轮所有序列。

### 同步方式

在支持 stream memory operations 的设备上，异步方案可能更好地隐藏查表延迟。当前 GB10 host handshake 以可用性和正确性优先，性能测试应该把等待时间单独观察，而不能只统计总 tokens/s。

## 实际部署时应观察哪些指标

排查 PLE 问题时，只看 GPU utilization 不够。至少需要同时记录：

- 每个 token 或每个 engine step 的 PLE row 数量；
- mmap major/minor page fault；
- 块设备随机读取量、平均延迟和 queue depth；
- page cache 大小及 <code>MemAvailable</code>变化；
- CPU worker gather 时间；
- packed rows 的 H2D copy 时间；
- GPU worker 等待 <code>done_flag</code>的时间；
- prefill 与 decode 分开统计的端到端延迟；
- 冷启动、冷 page cache 和稳定运行后的差异。

项目 README 中的 1,366KiB/token 与 57KiB/token 对比说明，磁盘读取量可以直接暴露错误的 mmap 访问策略。另一方面，读取量下降后 decode 速度没有明显变化，也提醒我们不要把所有 PLE 优化都解释成吞吐提升。对 UMA 机器来说，减少无效 page cache 本身就是重要结果。

## 小结

Qwen3.8-Flash-Next 的 PLE 是一个位于网络前部的 n-gram 条件记忆模块。模型主干仍然在 hidden states 上逐层计算；token ID 和 positions 作为辅助输入，沿另一条路径生成 n-gram ID 并查询大表。

MiaAI-Lab 项目把这条路径拆成了明确的 CPU/GPU 分工：

- CPU 计算 n-gram ID，从 NVMe-backed mmap table 中 gather packed rows；
- GPU 对查到的 NVFP4 数据反量化，并完成投影、门控、短卷积和残差融合；
- prefill 利用整段 token 已知的条件批量预取；
- decode 受上一轮生成 token 的依赖限制，无法无限提前；
- GB10 上用 host-side sequence-number handshake 代替不受支持的 CUDA stream memory operation。

packed 文件、<code>MADV_RANDOM</code>和显式 <code>WILLNEED</code>预取解决的是三个不同问题：packed 文件减少单行数据量并消除 checkpoint shard 解析；<code>MADV_RANDOM</code>避免随机查询带入无关页面；<code>WILLNEED</code>让已经确定的一批页面并行进入存储队列。三者共同使 26.82GiB PLE 表能够以可回收的 file-backed 形式参与推理。

下一篇将分析部署中的量化路径：NVFP4 code、FP8 block scale 和 global scale 如何表示模型权重，MXFP8 内核为什么会遇到特殊矩阵形状，以及 FP8 KV Cache 如何改变长上下文容量。

## 参考资料

- Qwen Team：[Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency](https://qwen.ai/blog?id=qwen3.8-flash-next)
- Qwen Team：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320)
- MiaAI-Lab：[Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
- Xin Cheng et al.：[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)
- Google AI for Developers：[Gemma 3n model overview](https://ai.google.dev/gemma/docs/gemma-3n)

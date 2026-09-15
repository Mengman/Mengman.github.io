---
title: 【Qwen3.8-Flash-Next 端侧部署】03-NVFP4 权重与 FP8 KV Cache
date: 2026-09-15T17:20:55.464Z
tags:
---

在单台 DGX Spark 的部署中，“量化”至少涉及三类不同数据：模型权重、PLE 查找表和 KV Cache。它们都是 FP4 或 FP8 类型，但量化对象、scale 粒度、读取方式和反量化位置并不相同。

如果只说“模型采用 NVFP4”，很容易产生两个误解。第一，checkpoint 中的每一个张量并不一定都使用同一种格式；第二，权重量化解决的是固定参数的容量问题，不能阻止 KV Cache 随上下文长度和并发数继续增长。

MiaAI-Lab 项目部署的是 [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4) checkpoint。这个仓库是 [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) 的镜像，不是 MiaAI-Lab 独立制作的第三种量化。项目又针对 vLLM 增加了两类适配：少量 MXFP8 线性层在原生内核不支持其矩阵形状时局部回退到 BF16；QSA 注意力内核则增加 FP8 E4M3 KV Cache 的读取和缩放路径。

本文依据 NVIDIA 的 [NVFP4 格式说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)、Qwen Team 的 [Qwen3.8-Flash-Next 架构文档](https://qwen.ai/blog?id=qwen3.8-flash-next)，并结合 local-inference-lab checkpoint 的量化配置及 [Qwen3.8-Flash-Next-Single-DGX-Spark 项目](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)中的 ModelOpt 和 QSA 补丁进行分析。相关格式、模型、权重和部署代码分别来自 NVIDIA、Qwen Team、local-inference-lab 和 MiaAI-Lab，感谢这些团队公开实现和测量数据。

## 先区分几种数值表示

当前部署路径中会遇到下面这些名称：

| 名称 | 主要保存的对象 | 数值和 scale | 运行时用途 |
| --- | --- | --- | --- |
| NVFP4 | 主模型中的量化权重，也用于 packed PLE 表 | E2M1 4-bit value；每 16 个值一个 FP8 E4M3 block scale；另有 FP32 global scale | 降低权重和 PLE 的存储量 |
| MXFP8 | checkpoint 中部分线性层的量化计算路径 | FP8 value；通常每 32 个值共享 E8M0 scale | 由支持的矩阵内核执行量化 GEMM |
| FP8 E4M3 KV Cache | QSA 层的 K/V 历史状态 | FP8 K/V 数据及独立的 K scale、V scale | 降低长上下文缓存容量 |
| BF16 | 激活、部分状态和兼容性回退 | 16-bit 浮点 | GPU 主干计算及少量不兼容线性层 |

NVFP4 中包含 FP8 scale，不代表 NVFP4 权重本身是 FP8；FP8 KV Cache 也不是“用 NVFP4 的 scale 保存 K/V”。名称中虽然都出现 FP8 E4M3，它们属于不同张量的数据格式。

存储格式与计算格式也要分开。权重可以用 4 bit 保存，但进入 Tensor Core 计算时会配合 scale 和相应内核完成解码；KV Cache 可以用 FP8 保存，而 query、tile 转换和点积累加仍然使用 BF16 或 FP32。低精度存储不等于整条计算链都只有相同位宽。

## NVFP4 怎样表示一个权重

NVFP4 使用 E2M1 格式：1 个 sign bit、2 个 exponent bit 和 1 个 mantissa bit。它能表示的非负幅值集合为：

~~~text
0, 0.5, 1, 1.5, 2, 3, 4, 6
~~~

加上 sign bit 后得到正负数值。单个 4-bit code 的范围和精度都很有限，因此不能直接把 BF16 权重逐元素截成 E2M1。NVFP4 使用两级 scale 恢复动态范围：

~~~text
近似权重
  = E2M1 code value
  × FP8 E4M3 block scale
  × FP32 global scale
~~~

每 16 个连续值共享一个 block scale。block scale 使用带有 3 位 mantissa 的 E4M3  FP8 格式保存，而不是只能表示 2 的整数次幂的 E8M0，因此能更细致地贴近该 block 的实际范围。global scale 再让整张量落入 E4M3 scale 与 E2M1 value 的组合可表示范围。

仅看主要存储项，每 16 个权重需要：

~~~text
16 × 4 bit values = 8 byte
1 × FP8 scale     = 1 byte
合计                = 9 byte
~~~

也就是平均每个值 4.5 bit，再加一份相对于大张量很小的 FP32 global scale。与每个值 16 bit 的 BF16 相比，理想存储比例约为：

~~~text
16 / 4.5 ≈ 3.56
~~~

所以 NVFP4 相比 BF16 的实际容量收益接近 3.5 倍，而不是忽略 scale 后得到的整四倍。完整 checkpoint 还包含索引、norm、router、scale 以及其他精度的参数，文件体积不能直接用总参数量乘 4 bit 得到。

## “NVFP4 checkpoint”仍然可能是混合量化

项目使用的 checkpoint 通过配置为不同模块声明不同量化算法。主模型路径中既有 NVFP4 数据，也有 MXFP8 线性层；PLE 则使用上一篇介绍的 NVFP4 packed row。部署代码必须根据模块配置选择正确的 quant method，不能只根据模型目录名全局决定。

这种混合方式有实际原因。不同权重形状和算子对低精度内核的支持程度不同，embedding lookup、普通 linear 和 fused MoE 也不是同一种计算。强行把所有张量塞进一个内核路径，往往会在加载或第一次执行特殊形状时失败。

项目的 [patch_ple_layer.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_ple_layer.py) 就专门读取 <code>ple_embedding_dtype</code>，为 NVFP4 PLE 和 FP8 PLE 选择了不同的实现。本文后面讨论的 [patch_modelopt_mxfp8.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_modelopt_mxfp8.py)，则是处理 MXFP8 linear kernel 的矩阵形状限制。

## local-inference-lab 的量化方案

### 整体量化方案

local-inference-lab 的量化方案是一套的混合量化方案，每个模块采用的不同的量化方案：

| 模块 | 主要格式 | 说明 |
| --- | --- | --- |
| Routed MoE experts | NVFP4 | 4-bit value、每 16 值一个 FP8 block scale |
| GDN 与 QSA 的多类 projection | MXFP8 | 包括大量 linear attention 和 sparse attention 线性权重 |
| Shared experts | MXFP8 | 每 token 都会经过的 dense expert 也被压缩 |
| PLE / n-gram embedding | NVFP4 | `ple_embedding_dtype` 明确设置为 `nvfp4` |
| Norm、router、部分 embedding 与其他张量 | BF16 或原始格式 | 不适合上述量化组的张量没有被强行改成同一种 dtype |

因此，“NVFP4 checkpoint” 在这里并不是“所有参数都以 4 bit 保存”。更准确的描述是：routed experts 和容量最大的 PLE 表采用 NVFP4，大量 dense linear 采用 MXFP8，其余敏感或不适配的张量保留更高精度。这种布局同时压缩稀疏专家、dense side layers 和 PLE，目标明显偏向尽可能降低整个 checkpoint 与运行时权重的容量。

### PLE 使用 NVFP4 是最关键的容量选择

Qwen3.8-Flash-Next 在 125B 主模型参数之外，还有约 51B 参数的 PLE n-gram embedding。PLE 是根据离散 n-gram ID 查找少量 row 的大表，不是对整张表执行 GEMM。local-inference-lab 仍然把这部分设为 NVFP4，并由项目的 PLE 补丁在运行时读取 4-bit code、FP8 block scale 和 FP32 global scale。

对 160 维 PLE row，项目的 packed 格式包含：

~~~text
160 个 NVFP4 value    = 80 byte
10 个 FP8 block scale = 10 byte
合计                  = 90 byte / row
~~~

如果同一行使用纯 FP8 value，则主要数据为 160 byte。NVFP4 packed row 因而比 FP8 row 小约 44%，但不能把它简单说成严格减半，因为每 16 个 value 还要保存一个 block scale。项目生成的完整 packed PLE 表约为 26.8GiB；相较约 51B 参数的 FP8 表，它节省的二十多 GB 正是该 checkpoint 能进入约 100GB 级别的主要原因。

这个变化对单台 DGX Spark 尤其重要。128GB UMA 不只要容纳固定权重，还要给 CUDA、vLLM host process、激活、MTP、recurrent state、KV Cache 和 PLE page cache 留空间。即使部署时把 PLE mmap 到 NVMe，较小的 packed row 仍然可以减少磁盘文件、随机读取量、page cache 压力和 CPU 到 GPU 的复制字节数。

### 激进量化换来了什么，又承担什么风险

local-inference-lab 方案的主要收益是容量和数据移动，而不是让 PLE 直接获得 NVFP4 Tensor Core GEMM 加速。Routed expert 是矩阵乘法，可以使用针对 W4A4 NVFP4 优化的 kernel；PLE 是 embedding lookup，量化后仍要查找 codes 与 scales，再恢复成后续计算需要的数值。

把 PLE 从 FP8 进一步降到 NVFP4，**会扩大单个 embedding value 的舍入误差**。PLE 查找结果会进入模型残差路径，这种误差是否影响长文本、代码、工具调用或多模态任务，不能只由文件大小推断。当前项目已有 reasoning、needle 和多模态运行验证，但没有与 NVIDIA checkpoint 使用同一评测集、同一推理配置进行直接质量 A/B 测试比较。因此可以确认它更激进、更节省容量，但是相比 Nvidia 的 NVFP4 量化方案相比有多少性能损失，无法直接得出结论。

## NVIDIA 的 NVFP4 量化方案

NVIDIA 后来发布的 [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) 同样使用 ModelOpt，但它对不同模块采取了更保守的精度边界。官方模型卡给出的布局是：

| 模块 | NVIDIA checkpoint 的格式 |
| --- | --- |
| 主模型 routed MoE expert linear | W4A4 NVFP4，weight scale 使用 MSE calibration |
| Attention、shared experts 和其他主模型层 | BF16 |
| MTP routed experts | 128×128 block-scaled FP8 |
| PLE n-gram embedding | per-tensor FP8 |

NVIDIA 只把主模型中最适合低精度 GEMM 的 routed experts 量化为 NVFP4。PLE 保持 FP8，attention、shared experts 和其他主模型层保持 BF16；官方还说明 MTP 与 PLE 张量来自 Qwen 的 FP8 checkpoint。最终 checkpoint 约为 BF16 源模型的 1/2.7，也就是磁盘体积减少约 63%。

这个方案的目标不是让文件尽可能小，而是在 Blackwell、ModelOpt 与 vLLM 的官方部署路径上控制量化误差。NVIDIA 公布了 NVFP4 checkpoint 相对 FP8 baseline 的多项任务结果，两者在不同项目上各有小幅高低，至少说明官方方案经过了一组覆盖推理、代码、工具使用、长上下文和多模态的质量评测。不过，这组数据没有包含 local-inference-lab checkpoint，不能充当两个社区方案的直接 A/B。

官方列出的目标硬件是 B200 和 B300，并给出 TP=8 的 vLLM 示例。其 checkpoint 约 133GB，已经超过 DGX Spark 的 128GB 统一内存总量，还没有计入运行时和 KV Cache。因此若要在单台 Spark 上使用，仍需把大约 51GB 级别的 FP8 PLE 从常驻权重中卸载，或者采用其他 CPU/NVMe offload；不能把面向多卡服务器的官方启动命令原样移植到 TP=1。

## 两种量化方案的取舍

两者都叫 Qwen3.8-Flash-Next NVFP4，但 NVFP4 的覆盖边界不同：

| 比较项 | local-inference-lab 方案 | NVIDIA 方案 |
| --- | --- | --- |
| Routed experts | NVFP4 | NVFP4 |
| PLE | NVFP4 | FP8 |
| Attention projection | 大量采用 MXFP8 | BF16 |
| Shared experts | MXFP8 | BF16 |
| MTP | checkpoint 内混合量化 | Routed experts 为 FP8，其余保持相应 FP8/BF16 来源格式 |
| Hugging Face 显示体积 | 约 106GB；Mia 镜像属于同一权重来源 | 约 133GB |
| 主要取向 | 容量、单机可部署性和较少的数据移动 | 精度保留、官方 kernel/runtime 兼容与质量验证 |

local-inference-lab 是更激进的量化方案。它不只压缩 routed experts，还把 dense projection、shared experts 和约 51B 参数的 PLE 纳入低精度范围。这样可以少占二十多 GB，并让 128GB UMA 上的单机部署获得决定性的内存余量；代价是更多模块进入较低精度路径，也需要项目补丁处理 MXFP8 shape、NVFP4 PLE 反量化和卸载通信。

NVIDIA 方案对 PLE 保留 FP8，对 attention、shared experts 等模块保留 BF16，因而理论上具有更小的量化误差和更好的模型精度保留。它还拥有官方 ModelOpt/vLLM 部署说明及一组相对 FP8 baseline 的评测证据。这里的“性能更好”首先应理解为模型任务效果、数值保真度和官方运行时兼容性更有保障，而不是已经证明每秒 token 数更高。

推理吞吐需要单独判断。在 B200/B300 的多卡部署中，NVIDIA 方案更贴近官方优化路径；在 DGX Spark 的 PLE mmap 路径上，FP8 PLE row 比 NVFP4 packed row 更宽，可能增加 NVMe、page cache 和 H2D(Host to Device) 流量，而更多 BF16 dense 权重也会增加 decode 的持续带宽。因此，哪一个 checkpoint 的 prefill 或 decode 更快，必须在相同 vLLM commit、offload 实现、MTP、KV dtype、prompt、并发和缓存冷热条件下实测，不能仅由量化位宽或官方身份推出。

可以将选择原则概括为：

- local-inference-lab 方案是“容量优先”：更适合把 Qwen3.8-Flash-Next 压入一台 128GB DGX Spark；
- NVIDIA 方案是“精度与官方兼容优先”：更适合具有充足显存的 B200/B300 多卡环境，并预期获得更稳健的质量保留；

## MXFP8 线性层的形状限制

MXFP8 与 NVFP4 不是同一个格式。NVIDIA Transformer Engine 的[低精度格式说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)中，MXFP8 使用 FP8 value，并为每 32 个连续值保存一个 E8M0 scale。E8M0 scale 是 2 的幂，硬件处理简单，但尺度选择比 E4M3 粗。

当前 vLLM 镜像使用 FlashInfer CUTLASS MXFP8 linear kernel。项目在 GB10 上测得，该内核要求实际参与计算的权重矩阵 <code>[N, K]</code>满足：

~~~text
N >= 128
N % 32 == 0
K >= 128
K % 32 == 0
~~~

这里的 N 和 K 是张量并行切分后的 per-partition shape，不一定等于 checkpoint 中看到的全局矩阵形状。token 数对应的 M 维不影响上述限制。

Qwen3.8-Flash-Next 中至少有两类形状违反条件：

| 模块 | 权重形状 | 不满足的条件 |
| --- | ---: | --- |
| 线性注意力的 <code>in_proj_a/b</code> | <code>[48, 2560]</code> | N 小于 128 |
| 视觉 MLP 的 <code>linear_fc1</code> | <code>[4304, 1152]</code> | N 除以 32 余 16 |

如果仍然调用原生内核，第一类会在 engine startup 报出 N 太小，第二类则会在视觉 profile 或多模态请求中报告矩阵尺寸不受支持。

## 局部 BF16 回退

最直接的兼容方案是把 MXFP8 权重反量化成 BF16，再调用普通 linear。问题在于，如果对整个模型这样做，数十 GiB 量化权重会大幅膨胀，单台 DGX Spark 的内存预算立即失效。

项目的做法是按层检查实际 <code>[N, K]</code>。满足条件的层继续使用 FlashInfer MXFP8 kernel；不满足条件的层改用 <code>EmulationMxfp8LinearKernel</code>，在加载时把该层权重展开为 BF16，之后作为普通 BF16 linear 运行。

[`files/patch_modelopt_mxfp8.py`](./files/patch_modelopt_mxfp8.py) 对应的原始判断如下：

```python
def _mxfp8_native_kernel_supports(n: int, k: int) -> bool:
    return n >= 128 and n % 32 == 0 \
        and k >= 128 and k % MXFP8_BLOCK_SIZE == 0

if not _mxfp8_native_kernel_supports(
    output_size_per_partition, input_size_per_partition
):
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    if not isinstance(self.kernel, EmulationMxfp8LinearKernel):
        self.kernel = _mxfp8_emulation_kernel()
```

检查发生在 `create_weights` 中，使用的是 TP 切分后的 `output_size_per_partition` 和 `input_size_per_partition`。这就是为什么文章需要强调 per-partition shape，而不能只查 checkpoint 的全局 shape。

~~~text
读取某个 MXFP8 linear 的 per-partition shape
                   │
          ┌────────┴────────┐
          │                 │
      shape 支持         shape 不支持
          │                 │
          ▼                 ▼
  FlashInfer MXFP8     BF16 emulation
        kernel          仅该层展开
~~~

每一个 linear layer 有自己的 quant method 实例，因此更换 kernel 只影响当前层。项目估算，36 个线性注意力层中的 <code>48 × 2560</code>投影全部回退，额外 BF16 权重约为 17MB，和全模型回退相比很小。

视觉模块还保留了额外的 prefix rule，使 <code>visual.*</code>优先采用已经验证过的 emulation 路径。这不是说所有视觉计算都必须使用 BF16，而是当前 checkpoint 和内核组合中，局部展开视觉相关的不兼容 MXFP8 权重比等待运行时失败更稳妥。

这种补丁的意义不是提高量化率，而是控制回退范围。对于内存极限部署，“遇到不支持就全局解量化”通常不可接受；真正需要的是按 tensor 或按 layer 建立兼容性边界。

## KV Cache 的量化方案

权重加载完成后，其容量基本固定。KV Cache 则随请求增长。

在自回归注意力中，每个新 token 会产生 key 和 value。后续 token 需要读取历史 K/V，因此服务必须为已经处理的 token 保留这些状态。对一个请求，主 KV Cache 大致与上下文长度成正比；多个并发请求再把各自 token 数相加。

Qwen3.8-Flash-Next 使用 36 个 GDN 层和 12 个全局注意力/QSA 层。GDN 把历史压缩为固定大小状态，不需要为每个历史 token 保存标准 K/V；12 个 QSA 层仍然需要 paged KV Cache。此外，QSA indexer、compressor 或 GDN 还有其他 side state，不能只计算主 K/V。

### BF16 格式内存占用

项目中的 QSA 配置有 12 个全局注意力层、2 个 KV head、每个 head 256 维。K 和 V 各保存一份。BF16 每个值 2 字节，所以主 KV Cache 的每 token 占用是：

~~~text
12 layers
× 2 KV heads
× 256 dims
× 2 (K + V)
× 2 bytes
= 24,576 bytes/token
~~~

当前启动脚本使用的全架构 BF16 cache 测量值约为 29,482 bytes/token。主 K/V 占其中约 83%，剩余部分来自仍按 BF16 保存的 QSA side/compressor cache 和其他状态。

这解释了两个现象：

1. 把主 K/V 从 BF16 改成 FP8 会带来很大收益，因为它占总缓存的大部分。
2. 总缓存容量不会严格翻倍，因为剩余状态没有随主 K/V 一起减半。

### FP8 后的预算估算

主 K/V 使用 FP8 后，每 token 从 24,576 字节降到约 12,288 字节。若其余约 4,906 字节保持不变，总量约为：

~~~text
12,288 + 4,906 ≈ 17,194 bytes/token
~~~

相对于 29,482 bytes/token，比例约为 0.58。项目启动脚本正是使用 <code>KV_MULT=0.58</code>估算 FP8 cache 需求。按这个保守预算，token 容量约提高 1.7 倍；项目在不同版本和配置上的实际测量约为 1.7 至 1.9 倍。

测量值会受 vLLM block 分配、graph profile、cache metadata 和其他状态影响，因此不应把某一次“约 1.85 倍”当作数值格式的固定常数。

[`start.sh`](./start.sh) 中的预算常量直接保留了这一实现假设：

```bash
KV_BYTES_PER_TOKEN=29482
KV_MULT=1.0
# FP8 halves the main KV, but QSA side/compressor caches stay BF16.
[[ "$KV_CACHE_DTYPE" == fp8* ]] && KV_MULT=0.58
```

因此 `0.58` 是这套模型与 cache 布局的工程测量值，不是 FP8 数据类型本身的通用比例。

## vLLM 原路径为什么不能直接使用 FP8 KV

vLLM 已经能够按 FP8 cache dtype 分配量化 KV storage，但当时 Qwen3.8-Flash-Next 专用 QSA backend 只声明支持 <code>auto</code>和 <code>bfloat16</code>。通用 cache 配置支持 FP8，不代表模型专用注意力 kernel 已经知道怎样读取它。

在量化 cache 中，vLLM 可能把底层 allocation 表示为 <code>uint8</code>。如果 QSA kernel 把这些字节当作 BF16 或普通整数读取，结果不会只是精度下降，而是数值语义完全错误。项目补丁因此同时修改 dtype capability、wrapper 检查和 Triton kernel。

[patch_qsa_fp8_kv.py](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/files/patch_qsa_fp8_kv.py) 完成了几件事：

- QSA backend 声明支持 <code>fp8</code>和 <code>fp8_e4m3</code>；
- <code>uint8</code> cache 在不复制数据的情况下 reinterpret 为 <code>float8_e4m3fn</code>；
- 如果 cache 是量化字节但 quant mode 没有启用，立即报错，避免返回看似正常的错误输出；
- QSA block-selection indexer 接收 K scale；
- sparse attention kernel 同时接收 K scale 和 V scale；
- FP8 分支由 compile-time constant 控制，BF16 配置不会保留多余的运行时分支。

## FP8 KV 在 QSA 中怎样参与计算

QSA 有两个阶段会读取历史 key 。

第一阶段是 compressed lightweight indexer。它根据 query 与压缩后的历史 key 计算 block importance，再选出需要进入核心 sparse attention 的上下文块。K 的量化误差可能改变 block 排名，因此 FP8 key 不只影响点积数值，也可能影响“模型选中了哪些历史位置”。

第二阶段是对选中位置执行 sparse attention，读取完整 K/V tile，计算 attention score 和加权 value。

补丁没有先构造完整 FP32 dequantized K/V tile，而是利用 scale 可以移到点积之外这一关系。对于一个共享标量 K scale：

~~~text
dot(Q, K_fp8 × K_scale)
= dot(Q, K_fp8) × K_scale
~~~

对于 V scale：

~~~text
sum(P × (V_fp8 × V_scale))
= sum(P × V_fp8) × V_scale
~~~

kernel 先把当前 FP8 tile 转成 query 使用的 BF16 dtype，让 Tensor Core 完成点积；score 以 FP32 输出并乘 K scale，归一化后的 value 聚合结果再乘 V scale。这样既保留正确的缩放关系，也避免为每个 tile 额外保存一份 FP32 K/V。

## 权重量化和 KV 量化的对比

两类量化最终作用在推理不同的阶段上：

~~~text
服务基础占用
  = NVFP4 / mixed-precision 主模型权重
  + vLLM/CUDA 固定开销
  + MTP 权重

请求相关占用
  = FP8 或 BF16 KV Cache
  + GDN/QSA 其他状态
  + 临时激活和工作区
~~~

主模型 NVFP4 决定服务能否完成加载；FP8 KV 决定加载之后还能留下多少上下文和并发容量。

当前已知配置大致为：

| 项目 | 当前示例配置 |
| --- | --- |
| checkpoint | <code>Mia-AiLab/Qwen3.8-Flash-Next-NVFP4</code> |
| GPU 侧非 PLE 权重 | 约 71.75GiB |
| KV Cache dtype | FP8 |
| BF16 cache 预算基准 | 约 29,482 bytes/token |
| FP8 cache 预算系数 | 0.58 |
| KV Cache 目标 | 16GiB |
| 原生 context | 262,144 token |

这些数字属于当前项目和 checkpoint，不是 NVFP4 或 FP8 格式本身的通用规格。

## 小结

单台 DGX Spark 上的量化方案是一个综合的量化方案，针对不同数据对象分别处理：

- NVFP4 通过 4-bit E2M1 value、FP8 E4M3 block scale 和 FP32 global scale 压缩大部分权重；
- local-inference-lab 进一步把约 51B 参数的 PLE 量化为 NVFP4，并对大量 dense linear 使用 MXFP8，是偏向容量和单机部署的激进方案；
- NVIDIA 只对主模型 routed experts 使用 NVFP4，让 PLE、MTP 和其他模块保留 FP8 或 BF16，是偏向精度保留与官方兼容的保守方案；
- checkpoint 中部分线性层走 MXFP8 路径，特殊 N/K 形状只在对应层回退为 BF16；
- QSA 主 KV Cache 使用 FP8 E4M3 保存，读取时转换为 BF16 tile，并在点积和输出位置应用 K/V scale；
- GDN 和 QSA 的其他状态没有全部减半，所以总 KV 容量增益低于两倍；
- FP8 key 可能改变 QSA 的 block 选择，必须把长上下文质量作为部署验证的一部分。

下一篇将讨论这些量化权重和缓存进入 vLLM 之后，MTP、CUDA Graph、动态批处理和 Chunked Prefill 如何组织实际推理。

## 参考资料

- NVIDIA：[NVFP4 — Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- NVIDIA：[Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- NVIDIA：[Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- NVIDIA：[Qwen3.8-Flash-Next-NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- Qwen Team：[Qwen3.8-Flash-Next Architecture](https://qwen.ai/blog?id=qwen3.8-flash-next)
- local-inference-lab：[Qwen3.8-Flash-Next-NVFP4 checkpoint](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
- local-inference-lab：[Qwen3.8-Flash-Next-NVFP4 config.json](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/blob/main/config.json)
- Mia-AiLab：[Qwen3.8-Flash-Next-NVFP4 mirror](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
- MiaAI-Lab：[Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
- vLLM：[FP8 KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html)

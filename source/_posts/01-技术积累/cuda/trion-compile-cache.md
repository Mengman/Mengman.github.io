---
title: 从 CUDA 到 Triton：深入理解 Triton 的即时编译与自动调优
date: 2026-08-06 18:30:59
tags: [machinelearning, deeplearning, CUDA]
categories: CUDA
typora-root-url: ../../../
---

flex-gemm 是一个使用 triton 开发的稀疏卷积库，在运行的过程中因为要根据输入数据进行算子调优，所以在计算时间上有很大的波动。在解决 flex-gemm triton 算子编译优化的过程中，系统性的学习了一下 triton 的编译流程。 这里写一篇文章做个记录。
## 一、CUDA 编程核心

### 1.1 线程架构

NVIDIA GPU 的设计哲学是“用海量线程隐藏内存延迟”。为了高效管理这些线程，CUDA 定义了一套严格的层次结构：
- **线程（Thread）**：最基本的执行单元，运行在 CUDA 核上。
- **线程块（Thread Block）**：一组线程的集合，同一个 Block 内的线程可以通过共享内存（Shared Memory）高效通信，也能做同步（`__syncthreads()`）。一个 Block 会被调度到一个流多处理器（SM）上执行。
- **线程网格（Grid）**：所有 Block 的集合，代表一次完整的 Kernel 启动。

理解这套层次，就理解了 GPU 编程的第一个关键：你需要把计算任务分解成“可以独立并行的块”，再把每个块分解成“可以协同工作的线程”。这个分解方案会直接影响性能。

![CUDA线程层级架构](/image/trion-compile-cache/cuda_thread_hierarchy_01.png)

### 1.2 CUDA 程序调优

为了能写出高性能的 CUDA 程序，开发者需要根据**数据特性和硬件架构**手工调整一系列参数和实现细节。
以矩阵乘法 `C = A × B` 为例，我们看看有多少东西需要亲手“捏”：

#### Block 大小与 Grid 大小
Block 包含多少线程？是 `(16, 16)` 还是 `(128, 1)`？这决定了每个 Block 的计算量和资源占用。Grid 又该由多少个 Block 组成才能完整覆盖输出矩阵？这里需要考虑矩阵维度是否正好被 Block 大小整除，边界如何处理。

#### 内存层次的使用
GPU 有全局内存、L2 缓存、共享内存和寄存器，带宽和容量差异巨大。一个标准优化是使用**共享内存分块（Tiling）**：把矩阵切分成小块，先将子块从全局内存协作加载到共享内存，再让 Block 内线程从共享内存读取做乘法累加。这需要你手动规划“块”的大小、分配共享内存、插入 `__syncthreads()` 同步屏障。

#### 内存合并访问
同一个 Warp（32 个线程）访问全局内存时，如果地址是连续的、对齐的，就能在一次内存事务中完成——这叫合并访问。否则性能会断崖式下降。你需要精心设计线程到数据的映射关系。

#### 占用率与寄存器压力
一个 SM 上同时驻留的 Warp 越多，越能隐藏内存延迟。但每个线程用太多寄存器或共享内存，就会限制同时驻留的 Block 数量。因此需要在“用寄存器缓存数据”和“让更多 Warp 干活”之间找平衡。
每一个选择背后都有复杂的权衡，而且这种权衡**随 GPU 型号变化**：A100 的共享内存大小和 SM 数量与 V100 不同，最优 Block 大小也就不一样。这导致 CUDA 代码往往需要为不同 GPU 写不同版本的 Kernel，或手工编写复杂的参数搜索逻辑。



## 二、triton 如何简化这一切

Triton 的核心思想是：**让开发者以 Block 为单位编写计算逻辑，而不是管理每个线程**。你只需要描述“一个 Block 要做什么”，而 Block 内部的线程化细节、共享内存分配、同步、内存合并等全部交给编译器。

同样以矩阵乘法为例，Triton 代码大概长这样：

```python
 @triton.jit  
 def matmul_kernel(  
     A_ptr, B_ptr, C_ptr,  
     M, N, K,  
     BLOCK_M: tl.constexpr,   
     BLOCK_N: tl.constexpr,   
     BLOCK_K: tl.constexpr,  
 ):  
     pid_m = tl.program_id(0)  
     pid_n = tl.program_id(1)# 计算本 block 对应 A、B 的 tile 指针  
     a_tile = tl.load(A_ptr + offsets_a)  
     b_tile = tl.load(B_ptr + offsets_b)# 在寄存器中累加  
     acc = tl.dot(a_tile, b_tile)  
     tl.store(C_ptr + offsets_c, acc)

```

你不再需要：
- 写 `threadIdx.x`、`blockIdx.x` 来手动算下标
- 手动申请共享内存并处理 bank conflict
- 插入 `__syncthreads()`
- 担心内存合并——Triton 编译器会自动生成合并访问模式
- 担心 Warp 发散

Triton 把“写单线程”提升到了“写 Block 级程序”，极大降低了 GPU 编程的心理负担。



## 四、Auto-Tune：让编译器帮你搜索最优参数

虽然开发者不需要管线程，但 **Block 的大小（即** `BLOCK_M`**、**`BLOCK_N`**、**`BLOCK_K`**）** 仍然影响性能。这个参数该设为 64 还是 128？不同的矩阵形状、不同 GPU 都有不同答案。
Triton 提供了 `@triton.autotune` 装饰器来自动解决这个问题：
```python
 @triton.autotune(  
     configs=[  
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),  
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),  
         # 更多配置...  
     ],  
     key=['M', 'N', 'K'],  
 )  
 @triton.jit  
 def matmul_kernel(...):  
     ...

```

**Auto-Tune 原理**：

1. 当你第一次用某组 `(M, N, K)` 调用函数时，Triton 会遍历所有提供的 `configs`。
2. 对每个 config，生成一个带有具体 `BLOCK_M` 等值的 Kernel 变体，并在真实 GPU 上运行一次 Benchmark。
3. 记录每个 config 的执行时间，选择最快的那个。
4. 将最优 config 与该 `key` 绑定：下次再遇到相同形状的矩阵乘法，直接用最优版本，不再重复搜索。

这个过程相当于**自动完成了 CUDA 专家需要手工完成的参数扫描**。而且由于搜索发生在运行时，它可以针对你手头这块 GPU 精准优化——代码是“可移植的性能”。



## 五、Triton 的 JIT 编译流程

上面多次提到“生成 Kernel 变体”。那么 Python 代码是怎么变成 GPU 能执行的机器码的？这就轮到 Triton 的 JIT（即时编译）系统登场。
Triton 编译流程可以概括为 **Python → Triton IR → Triton GPU IR → LLVM IR / PTX → cubin**：
1. **前端解析**：`@triton.jit` 装饰的函数被第一次调用时，Triton 会解析 Python 抽象语法树（AST），将其转换为 **Triton 中间表示（Triton IR）**。Triton IR 是一种面向块级别计算的 SSA 表示，保留了高层语义（如 `tl.dot` 直接表示矩阵乘）， 文件后缀名为 "ttir"。
2. **Triton GPU IR**：Triton IR 经过一系列优化和降级，变成 Triton GPU IR。这个阶段会进行 Block 级别的优化，例如内存合并、共享内存分配、同步插入等。此刻代码还是相对高层的。文件后缀名为 “ttigr”。
3. **生成 LLVM IR / PTX**：Triton GPU IR 进一步转换为 LLVM IR，文件后缀名称 "llir"。然后利用 NVIDIA 的 PTX 后端生成 PTX 代码（一种接近机器码的虚拟 ISA）。PTX 是 NVIDIA GPU 的“汇编语言”，但仍是文本格式，尚未绑定到具体 GPU 架构。
4. **生成 cubin**：PTX 被 NVIDIA 的 `ptxas` 编译器编译成 **cubin**（CUDA Binary），这是最终加载到 GPU 上执行的二进制文件。cubin 与特定 GPU 的计算能力（Compute Capability）绑定。CUDA 程序在新的环境也需要 warmup，这个过程就是将 PTX 代码在特定的硬件上编译成 cubin。



## 六、编译缓存：让速度起飞的关键

如果每次 Kernel 调用都要完整走一遍上述编译流程，Triton 的启动延迟将无法忍受——尤其是 Auto-Tune 阶段，可能要对几十套配置分别编译，这将极其缓慢。
Triton 的解决方案是**基于文件系统的编译缓存**。

### 6.1 缓存键的构成
每当 Triton 需要编译一个 Kernel，它会基于以下内容生成一个哈希值作为缓存键：
- Triton 编译器版本等
- 源代码
- GPU 架构标识（计算能力版本）
- GPU 编译器参数
- 相关的环境变量

``` python
 # triton.runtime.cache.py  
 def **get_cache_key**(src, backend, backend_options, env_vars):  
     key = f"{triton_key()}-{src.hash()}-{backend.hash()}-{backend_options.hash()}-{str(**sorted**(env_vars.items()))}"  
     return key
```

只要这些输入不变，哈希就不变，意味着编译产物可以安全复用。



### 6.2 缓存文件位置

**编译缓存**

缓存默认存储在 `~/.triton/cache/` 目录下。在该目录中你可以看到一系列以哈希命名的文件夹，每个文件夹内包含：
- `kernel.cubin`：编译好的 GPU 二进制
- `kernel.ttir` / `kernel.ttgir`：中间表示文件（可选，用于调试）
- `metadata.json` 或类似文件，记录编译参数

下面就是 flex_gemm 一个缓存目录下的实际文件。

![flex-gemm 缓存文件夹内容](/image/trion-compile-cache/flex_gemm_triton_cache_files.jpg)

**Autocache 配置环境**

flex_gemm 项目对于 triton autotune cache 做了自定义的优化。

每个算子根据输入参数的不同对应一条最优的编译参数

```json
 {  
     "NVIDIA GeForce RTX 4090 D": {  
         "flex_gemm.kernels.triton.spconv.sparse_submanifold_conv_fwd_masked_implicit_gemm.sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel": {  
             "(14, 32, 32, 27, True, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.uint32', 'torch.int64', 'torch.float32')": {  
                 "kwargs": {  
                     "B1": 128,  
                     "B2": 32,  
                     "BK": 32  
                 },  
                 "num_warps": 4,  
                 "num_ctas": 1,  
                 "num_stages": 4,  
                 "num_buffers_warp_spec": 0,  
                 "num_consumer_groups": 0,  
                 "reg_dec_producer": 0,  
                 "reg_inc_consumer": 0,  
                 "maxnreg": null,  
                 "pre_hook": null  
             },  
             "(14, 32, 8, 27, True, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.uint32', 'torch.int64', 'torch.float32')": {  
                 "kwargs": {  
                     "B1": 128,  
                     "B2": 32,  
                     "BK": 32  
                 },  
                 "num_warps": 4,  
                 "num_ctas": 1,  
                 "num_stages": 4,  
                 "num_buffers_warp_spec": 0,  
                 "num_consumer_groups": 0,  
                 "reg_dec_producer": 0,  
                 "reg_inc_consumer": 0,  
                 "maxnreg": null,  
                 "pre_hook": null  
             }  
         }  
     },  
     "NVIDIA A100-SXM4-40GB": {},  
     "NVIDIA A100 80GB PCIe": {},  
     "AMD Instinct MI300X VF": {}  
 }
```

```python
 @triton_autotune(  
     configs=config.autotune_config,  
     key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32'],  
 )  
 @triton.heuristics(heuristics)  
 @triton.jit  
 def sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel(  
     input,  
     weight,  
     bias,  
     neighbor,  
     sorted_idx,  
     output,  
     # Tensor dimensions
     N, LOGN, Ci, Co, V: tl.constexpr,  
     # Meta-parameters
     B1: tl.constexpr,   # Block size for N dimension
     B2: tl.constexpr,   # Block size for Co dimension
     BK: tl.constexpr,   # Block size for K dimension (V * Ci)
     allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
     # Huristic parameters 
     valid_kernel,  
     valid_kernel_seg,  
 ):
```

`key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32']` 分别对应 autotune 配置的前五个参数 `14, 32, 8, 27, True` 。 后续的类型是 kernel 接受的参数的类型。



### 缓存加载流程

当你调用一个 Triton Kernel 时：
1. **计算哈希** → 去缓存目录查找对应文件夹。
2. **命中**：直接加载 `kernel.cubin`，跳过所有编译步骤，内核瞬间启动。这也是 Auto-Tune benchmark 阶段能快速跑完多个 config 的原因——缓存在，就只有“加载 + 执行”的开销。
3. **未命中**：触发完整的 JIT 编译流程，生成 cubin 后**自动存入缓存目录**，下次同样输入即可命中。

对于 `autotune` 装饰的 Kernel，搜索最优 config 的过程会产生一大批缓存条目——每个 config 对应一个不同的哈希。一旦某个 `(M, N, K)` 组合的最优 config 被确定，以后对该组合的调用只会加载那一个最优 cubin，不会再尝试其他 config，进一步减少开销。

**flex_gemm autotune 缓存**
flex_gemm 自己扩展了 triton 的 autotune 缓存机制，在 `import flex_gemm`的时候，会自动加载 autotune 缓存配置文件 "autotune_cache.json"。

**默认位置**
triton_cache 默认位置是 "~/.triton/cache" 可以通过环境变量 "TRITON_CACHE_DIR" 来修改
flex_gemm autotune 缓存的默认位置 "~/.flex_gemm/autotune_cache.json", 通过环境变量 “AUTOTUNE_CACHE_PATH” 来修改。



## 七、 工程实践：如何部署 Triton Cache？

1. **首次启动预热**：准备一套常用的数据，在容器启动的时候跑一遍所有的 kernel 进行编译缓存
    1. 优点：简单，不限制硬件和软件环境。
    2. 缺点：耗时，准备数据困难，数据不易覆盖所有场景
2. **打包 Cache**：将 cache 打包的 wheel 或者 docker 中
    1. 优点：不需要预热，使用过程无感
    2. 缺点：只能针对确定的环境



在实际的部署过程中，可以解析 autotune_cache.json 文件中每个算子的每条最优配置；然后用这些配置参数加上源代码就可以解析出缓存目录的 base64 名称，这样只需要将使用到的编译结果打包到 wheel 中就可以了，进一步优化 wheel 包的体积

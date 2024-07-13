---
layout: post
title: CUDA 代码编译流程
date: 2024-01-04 00:00:00 +0800
categories: machinelearning deeplearning softwareenginer CUDA
typora-root-url: ..
---

# 1 编译流程

随着 Nvidia 不断更新 GPU 硬件设计架构，每一代架构更迭都会引入新的功能，在同一代架构中不同的 GPU 型号在硬件配置上与性能上也会有一些细小的差异。

Nvidia 提供了一套代号来标识 GPU 的**真实架构**，它的格式是 `sm_xy`，`x` 代表 GPU 架构世代的编号，`y`代表 GPU 在这个架构世代中的版本号。为了方便对比 GPU 能力，如果 `x1y1 <= x2y2` 那么`sm_x1y1`中所有非 ISA 相关的功能都包含在 `sm_x2y2` 中了。

然而这并不能保证新的 GPU 架构就能兼容老的 GPU 架构的二进制代码，比如在 Fermi 架构上编译的代码就无法在Kepler GPU 上运行，因为它们的指令集与指令编码都是不一样的。只用在相同的架构下才能保持二进制兼容。为了解决这个问题，CUDA 代码采用了两阶段编译过程。

**第一个阶段**

nvcc 编译器将 CUDA 代码编译成 PTX 代码，这是一种类似汇编的代码。它基于**虚拟架构**进行编译，虚拟架构是根据 GPU 能力、功能来定义的。

**第二个阶段**

nvcc 编译器是包含在 GPU 驱动中，编译器可以预先或者在运行的时候根据**真实架构**将 PTX 代码编译成实际的二进制代码。

第一个阶段的**虚拟架构**要兼容第二个阶段的**真实架构**，**虚拟架构**的版本越低就越能兼容更多的**真实架构**，当然CUDA 代码支持的功能也就越少。

![Two-Staged Compilation with Virtual and Real Architectures](/assets/virtual-architectures.png)

对于在 CPU 上运行的 host code 与 GPU 上运行的 device code，编译器会分开进行编译。

1. nvcc 会将 device code (kernel 函数) 编译成 PTX code 或者二进制格式 (cubin object) 。
2. nvcc 会修改 host code 中调用 kernel 的部分，修改为 PTX code 或者 cubin object 中实际的代码。



**架构代号列表**

| 虚拟架构代号 | 真实架构代号 | CUDA支持版本 | 支持架构  | 支持硬件                    |
| ------------ | ------------ | ------------ | --------- | --------------------------- |
| compute_50   | sm_50        | CUDA 6~11    | Maxwell   | Tesla/Quadro M series       |
| compute_52   | sm_52        | CUDA 6~11    | Maxwell   | GTX-980, GTX Titan X        |
| compute_53   | sm_53        | CUDA 6~11    | Maxwell   | Tegra TX1, Jetson Nano      |
| compute_60   | sm_60        | CUDA 8       | Pascal    | Tesla P100                  |
| compute_61   | sm_61        | CUDA 8       | Pascal    | GTX 1080, GTX1070           |
| compute_62   | sm_62        | CUDA 8       | Pascal    | Jetson TX2                  |
| compute_70   | sm_70        | CUDA 9       | Volta     | Tesla V100                  |
| compute_72   | sm_72        | CUDA 9       | Volta     | Jetson AGX Xavier           |
| compute_75   | sm_75        | CUDA 10      | Turing    | RTX 2080, RTX 2070 Tesla T4 |
| compute_80   | sm_80        | CUDA 11.1    | Ampere    | A100                        |
| compute_86   | sm_86        | CUDA 11.1    | Ampere    | RTX 3090                    |
| compute_87   | sm_87        | CUDA 11.1    | Ampere    | Jetson AGX Orin             |
| compute_89   | sm_89        | CUDA 11.8    | Lovelace  | RTX 4090                    |
| compute_90   | sm_90        | CUDA 12      | Hopper    | H100 H200                   |
| compute_95   | sm_95        | CUDA 12      | Blackwell | B100                        |



# 2 PTX

上一节提到的 **PTX** (Parallel Thread Execution) 是 Nvidia 为 CUDA 设计的一种低级虚拟机和指令架构，它是CUDA 代码的中间表示形式，有以下特征：

1. 由虚拟指令集定义：用一套虚拟的并行指令集，用于表示 CUDA 设备功能。
2. 目标独立：采用抽象的寄存器和线程模型，不依赖于具体的 GPU 架构，可以针对不同的 GPU 生成优化机器代码。
3. 可移植： PTX 代码可以在不同的 CUDA 运行环境和 GPU设备上执行。
4. JIT 编译： PTX 代码在执行前，有 GPU 驱动编译生成针对特定 GPU 的机器代码。
5. 可读性好：PTX 代码结构类似汇编，以文本形式保存，便于阅读和调试。
6. 虚拟寻址：提供了统一的虚拟寄存器和寻址空间



# 3 JIT 编译

PTX 代码在目标机器上执行前，会由 GPU 驱动将 PTX 代码优化、编译为机器代码。PTX 代码的编译结果会被缓存下来，以免重复编译。但是如果更新了 GPU 驱动，那么 PTX 代码会重新编译。

**在不同操作系统下 JIT 缓存目录**

| 操作系统 | 缓存目录                                                |
| -------- | ------------------------------------------------------- |
| Linux    | `~/.nv/ComputeCache`                                    |
| Windows  | `%APPDATA\%NVIDIA\ComputeCache`                         |
| MacOS    | `$HOME/Library/Application Support/NVIDIA/ComputeCache` |

**与 JIT 相关的环境变量配置**

| 环境变量              | 默认值 | 说明                                                         |
| --------------------- | ------ | ------------------------------------------------------------ |
| `CUDA_CACHE_DISABLE`  | 0      | 是否禁用 JIT cache，`1` 代表禁用 `0` 代表不禁用              |
| `CUDA_CACHE_MAXSIZE`  | 256MB  | cache 的尺寸，最大值 4GB， 在 334 版本驱动前，默认值是 32MB  |
| `CUDA_CACHE_PATH`     | 略     | 设置 cache 目录                                              |
| `CUDA_FORECE_PTX_JIT` | 0      | 设置为 `1` 时，强制忽略预编译好的机器代码，使用 JIT 从 PTX 代码中编译出机器代码 |



# 4 编译参数 arch & code

`nvcc` 编译器在编译 CDUA 代码的时候提供了 `-gencode` 参数，这个参数接收 `arch=compute_xx` 与 `code=sm_xx` 作为参数。也可以单独使用 `-arch` 和 `-code` 来指定。

```bash
nvcc x.cu -gencode arch=compute_50,code=sm_50
```



## --gpu-architecture(-arch)

这个参数接受**虚拟架构**作为参数，通常来说这个参数与最终编译出来的 PTX 代码无关，它只是作为编译 CUDA 代码时候的预处理参数。

在 CUDA 编程中提供了宏 `__CUDA_ARCH__` 可以通过这个宏来控制编译的内容，当编译参数为 `-arch=compute_35` 时，`__CUDA__ARCH__` 的值就是 350。



## --gpu-code(-code)

这个参数指定了 CUDA 代码实际要编译、优化的 PTX 代码对应的 Nvidia GPU。 其中虚拟架构代码 `compute_xx` 用来指定 PTX 代码版本，真实架构代码 `sm_xx` 用来指定机器代码。 如果程序在执行前无法找到设备 GPU 对应的二进制代码，那么就是使用 JIT 从 PTX 代码中编译出对应的机器代码。

同时使用 `-arch` 与 `-code` 参数时候，要确保使用的虚拟架构与真实架构兼容的。



当编译时只提供了 `-arch` 参数，这个时候其实是 `-arch` 与 `-code` 的简写，`-code` 的值默认等于前者。

```bash
# 提供虚拟架构代码
nvcc -arch=compute_61 
# 等价于
nvcc -arch=compute_61 -code=compute_61

# 提供真实架构代码
nvcc -arch=sm_61 
# 等价于
nvcc -arch=compute_61 -code=sm_61,compute_61
```



## 胖二进制(Fat Binaries)

```bash
nvcc x.cu 
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_79,code=\'compute_70,sm_70\'
			
```

在编译的时候可以同时制定多套参数，编译出来的代码就可以支持多种不同架构的 GPU 设备，在具体执行的时候，驱动会选择当前真实架构对应的机器代码，如果找不到，那么就使用兼容的 PTX 代码 JIT 编译出对应的机器代码，这会导致在第一次执行时候速度会比较慢。 

这种将同一份代码编译出多个不同架构代码的方式就叫"fat binaries"， 它带来的好处是让代码一次编译就能支持在多种设备上运行；它的缺点是，随着指定的架构版本越来越多，编译速度也越来越慢，编译出来的代码体积也越大。

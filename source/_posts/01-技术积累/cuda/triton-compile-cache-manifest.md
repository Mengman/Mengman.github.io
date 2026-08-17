---
title: Triton 编译缓存 manifest 路径对于缓存命中的影响
date: 2026-08-17T11:31:57.372Z
tags: [machinelearning, deeplearning, CUDA]
categories: CUDA
---

我在上一篇文章 “从 CUDA 到 Triton：深入理解 Triton 的即时编译与自动调优” 中介绍了 triton 编译缓存，同时还介绍了如何在部署环境复用编译缓存。 本篇文章我将介绍编译文件中的 manifest 文件对于缓存命中的影响。



## 回顾 Triton 编译缓存文件

缓存默认存储在 `~/.triton/cache/` 目录下。在该目录中你可以看到一系列以哈希命名的文件夹，每个文件夹内包含：

- `kernel.cubin`：编译好的 GPU 二进制
- `kernel.ttir` / `kernel.ttgir`：中间表示文件（可选，用于调试）
- `__grp__<kernel_name>.json`:  这是编译文件清单，里面记录了每个文件在系统中的绝对路径。

具体文件内容：

```json
{
  "child_paths": {
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ttir": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ttir",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ttgir": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ttgir",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.llir": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.llir",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ptx": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.ptx",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.cubin": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.cubin",
    "sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.json": "/home/root/.triton/cache/PNPC3BUAFXZSIJBEXTKX3IJQIKDKJGATIMRJU26VBJG2LR4D7O5A/sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel.json"
  }
}
```

文件中记录了编译文件在文件系统中的绝对路径。 经过我测试发现，**如果这些文件的实际路径与清单中的路径不符**，**那么 triton 就会重新编译整个算子**。



## 实验记录

**测试环境**

triton=3.3.1



1. **没有缓存情况下第一次跑**

GPU:                 NVIDIA GeForce RTX 4090 D (cuda:0)

Triton cache:        /home/jovyan/.triton/cache

Cache files:         1 -> 283

Cache bytes:         22328 -> 13471311

**First call:          23414.190 ms**

Warm median:         0.147 ms

Warm minimum:        0.145 ms

Output finite:       True

**这里做了 auto_tunning + 算子编译**



2. **已经有缓存文件**

GPU:                 NVIDIA GeForce RTX 4090 D (cuda:0)

Triton cache:        /home/jovyan/.triton/cache

Cache files:         283 -> 283

Cache bytes:         13471311 -> 13471311

**First call:          286.002 ms**

Warm median:         0.166 ms

Warm minimum:        0.156 ms

Output finite:       True



First call 用时，说明 triton 既没有做 auto-tune 也没有重新编译算子



3. **删除  grp.json 文件**

GPU:                 NVIDIA GeForce RTX 4090 D (cuda:0)

Triton cache:        /home/jovyan/.triton/cache

Cache files:         243 -> 244

Cache bytes:         13417943 -> 13420877

**First call:          608.326 ms**

Warm median:         0.265 ms

Warm minimum:        0.233 ms

Output finite:       True

**跑完之后重新生成了 一个 `__grp__*.json` 文件, 这个算子应该就是 auto-tuning 出来的最优算子**。



4. **再重复跑一次**

GPU:                 NVIDIA GeForce RTX 4090 D (cuda:0)

Triton cache:        /home/jovyan/.triton/cache

Cache files:         244 -> 244

Cache bytes:         13420877 -> 13420877

**First call:          288.451 ms**

Warm median:         0.161 ms

Warm minimum:        0.152 ms

Output finite:       True



这次 first call 又变成 200多毫秒，说明命中了缓存。



5. **将 grp.json 的路径从 "/home/jovyan/.triton"  替换成 "/home/root/.triton"**

GPU:                 NVIDIA GeForce RTX 4090 D (cuda:0)

Triton cache:        /home/jovyan/.triton/cache

Cache files:         244 -> 244

Cache bytes:         13420866 -> 13420877

**First call:          623.925 ms**

Warm median:         0.208 ms

Warm minimum:        0.182 ms

Output finite:       True



`__grp__*.json` **文件中的路径重新变为 "/home/jovyan/.triton"**

**所有编译文件的创建时间都更新了，说明这个算子被 triton 重新编译了**。



## 结论

1. `__grp__*.json` **文件是否存在和文件中的路径是否正确，确实会影响缓存命中。**
2. **如果** `__grp__*.json` **文件不存在，或者路径不符合当前环境，triton 会重新编译这个算子，但是不会重新进行 auto-tuning。**

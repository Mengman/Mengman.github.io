---
title: 【投机解码专题】核心论文： Speculative-Decoding-Fast-Inference-from-Transformers-via-Speculative-Decoding
date: 2026-07-02 23:43:18
tags: [LLM, Speculative-Decoding]
categories: paper
typora-root-url: ..
---
**投机解码**（Speculative Decoding）的核心思想，在 2022–2023 年间由 Google Research 和 DeepMind 两个团队几乎同时独立提出，代表性工作分别是：

- **"Fast Inference from Transformers via Speculative Decoding"**（Leviathan et al., 2023）
- **"Accelerating Large Language Model Decoding with Speculative Sampling"**（Chen et al., 2023）

两篇论文共同奠定了这一方向的理论基础，本文笔记以第一篇（Leviathan 等）为主线展开。

# Fast Inference from Transformers via Speculative Decoding

## 🚀五分钟全景图

**一句话概括：**

本文提出了一种**投机解码**方案：先用一个极轻量的“草稿模型”并行生成多个候选 Token，再由目标大模型做一次性验证。在**输出分布数学上严格不变**的前提下，将 Transformer 推理速度提升了约 **2～3 倍**。

**创新点 (Contribution)：**

1. **分布无损加速**：与知识蒸馏或剪枝不同，本文设计的“投机采样”算法，在数学上严格保证了最终采样分布与大模型原生分布完全一致，做到了“加速但不改输出”。
2. **开箱即用**：无需改动大模型架构或重新训练，直接复用现成的小模型作为草稿即可。
3. **模型无关**：适用于任意自回归架构（T5、LaMDA、GPT 等）。

**启发 (Inspiration)：**

- **工程实践：** 在显存带宽受限（Memory-bound）的场景下，通过增加计算并发度（Compute Concurrency）来换取延迟降低是极其有效的策略。
- **模型协作：** 本文开启了“大小模型协同推理”的新思路。小模型负责快速生成候选，大模型负责精准验证，分工明确。



---

## 1. Introduction

当前主流的基于 Transformer 的自回归 LLM，在解码阶段面临一个根本性瓶颈：生成 K 个 Token 需要串行执行 K 次前向传播，模型越大，单步延迟越明显。

但论文中有一个关键的观察：并非所有解码步骤都同样困难。有些位置的预测几乎是确定性的（比如“喜马拉雅**山**”），而有些位置则面临较大的不确定性（比如“世界上最高的**山**”，候选词还可能是“树”“人”“动物”等）。这意味着，如果有一个轻量级模型能先“猜”几步，再由大模型一次性确认，就有可能跳过一部分串行开销。

受 CPU 设计中“投机执行”思想的启发，论文将此思路迁移到 LLM 推理中：用一个高效的近似模型（草稿模型）先生成多个连续的候选 Token，然后让大模型并行验证这些候选，只保留那些符合大模型分布的结果。

![fig-1](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/fig-1.png)

图 1 展示了一个具体示例：一个 6M 参数的草稿模型生成了若干候选 Token（绿、红、蓝三色），随后 97M 的目标模型一次性完成验证。其中，<span style="color:green">绿色</span> Token 被目标模型接受，<span style="color:red">红色</span> 被拒绝，而 <span style="color:blue">蓝色</span> Token 由于位于首个被拒绝 Token 之后，也被一并丢弃。

在 T5-XXL（11B）上的实验表明，该方法可获得约 2～3 倍的端到端加速。



## 2. Speculative Decoding

### 2.1 Overview

$M_p$ 表示目标模型，$p(x_t|x_{<t})$ 表示目标模型在给定 t-1 个前缀token情况下第 t 个token的输出概率分布； $M_q$ 代表草稿模型，$q(x_t|x_{<t})$ 表示草稿模型对于第 t 个token输出的概率分布。

流程：

1. 使用草稿模型 $M_q$ 生成 $\gamma$ 个预测 token；
2. 使用目标模型 $M_p$ 一次性并行验证所有的 token，只接受可以获得与 $M_p$ 概率分布相同的 token；
3. 如果第 t 个 token 被拒绝了，那么第 t 个 token 就以 $M_p$ 预测的第 t 个token 为准，总共生成 t 个token； 如果所有 $\gamma$ 个 token 都被接受了，那么再加上验证过程中 $M_p$ 生成的一个 token，总共生成了 $\gamma + 1$ 个 token。 

### 2.2 Standardized Sampling

为了后续数学证明的严谨性，将 Top-k, Nucleus, Temperature 等采样策略统一视为从调整后的概率分布中采样。

$p(x)$ 和 $q(x)$ 分别代表目标模型 $M_p$ 和 草稿模型 $M_q$ 在根据采样策略调整之后的概率分布。

### 2.3 Speculative Sampling

本节是整篇论文的理论核心。Algorithm 1 结合附录 A.1 给出了一个关键证明：采用投机采样策略后，最终被接受的 Token 的边缘分布，与直接使用目标模型采样完全一致。

![alg-1](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/alg-1.png)

验证阶段的核心逻辑如下：

- 若 $q(x)≤p(x)$（草稿模型“并未高估”该 Token），则直接接受；
- 若 $q(x)>p(x)$（草稿模型“过度自信”），则以概率 $β=\frac{p(x)}{q(x)}$接受，其中 $\beta$ 称为该 Token 的**接受率**；
- 一旦某个 Token 被拒绝，则从修正分布 $p′(x)=norm(max⁡(0,p(x)−q(x)))$ 中重新采样，作为该位置的最终输出。



**分布一致性证明**

某个 Token x′ 被最终采样的概率，由两部分组成：（1）作为草稿模型的候选被直接接受；（2）草稿候选被拒绝后，从修正分布中重新采样选中。
$$
P(x=x')=P_{accpeted}(x=x') + P_{rejected}(x=x') \tag{1}
$$


$P_{accpeted}$ 等于草稿模型输出的概率乘以token 的接受率 $\beta$ 
$$
P_{accpeted}(x=x') = q(x') \times \beta \tag{2}
$$
$\beta$ 的具体形式如下：
$$
\beta = 
\begin{cases}
1, & \text{if } p(x') \geq q(x') \\
\frac{p(x')}{q(x')}, & \text{if } p(x') < q(x')
\end{cases} = \min \left(1, \frac{p(x')}{q(x')} \right) \tag{3}
$$
把 (3) 带入 (2)
$$
P_{accpeted}(x=x') = q(x') \times min(1, \frac{p(x')}{q(x')})=min(q(x'), p(x')) \tag{4}
$$


$P_{rejected}$ 等于被拒绝的概率乘以调整概率之后x'的采样概率
$$
P_{rejected}(x=x')=(1-\beta) \times p'(x') \tag{5}
$$
调整后的概率等于 
$$
\begin{align}
p'(x) &= \text{norm}( \max(0, p(x) - q(x)) ) \\
 &=  \frac{p(x) - max(0, p(x)-q(x))}{\sum_{x'}{p(x')-min(q(x'), p(x'))}} \\
 &= \frac{p(x)-min(q(x), p(x))}{1-\beta}
 \end{align}  \tag{6}
$$
把 (6) 带入 (5)
$$
P_{rejected}(x=x')=p(x')-min(q(x'), p(x')) \tag{7}
$$
最后再把 (4) 和 (7)  都带入 (1)
$$
\begin{align}
P(x=x') &= min(q(x'), p(x')) + p(x') - min(q(x), p(x')) \\
&= p(x')
\end{align} \tag{8}
$$
这说明，投机采样的最终输出分布与目标模型原生分布 **完全一致**，没有任何近似或偏差。



**具体举例说明**

假设在某个位置，草稿模型$M_p$ 和目标模型 $M_q$的输出分布如下：

| token | 草稿模型 q(x) | 目标模型 p(x) | 投机采样最终概率 P(x)                                        |
| ----- | ------------- | ------------- | ------------------------------------------------------------ |
| A     | 0.8           | 0.4           | $P_{accpeted}(A)=0.8 \times \frac{0.4}{0.8}=0.4$ <br />因为 $q(x) > p(x) \rightarrow P_{rejected} = 0$<br />$P(A)=0.4 + 0 = 0.4$ |
| B     | 0.1           | 0.3           | $P_{accpeted}(B)=0.1 \times 1=0.1$<br />只有选中 token A 才可能被拒绝，$P_{rejected}(B) = q(A) \times (1-\beta(A)) \times \frac{p(B)-q(B)}{p(B)-q(B) + p(C)-q(C)} = 0.8 \times \frac{0.4}{0.8} \times \frac{0.3-0.1}{0.2 + 0.2} = 0.2$ <br />$P(B)=0.1+0.2=0.3$ |
| C     | 0.1           | 0.3           | $P_{accpeted}(C)=0.1 \times 1 = 0.1$<br />$P_{rejected}(C)=q(A)\times (1-\beta(A))\times \frac{p(C)-q(C)}{p(B)-q(B) + p(C)-q(C)}= 0.2$ <br />$P(C)=0.1+0.2=0.3$ |





## 3. Analysis (理论分析)

### 3.1 Number of Generated Tokens (生成 Token 数量)

这一节分析单轮迭代（草稿生成 + 大模型验证）能产出多少个 Token 的期望值，它是后续加速比公式的理论起点。

论文假设各位置的接受率 $\beta$ 是独立同分布（i.i.d.），且令 $\alpha=E(\beta)$（即平均接受概率）。

那么，生成 token 数量的期望为：

- 至少生成 1 个 token 的概率是 1（必然事件）。
- 至少生成 2 个 token 的概率等于第 1 个猜测被接受，即 $\alpha$。
- 至少生成 3 个 token 的概率等于前 2 个猜测都被接受，即 $\alpha^2$。
- ……
- 至少生成 $\gamma+1$ 个 token 的概率等于前 $\gamma$ 个猜测全部被接受，即 $\alpha^\gamma$。

因此： $E( \text{# generated tokens}) = 1 + \alpha + \alpha^2 + \dots + \alpha^\gamma$

这是一个标准的等比数列求和，首项为 1，公比为 $\alpha$，项数为 $\gamma+1$。代入求和公式即得：
$$
E(\text{# generated tokens}) = \frac{1 \times (1 - \alpha^{\gamma+1})}{1 - \alpha}
$$

- 当 $\alpha \to 1$ 时（草稿模型完美预测），$E \to \gamma + 1$。即每轮生成上限个数，加速比最大。
- 当 $\alpha \to 0$ 时（草稿模型完全随机），$E \to 1$。即退化成普通自回归，每轮只生成 1 个 token，无加速也无损失（最坏情况也不比基线更慢）。



![fig-2](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/fig-2.png)

Figure 2 是以 $\alpha$ 为横轴（范围 0~1），纵轴为 $E(\text{# tokens})$，并画出了不同 $\gamma$（1, 2, 3, 5, 7, $\infty $）下的曲线簇。从图2中可以获得两点启示：

1. **投机解码的收益高度依赖于草稿模型的质量**：当 $\alpha < 0.6$ 时，所有曲线都非常扁平，无论 $\gamma$ 有多大，被接受的 token 数都不大于3。如果草稿模型只有 50% 的准确率，拉长猜测窗口（增大 $\gamma$）纯属浪费计算资源（因为第一个错误出现得很早）；只有当你确信小模型在大样本上能达到 80%~90% 的接受率时，加大 $\gamma$ 才有爆发式的线性增益。
2. **边际递减效应明显**：对于固定的 $\alpha$（例如 $\alpha=0.8$），从 $\gamma=1$ 提升到 $\gamma=3$ 收益明显，但从 $\gamma=5$ 提升到 $\gamma=7$ 收益急剧放缓（曲线间距压缩）。草稿模型的计算成本随着 $\gamma$ 的增大线性增大，但是收益却没有等比例的提高。



### 3.2 Calculating $\alpha$

引入 $D_{LK}$ 散度: 

$D_{LK}(p, q) = \sum |p(x) - q(x)| = 1 - \sum \min(p(x), q(x))$

推导出  $\alpha = 1 - E(D_{LK}(p, q))=E(min(p,q))$。说明 $M_q$ 和 $M_p$ 越像，$\alpha$ 越大，加速效果越好。



### 3.3 Walltime Improvment  (延迟提升)

引入**成本系数** c 它等于**草稿模型与目标模型单次推理时间的比值**： $c=\frac{T_{M_q}}{T_{M_p}}$ ，结合 $\frac{1 \times (1 - \alpha^{\gamma+1})}{1 - \alpha}$ 

获得总加速比：
$$
\begin{align}
\text{Speedup} &= \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)} \\
&= \frac{1+\alpha}{1+c}
\end{align}
$$


* $\alpha$ 越大，草稿模型猜的越准，加速效果越好。
* $c$ 越小，草稿模型相比目标模型的推理成本越低，加速效果越好



### 3.4. Number of Arithmetic Operations (计算量分析)

在投机解码中如果 token 被接受了，那么在没有额外增加计算量的情况下 “免费” 获得了 token；如果 token 被拒绝，那么草稿模型对于被拒绝 token 的计算量就是额外的成本。

另外 Decoding 是访存密集型场景，就算增加计算量来换取减少内存带宽的占用也是值得的。 因为 LLM Decoding 阶段主要时间在搬运权重（读显存），而不是计算。并行验证 5 个 token 搬运权重的次数和验证 1 个是一样的。



### 3.5. Choosing $\gamma$ 

**(1) 优化的目标函数（核心准则）**

寻找最优 $\gamma$ 的唯一标准是**最大化实际墙钟加速比（Walltime Improvement）**，即最大化 Theorem 3.8 中的公式：
$$
\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
$$
这里存在一个明确的**权衡**：

- **分子（$1 - \alpha^{\gamma+1}$）**：增大 $\gamma$ 能提升期望产出 Token 数（来自公式 1），这是**收益**。
- **分母（$\gamma c + 1$）**：增大 $\gamma$ 意味着草稿模型需要串行执行更多步，耗时近似线性增长 $(\gamma c + 1)$，这是**成本**。

因此，**最优 $\gamma$ 是边际收益等于边际成本的均衡点**。由于 $\gamma$ 是正整数，论文建议直接**数值搜索**（参见 Figure 3），无需解析求解。

![fig-3](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/fig-3.png)

**(2) Figure 3 的工程启示**

Figure 3 展示了不同 c下最优 $\gamma$ 随 $\alpha$ 的变化趋势，从中可以总结两条实用规则：

- **$\alpha$ 越高，最优 $\gamma$ 越大**：当小模型很准（如 $\alpha > 0.8$）时，曲线陡升，值得“赌”一个很长的猜测序列，因为第一个错误很少出现。
- **$c$ 越大，最优 $\gamma$ 越小**：如果小模型跑得不够快（比如 $c=0.1$），最优 $\gamma$ 被严厉压制（几乎不超过 3~4）。因为每多猜一个 token 的延迟成本太高，不如频繁地切回大模型验证。



**(3) 核心洞见：固定 $\gamma$ 并非最优（自适应 $\gamma$ 的潜力）**

这是 3.5 节最具前瞻性的分析。论文指出：

> 在整个推理过程中，接受率 $\beta$ **并不是恒定的**（因上下文而异）。某些前缀下小模型极准（$\beta$ 很高），某些前缀下小模型极差（$\beta$ 很低）。

如果全流程死守一个固定的 $\gamma$，就会导致：
- 在“简单前缀”下，$\gamma$ 设得太小，浪费了高接受率的潜力；
- 在“困难前缀”下，$\gamma$ 设得太大，导致频繁在早期拒绝，浪费计算。



**(4) “预言机（Oracle）”上界分析**

为了量化动态调整的价值，论文假设存在一个**完美的预言机（Oracle）**，能提前预知当前的 $\beta$ 并实时调整最优 $\gamma$。

在这种情况下，期望生成的 token 数将退化为：
$$
E(\text{# tokens}) = \frac{1}{1 - \alpha}
$$
（这是几何分布的无记忆期望，不再受 $\gamma$ 的截断限制）。

论文进一步量化：**相比于固定 $\gamma$ 的方案，使用动态最优 $\gamma$ 的加速比上限可以再提升高达 ~60%**（在典型参数下）。



**(5) 论文的最终立场**

尽管动态 $\gamma$ 极具诱惑力，论文在 3.5 节末尾明确将此列为 **“未来工作”（Future Work）**。在本次实验（第 4 章）中，他们为了验证理论的可复现性，依然选择了**固定的 $\gamma$**（在表 2 中给出具体数值），并未在线上推理时实时改变 $\gamma$。



### 3.6 Approximation Models(近似模型选择)

论文讨论了以下几类草稿模型的选择：

1. **小 Transformer：** 参数量比大模型小 2 个数量级效果时，能在  c 和 $\alpha$ 之间取得最佳平衡。
2. **N-gram：** 极低成本的模型也能提供微弱加速（$\alpha \approx 0.2$）。
3. **非自回归模型：** 也可以作为草稿模型。



## 4. Experiments

### 4.1 Empirical Walltime Improvement

**实验设置**

* 目标模型： T5-XXL 参数量 11B
* 任务：
  * 英译德（WMT EnDe）
  * 文本摘要（CNN/DailyMail）
* 草稿模型：T5-SMALL(77M)，T5-base(250M), T5-large(800M)
* 采样策略: 贪婪 argmax(temp=0), 标准随机采样 (temp=1)

![table-2](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/table-2.png)

实验总结：

* T5-Small 作为草稿模型效果最好（平衡了速度和准确率）。
* **加速比：** 2.6X (采样) 到 3.4X (贪婪搜索)。
* 实测结果与理论公式（Theorem 3.8）吻合。



### 4.2 Empirical $\alpha$ Values

实验不同类型的草稿模型的 $\alpha$ 值

* 与目标模型**架构相似**的小模型可以获得较高的 $\alpha$
* 即使是 Unigram/Bigram 这种弱模型，\alpha 也有 0.05-0.20，说明只要有相关性就能加速。

![](/image/Fast-Inference-from-Transformers-via-Speculative-Decoding/table-3.png)

## 5. Related work 

* **自适应计算 (如 Early Exit)：** 通常需要修改模型结构或重新训练，本文无需任何改动。

- **Blockwise Parallel Decoding：** 只能做贪婪搜索，且需要训练辅助头。本文支持采样，且无需训练。
- **知识蒸馏：** 会改变输出分布。本文在数学上保证了分布不变。


## 6. Discussion

**局限性**

投机解码的本质是用额外计算换取延迟降低，因此要求硬件有足够的并行计算余量。如果 GPU 利用率已接近饱和，该方法不会带来收益。

**值得关注的后续方向**

- 将投机解码与 Beam Search 结合（论文附录 A.4 指出会有一定性能损失）；
- 训练专门以最大化 $\alpha$ 为目标的草稿模型；
- 层级化投机：用更小的模型加速草稿模型本身；
- 跨模态推广（如图像生成等非文本自回归任务）。
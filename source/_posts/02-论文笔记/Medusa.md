---
title: 【投机解码专题】自主多头预测架构-1：MEDUSA
date: 2026-07-09 13:21:36
tags: [LLM, Speculative-Decoding]
categories: paper
typora-root-url: ../../
---

独立草稿模型的两大痛点
1. 草稿模型输出 token 的概率分布与目标模型存在偏差，限制了草稿的接受长度。
2. 在部署的时候再额外维护一个小模型的成本太高。



自主多头预测架构 (Self-Speculative / Multi-Head Decoding) 的解法：**把草稿模型“吞进”目标模型内部**。

1. 在目标模型上增加多个独立的预测头作为草稿模型，并且复用目标模型的 hidden-state 来确保在概率分布上接近。
2. 在部署过程中只用维护一个模型。



这个方向的核心论文：
**MEDUSA (美杜莎):**

* _Paper:_ **"MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (2024, Cai et al.)
- _贡献:_ 投机采样领域的重大里程碑。在冻结的 Base LLM 顶层增加多个并行的 ResBlock Heads，每个 Head 分别预测 $t+1, t+2, \dots, t+k$ 个位置的 Token，并结合 Tree-structured Attention 组织草稿树，由大模型一次性验证多条路径。

 **EAGLE 系列:**

- _Paper:_ **"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"** (2024, Li et al.)
  	    - Eagle-2: Faster Inference of Language Models with Dynamic Draft Trees
  	    - Eagle-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test
     - _贡献:_ 认为 MEDUSA 忽略了 Token 之间的顶层特征依赖（Feature Uncertainty）。EAGLE 在隐藏层（Feature Level）进行轻量级的自回归外推，使得草稿头在预测更远的 Token 时，准确率大幅超越 MEDUSA，成为目前业界单设备投机采样的 SOTA（State-of-the-Art）之一。目前已迭代至 **EAGLE-3**。

本次先从 MEDUSA 开始做阅读笔记。



## 五分钟全景图

论文名称：[**"MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (2024, Cai et al.)](https://arxiv.org/abs/2401.10774)

 **一句话概括**：
 这篇论文提出了 **MEDUSA**，一种通过在 LLM 顶部添加多个轻量级解码头（Decoding Heads）来并行预测未来多个 token，从而打破自回归（Auto-regressive）串行瓶颈的推理加速框架。

**创新点 (Contribution)**：
1.  **架构创新**：抛弃了传统投机采样（Speculative Decoding）中需要维护一个独立“草稿模型（Draft Model）”的复杂设计，直接在原模型上“嫁接”多个单层解码头。
2.  **算法突破**：提出了 **Tree Attention** 机制，允许在一次前向传播中并行验证多个候选 token 序列（而非单条链）。
3.  **训练策略**：设计了 **MEDUSA-1**（冻结主干，仅训头）和 **MEDUSA-2**（联合微调）两种方案，并引入了 **Self-Distillation** 解决无训练数据的问题。
4.  **采样优化**：提出了 **Typical Acceptance** 方案，替代传统的拒绝采样，在高温度采样下表现更好。

**启发 (Inspiration)**：
* **工程上**：无需引入额外的模型文件，极大降低了部署复杂度（尤其是分布式环境）。
* **研究上**：证明 LLM 的 hidden state 包含未来 token 的预测信息，为“非自回归生成”提供了参数高效微调（PEFT）的新路径。



## 1. Introduction

核心问题：投机采样能加速 LLM 推理，但维护独立草稿模型的工程成本太高，特别是在分布式系统中。

**MEDUSA 方案**

1. MEDUSA heads: 将多个解码头连接到目标模型上，作为草稿模型。
2. 多头并行预测：相比于草稿模型串行预测后续 K 个 token，MEDUSA 多个头并行预测后续 token，加速草稿模型的推理。
3. 更高效的验证：用 **典型接受方案**（typical acceptance scheme）替代拒绝采样方案，提升采样效率。

**训练方案**

* MEDUSA-1：冻结骨干，仅训练 Heads，显存占用低，**但预测准确率低于 MEDUSA-2**。
* MEDUSA-2：融入到 SFT 和后续的RLHF 训练过程中，可以提供更好的预测质量，但是训练流程更加复杂。



## 2. Methodology

![图1 Medusa架构和流程](/image/Medusa/medusa-fig-1.png)

图1 展示了 MEDUSA 整体的架构与流程



### 2.1 Key components

#### 2.1.1 MEDUSA HEADS

**模型结构**
* 在 LLM 最后一个 transform layer，添加 $K$ 个额外的解码头。
* 第 $k$ 个头负责预测第 $t+k+1$ 个 token。
* 结构极简 由 Feed-Forward Network (单层) + 残差连接组成，参数量极小，用最后一层的 hidden state 作为输入。

$$
p^{(k)}_t = softmax(W^{(k)}_2 \cdot ( \text{SiLU}(W^{(k)}_1 \cdot h_t) + h_t))
$$

* $p^{(k)}_t$：第 k 个 head 的预测的 token 概率分布
* $h_t$： 为目标模型最后一层在 t 位置输出的 hidden state

#### 2.1.2 Tree Attention
为了提高草稿模型预测的准确率，论文选择每个 MEDUSA head 输出的 top-k 个 token 进入验证阶段， 这些候选的 token 构成一颗 token 树。但是这带来一个问题，随着 top-k 和 预测长度 K 的增大，验证路径也指数级增加。

为了高效的验证，论文提出了 Tree Attention 的机制，通过改造 Attention Mask，在一次前向传播中并行验证整棵候选树。

![图2 Tree Attention](/image/Medusa/medusa-fig-2.png)

![标准Causal mask](/image/Medusa/medusa-Attention-mask.png)

在验证前，将所有候选 token 拼接成一个序列，长度为：
$$
\sum^K_{k=1}{\prod^k_{i=1}{s_i}}
$$
$s_i$ 代表第 i 个 head 输出的候选 token 数量。以图2为例， $s_1=2$ , $s_2=3$, 验证序列的长度为 $2 + 2 \times 3 = 8$

标准 causal mask 中，token 能看到自己及之前的所有 token。Tree Attention 同理，但树中节点的“前序”不是按序列位置排列的，需要重新调整 mask，让每个节点只能看到自己的祖先节点。 仔细看图2的 mask 就可以发现这其实就是一个树型结构。

第一组浅黄色的 token `[is, ', the]` 作为 `It` 的子节点；第二组浅绿色的 token 作为 `I` 的子节点。

![解码树](/image/Medusa/medusa-解码树.png)
这种按 Cartesian Product 构建的树，后文称作**笛卡尔积树**（Cartesian Product Tree）或者**密集树**（Dense Tree)。



### 2.2 Training Strategies

MEDUSA 有两种训练策略

**MEDUSA-1**：单独 fine-tune MEDUSA head

* 冻结 Backbone，只训练 MEDUSA Heads。
* **Loss** 函数
	* 使用 Cross-Entropy loss 作为训练 loss
	* 随着 MEDUSA head 序号增大，预测的不确定增大，loss 也增大；
	* 引入 $\lambda_k$ 作为权重，平衡各 head 之间的 loss； $\mathcal{L}_{\mathrm{MEDUSA-1}}=\displaystyle \sum_{k=1}^K -\lambda_k log{p_t^{(k)}}(y_{t+k+1})$

* 优点：
	* 微调(fine-tune) 显存占用低，可量化训练（QLoRA 风格）
	* 不破坏原模型能力
* 缺点： 模型准确性较差



**MEDUSA-2**： 将 Backbone 和 MEDUSA Heads 联合训练，可以获得更好的预测准确度。

论文提出了三个训练 trick



*  **Combined Loss**：
	* $\mathcal{L}_{\mathrm{MEDUSA-2}}=\mathcal{L}_{LM} + \lambda_0 \mathcal{L}_{\mathrm{MEDUSA-1}}$  
	* 将原模型的 loss $\mathcal{L}_{LM}$  和 MEDUSA-1 的 loss 进行组合，并使用 $\lambda_0$ 来设置权重。
* **Differential LR**
	* 为 Backbone 和 Heads 的训练状态不同，需要分别设置不同的学习率；
	* BackBone 学习率小，Heads 学习率大。
* **Heads Warmup**
	* Stage-1： 按照 MEDUSA-1 进行训练
	* Stage-2：先单独训 Backbone 几个 Epoch；在联合 Backbone 和 Heads 一起训练



### 2.3 Extensions

#### 2.3.1 Typical Acceptance
投机解码依赖**拒绝采样（Rejection Sampling）** 来保证输出分布与原始模型严格一致。但在工程实践中，这带来了两个致命缺陷：
1. 当采样温度（Temperature）升高时，接受率急剧下降，导致加速失效；
2. 即使草稿模型与原始模型分布完全一致（理想情况），由于二者独立采样，拒绝采样依然会引入额外随机开销，造成不必要的拒绝。

**Typical Acceptance**
**改进**：不追求和目标模型的概率分布强一致，而是设定一个阈值（基于 Entropy），只要候选 token 的概率“足够典型”（不是极低概率的异常值）就接受。
$$
\begin{aligned} p_{\text{original}}(x_{n+k} \mid x_1, x_2, \dots , x_{n+k-1}) &> \\
\min \big(\epsilon, \delta \exp \left( -H(p_{\text{original}}( \cdot \mid x_1, x_2, \dots , x_{n+k-1})) \right) \big) \end{aligned}
$$
* $H(\cdot)$ 表示当前上下文下原始模型预测分布的**熵**。
* $\epsilon$ 是硬阈值（Hard threshold），防止极低概率词被接受。
* $\delta$ 是熵依赖阈值的缩放系数。

公式源于截断采样（Truncation Sampling）研究。其核心洞察在于：
1. 当模型预测的熵很高时（即模型非常“不确定”，如创意写作任务），$\delta \cdot e^{-H}$变小，阈值由该值主导，比硬阈值 $\epsilon$  更低，接受门槛更宽松。
2. 当熵很低时（即模型非常“确定”，如代码补全），$\delta \cdot e^{-H}$变大 , 超过 $\epsilon$  ，阈值由 $\epsilon$  主导（相对更严）。

为了保证解码永不卡死（至少推进1个token），**对第一个预测token（即原始LM Head的预测）采用贪心解码并无条件接受**，后续的MEDUSA Heads预测才应用典型接受阈值。这彻底避免了拒绝采样带来的“全盘回退”计算浪费，实验表明（图5）该方案在温度>0时，其生成质量（MT-Bench分数）逼近甚至超越随机采样，且加速比随温度升高而增大。



#### 2.3.2 Self-distillation

在训练 MEDUSA 时候，由于原模型的训练数据没有公开，或者模型经过了RLHF。如果用其他数据直接去训练，会导致模型输出**分布偏移**。

论文采用**自蒸馏**(self-distillation) 的方式在缺乏原模型训练数据的情况下进行训练。

* **生成训练数据**：利用相似领域训练数据作为**种子数据集**(seed dataset)，比如 ShareGPT 数据集来训练 Chat 模型，让原模型根据数据集的 prompt 生成大量训练数据。

* **MEDUSA-1**(冻结Backbone)：直接使用模型自生成的数据集作为Ground Truth来训练Heads，因为骨干权重不变，仅需拟合模型当前的输出分布，足够有效。

* **MEDUSA-2**(联合训练)：仅用硬标签（Hard Labels，即采样出的离散token）训练会严重损害骨干能力。因此引入 **知识蒸馏（Knowledge Distillation）** 损失，将原始模型视为教师模型。其损失函数替换为：

$$
\mathcal{L}_{\mathrm{LM-distill}} = KL\left(p_{\text{original},t}^{(0)} \ \middle|\middle|\ p_{t}^{(0)}\right)
$$

即让当前正在训练的骨干输出的概率分布，去拟合原始冻结教师模型的概率分布（软标签Soft Labels），而非拟合单一的离散Token。

**显存优化**
训练 MEDUSA-2 若同时加载教师和学生两份模型，显存吃不消。论文**使用LoRA（低秩适配器）微调骨干网络**，关闭 LoRA 就是教师，开启 LoRA 就是学生，共享同一份权重，因此**无需额外加载一份模型副本**（这在33B级别模型中可节省约60GB显存）。但论文特别警告：此场景下LoRA**不能配合量化（Quantization）使用**，否则教师模型会退化为量化版本，导致蒸馏目标质量降低，影响最终效果。



#### 2.3.3 Searching for the Optimized Tree Construction

2.1.2节提出的**笛卡尔积树**虽然简单，但在固定节点预算下（受限于计算开销），**均匀扩展每一层并不是最优的**。直觉上，离当前token越近（即kk越小）的Head预测越准，其Top-1准确率远高于远处Head的Top-5。因此，我们需要将宝贵的树节点资源分配给那些**条件概率更高**的分支。

**数学建模与贪心搜索算法**：

- 定义 $a_k^{(i)}$ 为第 k 个Head的第 i 个Top预测在校准集（Calibration Dataset）上的**经验准确率**。
- 假设各Head预测独立，则一条候选路径（由各Head的Top-$[i_1, i_2, ..., i_k]$ 组成）的期望接受长度为：
  $$
  \sum_{[i_1,\dots,i_k]\in I} \prod_{j=1}^k a_j^{(i_j)}
  $$
  

构建树是一个**节点分配**问题。关键推导结论是：
> 在树中添加一个新节点对期望接受长度的**边际贡献**，恰好等于该节点所在路径的累积准确率（即该节点自身的准确率）。

基于此，论文提出了**贪心生长算法**：
1. 从根节点（原始预测）出发。
2. 每一步，从当前树边界上的待选节点中，挑选**准确率最高**的那个节点加入树中。
3. 重复直到树的总节点数达到预设预算（如64个节点）。

这种方法生成的树被称为**稀疏树（Sparse Tree）**。



## 3 Experiments

### 3.1 Case Study: MEDUSA-1 v.s. MEDUSA-2 on Vicuna7B and 13B
**实验目的**：验证 MEDUSA-1 和 MEDUSA-2 两种训练策略，在不同大小(7B, 13B)模型下的性能表现。

**实验配置**：

* 原模型：Vicuna 7B，13B
* 原模型训练数据：ShareGPT

![图3 Medusa-1 对比 Medusa-2](/image/Medusa/medusa-fig-3.png)

**实验结论**

- **Vicuna-7B**：MEDUSA-1 实现 **2.18倍** 加速；MEDUSA-2 进一步提升至 **2.83倍**。    
- **Vicuna-13B**：MEDUSA-1 实现 **2.33倍** 加速；MEDUSA-2 同样稳定在 **2.83倍**。

**任务敏感性（分类别分析）**：
论文特别分析了 MT-Bench 的 8 个任务类别，发现 **代码（Coding）** 和 **信息抽取（Extraction）** 任务的加速比最高（分别达 3.29倍 和 3.62倍）。这暗示 MEDUSA 对结构化、高确定性任务具有天然的优势，为代码大模型的优化提供了强有力支撑。



### 3.2 Case Study: Training with Self-Distillation on Vicuna-33B and Zephyr-7B

**实验目的**：验证在**训练数据不可用或模型经过 RLHF** 的“脏数据”场景下，2.3.2节提出的自蒸馏管线是否有效。

**实验配置**

- 原模型：Vicuna-33B（私有数据集训练）， Zephyr-7B（经过 RLHF 对齐）
- 数据集：利用 ShareGPT 和 UltraChat 作为种子提示词，让目标模型自生成约 10万 条样本。
- 训练策略：MEDUSA-2

![table-1](/image/Medusa/medusa-table-1.png)
**实验结论**

- **加速比（Acc. Rate）**：Vicuna-33B 为 3.01倍，Zephyr-7B 为 3.14倍，虽然略低于 Vicuna-7B（3.47倍），但依然显著。

- **额外计算开销（overhead）**：使用 MEDUSA-2 之后的单次推理计算开销增加了 20%左右，但是换来了 3 倍多的加速。

- **质量保持（Quality）**：这是最大的亮点。通过自蒸馏+KL散度约束，生成质量几乎无损（Vicuna-33B 得分 +0.05，Zephyr-7B 仅 -0.07，Vicuna-13B -0.14）。这证明了自蒸馏能够完美对齐模型的真实输出分布。

- **VS 推测解码（Speculative Decoding）**：在相同模型上，推测解码（使用开源的 Tiny-Vicuna 等草稿模型）的加速比仅为 1.47-1.60倍，远低于 MEDUSA（2.35-2.83倍），凸显了 MEDUSA 在工程简洁性和加速上限上的双重优势。



### 3.3  Ablation Study

#### 3.3.1 configuration of the tree attention
**实验目的**：对比两种 Attention Tree （密集树 v.s. 稀疏树）机制的加速效果。

**实验配置**

* 模型：MEDUSA-2 Vicuna 7B
* 测试项目：MT-Bench 写作和角色扮演

**实验结论**

![图4](/image/Medusa/medusa-fig-4.png)
图 a 中展示不同的 Tree Attention 每轮接受的 token 长度与候选 token 长度的关系；图 b 展示了候选 token 的长度与推理速度之间的关系。

- **密集树（Dense）vs. 稀疏优化树（Sparse）**：实验发现，若单纯靠笛卡尔积堆砌节点（密集树），即使节点数达到 256，其加速效果也不如经过 2.3.3 节贪心算法优化后的 **64节点稀疏树**。

- **边际递减效应**：加速比随节点数增加呈**对数增长**（图4a）。当节点数超过 64 后，增加的树节点带来的计算开销（矩阵乘法与自注意力负荷）已超过其带来的加速收益，整体速度（tokens/s）反而开始下降（图4b）。这界定了 MEDUSA 的最优树规模。



#### 3.3.2 Thresholds of Typical Acceptance

**实验目的**：研究阈值 $\epsilon$ 对于  Typical Acceptance 的影响。

**实验结论**
![图5](/image/Medusa/medusa-fig-5.png)

- **质量与速度的权衡**：随着超参数 $\epsilon$ 从 0.01 增至 0.25，**接受门槛提高（更严格）**，**加速比下降**，但 **MT-Bench 质量分数提高**（图5）。

- **创造性优势**：在温度为 0.7 的**创造力任务**中，Typical Acceptance 的生成质量曲线逼近甚至超过传统的随机采样（RS），且在相同质量下，Typical Acceptance的加速比远高于随机采样。这验证了 2.3.1 节“放弃严格保序，换取更高效率”的核心思想。



#### 3.3.3 Effectiveness of Two-Stage fine-tuning

**实验目的**： 两阶段微调（Two-Stage Fine-tuning）的必要性

**实验结论**
![表-2](/image/Medusa/medusa-table-2.png)

- **直接联合微调（Direct Fine-tuning）**：模型质量**明显下滑**（从 6.17 降至 5.92），验证了同时更新骨干网络会破坏 LLM 原有能力。
- **MEDUSA-1**（冻结骨干）：质量保持 6.23，加速 2.18倍。
- **MEDUSA-2**（两阶段训练）：质量恢复至 6.18（接近基线），且加速飙升到 2.83倍。



**论文中的各项技术对于加速的影响**
![表-3](/image/Medusa/medusa-table-3.png)
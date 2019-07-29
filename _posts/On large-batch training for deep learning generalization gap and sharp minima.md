---
layout: post
title:  "On large-batch training for deep leanring: generalization gap and sharp minima 笔记"
date:   2019-07-29 17:26:21 +0800
categories: machinelearning deeplearning math
---

**论文题目:** On large-batch training for deep leanring: generalization gap and sharp minima

**论文内容：** 在SGD中使用较大的batch-size会导致模型的泛化能力退化，因为较大的batch size 模型更加容易在 sharp minima处收敛，较小的batch size使模型在 flat minima处收敛。

**论文亮点**

> Many theoretical properties of these methods (SGD and its variants) are known. These include guarantees of: 
>
> (a) convergence to minimizers of strongly-convex functions and to stationary points for non-convex functions, 
>
> (b) saddle-point avoidance and 
>
> (c) robustness to input data. 
>
>  
>
> Stochastic gradient methods have, however, a major drawback: owing to the sequential nature of the iteration and small batch sizes, there is limited avenue for parallelization.



### 大 batch size 的缺点

大 batch size 的泛化能力差是因为，它倾向收敛在 sharp minima 处。 极小值的 sharpness 是由 $\nabla ^2 f(x) $ 正特征值的大小来决定的，特征值越大，极小值就越 sharp， 泛化能力也就越差。


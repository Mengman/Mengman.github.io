---
layout: post
title: Softmax Cross Entropy Loss笔记
date: '2019-02-22 17:26:21 +0800'
categories: machinelearning deeplearning math
created: '2019-07-10T05:02:46.731Z'
modified: '2019-07-10T05:02:46.731Z'
tags: [Import-e2cb]
---

## Softmax函数
Softmax函数是在机器学习做分类预测中一个重要的工具函数。Softmax函数的数学定义如下：

$$
\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^Ne^{z_k}} 
\text{ for }j = 1, ..., N  \text{ and } \mathbf{z} = (z_1, z_2, ..., z_k)
$$

如果用神经网络做一个分类手写数字预测，那么神经网络的最后一层有10个输出，分别作为预测为 0 ~ 9 的概率。那么怎么判断预测结果到底是哪个数字呢？ 一个简单的方法是找到10个输出里最大的那个作为预测结果。 但是这10个输出是相互独立的，其和不等于1。所以一个更好的办法是将所有的输出的和缩放到1，然后看每个输出占的比重，将这个值作为预测概率，而 Softmax 函数恰好可以做到这一点。对于分类  i  的预测概率为  $$p_i =\sigma(z)_i$$

Softmax 函数非常好实现，下面是用 numpy 实现的 Softmax

```python
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
```
在实现 Softmax 函数的时候，由于涉及到指数运算，在输出比较大的时候，非常容易造成结果溢出。
为了使得 softmax 函数在数值上更加稳定，我们可以先对 $\mathbf{z}$ 进行 normalization 。 Normalization 除了统计学上的`标准分`( $$\frac{\mathbf{X} - \mu}{\sigma}$$ ) , 机器学习中还常用三种其他的方法： L1, L2 和 max

L1: $$\frac{\mathbf{z}}{\sum_{i=1}^n{\vert z_i \vert }}$$

L2:  $$\frac{\mathbf{z}}{\sqrt{\sum_{i=1}^n{z_i^2}}}$$

max: $$\frac{\mathbf{z}}{\max{\mathbf{z}}}$$

于是可以把上面的代码改写为
```python
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)
```

## Softmax 求导
由于 softmax 的良好性质，在用神经网络做分类任务时，通常直接将最后一层设置为 softmax, 这个时候就要考虑如何对 softmax 进行求导。神经网络输出 $\mathbf{z}$ 长度为 k ，那么要对这个 k 个输出进行求偏导。

$$
\frac{\partial\sigma(\mathbf{z})}{\partial\mathbf{z}} = 
  \begin{bmatrix}
      \frac{\partial\sigma(z)_0}{\partial z_0} & \frac{\partial\sigma(z)_1}{\partial z_0} & ... & \frac{\partial\sigma(z)_N}{\partial z_0} \\
      \frac{\partial\sigma(z)_0}{\partial z_1} & \frac{\partial\sigma(z)_1}{\partial z_1} & ... & \frac{\partial\sigma(z)_N}{\partial z_1} \\
      ... \\
      \frac{\partial\sigma(z)_0}{\partial z_N} & \frac{\partial\sigma(z)_1}{\partial z_N} & ... & \frac{\partial\sigma(z)_N}{\partial z_N} \\
  \end{bmatrix}
$$

对于矩阵中的元素  $$\frac{\partial\sigma(z)_i}{\partial z_j}$$ ， 如果  $$i \neq j$$  :

$$
\begin{aligned} \frac{\partial\frac{e^{z_i}}{\sum_{k=1}^Ne^{z_k}} }{\partial z^j} & = \frac{0 - e^{z_i} e^{z_j}}{ (\sum_{k=1}^Ne^{z_k})^2 } \\
& = \frac{-e^{z_i}}{\sum_{k=1}^Ne^{z_k}} \times \frac{e^{z_j}}{\sum_{k=1}^Ne^{z_k}} \\
& = -p_i \times p_j\end{aligned}
$$

如果 i = j :

$$
\begin{aligned}\frac{\partial\frac{e^{z_i}}{\sum_{k=1}^Ne^{z_k}} }{\partial z^i} &= \frac{ e^{z_i} }{ \sum_{k=1}^Ne^{z_k} } - \frac{ e^{2z_i} }{ (\sum_{k=1}^Ne^{z_k})^2 } \\
& = \frac{ e^{z_i} (\sum_{k=1}^Ne^k - e^{z_i}) }{ (\sum_{k=1}^Ne^{z_k})^2 } \\
& = p_i (1-p_i)
\end{aligned}
$$


求导矩阵可以改写为：

$$
\frac{\partial\sigma(\mathbf{z})}{\partial\mathbf{z}} = 
  \begin{bmatrix}
      p_0(1-p_0) & -p_1 \times p_0 & ... & - p_N * p_0 \\
      -p_0 \times p_1 & p_1(1-p_1) & ... & - p_N * p_1 \\
      ... \\
      -p_0 \times p_N & -p_1 \times p_N & ... &  p_N(1-p_N) \\
\end{bmatrix}
$$

## 交叉熵

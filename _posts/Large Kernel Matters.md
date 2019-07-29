**论文题目:** Large Kernel Matters - Improve Semantic Segmentation by Global Convolutional Network

**论文内容:** 使用全局卷积网络(GCN)解决分割模型中分类与分割任务冲突的问题，使用Boundary Refinement block 提升物体边缘的分割能力。

# 论文亮点

> 1. Classification and localization are two naturally contradictory tasks. For the classification task, the models are required to be invariant to various transformations like translation and rotation. But for localization task, models should be transformation-sensitive, precisely locate every pixel for each semantic category.



# Approach

## Global Convolutional Network

GCN 的目的是同时满足在分割网络中对于分类和定位的需求。从定位任务的要求来说网络结构是一个不包含全连接和全局池化的全卷积结构，因为全局池化会抹去定位信息；从分类角度来说网络需要稠密连接结构，所以卷积核的尺寸应该尽可能的大。

![1560924841688](/home/tx-deepocean/.config/Typora/typora-user-images/1560924841688.png)



GCN中使用 1×k + k×1 的卷积核来代替 k×k的卷积核降低计算量。



## Boundary Refinement

BR 块是一种残差结构，用于优化物体边缘分割的能力

![1560925157354](/home/tx-deepocean/.config/Typora/typora-user-images/1560925157354.png)


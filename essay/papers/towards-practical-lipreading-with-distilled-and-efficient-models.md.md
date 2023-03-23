# TOWARDS PRACTICAL LIPREADING WITH DISTILLED AND EFFICIENT MODELS

The paper works on single words, not sentences!

In this work, we propose a series of innovations that significantly bridge that gap: first, we raise the state-of-the-art performance by a wide margin on LRW and LRW-1000 to 88.5 % and 46.6 %, respectively using self-distillation. Secondly, we propose a series of architectural changes, including a novel Depthwise Separable Temporal Convolutional Network (DS-TCN) head, that slashes the computational cost to a fraction of the (already quite efficient) original model. Thirdly, we show that knowledge distillation is a very effective tool for recovering performance of the lightweight models

This paper proposes:

1. Self-distillation to improve the model performance.
2. Introduces the Depth Separable Temporal Convolution Network (DS-TCN) head to reduce the computation cost compared to a standard MS-TCN.
3. Replaces ResNet-18 with another backbone based on Shuffle Net V2, because the latter is more lightweight.

Data augmentation methods used:

1. Horizontal flip with probability 0.5
2. Random crop of 88x88 area
3. Mixup with weight of 0.4

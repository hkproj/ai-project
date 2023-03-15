# TOWARDS PRACTICAL LIPREADING WITH DISTILLED AND EFFICIENT MODELS

The paper works on single words, not sentences!

This paper proposes:

1. Self-distillation to improve the model performance.
2. Introduces the Depth Separable Temporal Convolution Network (DS-TCN) head to reduce the computation cost compared to a standard MS-TCN.
3. Replaces ResNet-18 with another backbone based on Shuffle Net V2, because the latter is more lightweight.

Data augmentation methods used:

1. Horizontal flip with probability 0.5
2. Random crop of 88x88 area
3. Mixup with weight of 0.4

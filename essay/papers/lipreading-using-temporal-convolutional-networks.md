# LIPREADING USING TEMPORAL CONVOLUTIONAL NETWORKS

In this work, we address the limitations of this model and we propose changes which further improve its performance. Firstly, the BGRU layers are replaced with Temporal Convolutional Networks (TCN). Secondly, we greatly simplify the training procedure, which allows us to train the model in one single stage. Thirdly, we show that the current state-of-the-art methodology produces models that do not generalize well to variations on the sequence length, and we addresses this issue by proposing a variable-length augmentation

This paper works on single words, not sentences!

The paper shows that a model based on TCN instead of Bi-GRU achieves better results. The architecture is the standard Image Seq->ResNet-18->MS-TCN->Softmax

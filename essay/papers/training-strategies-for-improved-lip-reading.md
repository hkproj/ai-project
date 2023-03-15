# TRAINING STRATEGIES FOR IMPROVED LIP-READING

The paper investigates the performance of state-of-the-art data augmentation and training strategies used for the lip reading task.

## Outcomes

1. Using DC-TCN (densely connected temporal convolution networks) along with all data augmentation strategies described can achieve a new state-of-the-art result on the LRW dataset.
2. Time Masking is the most effective data augmentation method followed by mixup. The use of DC-TCN significantly outperforms the MS-TCN, which in turn outperforms Bi-GRU.
3. The use of boundary indicators and self distillation is also effective in improving the accuracy of the model.

## Data augmentation techniques

4. Random Cropping: randomly crop a 88x88 patch from the mouth ROI (region of interest) during training. At test time, simply crop the central patch.
5. Flipping: randomly flip all the frames in a video with probability of 0.5
6. Mixup: create a new training sample by combining two video sequences and the corresponding targets.
7. Time masking: mask N consecutive frames for each training sequence where N is sampled from a uniform distribution between 0 and Nmax. Each masked frame is replaced with the mean frame of the sequence it belongs to. This technique has been successful in ASR applications.

### Word boundary indicator

A matrix is added to the input sequence (sequence of frames) to indicate which frame contains which word.

### Self distillation

Self distillation is used to train student models starting from a teacher model, where students model show improvement over the teacher. Student and teacher models share the same architecture and hyper-parameters.
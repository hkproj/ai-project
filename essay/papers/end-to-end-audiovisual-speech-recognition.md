# END-TO-END AUDIOVISUAL SPEECH RECOGNITION

In this work, we present an end-toend audiovisual model based on residual networks and Bidirectional Gated Recurrent Units (BGRUs). To the best of our knowledge, this is the first audiovisual fusion model which simultaneously learns to extract features directly from the image pixels and audio waveforms and performs within-context word recognition on a large publicly available dataset (LRW). The model consists of two streams, one for each modality, which extract features directly from mouth regions and raw waveforms. The temporal dynamics in each stream/modality are modeled by a 2-layer BGRU and the fusion of multiple streams/modalities takes place via another 2-layer BGRU. A slight improvement in the classification rate over an end-toend audio-only and MFCC-based model is reported in clean audio conditions and low levels of noise

## Architecture

1. We use a ResNet for the audio stream instead of a rather shallow 2-layer CNN
2. We do not use a pretrained ResNet for the visual stream but we train a ResNet from scratch
3. We use BGRUs in each stream which help modeling the temporal dynamics of each modality instead of using just one BLSM layer at the top and
4. We use a training procedure which allows for efficient end-to-end training of the entire network.

## Visual stream

The visual stream is similar to [13] and consists of a spatiotemporal convolution followed by a 34-layer ResNet and a 2-layer BGRU.

## Audio stream

The audio stream consists of an 18-layer ResNet followed by two BGRU layers.

## Classification layer

The BGRU outputs of each stream are concatenated and fed to another 2-layer BGRU in order to fuse the information from the audio and visual streams and jointly model their temporal dynamics. The output layer is a softmax layer which provides a label to each frame. The sequence is labeled based on the highest average probability.

## Training

Training is divided into 2 phases: first the audio/visual streams are trained independently and then the audiovisual network is trained end-to-end. During training data augmentation is performed on the video sequences of mouth ROIs. This is done by applying random cropping and horizontal flips with probability 50% to all frames of a given clip. Data augmentation is also applied to the audio sequences. During training babble noise at different levels (between -5 dB to 20 db) might be added to the original audio clip. The selection of one of the noise levels or the use of the clean audio is done using a uniform distribution.

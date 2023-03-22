# A Multimodal German Dataset for Automatic Lip Reading Systems and Transfer Learning

In this paper we present the dataset GLips (German Lips) consisting of 250,000 publicly available videos of the faces of speakers of the Hessian Parliament, which was processed for word-level lip reading using an automatic pipeline. The format is similar to that of the English language LRW (Lip Reading in the Wild) dataset, with each video encoding one word of interest in a context of 1.16 seconds duration, which yields compatibility for studying transfer learning between both datasets.

We demonstrate learning from scratch and show that transfer learning from LRW to GLips and vice versa improves learning speed and performance, in particular for the validation set.

## Dataset

    GLips consists of 250,000 H264-compressed MPEG-4 videos of speakers’ faces from parliamentary sessions of the Hessian Parliament, which are divided into 500 different words of 500 instances each.
    As with LRW, each video is 1.16s long at a frame rate of 25fps. The audio track was stored separately in an MPEG AAC audio file (.m4a). For each video there is an additional metadata textfile with the fields:
    - Spoken word 
    - Start time of utterance in seconds
    - End time of utterance in seconds
    - Duration of utterance in seconds
    - Corresponding numerical filename in the database

    In order to create GLips, we also need the exact time of pronunciation and the duration of the utterance for each selected word. However, the subtitle files only contain one interval for each of several words. The solution to this problem via alignment using the WebMAUS serviceIn order to create GLips, we also need the exact time of pronunciation and the duration of the utterance for each selected word. However, the subtitle files only contain one interval for each of several words. The solution to this problem via alignment using the WebMAUS service

## Face recognition

Very few complications occur in the videos that these processing conditions do not satisfy, s.a. the speaker moving too vividly or being very tall which could cause the face detection to confuse the speakers face with the person sitting in the elevated position behind him

## Model evaluation

We chose the X3D convolutional neural network model by Feichtenhofer (2020) since it is efficient for video classification in terms of accuracy and computational cost and well designed for the processing of spatiotemporal features

Implementation available here: https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md

## Transfer learning

The authors perform transfer learning between their german dataset and the LRW, to verify if the lip reading models can be improved by transfer learning.
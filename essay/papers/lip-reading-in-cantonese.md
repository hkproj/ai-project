# Lip Reading in Cantonese

1. We present a large scale Cantonese lip reading dataset, 94 named CLRW with 800 word classes and each class consists of one or several Chinese characters. There are a total of 400,000 samples, with an average of 500 samples per class.
2. We have established a pipeline to automatically collect lip reading datasets. To be closer to the real scene distribution, CLRW contains huge variations in diversity of speakers, background clutters, sample lengths, etc.
3. In the next step, we propose a two-branch model, named TBGL, which contains a global branch and a local branch. In this paper, we introduce a bidirectional knowledge distillation loss for jointly training the two branches. Finally, we evaluate our methods on LRW, 107 CAS-VSR-W1K, and CLRW respectively, and demonstrate the effectiveness of our methods.

## Lip Reading Datasets

The paper shows and compares various lip reading datasets and their features.

## Lip Reading Methods

The paper compares various lip reading methods used in literature.

## Dataset

### Data collection

1. The collected videos should include a talking person facing the camera. Moreover, video clips should not contain invalid frames (no speakers or multiple speakers).
2. The video source should contain clear speech and must be discarded if the ambient noise is too large.
3. The examples of speakers in our dataset are shown in Fig.1, in order to make our dataset closer to the real scene distribution, we don’t make too many restrictions on gender, age, speaking speed and light conditions, etc.

We have collected various forms of video sources including Cantonese news programs, Cantonese variety shows, Cantonese talk shows, Cantonese vlogs, and Cantonese character interviews, to achieve data diversity.

### Scene boundary detection

We use the global histogram of the image to judge the switching between a single speaker and other scenes in the video and obtain a rough single speaker video clip. The global histogram calculates the difference between adjacent frames according to Formula 1 by counting the number of all pixels in the frame at each gray level.

### Audio-Video Synchronization

We download videos from Bilibili, YouTube, Guangzhou 361 Radio and Television, TVB, and many other websites. The video and audio stream are inevitably out of sync in the process of repeated encoding.

We first manually filter out the video samples which audios and videos are obviously out of sync. But for small out- of-sync videos, we introduce the SyncNet Model [26] to solve this problem.

### Transcription

The authors generate the transcript using iFlytek service, but then manually verify its quality.

### Face detection and mouth region extraction

The authors use MediaPipe toolkit.

### Dataset statistics

We download a total of 417 hours of original videos from Bilibili, YouTube, TVB, Guangzhou Broadcasting Network and many other websites including news, interviews, talk shows and other forms. We end up with about 65 hours of valid video with 30,000 face image sequences and 400,000 video clips in total. Finally, we keep the 800 most frequent word classes, with an average of 500 samples per word classes.

## Model architecture

See the paper.


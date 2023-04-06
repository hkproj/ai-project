# Italian lipreading dataset

## Objective

The main goal of this project is to train a model to lipread the Italian language using a variety of speakers taken from the web. The primary source for the videos is the popular video sharing platfrom YouTube. Since it is not possible to share videos downloaded from YouTube without its parent company Alphabet's or the video owner's permissions, the videos will not be shared, but rather, links to the videos and the timestamps to crop will be the main output of this dataset building process. Each researcher can use the pipeline to download its own copy of the videos to perform further processing.

### Video selection

Videos have been selected so as to bring as much variety as possible into the training data. This includes videos of males and females, videos of people with different skin tones and background. Videos in which there is primarily one speaker whose face is fully visible have been preferred, as our goal is to catch the lip area in the video and map it to the corresponding text. Educational videos, self-help videos and videos in which there is one active speaker telling a story are more suitable for this task. Even with this kind of content, however, there are some challenges:

1. Italian people are famous all over the world for their hand gestures, and more often than not, hands come in between the speaker's face and the camera, covering the lip area partially or fully.
2. The face of the active speaker is not always facing the camera, but just like in natural conversations, the face may be facing left, right, down or up from time to time.
3. It is not uncommon for Italian people to grow a beard or mustache. This can sometimes partially cover the lip.
4. All the videos have different lighting conditions and the quality of the video (pixel density) is also not homogeneous, as authors may have used different cameras and settings.

All of the abovementioned challenges augment the data variety, and for this reason, no action has been taken to avoid or remove these details from the videos. This way, the data is more robust and we can avoid using data augmentation techniques.

Moreover, there are challenges in the spoken language:

1. Most speakers employ terms taken from the English language, for example the word `online`, `training`, `web`, etc. Since a lipreading model is concerned with the sequence of sounds each word produces, non-Italian words that are used in daily spoken language will be kept as is.
2. Words with accents, whose prononciation depends on the type of accent, will also be kept without any further post-processing. For example, in the Italian language the words *cantò* e *canto* are two conjucations of the same verb *cantare* (to sing); they are written similarly but pronounced differently. However, there not much difference in the visemes, that is, the configuration of the lip area when pronouncing these two variants. Our sequence to sequence model should learn to select the right word based on the left context of the sequence. This is a clear indication that the model must learn to handle long-term dependencies.

## Pipeline

## General overview

A lipreading model takes as input a sequence of images representing the lip area of a speaker and produces a text. No audio is ever sent to the model, which must learn to read the lip area only. To produce the dataset given a set of videos, the first operation is to extract clips from the video in which the face of the active speaker is fully visible. Secondly, the audio needs to be transcribed and aligned to the video.
Given a clip in which the active speaker is fully visible and its aligned transcript, it is possible to generate shorter clips of a few words each as data samples for training and validation.
For example, given a video clip that maps to the following short sentence of `N=15` words:

*Il gatto correva veloce nella prateria, mentre il cane lo guardava da sotto un albero.*

It is possible to generate `T=N-K+1` data samples of by taking a sliding window of size `K`. Punctuation is not counted, as it does not map to any viseme. For example, with `K=8`, it is possible to have the following `T=8` sliding windows:

word1|word2|word3|word4|word5|word6|word7|word8
---|---|---|---|---|---|---|---
Il|gatto|correva|veloce|nella|prateria|mentre|**il**
gatto|correva|veloce|nella|prateria|mentre|**il**|cane
correva|veloce|nella|prateria|mentre|**il**|cane|lo
veloce|nella|prateria|mentre|**il**|cane|lo|guardava
nella|prateria|mentre|**il**|cane|lo|guardava|da
prateria|mentre|**il**|cane|lo|guardava|da|sotto
mentre|**il**|cane|lo|guardava|da|sotto|un
**il**|cane|lo|guardava|da|sotto|un|albero

The sliding window size `K` is a hyper-parameter of the model. This allows to learn as much transitions as possible between words. The sliding window approach is made possible only because each word is aligned to the video. It is clear that, using this method, the model will be overfit on some words and less on others. For example, the model will maximize its learning of the word *il*, which appears in all the rows of the table. This indication can be useful in debugging the model's performance, because we expect the most frequent words to have a lower error rate.

The pipeline is made up of the following steps:

## 1 - Video download

There are many tools available online to download videos from the web. I will be using the popular `youtube-dl`, which allows to download videos in the chosen format and with the chosen quality. I will be using the following argument for the format *parameter*:

```bash
youtube-dl [...] -f "best[ext=mp4]" [...]
```

This tells the tool to download the best available format that uses the `mp4` extension. Different videos may have a different pixel density.

## 2 - Facial recognition

Facial recognition is important for two reasons:

1. Find all the faces that appear in the video so as to select which one to consider then cutting the video into clips, in which only selected face is visible (there may be other faces, but are ignored).
1. Make sure the face is actually visible throughout the video, insomuch that a facial recognition software detects it.

Facial recognition is performed on each frame for each video. While perming this job, the following challenges have been faced:

1. The same face may not be recognized as same by the facial recognition software. This can be due to many reasons, the easiest solution is to use a stronger model to perform this task. However, every model will come with a certain degree of imperfection.
1. The face may not be fully visible, even if it is recognized. For example, the face of a person that is initially facing the camera and after some time is facing left can be recognized by a facial recognition library as still belonging to the same person, because facial recognition software can reconstruct the 3D mesh of the face even if it is not fully visible. The lip area, however, may not be fully visible. It is possible to discard such intervals of time in which the lip area is not fully visible, but since it can be still recognized by the facial recognition software, and since these intervals are not the majority of the intervals, they are kept in the final clips and the model is trained upon them.

The output of this phase is a list of timestamps, corresponding to the frames in which a particular face is visible. This is done for all the faces recognized in the video.

## Face selection and intervals merging

A video may contain multiple faces, so the facial recognition software exports one picture for every face it has recognized. For each face, the software does export not only a picture of the frame in which the face was first seen, but also a list of timestamps (in a text file), one for each of the frames in which the same face is seen again.
With this approach, there are still some challenges to overcome:

1. The facial recognition software may not recognize the same face but treat two different frames with the same person as different people. Of course this is a limitation of the facial recognition library used, but every library has its own non-zero error rate. This can be recognized by the user because the software exports two pictures of the same person with their own list of timestamps.
1. A face may not be recognized in a given frame. This usually happens when the frame is blurry (the camera is out of focus).

For example, a video in which a single person speaks from the time 00:00:13.000ms to 00:00:20:000ms, may produce the following output files:

1. `Face 0.jpg`, with timestamps 00:00.13.000ms, 00:00:13.030ms, 00:00:13.060ms, [...], 00:00:18.150ms, [**missing 2 frames**], 00:00:18.240ms, [...]
1. `Face 1.jpg`, with timestamps 00:00:18.180ms, 00:00:18.210ms

The facial recognition software may treat the same person as two different people, one of them appearing only for two frames. This is obviously an error and the user can merge the two intervals into one big interval by telling the software that these two faces actually belong to the same person.

Gaps in the intervals shorter than a threshold, which is a hyperparameter set to 200ms are also merged as a contiguous interval, because it means that the facial recognition library just didn't recognize the person for a short duration of time because of a blurry frame or a malfunction of the facial recognition library itself.

## 4 - Clips extraction from the videos

The output of the previous step is a list of contiguous intervals in which the same face appears. This is used to cut the video into clips of a minimum duration (a hyperparameter set to 5s).
For example, a video of many minutes in which the speaker alternates with a screencast of his computer, should automatically be cut and resulting in shorter clips in which only the speaker is visible.

## 5 - Audio extraction

The audio is extracted for each of the clips produced by the previous step

## 6 - Audio transcription and alignment

The audio extracted by the previous step is transcribed using the `Whisper` framework from OpenAI. Since `Whisper` does not automatically generate aligned subtitle files, I have used `WhisperX`, a popular fork of `Whisper` that automatically generates `SRT` files, with one line for each word spoken and the corresponding time bounds.

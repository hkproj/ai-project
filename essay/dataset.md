# Italian lipreading dataset

## Objective

The main goal of this project is to train a model to lipread the Italian language using a variety of speakers taken from the web. The primary source for the videos is the popular video sharing platfrom YouTube. Since it is not possible to share videos downloaded from YouTube without Alphabet's or the owner's permissions, the videos will not be shared, but rather, links to the videos and the timestamps to crop will be the main output of this dataset building process. Each researcher can use the pipeline to download its own copy of the videos to perform further processing.

### Video selection

Videos have been selected so as to bring as much variety as possible into the training data. This includes videos of males and females, videos of people with different skin tones and background. Videos where there is primarily one speaker whose face is fully visible have been preferred, as our goal is to catch the lip area in the video and map it to the corresponding text. Educational videos, self-help videos and videos in which there is one active speaker telling a story are more suitable for this task. Even with this kind of content, however, there are some challenges:

1. Italian people are famous all over the world for their hand gestures, and more often than not, hands come in between the speaker's face and the camera, covering the lip area partially or fully.
2. The face of the active speaker is not always facing the camera, but just like in natural conversations, the face may be facing left, right, down or up from time to time.
3. Italian people and most southern Europeans tend to be more hairy than Northern Europeans and it is not uncommon for Italian teenagers and men to grow a beard or mustache. This can sometimes partially cover the lip.
4. All the videos have different lighting conditions and the quality of the video (pixel density) is also not homogeneous, as authors may have used different cameras and settings.

All of the abovementioned challenges augment the data variety, and for this reason, no action has been taken to avoid or remove these details from the videos. This way, the data is more robust and we can avoid using data augmentation techniques.

## Pipeline

### Video selection



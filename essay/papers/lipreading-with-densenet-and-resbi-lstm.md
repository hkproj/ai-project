# Lipreading with DenseNet and resBi-LSTM

The authors builds a novel dataset called "News, Speech, Talk show" dataset for the Mandarin language.
    For the convenience of lipreading, we choose the programs in which speakers face up to the video camera, so that we use the TV programs such as News, Speech and Talk shows from the Internet.

## Dataset

### Video preparation

    We first determine the shot boundaries by comparing color histograms of adjacent frames. Within each shot, we use a face detector in SeetaFaceEngine1 to do face detection and tracking. If there is no face or the scale of face is not lager than 64 × 64 pixels, we reject the shot as not containing any potential speakers.

### AV Synchronization

    In the videos, we find that similar to [3,16], the video and audio streams may be out of sync for about one second. To settle this problem, we adopt the SyncNet model introduced in [27]. The model uses the two-stream CNN architecture with a contrastive loss to estimate the correlation between the mouth movement of the video and the audio track of video.

### Video segmentation

    Some of the cropped videos are too long to train on our devices, because of the GPU memory constraints. For this reason, we choose to split the videos into 3-second video clips, as well as the audio

### ASR

    The subtitles of the TV programs may not have access to obtain. Therefore, we use the audio streams to do speech recognition to get the text via the service of Baidu Aip-Speech.

### Lip area extraction

    The videos in our dataset are the individual speech and the frame rate of videos is 25fps. In this step, we extract the mouth regions from videos. We first apply dlib [28] to detect the facial landmarks. Then, according to the landmarks, we use a mouth-centered RoI to extract lip area for each frame. Since the information of lip movement is the most important for lipreading, we only retain the lip area

## Model

### CNN

    Spatiotemporal convolutional layers can better capture time feature information in the lip sequence, while 2D convolutional layers are used to extract spatial features from input images. Therefore, we first apply a convolutional layer with 64 three-dimensional (3D) kernels of 5 × 7 × 7 size (time/height/width) on the input mouth

    After the spatiotemporal convolutional layer, a dense connection network (DenseNet) is employed to extract more spatial features at each time step and we use the 121-layer version.

### RNN

    A convolutional layer with 1×1 kernels and two-layer resBiLSTM are applied following the visual model. The goal of the convolutional layer is to reduce the dimension from 1024 to 512. Meanwhile, a shortcut is utilized in our resBi-LSTM layer to add the original information before layer to the information after the Bi-LSTM layer. Residual structure used here allows the information extracted from visual module to be perceived by all the LSTM layers.


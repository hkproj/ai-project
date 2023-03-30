# Syllable-Based Indonesian Lip Reading Model

The goal of this research is to create an Indonesian Lip reading model that can handle OOV, which would make it applicable in the real world settings using deep learning.

## Architecture

The proposed architecture is built with Spatiotemporal Convolution Neural Network (StCNN), as illustrated in Fig. 1. The input is a sequence of frames with a length of t. Then it processed in a 3D architecture consisting of StCNN, Batch Normalization, Spatial Dropout, and Max Polling. This model also used BiGRU, though the input is in 2D form, flatten process is required beforehand. Finally, the output of Bi-GRU passed through a fully connected layer to classify the sequence of frames.

## Dataset

The data set used in this research is recorded using 13 MP phone camera with 30 frames per second (fps). Each video contains a different sentence spoken by speakers.

1. The speakers consisted of five women and five men
1. The speakers aged 20 to 50 years old.
1. Each speaker has different facial features and various pronunciations.
1. Each speaker spoke five sentences
1. Total video used 50 video
1. Total syllables from 5 sentences are 29 (included silence)

## Data augmentation

Some examples include: adding specific values to all pixel intensities in the data which makes the image look brighter, giving some noise such as changing some values to 0 so that black dots will appear, changing some pixel values to 255 so that the white dots will appear, rotate images to a certain level, blur images using Gaussian, and others

## Training

there are 2000 videos for training and 50 videos for testing

## Results

Table III illustrates the results of training using 6, 7, and 8 frames for each syllable. It shows that the proposed model produces an accuracy of 100%, which is better than the LipNet that gives an accuracy of 88.6%

import random
import shutil
from srtloader import SRTLoader
from pathlib import Path
import fstools
import os
from tqdm import tqdm

if __name__ == '__main__':
    # Create the output dir if it doesn't exist
    lipnetDataPath = Path("./lipnet_data")
    os.makedirs(lipnetDataPath, exist_ok=True)
    GRIDAlignTXT = lipnetDataPath / "GRID_align_txt"
    lipPath = lipnetDataPath / "lip"
    os.makedirs(GRIDAlignTXT, exist_ok=True)
    os.makedirs(lipPath, exist_ok=True)
    helper = fstools.DatasetFSHelper()
    miniclipsPath = Path(helper.getMiniClipsFolderPath())
    allVideos = {}

    allChars = set()

    maxCharsCount = 0

    for videoPath in tqdm(miniclipsPath.iterdir(), desc="Processing videos", unit = ' video', position=0):
        if videoPath.is_dir():
            videoId = videoPath.name
            allVideos[videoId] = []
            for miniclip in tqdm(videoPath.iterdir(), desc= "Processing miniclips", unit = ' miniclip', position=1, leave=False):
                if miniclip.is_dir():
                    miniclipId = miniclip.name
                    allVideos[videoId].append(miniclipId)

                    # Output paths
                    lipnetFramesDir = lipPath / videoId / "video" / "mpg_6000" / miniclipId
                    lipnetFramesDir.mkdir(parents=True, exist_ok=True)
                    lipnetTranscriptDir = GRIDAlignTXT / videoId / "align"
                    lipnetTranscriptDir.mkdir(parents=True, exist_ok=True)
                    
                    # Input paths
                    framesPath = miniclip / fstools.MINI_CLIP_FRAMES_DIR
                    lipsPath = miniclip / fstools.MINI_CLIP_LIPS_DIR
                    lipsDebugPath = miniclip / fstools.MINI_CLIP_LIPS_DEBUG_DIR
                    srtPath = miniclip / (fstools.MINI_CLIP_TRANSCRIPT_NAME + fstools.TRANSCRIPTION_FILE_EXTENSION)
                    
                    # Copy all frames from lipsPath into lipnetFramesDir
                    for lipFilePath in lipsPath.iterdir():
                        if lipFilePath.name.endswith(fstools.FRAME_FILE_EXTENSION):
                            destPath = lipnetFramesDir / lipFilePath.name
                            
                            shutil.copy(lipFilePath, destPath)

                    srtLoader = SRTLoader(srtPath)
                    allWords = srtLoader.getAllWords()
                    # Save all words in the align file
                    alignFilePath = lipnetTranscriptDir / (miniclipId + '.align')
                    totalLength = 0
                    with open(alignFilePath, 'w') as alignFile:
                        for entry in allWords:
                            _ , _,_ , word = entry
                            for token in word.split(' '):
                                toWrite = token.upper()
                                alignFile.write(f'{int(0*1000)} {int(0*1000)} {toWrite}\n')
                                totalLength += len(toWrite)
                                for char in token.upper():
                                    allChars.add(char)
                    # Save the max number of chars
                    maxCharsCount = max(maxCharsCount, totalLength)
                    # Check that the totalLength is less or equal to 200
                    limit = 200
                    assert totalLength <= limit, f'The total length of the transcript {videoId} {miniclipId} is {totalLength} which is greater than {limit}'

    print(f'The max number of chars is {maxCharsCount}')

    lipnetDataSetPath = Path("./lipnet_datasets")
    os.makedirs(lipnetDataSetPath, exist_ok=True)

    allVideosIds = list(allVideos.keys())
    # Randomize the order of the videos
    # Select 80% of the videos for training
    trainIndex = int(len(allVideosIds)*0.8)
    unseenTrainVideosIds = allVideosIds[:trainIndex]
    # Select 20% of the videos for validation
    unseenValVideosIds = allVideosIds[trainIndex:]

    # Create a txt file with one line per miniclip for the training set
    with open(lipnetDataSetPath / "unseen_train.txt", 'w') as trainFile:
        for videoId in unseenTrainVideosIds:
            for miniclipId in allVideos[videoId]:
                trainFile.write(f'{videoId}/video/mpg_6000/{miniclipId}\n')

    # Create a txt file with one line per miniclip for the validation set
    with open(lipnetDataSetPath / "unseen_val.txt", 'w') as valFile:
        for videoId in unseenValVideosIds:
            for miniclipId in allVideos[videoId]:
                valFile.write(f'{videoId}/video/mpg_6000/{miniclipId}\n')

    allPairs = [(videoId, miniclipId) for videoId in allVideosIds for miniclipId in allVideos[videoId]]
    # Randomize the order of the array allPairs
    random.shuffle(allPairs)
    # Select 80% of the videos for training
    trainIndex = int(len(allPairs)*0.8)
    trainPairs = allPairs[:trainIndex]
    # Select 20% of the videos for validation
    valPairs = allPairs[trainIndex:]

    # Create a txt file with one line per miniclip for the training set
    with open(lipnetDataSetPath / "overlap_train.txt", 'w') as trainFile:
        for videoId, miniclipId in trainPairs:
            trainFile.write(f'{videoId}/video/mpg_6000/{miniclipId}\n')

    # Create a txt file with one line per miniclip for the validation set
    with open(lipnetDataSetPath / "overlap_val.txt", 'w') as valFile:
        for videoId, miniclipId in valPairs:
            valFile.write(f'{videoId}/video/mpg_6000/{miniclipId}\n')

    # Make sure the alphabet is present
    for charIndex in range(ord('A'), ord('Z') + 1):
        allChars.add(chr(charIndex))

    # Save the vocabulary (allChars) in a file, one letter per line
    with open(lipnetDataSetPath / "vocabulary.txt", 'w') as vocabFile:
        for char in sorted(allChars):
            vocabFile.write(f'{char}\n')


    
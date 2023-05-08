from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import fstools

BRANCH_NAME = 'main'

def checkMiniClipsStats(videoId, miniclip):
    miniclipId = miniclip.name
    framesPath = miniclip / fstools.MINI_CLIP_FRAMES_DIR
    lipsPath = miniclip / fstools.MINI_CLIP_LIPS_DIR
    lipsDebugPath = miniclip / fstools.MINI_CLIP_LIPS_DEBUG_DIR
    numFrames = len([file for file in framesPath.iterdir() if file.name.endswith(fstools.FRAME_FILE_EXTENSION)])
    numLips = len([file for file in lipsPath.iterdir() if file.name.endswith(fstools.FRAME_FILE_EXTENSION)])
    numLipsDebug = len([file for file in lipsDebugPath.iterdir() if file.name.endswith(fstools.FRAME_FILE_EXTENSION)])
    frameIsBlack = []
    for frameFilePath in framesPath.iterdir():
        if frameFilePath.name.endswith(fstools.FRAME_FILE_EXTENSION):
            # Open the file with OpenCV
            frame = cv2.imread(str(frameFilePath))
            height, width, channels = frame.shape
            allBlackImage = np.zeros((height, width, 3), np.uint8)
            assert allBlackImage.shape == frame.shape
            assert allBlackImage.dtype == frame.dtype
            # Check if the image is an array of all black pixels
            isBlack = frame == allBlackImage
            if (np.all(isBlack)):
                frameIsBlack.append(frameFilePath.name)
    return videoId, miniclipId, frameIsBlack, numFrames, numLips, numLipsDebug

if __name__ == '__main__':
    helper = fstools.DatasetFSHelper()
    miniclipsPath = Path(helper.getMiniClipsFolderPath())
    entries = []
    futures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for videoPath in miniclipsPath.iterdir():
            if videoPath.is_dir():
                videoId = videoPath.name
                for miniclip in videoPath.iterdir():
                    if miniclip.is_dir():
                        futures.append(executor.submit(checkMiniClipsStats, videoId, miniclip))
        for future in futures:
            videoId, miniclipId, frameIsBlack, numFrames, numLips, numLipsDebug = future.result()
            numBlackFrames = len(frameIsBlack)
            entries.append((videoId, miniclipId, numFrames, numLips, numLipsDebug, numBlackFrames))
            print(f'{videoId:<20}\t{miniclipId:<15}\tframes: {numFrames:>5}\tlips: {numLipsDebug:>5}\tlipsdebug: {numLipsDebug:>5}\tblack: {numBlackFrames:>5}')
        # Transform entries into a DataFrame
        df = pd.DataFrame(entries, columns=['videoId', 'miniclipId', 'numFrames', 'numLips', 'numLipsDebug', 'numBlackFrames'])
        # Save the DataFrame to a CSV file
        df.to_csv(Path('miniclips_stats.csv'), index=False)

import shutil
from srtloader import SRTLoader
from pathlib import Path
import fstools
import os

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
    for videoPath in miniclipsPath.iterdir():
        if videoPath.is_dir():
            videoId = videoPath.name
            for miniclip in videoPath.iterdir():
                if miniclip.is_dir():
                    lipnetFramesDir = lipPath / videoId / "video" / "mpg_6000" / miniclip.name
                    lipnetFramesDir.mkdir(parents=True, exist_ok=True)
                    lipnetTranscriptDir = GRIDAlignTXT / videoId / "align"
                    lipnetTranscriptDir.mkdir(parents=True, exist_ok=True)
                    miniclipId = miniclip.name
                    framesPath = miniclip / fstools.MINI_CLIP_FRAMES_DIR
                    lipsPath = miniclip / fstools.MINI_CLIP_LIPS_DIR
                    lipsDebugPath = miniclip / fstools.MINI_CLIP_LIPS_DEBUG_DIR
                    srtPath = miniclip / (fstools.MINI_CLIP_TRANSCRIPT_NAME + fstools.TRANSCRIPTION_FILE_EXTENSION)
                    srtLoader = SRTLoader(srtPath)
                    allWords = srtLoader.getAllWords()
                    # Copy all frames from lipsPath into lipnetFramesDir
                    for lipFilePath in lipsPath.iterdir():
                        if lipFilePath.name.endswith(fstools.FRAME_FILE_EXTENSION):
                            destPath = lipnetFramesDir / lipFilePath.name
                            shutil.copy(lipFilePath, destPath)
                    # Save all words in the align file
                    alignFilePath = lipnetTranscriptDir / (miniclipId + '.align')
                    with open(alignFilePath, 'w') as alignFile:
                        for entry in allWords:
                            _ , start, end, word = entry
                            alignFile.write(f'{int(start*1000)} {int(end*1000)} {word}\n')
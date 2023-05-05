from pathlib import Path
import os
import shutil

DATA_DIR = 'data'
RAW_VIDEOS_DIR = 'raw_videos'
FACES_DIR = 'faces'
CLIPS_DIR = 'clips'
MINI_CLIPS_DIR = 'miniclips'
BRANCHES_DIR = 'branches'

MINI_CLIP_VIDEO_NAME = 'video'
MINI_CLIP_REDUCED_FPS_NAME = 'video_25fps'
MINI_CLIP_TRANSCRIPT_NAME = 'transcript'
MINI_CLIP_FRAMES_DIR = 'frames'

MODEL_BRANCH_NAME = '_model'
DEFAULT_BRANCH_NAME = 'main'

VIDEO_FILE_EXTENSION = '.mp4'
AUDIO_FILE_EXTENSION = '.aac'
TRANSCRIPTION_FILE_EXTENSION = '.aac.word.srt'
REGULARIZED_TRANSCRIPTION_FILE_EXTENSION = '.aac.regularized-word.srt'
CLEANED_TRANSCRIPTION_FILE_EXTENSION = '.aac.cleaned-word.srt'


class DatasetFSHelper:

    def __init__(self, branchName: str=DEFAULT_BRANCH_NAME) -> None:
          self.setBranchName(branchName)

    def setBranchName(self, branchName: str) -> None:
        if not branchName or len(branchName) < 1 or branchName == MODEL_BRANCH_NAME:
            raise ValueError('Illegal branch name')
        self.branch = branchName

    def getVideoIdFromFileName(fileName: str) -> str:
        return os.path.splitext(os.path.basename(fileName))[0]

    def getVideoFileName(videoId: str) -> str:
        return videoId + VIDEO_FILE_EXTENSION

    def getRawVideoPath(self, videoId: str) -> str:
        return str(Path(DatasetFSHelper.getRawVideoFolderPath()) / DatasetFSHelper.getVideoFileName(videoId))

    def getRawVideoFolderPath() -> str:
        return str(Path('.') / DATA_DIR / RAW_VIDEOS_DIR)

    def getFacesPath(self, videoId: str) -> str:
        path = Path('.') / DATA_DIR / BRANCHES_DIR / self.branch / FACES_DIR / videoId
        return str(path)

    def getClipsFolderPath(self) -> str:
        return str(Path('.') / DATA_DIR / BRANCHES_DIR / self.branch / CLIPS_DIR)

    def getClipsPath(self, videoId: str) -> str:
        path = Path('.') / DATA_DIR / BRANCHES_DIR / self.branch / CLIPS_DIR / videoId
        return str(path)
    
    def getMiniClipsPath(self, videoId: str) -> str:
        path = Path('.') / DATA_DIR / BRANCHES_DIR / self.branch / MINI_CLIPS_DIR / videoId
        return str(path)
    
    def ensureBranchPathExists(self) -> None:
        branchPath = str(Path('.') / DATA_DIR / BRANCHES_DIR / self.branch)
        if Path.exists(Path(branchPath)):
            return
        modelBranchPath =  str(Path('.') / DATA_DIR / BRANCHES_DIR / MODEL_BRANCH_NAME)
        shutil.copytree(modelBranchPath, branchPath)
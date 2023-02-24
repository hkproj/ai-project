from pathlib import Path
import os

DATA_DIR = 'data'
RAW_VIDEOS_DIR = 'raw_videos'
FACES_DIR = 'faces'
CLIPS_DIR = 'clips'

VIDEO_FILE_EXTENSION = '.mp4'
AUDIO_FILE_EXTENSION = '.aac'

def getVideoIdFromFileName(fileName: str) -> str:
    return os.path.splitext(os.path.basename(fileName))[0]

def getVideoFileName(videoId: str) -> str:
    return videoId + VIDEO_FILE_EXTENSION

def getRawVideoPath(videoId: str) -> str:
    return str(Path(getRawVideoFolderPath()) / getVideoFileName(videoId))

def getRawVideoFolderPath() -> str:
    return str(Path('.') / DATA_DIR / RAW_VIDEOS_DIR)

def getFacesPath(videoId: str) -> str:
    path = Path('.') / DATA_DIR / FACES_DIR / videoId
    return str(path)

def getClipsFolderPath() -> str:
    return str(Path('.') / DATA_DIR / CLIPS_DIR)

def getClipsPath(videoId: str) -> str:
    path = Path('.') / DATA_DIR / CLIPS_DIR / videoId
    return str(path)
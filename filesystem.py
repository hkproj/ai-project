from pathlib import Path
import os

DATA_DIR = 'data'
RAW_VIDEOS_DIR = 'raw_videos'
FACES_DIR = 'faces'

def getVideoId(fileName: str) -> str:
    return os.path.splitext(os.path.basename(fileName))[0]

def getVideoFileName(videoId: str) -> str:
    return videoId + '.mp4'

def getRawVideoPath(videoId: str) -> str:
    return str(Path(getRawVideoFolderPath()) / getVideoFileName(videoId))

def getRawVideoFolderPath() -> str:
    return str(Path('.') / DATA_DIR / RAW_VIDEOS_DIR)

def getFacesPath(videoId: str) -> str:
    facesPath = Path('.') / DATA_DIR / FACES_DIR / videoId
    return str(facesPath)

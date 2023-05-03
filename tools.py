import os
import datetime

def getVideoDuration(videoPath: str) -> float:
    command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{videoPath}\""
    duration = os.popen(command).read()
    return float(duration)

def cutVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    command = f"ffmpeg -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:a copy \"{outputPath}\""
    os.system(command)

def cutExactVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    #ffmpeg -accurate_seek -i INPUT.mp4 -ss 00:00:00.581 -to 00:00:03.469 -c:v copy -c:a copy OUTPUT.mp4
    command = f"ffmpeg -accurate_seek -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:v copy -c:a copy \"{outputPath}\""
    os.system(command)

def extractAudio(inputPath: str, outputPath: str) -> None:
    command = f"ffmpeg -i \"{inputPath}\" -vn -acodec copy \"{outputPath}\""
    os.system(command)

def transcribeAudio(inputPath: str, outputDir: str) -> None:
    command = f"./transcribe_audio.sh \"{inputPath}\" \"{outputDir}\""
    os.system(command)
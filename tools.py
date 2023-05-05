import os
import datetime

def getVideoDuration(videoPath: str) -> float:
    command = f"/usr/bin/ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{videoPath}\""
    duration = os.popen(command).read()
    return float(duration)

def extractFramesFromVideo(inputPath: str, outputDir: str, fps: int) -> None:
    command = f"/usr/bin/ffmpeg -i \"{inputPath}\" -qscale:v 2 -r {fps} \"{outputDir}/%d.jpg\""
    os.system(command)

def changeVideoFPS(inputPath: str, outputPath: str, fps: int) -> None:
    command = f"/usr/bin/ffmpeg -i \"{inputPath}\" -filter:v fps={fps} \"{outputPath}\""
    os.system(command)

def cutVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    command = f"/usr/bin/ffmpeg -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:a copy -crf 10 \"{outputPath}\""
    os.system(command)

def cutExactVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    #ffmpeg -accurate_seek -i INPUT.mp4 -ss 00:00:00.581 -to 00:00:03.469 -c:v copy -c:a copy OUTPUT.mp4
    command = f"/usr/bin/ffmpeg -accurate_seek -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:a copy -crf 10 \"{outputPath}\""

    os.system(command)

def extractAudio(inputPath: str, outputPath: str) -> None:
    command = f"/usr/bin/ffmpeg -i \"{inputPath}\" -vn -acodec copy \"{outputPath}\""
    os.system(command)

def transcribeAudio(inputPath: str, outputDir: str) -> None:
    command = f"./transcribe_audio.sh \"{inputPath}\" \"{outputDir}\""
    os.system(command)
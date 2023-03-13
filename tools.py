import os
import datetime

def cutVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    command = f"ffmpeg -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:a copy \"{outputPath}\""
    os.system(command)

def extractAudio(inputPath: str, outputPath: str) -> None:
    command = f"ffmpeg -i \"{inputPath}\" -vn -acodec copy \"{outputPath}\""
    os.system(command)

def transcribeAudio(inputPath: str, outputDir: str) -> None:
    command = f"./transcribe_audio.sh \"{inputPath}\" \"{outputDir}\""
    os.system(command)
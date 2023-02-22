import os
import datetime

def cutVideo(inputPath: str, outputPath: str, fromTs: float, toTs: float) -> None:
    fromTsString = str(datetime.timedelta(seconds=(fromTs / 1000)))
    toTsString = str(datetime.timedelta(seconds=(toTs / 1000)))

    command = f"ffmpeg -i \"{inputPath}\" -ss {fromTsString} -to {toTsString} -c:a copy \"{outputPath}\""
    os.system(command)
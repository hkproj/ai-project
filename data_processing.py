import cv2
from pathlib import Path
import fstools
from datasets import VideoDataset
import face_recognition
import shutil
import time
import logging
from log import getLogger
import concurrent.futures
import argparse
import os
import datetime
import videotools

logger = getLogger(os.path.splitext(os.path.basename(__file__))[0])

def _extractFacesFromFrame(index: int, rgbFrame):
    faceLocations = face_recognition.face_locations(rgbFrame)
    faceEncodings = face_recognition.face_encodings(rgbFrame, faceLocations)
    return index, faceLocations, faceEncodings

def extractAllFacesFromVideo(videoId: str, sleep: int, frameBatchSize: int = 1) -> None:

    # If the directory already exists, ignore this video
    faceSavePath = fstools.getFacesPath(videoId)
    # Delete the directory and then re-create it
    if Path.exists(Path(faceSavePath)):
        logger.info(f'Ignoring video {videoId} because its directory already exists')
        return

    # Get the video's path
    videoPath = fstools.getRawVideoPath(videoId)
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # results
    faceEncodings = []
    faceTimestamps = {}
    faceFrames = {}

    # stats
    totalFramesCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    globalFrameIndex = 0
    startTime = time.time()
    lastStatTime = time.time()
    STAT_PRINT_INTERVAL = 10.0

    batch = []

    while cap.isOpened():
        frameExists, frame = cap.read()

        # Sleep if necessary
        if (sleep > 0):
            time.sleep(float(sleep) / 1000)

        if frameExists:
            globalFrameIndex += 1

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgbFrame = frame[:, :, ::-1]
            # Add the frame and timestamp to the batch
            batch.append((frame, rgbFrame, timestamp))

        isLastBatch = not frameExists

        # Process the batch only if it reached the batch size or no more frames are available
        if len(batch) == frameBatchSize or (isLastBatch and len(batch) > 0):

            batchFaceLocations = [None] * len(batch)
            batchFaceEncodings = [None] * len(batch)

            # Parallelize the process
            with concurrent.futures.ProcessPoolExecutor(max_workers=frameBatchSize) as pool:
                futures = [pool.submit(_extractFacesFromFrame, index, rgbFrame) for index, (_, rgbFrame, _) in enumerate(batch)]
                for future in futures:
                    # Make sure the indices correspond
                    i, l, e = future.result()
                    batchFaceLocations[i] = l
                    batchFaceEncodings[i] = e

            for (frame, rgbFrame, timestamp), frameFaceLocations, currentFrameFaceEncodings in zip(batch, batchFaceLocations, batchFaceEncodings):

                for (top, right, bottom, left), currFaceEncoding in zip(frameFaceLocations, currentFrameFaceEncodings):
                    matches = face_recognition.compare_faces(faceEncodings, currFaceEncoding)

                    if True in matches:
                        # Get the index of the first face it matches with
                        faceIndex = matches.index(True)
                        faceTimestamps[faceIndex].append(timestamp)
                    else:
                        # Save the face in the list
                        insertIndex = len(faceEncodings)
                        faceEncodings.append(currFaceEncoding)
                        faceTimestamps[insertIndex] = [timestamp]
                        faceFrames[insertIndex] = frame
                        logger.debug(f'Video {videoId} - New face found at {timestamp / 1000 :.4f}s. Total faces: {len(faceEncodings)}')
                if (time.time() - lastStatTime) > STAT_PRINT_INTERVAL:
                    processingSpeed = (globalFrameIndex / (time.time() - startTime))
                    timestampSeconds = timestamp / 1000
                    percentage = float(globalFrameIndex) / totalFramesCount * 100.0
                    estimatedTimeToComplete = datetime.timedelta(seconds=int((totalFramesCount - globalFrameIndex) / (processingSpeed + 0.0001)))
                    logger.info(f'Video {videoId} - Speed: {processingSpeed:.1f} fps. Timestamp: {timestampSeconds:.3f}s. Percentage: {percentage:.2f}. ETA: {estimatedTimeToComplete}')
                    lastStatTime = time.time()

            # Reset the batch
            batch = []

        if isLastBatch:
            break

    cap.release()

    # Make sure the directory exists
    Path(faceSavePath).mkdir(parents=True, exist_ok=True)

    # Save all the faces found
    for i in range(len(faceEncodings)):
        # Save frame image
        frame = faceFrames[i]
        faceImageFilePath = str(Path(faceSavePath) / (str(i) + '.jpg'))
        cv2.imwrite(faceImageFilePath, frame)

        # Save timestamps
        timestamps = faceTimestamps[i]
        timestampsFilePath = str(Path(faceSavePath) / (str(i) + '.csv'))
        writeRawTimestampsFile(timestampsFilePath, timestamps)

    logger.debug(
        f'Video {videoId} - Processed {totalFramesCount} frames in {(time.time() - startTime):.2f}s')

def writeRawTimestampsFile(filePath: str, timestamps: list[float]) -> None:
    with open(filePath, 'w') as file:
        for ts in timestamps:
            file.write(f'{ts:.4f}\n')

def loadRawTimestampsFile(filePath: str) -> list[float]:
    with open(filePath, 'r') as file:
        lines = file.readlines()
        numbers = [float(line) for line in lines if len(line.strip()) > 0]
        return numbers
    
def writeIntervalsFile(filePath: str, intervals: list[tuple[float, float]]) -> None:
    with open(filePath, 'w') as file:
        for start,end in intervals:
            file.write(f'{start:.4f};{end:.4f}\n')

def loadIntervalsFile(filePath: str) -> list[tuple[float, float]]:
    intervals = []
    with open(filePath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            values = line.split(';')
            if len(values) != 2:
                raise Exception('Invalid file format')
            start,end=float(values[0]), float(values[1])
            intervals.append((start,end))
    return intervals

def handleExtractFacesCommand(workers: int, videoIds: list[int], sleep: int) -> None:
    if videoIds is None:
        ds = VideoDataset(fstools.getRawVideoFolderPath())
        videoIds = [videoId for videoId in ds]
    for videoId in videoIds:
        # Get the video's path
        videoPath = fstools.getRawVideoPath(videoId)
        if not Path.exists(Path(videoPath)):
            raise FileNotFoundError()
    logger.debug(f'Will extract faces from {len(videoIds)} videos')
    for videoId in videoIds:
        extractAllFacesFromVideo(videoId, sleep, frameBatchSize=workers)

def handleMergeFaces(videoId: str, sourceList: list, target: int) -> None:
    # Get the video's path
    videoPath = fstools.getFacesPath(videoId)
    
    # Check arguments
    assert sourceList and len(sourceList) > 0, "Please indicate at least one source face"
    assert target is not None, "Please indicate the target"

    allTimestampLists = []

    # Very all the source files exist
    for source in sourceList:
        faceTimestampsFilePath = Path(videoPath) / (str(source) + '.csv')
        if not Path.exists(faceTimestampsFilePath):
            raise FileNotFoundError(str(faceTimestampsFilePath))
        else:
            # If the file exists, then load it in memory and delete it
            sourceTimestamps = loadRawTimestampsFile(faceTimestampsFilePath)
            allTimestampLists.append(sourceTimestamps)
            logger.debug(f'Loaded target {source} of size {len(sourceTimestamps)}')
            os.remove(faceTimestampsFilePath)
    
    # Verify if the target file exists
    targetTimestampsFilePath = Path(videoPath) / (str(target) + '.csv')
    if not Path.exists(targetTimestampsFilePath):
        raise FileNotFoundError(str(targetTimestampsFilePath))
    else:
        # If the target file exists, load it in memory and delete it
        targetTimestamps = loadRawTimestampsFile(targetTimestampsFilePath)
        allTimestampLists.append(targetTimestamps)
        logger.debug(f'Target size before merging: {len(targetTimestamps)}')
        os.remove(targetTimestampsFilePath)

    # Merge the timestamps
    mergedTimestamps = set()
    for timestampList in allTimestampLists:
        for item in timestampList:
            mergedTimestamps.add(item)

    mergedTimestamps = sorted(list(mergedTimestamps))
    logger.debug(f'Target size after merging: {len(mergedTimestamps)}')

    # Write the result of the merge back into the target file
    writeRawTimestampsFile(targetTimestampsFilePath, mergedTimestamps)

def handleCreateIntervals(videoId: str, target: int, maxDifference: int, warnLimit: int) -> None:
    # Get the video's path
    videoPath = fstools.getFacesPath(videoId)
    if not Path.exists(Path(videoPath)):
        raise FileNotFoundError(videoPath)
    
    # Check arguments
    assert target is not None, "Please indicate the target"
    assert maxDifference is not None, "Please indicate the max difference"
    assert warnLimit is not None and warnLimit > maxDifference, "Indicate a correct warning limit"

    # Verify if the target file exists
    originalFilePath = Path(videoPath) / (str(target) + '.csv')
    if not Path.exists(originalFilePath):
        raise FileNotFoundError(str(originalFilePath))
    
    #load it in memory
    timestamps = loadRawTimestampsFile(originalFilePath)

    # Add the first interval
    assert len(timestamps) > 0, "File is empty"
    intervals = []
    intervals.append((timestamps[0], timestamps[0]))

    # Iterate and, based on the difference, decide if a new interval is needed
    for timestamp in timestamps[1:]:
        lastStart, lastEnd = intervals[-1]
        assert timestamp > lastEnd, "File is not sorted!"
        if (timestamp - lastEnd) > maxDifference:
            # Close the timestamp and start a new one
            intervals.append((timestamp, timestamp))
        else:
            intervals[-1] = lastStart, timestamp

    # Check if any interval is triggering the limit
    for start, end in intervals:
        if (end - start) < warnLimit:
            logger.warning(f'Found interval of {(end-start)}ms, starting at {start}ms')

    logger.debug(f'Total intervals: {len(intervals)}')
    for start, end in intervals:
        duration = end - start
        logger.info(f'Interval: {str(datetime.timedelta(seconds=(start / 1000))).ljust(20)} {str(datetime.timedelta(seconds=(end / 1000))).ljust(20)} - Duration {str(datetime.timedelta(seconds=(duration / 1000))).ljust(20)}')

    # Write output file
    outputFilePath = str(Path(videoPath) / (str(target) + '_intervals.csv'))
    if Path.exists(Path(outputFilePath)):
        os.remove(outputFilePath)
    writeIntervalsFile(outputFilePath, intervals)

def handleCreateClips(videoId: str, targetIntervals: int, minDuration: int, workers: int=1) -> None:
    # Check args
    assert targetIntervals is not None, "Target not specified"
    assert minDuration > 0, "Minimum duration not specified"

    # Get the video's path
    videoPath = fstools.getFacesPath(videoId)
    # Verify if the target file exists
    intervalsFilePath = Path(videoPath) / (str(targetIntervals) + '_intervals.csv')
    if not Path.exists(intervalsFilePath):
        raise FileNotFoundError(str(intervalsFilePath))
    
    # Verify the raw video exists
    inputVideoPath = fstools.getRawVideoPath(videoId)
    if not Path.exists(Path(inputVideoPath)):
        raise FileNotFoundError(str(inputVideoPath))

    # load the intervals in memory
    intervals = loadIntervalsFile(intervalsFilePath)

    # Make sure the output path exists
    clipsPath = fstools.getClipsPath(videoId)
    Path(clipsPath).mkdir(parents=True, exist_ok=True)

    for index, (start,end) in enumerate(intervals):
        durationSecs = float(end - start) / 1000
        if durationSecs > minDuration:
            # Delete existing clip, if it exists
            outputFilePath = Path(clipsPath) / f"{targetIntervals}_{index}{fstools.VIDEO_FILE_EXTENSION}"
            if Path.exists(outputFilePath):
                os.remove(outputFilePath)
            logger.debug(f'Cutting video from {str(datetime.timedelta(seconds=(start / 1000))).ljust(20)} to {str(datetime.timedelta(seconds=(end / 1000))).ljust(20)} and saving into {outputFilePath}')
            # Run tool to cut video
            videotools.cutVideo(inputVideoPath, outputFilePath, start, end)
        else:
            logger.info(f'Ignoring interval {index} because too short')

def handleExtractAudio(videoIds: list[str]) -> None:
    if videoIds is None:
        # Get all the videoIds from the clips folder
        videoIds = []
        path = Path(fstools.getClipsFolderPath())
        for item in path.iterdir():
            if not item.is_file():
                videoIds.append(item.name)

    for videoId in videoIds:
        # Get the video's path
        clipsPath = fstools.getClipsPath(videoId)
        if not Path.exists(Path(clipsPath)):
            raise FileNotFoundError()
    
    logger.debug(f'Will extract audio from {len(videoIds)} videos')

    for videoId in videoIds:
        clipsPath = Path(fstools.getClipsPath(videoId))
        # Get all the video files in this folder
        for item in clipsPath.iterdir():
            if item.name.endswith(fstools.VIDEO_FILE_EXTENSION):
                baseFileName = os.path.splitext(item.name)[0]
                outputFileName = clipsPath / (baseFileName + fstools.AUDIO_FILE_EXTENSION)
                videotools.extractAudio(str(item), str(outputFileName))

def handleTranscribeAudio(videoIds: list[str]) -> None:
    if videoIds is None:
        # Get all the videoIds from the clips folder
        videoIds = []
        path = Path(fstools.getClipsFolderPath())
        for item in path.iterdir():
            if not item.is_file():
                videoIds.append(item.name)

    for videoId in videoIds:
        # Get the video's path
        clipsPath = fstools.getClipsPath(videoId)
        if not Path.exists(Path(clipsPath)):
            raise FileNotFoundError()
    
    logger.debug(f'Will transcribe audio from {len(videoIds)} videos')

    for videoId in videoIds:
        clipsPath = Path(fstools.getClipsPath(videoId))
        # Get all the audio files in this folder
        for item in clipsPath.iterdir():
            if item.name.endswith(fstools.AUDIO_FILE_EXTENSION):
                raise NotImplementedError()

if __name__ == '__main__':

    COMMAND_EXTRACT_FACES = 'extract-faces'
    COMMAND_MERGE_FACES = 'merge-faces'
    COMMAND_CREATE_INTERVALS = 'create-intervals'
    COMMAND_CREATE_CLIPS = "create-clips"
    COMMAND_EXTRACT_AUDIO = "extract-audio"

    parser = argparse.ArgumentParser(prog = 'Video Processing',description = 'video processing utility')
    parser.add_argument('command', type=str, choices=[COMMAND_EXTRACT_AUDIO, COMMAND_CREATE_CLIPS, COMMAND_EXTRACT_FACES, COMMAND_MERGE_FACES, COMMAND_CREATE_INTERVALS], help='The operation to execute')
    parser.add_argument('-w', '--workers', type=int, required=False, default=1, help='Number of workers')
    parser.add_argument('--video-id', nargs='*', type=str, required=False, help='Video ID(s) to process')
    parser.add_argument('--source', nargs='+', type=int, required=False, help='Source faces to merge')
    parser.add_argument('--target', type=int, required=False, help='Target to merge to')
    parser.add_argument('--max-diff', type=int, required=False, help='Max timestamp difference')
    parser.add_argument('--warn-limit', type=int, required=False, help='Warning limit')
    parser.add_argument('--sleep', type=int, required=False, default=0, help='Sleep time after every iteration')
    parser.add_argument('--min-duration', type=int, required=False, default=0, help='Minimum duration of clip')

    args = parser.parse_args()
    print(args)

    if args.command == COMMAND_EXTRACT_FACES:
        handleExtractFacesCommand(args.workers, args.video_id, args.sleep)
    elif args.command == COMMAND_MERGE_FACES:
        handleMergeFaces(args.video_id[0], args.source, args.target)
    elif args.command == COMMAND_CREATE_INTERVALS:
        handleCreateIntervals(args.video_id[0], args.target, args.max_diff, args.warn_limit)
    elif args.command == COMMAND_CREATE_CLIPS:
        handleCreateClips(args.video_id[0], args.target, args.min_duration, args.workers)
    elif args.command == COMMAND_EXTRACT_AUDIO:
        handleExtractAudio(args.video_id)
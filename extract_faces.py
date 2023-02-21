import cv2
from pathlib import Path
from filesystem import getRawVideoPath, getFacesPath, getRawVideoFolderPath
from datasets import VideoDataset
import face_recognition
import shutil
import time
import logging
from log import getLogger
import concurrent.futures

logger = getLogger(__name__)

def extractAllFacesFromVideo(videoId: str, maxIntervalDelta=200, frameBatchSize=1, smallIntervalWarning=1000) -> None:
    # Get the video's path
    videoPath = getRawVideoPath(videoId)
    if not Path.exists(Path(videoPath)):
        raise FileNotFoundError()

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
            # Only if batch processing is enabled...
            if frameBatchSize > 1:
                batchFaceLocations = face_recognition.batch_face_locations([rgbFrame for _, rgbFrame, _ in batch])
            else:
                _, rgbFrame, _ = batch[0]
                batchFaceLocations = [face_recognition.face_locations(rgbFrame)]

            assert len(batchFaceLocations) == len(batch), "dimension of face locations result must match batch size"

            for (frame, rgbFrame, timestamp), frameFaceLocations in zip(batch, batchFaceLocations):
                currentFrameFaceEncodings = face_recognition.face_encodings(rgbFrame, frameFaceLocations)

                for (top, right, bottom, left), currFaceEncoding in zip(frameFaceLocations, currentFrameFaceEncodings):
                    matches = face_recognition.compare_faces(faceEncodings, currFaceEncoding)

                    if True in matches:
                        # Get the index of the first face it matches with
                        faceIndex = matches.index(True)
                        lastTimeStampStart, lastTimeStampEnd = faceTimestamps[faceIndex][-1]
                        if timestamp - lastTimeStampEnd > maxIntervalDelta:
                            # Check if the last interval is too short; if it is, log a warning
                            if lastTimeStampEnd - lastTimeStampStart <= smallIntervalWarning:
                                logger.warning(f'Video {videoId} - Small interval detected, only {(lastTimeStampEnd - lastTimeStampStart):.3f}ms')
                            # Start a new timestamp
                            faceTimestamps[faceIndex].append((timestamp, timestamp))
                        else:
                            # Merge with existing latest timestamp
                            faceTimestamps[faceIndex][-1] = (lastTimeStampStart, timestamp)
                        continue
                    else:
                        # Save the face in the list
                        insertIndex = len(faceEncodings)
                        faceEncodings.append(currFaceEncoding)
                        faceTimestamps[insertIndex] = [(timestamp, timestamp)]
                        faceFrames[insertIndex] = frame
                        logger.debug(f'Video {videoId} - New face found at {timestamp / 1000 :.4f}s. Total faces: {len(faceEncodings)}')
                if (time.time() - lastStatTime) > STAT_PRINT_INTERVAL:
                    logger.info(f'Video {videoId} - Speed: {(globalFrameIndex / (time.time() - startTime)) :.1f} fps. Timestamp: {timestamp / 1000 :.3f}s. Percentage: {(float(globalFrameIndex) / totalFramesCount * 100.0):.2f}')
                    lastStatTime = time.time()

            # Reset the batch
            batch = []

        if isLastBatch:
            break

    cap.release()

    faceSavePath = getFacesPath(videoId)
    # Delete the directory and then re-create it
    if Path.exists(Path(faceSavePath)):
        shutil.rmtree(faceSavePath)
    # Re-create it
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
        with open(timestampsFilePath, 'w') as file:
            for ts in timestamps:
                file.write(f'{ts[0]:.4f};{ts[1]:.4f}\n')

    logger.debug(
        f'Video {videoId} - Processed {totalFramesCount} frames in {(time.time() - startTime):.2f}s')


if __name__ == '__main__':
    ds = VideoDataset(getRawVideoFolderPath())
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(extractAllFacesFromVideo, videoId, frameBatchSize=1) for videoId in ds]
        for future in futures:
            future.result()
